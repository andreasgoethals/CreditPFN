"""Submit the isolated v2/v2.6 workflow without a running-task throttle.

Queue-admission refusals are retried with smaller arrays, preserving task IDs.
Every accepted job is durable; an ambiguous sbatch outcome stops for inspection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from omegaconf import OmegaConf
from src.utils.atomic import write_json
from src.utils.legacy_design import root, scope, validate_id
from src.utils.paths import REPO_ROOT, manifests_dir


def test_plan(identifier):
    common = ["--partition=batch_graniterapids", "--cpus-per-task=8"]
    return [
        dict(key="prepare", needs=[], tasks=None, args=[*common, "scripts/slurm/legacy_cpu.slurm", "prepare", identifier]),
        dict(key="smoke", needs=["prepare"], tasks=list(range(12)), args=["--time=01:00:00", "scripts/slurm/legacy_gpu.slurm", "smoke", identifier]),
        dict(key="capacity", needs=["smoke"], tasks=list(range(68)), args=["--time=03:00:00", "scripts/slurm/legacy_gpu.slurm", "capacity", identifier]),
        dict(key="inspect", needs=["capacity"], tasks=None, args=[*common, "scripts/slurm/legacy_cpu.slurm", "inspect", identifier]),
    ]


def research_plan(identifier):
    with scope(identifier):
        return _research_plan(identifier)


def _research_plan(identifier):
    from src.utils.legacy_checks import require_gate
    from src.train.config import load_train_config, resolve_grid
    from src.utils.prepare_experiment import check_prepared
    workflow = require_gate(identifier)
    training, context, benchmarks = [], [], []
    for key, path in sorted(workflow["production"].items()):
        if key.startswith("e10_"):
            cfg = OmegaConf.load(path)
            with scope(identifier, "experiment4"):
                plan_path = manifests_dir("experiment4") / "plans" / (cfg.run_name + ".json")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            context.append(dict(key=key, needs=[], tasks=list(range(len(plan["tasks"]))),
                args=[f"--export=ALL,CREDITPFN_LEGACY_ID={identifier}",
                      "scripts/slurm/context_benchmark.slurm", str(plan_path)]))
            continue
        cfg = load_train_config(config_path=path)
        check_prepared(Path(path))
        group, track = str(cfg.experiment.output_group), str(cfg.track)
        for split in range(4):
            token = f"{key}_s{split}"
            export = (f"--export=ALL,CREDITPFN_LEGACY_ID={identifier},CREDITPFN_EXPERIMENT={group},"
                      f"CREDITPFN_CONFIG={path},CREDITPFN_SPLIT_INDEX={split},CREDITPFN_USE_SCRATCH=1,"
                      "CREDITPFN_REQUIRE_STAGING=1,CREDITPFN_TRIALS_PER_TASK=1,CREDITPFN_EVAL_KIND=foundation,CREDITPFN_EVAL_FAMILY=tabpfn,")
            training.append(dict(key=token, needs=[], tasks=list(range(len(resolve_grid(cfg, single=False)))),
                args=["--time=01:40:00", "--signal=B:USR1@600", "--requeue",
                      export + "CREDITPFN_SEGMENT_SECONDS=5400,CREDITPFN_AUTO_REQUEUE=1,CREDITPFN_MAX_REQUEUES=20",
                      f"scripts/slurm/train_{track}.slurm"]))
            benchmarks.append(dict(key=token + "_benchmark", needs=[token],
                tasks=list(range(16)), args=["--time=04:00:00", export + "EVAL_TASKS=16",
                    f"scripts/slurm/eval_{track}.slurm"]))
    return training + context + benchmarks


def submission_environment():
    """A submitted batch starts on the host, not inside its submitting container.

    Preserve requested runtime/scientific settings, but never export parent
    container markers, container injection options or job-owned cache/log paths.
    Each child activates its own runtime and creates its own cache and log.
    """
    inherited_overrides = {"CREDITPFN_STOP_AFTER_UPDATES", "CREDITPFN_STOP_REQUESTED", "CREDITPFN_STOP_REQUEST_FILE",
                           "CREDITPFN_EVAL_TASK_INDEX", "CREDITPFN_EVAL_FAMILY",
                           "CREDITPFN_OWNED_CACHE", "CREDITPFN_ACTIVE_LOG"}
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("SBATCH_", "APPTAINER_", "SINGULARITY_",
                                   "APPTAINERENV_", "SINGULARITYENV_"))
            and key not in inherited_overrides}


def submit_plan(plan, destination, *, runner=subprocess.run, sleep=time.sleep, max_array_size=None):
    """Resume confirmed submissions only; genuine prerequisites use afterok."""
    state = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else {
        "plan": plan, "accepted": [], "in_flight": None, "index": 0, "offset": 0}
    if state["plan"] != plan or state["in_flight"] is not None:
        raise RuntimeError("Changed plan or uncertain submission: inspect its receipt and Slurm before retrying")
    if state["index"] == len(plan):
        return state
    if max_array_size is None:
        limits = runner(["scontrol", "-M", "mindwell", "show", "config"], text=True,
                        capture_output=True, check=True, timeout=30)
        match = re.search(r"^\s*MaxArraySize\s*=\s*(\d+)", limits.stdout, re.M)
        if not match:
            raise RuntimeError("Cannot verify the controller's array limit")
        max_array_size = int(match[1])
    for item in plan:
        if item["tasks"] and (min(item["tasks"]) < 0 or max(item["tasks"]) >= max_array_size):
            raise ValueError("Stable task IDs exceed the verified MaxArraySize")
    environment = submission_environment()
    for index in range(state["index"], len(plan)):
        item = plan[index]
        tasks = item["tasks"]
        pending = len(tasks) - state["offset"] if tasks is not None else 1
        batch = pending  # Submit every ready task; only definite site admission limits split it.
        while pending:
            dependencies = [r["id"] for r in state["accepted"] if r["key"] in item["needs"]]
            if set(item["needs"]) - {r["key"] for r in state["accepted"]}:
                raise ValueError("Submission prerequisite was not accepted")
            command = ["sbatch", "--parsable", "--clusters=mindwell", "--account=lp_verbekelab"]
            if dependencies:
                command += ["--dependency=afterok:" + ":".join(dependencies), "--kill-on-invalid-dep=yes"]
            selected = tasks[state["offset"]:state["offset"] + batch] if tasks is not None else None
            if selected is not None:
                command += ["--array=" + ",".join(map(str, selected))]
            command += item["args"]
            state["in_flight"] = dict(key=item["key"], tasks=selected, command=command)
            write_json(destination, state)
            result = runner(command, cwd=REPO_ROOT, env=environment, text=True,
                            capture_output=True, check=False, timeout=60)
            response = result.stdout.strip()
            if result.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;mindwell)?", response):
                job = dict(key=item["key"], tasks=selected, id=response.split(";")[0])
                state["accepted"].append(job)
                state["in_flight"] = None
                state["offset"] += batch
                pending -= batch
                if not pending:
                    state["index"], state["offset"] = index + 1, 0
                write_json(destination, state)
                print(f"Queued {job['key']}: {job['id']} ({batch} task(s), no running throttle)", flush=True)
                batch = pending
            elif result.returncode and not response and re.search(
                r"(?:QOS|Assoc).*SubmitJob.*Limit|maximum number of submitted jobs", result.stderr, re.I):
                state["in_flight"] = None
                write_json(destination, state)
                if batch > 1:
                    batch = max(1, batch // 2)
                else:
                    print("Site submission ceiling reached; retrying admission in 60 seconds.", flush=True)
                    sleep(60)
                    batch = pending
            else:
                if result.returncode and not response:
                    state["in_flight"] = None
                    state["last_rejection"] = result.stderr.strip()
                    write_json(destination, state)
                raise RuntimeError(f"Submission unconfirmed; preserve {destination} and inspect Slurm: {result.stderr.strip()}")
    return state


def submit(identifier, phase):
    validate_id(identifier)
    if not os.environ.get("VSC_DATA"):
        raise RuntimeError("Submit this workflow from VSC")
    directory = root(identifier)
    marker = directory / ".scope.json"
    if phase == "tests" and not marker.exists():
        directory.mkdir(parents=True, exist_ok=True)
        write_json(marker, dict(kind="creditpfn_legacy_scope", id=identifier), exclusive=True)
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != dict(kind="creditpfn_legacy_scope", id=identifier):
        raise ValueError("Extension has no valid ownership marker")
    lock = directory / (phase + ".submit.lock")
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        from src.utils.legacy_checks import fingerprint
        record = directory / (phase + ".source.json")
        identity = dict(fingerprint=fingerprint())
        if record.exists() and json.loads(record.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Source changed since submission; use a fresh extension ID")
        write_json(record, identity)
        plan = test_plan(identifier) if phase == "tests" else research_plan(identifier)
        state = submit_plan(plan, directory / (phase + ".submission.json"))
        print(f"All {len(state['accepted'])} submissions accepted for {phase}. See scoped job logs for outcomes.", flush=True)
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("tests", "research"))
    parser.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    submit(args.id, args.phase)


if __name__ == "__main__":
    main()
