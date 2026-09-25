"""Fail-closed experiment-0 stages, advanced by completion callbacks rather than idle jobs.

All ledger updates are locked on DATA. The last successful task schedules a short
CPU audit on wICE; that audit alone may release the next Mindwell GPU stage.
No cross-controller Slurm dependencies or polling allocations are needed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import uuid

from src.utils.atomic import write_json
from src.utils.experiment import code_identity, digest_json, file_digest
from src.utils.paths import REPO_ROOT, manifests_dir

COUNTS = {"null": 8, "pilot": 16, "recovery": 4, "budget": 4}  # per track
PARTS = ("part1", "part2", "recovery")


def root() -> Path:
    return manifests_dir("experiment0") / "workflow"


def fingerprint() -> str:
    files = [REPO_ROOT / "config/train.yaml", REPO_ROOT / "config/retention.yaml",
             REPO_ROOT / "src/utils/experiment0.py", REPO_ROOT / "src/utils/recovery_check.py"]
    files += [REPO_ROOT / f"config/experiment0/{phase}_{track}.yaml"
              for phase in ("null", "pilot", "recovery") for track in ("pd", "lgd")]
    return digest_json({"training_code": code_identity(), "evaluation_code": code_identity(stage="eval"),
                        "configs": {p.name: file_digest(p) for p in files}})


@contextmanager
def locked(identifier: str):
    if not identifier.isalnum():
        raise ValueError("Invalid workflow identifier")
    import fcntl
    path = root() / identifier / "state.json"
    with path.with_suffix(".lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint():
            raise RuntimeError("Source changed during experiment 0; inspect the workflow before proceeding")
        yield path, state


def _submit(command: list[str]) -> str:
    # A timeout is ambiguous: never retry a submission that might have succeeded.
    result = subprocess.run(command, text=True, capture_output=True, timeout=45, check=True)
    job = result.stdout.strip().split(";", 1)[0]
    if not job.isdigit():
        raise RuntimeError(f"Uncertain submission response: {result.stdout!r}; inspect squeue")
    print(f"Submitted {result.stdout.strip()}", flush=True)
    return job


def _cpu(identifier: str, action: str, phase: str = "") -> str:
    return _submit(["sbatch", "--parsable", "--clusters=wice", "--partition=batch",
        "--time=00:30:00", "--export=ALL,CREDITPFN_EXPERIMENT=experiment0",
        "scripts/slurm/maintenance.slurm", "experiment0", action,
        "--id", identifier, *( ["--phase", phase] if phase else [])])


def start(part: str) -> str:
    if part not in PARTS:
        raise ValueError(f"Unknown experiment-0 part: {part}")
    if not os.environ.get("VSC_DATA"):
        raise RuntimeError("Submit experiment 0 from a VSC login node")
    identity = fingerprint()
    if part == "part2":
        receipt = root() / "part1_passed.json"
        if not receipt.exists() or json.loads(receipt.read_text())["fingerprint"] != identity:
            raise RuntimeError("Part 2 requires a passing part-1 receipt for the current code and controls")
    identifier = uuid.uuid4().hex
    folder = root() / identifier
    folder.mkdir(parents=True, exist_ok=False)
    state = {"id": identifier, "part": part, "fingerprint": identity,
             "phase": "prepare", "status": "submitting", "jobs": [], "done": [], "failed": []}
    write_json(folder / "state.json", state, exclusive=True)
    # One workflow per part/source prevents duplicate writers to the same trial files.
    claim = root() / f"{part}_{identity[:20]}.json"
    write_json(claim, {"id": identifier}, exclusive=True)
    with locked(identifier) as (path, state):
        state["jobs"].append({"phase": "prepare", "cluster": "wice", "id": _cpu(identifier, "prepare")})
        state["status"] = "preparing"
        write_json(path, state)
    print(f"Workflow {identifier}: {folder}", flush=True)
    return identifier


def prepare(identifier: str):
    from src.utils.prepare_experiment import prepare as prepare_plan
    from src.utils.stage_inputs import stage
    from src.utils.preflight import main as preflight
    with locked(identifier) as (_, state):
        part = state["part"]
    phases = {"part1": ("null", "pilot"), "part2": ("budget",), "recovery": ()}[part]
    configs = [Path(f"config/experiment0/{p}_{t}.yaml") for p in phases for t in ("pd", "lgd")]
    if part in ("part1", "recovery"):
        from src.utils.recovery_check import make_configs
        configs.extend(make_configs(root() / identifier, diagnostic=part == "recovery"))
    else:
        for phase in ("null", "pilot"):
            for track in ("pd", "lgd"):
                prepare_plan(Path(f"config/experiment0/{phase}_{track}.yaml"), check=True)
    if preflight([arg for config in configs for arg in ("--config", str(config))]):
        raise RuntimeError("CPU preflight failed; no GPU jobs submitted")
    for config in configs:
        prepare_plan(config, write=True)
    stage(Path(os.environ["VSC_SCRATCH_GPFS1"]) / "CreditPFN", write=True)
    launch(identifier, "recovery" if part == "recovery" else phases[0])


def launch(identifier: str, phase: str):
    with locked(identifier) as (path, state):
        state.update(phase=phase, status="submitting", done=[], failed=[], submissions_complete=False)
        write_json(path, state)
    env = dict(os.environ, CREDITPFN_FLOW_ID=identifier, CREDITPFN_FLOW_PHASE=phase,
               CREDITPFN_EXPERIMENT="experiment0", CREDITPFN_USE_SCRATCH="1",
               CREDITPFN_REQUIRE_STAGING="1", STAGES="train", TRIALS_PER_TASK="1",
               SEGMENT_MINUTES="0", CREDITPFN_AUTO_REQUEUE="0")
    if phase == "recovery":
        from src.utils.submit_bounded import submit
        cmd = ["sbatch", "--parsable", "--clusters=mindwell", "--partition=gpu_b200",
               "--array=0-7%4", f"--time={os.environ.get('RECOVERY_WALLTIME', '00:30:00')}",
               f"--export=ALL,CREDITPFN_FLOW_ID={identifier},CREDITPFN_EXPERIMENT=experiment0,CREDITPFN_USE_SCRATCH=1",
               "scripts/slurm/recovery.slurm"]
        response = submit(cmd, slots=int(os.environ.get("GLOBAL_CONCURRENCY", "16")), limit=4, cluster="mindwell")
        print(f"Recovery array {response}", flush=True)
    else:
        # Null controls are short. Positive-LR pilots retain a one-hour rail until measured.
        env["WALLTIME"] = os.environ.get(f"{phase.upper()}_WALLTIME", "00:30:00" if phase == "null" else "01:00:00")
        if phase == "budget":
            env.pop("WALLTIME")
            # Part 1 has verified recovery before these longer runs may use it.
            # Each allocation requests only one segment plus the save/monitor margin.
            env["SEGMENT_MINUTES"] = os.environ.get("BUDGET_SEGMENT_MINUTES", "120")
            env["CREDITPFN_AUTO_REQUEUE"] = "1"
        for track in ("pd", "lgd"):
            subprocess.run(["bash", "scripts/slurm/run_experiment.sh", f"config/experiment0/{phase}_{track}.yaml"],
                           env=env, check=True)
    with locked(identifier) as (path, state):
        state["submissions_complete"] = True
        state["status"] = "running" if not state["failed"] else "failed"
        write_json(path, state)
        _advance(path, state)


def _advance(path: Path, state: dict):
    phase = state["phase"]
    expected = {f"{track}:{i}" for track in ("pd", "lgd") for i in range(COUNTS[phase])}
    if (state["status"] != "running" or not state.get("submissions_complete")
            or state["failed"] or set(state["done"]) != expected):
        return
    state["status"] = "submitting_audit"
    write_json(path, state)  # Uncertain sbatch acceptance must never be retried automatically.
    job = _cpu(state["id"], "audit", phase)
    state["jobs"].append({"phase": phase + "_audit", "cluster": "wice", "id": job})
    state["status"] = "auditing"
    write_json(path, state)


def complete(identifier: str, phase: str, track: str, trial: int, rc: int):
    key = f"{track}:{trial}"
    if phase not in COUNTS or track not in ("pd", "lgd") or not 0 <= trial < COUNTS[phase]:
        raise ValueError("Unexpected completion callback")
    with locked(identifier) as (path, state):
        if state["phase"] != phase:
            raise RuntimeError("Callback belongs to a different workflow phase")
        if rc:
            state["failed"] = sorted(set(state["failed"]) | {key})
            state["status"] = "failed"
        else:
            state["done"] = sorted(set(state["done"]) | {key})
        write_json(path, state)
        _advance(path, state)


def audit(identifier: str, phase: str):
    from src.utils.audit_experiment import audit as audit_trials
    with locked(identifier) as (_, state):
        if state["phase"] != phase or state["status"] != "auditing":
            raise RuntimeError("Audit does not belong to the current completed workflow stage")
    folder = root() / identifier
    if phase == "recovery":
        reports = [json.loads(p.read_text()) for p in sorted(folder.glob("recovery_*_*.json"))]
        if len(reports) != 8 or not all(r["passed"] for r in reports):
            raise RuntimeError("Recovery checks incomplete or unequal; workflow stopped")
    else:
        reports = [audit_trials(Path(f"config/experiment0/{phase}_{t}.yaml"), null=phase == "null") for t in ("pd", "lgd")]
        write_json(folder / f"{phase}_audit.json", reports)
        if not all(r["passed"] and r["diverged"] == 0 for r in reports):
            raise RuntimeError("Training audit failed or contains divergence; workflow stopped")
    if phase in ("null", "pilot"):
        launch(identifier, "pilot" if phase == "null" else "recovery")
    else:
        with locked(identifier) as (path, state):
            state["status"] = "passed"
            write_json(path, state)
            write_json(root() / f"{state['part']}_passed.json", state)
        print(f"Experiment 0 {state['part']} PASSED. No later experiments were submitted.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "prepare", "complete", "audit"))
    parser.add_argument("--part", choices=PARTS, default="part1")
    parser.add_argument("--id")
    parser.add_argument("--phase", choices=tuple(COUNTS))
    parser.add_argument("--track", choices=("pd", "lgd"))
    parser.add_argument("--trial", type=int)
    parser.add_argument("--rc", type=int, default=0)
    args = parser.parse_args(argv)
    if args.action == "start":
        start(args.part)
    else:
        try:
            if args.action == "prepare":
                prepare(args.id)
            elif args.action == "audit":
                audit(args.id, args.phase)
            else:
                complete(args.id, args.phase, args.track, args.trial, args.rc)
        except Exception:
            if args.id:
                with locked(args.id) as (path, state):
                    state["status"] = "failed"
                    write_json(path, state)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
