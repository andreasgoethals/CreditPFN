"""Resume admission from a separate Git checkout without changing scientific source.

Execute this file by absolute path. Imports are deliberately delayed until the
frozen scientific checkout has been selected. Original plans/receipts stay intact;
replacement jobs and actual dependency translations have separate durable records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess


def live_tasks(text):
    result = set()
    for line in text.splitlines():
        parent, job, state = line.split("|")
        if "_" in job:
            result.add((parent, int(job.rsplit("_", 1)[1])))
    return result


def unfinished(report, split):
    if report["problems"] or report["diverged"]:
        raise RuntimeError("Scientific artifact audit failed")
    width, remainder = divmod(report["expected"], 4)
    if remainder or len(report["trials"]) != report["expected"]:
        raise RuntimeError("Unexpected prepared trial roster")
    rows = report["trials"][split * width:(split + 1) * width]
    if any(r["status"] not in ("OK", "PENDING") for r in rows):
        raise RuntimeError("Unexpected scientific trial state")
    return [i for i, row in enumerate(rows) if row["status"] == "PENDING"]


def replacement_plan(plan, accepted, reports, live, states):
    """Select terminal, incomplete tasks only; never trust TIMEOUT over artifacts."""
    result = []
    terminal = {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
    for item in plan:
        key = item["key"]
        if key.endswith("_benchmark") or key.startswith("e10_"):
            continue
        config, split = key.rsplit("_s", 1)
        missing = unfinished(reports[config], int(split))
        owned = [r for r in accepted if r["key"] == key]
        selected = []
        for task in missing:
            owners = [r for r in owned if task in r["tasks"]]
            if len(owners) != 1:
                raise RuntimeError("Missing or ambiguous original task submission")
            parent = owners[0]["id"]
            if (parent, task) in live:
                continue
            state = (states.get(f"{parent}_{task}", "").split() or [""])[0]
            if state not in terminal:
                raise RuntimeError(f"Unconfirmed missing training task: {parent}_{task}: {state}")
            selected.append(task)
        if selected:
            result.append(dict(deepcopy(item), tasks=selected))
    return result


def dependencies(parent, item, report, live, replacements):
    """Depend only on unfinished tasks which still exist in the live controller."""
    split = int(item["key"].rsplit("_s", 1)[1])
    result = []
    for task in unfinished(report, split):
        candidates = [r["id"] for r in replacements
                      if r["key"] == item["key"] and task in r["tasks"]]
        if len(candidates) > 1:
            raise RuntimeError("Duplicate replacement ownership")
        owner = candidates[0] if candidates else parent
        if (owner, task) not in live:
            raise RuntimeError(f"Incomplete training task is not live: {owner}_{task}")
        result.append(f"{owner}_{task}")
    return result


def translate(command, resolve, barrier):
    """Leave resources, scientific arguments and tasks byte-for-byte unchanged."""
    result, changes = [], []
    dispatch = "scripts/slurm/recovery_cpu.slurm" in command and "dispatch" in command
    for arg in command:
        if arg.startswith("--dependency=afterok:"):
            previous = arg.split(":")[1:]
            current = sorted({v for parent in previous for v in resolve(parent)})
            if current:
                result.append("--dependency=afterok:" + ":".join(current))
            changes.append(dict(kind="afterok", before=previous, after=current))
        elif dispatch and arg.startswith("--dependency=afterany:"):
            continue  # A final barrier is rebuilt from actual live owned arrays below.
        else:
            result.append(arg)
    if dispatch:
        current = sorted(barrier())
        if current:
            result.insert(1, "--dependency=afterany:" + ":".join(current))
        changes.append(dict(kind="afterany", after=current))
    return result, changes


def run(checkout, recovery):
    checkout = checkout.resolve()
    os.chdir(checkout)
    import sys
    sys.path.insert(0, str(checkout))
    from src.utils import campaign_recovery as cr, legacy_submit as submit
    from src.utils.legacy_design import scope, root as extension_root
    from src.utils.source_transition import authorization
    from src.utils.audit_experiment import audit
    from src.utils.atomic import write_json
    from src.utils.experiment import file_digest
    folder = cr.root(recovery)
    cfg = cr.record(recovery)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != cfg["source"]:
        raise RuntimeError("Frozen scientific checkout changed")
    with cr.writer(folder / "outage-resume.lease"):
        with cr.writer(folder / "research.lease"):
            pass  # Refuse a still-running original submitter before creating replacements.
        authorization(cfg["original"])
        workflow = json.loads((extension_root(cfg["original"]) / "workflow.json").read_text())
        plan = json.loads((folder / "research.plan.json").read_text())
        state = json.loads((folder / "research.submission.json").read_text())
        if state["in_flight"] is not None:
            raise RuntimeError("Ambiguous original submission; inspect before retrying")
        if state["plan"] != plan:
            raise RuntimeError("Original admission plan changed")
        lineage = folder / "outage-resume-source.json"
        binding = dict(scientific_source=actual, original_plan_sha256=file_digest(folder / "research.plan.json"),
                       helper_sha256=file_digest(Path(__file__)))
        if lineage.exists():
            if json.loads(lineage.read_text()) != binding:
                raise RuntimeError("Outage controller source/plan changed")
        else:
            write_json(lineage, binding, exclusive=True)
        reports = {}

        def report(key):
            with scope(cfg["original"]):
                reports[key] = audit(Path(workflow["production"][key]))
            if reports[key]["problems"] or reports[key]["diverged"]:
                raise RuntimeError(f"Artifact audit failed: {key}")
            print(f"Artifact audit {key}: {reports[key]['completed']}/{reports[key]['expected']} complete", flush=True)
            return reports[key]

        for key in workflow["production"]:
            if not key.startswith("e10_"):
                report(key)
        write_json(folder / "outage-training-audit.json", reports)

        def queue():
            text = subprocess.check_output(["squeue", "-M", "mindwell", "-u", os.environ["USER"],
                "-r", "-h", "-o", "%F|%i|%T"], text=True, timeout=30)
            return live_tasks(text)

        replacement_file = folder / "outage-replacements.submission.json"
        roster = folder / "outage-replacements.plan.json"
        if roster.exists():
            repairs = json.loads(roster.read_text())
            bykey = {item["key"]: item for item in plan}
            for item in repairs:
                expected = dict(deepcopy(bykey[item["key"]]), tasks=item["tasks"])
                if item != expected or not set(item["tasks"]).issubset(bykey[item["key"]]["tasks"]):
                    raise RuntimeError("Replacement roster changed scientific submission arguments")
        else:
            ids = sorted({v["id"] for v in state["accepted"] if not v["key"].endswith("_benchmark")})
            text = subprocess.check_output(["sacct", "-M", "mindwell", "-j", ",".join(ids),
                "-X", "-n", "-P", "-o", "JobID,State"], text=True, timeout=30)
            accounting = dict(line.split("|")[:2] for line in text.splitlines() if line)
            repairs = replacement_plan(plan, state["accepted"], reports, queue(), accounting)
            # Every selected task must retain an actual durable recovery checkpoint.
            for item in repairs:
                key, split = item["key"].rsplit("_s", 1)
                rows = reports[key]["trials"]
                width = len(rows) // 4
                config = workflow["production"][key]
                from src.train.config import load_train_config
                from src.utils.paths import resolve_staging_path
                with scope(cfg["original"]):
                    model_cfg = load_train_config(config_path=config)
                    directory = resolve_staging_path(model_cfg.checkpoint.trained_dir) / model_cfg.track
                for task in item["tasks"]:
                    checkpoint = directory / (rows[int(split)*width+task]["trial"] + ".ckpt.resume.pt")
                    if not checkpoint.is_file():
                        raise RuntimeError(f"Missing durable recovery checkpoint: {checkpoint.name}")
            write_json(roster, repairs, exclusive=True)
        real_submit = submit.submit_plan
        replaced = real_submit(repairs, replacement_file, sleep=cr.interruptible_sleep)
        replacements = replaced["accepted"]
        original = {v["id"]: next(i for i in plan if i["key"] == v["key"])
                    for v in state["accepted"] if not v["key"].endswith("_benchmark") and not v["key"].startswith("e10_")}

        def resolve(parent):
            if parent not in original:
                raise RuntimeError(f"Unknown training prerequisite: {parent}")
            item = original[parent]
            config = item["key"].rsplit("_s", 1)[0]
            # Re-read artifacts each time: a dependency can finish and be purged while
            # admission waits at the site's real submitted-job ceiling.
            return dependencies(parent, item, report(config), queue(), replacements)

        def barrier():
            current = json.loads((folder / "research.submission.json").read_text())
            owned = {r["id"] for r in current["accepted"] + replacements}
            return {parent for parent, task in queue() if parent in owned}

        journal = folder / "outage-admission-transports"
        journal.mkdir(exist_ok=True)

        def runner(command, **kwargs):
            if command[0] != "sbatch":
                return subprocess.run(command, **kwargs)
            try:
                actual_command, changes = translate(command, resolve, barrier)
            except Exception as exc:
                # No RPC occurred. Let the durable submitter record a definite rejection,
                # rather than leaving an ambiguous in-flight record for a local refusal.
                return subprocess.CompletedProcess(command, 1, stdout="",
                    stderr=f"Dependency reconciliation refused before sbatch: {exc}")
            import uuid
            path = journal / (uuid.uuid4().hex + ".json")
            record = dict(planned_command=command, actual_command=actual_command, changes=changes,
                          science_source=actual, helper_sha256=file_digest(Path(__file__)), outcome=None)
            write_json(path, record, exclusive=True)
            try:
                result = subprocess.run(actual_command, **kwargs)
            except BaseException as exc:
                record["uncertain"] = type(exc).__name__
                write_json(path, record)
                raise
            record["outcome"] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            write_json(path, record)
            return result

        def resumed(plan, destination, **kwargs):
            return real_submit(plan, destination, runner=runner, **kwargs)

        submit.submit_plan = resumed
        cr.recover(recovery)
        write_json(folder / "outage-admission-complete.json", dict(passed=True,
            scientific_source=actual, helper_sha256=file_digest(Path(__file__)),
            replacements_sha256=file_digest(replacement_file),
            research_sha256=file_digest(folder / "research.submission.json"),
            finish_sha256=file_digest(folder / "finish.submission.json")))
        print("PASS: remaining admissions accepted; live replacement jobs included in final barrier.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--recovery", required=True)
    args = parser.parse_args()
    run(args.checkout, args.recovery)
