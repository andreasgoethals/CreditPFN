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

COUNTS = {"null": 8, "pilot": 16, "recovery": 4, "budget": 4, "auxiliary": 4}  # per track
PARTS = ("part1", "part2", "recovery", "pilot", "auxiliary")
BASES = ("v2", "v2.6", "v3", "tabicl")


def root() -> Path:
    return manifests_dir("experiment0") / "workflow"


def fingerprint() -> str:
    files = [REPO_ROOT / "config/train.yaml", REPO_ROOT / "config/retention.yaml",
             REPO_ROOT / "src/utils/experiment0.py", REPO_ROOT / "src/utils/recovery_check.py"]
    files += [REPO_ROOT / f"config/experiment0/{phase}_{track}.yaml"
              for phase in ("null", "pilot", "recovery") for track in ("pd", "lgd")]
    files += [REPO_ROOT / "config/experiment0/auxiliary.yaml", REPO_ROOT / "src/utils/auxiliary_check.py",
              REPO_ROOT / "scripts/slurm/auxiliary.slurm"]
    files += [REPO_ROOT / f"config/experiment{exp}/{track}.yaml" for exp in (1, 2, 3) for track in ("pd", "lgd")]
    return digest_json({"training_code": code_identity(), "evaluation_code": code_identity(stage="eval"),
                        "configs": {p.relative_to(REPO_ROOT).as_posix(): file_digest(p) for p in files}})


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


def _pilot_selection(bases: dict | None) -> dict:
    bases = bases or {}
    if set(bases) - {"pd", "lgd"}:
        raise ValueError("Pilot bases must be selected by pd/lgd track")
    selected = {}
    for track in ("pd", "lgd"):
        requested = list(bases.get(track, BASES))
        if not requested or len(set(requested)) != len(requested) or set(requested) - set(BASES):
            raise ValueError(f"Invalid {track} pilot base selection")
        selected[track] = [base for base in BASES if base in requested]
    return selected


def _phase_configs(identifier: str, phase: str, part: str) -> list[Path]:
    folder = root() / identifier if part == "pilot" else Path("config/experiment0")
    return [folder / f"{phase}_{track}.yaml" for track in ("pd", "lgd")]


def _make_pilot_configs(folder: Path, bases: dict):
    """Copy the existing pilot recipe, selecting bases without replacing old plans."""
    from omegaconf import OmegaConf
    from src.train.config import load_train_config
    for track, selected in _pilot_selection(bases).items():
        cfg = load_train_config(config_path=f"config/experiment0/pilot_{track}.yaml")
        axis = "classifier_base_paths" if track == "pd" else "regressor_base_paths"
        prefixes = ["tabicl-" if base == "tabicl" else f"tabpfn-{base}-" for base in selected]
        paths = [p for p in cfg.tunable[axis] if Path(p).name.startswith(tuple(prefixes))]
        if len(paths) != len(selected):
            raise ValueError(f"Pilot selection does not match configured {track} checkpoints")
        cfg.tunable[axis] = paths
        cfg.run_name += "_probe_" + folder.name[:8]
        with (folder / f"pilot_{track}.yaml").open("x", encoding="utf-8") as stream:
            OmegaConf.save(cfg, stream)


def _expected(state: dict) -> set[str]:
    counts = state.get("trial_counts", {track: COUNTS[state["phase"]] for track in ("pd", "lgd")})
    return {f"{track}:{i}" for track, count in counts.items() for i in range(count)}


def start(part: str, *, pilot_bases: dict | None = None) -> str:
    if part not in PARTS:
        raise ValueError(f"Unknown experiment-0 part: {part}")
    if pilot_bases and part != "pilot":
        raise ValueError("Base selection is only available for the standalone pilot stage")
    selected = _pilot_selection(pilot_bases) if part == "pilot" else None
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
    if selected is not None:
        state["pilot_bases"] = selected
    write_json(folder / "state.json", state, exclusive=True)
    # One workflow per part/source prevents duplicate writers to the same trial files.
    claim_identity = digest_json({"fingerprint": identity, "bases": selected}) if selected is not None else identity
    claim = root() / f"{part}_{claim_identity[:20]}.json"
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
    phases = {"part1": ("null", "pilot"), "part2": ("budget",), "recovery": (), "pilot": ("pilot",), "auxiliary": ()}[part]
    if part == "pilot":
        _make_pilot_configs(root() / identifier, state["pilot_bases"])
    configs = [config for phase in phases for config in _phase_configs(identifier, phase, part)]
    if part == "auxiliary":
        from src.utils.auxiliary_check import make_configs
        configs.extend(make_configs(root() / identifier))
    if part in ("part1", "recovery"):
        from src.utils.recovery_check import make_configs
        configs.extend(make_configs(root() / identifier, diagnostic=part == "recovery"))
    elif part == "part2":
        for phase in ("null", "pilot"):
            for track in ("pd", "lgd"):
                prepare_plan(Path(f"config/experiment0/{phase}_{track}.yaml"), check=True)
    if preflight([arg for config in configs for arg in ("--config", str(config))]):
        raise RuntimeError("CPU preflight failed; no GPU jobs submitted")
    for config in configs:
        prepare_plan(config, write=True)
    stage(Path(os.environ["VSC_SCRATCH_GPFS1"]) / "CreditPFN", write=True)
    launch(identifier, part if part in ("recovery", "auxiliary") else phases[0])


def launch(identifier: str, phase: str):
    with locked(identifier) as (path, state):
        configs = _phase_configs(identifier, phase, state["part"])
        if state["part"] == "pilot":
            from src.train.config import load_train_config, resolve_grid
            counts = {track: len(resolve_grid(load_train_config(config_path=str(config)), single=False))
                      for track, config in zip(("pd", "lgd"), configs)}
        else:
            counts = {track: COUNTS[phase] for track in ("pd", "lgd")}
        state.update(phase=phase, status="submitting", done=[], failed=[], submissions_complete=False)
        state["trial_counts"] = counts
        write_json(path, state)
    env = dict(os.environ, CREDITPFN_FLOW_ID=identifier, CREDITPFN_FLOW_PHASE=phase,
               CREDITPFN_EXPERIMENT="experiment0", CREDITPFN_USE_SCRATCH="1",
               CREDITPFN_REQUIRE_STAGING="1", STAGES="train", TRIALS_PER_TASK="1",
               SEGMENT_MINUTES="0", CREDITPFN_AUTO_REQUEUE="0")
    if phase in ("recovery", "auxiliary"):
        from src.utils.submit_bounded import submit
        cmd = ["sbatch", "--parsable", "--clusters=mindwell", "--partition=gpu_b200",
               "--array=0-7%4", f"--time={os.environ.get(phase.upper() + '_WALLTIME', '00:15:00' if phase == 'auxiliary' else '00:30:00')}",
               f"--export=ALL,CREDITPFN_FLOW_ID={identifier},CREDITPFN_EXPERIMENT=experiment0,CREDITPFN_USE_SCRATCH=1",
               f"scripts/slurm/{phase}.slurm"]
        response = submit(cmd, slots=int(os.environ.get("GLOBAL_CONCURRENCY", "16")), limit=4, cluster="mindwell")
        print(f"{phase.capitalize()} array {response}", flush=True)
        with locked(identifier) as (path, state):
            state["jobs"].append({"phase": phase, "cluster": "mindwell", "id": response.split(";", 1)[0]})
            write_json(path, state)
    else:
        # Null controls are short. Positive-LR pilots retain a one-hour rail until measured.
        env["WALLTIME"] = os.environ.get(f"{phase.upper()}_WALLTIME", "00:30:00" if phase == "null" else "01:00:00")
        if phase == "budget":
            env.pop("WALLTIME")
            # Part 1 has verified recovery before these longer runs may use it.
            # Each allocation requests only one segment plus the save/monitor margin.
            env["SEGMENT_MINUTES"] = os.environ.get("BUDGET_SEGMENT_MINUTES", "120")
            env["CREDITPFN_AUTO_REQUEUE"] = "1"
        for config in configs:
            subprocess.run(["bash", "scripts/slurm/run_experiment.sh", str(config)],
                           env=env, check=True)
    with locked(identifier) as (path, state):
        state["submissions_complete"] = True
        state["status"] = "running" if not state["failed"] else "failed"
        write_json(path, state)
        _advance(path, state)


def _advance(path: Path, state: dict):
    phase = state["phase"]
    expected = _expected(state)
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
    if phase not in COUNTS or track not in ("pd", "lgd") or trial < 0:
        raise ValueError("Unexpected completion callback")
    with locked(identifier) as (path, state):
        if state["phase"] != phase:
            raise RuntimeError("Callback belongs to a different workflow phase")
        if key not in _expected(state):
            raise ValueError("Unexpected completion callback")
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
        part = state["part"]
    folder = root() / identifier
    if phase == "auxiliary":
        from src.utils.auxiliary_check import audit_reports
        write_json(folder / "auxiliary_audit.json", audit_reports(folder))
    elif phase == "recovery":
        reports = [json.loads(p.read_text()) for p in sorted(folder.glob("recovery_*_*.json"))]
        if len(reports) != 8 or not all(r["passed"] for r in reports):
            raise RuntimeError("Recovery checks incomplete or unequal; workflow stopped")
    else:
        reports = [audit_trials(config, null=phase == "null")
                   for config in _phase_configs(identifier, phase, part)]
        write_json(folder / f"{phase}_audit.json", reports)
        if not all(r["passed"] and r["diverged"] == 0 for r in reports):
            raise RuntimeError("Training audit failed or contains divergence; workflow stopped")
    if phase in ("null", "pilot") and part != "pilot":
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
    parser.add_argument("--pd-bases", nargs="+", choices=BASES)
    parser.add_argument("--lgd-bases", nargs="+", choices=BASES)
    parser.add_argument("--id")
    parser.add_argument("--phase", choices=tuple(COUNTS))
    parser.add_argument("--track", choices=("pd", "lgd"))
    parser.add_argument("--trial", type=int)
    parser.add_argument("--rc", type=int, default=0)
    args = parser.parse_args(argv)
    selection = {track: bases for track in ("pd", "lgd") if (bases := getattr(args, f"{track}_bases")) is not None}
    if selection and (args.action != "start" or args.part != "pilot"):
        parser.error("Base selection is only available for start --part pilot")
    if args.action == "start":
        start(args.part, pilot_bases=selection or None)
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
