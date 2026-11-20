"""Experiment 0 releases are tied to a submitted run and its actual execution inputs."""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from src.utils.atomic import write_json
from src.utils.experiment import digest_json, file_digest
from src.utils.paths import REPO_ROOT

RUN_ENV = "CREDITPFN_PILOT_RUN"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def directory(root: Path, run: str) -> Path:
    if not re.fullmatch("[0-9a-f]{32}", run):
        raise ValueError("Invalid Experiment 0 run identifier")
    return root / "runs" / run


def identity(*, verify_project: bool = False) -> dict:
    from src.utils import runtime_validation, stage_inputs

    version = os.environ["CREDITPFN_RUNTIME_VERSION"]
    accepted = runtime_validation.require(version)
    override = os.environ.get("CREDITPFN_INPUT_POINTER")
    pointer = Path(override) if override else Path(os.environ["VSC_SCRATCH_GPFS1"]) / "CreditPFN/ACTIVE_INPUTS.json"
    snapshot = stage_inputs.resolve(pointer)
    inventory = read(snapshot / "inventory.json")
    if verify_project:
        from src.utils.paths import resolve_staging_path
        for name, entry in inventory["files"].items():
            if file_digest(resolve_staging_path(name), fresh=True) != entry["sha256"]:
                raise RuntimeError("PROJECT inputs have changed since scratch was staged; refresh the input snapshot")
            if file_digest(snapshot / name, fresh=True) != entry["sha256"]:
                raise RuntimeError("A staged scratch input has changed; stage a fresh input snapshot")
    if os.environ.get("CREDITPFN_USE_SCRATCH") == "1" and Path(os.environ["CREDITPFN_DATA_ROOT"]).resolve() != snapshot.resolve():
        raise RuntimeError("The worker's mounted inputs differ from the current input snapshot")
    files = sorted(p for base in ("config", "scripts", "tests") for p in (REPO_ROOT / base).rglob("*")
                   if p.suffix in {".yaml", ".py", ".sh", ".slurm"})
    return dict(runtime=version, runtime_manifest=accepted["manifest_sha256"], code=accepted["identity"]["code"],
                inputs=inventory["sha256"],
                protocol=digest_json({p.relative_to(REPO_ROOT).as_posix(): file_digest(p) for p in files}))


def with_dependencies(selection: dict | None) -> dict | None:
    """Refresh evaluations that use a checkpoint selected for retraining."""
    from src.utils.pilot_checks import MODELS_PER_TRACK, SCALE_PILOTS, TRACKS
    if not selection or not selection.get("pilots"):
        return selection
    result = dict(selection)
    for track_index, track in enumerate(TRACKS):
        models = range(track_index * MODELS_PER_TRACK, (track_index + 1) * MODELS_PER_TRACK)
        sources = {"evaluation": ("reference", "domain_control", "episode2048"), "scale": (SCALE_PILOTS[track],)}
        for kind, pilots in sources.items():
            if any(f"{pilot}_{track}.yaml" in selection["pilots"] for pilot in pilots):
                result[kind] = sorted(set(result.get(kind, [])) | set(models))
    return result


def begin(root: Path, selection: dict | None) -> str:
    """Invalidate the release before even preparing plans. Preserve past evidence by run."""
    from src.utils.pilot_checks import CHECK_COUNTS, FOLDER, pilot_paths
    if selection is not None:
        if not isinstance(selection, dict) or not any(selection.values()):
            raise ValueError("Experiment 0 requires a non-empty selection")
        allowed = {**{k: set(range(v)) for k, v in CHECK_COUNTS.items()},
                   "pilots": {p.name for p in pilot_paths(FOLDER)}, "classical": {"pd", "lgd"},
                   "prepare": {p.name for p in FOLDER.glob("*.yaml")}}
        for kind, values in selection.items():
            if (kind not in allowed or not isinstance(values, list) or not set(values) <= allowed[kind]
                    or (kind in CHECK_COUNTS and any(type(i) is not int for i in values))):
                raise ValueError(f"Invalid Experiment 0 selection for {kind}")
    selection = with_dependencies(selection)
    run = uuid.uuid4().hex
    active = root / "active.json"
    previous = read(active)["run"] if active.exists() else None
    write_json(active, {"run": run})
    spec = dict(run=run, evidence=run, identity=identity(verify_project=True), selection=selection, checks={})
    if previous and selection is not None:
        prior = directory(root, previous) / "request.json"
        if prior.is_file():
            parent = read(prior)
            if parent["identity"] == spec["identity"]:
                spec["evidence"] = parent["evidence"]
                spec["checks"].update(parent["checks"])
    for kind, count in CHECK_COUNTS.items():
        for task in range(count) if selection is None else selection.get(kind, []):
            spec["checks"][f"{kind}_{task:02d}"] = run
    write_json(directory(root, run) / "request.json", spec, exclusive=True)
    return run


def pending_controllers() -> bool:
    """Slurm's Command field omits script arguments. SubmitLine identifies the controller action."""
    import subprocess
    queued = subprocess.run(["squeue", "-M", "mindwell", "-u", os.environ["USER"], "-h",
                             "-n", "creditpfn-campaign-cpu", "-o", "%i"], check=True,
                            text=True, capture_output=True, timeout=60)
    ids = {j for j in queued.stdout.split() if j.isdigit() and j != os.environ.get("SLURM_JOB_ID")}
    if not ids:
        return False
    result = subprocess.run(["sacct", "-M", "mindwell", "-X", "-nP", "-j", ",".join(sorted(ids)),
                             "-o", "JobID,SubmitLine"], check=True, text=True, capture_output=True, timeout=60)
    # If accounting cannot identify another same-project controller, it cannot grant clearance.
    submitters = {line.partition("|")[0] for line in result.stdout.splitlines()
                  if re.search(r"campaign_cpu\.slurm\s+submit\s+E\d+\b", line)}
    return bool(ids - submitters)


def incomplete_jobs(expected: set[str]) -> list[str]:
    """Array elements may fail independently even though an afterany summary is released."""
    import subprocess
    if not expected:
        return []
    result = subprocess.run(["sacct", "-M", "mindwell", "-X", "-nP", "-j", ",".join(sorted(expected)),
                             "-o", "JobID,State,ExitCode"], check=True, text=True, capture_output=True, timeout=60)
    seen, problems = set(), []
    for line in result.stdout.splitlines():
        job, state, code = line.strip().split("|")
        parent = job.split("_", 1)[0]
        matched = expected.intersection((job, parent))
        if matched:
            seen.update(matched)
            if (state, code) != ("COMPLETED", "0:0"):
                problems.append(line)
    return sorted(problems + [f"{j}: no accounting record" for j in expected - seen])


def current(root: Path, run: str, *, verify_project: bool = False) -> dict:
    if read(root / "active.json")["run"] != run:
        raise RuntimeError("Experiment 0 was superseded by another controller")
    spec = read(directory(root, run) / "request.json")
    if spec["run"] != run or spec["identity"] != identity(verify_project=verify_project):
        raise RuntimeError("Experiment 0 code, configuration, runtime or input snapshot changed")
    return spec


def record_check(root: Path, name: str, record: dict) -> Path:
    run = os.environ.get(RUN_ENV)
    if run:
        spec = current(root, run)
        if spec["checks"].get(Path(name).stem) != run:
            raise RuntimeError(f"Check {name} was not requested by this controller")
        root = directory(root, run)
        record = {**record, "run": run, "array": os.environ.get("SLURM_ARRAY_JOB_ID")}
    path = root / name
    write_json(path, record)
    return path


def require(root: Path) -> dict:
    active = root / "active.json"
    if active.is_file():
        run = read(active)["run"]
        summary = directory(root, run) / "summary.json"
        if summary.is_file():
            result = read(summary)
            if result.get("passed") is True and result.get("run") == run:
                spec = current(root, run, verify_project=True)
                if result.get("identity") == spec["identity"]:
                    return spec
    raise RuntimeError(f"Experiment 0 has not passed for the current run and inputs ({active})")
