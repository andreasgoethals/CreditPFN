"""Prepare and submit the campaign from inside an accepted runtime job, one experiment at a time.

``prepare`` writes the immutable plan of every config in a folder. ``submit-experiment0`` sends
every Experiment 0 part at once. ``submit E<n>`` submits one experiment:

* training experiments: every arm config through ``scripts/slurm/run_experiment.sh`` with
  ``STAGES="train eval"`` (each partition's five-fold evaluation waits for its training with
  ``afterok``); E1 also queues the classical baselines once (they do not depend on CPT);
* E5 (target fine-tuning) and E6 (context curves): ``campaign_eval_gpu.slurm`` arrays over
  ``src.eval.campaign_tasks``, only after every E1 checkpoint (E6: and every E4 checkpoint) exists.

Mindwell allows 500 queued jobs per user and counts every array element (other projects of the
same user included). Each submission is sized first; when the next config would not fit under
``QUEUE_BUDGET`` the launcher stops cleanly and records what was submitted, so running the same
command later continues where it stopped. Children receive the scrubbed submission environment.

    python -m src.utils.campaign_launch prepare config/experiments/E00_pilots
    python -m src.utils.campaign_launch submit-experiment0
    python -m src.utils.campaign_launch submit E8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from src.utils.atomic import write_json
from src.utils.paths import REPO_ROOT

NO_CAP = "4096"  # lane pool far larger than any submission: Slurm admission is the only cap
QUEUE_BUDGET = 480  # of Mindwell's 500 queued jobs per user, keep 20 free for interactive work
EVAL_TASKS = 4      # five-fold evaluation shards per trained partition
# Rejected endpoint in the registered challenge extract, verified against LendingClub's
# 2007-2011 LoanStats3a: 25,428 unique matches, including 1,185 fully paid loans in class zero.
REJECTED_PD_0006_TARGET = "fda2cc44895adf82dc02e2e689ae6056cb75e55ae3aee41f34ad05fe4bc197d2"


def prepare(folder: Path, only: list[str] | None = None) -> list[dict]:
    from src.utils.prepare_experiment import prepare as prepare_config
    reports = []
    for path in sorted(Path(folder).glob("*.yaml")):
        if only is not None and path.name not in set(only):
            continue
        report = prepare_config(path, write=True)
        reports.append({"config": path.name, "run": report["run"], "track": report["track"],
                        "trials": report["training_trials"], "plan": report["plan"]})
        print(json.dumps(reports[-1]), flush=True)
    return reports


def pd_target_issue(dataset_id: str) -> str | None:
    """Reject the known challenge labels for PD 0006, including a global label flip."""
    import hashlib

    import numpy as np
    import pandas as pd

    from src.data.preprocessing import DATASET_METADATA
    from src.utils.paths import processed_dir

    if dataset_id.partition(".")[0] != "0006":
        return None
    target = DATASET_METADATA[dataset_id]["target_column"]
    fields = ["loan_amnt", "annual_inc", "int_rate", "dti"]
    frame = pd.read_csv(processed_dir("pd", dataset_id + ".sanitized.csv"), usecols=[*fields, target])
    # Fingerprint the membership of each label group, independent of row order or the names
    # of the labels. No individual loan identifiers or records are needed in the repository.
    profiles = pd.util.hash_pandas_object(frame[fields].astype("float64"), index=False)
    groups = [hashlib.sha256(np.sort(profiles.loc[index].to_numpy(dtype="<u8")).tobytes()).hexdigest()
              for index in frame.groupby(target, dropna=False).groups.values()]
    signature = hashlib.sha256(" ".join(sorted(groups)).encode("ascii")).hexdigest()
    if signature == REJECTED_PD_0006_TARGET:
        return ("PD 0006 has the challenge target contradicted by LendingClub terminal outcomes. "
                "A label flip also mixes fully paid and charged-off loans. Resolve the corpus before launching.")
    return None


def grid_preflight() -> dict:
    """Resolve all training arms and evaluation task lists on the campaign's real corpus, CPU only."""
    from src.eval import campaign_tasks as ct
    from src.utils.design import experiment_folder, load_design
    from src.utils.prepare_experiment import prepare as inspect
    design = load_design()
    configs = {name: [inspect(p) for p in sorted(experiment_folder(name, design).glob("*.yaml"))]
               for name in design["experiments"] if name not in ("E5", "E6")}
    if not all(configs.values()):
        raise RuntimeError("A training experiment has no generated configurations")
    tasks, problems = {}, []
    for kind in ct.KINDS:
        cells = ct.tasks(design, kind=kind)
        keys = {(t["track"], t["model"], t.get("source"), t.get("arm"), t["dataset_id"], t["fold"])
                for t in cells}
        if not keys or len(keys) != len(cells):
            raise RuntimeError(f"{kind} has missing or duplicate task identities")
        tasks[kind] = len(cells)
        if kind == "finetune":
            for dataset in sorted({t["dataset_id"] for t in cells if t["track"] == "pd"}):
                if issue := pd_target_issue(dataset):
                    problems.append(issue)
    return dict(configs={name: len(rows) for name, rows in configs.items()},
                training_runs={name: sum(r["training_trials"] for r in rows) for name, rows in configs.items()},
                evaluation_tasks=tasks, input_problems=problems, passed=not problems)


def submission_environment() -> dict:
    """A submitted batch starts on the host, not inside its submitting container.

    Preserve requested runtime/scientific settings, but never export parent
    container markers, container injection options or job-owned cache/log paths.
    Each child activates its own runtime and creates its own cache and log.
    """
    inherited_overrides = {"CREDITPFN_STOP_AFTER_UPDATES", "CREDITPFN_STOP_REQUESTED", "CREDITPFN_STOP_REQUEST_FILE",
                           "CREDITPFN_RUNTIME_ACCEPTANCE", "CREDITPFN_LEGACY_RUNTIME_COMPARISON",
                           "CREDITPFN_EVAL_TASK_INDEX", "CREDITPFN_EVAL_FAMILY",
                           "CREDITPFN_OWNED_CACHE", "CREDITPFN_ACTIVE_LOG"}
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("SBATCH_", "APPTAINER_", "SINGULARITY_",
                                   "APPTAINERENV_", "SINGULARITYENV_"))
            and key not in inherited_overrides}


def _environment(**extra) -> dict:
    env = submission_environment()
    # The controller runs with PROJECT inputs and its own container paths; children stage and
    # activate their own (scratch inputs, as in production).
    for key in ("CREDITPFN_USE_SCRATCH", "CREDITPFN_DATA_ROOT", "CREDITPFN_BASE_CACHE_ROOT",
                "PYTHONPATH", "LD_LIBRARY_PATH", "CREDITPFN_RUNTIME_ROOT", "CREDITPFN_PILOT_RUN",
                "CREDITPFN_PILOT_SET", "CREDITPFN_RELEASE_RUN"):
        env.pop(key, None)
    env["CREDITPFN_USE_SCRATCH"] = "1"
    if not re.fullmatch(r"[A-Za-z0-9]+", env.get("CREDITPFN_RUNTIME_VERSION", "")):
        raise RuntimeError("Submission requires the accepted CREDITPFN_RUNTIME_VERSION")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _job_ids(text: str) -> list[str]:
    return re.findall(r"Submitted train array (\d+)", text)


def _sbatch(args: list[str], env: dict) -> str:
    # A timeout is ambiguous: never retry a submission that might have succeeded.
    out = subprocess.run(["sbatch", "--parsable", *args], env=env, text=True,
                         capture_output=True, timeout=60, check=True).stdout.strip()
    job = out.split(";", 1)[0]
    if not job.isdigit():
        raise RuntimeError(f"Uncertain submission response {out!r}; inspect squeue")
    return job


class UncertainSubmission(RuntimeError):
    """A launcher failed or timed out after possibly submitting some arrays."""

    def __init__(self, message: str, output: str):
        super().__init__(message)
        self.job_ids = re.findall(r"^(\d+)(?:;\w+)?$", output, re.M) + _job_ids(output)


def _run_launcher(path: Path, env: dict) -> str:
    try:
        run = subprocess.run(["bash", "scripts/slurm/run_experiment.sh", str(path)], cwd=REPO_ROOT,
                             env=env, text=True, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode()
        raise UncertainSubmission(f"Launcher timed out for {path.name}", output) from exc
    print(run.stdout + run.stderr, flush=True)
    if run.returncode != 0:
        raise UncertainSubmission(f"Submission failed for {path.name} (STAGES={env.get('STAGES')})",
                                  run.stdout)
    return run.stdout


def _launch(state: dict, key: str, path: Path, env: dict, receipt: Path) -> str:
    """Run one launcher; a failure is recorded as uncertain (its printed job IDs kept) and the
    key is refused on later reruns until someone has inspected squeue and removed the entry."""
    if key in state.setdefault("uncertain", {}):
        raise RuntimeError(f"{key} may be partly submitted ({state['uncertain'][key]}); inspect squeue, "
                           f"cancel or keep its jobs, then delete its 'uncertain' entry in {receipt}")
    try:
        return _run_launcher(path, env)
    except UncertainSubmission as exc:
        state["uncertain"][key] = dict(error=str(exc), job_ids=exc.job_ids)
        write_json(receipt, state)
        raise


def submit_experiment0(folder: Path, receipt: Path, only: dict | None = None) -> dict:
    """Every Experiment 0 part, all independent work at once (no running cap).

    Pilots train; the two reference pilots are also benchmarked end to end (foundation
    five-fold on GPU after training, classical baselines on CPU). GPU checks run as arrays;
    the evaluation checks wait for the reference pilots; the summary waits for everything.

    ``only`` resubmits a part: {"pilots": [config names], "classical": [tracks], "recovery": [task
    indices], "capacity": [...], "precision": [...], "evaluation": [...]}; a missing key submits
    nothing of that kind. The summary always runs and covers every receipt.
    """
    from src.utils.pilot_checks import (
        CAPACITY_TASKS,
        EVALUATION_TASKS,
        PRECISION_TASKS,
        RECOVERY_TASKS,
        pilot_paths,
        recovery_case,
    )
    from src.utils import pilot_gate
    if receipt.exists():
        raise RuntimeError(f"Experiment 0 was already submitted; inspect {receipt}")
    root = receipt.parent
    run = os.environ.get(pilot_gate.RUN_ENV)
    if not run or not (pilot_gate.directory(root, run) / "preflight.json").is_file():
        raise RuntimeError("Submit Experiment 0 through campaign_cpu.slurm so its CPU preflight runs first")
    request = pilot_gate.current(root, run)
    if pilot_gate.read(pilot_gate.directory(root, run) / "preflight.json").get("passed") is not True:
        raise RuntimeError("Experiment 0 CPU preflight has not passed")
    only = pilot_gate.with_dependencies(only)
    if request["selection"] != only:
        raise RuntimeError("Experiment 0 selection differs from the current request")
    only = only or {}

    def chosen(kind: str, count: int) -> list[int]:
        if not only:
            return list(range(count))
        return sorted({int(i) for i in only.get(kind, []) if 0 <= int(i) < count})

    def array(indices: list[int]) -> str:
        return "--array=" + ",".join(str(i) for i in indices)
    env = _environment(CREDITPFN_EXPERIMENT="experiment0", SPLITS=1, THROTTLE=NO_CAP,
                       GLOBAL_CONCURRENCY=NO_CAP, EVAL_CONCURRENCY=NO_CAP, STAGES="train",
                       EVAL_TASKS=EVAL_TASKS,
                       # 250-update pilots measured 3-10 minutes on the B200 (09-11-2026); short
                       # requests backfill sooner. The reference benchmark (v3/TabICL at a
                       # 467,000-row context on the largest table) is still unmeasured.
                       WALLTIME="01:30:00", EVAL_WALLTIME="12:00:00")
    state = {"pilots": {}, "benchmark": {}, "classical": {}, "recovery": None, "capacity": None,
             "precision": None, "evaluation": None, "scale": None, "summary": None, "run": run}
    env[pilot_gate.RUN_ENV] = run
    env["CREDITPFN_PILOT_SET"] = request["evidence"]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        for path in pilot_paths(folder):
            if only and path.name not in set(only.get("pilots", [])):
                continue
            stages = "train eval" if path.name.startswith("reference_") else "train"
            out = _run_launcher(path, {**env, "STAGES": stages})
            state["pilots"][path.name] = _job_ids(out)
            if stages != "train":
                state["benchmark"][path.name] = re.findall(r"^(\d+)(?:;\w+)?$", out, re.M)
        for track in (("pd", "lgd") if not only else only.get("classical", [])):
            out = _run_launcher(folder / f"reference_{track}.yaml",
                                {**env, "STAGES": "eval", "EVAL_KIND": "classical"})
            state["classical"][track] = re.findall(r"^(\d+)(?:;\w+)?$", out, re.M)
        gpu = "scripts/slurm/pilot_gpu.slurm"
        for kind, count, extra in (("recovery", RECOVERY_TASKS, ["--time=01:00:00"]),
                                   ("capacity", CAPACITY_TASKS, []),
                                   ("precision", PRECISION_TASKS, ["--time=01:00:00"])):
            tasks = chosen(kind, count)
            if tasks:
                check_env = env
                if kind == "recovery" and request["evidence"] != run:
                    # A retry must start both resume arms empty; put its plans and weights
                    # in this attempt's directory, retaining the reference pilots for scoring.
                    names = {f"recovery_{variant}_{arm}_{track}.yaml" for task in tasks
                             for variant, track, _ in [recovery_case(task)]
                             for arm in ("uninterrupted", "resumed")}
                    subprocess.run([sys.executable, "-m", "src.utils.campaign_launch", "prepare", str(folder),
                                    "--only", json.dumps({"prepare": sorted(names)})],
                                   env={**os.environ, "CREDITPFN_PILOT_SET": run}, check=True)
                    check_env = {**env, "CREDITPFN_PILOT_SET": run}
                state[kind] = _sbatch([array(tasks), *extra, gpu, kind], check_env)
        # The evaluation check scores the reference pilots and, for E12, the 2,048-row episode pilot.
        reference = [j for name, jobs in state["pilots"].items()
                     if name.startswith(("reference_", "episode2048_", "domain_control_")) for j in jobs]
        tasks = chosen("evaluation", EVALUATION_TASKS)
        if tasks:
            state["evaluation"] = _sbatch(([f"--dependency=afterany:{':'.join(reference)}"] if reference else [])
                                          + [array(tasks), gpu, "evaluation"], env)
        tasks = chosen("scale", EVALUATION_TASKS)
        if tasks:
            state["scale"] = _sbatch(([f"--dependency=afterok:{':'.join(reference)}",
                                       "--kill-on-invalid-dep=yes"] if reference else [])
                                     + [array(tasks), f"--time={TASK_WALLTIME}", gpu, "scale"], env)
        after = ([j for jobs in state["pilots"].values() for j in jobs]
                 + [j for jobs in state["benchmark"].values() for j in jobs]
                 + [j for jobs in state["classical"].values() for j in jobs]
                 + [state[k] for k in ("recovery", "capacity", "precision", "evaluation", "scale") if state[k]])
        if os.environ.get("SLURM_JOB_ID"):
            after.append(os.environ["SLURM_JOB_ID"])
        state["summary"] = _sbatch(([f"--dependency=afterany:{':'.join(after)}"] if after else [])
                                   + ["scripts/slurm/campaign_cpu.slurm", "summary"], env)
    finally:
        write_json(receipt, state)
        write_json(pilot_gate.directory(root, run) / "jobs.json", state, exclusive=True)
    print(json.dumps(state, indent=2), flush=True)
    return state


def queued_jobs() -> int:
    """Array elements this user has queued or running on Mindwell (what the 500 limit counts)."""
    out = subprocess.run(["squeue", "--clusters=mindwell", "--user", os.environ["USER"], "--noheader",
                          "--array", "--format=%i"], text=True, capture_output=True, timeout=60, check=True)
    return sum(1 for line in out.stdout.splitlines() if line.strip() and not line.startswith("CLUSTER"))


def stages(path: Path) -> str:
    """Single-source runs (E7) are read from their monitor (every other table, one fixed split);
    a five-fold evaluation of 100 runs on all tables would cost more than their training."""
    from src.train.config import load_train_config
    single = load_train_config(config_path=str(path)).corpus.get("single_source", None)
    return "train" if single is not None else "train eval"


def config_cost(path: Path) -> int:
    """Queued jobs one arm config creates: one per model and partition, plus evaluation shards."""
    from src.train.config import load_train_config, resolve_grid
    cfg = load_train_config(config_path=str(path))
    partitions = int(cfg.corpus.n_splits)
    evaluation = EVAL_TASKS * partitions if stages(path) == "train eval" else 0
    return len(resolve_grid(cfg, single=False)) * partitions + evaluation


def _receipt(slug: str) -> Path:
    from src.utils.paths import manifests_dir
    return manifests_dir(slug) / "submission.json"


#: Walltime rails for campaign jobs until Experiment 0 has measured them; override with
#: CREDITPFN_TRAIN_WALLTIME / CREDITPFN_EVAL_WALLTIME (HH:MM:SS). The run_experiment.sh default
#: (two hours, no segments) is far below first-study timings for 26,000-row trials (2.1-2.4 h
#: with 2,000-row monitors; monitors are now 10,000 rows), and a timed-out training job
#: cancels its evaluation.
# Measured in Experiment 0 (16-11-2026): 250-update pilots 7 min median, 35 min at most (the
# campaign runs 2,500 updates, E10 also 10,000 and full passes); one PD five-fold evaluation shard
# 5.1-6.5 h, LGD 12-15 min. Without SEGMENT_MINUTES a job that reaches its walltime fails rather
# than requeues, so training keeps a wide margin.
DEFAULT_WALLTIMES = {"WALLTIME": "12:00:00", "EVAL_WALLTIME": "09:00:00"}
#: E5/E6/E12 task arrays (campaign_eval_gpu.slurm); E5 fine-tuning on the largest tables is not
#: yet measured, so the wrapper's 12 h stays the default. Override: CREDITPFN_TASK_WALLTIME.
TASK_WALLTIME = "12:00:00"


def campaign_walltimes() -> dict:
    values = {"WALLTIME": os.environ.get("CREDITPFN_TRAIN_WALLTIME", DEFAULT_WALLTIMES["WALLTIME"]),
              "EVAL_WALLTIME": os.environ.get("CREDITPFN_EVAL_WALLTIME", DEFAULT_WALLTIMES["EVAL_WALLTIME"])}
    for value in values.values():
        if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", value):
            raise ValueError(f"Walltime must be HH:MM:SS, got {value!r}")
    return values


def require_experiment0() -> dict:
    """Require current evidence, including when a newer controller is still queued."""
    from src.utils import pilot_gate
    from src.utils.pilot_checks import receipts
    request = pilot_gate.require(receipts())
    if os.environ.get("SLURM_JOB_ID") and pilot_gate.pending_controllers():
        raise RuntimeError("An Experiment 0 controller or summary is queued or running")
    return request


def submit_experiment(name: str, *, budget: int = QUEUE_BUDGET, queued=queued_jobs) -> dict:
    """Submit one experiment, within the queue budget; resumable by rerunning the same command."""
    release = require_experiment0()
    identity = release["identity"]
    from src.utils.design import experiment_folder, load_design
    design = load_design()
    spec = design["experiments"][name]
    slug, kind = spec["slug"], spec.get("kind", "training")
    receipt = _receipt(slug)
    # The launcher's own queue ceiling equals the controller's budget: a lower one would make it
    # wait (and be killed by the 15-minute launcher timeout) on work the controller just admitted.
    env = _environment(CREDITPFN_EXPERIMENT=slug, CREDITPFN_RELEASE_RUN=release["run"],
                       THROTTLE=NO_CAP, GLOBAL_CONCURRENCY=NO_CAP,
                       EVAL_CONCURRENCY=NO_CAP, EVAL_TASKS=EVAL_TASKS, MAX_QUEUED=str(budget),
                       **campaign_walltimes())
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # One controller per experiment at a time: two would both see the same unsubmitted work.
    lock = receipt.with_suffix(".lock")
    try:
        lock.open("x").close()
    except FileExistsError:
        raise RuntimeError(f"Another submission of {name} holds {lock}; if none runs (squeue), delete the lock") from None
    try:
        state = (json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists()
                 else {"submitted": {}, "experiment0_identity": identity})
        if state.get("experiment0_identity") != identity:
            raise RuntimeError("Existing campaign submission used different or unverified Experiment 0 evidence")
    except BaseException:
        lock.unlink(missing_ok=True)
        raise
    save = lambda: write_json(receipt, state)  # noqa: E731
    try:
        if kind in ("finetune", "inference"):
            _submit_evaluation_tasks("finetune" if slug == "target_finetune" else "context",
                                     state, env, budget, queued, save)
        else:
            configs = sorted(experiment_folder(name, design).glob("*.yaml"))
            prepare(experiment_folder(name, design))
            for path in configs:
                if path.name in state["submitted"]:
                    continue
                cost = config_cost(path)
                if queued() + cost > budget:
                    print(f"Queue budget reached before {path.name}; rerun later to continue.", flush=True)
                    break
                out = _launch(state, path.name, path, {**env, "STAGES": stages(path)}, receipt)
                state["submitted"][path.name] = dict(train=_job_ids(out), cost=cost)
                write_json(receipt, state)
            if "matched_context_evaluation" in spec and len(state["submitted"]) == len(configs):
                # E12: matched-context scoring once every episode-size checkpoint exists; rerun later otherwise.
                matched = state.setdefault("matched_context", {"submitted": {}})
                try:
                    _submit_evaluation_tasks("episode_context", matched, env, budget, queued, save)
                except RuntimeError as exc:
                    print(f"Matched-context evaluation not submitted yet: {exc}", flush=True)
            if slug == "reference" and len(state["submitted"]) == len(configs):
                classical = state.setdefault("classical", {})
                for track in ("pd", "lgd"):
                    if track in classical:
                        continue
                    out = _launch(state, f"classical_{track}", experiment_folder(name, design) / f"{track}_a0.yaml",
                                  {**env, "STAGES": "eval", "EVAL_KIND": "classical"}, receipt)
                    classical[track] = re.findall(r"^(\d+)(?:;\w+)?$", out, re.M)
                    write_json(receipt, state)
    finally:
        write_json(receipt, state)
        lock.unlink(missing_ok=True)
    print(json.dumps(state, indent=2), flush=True)
    return state


def _submit_evaluation_tasks(kind, state, env, budget, queued, save=lambda: None):
    """Chunks of an evaluation task array, only once every checkpoint it scores exists.

    Each chunk is recorded as ``uncertain`` (and saved) before ``sbatch``: a timeout after Slurm
    accepted it then leaves a marker that blocks resubmission until squeue has been inspected."""
    uncertain = [k for k, v in state.get("submitted", {}).items() if v == "uncertain"]
    if uncertain:
        raise RuntimeError(f"{kind}: chunks {uncertain} may have been submitted; inspect squeue, then set "
                           "their job ids (or delete the entries) in the receipt")
    from src.eval import campaign_tasks as ct
    from src.utils.design import config_path
    tasks = ct.tasks(kind=kind)

    def references(task):
        return ({"pd": config_path("episode_size", "pd", task["arm"])} if kind == "episode_context" else None)
    checkpoints = {(t.get("source", "reference"), t["track"], t["model"], t["partition"], t.get("arm")): t
                   for t in tasks}
    missing = [key for key, task in checkpoints.items()
                if not _exists(lambda task=task: ct.cpt_checkpoint(task, references(task)))]
    if missing:
        raise RuntimeError(f"{len(missing)} checkpoints are missing; finish their training first: {missing[:3]}")
    ct.write_tasks_cache(kind, tasks)   # workers read their task from it (no corpus re-read)
    walltime = os.environ.get("CREDITPFN_TASK_WALLTIME", TASK_WALLTIME)
    if not re.fullmatch(r"\d{1,3}:[0-5]\d:[0-5]\d", walltime):
        raise ValueError(f"CREDITPFN_TASK_WALLTIME must be HH:MM:SS, got {walltime!r}")
    start = int(state.get("next_task", 0))
    while start < len(tasks):
        room = budget - queued()
        if room <= 0:
            print("Queue budget reached; rerun later to continue.", flush=True)
            break
        stop = min(len(tasks), start + room, start + 1000)   # Slurm MaxArraySize is 1001
        # Array indices restart at 0 for every chunk; the wrapper adds TASK_OFFSET back.
        key = f"{start}-{stop - 1}"
        state.setdefault("submitted", {})[key] = "uncertain"
        save()
        job = _sbatch([f"--array=0-{stop - start - 1}", f"--time={walltime}", f"--export=ALL,TASK_OFFSET={start}",
                       "scripts/slurm/campaign_eval_gpu.slurm", kind], env)
        state["submitted"][key] = job
        state["next_task"] = start = stop
        save()


def _exists(resolve) -> bool:
    try:
        resolve()
        return True
    except (FileNotFoundError, RuntimeError):
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("begin", "preflight", "prepare", "submit-experiment0", "rerun-experiment0", "submit"))
    parser.add_argument("--only", help="rerun-experiment0: JSON {pilots, prepare, classical, recovery, capacity, "
                                        "precision, evaluation, scale}")
    parser.add_argument("target", nargs="?", default=str(REPO_ROOT / "config/experiments/E00_pilots"),
                        help="folder for prepare/submit-experiment0; experiment number (E1..E12) for submit")
    args = parser.parse_args(argv)
    if args.action == "begin":
        from src.utils import pilot_gate
        from src.utils.pilot_checks import receipts
        root = receipts()
        run = pilot_gate.begin(root, json.loads(args.only) if args.only else None)
        print(run, pilot_gate.read(pilot_gate.directory(root, run) / "request.json")["evidence"])
    elif args.action == "preflight":
        from src.utils import pilot_gate
        from src.utils.pilot_checks import receipts
        run = os.environ[pilot_gate.RUN_ENV]
        pilot_gate.current(receipts(), run)
        report = grid_preflight()
        write_json(pilot_gate.directory(receipts(), run) / "preflight.json", report, exclusive=True)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            return 1
    elif args.action == "prepare":
        selection = json.loads(args.only) if args.only else None
        # A selection prepares only the pilots it names (an evaluation-only rerun prepares none).
        prepare(Path(args.target), only=(list(selection.get("pilots", [])) + list(selection.get("prepare", [])))
                if selection is not None else None)
    elif args.action == "rerun-experiment0":
        from uuid import uuid4
        from src.utils.pilot_checks import receipts
        stamp = uuid4().hex
        selection = json.loads(args.only or "{}")
        if not any(selection.values()):
            raise SystemExit("rerun-experiment0 needs a non-empty selection; an empty one would rerun everything")
        submit_experiment0(Path(args.target), receipts() / f"submission_rerun_{stamp}.json", only=selection)
    elif args.action == "submit-experiment0":
        from src.utils.pilot_checks import receipts
        from uuid import uuid4
        root = receipts()
        receipt = root / "submission.json"
        if receipt.exists():
            receipt = root / f"submission_full_{uuid4().hex}.json"
        submit_experiment0(Path(args.target), receipt)
    else:
        submit_experiment(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
