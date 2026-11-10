"""Prepare and submit the campaign from inside an accepted runtime job, one experiment at a time.

``prepare`` writes the immutable plan of every config in a folder. ``submit-experiment0`` sends
every Experiment 0 part at once. ``submit E<n>`` submits one experiment:

* training experiments: every arm config through ``scripts/slurm/run_experiment.sh`` with
  ``STAGES="train eval"`` (each partition's five-fold evaluation waits for its training with
  ``afterok``); E1 also queues the classical baselines once (they do not depend on CPT);
* E4 (target fine-tuning) and E11 (context curves): ``campaign_eval_gpu.slurm`` arrays over
  ``src.eval.campaign_tasks``, only after every E1 checkpoint exists.

Mindwell allows 500 queued jobs per user and counts every array element (other projects of the
same user included). Each submission is sized first; when the next config would not fit under
``QUEUE_BUDGET`` the launcher stops cleanly and records what was submitted, so running the same
command later continues where it stopped. Children receive the scrubbed submission environment.

    python -m src.utils.campaign_launch prepare config/experiments/E00_pilots
    python -m src.utils.campaign_launch submit-experiment0
    python -m src.utils.campaign_launch submit E5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from src.utils.atomic import write_json
from src.utils.paths import REPO_ROOT

NO_CAP = "4096"  # lane pool far larger than any submission: Slurm admission is the only cap
QUEUE_BUDGET = 480  # of Mindwell's 500 queued jobs per user, keep 20 free for interactive work
EVAL_TASKS = 4      # five-fold evaluation shards per trained partition


def prepare(folder: Path) -> list[dict]:
    from src.utils.prepare_experiment import prepare as prepare_config
    reports = []
    for path in sorted(Path(folder).glob("*.yaml")):
        report = prepare_config(path, write=True)
        reports.append({"config": path.name, "run": report["run"], "track": report["track"],
                        "trials": report["training_trials"], "plan": report["plan"]})
        print(json.dumps(reports[-1]), flush=True)
    return reports


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
                "PYTHONPATH", "LD_LIBRARY_PATH", "CREDITPFN_RUNTIME_ROOT"):
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


def submit_experiment0(folder: Path, receipt: Path) -> dict:
    """Every Experiment 0 part, all independent work at once (no running cap).

    Pilots train; the two reference pilots are also benchmarked end to end (foundation
    five-fold on GPU after training, classical baselines on CPU). GPU checks run as arrays;
    the evaluation checks wait for the reference pilots; the summary waits for everything.
    """
    from src.utils.pilot_checks import (
        CAPACITY_TASKS,
        EVALUATION_TASKS,
        PRECISION_TASKS,
        RECOVERY_TASKS,
        pilot_paths,
    )
    if receipt.exists():
        raise RuntimeError(f"Experiment 0 was already submitted; inspect {receipt}")
    env = _environment(CREDITPFN_EXPERIMENT="experiment0", SPLITS=1, THROTTLE=NO_CAP,
                       GLOBAL_CONCURRENCY=NO_CAP, EVAL_CONCURRENCY=NO_CAP, STAGES="train",
                       EVAL_TASKS=EVAL_TASKS,
                       # 250-update pilots measured 3-10 minutes on the B200 (09-11-2026); short
                       # requests backfill sooner. The reference benchmark (v3/TabICL at a
                       # 467,000-row context on the largest table) is still unmeasured.
                       WALLTIME="01:30:00", EVAL_WALLTIME="12:00:00")
    state = {"pilots": {}, "benchmark": {}, "classical": {}, "recovery": None, "capacity": None,
             "precision": None, "evaluation": None, "summary": None}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        for path in pilot_paths(folder):
            stages = "train eval" if path.name.startswith("reference_") else "train"
            out = _run_launcher(path, {**env, "STAGES": stages})
            state["pilots"][path.name] = _job_ids(out)
            if stages != "train":
                state["benchmark"][path.name] = re.findall(r"^(\d+)(?:;\w+)?$", out, re.M)
        for track in ("pd", "lgd"):
            out = _run_launcher(folder / f"reference_{track}.yaml",
                                {**env, "STAGES": "eval", "EVAL_KIND": "classical"})
            state["classical"][track] = re.findall(r"^(\d+)(?:;\w+)?$", out, re.M)
        gpu = "scripts/slurm/pilot_gpu.slurm"
        state["recovery"] = _sbatch([f"--array=0-{RECOVERY_TASKS - 1}", "--time=01:00:00", gpu, "recovery"], env)
        state["capacity"] = _sbatch([f"--array=0-{CAPACITY_TASKS - 1}", gpu, "capacity"], env)
        state["precision"] = _sbatch([f"--array=0-{PRECISION_TASKS - 1}", "--time=01:00:00", gpu, "precision"], env)
        # The evaluation check scores the reference pilots and, for E10, the 2,048-row episode pilot.
        reference = [j for name, jobs in state["pilots"].items()
                     if name.startswith(("reference_", "episode2048_")) for j in jobs]
        state["evaluation"] = _sbatch([f"--dependency=afterany:{':'.join(reference)}",
                                       f"--array=0-{EVALUATION_TASKS - 1}", gpu, "evaluation"], env)
        after = ([j for jobs in state["pilots"].values() for j in jobs]
                 + [j for jobs in state["benchmark"].values() for j in jobs]
                 + [j for jobs in state["classical"].values() for j in jobs]
                 + [state["recovery"], state["capacity"], state["precision"], state["evaluation"]])
        state["summary"] = _sbatch([f"--dependency=afterany:{':'.join(after)}",
                                    "scripts/slurm/campaign_cpu.slurm", "summary"], env)
    finally:
        write_json(receipt, state)
    print(json.dumps(state, indent=2), flush=True)
    return state


def queued_jobs() -> int:
    """Array elements this user has queued or running on Mindwell (what the 500 limit counts)."""
    out = subprocess.run(["squeue", "--clusters=mindwell", "--user", os.environ["USER"], "--noheader",
                          "--array", "--format=%i"], text=True, capture_output=True, timeout=60, check=True)
    return sum(1 for line in out.stdout.splitlines() if line.strip() and not line.startswith("CLUSTER"))


def stages(path: Path) -> str:
    """Single-source runs (E12) are read from their monitor (every other table, one fixed split);
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
DEFAULT_WALLTIMES = {"WALLTIME": "24:00:00", "EVAL_WALLTIME": "24:00:00"}


def campaign_walltimes() -> dict:
    values = {"WALLTIME": os.environ.get("CREDITPFN_TRAIN_WALLTIME", DEFAULT_WALLTIMES["WALLTIME"]),
              "EVAL_WALLTIME": os.environ.get("CREDITPFN_EVAL_WALLTIME", DEFAULT_WALLTIMES["EVAL_WALLTIME"])}
    for value in values.values():
        if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", value):
            raise ValueError(f"Walltime must be HH:MM:SS, got {value!r}")
    return values


def require_experiment0() -> None:
    """Campaign work starts only after a passing Experiment 0 summary."""
    from src.utils.pilot_checks import receipts
    path = receipts() / "summary.json"
    if not path.is_file() or not json.loads(path.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError(f"Experiment 0 has not passed ({path}); run and inspect it first")


def submit_experiment(name: str, *, budget: int = QUEUE_BUDGET, queued=queued_jobs) -> dict:
    """Submit one experiment, within the queue budget; resumable by rerunning the same command."""
    require_experiment0()
    from src.utils.design import experiment_folder, load_design
    design = load_design()
    spec = design["experiments"][name]
    slug, kind = spec["slug"], spec.get("kind", "training")
    receipt = _receipt(slug)
    state = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {"submitted": {}}
    env = _environment(CREDITPFN_EXPERIMENT=slug, THROTTLE=NO_CAP, GLOBAL_CONCURRENCY=NO_CAP,
                       EVAL_CONCURRENCY=NO_CAP, EVAL_TASKS=EVAL_TASKS, **campaign_walltimes())
    receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        if kind in ("finetune", "inference"):
            _submit_evaluation_tasks("finetune" if slug == "target_finetune" else "context",
                                     state, env, budget, queued)
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
                # E10: matched-context scoring once every episode-size checkpoint exists; rerun later otherwise.
                matched = state.setdefault("matched_context", {"submitted": {}})
                try:
                    _submit_evaluation_tasks("episode_context", matched, env, budget, queued)
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
    print(json.dumps(state, indent=2), flush=True)
    return state


def _submit_evaluation_tasks(kind, state, env, budget, queued):
    """Chunks of an evaluation task array, only once every checkpoint it scores exists."""
    from src.eval import campaign_tasks as ct
    from src.utils.design import config_path
    tasks = ct.tasks(kind=kind)

    def references(task):
        return ({"pd": config_path("episode_size", "pd", task["arm"])} if kind == "episode_context" else None)
    missing = sorted({(t["track"], t["model"], t["partition"], t.get("arm")) for t in tasks
                      if not _exists(lambda: ct.cpt_checkpoint(t, references(t)))})
    if missing:
        raise RuntimeError(f"{len(missing)} checkpoints are missing; finish their training first: {missing[:3]}")
    start = int(state.get("next_task", 0))
    while start < len(tasks):
        room = budget - queued()
        if room <= 0:
            print("Queue budget reached; rerun later to continue.", flush=True)
            break
        stop = min(len(tasks), start + room, start + 1000)   # Slurm MaxArraySize is 1001
        # Array indices restart at 0 for every chunk; the wrapper adds TASK_OFFSET back.
        job = _sbatch([f"--array=0-{stop - start - 1}", f"--export=ALL,TASK_OFFSET={start}",
                       "scripts/slurm/campaign_eval_gpu.slurm", kind], env)
        state["submitted"][f"{start}-{stop - 1}"] = job
        state["next_task"] = start = stop


def _exists(resolve) -> bool:
    try:
        resolve()
        return True
    except (FileNotFoundError, RuntimeError):
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit-experiment0", "submit"))
    parser.add_argument("target", nargs="?", default=str(REPO_ROOT / "config/experiments/E00_pilots"),
                        help="folder for prepare/submit-experiment0; experiment number (E1..E12) for submit")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(Path(args.target))
    elif args.action == "submit-experiment0":
        from src.utils.pilot_checks import receipts
        submit_experiment0(Path(args.target), receipts() / "submission.json")
    else:
        submit_experiment(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
