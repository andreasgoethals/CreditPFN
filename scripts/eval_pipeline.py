"""End-to-end orchestrator for the cross-model benchmark.

Runs the **final benchmark** that the project ultimately exists to
produce: every TabPFN variant (untuned + every continued-pretrained
checkpoint listed in the training manifest) plus the classical
baselines (XGBoost, CatBoost, LogReg/LinReg) scored on the **same
held-out test chunks** the training pipeline produced, with
``cv.n_folds`` of cross-validation per chunk.

Two execution modes
-------------------

1. **Local / single process (default)** — iterate every model on
   every test chunk in one process. Useful for small corpora and
   debugging::

       python scripts/eval_pipeline.py track=pd

2. **Slurm-array (one (model × test_dataset) per task)** — for the
   3 000-dataset corpus, iterating sequentially is too slow when
   XGBoost/CatBoost each spend several minutes on per-fold Optuna
   HPO. The script supports task indexing so each slurm array task
   processes ONE (model_name, test_dataset_id) pair::

       N=$(python scripts/eval_pipeline.py --list-tasks track=pd)
       sbatch --array=0-$((N - 1))%32 scripts/slurm/eval_pd.slurm

   Each task writes its own row(s) to
   ``results/<TRACK>/<method>/<run_name>_<timestamp>_<dataset>.csv``,
   so different tasks never write to the same file (no locking).

Optional filters
----------------
* ``--method <name>``         — score only this model.  Repeatable.
* ``--test-dataset <id>``     — score only this test dataset.  Repeatable.
* ``--task-index N``          — pick the Nth (method, dataset) pair.
* ``--list-tasks``            — print the total task count and exit.

CSV layout (long format, one row per model × chunk × fold)::

    track, task_type, model_name, model_source, model_path,
    test_dataset_id, test_chunk_idx, fold_idx,
    n_train_rows, n_test_rows,
    metric_name, metric_value, elapsed_sec, timestamp, status, error
"""

from __future__ import annotations

import argparse
import logging
import sys as _sys
import time
from itertools import product
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from src.utils.paths import resolve_output_path  # noqa: E402
from src.utils.run_log import resolve_run_log, setup_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cfg loading
# --------------------------------------------------------------------------- #


def _load_cfgs(eval_overrides: list[str], train_overrides: list[str]):
    """Load both yamls + apply CLI overrides.

    The eval cfg points to the train cfg's path so the test-split
    definition (seed, fractions, multi_chunk_policy,
    train_dataset_ids, test_dataset_ids) lives in one place.
    """
    from omegaconf import OmegaConf
    eval_cfg = OmegaConf.load("config/eval.yaml")
    if eval_overrides:
        eval_cfg = OmegaConf.merge(eval_cfg, OmegaConf.from_dotlist(eval_overrides))
    train_cfg = OmegaConf.load(eval_cfg.train_cfg_path)
    if train_overrides:
        train_cfg = OmegaConf.merge(train_cfg, OmegaConf.from_dotlist(train_overrides))
    return eval_cfg, train_cfg


# --------------------------------------------------------------------------- #
# Roster + tasks
# --------------------------------------------------------------------------- #


def _build_roster(eval_cfg, train_cfg, track: str):
    """Build (handles_and_models, test_chunks) used by both --list-tasks
    and run().
    """
    from src.train.corpus import split_from_cfg
    from src.model import build_baselines
    from src.eval.benchmark import load_trained_handles

    split = split_from_cfg(train_cfg, track=track)

    bases = (
        list(train_cfg.tunable.classifier_base_paths) if track == "pd"
        else list(train_cfg.tunable.regressor_base_paths)
    )
    hpo_xgb = dict(eval_cfg.hpo.xgboost)  if hasattr(eval_cfg, "hpo") else {}
    hpo_cb  = dict(eval_cfg.hpo.catboost) if hasattr(eval_cfg, "hpo") else {}
    baselines = build_baselines(
        track=track,
        base_paths_for_tabpfn_untuned=bases,
        enabled=list(eval_cfg.baselines.enabled),
        device=str(train_cfg.device),
        n_estimators_tabpfn=int(eval_cfg.tabpfn_n_estimators),
        seed=int(train_cfg.seed),
        hpo_xgboost=hpo_xgb,
        hpo_catboost=hpo_cb,
    )

    manifest_csv = (
        resolve_output_path("manifests") / f"{train_cfg.run_name}_{track}.csv"
    )
    trained = load_trained_handles(
        manifest_csv, track=track,
        device=str(train_cfg.device),
        n_estimators=int(eval_cfg.tabpfn_n_estimators),
    )

    return baselines + trained, split, manifest_csv


def _enumerate_tasks(handles_and_models, test_chunks):
    """Cartesian product of (model_index, dataset_id) — every parallel
    slurm-array task scores one (model × dataset) pair across all
    chunks of that dataset and all folds.

    We collapse on dataset_id (not chunk_idx) because chunks of the
    same parent dataset are typically tiny relative to per-fold HPO
    cost; running them together avoids re-fitting the model per chunk.
    """
    dataset_ids = sorted({c.dataset_id for c in test_chunks})
    return list(product(range(len(handles_and_models)), dataset_ids))


def _filter_roster(handles_and_models, test_chunks, *,
                   method_filter: list[str], dataset_filter: list[str],
                   task_index: int | None):
    """Apply --method / --test-dataset / --task-index filters.

    Filter precedence (innermost wins):
      1. --task-index N (pick THE Nth pair from the cartesian grid)
      2. --method / --test-dataset (intersect the lists)
    """
    pairs = _enumerate_tasks(handles_and_models, test_chunks)

    if task_index is not None:
        if not 0 <= task_index < len(pairs):
            raise IndexError(
                f"--task-index={task_index} is out of bounds; this cfg "
                f"has {len(pairs)} task(s) (indices 0..{len(pairs) - 1})."
            )
        m_idx, ds_id = pairs[task_index]
        kept_handle, kept_model = handles_and_models[m_idx]
        return [(kept_handle, kept_model)], [c for c in test_chunks if c.dataset_id == ds_id]

    keep_models = handles_and_models
    if method_filter:
        keep_models = [
            (h, m) for h, m in handles_and_models if h.name in method_filter
        ]
    keep_chunks = test_chunks
    if dataset_filter:
        keep_chunks = [c for c in test_chunks if c.dataset_id in dataset_filter]
    return keep_models, keep_chunks


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def run(
    eval_overrides: list[str] | None = None,
    train_overrides: list[str] | None = None,
    log_path: Path | str | None = None,
    method_filter: list[str] | None = None,
    dataset_filter: list[str] | None = None,
    task_index: int | None = None,
) -> int:
    eval_cfg, train_cfg = _load_cfgs(eval_overrides or [], train_overrides or [])
    track = str(train_cfg.track)

    log, _ = resolve_run_log(log_path, task_name=f"eval_{track}")
    setup_logging(log.path)
    LOGGER.info("eval_pipeline: log=%s  track=%s", log.path, track)

    handles_and_models, split, manifest_csv = _build_roster(eval_cfg, train_cfg, track)
    LOGGER.info("Test split: %s", split.summary)
    LOGGER.info("Roster: %d models (manifest=%s)",
                len(handles_and_models), manifest_csv)

    keep_models, keep_chunks = _filter_roster(
        handles_and_models, split.test,
        method_filter=method_filter or [],
        dataset_filter=dataset_filter or [],
        task_index=task_index,
    )
    LOGGER.info("After filtering: %d model(s) × %d test chunk(s)",
                len(keep_models), len(keep_chunks))

    metric_name = (
        eval_cfg.metric.classification if track == "pd"
        else eval_cfg.metric.regression
    )
    n_folds = int(eval_cfg.cv.n_folds) if hasattr(eval_cfg, "cv") else 5
    results_base = (
        eval_cfg.results.base_dir if hasattr(eval_cfg, "results") else "results"
    )

    from src.eval.benchmark import run_benchmark
    t0 = time.monotonic()
    rows = run_benchmark(
        test_chunks=keep_chunks,
        handles_and_models=keep_models,
        track=track,
        metric_name=metric_name,
        run_name=str(train_cfg.run_name),
        n_folds=n_folds,
        seed=int(train_cfg.seed),
        results_base_dir=results_base,
        # Per-task suffix for the per-method CSV (so two parallel slurm
        # tasks for the same method-but-different-datasets don't write
        # to the same file).
        per_task_tag=(
            f"task{task_index}_ds-{keep_chunks[0].dataset_id}"
            if task_index is not None and keep_chunks else None
        ),
    )
    elapsed = time.monotonic() - t0

    n_ok   = sum(1 for r in rows if r.status == "OK")
    n_fail = sum(1 for r in rows if r.status == "FAIL")
    line = (
        f"eval_pipeline: status={'OK' if n_fail == 0 else f'FAIL[{n_fail}]'}  "
        f"track={track}  metric={metric_name}  "
        f"models={len(keep_models)}  test_chunks={len(keep_chunks)}  "
        f"folds={n_folds}  cells_ok={n_ok}  cells_fail={n_fail}  "
        f"results_base={results_base}  elapsed={elapsed:.1f}s"
    )
    log.write(line)
    print(line)
    return 0 if n_fail == 0 else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Cross-model benchmark on the held-out test split.",
    )
    p.add_argument(
        "--log-path", default=None,
        help="Append the run summary to this log file (skip the auto-naming).",
    )
    p.add_argument(
        "--method", action="append", default=None,
        help="Score only this model (repeatable). Match against ``handle.name``.",
    )
    p.add_argument(
        "--test-dataset", action="append", default=None,
        help="Score only this test dataset_id (repeatable).",
    )
    p.add_argument(
        "--task-index", type=int, default=None,
        help="Pick the Nth (model × dataset) task. For SLURM arrays — set "
             "to $SLURM_ARRAY_TASK_ID. Mutually exclusive with --method "
             "and --test-dataset.",
    )
    p.add_argument(
        "--list-tasks", action="store_true",
        help="Print the total (model × dataset) task count for the current "
             "cfg and exit. Useful for sizing slurm arrays.",
    )
    args, unknown = p.parse_known_args(argv)

    # Hydra-style overrides: split between eval and train cfgs by prefix.
    eval_keys_prefixes = (
        "train_cfg_path", "baselines.", "metric.", "tabpfn_n_estimators",
        "cv.", "hpo.", "results.",
    )
    eval_overrides, train_overrides = [], []
    for a in unknown:
        if "=" not in a or a.startswith("-"):
            p.error(f"unrecognised argument: {a}")
        if any(a.startswith(pre) for pre in eval_keys_prefixes):
            eval_overrides.append(a)
        else:
            train_overrides.append(a)

    if args.task_index is not None and (args.method or args.test_dataset):
        p.error("--task-index cannot be combined with --method / --test-dataset")
    return args, eval_overrides, train_overrides


if __name__ == "__main__":
    args, eval_overrides, train_overrides = _parse_args()
    if args.list_tasks:
        eval_cfg, train_cfg = _load_cfgs(eval_overrides, train_overrides)
        track = str(train_cfg.track)
        # Suppress the FileHandler when --list-tasks (no log file needed).
        logging.basicConfig(level=logging.WARNING, force=True)
        handles_and_models, split, _ = _build_roster(eval_cfg, train_cfg, track)
        print(len(_enumerate_tasks(handles_and_models, split.test)))
        raise SystemExit(0)
    raise SystemExit(run(
        eval_overrides=eval_overrides,
        train_overrides=train_overrides,
        log_path=args.log_path,
        method_filter=args.method,
        dataset_filter=args.test_dataset,
        task_index=args.task_index,
    ))
