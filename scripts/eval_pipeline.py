"""End-to-end orchestrator for the cross-model benchmark.

Scores every model on every held-out test dataset. The eval reads
the **processed CSVs** under ``data/processed/{track}/<id>.sanitized.csv``
(NOT the cached `.npz` chunks — those exist for TabPFN training only;
their 20k row cap is the wrong granularity for XGBoost / CatBoost on
real-sized datasets).

Per (model × dataset) the eval runs ``cfg.cv.n_folds`` cross-validation
with an INNER train/val split for HPO + F1-threshold tuning:

    outer split:  80% train,  20% test          (per fold)
    inner split:  80% sub-train, 20% validation (within the train fold)

Two execution modes
-------------------

1. **Local / single process (default)** — iterate every model on
   every test dataset in one process::

       python scripts/eval_pipeline.py track=pd

2. **Slurm-array (one (model × test_dataset) per task)** — for the
   3,000-dataset corpus, the cartesian product is large and each cell
   includes an Optuna HPO study. Each task processes ONE pair::

       N=$(python scripts/eval_pipeline.py --list-tasks track=pd)
       sbatch --array=0-$((N - 1))%32 scripts/slurm/eval_pd.slurm

   Each task writes its own
   ``results/<TRACK>/<method>/<run>_<timestamp>__ds-<id>.csv``,
   so concurrent tasks never write to the same file (no locking).

Re-runs and skip-existing
-------------------------
By default the eval is **idempotent across reruns**: before scoring,
each (model × test_dataset) pair is checked against existing CSVs
under ``<results.base_dir>/<TRACK>/<method-dirname>/``. If at least
one CSV already records an ``OK`` row for that pair, the pair is
skipped. This means adding a new trained checkpoint and resubmitting
the eval only triggers work for the new (and any previously failed)
cells — XGBoost, CatBoost, LogReg, LinReg, and untuned-TabPFN do not
re-run. Pass ``--rerun`` to force fresh scoring.

Optional filters
----------------
* ``--method <name>``         — score only this model.  Repeatable.
* ``--test-dataset <id>``     — score only this test dataset.  Repeatable.
* ``--task-index N``          — pick the Nth (model, dataset) pair.
* ``--list-tasks``            — print the total task count and exit.
* ``--rerun``                 — disable the skip-already-scored guard.

Test-dataset resolution
-----------------------
For ``tabpfn-trained`` models the test datasets come from each
checkpoint's ``.provenance.json`` (so every checkpoint is scored on
its OWN held-out set — recorded at training time). For
``tabpfn-untuned`` and classical baselines the test datasets come
from the train.yaml corpus split. Both routes give the same set
when the seed and fractions match (which they do by default), so
the comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import logging
import sys as _sys
import time
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
    from omegaconf import OmegaConf
    eval_cfg = OmegaConf.load("config/eval.yaml")
    if eval_overrides:
        eval_cfg = OmegaConf.merge(eval_cfg, OmegaConf.from_dotlist(eval_overrides))
    train_cfg = OmegaConf.load(eval_cfg.train_cfg_path)
    if train_overrides:
        train_cfg = OmegaConf.merge(train_cfg, OmegaConf.from_dotlist(train_overrides))
    return eval_cfg, train_cfg


# --------------------------------------------------------------------------- #
# Auto-cache hook (Gemini's #2 fix)
# --------------------------------------------------------------------------- #
#
# The eval reads PROCESSED CSVs under
# `data/processed/{track}/<id>.sanitized.csv` plus the per-track manifest.
# Both are produced by the data pipeline. Mirror `scripts/train_pipeline.py`'s
# auto-cache hook: if any dataset we're about to score is missing, run the
# data pipeline for just those IDs.


def _ensure_processed(plan, *, log_path):
    """Materialise any missing processed-CSV / manifest entry before
    the eval loop starts.

    A processed CSV is considered "present" iff
    `data/processed/{track}/{dataset_id}.sanitized.csv` exists. We
    don't check fingerprint validity here — the data pipeline does
    that on its own (`skip_if_cached` in the dataset stage). For
    missing rows, invoke `scripts/data_pipeline.run(datasets=missing)`.
    """
    from src.data.preprocessing import DATASET_METADATA
    from src.utils.paths import resolve_data_path

    needed: set[str] = set()
    for (handle_and_model, ds_ids) in plan:
        needed.update(ds_ids)

    tracks = {d: m["track"] for d, m in DATASET_METADATA.items()}
    missing: list[str] = []
    for did in sorted(needed):
        tr = tracks.get(did)
        if tr is None:
            continue
        p = resolve_data_path(
            f"data/processed/{tr}/{did}.sanitized.csv"
        )
        if not p.exists():
            missing.append(did)

    if not missing:
        LOGGER.info(
            "Auto-cache OK: every test dataset's processed CSV is on disk."
        )
        return

    LOGGER.info(
        "Auto-cache MISS: %d processed CSV(s) missing — running data "
        "pipeline to fill them: %s", len(missing), missing,
    )
    from scripts import data_pipeline
    rc = data_pipeline.run(
        fresh=False, datasets=missing, log_path=log_path,
    )
    if rc != 0:
        raise RuntimeError(
            f"data pipeline returned non-zero exit code while filling "
            f"{len(missing)} missing processed CSV(s); see logs."
        )


# --------------------------------------------------------------------------- #
# Roster + tasks
# --------------------------------------------------------------------------- #


def _build_roster(eval_cfg, train_cfg, track: str):
    """Build (handles_and_models, cfg_test_dataset_ids, manifest_csv_path).

    The cfg_test_dataset_ids list is the FALLBACK test set used by
    untuned/classical models (and by trained models whose provenance
    is missing). Trained models prefer their own provenance test set.
    """
    from src.train.corpus import split_from_cfg
    from src.model import build_baselines
    from src.eval.benchmark import load_trained_handles

    split = split_from_cfg(train_cfg, track=track)
    cfg_test_ids = sorted({c.dataset_id for c in split.test})

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

    return baselines + trained, cfg_test_ids, manifest_csv


def _enumerate_tasks(handles_and_models, cfg_test_ids: list[str]):
    """Cartesian product of (model_idx, dataset_id) — one slurm-array
    task scores one pair across all CV folds.

    For tabpfn-trained models with provenance, the dataset list comes
    from their own provenance (each checkpoint scored on its own test
    set). For other models, falls back to ``cfg_test_ids``.
    """
    from src.eval.benchmark import resolve_test_datasets
    pairs: list[tuple[int, str]] = []
    for m_idx, (handle, _) in enumerate(handles_and_models):
        ds_ids = resolve_test_datasets(handle, cfg_test_dataset_ids=cfg_test_ids)
        for did in sorted(ds_ids):
            pairs.append((m_idx, did))
    return pairs


def _filter_roster(handles_and_models, cfg_test_ids, *,
                   method_filter, dataset_filter, task_index):
    """Apply --method / --test-dataset / --task-index filters."""
    pairs = _enumerate_tasks(handles_and_models, cfg_test_ids)

    if task_index is not None:
        if not 0 <= task_index < len(pairs):
            raise IndexError(
                f"--task-index={task_index} is out of bounds; this cfg "
                f"has {len(pairs)} task(s) (indices 0..{len(pairs) - 1})."
            )
        m_idx, ds_id = pairs[task_index]
        return [(handles_and_models[m_idx], [ds_id])]

    keep_models = (
        [(h, m) for h, m in handles_and_models if h.name in set(method_filter)]
        if method_filter else list(handles_and_models)
    )
    out: list[tuple[tuple, list[str]]] = []
    for handle_and_model in keep_models:
        handle, _ = handle_and_model
        from src.eval.benchmark import resolve_test_datasets
        ds_ids = resolve_test_datasets(handle, cfg_test_dataset_ids=cfg_test_ids)
        if dataset_filter:
            ds_ids = [d for d in ds_ids if d in set(dataset_filter)]
        if ds_ids:
            out.append((handle_and_model, sorted(ds_ids)))
    return out


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
    rerun: bool = False,
) -> int:
    eval_cfg, train_cfg = _load_cfgs(eval_overrides or [], train_overrides or [])
    track = str(train_cfg.track)

    log, _ = resolve_run_log(log_path, task_name=f"eval_{track}")
    setup_logging(log.path)
    LOGGER.info("eval_pipeline: log=%s  track=%s", log.path, track)

    handles_and_models, cfg_test_ids, manifest_csv = _build_roster(
        eval_cfg, train_cfg, track,
    )
    LOGGER.info("Cfg test split: %d datasets", len(cfg_test_ids))
    LOGGER.info("Roster: %d models (manifest=%s)",
                len(handles_and_models), manifest_csv)

    plan = _filter_roster(
        handles_and_models, cfg_test_ids,
        method_filter=method_filter or [],
        dataset_filter=dataset_filter or [],
        task_index=task_index,
    )
    LOGGER.info("After filtering: %d (model × dataset-list) pair(s)", len(plan))

    # Rerun-skip: drop (handle × dataset_id) pairs that already have an
    # OK row on disk. The cartesian product over the 3 000-dataset corpus is
    # ~25 models × ~3 datasets = 75 cells per re-run; skipping the
    # already-scored ones means a new tabpfn-trained variant only triggers
    # work for the new cells. Pass `--rerun` to force-rescore everything.
    results_base_for_skip = (
        eval_cfg.results.base_dir if hasattr(eval_cfg, "results") else "results"
    )
    n_folds_required = (
        int(eval_cfg.cv.n_folds) if hasattr(eval_cfg, "cv") else 5
    )
    if rerun:
        LOGGER.info("--rerun set: existing CSVs will NOT be consulted.")
    else:
        from src.eval.benchmark import find_existing_results
        pruned_plan: list[tuple[tuple, list[str]]] = []
        n_skipped = 0
        skipped_pairs: list[str] = []
        for (handle_and_model, ds_ids) in plan:
            handle, _ = handle_and_model
            keep_ids = []
            for did in ds_ids:
                existing = find_existing_results(
                    handle, did, track=track,
                    results_base_dir=results_base_for_skip,
                    n_folds_required=n_folds_required,
                )
                if existing:
                    n_skipped += 1
                    skipped_pairs.append(f"{handle.name} × {did}")
                else:
                    keep_ids.append(did)
            if keep_ids:
                pruned_plan.append((handle_and_model, keep_ids))
        if n_skipped:
            LOGGER.info(
                "Skipping %d already-scored (model × dataset) pair(s) "
                "(all %d folds OK on disk); pass --rerun to force a "
                "fresh scoring. First 5: %s",
                n_skipped, n_folds_required, skipped_pairs[:5],
            )
        plan = pruned_plan
        LOGGER.info(
            "After rerun-skip: %d (model × dataset-list) pair(s) remain",
            len(plan),
        )

    if not plan:
        LOGGER.warning("nothing to do.")
        return 0

    # Auto-cache hook (Gemini's #2 fix) — the eval reads PROCESSED
    # CSVs and the per-track manifest. If any of the datasets we're
    # about to score is missing on disk, run the data pipeline for
    # just those IDs. Idempotent if everything is already there.
    _ensure_processed(plan, log_path=log.path)

    # Per-method row caps and CV settings.
    n_folds = int(eval_cfg.cv.n_folds)             if hasattr(eval_cfg, "cv") else 5
    inner   = float(eval_cfg.cv.inner_val_fraction) if hasattr(eval_cfg, "cv") else 0.20
    max_rows_tabpfn = (
        int(eval_cfg.max_rows_tabpfn)
        if hasattr(eval_cfg, "max_rows_tabpfn")
        and eval_cfg.max_rows_tabpfn is not None
        else None
    )
    results_base = (
        eval_cfg.results.base_dir if hasattr(eval_cfg, "results") else "results"
    )

    from src.eval.benchmark import run_benchmark
    t0 = time.monotonic()

    all_rows = []
    n_fail = 0
    for (handle_and_model, ds_ids) in plan:
        handle, _model = handle_and_model
        # In single-task slurm mode, tag the output file with the dataset_id.
        per_task_tag = (
            f"task{task_index}_ds-{ds_ids[0]}"
            if task_index is not None and len(ds_ids) == 1 else None
        )
        rows = run_benchmark(
            test_dataset_ids=ds_ids,
            handles_and_models=[handle_and_model],
            track=track,
            run_name=str(train_cfg.run_name),
            n_folds=n_folds,
            inner_val_fraction=inner,
            seed=int(train_cfg.seed),
            results_base_dir=results_base,
            max_rows_tabpfn=max_rows_tabpfn,
            per_task_tag=per_task_tag,
        )
        all_rows.extend(rows)
        n_fail += sum(1 for r in rows if r.status == "FAIL")

    elapsed = time.monotonic() - t0
    n_ok = sum(1 for r in all_rows if r.status == "OK")

    line = (
        f"eval_pipeline: status={'OK' if n_fail == 0 else f'FAIL[{n_fail}]'}  "
        f"track={track}  pairs={len(plan)}  folds={n_folds}  "
        f"cells_ok={n_ok}  cells_fail={n_fail}  "
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
        description="Cross-model benchmark on the held-out test datasets.",
    )
    p.add_argument("--log-path", default=None,
                   help="Append the run summary to this log file "
                        "(skip the auto-naming).")
    p.add_argument("--method", action="append", default=None,
                   help="Score only this model (repeatable).")
    p.add_argument("--test-dataset", action="append", default=None,
                   help="Score only this test dataset_id (repeatable).")
    p.add_argument("--task-index", type=int, default=None,
                   help="Pick the Nth (model × dataset) pair. For SLURM "
                        "arrays — set to $SLURM_ARRAY_TASK_ID.")
    p.add_argument("--list-tasks", action="store_true",
                   help="Print the total (model × dataset) task count for "
                        "the current cfg and exit.")
    p.add_argument("--rerun", action="store_true",
                   help="Force re-scoring every (model × dataset) pair, "
                        "even ones that already have an OK row in the "
                        "results directory. By default the eval skips "
                        "already-scored pairs so adding a new trained "
                        "checkpoint only triggers work for the new cells.")
    args, unknown = p.parse_known_args(argv)
    eval_keys_prefixes = (
        "train_cfg_path", "baselines.", "max_rows_tabpfn",
        "tabpfn_n_estimators", "cv.", "hpo.", "results.", "metrics.",
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
        logging.basicConfig(level=logging.WARNING, force=True)
        handles_and_models, cfg_test_ids, _ = _build_roster(
            eval_cfg, train_cfg, track,
        )
        print(len(_enumerate_tasks(handles_and_models, cfg_test_ids)))
        raise SystemExit(0)
    raise SystemExit(run(
        eval_overrides=eval_overrides,
        train_overrides=train_overrides,
        log_path=args.log_path,
        method_filter=args.method,
        dataset_filter=args.test_dataset,
        task_index=args.task_index,
        rerun=args.rerun,
    ))
