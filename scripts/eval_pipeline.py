"""End-to-end orchestrator for the cross-model benchmark.

Runs the **final benchmark** that the project ultimately exists to
produce: every TabPFN variant (untuned + every continued-pretrained
checkpoint listed in the training manifest) plus the classical
baselines (XGBoost, CatBoost, LogReg/LinReg) scored on the **same
held-out test chunks** the training pipeline produced.

The comparison CSV the eval writes is long-format:

    track, task_type, model_name, model_source, model_path,
    test_dataset_id, test_chunk_idx, n_train_rows, n_test_rows,
    metric_name, metric_value, elapsed_sec, status, error

…so any aggregation (mean per model, per-dataset breakdown, …) is
one ``pd.read_csv`` + ``groupby`` away.

CLI usage
---------
::

    # Default — uses cfg.train_cfg_path's `track` key
    python scripts/eval_pipeline.py

    # Explicit track override
    python scripts/eval_pipeline.py track=pd
    python scripts/eval_pipeline.py track=lgd

    # Hydra-style overrides on either yaml
    python scripts/eval_pipeline.py baselines.enabled=[xgboost,tabpfn-untuned]
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
from src.utils.run_log import resolve_run_log  # noqa: E402

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cfg loading
# --------------------------------------------------------------------------- #


def _load_cfgs(eval_overrides: list[str], train_overrides: list[str]):
    """Load both yamls + apply CLI overrides.

    The eval cfg points to the train cfg's path so we can keep the
    test-split definition (seed, fractions, multi_chunk_policy,
    train_dataset_ids, test_dataset_ids) authoritative in one place.
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
# Public entry point
# --------------------------------------------------------------------------- #


def run(
    eval_overrides: list[str] | None = None,
    train_overrides: list[str] | None = None,
    log_path: Path | str | None = None,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    log, _ = resolve_run_log(log_path)
    eval_cfg, train_cfg = _load_cfgs(eval_overrides or [], train_overrides or [])
    track = str(train_cfg.track)

    # Lazy imports — keep --help fast.
    from src.train.corpus import split_from_cfg
    from src.model import build_baselines
    from src.eval.benchmark import (
        load_trained_handles, run_benchmark,
    )

    # 1) Recreate the EXACT test split the training pipeline used.
    split = split_from_cfg(train_cfg, track=track)
    LOGGER.info("Test split: %s", split.summary)

    # 2) Build the baseline + tabpfn-untuned roster.
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
    LOGGER.info("Baselines + tabpfn-untuned: %d models",
                len(baselines))

    # 3) Pull every continued-pretrained TabPFN from the training manifest.
    manifest_csv = resolve_output_path("logs/runs") / f"{train_cfg.run_name}_{track}.csv"
    trained = load_trained_handles(
        manifest_csv, track=track,
        device=str(train_cfg.device),
        n_estimators=int(eval_cfg.tabpfn_n_estimators),
    )
    LOGGER.info("TabPFN-trained: %d checkpoints from %s",
                len(trained), manifest_csv)

    handles_and_models = baselines + trained

    # 4) Pick the metric for this track.
    metric_name = (
        eval_cfg.metric.classification if track == "pd"
        else eval_cfg.metric.regression
    )

    # 5) Run the benchmark — writes one CSV per method, never clobbers.
    n_folds = int(eval_cfg.cv.n_folds) if hasattr(eval_cfg, "cv") else 5
    results_base = (
        eval_cfg.results.base_dir if hasattr(eval_cfg, "results")
        else "results"
    )
    t0 = time.monotonic()
    rows = run_benchmark(
        test_chunks=split.test,
        handles_and_models=handles_and_models,
        track=track,
        metric_name=metric_name,
        run_name=str(train_cfg.run_name),
        n_folds=n_folds,
        seed=int(train_cfg.seed),
        results_base_dir=results_base,
    )
    elapsed = time.monotonic() - t0

    n_ok   = sum(1 for r in rows if r.status == "OK")
    n_fail = sum(1 for r in rows if r.status == "FAIL")
    line = (
        f"eval_pipeline: status={'OK' if n_fail == 0 else f'FAIL[{n_fail}]'}  "
        f"track={track}  metric={metric_name}  "
        f"models={len(handles_and_models)}  test_chunks={len(split.test)}  "
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
        description="Cross-model benchmark on the held-out test split."
    )
    p.add_argument(
        "--log-path", default=None,
        help="Append the run summary to this log file instead of creating "
             "a fresh logs/<timestamp>.log file.",
    )
    args, unknown = p.parse_known_args(argv)
    # Hydra-style overrides: split between eval and train cfgs by prefix.
    # eval keys: train_cfg_path / baselines.* / metric.* / tabpfn_n_estimators
    # everything else → train cfg.
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
    return args, eval_overrides, train_overrides


if __name__ == "__main__":
    args, eval_overrides, train_overrides = _parse_args()
    raise SystemExit(run(
        eval_overrides=eval_overrides,
        train_overrides=train_overrides,
        log_path=args.log_path,
    ))
