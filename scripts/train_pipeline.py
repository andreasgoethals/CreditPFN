"""End-to-end orchestrator for continued pretraining.

Mirrors ``scripts/data_pipeline.py``. The actual training math lives in
:mod:`src.train.loop`; this script's job is to:

  1. **Resolve the training plan**: which (base_checkpoint, learning_rate,
     multi_chunk_policy) tuples to train. By default this is the full
     cartesian product of every list under ``cfg.tunable``. With
     ``--single`` the script uses only the FIRST value of each list
     (one trial). With ``--trial-index N`` only the Nth trial of the
     cartesian product is run — designed for slurm arrays where each
     array task takes one trial.

  2. **Auto-cache hook**: before training starts, check whether every
     dataset the run will touch is on disk under
     ``cfg.corpus.cached_dir``. If any are missing,
     ``scripts/data_pipeline.py`` is invoked transparently for just
     those IDs. This lets you train without ever calling the data
     pipeline by hand — though running it once up-front is still the
     recommended workflow for large corpora.

  3. **Per-trial training**: call :func:`src.train.loop.train_one_config`.
     Each trained checkpoint is saved to
     ``cfg.checkpoint.trained_dir/<track>/<descriptive_name>.ckpt``.

  4. **Manifest CSV**: append a row per trial to
     ``logs/runs/<run_name>_<track>.csv`` (HP-tuple, test metric,
     checkpoint path, walltime, OK/FAIL). The eval pipeline
     (`scripts/eval_pipeline.py`) reads this CSV to know which
     checkpoints to benchmark against the baselines.

CLI usage
---------
::

    # Local: cartesian product over `cfg.tunable.*`
    python scripts/train_pipeline.py

    # Local: only one trial (first value of every tunable list)
    python scripts/train_pipeline.py --single

    # Slurm array (one task per trial):
    #   sbatch --array=0-$(($(python scripts/train_pipeline.py --list-trials)-1)) \
    #          scripts/slurm/train_pd.slurm
    python scripts/train_pipeline.py --trial-index $SLURM_ARRAY_TASK_ID

    # How many trials does the current cfg expand to?
    python scripts/train_pipeline.py --list-trials

    # Debug: train on one specific dataset only
    python scripts/train_pipeline.py corpus.train_dataset_ids=[0001.gmsc]

    # Hydra-style overrides (any cfg key)
    python scripts/train_pipeline.py track=lgd train.epochs=10
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys as _sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow `python scripts/train_pipeline.py` (vs `-m scripts.train_pipeline`).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from src.utils.paths import resolve_output_path  # noqa: E402
from src.utils.run_log import resolve_run_log  # noqa: E402

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cfg loading + Hydra-style overrides
# --------------------------------------------------------------------------- #


def _load_cfg(overrides: list[str] | None = None):
    """Load ``config/train.yaml`` and apply ``key=value`` overrides."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/train.yaml")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _resolve_grid(cfg, *, single: bool) -> list[tuple[str, float, str]]:
    """Materialise the (base, lr, policy) tuples to train.

    ``single=True``: head of every tunable list (one trial).
    Otherwise: full cartesian product.
    """
    track = str(cfg.track)
    bases = (
        list(cfg.tunable.classifier_base_paths) if track == "pd"
        else list(cfg.tunable.regressor_base_paths)
    )
    lrs = [float(x) for x in cfg.tunable.learning_rates]
    policies = list(cfg.tunable.multi_chunk_policies)

    if single:
        return [(str(bases[0]), float(lrs[0]), str(policies[0]))]
    return [
        (str(b), float(lr), str(p))
        for b, lr, p in itertools.product(bases, lrs, policies)
    ]


# --------------------------------------------------------------------------- #
# Auto-cache hook
# --------------------------------------------------------------------------- #


def _ensure_cache(cfg, log_path: Path | str | None) -> None:
    """Run the data pipeline for any dataset that the training run will
    need but that isn't cached yet.

    Strategy:
      * Pull the canonical ID list from ``DATASET_METADATA`` (one
        entry per registered dataset) and filter to the active track.
      * Restrict further to the IDs the user actually asked for in
        ``cfg.corpus.train_dataset_ids`` ∪ ``cfg.corpus.test_dataset_ids``
        (if either list is non-empty); otherwise consider every
        registered ID for the track.
      * Use :func:`src.data.cache.find_uncached_datasets` to compute
        the missing subset; if it is non-empty, invoke
        :func:`scripts.data_pipeline.run` with ``datasets=missing``.

    Idempotent: a fully-cached corpus walks through this function
    in O(#datasets) ``Path.exists()`` calls and does nothing else.
    """
    from src.data.preprocessing import DATASET_METADATA
    from src.data.cache import find_uncached_datasets

    track = str(cfg.track)
    corpus = cfg.corpus

    # Tracks lookup (used by find_uncached_datasets)
    tracks = {did: m["track"] for did, m in DATASET_METADATA.items()}
    track_ids = sorted([d for d, m in DATASET_METADATA.items()
                        if m["track"] == track])

    # Restrict to whatever the user explicitly asked for, if anything.
    train_explicit = list(corpus.get("train_dataset_ids", []) or [])
    test_explicit  = list(corpus.get("test_dataset_ids",  []) or [])
    explicit = set(train_explicit) | set(test_explicit)
    candidate_ids = sorted(explicit & set(track_ids)) if explicit else track_ids

    missing = find_uncached_datasets(
        corpus.cached_dir,
        dataset_ids=candidate_ids,
        tracks=tracks,
    )
    if not missing:
        LOGGER.info("Cache OK: all %d candidate dataset(s) for track=%s "
                    "are materialised.", len(candidate_ids), track)
        return

    LOGGER.info("Cache MISS: %d dataset(s) missing — running data pipeline "
                "to fill them: %s", len(missing), missing)
    # Lazy import: only loads omegaconf again etc. No circular refs.
    from scripts import data_pipeline
    rc = data_pipeline.run(
        fresh=False, datasets=missing, log_path=log_path,
    )
    if rc != 0:
        raise RuntimeError(
            f"data pipeline returned non-zero exit code while filling "
            f"{len(missing)} missing dataset(s); see logs."
        )


# --------------------------------------------------------------------------- #
# CSV manifest row
# --------------------------------------------------------------------------- #


@dataclass
class RunRow:
    """One row of the per-track training manifest."""
    track: str
    base_checkpoint: str
    learning_rate: float
    multi_chunk_policy: str
    seed: int
    test_metric_name: str
    test_metric_raw: float | None
    n_train_datasets: int
    n_test_datasets: int
    n_train_chunks: int
    n_test_chunks: int
    final_ckpt_path: str | None
    elapsed_sec: float
    status: str                       # "OK" | "FAIL"
    error: str | None


def _write_csv(rows: list[RunRow], path: Path, *, append: bool) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    write_header = (not append) or (not path.exists())
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def run(
    single: bool = False,
    trial_index: int | None = None,
    overrides: list[str] | None = None,
    log_path: Path | str | None = None,
    cfg=None,
) -> int:
    """Train one trial (``--single`` / ``--trial-index``) or every (base × lr × policy) tuple.

    ``trial_index`` takes precedence over ``single`` if both are set.

    Returns
    -------
    ``0`` on full success, ``1`` if any trial raised.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    if cfg is None:
        cfg = _load_cfg(overrides)
    log, _ = resolve_run_log(log_path)

    track = str(cfg.track)
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    # ---- 1) auto-cache hook (always runs, near-zero cost when cache is OK)
    _ensure_cache(cfg, log_path=log.path if hasattr(log, "path") else None)

    # ---- 2) resolve which trials to run
    full_grid = _resolve_grid(cfg, single=False)

    if trial_index is not None:
        if not 0 <= trial_index < len(full_grid):
            raise IndexError(
                f"trial_index={trial_index} is out of bounds; this cfg "
                f"has {len(full_grid)} trial(s) (indices 0..{len(full_grid) - 1})."
            )
        plan = [full_grid[trial_index]]
        plan_label = f"trial {trial_index}"
        # When running one trial of a slurm array, append (don't clobber).
        csv_append = True
    elif single:
        plan = [full_grid[0]]
        plan_label = "single (--single)"
        csv_append = False
    else:
        plan = full_grid
        plan_label = "cartesian grid"
        csv_append = False

    LOGGER.info(
        "Training plan: %d run(s) on track=%s (%s; full grid has %d)",
        len(plan), track, plan_label, len(full_grid),
    )

    csv_path = resolve_output_path("logs/runs") / f"{cfg.run_name}_{track}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 3) per-trial training
    from src.train.loop import train_one_config

    rows: list[RunRow] = []
    failures = 0
    t_outer = time.monotonic()

    for trial_idx_local, (base, lr, policy) in enumerate(plan, start=1):
        global_idx = (
            trial_index if trial_index is not None
            else (trial_idx_local - 1)
        )
        LOGGER.info(
            "\n=== Trial %d/%d (global %d)  base=%s  lr=%g  policy=%s ===",
            trial_idx_local, len(plan), global_idx,
            Path(base).name, lr, policy,
        )
        t_trial = time.monotonic()
        try:
            result = train_one_config(
                cfg, track=track,
                base_checkpoint=base,
                learning_rate=lr,
                multi_chunk_policy=policy,
            )
            rows.append(RunRow(
                track=track, base_checkpoint=base, learning_rate=lr,
                multi_chunk_policy=policy, seed=int(cfg.seed),
                test_metric_name=result.test_metric_name,
                test_metric_raw=result.test_metric_raw,
                n_train_datasets=result.n_train_datasets,
                n_test_datasets=result.n_test_datasets,
                n_train_chunks=result.n_train_chunks,
                n_test_chunks=result.n_test_chunks,
                final_ckpt_path=str(result.final_ckpt_path),
                elapsed_sec=result.elapsed_sec,
                status="OK", error=None,
            ))
        except Exception as exc:                           # noqa: BLE001
            failures += 1
            LOGGER.error("Trial %d failed: %s", trial_idx_local, exc, exc_info=True)
            rows.append(RunRow(
                track=track, base_checkpoint=base, learning_rate=lr,
                multi_chunk_policy=policy, seed=int(cfg.seed),
                test_metric_name="",
                test_metric_raw=None,
                n_train_datasets=0, n_test_datasets=0,
                n_train_chunks=0, n_test_chunks=0,
                final_ckpt_path=None,
                elapsed_sec=time.monotonic() - t_trial,
                status="FAIL", error=f"{type(exc).__name__}: {exc}",
            ))

        _write_csv(rows, csv_path, append=csv_append)
        # After the first row of an appending run (slurm-array mode), we
        # keep appending; for a clobbering run we stay in 'write' mode
        # but the file already exists so subsequent writes append rows
        # under the same header.
        if not csv_append:
            csv_append = True   # subsequent rows in the same process append

    elapsed = time.monotonic() - t_outer
    summary = (
        f"train_pipeline: status={'OK' if failures == 0 else f'FAIL[{failures}/{len(plan)}]'}  "
        f"track={track}  mode={plan_label}  "
        f"trials={len(plan)}  csv={csv_path}  elapsed={elapsed:.1f}s"
    )
    log.write(summary)
    print(summary)
    return 0 if failures == 0 else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Continued pretraining for TabPFN on the credit corpus.",
    )
    p.add_argument(
        "--single", action="store_true",
        help="Train only ONE trial (the first value of every list under "
             "cfg.tunable). Default: cartesian product of all tunable lists.",
    )
    p.add_argument(
        "--trial-index", type=int, default=None,
        help="Train only the Nth trial of the cartesian grid (0-indexed). "
             "Designed for slurm arrays — set to $SLURM_ARRAY_TASK_ID.",
    )
    p.add_argument(
        "--list-trials", action="store_true",
        help="Print the number of trials in the current cfg's cartesian "
             "grid and exit. Useful for sizing slurm arrays.",
    )
    p.add_argument(
        "--log-path", default=None,
        help="Append the run summary to this log file instead of creating "
             "a fresh logs/<timestamp>.log file.",
    )
    args, unknown = p.parse_known_args(argv)
    overrides = [a for a in unknown if "=" in a and not a.startswith("-")]
    leftover = [a for a in unknown if a not in overrides]
    if leftover:
        p.error(f"unrecognised arguments: {leftover}")
    return args, overrides


if __name__ == "__main__":
    args, overrides = _parse_args()
    if args.list_trials:
        cfg = _load_cfg(overrides)
        print(len(_resolve_grid(cfg, single=False)))
        raise SystemExit(0)
    raise SystemExit(run(
        single=args.single,
        trial_index=args.trial_index,
        overrides=overrides,
        log_path=args.log_path,
    ))
