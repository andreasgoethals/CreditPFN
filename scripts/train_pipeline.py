"""End-to-end orchestrator for continued pretraining.

Mirrors ``scripts/data_pipeline.py``. The actual training math lives in
:mod:`src.train.loop`; this script's job is to:

  1. **Resolve the training plan**: which (base_checkpoint, learning_rate)
     tuples to train. By default this is the full cartesian product of
     every list under ``cfg.tunable``. With ``--single`` the script uses
     only the FIRST value of each list (one trial). With ``--trial-index
     N`` only the Nth trial of the cartesian product is run — designed
     for slurm arrays where each array task takes one trial.

     The multi-chunk policy is fixed (one chunk per parent dataset);
     see :data:`src.train.loop.MULTI_CHUNK_POLICY`.

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

  4. **Manifest CSV** + **per-epoch CSV**:
     * One row per trial appended to
       ``manifests/<run_name>_<track>.csv`` (HP-tuple, checkpoint path,
       walltime, OK/FAIL). The eval pipeline
       (`scripts/eval_pipeline.py`) reads this to know which
       checkpoints to benchmark against the baselines.
     * One CSV per trial under
       ``results/training/<track>/<descriptive_name>.csv`` with the
       per-epoch ``(epoch, train_loss, lr, elapsed_sec)`` — useful
       for diagnosing how the loss evolves across epochs.

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

from src.utils.paths import apply_data_source_from_cfg, resolve_output_path  # noqa: E402
from src.utils.run_log import resolve_run_log, setup_logging  # noqa: E402

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


def _resolve_grid(cfg, *, single: bool) -> list[tuple[str, float, bool]]:
    """Materialise the (base, lr, use_lora) tuples to train.

    ``single=True``: head of every tunable list (one trial).
    Otherwise: full cartesian product over base × lr × use_lora.

    ``cfg.tunable.use_lora`` may be a single bool, a list of bools,
    or missing entirely (defaults to ``[False]``).
    """
    track = str(cfg.track)
    bases = (
        list(cfg.tunable.classifier_base_paths) if track == "pd"
        else list(cfg.tunable.regressor_base_paths)
    )
    lrs = [float(x) for x in cfg.tunable.learning_rates]
    raw_lora = getattr(cfg.tunable, "use_lora", [False])
    if isinstance(raw_lora, bool):
        loras = [bool(raw_lora)]
    else:
        loras = [bool(x) for x in raw_lora]

    if single:
        return [(str(bases[0]), float(lrs[0]), bool(loras[0]))]
    return [
        (str(b), float(lr), bool(lo))
        for b, lr, lo in itertools.product(bases, lrs, loras)
    ]


# --------------------------------------------------------------------------- #
# Explicit-ID validation
# --------------------------------------------------------------------------- #


def _validate_corpus_ids_or_raise(cfg, *, track: str) -> None:
    """Fail fast if ``cfg.corpus.train_dataset_ids`` / ``test_dataset_ids``
    contain IDs that aren't registered in ``DATASET_METADATA`` for the
    active track.

    Without this, a typo (e.g. ``0002.heloc`` instead of
    ``0002.taiwan_creditcard``) would be silently dropped by the
    auto-cache hook's set-intersection and then the corpus splitter's
    warn-and-continue, leaving the user with a quietly smaller training
    set than intended. We'd rather crash with a message that lists the
    valid IDs.
    """
    from src.data.preprocessing import DATASET_METADATA

    known = {d for d, m in DATASET_METADATA.items() if m["track"] == track}
    train_ids = list(cfg.corpus.get("train_dataset_ids", []) or [])
    test_ids  = list(cfg.corpus.get("test_dataset_ids", []) or [])

    bad_train = [d for d in train_ids if d not in known]
    bad_test  = [d for d in test_ids  if d not in known]
    if not (bad_train or bad_test):
        return

    valid_sorted = "\n  ".join(sorted(known))
    raise ValueError(
        f"Unknown dataset_id(s) for track={track!r}:\n"
        f"  train_dataset_ids: {bad_train}\n"
        f"  test_dataset_ids:  {bad_test}\n"
        f"Valid IDs for this track:\n  {valid_sorted}"
    )


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
    """One row of the per-track training manifest.

    No test-set metric — the training pipeline does not score models.
    Use `scripts/eval_pipeline.py` for that. The eval reads the
    `final_ckpt_path` and the corresponding sidecar
    ``<final_ckpt_path>.provenance.json`` to recover the
    ``test_dataset_ids`` for this checkpoint.
    """
    track: str
    base_checkpoint: str
    learning_rate: float
    use_lora: bool
    seed: int
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
    """Train one trial (``--single`` / ``--trial-index``) or every (base × lr) tuple.

    ``trial_index`` takes precedence over ``single`` if both are set.

    Returns
    -------
    ``0`` on full success, ``1`` if any trial raised.
    """
    if cfg is None:
        cfg = _load_cfg(overrides)
    track = str(cfg.track)
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    # Apply paths.data_source from config/data.yaml (single source of truth)
    # BEFORE any path resolution downstream. See apply_data_source_from_cfg.
    from omegaconf import OmegaConf
    apply_data_source_from_cfg(OmegaConf.load("config/data.yaml"))

    # ---- 0) one log file per task: logs/<task>_<ts>.log -----------
    log, _ = resolve_run_log(log_path, task_name=f"train_{track}")
    setup_logging(log.path)
    LOGGER.info("train_pipeline: log=%s  cfg.track=%s  cfg.run_name=%s",
                log.path, track, cfg.run_name)

    # Validate any explicit corpus IDs against the dataset registry
    # NOW (before the auto-cache hook silently drops typos and the
    # downstream splitter quietly skips them). A typo in a CLI command
    # that copies an outdated README snippet would otherwise end up
    # training on fewer datasets than the user intended.
    _validate_corpus_ids_or_raise(cfg, track=track)

    # ---- 1) auto-cache hook (always runs, near-zero cost when cache is OK)
    _ensure_cache(cfg, log_path=log.path if hasattr(log, "path") else None)

    # ---- 2) resolve which trials to run
    full_grid = _resolve_grid(cfg, single=False)

    if trial_index is not None:
        if not 0 <= trial_index < len(full_grid):
            # Soft no-op: an over-sized slurm array (e.g. --array=0-31
            # against a 9-trial grid) is a legitimate pattern when the
            # grid size changes between submissions, and we want the
            # surplus tasks to exit zero cleanly rather than spam the
            # cluster with FAILED jobs. Direct CLI users still get a
            # clear message in the log.
            LOGGER.warning(
                "trial_index=%d is out of bounds for the %d-trial grid "
                "(valid indices 0..%d). Nothing to do — exiting cleanly.",
                trial_index, len(full_grid), len(full_grid) - 1,
            )
            print(
                f"train_pipeline: SKIP  trial_index={trial_index} "
                f"out of bounds for {len(full_grid)}-trial grid; exit 0",
            )
            return 0
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

    csv_path = resolve_output_path("manifests") / f"{cfg.run_name}_{track}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 3) per-trial training
    from src.train.loop import descriptive_name, train_one_config

    # Per-epoch CSVs live in results/training/<track>/<descriptive_name>.csv
    epoch_csv_dir = resolve_output_path("results/training") / track
    epoch_csv_dir.mkdir(parents=True, exist_ok=True)

    rows: list[RunRow] = []
    failures = 0
    t_outer = time.monotonic()

    for trial_idx_local, (base, lr, use_lora) in enumerate(plan, start=1):
        global_idx = (
            trial_index if trial_index is not None
            else (trial_idx_local - 1)
        )
        LOGGER.info(
            "\n=== Trial %d/%d (global %d)  base=%s  lr=%g  lora=%s ===",
            trial_idx_local, len(plan), global_idx,
            Path(base).name, lr, use_lora,
        )

        # Per-epoch CSV path (mirrors the descriptive name of the checkpoint)
        run_basename = descriptive_name(
            run_name=str(cfg.run_name), track=track,
            base_path=base, learning_rate=lr, seed=int(cfg.seed),
            use_lora=use_lora,
        ).removesuffix(".ckpt")
        epoch_csv = epoch_csv_dir / f"{run_basename}.csv"
        if epoch_csv.exists():
            epoch_csv.unlink()              # fresh file per run
        _epoch_csv_init: dict[str, bool] = {"written_header": False}

        def _on_epoch_end(rec, _path=epoch_csv, _flag=_epoch_csv_init) -> None:
            row = {
                "epoch":          int(rec.epoch),
                "train_loss":     float(rec.train_loss),
                "lr":             float(rec.lr),
                "metric_name":    str(rec.metric_name),
                "train_metric":   float(rec.train_metric),
                "test_metric":    float(rec.test_metric),
                "epoch_time_sec": float(rec.epoch_time_sec),
                "elapsed_sec":    float(rec.elapsed_sec),
            }
            write_header = not _flag["written_header"]
            with _path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                    _flag["written_header"] = True
                w.writerow(row)

        t_trial = time.monotonic()
        try:
            result = train_one_config(
                cfg, track=track,
                base_checkpoint=base,
                learning_rate=lr,
                use_lora=use_lora,
                on_epoch_end=_on_epoch_end,
            )
            rows.append(RunRow(
                track=track, base_checkpoint=base, learning_rate=lr,
                use_lora=use_lora, seed=int(cfg.seed),
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
                use_lora=use_lora, seed=int(cfg.seed),
                n_train_datasets=0, n_test_datasets=0,
                n_train_chunks=0, n_test_chunks=0,
                final_ckpt_path=None,
                elapsed_sec=time.monotonic() - t_trial,
                status="FAIL", error=f"{type(exc).__name__}: {exc}",
            ))

        # Write ONLY the row from this trial (not the full accumulated
        # `rows` list — that would re-append rows 1..N-1 every iteration).
        # `csv_append` flips to True after the first write so subsequent
        # trials append under the existing header.
        _write_csv([rows[-1]], csv_path, append=csv_append)
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
