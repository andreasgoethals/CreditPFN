"""End-to-end orchestrator for continued pretraining.

Mirrors ``scripts/data_pipeline.py``. The actual training math lives in
:mod:`src.train.loop`; this script's job is to:

  1. **Resolve the training plan**: which (base_checkpoint, learning_rate)
     tuples to train. By default this is the full cartesian product of
     every list under ``cfg.tunable``. With ``--single`` the script uses
     only the FIRST value of each list (one trial). With ``--trial-index
     N`` only the Nth trial of the cartesian product is run — designed
     for slurm arrays where each array task takes one trial.

     Each parent dataset contributes exactly one training step per
     epoch (no chunking; see ``src/train/corpus.py`` for the rationale).

  2. **Auto-process hook**: before training starts, check whether the
     sanitized CSV exists under
     ``data/processed/<track>/<id>.sanitized.csv`` for every dataset
     the run will touch. If any are missing,
     ``scripts/data_pipeline.py`` is invoked transparently for just
     those IDs. This lets you train without ever calling the data
     pipeline by hand — though running it once up-front is still the
     recommended workflow for large corpora.

  3. **Per-trial training**: call :func:`src.train.loop.train_one_config`.
     Each trained checkpoint is saved to
     ``cfg.checkpoint.trained_dir/<track>/<descriptive_name>.ckpt``.

  4. **Manifest CSV** + **per-epoch CSV**:
     * One row per trial appended to
       ``output/manifests/<run_name>_<track>.csv`` (HP-tuple, checkpoint path,
       walltime, OK/FAIL). The eval pipeline
       (`scripts/eval_pipeline.py`) reads this to know which
       checkpoints to benchmark against the baselines.
     * One CSV per trial under
       ``output/manifests/epochs/<track>/<descriptive_name>.csv`` with the
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
import os
import sys as _sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow `python scripts/train_pipeline.py` (vs `-m scripts.train_pipeline`).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from src.utils.paths import (  # noqa: E402
    apply_data_source_from_cfg,
    manifests_dir,
    resolve_staging_path,
)
from src.utils.config import dump_resolved  # noqa: E402
from src.utils.logging_setup import resolve_run_log, setup_logging  # noqa: E402

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


def _resolve_grid(
    cfg, *, single: bool,
) -> list[tuple[str, float, bool, float, int, str]]:
    """Materialise the ``(base, lr, use_lora, query_fraction, accumulate,
    epoch_pass_mode)`` tuples to train.

    ``single=True``: head of every tunable list (one trial).
    Otherwise: full cartesian product over
    ``base × lr × use_lora × query_fraction × accumulate_grad_batches ×
    epoch_pass_modes``.

    All tunable lists accept either a scalar or a list. ``use_lora``
    defaults to ``[False]`` when absent; ``query_fractions`` defaults
    to ``[0.20]`` (the TabPFN documented default) when absent;
    ``accumulate_grad_batches`` defaults to ``[1]`` (TabPFN's official
    no-accumulation behaviour) when absent; ``epoch_pass_modes`` defaults
    to ``["one_sample"]`` (one step per dataset per epoch — the original
    behaviour) when absent.
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
    raw_qf = getattr(cfg.tunable, "query_fractions", [0.20])
    if isinstance(raw_qf, (int, float)):
        qfs = [float(raw_qf)]
    else:
        qfs = [float(x) for x in raw_qf]
    raw_acc = getattr(cfg.tunable, "accumulate_grad_batches", [1])
    if isinstance(raw_acc, int):
        accs = [int(raw_acc)]
    else:
        accs = [int(x) for x in raw_acc]
    raw_pm = getattr(cfg.tunable, "epoch_pass_modes", ["one_sample"])
    if isinstance(raw_pm, str):
        pms = [raw_pm]
    else:
        pms = [str(x) for x in raw_pm]

    if single:
        return [(
            str(bases[0]), float(lrs[0]), bool(loras[0]), float(qfs[0]),
            int(accs[0]), str(pms[0]),
        )]
    return [
        (str(b), float(lr), bool(lo), float(qf), int(ac), str(pm))
        for b, lr, lo, qf, ac, pm
        in itertools.product(bases, lrs, loras, qfs, accs, pms)
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
    from src.train.corpus import resolve_ids_for_track

    known = {d for d, m in DATASET_METADATA.items() if m["track"] == track}
    # Per-track aware: train/test ID configs may be a flat list or a
    # {pd: [...], lgd: [...]} mapping; resolve to the active track's pins.
    train_ids = list(resolve_ids_for_track(cfg.corpus.get("train_dataset_ids", None), track))
    test_ids  = list(resolve_ids_for_track(cfg.corpus.get("test_dataset_ids", None), track))

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
# Auto-process hook
# --------------------------------------------------------------------------- #


def _ensure_processed(cfg, log_path: Path | str | None) -> None:
    """Run the data pipeline for any dataset whose sanitized CSV is missing.

    The training pipeline reads ``data/processed/{track}/<id>.sanitized.csv``
    directly. If any of the datasets in the corpus split are missing from
    disk, this kicks off the data pipeline for just those IDs and lets it
    rebuild the manifest + sanitized CSV. Idempotent — when everything is
    on disk this function does O(#datasets) ``Path.exists()`` checks and
    returns.
    """
    from src.data.preprocessing import DATASET_METADATA
    from src.utils.paths import resolve_data_path
    from src.train.corpus import resolve_ids_for_track
    from omegaconf import OmegaConf

    track = str(cfg.track)
    corpus = cfg.corpus

    track_ids = sorted([d for d, m in DATASET_METADATA.items()
                        if m["track"] == track])

    # Restrict to whatever the user explicitly pinned, if anything. Use the
    # SAME per-track resolver the authoritative corpus builder (split_from_cfg)
    # uses: `train_dataset_ids` / `test_dataset_ids` may be a flat list OR a
    # per-track mapping {pd: [...], lgd: [...]}. A naive `list()` on the mapping
    # yields its KEYS (["lgd"]) — the latent bug that made this pre-check
    # resolve 0 candidates on 2026-07-08 (both tracks) and, with a too-strict
    # guard, fatally block the run.
    train_explicit = list(resolve_ids_for_track(corpus.get("train_dataset_ids", None), track))
    test_explicit  = list(resolve_ids_for_track(corpus.get("test_dataset_ids",  None), track))
    explicit = set(train_explicit) | set(test_explicit)
    candidate_ids = sorted(explicit & set(track_ids)) if explicit else track_ids

    data_cfg = OmegaConf.load("config/data.yaml")
    proc_root = resolve_data_path(data_cfg.paths.processed)

    # Best-effort pre-check ONLY: its job is to auto-run the data pipeline for
    # any missing sanitized CSV. The AUTHORITATIVE corpus is built and validated
    # by split_from_cfg downstream, so an empty candidate set here must NEVER be
    # fatal (a 2026-07-08 hard-error regression blocked a whole run this way).
    if not candidate_ids:
        LOGGER.warning(
            "Processed-CSV preflight resolved 0 candidate datasets for "
            "track=%r (corpus pins matched no DATASET_METADATA id). Skipping "
            "auto-processing; the corpus split is validated downstream.", track,
        )
        return

    missing = [
        did for did in candidate_ids
        if not (proc_root / track / f"{did}.sanitized.csv").exists()
    ]
    if not missing:
        LOGGER.info(
            "Processed-CSV check OK: all %d candidate dataset(s) for "
            "track=%s are on disk under %s.",
            len(candidate_ids), track, proc_root / track,
        )
        return

    LOGGER.info(
        "Processed-CSV miss: %d dataset(s) missing — running data pipeline "
        "to fill them: %s", len(missing), missing,
    )
    from scripts import data_pipeline
    rc = data_pipeline.run(fresh=False, datasets=missing, log_path=log_path)
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

    The training pipeline does NOT score the trained checkpoint via the
    full eval pipeline (``scripts/eval_pipeline.py`` for that, which does
    K-fold CV against every baseline). What the row DOES include is the
    epoch=-1 baseline (= pre-finetuning unmodified base TabPFN) and the
    last-good epoch's per-trial metrics, so you can answer at-a-glance:
    "did this trial improve over baseline, by how much, and on which
    metric?".
    """
    # Identity & hyperparameters
    track: str
    base_checkpoint: str
    learning_rate: float
    use_lora: bool
    query_fraction: float             # 0.20 / 0.40, see cfg.tunable.query_fractions
    accumulate_grad_batches: int      # 1 / 4, see cfg.tunable.accumulate_grad_batches
    seed: int

    # Corpus
    n_train_datasets: int
    n_test_datasets: int

    # Outputs
    final_ckpt_path: str | None
    elapsed_sec: float
    status: str                       # "OK" | "FAIL" | "SKIP" | "DIVERGED"
    error: str | None

    # NEW (2026-05-28) — per-trial summary metrics.
    # `primary_metric_name` = "roc_auc" for PD, "rmse" for LGD.
    # `secondary_metric_name` = "brier_score" for PD, "r2" for LGD.
    # The `baseline_*` numbers come from epoch=-1 (pre-FT) so a quick
    # comparison `final - baseline` shows the lift from finetuning.
    primary_metric_name:    str   = ""
    baseline_train_metric:  float = float("nan")
    baseline_test_metric:   float = float("nan")
    final_train_metric:     float = float("nan")
    final_test_metric:      float = float("nan")
    final_train_loss:       float = float("nan")
    secondary_metric_name:  str   = ""
    final_secondary_train:  float = float("nan")
    final_secondary_test:   float = float("nan")
    # Divergence record — only populated when status == "DIVERGED".
    diverged_at_epoch:      int | None = None
    diverge_reason:         str   = ""
    # Per-epoch step plan: "one_sample" (1 step/dataset/epoch) or
    # "full_pass" (size-proportional steps). See cfg.tunable.epoch_pass_modes.
    epoch_pass_mode:        str   = "one_sample"


_MANIFEST_THREAD_LOCK = threading.Lock()


@contextmanager
def _manifest_lock(path: Path):
    """Serialize manifest writes across local threads and SLURM processes.

    All array tasks append to one per-track CSV on NFS.  The old
    check-then-open sequence could let two first writers both choose ``"w"``
    and truncate one another.  Linux ``flock`` supplies the cross-process
    lock used on VSC; the in-process lock also makes local threaded tests and
    non-POSIX development safe.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_THREAD_LOCK:
        with lock_path.open("a+b") as lock_fh:
            try:
                import fcntl  # type: ignore[import-not-found]
            except ImportError:  # Windows: no multi-process SLURM writers.
                fcntl = None
            locked = False
            if fcntl is not None:
                try:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                    locked = True
                except OSError as exc:
                    # e.g. ENOSYS on a Lustre mount without `-o flock`, or ENOLCK
                    # if the NFS lock manager is down. Don't abort the trial's
                    # manifest write over this — fall back to the in-process lock
                    # (already held) with a one-time warning. Cross-node races
                    # become possible, but the append-only single-row writes make
                    # data loss unlikely, and the benchmark dedup is a backstop.
                    LOGGER.warning(
                        "manifest flock() unavailable on this filesystem (%s); "
                        "using the in-process lock only.", exc,
                    )
            try:
                yield
            finally:
                if locked:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _write_csv(rows: list[RunRow], path: Path, *, append: bool) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with _manifest_lock(path):
        exists = path.exists() and path.stat().st_size > 0
        write_header = (not append) or (not exists)
        mode = "a" if append and exists else "w"
        with path.open(mode, newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
            fh.flush()
            os.fsync(fh.fileno())


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

    # The resolved config, next to the results it produced (see src/utils/config.py).
    dump_resolved(cfg, f"train_{track}")
    LOGGER.info("train_pipeline: log=%s  cfg.track=%s  cfg.run_name=%s",
                log.path, track, cfg.run_name)

    # Validate any explicit corpus IDs against the dataset registry
    # NOW (before the auto-cache hook silently drops typos and the
    # downstream splitter quietly skips them). A typo in a CLI command
    # that copies an outdated README snippet would otherwise end up
    # training on fewer datasets than the user intended.
    _validate_corpus_ids_or_raise(cfg, track=track)

    # ---- 1) auto-process hook (always runs, near-zero cost when on-disk)
    _ensure_processed(cfg, log_path=log.path if hasattr(log, "path") else None)

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

    csv_path = manifests_dir() / f"{cfg.run_name}_{track}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 3) per-trial training
    from src.train.loop import descriptive_name, train_one_config

    # Per-epoch CSVs live in output/manifests/epochs/<track>/<descriptive_name>.csv
    epoch_csv_dir = manifests_dir() / "epochs" / track
    epoch_csv_dir.mkdir(parents=True, exist_ok=True)

    rows: list[RunRow] = []
    failures = 0
    divergences = 0
    t_outer = time.monotonic()

    for trial_idx_local, (base, lr, use_lora, query_fraction, accumulate, pass_mode) in enumerate(plan, start=1):
        global_idx = (
            trial_index if trial_index is not None
            else (trial_idx_local - 1)
        )
        LOGGER.info(
            "\n=== Trial %d/%d (global %d)  base=%s  lr=%g  lora=%s  qf=%.2f  acc=%d  pass=%s ===",
            trial_idx_local, len(plan), global_idx,
            Path(base).name, lr, use_lora, query_fraction, accumulate, pass_mode,
        )
        # Mini environment banner BEFORE any tabpfn import / model load, so a
        # crash during load still leaves version + path context in the log.
        # (The 2026-07-03 LGD failures died before the full debug banner —
        # those 32 logs carried no git sha / tabpfn version at all.)
        try:
            import torch as _torch
            from src.train.loop import _git_sha as _sha
            try:
                import tabpfn as _tp
                _tpv = getattr(_tp, "__version__", "?")
            except Exception as _e:                          # noqa: BLE001
                _tpv = f"IMPORT FAILED: {_e}"
            from src.utils.paths import get_roots as _gr
            _r = _gr()
            LOGGER.info(
                "[env] creditpfn_git=%s python=%s torch=%s tabpfn=%s | "
                "data_root=%s output_root=%s staging_root=%s",
                _sha(), _sys.version.split()[0], _torch.__version__, _tpv,
                _r.get("data_root"), _r.get("output_root"), _r.get("staging_root"),
            )
        except Exception:                                    # pragma: no cover
            pass

        # Per-epoch CSV path (mirrors the descriptive name of the checkpoint)
        run_basename = descriptive_name(
            run_name=str(cfg.run_name), track=track,
            base_path=base, learning_rate=lr, seed=int(cfg.seed),
            use_lora=use_lora, query_fraction=query_fraction,
            accumulate_grad_batches=accumulate, epoch_pass_mode=pass_mode,
        ).removesuffix(".ckpt")

        # ---- Rename the log file to include the trial's HPs --------- #
        # On Linux, renaming a file that's currently the target of an
        # `exec > $LOG` redirection works cleanly: the slurm shell holds
        # an open file descriptor on the inode, not on the directory
        # entry, so subsequent writes follow the renamed file. This
        # makes it trivial to find a specific (base × lr × qf × lora)
        # log: just glob for `*_<run_basename>.log` instead of having to
        # cross-reference the array task ID with the manifest.
        try:
            current_log = log.path if hasattr(log, "path") else log_path
            if current_log is not None:
                cur = Path(str(current_log))
                if cur.exists():
                    enriched = cur.with_name(
                        cur.stem + "__" + run_basename + cur.suffix
                    )
                    if enriched != cur:
                        cur.rename(enriched)
                        # Update our in-memory pointer so anyone who
                        # introspects log.path post-rename sees the new
                        # location.
                        try:
                            log.path = enriched                        # type: ignore[attr-defined]
                        except Exception:
                            pass
                        LOGGER.info(
                            "Log file renamed for this trial: %s",
                            enriched.name,
                        )
        except Exception as exc:                                       # pragma: no cover
            LOGGER.warning("log-rename failed (continuing): %s", exc)
        # ---- Resume: skip trial if checkpoint + provenance both exist --- #
        # Idempotency contract: if both the final .ckpt AND its
        # .provenance.json sidecar are on disk, the trial is considered
        # successfully completed. We emit a one-line "SKIP" record into
        # the manifest (so it still appears in the summary) and move on.
        # To force a rerun, delete the .ckpt (or use
        # `python -m src.utils.pipeline_clean --stages train`).
        # A finished checkpoint may live in staging OR — when staging wasn't
        # writable from the training node and the loop fell back — under the
        # output root. Check BOTH so re-submissions skip completed trials
        # regardless of where the artefact landed.
        from src.utils.paths import resolve_output_path as _rop
        _candidates = [
            resolve_staging_path(cfg.checkpoint.trained_dir) / track / f"{run_basename}.ckpt",
            _rop(cfg.checkpoint.trained_dir) / track / f"{run_basename}.ckpt",
        ]
        expected_ckpt = next(
            (c for c in _candidates
             if c.exists() and c.with_suffix(c.suffix + ".provenance.json").exists()),
            _candidates[0],
        )
        expected_prov = expected_ckpt.with_suffix(
            expected_ckpt.suffix + ".provenance.json",
        )
        if expected_ckpt.exists() and expected_prov.exists():
            LOGGER.info(
                "SKIP trial %d (global %d): checkpoint already exists at %s "
                "— delete the file or use `pipeline_clean --stages train` "
                "to force a rerun.",
                trial_idx_local, global_idx, expected_ckpt,
            )
            rows.append(RunRow(
                track=track, base_checkpoint=base, learning_rate=lr,
                use_lora=use_lora, query_fraction=query_fraction,
                accumulate_grad_batches=int(accumulate),
                epoch_pass_mode=pass_mode,
                seed=int(cfg.seed),
                n_train_datasets=0, n_test_datasets=0,
                final_ckpt_path=str(expected_ckpt),
                elapsed_sec=0.0,
                status="SKIP", error=None,
            ))
            _write_csv([rows[-1]], csv_path, append=csv_append)
            if not csv_append:
                csv_append = True
            continue

        epoch_csv = epoch_csv_dir / f"{run_basename}.csv"
        if epoch_csv.exists():
            epoch_csv.unlink()              # fresh file per run
        _epoch_csv_init: dict[str, bool] = {"written_header": False}

        def _on_epoch_end(rec, _path=epoch_csv, _flag=_epoch_csv_init) -> None:
            # `secondary_*` is the optional per-track second metric — R²
            # for LGD, empty for PD. Columns are present in both tracks'
            # CSVs so the downstream notebooks see a stable schema; for
            # PD the secondary columns hold NaN / empty string.
            row = {
                "epoch":                     int(rec.epoch),
                "train_loss":                float(rec.train_loss),
                "lr":                        float(rec.lr),
                "metric_name":               str(rec.metric_name),
                "train_metric":              float(rec.train_metric),
                "test_metric":               float(rec.test_metric),
                "secondary_metric_name":     str(rec.secondary_metric_name),
                "secondary_train_metric":    float(rec.secondary_train_metric),
                "secondary_test_metric":     float(rec.secondary_test_metric),
                "epoch_time_sec":            float(rec.epoch_time_sec),
                "elapsed_sec":               float(rec.elapsed_sec),
                "optimizer_steps":           int(rec.optimizer_steps),
                "amp_skipped_steps":         int(rec.amp_skipped_steps),
                "data_skipped_steps":        int(rec.data_skipped_steps),
            }
            # Per-dataset loss and per-stage drift, one column each. The
            # column SET is fixed by the first row written (the epoch=-1
            # baseline has neither, so the header is taken from the first
            # trained epoch instead — see the header logic below).
            row.update({f"loss__{k}": float(v)
                        for k, v in sorted(rec.per_dataset_loss.items())})
            row.update({f"drift__{k}": float(v)
                        for k, v in sorted(rec.stage_drift.items())})

            # The baseline row (epoch=-1) carries no per-dataset losses, and
            # stage drift only appears on MONITORED epochs — so the naive
            # "header = keys of the first row" rule would lock in a schema
            # that later rows overflow, and DictWriter would raise on the
            # extra keys. Keep the widest header seen so far and pad missing
            # cells, so the file stays a rectangle whatever the cadence.
            known = _flag.setdefault("fieldnames", [])
            for k in row:
                if k not in known:
                    known.append(k)
            write_header = not _flag["written_header"]
            if not write_header and len(known) != _flag.get("n_cols", len(known)):
                # A new column appeared after the header was written (first
                # monitored epoch adds the drift__* set). Rewrite the file
                # with the wider header so downstream readers see a rectangle.
                _rewrite_epoch_csv(_path, known)
            _flag["n_cols"] = len(known)
            with _path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=known, restval="")
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
                query_fraction=query_fraction,
                accumulate_grad_batches=accumulate,
                pass_mode=pass_mode,
                on_epoch_end=_on_epoch_end,
            )
            rows.append(RunRow(
                track=track, base_checkpoint=base, learning_rate=lr,
                use_lora=use_lora, query_fraction=query_fraction,
                accumulate_grad_batches=int(accumulate),
                epoch_pass_mode=pass_mode,
                seed=int(cfg.seed),
                n_train_datasets=result.n_train_datasets,
                n_test_datasets=result.n_test_datasets,
                final_ckpt_path=str(result.final_ckpt_path),
                elapsed_sec=result.elapsed_sec,
                status=("DIVERGED" if result.diverged else "OK"),
                error=(
                    f"diverged@epoch={result.diverged_at_epoch}"
                    f"({result.diverge_reason})"
                    if result.diverged else None
                ),
                primary_metric_name=result.primary_metric_name,
                baseline_train_metric=result.baseline_train_metric,
                baseline_test_metric=result.baseline_test_metric,
                final_train_metric=result.final_train_metric,
                final_test_metric=result.final_test_metric,
                final_train_loss=result.final_train_loss,
                secondary_metric_name=result.secondary_metric_name,
                final_secondary_train=result.final_secondary_train,
                final_secondary_test=result.final_secondary_test,
                diverged_at_epoch=result.diverged_at_epoch,
                diverge_reason=result.diverge_reason,
            ))
            if result.diverged:
                # A numerically completed but diverged checkpoint is excluded
                # from eval and must not make its SLURM task write a success
                # sentinel. Returning non-zero lets the post-training gate
                # detect an incomplete scientific grid.
                divergences += 1
        except Exception as exc:                           # noqa: BLE001
            failures += 1
            LOGGER.error("Trial %d failed: %s", trial_idx_local, exc, exc_info=True)
            rows.append(RunRow(
                track=track, base_checkpoint=base, learning_rate=lr,
                use_lora=use_lora, query_fraction=query_fraction,
                accumulate_grad_batches=int(accumulate),
                epoch_pass_mode=pass_mode,
                seed=int(cfg.seed),
                n_train_datasets=0, n_test_datasets=0,
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
    if failures:
        overall_status = f"FAIL[{failures}/{len(plan)}]"
    elif divergences:
        overall_status = f"DIVERGED[{divergences}/{len(plan)}]"
    else:
        overall_status = "OK"
    summary = (
        f"train_pipeline: status={overall_status}  "
        f"track={track}  mode={plan_label}  "
        f"trials={len(plan)}  csv={csv_path}  elapsed={elapsed:.1f}s"
    )
    log.write(summary)
    print(summary)
    return 0 if failures == 0 and divergences == 0 else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #



def _rewrite_epoch_csv(path, fieldnames: list[str]) -> None:
    """Re-emit an existing per-epoch CSV under a WIDER header.

    Needed because the column set grows during a run: the epoch=-1 baseline
    row has no per-dataset losses, and `drift__*` only appears on monitored
    epochs. Rather than pre-declaring every possible column (which would
    require knowing the dataset list and module names up front), we widen the
    header when a new column first appears and pad the earlier rows. Cheap —
    the file has one row per epoch.
    """
    import csv as _csv
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as fh:
        old = list(_csv.DictReader(fh))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        w.writeheader()
        for r in old:
            w.writerow({k: r.get(k, "") for k in fieldnames})

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
        "--trial-family", type=int, default=None, metavar="N",
        help="Print the MODEL FAMILY ('tabpfn' or 'tabicl') of the Nth trial "
             "and exit. Lets a slurm prolog run only the preflight checks "
             "that this trial actually needs, so a missing optional "
             "dependency fails one family's trials instead of the whole grid.",
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
    if args.trial_family is not None:
        from src.train.tabicl_compat import model_family
        cfg = _load_cfg(overrides)
        grid = _resolve_grid(cfg, single=False)
        if not 0 <= args.trial_family < len(grid):
            # Over-sized slurm arrays are a legitimate pattern; a surplus
            # index is not an error. Print nothing and exit 0 so the caller's
            # `[[ "$FAMILY" == tabicl ]]` test simply doesn't match.
            raise SystemExit(0)
        print(model_family(grid[args.trial_family][0]))
        raise SystemExit(0)
    raise SystemExit(run(
        single=args.single,
        trial_index=args.trial_index,
        overrides=overrides,
        log_path=args.log_path,
    ))
