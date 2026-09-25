"""Run a named continued-pretraining phase or inspect its trial grid.

Shared configuration and grid logic lives in src.train.config; training math,
sampling and checkpoint recovery live in src.train. This entry point prepares
missing processed inputs, runs selected trials and records independent attempt
and detailed histories under output CreditPFN/<experiment>/training/ on project storage.
Model weights use checkpoints/trained/<experiment>/.

Examples (from the repository root):
    python scripts/train_pipeline.py --config config/experiment0/null_pd.yaml --list-trials
    python scripts/train_pipeline.py --config config/experiment0/null_pd.yaml --split-index 0 --trial-index 0

Use scripts/slurm/run_experiment.sh for prepared VSC jobs. --single chooses the
first eligible grid entry; --trial-index selects its stable zero-based index.
"""

from __future__ import annotations

import argparse
import csv
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
from src.train.config import load_train_config, resolve_grid  # noqa: E402
from src.utils.experiment import apply_split_index  # noqa: E402
from src.utils.config import dump_resolved  # noqa: E402
from src.utils.logging_setup import resolve_run_log, setup_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _refuse_unusable_gpu() -> None:
    """Fail fast, with the reason, when the visible GPU cannot run this build.

    Two cases, both seen in practice:
      * a login node's display GPU (Quadro P6000, sm_61) — shared, tiny, and unsupported;
      * any card whose compute capability is absent from `torch.cuda.get_arch_list()`.

    Either way the symptom is a CUDA kernel error raised deep inside the first forward pass,
    long after the config and corpus have been logged, which reads like a model bug.
    """
    import os
    import socket

    try:
        import torch
        if not torch.cuda.is_available():
            return                       # CPU is a legitimate choice; the loop handles it
        props = torch.cuda.get_device_properties(0)
    except Exception:                                          # pragma: no cover
        return

    cap = f"sm_{props.major}{props.minor}"
    supported = set(torch.cuda.get_arch_list())
    host = socket.gethostname()
    on_login = "login" in host and not os.environ.get("SLURM_JOB_ID")
    if cap in supported and not on_login:
        return

    lines = [
        "=" * 78,
        "REFUSING TO TRAIN — the visible GPU cannot run this PyTorch build.",
        "=" * 78,
        f"  host             : {host}",
        f"  slurm job        : {os.environ.get('SLURM_JOB_ID', '<none — interactive>')}",
        f"  gpu              : {props.name} ({cap}, {props.total_memory / 1e9:.1f} GB)",
        f"  pytorch supports : {' '.join(sorted(supported))}",
    ]
    if on_login:
        lines.append("")
        lines.append("  A login node's GPU is a shared DISPLAY device, not a compute GPU.")
    if cap not in supported:
        lines.append("")
        lines.append(f"  This build has no {cap} kernels, so the first forward pass dies with")
        lines.append("  'no kernel image is available for execution on the device'.")
    lines += [
        "",
        "  Submit it as a batch job instead:",
        "      bash scripts/slurm/run_experiment.sh config/experiment1/pd.yaml",
        "  or, for a single trial:",
        "      sbatch --array=0-0 --export=ALL,CREDITPFN_CONFIG=config/experiment1/pd.yaml \\",
        "             scripts/slurm/train_pd.slurm",
        "",
        "  To train on CPU on purpose, pass device=cpu.",
        "=" * 78,
    ]
    raise SystemExit("\n".join(lines))


# --------------------------------------------------------------------------- #
# Explicit-ID validation
# --------------------------------------------------------------------------- #


def _validate_corpus_ids_or_raise(cfg, *, track: str) -> None:
    """Fail fast if ``cfg.corpus.train_dataset_ids`` / ``test_dataset_ids``
    contain IDs that aren't registered in ``DATASET_METADATA`` for the
    active track.

    Without this, a typo (e.g. ``0002.taiwan_creditcard`` instead of
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


def _run_provenance(cfg, base_checkpoint: str, l2sp_lambda: float | None = None) -> dict:
    """The settings that are NOT swept but still decide what a number means.

    Recorded per row rather than per run because a manifest is read on its own, months
    later, by someone reconstructing what produced a score. `git_commit` and the
    submodule pin answer "which code and which literature snapshot" — the cluster pulls
    `origin/main`, so the commit is the only reliable identifier of what actually ran.
    """
    import subprocess
    from src.train.tabicl_compat import model_family

    def _git(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), cwd=_REPO, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:                                      # pragma: no cover
            return ""

    fam = model_family(base_checkpoint)
    tag = "tabicl" if fam == "tabicl" else ("v2.6" if "v2.6" in base_checkpoint else
                                           "v3" if "v3" in base_checkpoint else "v2")
    caps = {}
    try:
        from omegaconf import OmegaConf
        data_cfg = OmegaConf.load("config/data.yaml")
        caps = OmegaConf.to_container(data_cfg.finetuning.max_rows_per_epoch, resolve=True)
    except Exception:                                          # pragma: no cover
        pass
    opt = getattr(cfg, "optimizer", None)
    sched = getattr(cfg, "scheduler", None)
    from src.train.config import limit_training_rows
    return {
        "max_rows_per_epoch": limit_training_rows(cfg, int(caps.get(tag, caps.get("default", 0)) or 0)),
        # The SWEPT per-trial value (what training actually used); falls back to the config
        # default only when λ is not an axis. Reading `opt` unconditionally here is what made
        # every row read 0.003 while the grid believed it swept {0, 0.003} (fixed 22-09-2026).
        "l2sp_lambda": float(
            l2sp_lambda if l2sp_lambda is not None
            else (getattr(opt, "l2sp_lambda", float("nan")) if opt is not None else float("nan"))
        ),
        "warmup_fraction": float(getattr(sched, "warmup_fraction", float("nan"))
                                 if sched is not None else float("nan")),
        "min_lr_fraction": float(getattr(sched, "min_lr_fraction", float("nan"))
                                 if sched is not None else float("nan")),
        "tfm_library_pin": _git("submodule", "status", "tfm-library")[:48],
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        **_device_snapshot(),
    }


def _device_snapshot() -> dict:
    """Which GPU this trial ran on, and how much of it was used.

    Recorded per trial rather than per job because a slurm array can land its tasks on
    different partitions, and `AGENTS_MEMORY.md` has to be able to say "120 GPU-hours on B200" without
    anyone re-reading the logs.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu_name": "cpu", "gpu_total_gb": float("nan"),
                    "peak_gpu_gb": float("nan")}
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu_name": str(props.name),
            "gpu_total_gb": round(props.total_memory / 1e9, 2),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
    except Exception:                                          # pragma: no cover
        return {"gpu_name": "", "gpu_total_gb": float("nan"),
                "peak_gpu_gb": float("nan")}


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

    # NEW (12-08-2026) — the run's own configuration, so the manifest is
    # self-describing. docs/AGENTS_MEMORY.md is written from these columns; without them a
    # score cannot be attributed to a setting, and the corpus keeps changing.
    min_train_rows:         int   = 0
    total_optimizer_steps:  int   = 0
    epochs_run:             int   = 0
    steps_per_epoch:        int   = 0
    train_rows_total:       int   = 0
    test_rows_total:        int   = 0
    train_dataset_ids:      str   = ""      # ";"-joined, so one CSV cell holds the corpus
    test_dataset_ids:       str   = ""
    final_drift:            float = float("nan")
    max_rows_per_epoch:     int   = 0       # the resolved per-step row cap for this base
    l2sp_lambda:            float = float("nan")
    warmup_fraction:        float = float("nan")
    min_lr_fraction:        float = float("nan")
    tfm_library_pin:        str   = ""      # which literature snapshot this ran against
    git_commit:             str   = ""

    # COMPUTE ACCOUNTING (19-08-2026). A paper reports what a result cost, and a reviewer asks
    # whether the comparison was compute-matched. `elapsed_sec` alone cannot answer either:
    # it hides which GPU ran the trial and how much of it was used, so two numbers from
    # different partitions look comparable when they are not.
    gpu_name:               str   = ""      # e.g. "NVIDIA B200"
    gpu_total_gb:           float = float("nan")
    peak_gpu_gb:            float = float("nan")   # torch.cuda.max_memory_allocated
    sec_per_step:           float = float("nan")   # elapsed_sec / total_optimizer_steps
    gpu_hours:              float = float("nan")   # elapsed_sec / 3600, the billable unit
    # COMPUTE ACTUALLY DONE, not just time spent. A paper reports the cost of a result and a
    # reviewer asks whether two arms were compute-matched; wall-clock cannot answer either,
    # because a frozen-backbone trial does a fraction of the work per step at the same
    # seconds-per-step. FLOPs are ESTIMATED as 2 x params x rows for the forward pass and 4 x
    # for forward+backward (the standard convention: backward is ~2x forward), summed over the
    # steps actually taken — an order-of-magnitude figure, not an instrumented count.
    trainable_params:       int   = 0
    total_params:           int   = 0
    est_tflops:             float = float("nan")   # 10^12 FLOPs over the whole trial
    rows_seen:              int   = 0              # sum of rows over all steps
    # Epoch a patience-4 early-stopping rule WOULD have fired at; -1 = still improving when
    # the step budget ran out. Diagnostic only — nothing stops early. Use it to size the
    # budget for the next run instead of guessing.
    would_stop_epoch:       int   = -1


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
        cfg = load_train_config(overrides)
    from src.utils.paths import activate_experiment
    activate_experiment(cfg)
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
    full_grid = resolve_grid(cfg, single=False)

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

    # Workspace settings must precede the device probe's CUDA initialization.
    from src.train.recovery import configure_execution
    configure_execution(bool(getattr(cfg.train, "deterministic", False)))
    _refuse_unusable_gpu()
    LOGGER.info(
        "Training plan: %d run(s) on track=%s (%s; full grid has %d)",
        len(plan), track, plan_label, len(full_grid),
    )

    csv_path = manifests_dir() / f"{cfg.run_name}_{track}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 3) per-trial training
    from src.train.loop import descriptive_name, train_one_config

    # Per-epoch CSVs live in output CreditPFN/<experiment>/training/<track>/<descriptive_name>.csv
    from src.utils.paths import training_dir
    epoch_csv_dir = training_dir(track)
    epoch_csv_dir.mkdir(parents=True, exist_ok=True)

    rows: list[RunRow] = []
    failures = 0
    divergences = 0
    t_outer = time.monotonic()

    for trial_idx_local, (base, lr, use_lora, query_fraction, accumulate, pass_mode,
                         min_train_rows, l2sp_lambda) in enumerate(plan, start=1):
        global_idx = (
            trial_index if trial_index is not None
            else (trial_idx_local - 1)
        )
        LOGGER.info(
            "\n=== Trial %d/%d (global %d)  base=%s  lr=%g  frozen_axis=%s  qf=%.2f  acc=%d  "
            "pass=%s  min_train_rows=%d  l2sp=%s ===",
            trial_idx_local, len(plan), global_idx,
            Path(base).name, lr, use_lora, query_fraction, accumulate, pass_mode,
            min_train_rows, "cfg" if l2sp_lambda is None else f"{l2sp_lambda:g}",
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
            min_train_rows=min_train_rows, l2sp_lambda=l2sp_lambda,
            adaptation_mode=("frozen_backbone" if use_lora and (
                not bool(getattr(cfg.lora, "enabled", False)) or "tabicl" in base) else None)
                if OmegaConf.select(cfg, "experiment.fingerprint", default=False) else None,
        ).removesuffix(".ckpt")

        identity = None
        if OmegaConf.select(cfg, "experiment.fingerprint", default=False):
            from src.utils.experiment import trial_identity
            identity = trial_identity(cfg, full_grid[global_idx])
            if OmegaConf.select(cfg, "experiment.require_plan", default=False):
                from src.utils.prepare_experiment import assert_prepared
                assert_prepared(cfg, global_idx, identity)

        # Enrich a standalone log, but keep the shell-owned job path stable:
        # recovery children all receive that same path from their parent.
        # Renaming it leaves later children writing orphan summary files.
        try:
            current_log = log.path if hasattr(log, "path") else log_path
            if current_log is not None and not os.environ.get("CREDITPFN_ACTIVE_LOG"):
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
        # `python -m src.utils.clean_run --clean --stages train`).
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
            # A DIVERGED checkpoint is saved for inspection but is NOT a completed trial, so
            # RE-RUN it on resubmit instead of skipping — this is what makes "resubmit and the
            # unfinished ones retrain" true. schema_version-1 provenance has no `diverged` key
            # (reads as not-diverged), so a pre-2026-09 diverged checkpoint must be deleted by
            # hand (clean_run --clean --stages train, or a manifest-driven sweep) to force it.
            _diverged_ckpt = False
            try:
                import json as _json
                with open(expected_prov, encoding="utf-8") as _pf:
                    _prov = _json.load(_pf)
                    _diverged_ckpt = bool(_prov.get("diverged", False))
            except (OSError, ValueError, TypeError) as exc:
                raise RuntimeError(f"Unreadable checkpoint provenance: {expected_prov}. "
                                   "Inspect it before resuming; no checkpoint was overwritten.") from exc
            from src.train.sampling import PROTOCOL_VERSION
            if _prov.get("training_protocol_version") != PROTOCOL_VERSION:
                raise RuntimeError("Existing checkpoint uses an older training protocol. "
                                   "Use a new run_name; preserve legacy results.")
            if identity and _prov.get("trial_identity", {}).get("sha256") != identity["sha256"]:
                raise RuntimeError("Existing checkpoint has a different configuration, corpus, base, "
                                   "code or environment fingerprint. Use a new run_name.")
            if not _diverged_ckpt or identity:
                LOGGER.info(
                    "SKIP trial %d (global %d): checkpoint already exists at %s "
                    "— delete the file or use `clean_run --clean --stages train` "
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
                    status="DIVERGED" if _diverged_ckpt else "SKIP",
                    error=_prov.get("diverge_reason") if _diverged_ckpt else None,
                    min_train_rows=int(min_train_rows),
                    **_run_provenance(cfg, base, l2sp_lambda),
                ))
                _write_csv([rows[-1]], csv_path, append=csv_append)
                if not csv_append:
                    csv_append = True
                continue
            LOGGER.info(
                "RE-RUN trial %d (global %d): existing checkpoint at %s is marked DIVERGED "
                "in its provenance — retraining it.",
                trial_idx_local, global_idx, expected_ckpt,
            )

        from src.utils.training_files import TrainingFiles
        diagnostic_files = TrainingFiles(epoch_csv_dir, run_basename)
        epoch_csv = diagnostic_files.directory / f"{run_basename}.csv"
        if epoch_csv.exists():
            epoch_csv.unlink()              # fresh file per run
        _epoch_csv_init: dict[str, bool] = {"written_header": False}
        trajectory_csv = epoch_csv.with_name(run_basename + ".trajectory.csv")
        trajectory_csv.unlink(missing_ok=True)
        _trajectory_csv_init = {"written_header": False}

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
                "successful_updates":       int(rec.successful_updates),
                "processed_rows":           int(rec.processed_rows),
                "record_type":              rec.record_type,
                "monitor_seconds":          float(rec.monitor_seconds),
                "training_seconds":         float(rec.training_seconds),
                "compute_seconds":          float(rec.compute_seconds),
                "data_wait_seconds":        float(rec.data_wait_seconds),
                "weight_drift":             float(rec.weight_drift),
                "l2sp_penalty":             float(rec.l2sp_penalty),
                # Diagnostics that were computed every epoch and only ever printed. These are
                # what distinguish "the model was moved and nothing happened" from "the model
                # was never moved", which is the whole question of the project.
                "grad_norm_mean":            float(rec.grad_norm_mean),
                "grad_norm_max":             float(rec.grad_norm_max),
                "clipped_frac":              float(rec.clipped_frac),
                "lr_applied":                float(rec.lr_applied),
            }
            # Per-dataset loss and per-stage drift, one column each. The
            # column SET is fixed by the first row written (the epoch=-1
            # baseline has neither, so the header is taken from the first
            # trained epoch instead — see the header logic below).
            row.update({f"loss__{k}": float(v)
                        for k, v in sorted(rec.per_dataset_loss.items())})
            row.update({f"metric__{k}": float(v) for k, v in sorted(rec.per_dataset_metric.items())})
            row.update({f"drift__{k}": float(v)
                        for k, v in sorted(rec.stage_drift.items())})
            row.update({f"pdrift__{k}": float(v)
                        for k, v in sorted(rec.layer_drift.items())})

            row.update({f"score__{k}": float(v) for k,v in sorted(rec.per_dataset_scores.items())})
            row["cuda_peak_allocated_bytes"] = rec.cuda_peak_allocated_bytes
            row["cuda_peak_reserved_bytes"] = rec.cuda_peak_reserved_bytes
            if rec.record_type == "trajectory" and rec.parameter_statistics:
                import gzip
                parameter_path = _path.with_name(run_basename + ".parameters.csv.gz")
                first = not _flag.get("parameters_started", False)
                with gzip.open(parameter_path, "wt" if first else "at", newline="", encoding="utf-8") as stream:
                    values = [dict(successful_updates=rec.successful_updates, **entry) for entry in rec.parameter_statistics]
                    writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                    if first:
                        writer.writeheader()
                    writer.writerows(values)
                _flag["parameters_started"] = True

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
            from src.train.telemetry import ResourceMonitor
            with diagnostic_files, ResourceMonitor(diagnostic_files.directory / (run_basename + ".resources.csv"),
                    enabled=bool(getattr(cfg.train, "resource_diagnostics", False)),
                    interval=float(getattr(cfg.train, "resource_interval_seconds", 20))):
                result = train_one_config(
                    cfg, track=track,
                    base_checkpoint=base,
                    learning_rate=lr,
                    use_lora=use_lora,
                    query_fraction=query_fraction,
                    accumulate_grad_batches=accumulate,
                    pass_mode=pass_mode,
                    min_train_rows=min_train_rows,
                    l2sp_lambda=l2sp_lambda,
                    on_epoch_end=_on_epoch_end,
                    on_trajectory_end=lambda rec: _on_epoch_end(
                        rec, _path=trajectory_csv, _flag=_trajectory_csv_init),
                    trial_identity=identity,
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
                # The run's own configuration — see the RunRow docstring.
                min_train_rows=int(min_train_rows or 0),
                total_optimizer_steps=result.total_optimizer_steps,
                epochs_run=result.epochs_run,
                steps_per_epoch=result.steps_per_epoch,
                train_rows_total=result.train_rows_total,
                test_rows_total=result.test_rows_total,
                train_dataset_ids=";".join(result.train_dataset_ids),
                test_dataset_ids=";".join(result.test_dataset_ids),
                final_drift=result.final_drift,
                # Compute accounting, so the manifest can answer "what did this cost"
                # and "were these two arms compute-matched" without the logs.
                trainable_params=int(getattr(result, 'trainable_params', 0) or 0),
                total_params=int(getattr(result, 'total_params', 0) or 0),
                est_tflops=float(getattr(result, 'est_tflops', float('nan'))),
                rows_seen=int(getattr(result, 'rows_seen', 0) or 0),
                would_stop_epoch=int(
                    getattr(result, 'would_stop_epoch', None) or -1),
                sec_per_step=(float(result.elapsed_sec)
                              / max(1, int(getattr(result, 'total_optimizer_steps', 0) or 1))),
                gpu_hours=float(result.elapsed_sec) / 3600.0,
                **_run_provenance(cfg, base, l2sp_lambda),
            ))
            if result.diverged:
                # A numerically completed but diverged checkpoint is excluded
                # from eval and must not make its SLURM task write a success
                # sentinel. Returning non-zero lets the post-training gate
                # detect an incomplete scientific grid.
                divergences += 1
        except Exception as exc:                           # noqa: BLE001
            from src.train.recovery import TrainingInterrupted
            if isinstance(exc, TrainingInterrupted):
                LOGGER.warning("INTERRUPTED: %s. Resubmit the same configuration to continue.", exc)
                row = RunRow(track=track, base_checkpoint=base, learning_rate=lr,
                    use_lora=use_lora, query_fraction=query_fraction,
                    accumulate_grad_batches=int(accumulate), epoch_pass_mode=pass_mode,
                    seed=int(cfg.seed), n_train_datasets=0, n_test_datasets=0,
                    final_ckpt_path="", elapsed_sec=time.monotonic() - t_trial,
                    status="INTERRUPTED", error=str(exc), min_train_rows=int(min_train_rows),
                    **_run_provenance(cfg, base, l2sp_lambda))
                _write_csv([row], csv_path, append=csv_append)
                return 75
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
                min_train_rows=int(min_train_rows),
                **_run_provenance(cfg, base, l2sp_lambda),
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
        "--config", default=None, metavar="PATH",
        help="Training config to run. Defaults to config/train.yaml; point it at "
             "config/experiment<N>_<track>.yaml to run one experiment.",
    )
    p.add_argument(
        "--split-index", type=int, default=None, metavar="K",
        help="Which random train/test dataset split to run (0..n_splits-1). Sets "
             "corpus.split_seed and tags the run name, so 28 splits of one grid land in "
             "28 distinct manifests instead of overwriting each other.",
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
    import signal
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: os.environ.__setitem__("CREDITPFN_STOP_REQUESTED", "1"))
    args, overrides = _parse_args()
    if args.list_trials:
        cfg = apply_split_index(load_train_config(overrides, getattr(args, 'config', None)), getattr(args, 'split_index', None))
        print(len(resolve_grid(cfg, single=False)))
        raise SystemExit(0)
    if args.trial_family is not None:
        from src.train.tabicl_compat import model_family
        cfg = apply_split_index(load_train_config(overrides, getattr(args, 'config', None)), getattr(args, 'split_index', None))
        grid = resolve_grid(cfg, single=False)
        if not 0 <= args.trial_family < len(grid):
            # Over-sized slurm arrays are a legitimate pattern; a surplus
            # index is not an error. Print nothing and exit 0 so the caller's
            # `[[ "$FAMILY" == tabicl ]]` test simply doesn't match.
            raise SystemExit(0)
        print(model_family(grid[args.trial_family][0]))
        raise SystemExit(0)
    raise SystemExit(run(
        cfg=apply_split_index(load_train_config(overrides, args.config), args.split_index),
        single=args.single,
        trial_index=args.trial_index,
        overrides=overrides,
        log_path=args.log_path,
    ))
