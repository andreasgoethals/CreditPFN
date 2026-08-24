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


#: The sweep this pipeline runs unless told otherwise. A PHASE config (see `config/phases/`)
#: is a full replacement for it, not a patch: each phase answers one question and carries its
#: own grid, so `docs/EXPERIMENT_PLAN.md` can be executed one file at a time without editing
#: the default and without a pile of `key=value` overrides in a job script.
DEFAULT_TRAIN_CONFIG = "config/train.yaml"


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
        "      bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml",
        "  or, for a single trial:",
        "      sbatch --array=0-0 --export=ALL,CREDITPFN_CONFIG=config/experiment1_pd.yaml \\",
        "             scripts/slurm/train_pd.slurm",
        "",
        "  To train on CPU on purpose, pass device=cpu.",
        "=" * 78,
    ]
    raise SystemExit("\n".join(lines))


def _load_cfg(overrides: list[str] | None = None, config_path: str | None = None):
    """Load the training config and apply ``key=value`` overrides.

    `config_path` selects a phase config; it defaults to `config/train.yaml`. Phase files are
    self-contained rather than deltas, because a delta that silently inherits an axis is how a
    run ends up sweeping something nobody intended.
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(DEFAULT_TRAIN_CONFIG)
    if config_path and str(config_path) != DEFAULT_TRAIN_CONFIG:
        # EXPERIMENT CONFIGS ARE DELTAS over config/train.yaml, not replacements. The training
        # path requires ~25 keys (cfg.checkpoint.trained_dir, cfg.lora.*, cfg.train.amp,
        # cfg.optimizer.lr, ...) that have nothing to do with the science of one experiment, so
        # a self-contained experiment file is either 100 lines of machinery or it crashes at
        # checkpoint-save time. The base holds the machinery; the experiment holds what differs.
        # The silent-inheritance risk this trades against is covered by dumping the RESOLVED
        # config to output/manifests/resolved/ and by the manifest recording every swept axis.
        cfg = OmegaConf.merge(cfg, OmegaConf.load(str(config_path)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _apply_split_index(cfg, split_index: int | None):
    """Point the config at one of the random dataset splits.

    The split is NOT a sweep axis: making it one would multiply the trial grid by 28 and put a
    split tag inside every trial name. Instead it seeds the draw and tags the run name, so each
    split writes its own manifest and the analysis averages across them.
    """
    if split_index is None:
        return cfg
    cfg.corpus.split_seed = int(split_index)
    cfg.run_name = f"{cfg.run_name}_s{int(split_index):02d}"
    return cfg


def _resolve_grid(
    cfg, *, single: bool,
) -> list[tuple[str, float, bool, float, int, str]]:
    """Materialise the ``(base, lr, use_lora, query_fraction, accumulate,
    epoch_pass_mode, min_train_rows)`` tuples to train.

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
    # REQUIRED, not defaulted. These three differ in every experiment, so they live only in the
    # experiment config — and a silent fallback here would run a grid nobody asked for. The
    # project has been bitten twice by exactly that (`query_fractions` defaulting to 0.20 when
    # the sweep wanted 0.40; `min_train_rows` inherited as a stale two-value axis).
    # PRESENCE is required; the VALUE may be null where null means something.
    # `l2sp_lambdas: null` legitimately means "not an axis, use optimizer.l2sp_lambda", so
    # absence and null are different situations and only absence is an error.
    def _present(key: str) -> bool:
        try:
            return key in cfg.tunable
        except TypeError:                       # SimpleNamespace, used by the tests
            return hasattr(cfg.tunable, key)

    _absent = [k for k in ("learning_rates", "l2sp_lambdas", "frozen_backbone")
               if not _present(k)]
    _null = [k for k in ("learning_rates", "frozen_backbone")
             if _present(k) and getattr(cfg.tunable, k, None) is None]
    if _absent or _null:
        _names = ", ".join(f"tunable.{k}" for k in _absent + _null)
        raise SystemExit(
            f"config error: {_names} must be set by the experiment config.\n"
            "  These axes differ per experiment, so config/train.yaml deliberately\n"
            "  does not define them - a silent fallback would run a grid nobody\n"
            "  asked for. Add them to the --config file, e.g.\n"
            "      tunable:\n"
            "        learning_rates: [3.0e-7, 1.0e-6, 1.0e-5, 1.0e-4]\n"
            "        l2sp_lambdas: [0.0, 0.003]   # null = not an axis\n"
            "        frozen_backbone: [false]\n"
        )
    lrs = [float(x) for x in cfg.tunable.learning_rates]
    # `frozen_backbone` is the honest name for this axis and the accepted spelling from
    # run-9: on TabICL it trains the head only, on TabPFN it means LoRA. `use_lora` is
    # still read so older configs keep working, but a config setting NEITHER gets [False]
    # rather than a silently dropped axis.
    raw_lora = getattr(cfg.tunable, "frozen_backbone", None)
    if raw_lora is None:
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

    raw_mtr = getattr(cfg.corpus, "min_train_rows", [0]) if hasattr(cfg, "corpus") else [0]
    if isinstance(raw_mtr, (int, float)):
        mtrs = [int(raw_mtr)]
    else:
        mtrs = [int(x) for x in raw_mtr]

    # ANCHOR STRENGTH, swept from run-9. `tunable.l2sp_lambdas` absent means "use the
    # single value in `finetuning.l2sp_lambda`", which is what every earlier run did.
    raw_l2 = getattr(cfg.tunable, "l2sp_lambdas", None)
    if raw_l2 is None:
        l2sps: list = [None]
    elif isinstance(raw_l2, (int, float)):
        l2sps = [float(raw_l2)]
    else:
        l2sps = [float(x) for x in raw_l2]

    #: Families for which the adapter arm (`use_lora: true`) is generated. Empty or
    #: absent = every family, which is what every run before run-8 did. LoRA on TabPFN
    #: was a measured no-op in runs 4, 6 and 7 and cost a third of the grid; on TabICLv2
    #: the same flag means freeze-backbone, which is a different mechanism and still
    #: worth measuring.
    adapter_families = [str(x).lower() for x in
                        (getattr(cfg.tunable, "adapter_families", None) or [])]

    def _adapter_allowed(base_path: str) -> bool:
        if not adapter_families:
            return True
        from src.train.tabicl_compat import model_family
        fam = model_family(base_path)
        name = str(base_path).lower()
        return any(f in (fam, name) or f in name for f in adapter_families)

    if single:
        return [(
            str(bases[0]), float(lrs[0]), bool(loras[0]), float(qfs[0]),
            int(accs[0]), str(pms[0]), int(mtrs[0]), l2sps[0],
        )]
    return [
        (str(b), float(lr), bool(lo), float(qf), int(ac), str(pm), int(mtr), l2)
        for b, lr, lo, qf, ac, pm, mtr, l2
        in itertools.product(bases, lrs, loras, qfs, accs, pms, mtrs, l2sps)
        if not lo or _adapter_allowed(str(b))
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


def _run_provenance(cfg, base_checkpoint: str) -> dict:
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
    tag = "tabicl" if fam == "tabicl" else ("v2.6" if "v2.6" in base_checkpoint else "v3")
    caps = {}
    try:
        from omegaconf import OmegaConf
        data_cfg = OmegaConf.load("config/data.yaml")
        caps = OmegaConf.to_container(data_cfg.finetuning.max_rows_per_epoch, resolve=True)
    except Exception:                                          # pragma: no cover
        pass
    opt = getattr(cfg, "optimizer", None)
    sched = getattr(cfg, "scheduler", None)
    return {
        "max_rows_per_epoch": int(caps.get(tag, caps.get("default", 0)) or 0),
        "l2sp_lambda": float(getattr(opt, "l2sp_lambda", float("nan"))
                             if opt is not None else float("nan")),
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
    different partitions, and `RESULTS.md` has to be able to say "120 GPU-hours on B200" without
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
    # self-describing. docs/RESULTS.md is written from these columns; without them a
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
        cfg = _apply_split_index(_load_cfg(overrides, getattr(args, 'config', None)), getattr(args, 'split_index', None))
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

    _refuse_unusable_gpu()
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

    for trial_idx_local, (base, lr, use_lora, query_fraction, accumulate, pass_mode,
                         min_train_rows, l2sp_lambda) in enumerate(plan, start=1):
        global_idx = (
            trial_index if trial_index is not None
            else (trial_idx_local - 1)
        )
        LOGGER.info(
            "\n=== Trial %d/%d (global %d)  base=%s  lr=%g  lora=%s  qf=%.2f  acc=%d  "
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
            row.update({f"drift__{k}": float(v)
                        for k, v in sorted(rec.stage_drift.items())})
            row.update({f"pdrift__{k}": float(v)
                        for k, v in sorted(rec.layer_drift.items())})

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
                min_train_rows=min_train_rows,
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
                **_run_provenance(cfg, base),
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
        "--config", default=None, metavar="PATH",
        help="Training config to run. Defaults to config/train.yaml; point it at "
             "config/phases/<phase>.yaml to run one phase of docs/EXPERIMENT_PLAN.md.",
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
    args, overrides = _parse_args()
    if args.list_trials:
        cfg = _apply_split_index(_load_cfg(overrides, getattr(args, 'config', None)), getattr(args, 'split_index', None))
        print(len(_resolve_grid(cfg, single=False)))
        raise SystemExit(0)
    if args.trial_family is not None:
        from src.train.tabicl_compat import model_family
        cfg = _apply_split_index(_load_cfg(overrides, getattr(args, 'config', None)), getattr(args, 'split_index', None))
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
