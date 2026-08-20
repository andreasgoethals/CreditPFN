"""Single-config continued-pretraining loop.

One call to :func:`train_one_config` =

    1. Build the corpus split (train / test by dataset_id) from cfg.
    2. Load the requested base TabPFN checkpoint.
    3. Wrap an AdamW optimiser around it + a linear-warmup-then-cosine-
       decay LR scheduler over the total number of optimisation steps.
       (See :func:`make_warmup_cosine_schedule` for the exact formula
       — it matches HuggingFace's ``get_cosine_schedule_with_warmup``,
       which is what TabPFN's own ``FinetunedTabPFNClassifier`` uses.)
    4. Run ``cfg.train.epochs`` epochs of:
         for chunk in train_chunks (shuffled):
             forward → loss → backward → (optional grad-clip) → step
       …with mixed precision on CUDA, gradient accumulation, and NO
       validation. There is no early stopping — the user explicitly
       chose fixed-epoch training (cf. discussion in chat 2026-05-04
       on the val-set noise problem with ~10 datasets).
    5. Save the FINAL-epoch weights to
       ``cfg.checkpoint.trained_dir/<descriptive_name>.ckpt`` in
       Prior Labs format (state_dict + config), so the file
       round-trips through ``TabPFNClassifier(model_path=...)`` /
       ``TabPFNRegressor(model_path=...)``.
    6. Compute the test metric ONCE on the held-out test split and
       return it. This number is reported but NEVER used to make any
       within-training decision — there is no leak.

The function is one config. Iterating over the cartesian product of
``cfg.tunable`` lists lives in ``scripts/train_pipeline.py``, not
here, because that's a script-level concern (the user's instruction).
"""

from __future__ import annotations

import logging
import math
import os
import platform
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.train.corpus import (
    CorpusSplit,
    DatasetRef,
    resolve_ids_for_track,
    split_corpus,
)
from src.train.dataloader import (
    ProcessedDatasetLoader, TabPFNBatch, identity_collate, prepare_eval_chunk,
)
from src.train.metrics import (
    classification_metric, regression_metric,
    mean_ignore_nan,
)
from src.train.model import load_tabpfn_for_training, save_finetuned
from src.train.tabicl_compat import model_family
from src.utils.paths import (
    resolve_output_path, resolve_staging_path, resolve_writable_staging_path,
)

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class EpochRecord:
    """One row of the training history.

    ``train_metric`` / ``test_metric`` carry the **primary** monitoring
    metric (ROC-AUC for PD, RMSE for LGD) averaged over a small
    subsample of the train- and test-dataset chunks at end of epoch.
    Both are ``NaN`` when ``cfg.train.epoch_eval_subsample_samples == 0``
    (per-epoch eval disabled).

    ``secondary_*`` carries an optional **secondary** metric also logged
    each epoch (R² for LGD; unused for PD, where the secondary fields
    stay NaN with ``secondary_metric_name == ""``). The forward pass is
    shared with the primary metric — see
    :func:`evaluate_on_split` — so the extra column is essentially free.

    ``elapsed_sec`` is **cumulative** training time since the loop
    started; ``epoch_time_sec`` is the wall-clock for just this epoch
    (so you can spot a slow epoch without diffing the cumulative
    column).
    """
    epoch: int
    train_loss: float
    elapsed_sec: float
    lr: float
    train_metric: float = float("nan")
    test_metric:  float = float("nan")
    metric_name:  str   = ""
    secondary_train_metric: float = float("nan")
    secondary_test_metric:  float = float("nan")
    secondary_metric_name:  str   = ""
    epoch_time_sec: float = 0.0
    optimizer_steps: int = 0
    amp_skipped_steps: int = 0
    data_skipped_steps: int = 0

    # Mean training loss for THIS epoch, per source dataset. The epoch's
    # scalar `train_loss` averages the whole corpus, which hides the question
    # continued pretraining actually raises: is the corpus being learned
    # uniformly, or do some datasets improve while others degrade? A
    # per-dataset trajectory answers that directly (and is the per-dataset
    # forgetting check). Written to the per-epoch CSV as `loss__<dataset_id>`.
    per_dataset_loss: dict[str, float] = field(default_factory=dict)

    # Relative weight drift ``‖w−w0‖/‖w0‖`` per top-level model stage, on
    # monitored epochs only (it needs a full pass over the parameters).
    # Answers WHERE credit-specialisation lands: for TabICLv2 the stages are
    # col_embedder / row_interactor / icl_predictor, for TabPFN whatever its
    # top-level modules are. Empty on non-monitored epochs and under LoRA
    # (no anchor exists there).
    stage_drift: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Returned by :func:`train_one_config`.

    The training loop produces checkpoints and records summary metrics
    (epoch=-1 baseline + final epoch train/test scores, divergence flag).
    Per-(model × dataset) scoring is still the eval pipeline's job; this
    summary just lets the manifest CSV answer "did this trial help?".
    """
    final_ckpt_path: Path
    history: list[EpochRecord] = field(default_factory=list)
    n_train_datasets: int = 0
    n_test_datasets: int = 0
    elapsed_sec: float = 0.0
    descriptive_name: str = ""           # the basename of final_ckpt_path

    # NEW (2026-05-28) — summary fields surfaced through to the manifest.
    diverged: bool = False               # True if the loop aborted early
    diverged_at_epoch: int | None = None # last good epoch index (None if no divergence)
    diverge_reason: str = ""             # short tag — "loss_const" / "auc_random" / "amp_skip_storm"
    baseline_train_metric: float = float("nan")    # epoch=-1, primary metric
    baseline_test_metric:  float = float("nan")
    final_train_metric:    float = float("nan")    # last good epoch
    final_test_metric:     float = float("nan")
    final_train_loss:      float = float("nan")    # CE for PD, NLL for LGD
    final_secondary_train: float = float("nan")    # brier (PD) / r2 (LGD)
    final_secondary_test:  float = float("nan")
    primary_metric_name:   str = ""                # "roc_auc" / "rmse"
    secondary_metric_name: str = ""                # "brier_score" / "r2"

    # NEW (12-08-2026) — what this trial ACTUALLY ran, as opposed to what the config
    # asked for. Every one of these has been the subject of a wrong conclusion:
    #   * realised steps    — run-7's LGD ran 800 of a 9 100 target and nobody saw it
    #   * epochs            — trimmed or extended per base, so it is not cfg.train.epochs
    #   * corpus rows/ids   — "17 datasets" means something different every quarter
    #   * min_train_rows    — a swept axis as of run-8
    #   * final drift       — the difference between "no effect" and "did not train"
    total_optimizer_steps: int = 0
    epochs_run:            int = 0
    steps_per_epoch:       int = 0
    min_train_rows:        int = 0
    train_dataset_ids:     tuple[str, ...] = ()
    test_dataset_ids:      tuple[str, ...] = ()
    train_rows_total:      int = 0
    test_rows_total:       int = 0
    final_drift:           float = float("nan")   # ||w - w0|| at the last monitored epoch


# --------------------------------------------------------------------------- #
# Public utility: descriptive checkpoint name
# --------------------------------------------------------------------------- #


_BASE_VERSION_RE = re.compile(r"tabpfn-(v\d+(?:\.\d+)?)-")


def _resolve_max_rows_per_epoch(base_checkpoint: str | Path, mapping) -> int:
    """Look up the per-version `max_rows_per_epoch` cap.

    Accepts either an int (legacy single-value config) or a mapping
    ``{"v3": 26000, "v2.6": 11000, ...}`` (the PD 2-member per-step caps;
    ``train_one_config`` scales them down for higher member counts). For a
    mapping, we extract
    the leading ``v<MAJOR>[.<MINOR>]`` from the base checkpoint's
    filename (e.g. ``tabpfn-v2.6-classifier-…`` → ``"v2.6"``) and
    look up that key, falling back to ``"default"`` if absent.
    """
    if isinstance(mapping, int):
        return int(mapping)
    name = Path(str(base_checkpoint)).name
    # Family key takes precedence over version parsing: a TabICLv2 base like
    # "tabicl-classifier-v2-20260212.ckpt" would otherwise match "v2" (or
    # fall to "default"), silently inheriting a TabPFN-derived cap. TabICLv2's
    # ICL stage is O(rows²) with its own memory profile → own config key.
    if "tabicl" in name.lower():
        key = "tabicl"
    else:
        m = _BASE_VERSION_RE.search(name)
        key = m.group(1) if m else "default"
    if hasattr(mapping, "get"):
        if key in mapping:
            return int(mapping[key])
        if "default" in mapping:
            return int(mapping["default"])
    raise ValueError(
        f"finetuning.max_rows_per_epoch is neither an int nor a mapping "
        f"with a usable key for base={name!r} (resolved version key={key!r}). "
        f"Got: {mapping!r}"
    )


def descriptive_name(
    *, run_name: str, track: str, base_path: str | Path,
    learning_rate: float, seed: int,
    use_lora: bool = False,
    query_fraction: float | None = None,
    accumulate_grad_batches: int | None = None,
    epoch_pass_mode: str | None = None,
    min_train_rows: int | None = None,
    l2sp_lambda: float | None = None,
) -> str:
    """Build the on-disk filename encoding the tunable HPs.

    Schema:
        <run_name>_<track>_<base-stem>_lr<lr>_seed<seed>[_qf<qf>][_acc<K>][_fullpass][_lora].ckpt

    ``query_fraction`` is part of the sweep grid as of 2026-05-21,
    ``accumulate_grad_batches`` as of 2026-05-27, ``epoch_pass_mode`` as
    of 2026-06-01. All are optional in the filename — passing ``None``
    (or ``"one_sample"`` for the pass mode) omits the segment, so legacy
    callers / the default one-step-per-dataset sweep produce identical
    names to before.
    """
    base_stem = Path(str(base_path)).stem
    lr_tag = f"{learning_rate:.0e}".replace("+", "")
    qf_tag = ""
    if query_fraction is not None:
        # 0.20 → "qf20", 0.30 → "qf30", 0.40 → "qf40"
        qf_tag = f"_qf{int(round(query_fraction * 100)):02d}"
    acc_tag = ""
    if accumulate_grad_batches is not None:
        acc_tag = f"_acc{int(accumulate_grad_batches)}"
    # Only the non-default "full_pass" mode adds a tag, so "one_sample"
    # (the default) keeps the exact pre-2026-06-01 filename.
    pass_tag = "_fullpass" if (epoch_pass_mode == "full_pass") else ""
    # Corpus composition is a swept axis as of run-8, so it has to be in the name:
    # two trials that differ only in which training datasets existed are different
    # experiments, and without a tag they would overwrite each other's checkpoint.
    # Absent / 0 keeps the pre-run-8 filename byte-for-byte.
    rows_tag = f"_min{int(min_train_rows)}" if min_train_rows else ""
    # Anchor strength, swept from run-9. `None` = "not swept" and emits nothing, so every
    # trial name written before run-9 is byte-identical and still parses. The tag sits
    # BEFORE the adapter tag because `_NAME_RE` anchors `_lora|_iclhead` to the end.
    l2sp_tag = "" if l2sp_lambda is None else f"_l2sp{float(l2sp_lambda):g}"
    # The `use_lora` grid axis means LoRA for the TabPFN family but
    # FREEZE-BACKBONE / train-ICL-head-only for TabICLv2 (its upstream
    # stage-3 regime; full SFT collapsed TabICLv2 in two independent
    # reports — see src/train/tabicl_compat.py). Tag names must be
    # honest about which adaptation actually ran. Derived from
    # `base_path` HERE so every caller (loop + train_pipeline
    # idempotency + epoch CSVs) stays consistent with zero call-site
    # changes.
    if use_lora and "tabicl" in base_stem.lower():
        lora_tag = "_iclhead"
    else:
        lora_tag = "_lora" if use_lora else ""
    return (
        f"{run_name}_{track}_{base_stem}_lr{lr_tag}_seed{seed}"
        f"{qf_tag}{acc_tag}{pass_tag}{rows_tag}{l2sp_tag}{lora_tag}.ckpt"
    )


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #


def _stage_drift(
    model: torch.nn.Module,
    anchor: dict[str, "torch.Tensor"],
    stage_names: dict[str, list[str]],
    stage_w0: dict[str, float],
) -> dict[str, float]:
    """Relative drift ``‖w−w0‖/‖w0‖`` per top-level model stage.

    One pass over the anchored parameters; called only on monitored epochs.
    Tells us WHERE continued pretraining actually changes the model — e.g.
    whether credit-specialisation reshapes the feature embedder or the
    in-context attention. That is a claim nobody has made for tabular
    foundation models, and it is invisible in the single aggregate drift
    number.
    """
    if not anchor or not stage_names:
        return {}
    live = dict(model.named_parameters())
    out: dict[str, float] = {}
    with torch.no_grad():
        for stage, names in stage_names.items():
            w0 = stage_w0.get(stage, 0.0)
            if w0 <= 0:
                continue
            acc = 0.0
            for n in names:
                p = live.get(n)
                if p is None:
                    continue
                acc += float((p.detach().double() - anchor[n].double())
                             .pow(2).sum().item())
            out[stage] = math.sqrt(acc) / w0
    return out


def _drift_pct(l2sp_value: float, lam: float, w0_norm: float) -> str:
    """Render L2-SP as relative weight drift, ``‖w-w0‖/‖w0‖`` in percent.

    ``l2sp = 0.5 * lam * ‖w-w0‖²`` ⇒ ``‖w-w0‖ = sqrt(2*l2sp/lam)``. Dividing by
    the norm of the anchored start weights gives a scale-free number that is
    directly comparable across architectures and learning rates — unlike the
    raw penalty, whose magnitude also depends on how many tensors are anchored.
    """
    if lam <= 0 or w0_norm <= 0 or l2sp_value < 0:
        return "n/a"
    dw = math.sqrt(2.0 * l2sp_value / lam)
    return f"{100.0 * dw / w0_norm:.3f}%"


def make_warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
    schedule_type: str,
    min_lr_fraction: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear-warmup → cosine-decay LR multiplier.

    Matches HuggingFace's ``get_cosine_schedule_with_warmup``
    (which is what TabPFN's ``FinetunedTabPFNClassifier`` uses
    internally; see ``tfm-library/repositories/TabPFN .txt``):

      * step 0           → multiplier = 0
      * step warmup_steps → multiplier = 1
      * step total_steps  → multiplier = 0  (cosine "warmup_cosine" only)

    ``schedule_type``:
        - ``"constant"``      — multiplier = 1 throughout
        - ``"warmup_only"``   — linear warmup, then constant 1
        - ``"warmup_cosine"`` — linear warmup, then cosine decay to
          ``min_lr_fraction`` × peak (0.0 = the classic decay-to-zero).

    ``min_lr_fraction`` exists because our runs are SHORT. Real-TabPFN decays
    over 20 000 steps; a 600-step run that also decays to exactly zero spends
    its final quarter taking steps of size ~0, so the effective budget is far
    smaller than the step count suggests. Holding a small floor keeps the tail
    of training useful. (Added 2026-08-06 after the first two-family run moved
    v3's weights by only ~0.02 % at lr=3e-7.)
    """
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if schedule_type == "constant":
            return 1.0
        if step < warmup_steps:
            return step / warmup_steps           # 0 at step 0, ~1 just before warmup_steps
        if schedule_type == "warmup_only":
            return 1.0
        if schedule_type == "warmup_cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            floor = min(1.0, max(0.0, float(min_lr_fraction)))
            return floor + (1.0 - floor) * cos
        raise ValueError(f"unknown schedule_type={schedule_type!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _l2sp_penalty(
    model: torch.nn.Module,
    anchor: dict[str, "torch.Tensor"],
    lam: float,
):
    """L2-SP regulariser: ``0.5 * lam * Σ ‖w − w₀‖²`` over anchored params.

    L2-SP (Li et al. 2018) penalises drift from the *starting* weights
    ``w₀`` — here the synthetic-prior checkpoint — instead of from the
    origin (which is what plain AdamW ``weight_decay`` does). Real-TabPFN
    (Garg et al. 2025, §4) uses exactly this term to fight catastrophic
    forgetting of the pretrained prior during continued pretraining.

    Returns ``None`` when nothing is anchored (e.g. LoRA trials, where the
    base is frozen so there is no full-weight drift to penalise).
    """
    total = None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        w0 = anchor.get(name)
        if w0 is None:
            continue
        term = (p - w0).pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        return None
    return 0.5 * float(lam) * total


def _make_optimizer_and_scheduler(
    model: torch.nn.Module, cfg, *, total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW (betas=(0.9, 0.999)) + linear-warmup → cosine-decay schedule.

    Optimizer family and schedule type are hardcoded; only `weight_decay`
    and `warmup_fraction` are exposed via cfg.
    """
    lr = float(cfg.optimizer.lr) if hasattr(cfg.optimizer, "lr") else None
    if lr is None:
        lr = float(cfg.tunable.learning_rates[0])

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.optimizer.weight_decay),
        betas=(0.9, 0.999),
    )
    sched = make_warmup_cosine_schedule(
        optim,
        total_steps=total_steps,
        warmup_fraction=float(cfg.scheduler.warmup_fraction),
        schedule_type="warmup_cosine",
        min_lr_fraction=float(getattr(cfg.scheduler, "min_lr_fraction", 0.0)),
    )
    return optim, sched


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #


def _resolve_device(cfg) -> str:
    pref = str(cfg.device).lower()
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def _resolve_amp_dtype(cfg, device: str) -> tuple[bool, torch.dtype | None]:
    """Resolve whether AMP is active and which CUDA autocast dtype to use.

    BF16 is preferred when supported: unlike FP16 it retains FP32's exponent
    range, so large TabPFN regression gradients do not repeatedly overflow and
    force ``GradScaler`` to discard optimizer steps.  The tensor width remains
    two bytes, so the B200/H100 memory budget is unchanged.
    """
    enabled = bool(cfg.train.amp) and device == "cuda"
    if not enabled:
        return False, None

    requested = str(getattr(cfg.train, "amp_dtype", "auto")).strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "auto": "auto",
    }
    if requested not in aliases:
        raise ValueError(
            "train.amp_dtype must be one of auto, bfloat16/bf16, "
            f"or float16/fp16; got {requested!r}"
        )
    requested = aliases[requested]
    bf16_supported = bool(
        getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    if requested == "auto":
        return True, torch.bfloat16 if bf16_supported else torch.float16
    if requested == "bfloat16" and not bf16_supported:
        LOGGER.warning(
            "train.amp_dtype=bfloat16 requested but this CUDA device does not "
            "report BF16 support; falling back to float16 + GradScaler."
        )
        return True, torch.float16
    return True, torch.bfloat16 if requested == "bfloat16" else torch.float16


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> str:
    """Short git commit of the CreditPFN checkout (``?`` if unavailable).

    Logged in the debug banner so a shared run log pins the EXACT code
    version that produced it — the first thing needed when debugging a run.
    """
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or "?"
    except Exception:                                              # pragma: no cover
        return "?"


def _gpu_total_mem_gb(device: str) -> float | None:
    """Total VRAM (GiB) of the active CUDA device, or None on CPU."""
    if device != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:                                              # pragma: no cover
        return None


def _log_debug_banner(
    *, track: str, device: str, base_checkpoint, save_path,
    learning_rate, use_lora, query_fraction, accumulate, pass_mode,
    n_estimators_finetune, max_rows_per_epoch, max_cells_per_epoch,
    epochs, total_steps, steps_per_epoch, weight_decay, l2sp_lambda,
    warmup_fraction, seed, n_train_ds, n_test_ds, use_amp, amp_dtype,
    context_sampling="?",
) -> None:
    """Emit a single, comprehensive DEBUG banner at training start.

    The goal (user request, 2026-06-23): if a trial crashes or is killed
    mid-run, the log alone must contain EVERY piece of context needed to
    reproduce and debug it — environment, cluster, hardware, library
    versions, resolved storage paths, the full hyperparameter set, and the
    corpus shape. This is logged BEFORE the first forward pass so it
    survives any later failure (OOM, SIGKILL, divergence).
    """
    # --- versions -------------------------------------------------------- #
    try:
        import tabpfn as _tabpfn
        tabpfn_ver = getattr(_tabpfn, "__version__", "?")
    except Exception:                                              # pragma: no cover
        tabpfn_ver = "?"
    try:
        from importlib.metadata import version as _pkg_ver
        tabicl_ver = _pkg_ver("tabicl")
    except Exception:                                              # pragma: no cover
        tabicl_ver = "-"
    # --- hardware -------------------------------------------------------- #
    gpu_name, gpu_mem, cuda_cap = "cpu", None, None
    if device == "cuda" and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_mem = props.total_memory / 1e9
            cuda_cap = f"{props.major}.{props.minor}"
        except Exception:                                          # pragma: no cover
            gpu_name = "cuda"
    # --- resolved storage roots ----------------------------------------- #
    try:
        from src.utils.paths import get_roots
        roots = get_roots()
    except Exception:                                              # pragma: no cover
        roots = {}

    sep = "─" * 78
    lines = [
        sep,
        f"CreditPFN — TRAINING DEBUG BANNER (track={track})",
        sep,
        "[run] "
        f"base={Path(base_checkpoint).name} lr={learning_rate:g} lora={use_lora} "
        f"qf={query_fraction:.2f} accumulate={accumulate} pass_mode={pass_mode} "
        f"seed={seed}",
        "[hyperparams] "
        f"epochs={epochs} steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"n_estimators_finetune={n_estimators_finetune} amp={use_amp} "
        f"amp_dtype={amp_dtype} "
        f"max_rows_per_epoch={max_rows_per_epoch} max_cells_per_epoch={max_cells_per_epoch} "
        f"context_sampling={context_sampling} "
        f"weight_decay={weight_decay:g} l2sp_lambda={l2sp_lambda:g} "
        f"warmup_fraction={warmup_fraction:g}",
        "[corpus] "
        f"n_train_datasets={n_train_ds} n_test_datasets={n_test_ds}",
        "[env] "
        f"host={socket.gethostname()} "
        f"slurm_job={os.environ.get('SLURM_JOB_ID', '-')} "
        f"array={os.environ.get('SLURM_ARRAY_JOB_ID', '-')}/{os.environ.get('SLURM_ARRAY_TASK_ID', '-')} "
        f"cluster={os.environ.get('SLURM_CLUSTER_NAME', '-')} "
        f"partition={os.environ.get('SLURM_JOB_PARTITION', '-')} "
        f"node={os.environ.get('SLURMD_NODENAME', '-')} "
        f"account={os.environ.get('SLURM_JOB_ACCOUNT', '-')}",
        "[hardware] "
        f"device={device} gpu={gpu_name} vram={f'{gpu_mem:.1f}GB' if gpu_mem else 'n/a'} "
        f"cuda_capability={cuda_cap} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '-')} "
        f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '-')}",
        "[versions] "
        f"creditpfn_git={_git_sha()} "
        f"python={platform.python_version()} torch={torch.__version__} "
        f"cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} "
        f"numpy={np.__version__} tabpfn={tabpfn_ver} tabicl={tabicl_ver} "
        f"platform={platform.platform()}",
        "[paths] "
        f"data_root={roots.get('data_root', '?')} "
        f"output_root={roots.get('output_root', '?')} "
        f"staging_root={roots.get('staging_root', '?')}",
        f"[paths] base_checkpoint={base_checkpoint}",
        f"[paths] save_target={save_path}",
        sep,
    ]
    LOGGER.info("\n".join(str(x) for x in lines))


def _amp_step_was_skipped(scaler: "torch.amp.GradScaler") -> bool:
    """Return True if the most recent ``scaler.step(optimizer)`` was a no-op.

    ``GradScaler.step()`` silently skips the optimizer step when any
    gradient was inf/NaN (the dynamic-loss-scaling escape hatch). The
    public API doesn't expose a return value indicating skip / no-skip
    when AMP is disabled, so we look at the scaler's private
    per-optimizer ``_per_optimizer_states`` dict, which records
    ``"found_inf_per_device"`` as a tensor of 0 / 1 per device. Any 1 ⇒
    the step was skipped.

    Falls back to ``False`` when AMP is disabled (the scaler is a no-op
    that always lets the step through).
    """
    if not getattr(scaler, "_enabled", True):
        return False
    try:
        states = scaler._per_optimizer_states                          # type: ignore[attr-defined]
        for state in states.values():
            found = state.get("found_inf_per_device", {})
            for v in found.values():
                if float(v.item() if hasattr(v, "item") else v) != 0.0:
                    return True
    except Exception:                                                  # pragma: no cover
        # Best-effort probe — if the private API ever moves we degrade to the
        # old behaviour (assume the step happened). Warn ONCE so a silently
        # broken probe (which would re-introduce the LR-scheduler desync this
        # guards against) is visible in the log rather than invisible.
        if not getattr(_amp_step_was_skipped, "_warned", False):
            LOGGER.warning(
                "Could not read GradScaler private state to detect AMP-skipped "
                "steps; assuming steps were taken. If inf/NaN grad skips occur, "
                "the LR schedule may drift. (PyTorch internals may have changed.)"
            )
            _amp_step_was_skipped._warned = True                       # type: ignore[attr-defined]
        return False
    return False


# --------------------------------------------------------------------------- #
# Forward pass + loss
# --------------------------------------------------------------------------- #


def _forward(
    model: torch.nn.Module,
    batch: TabPFNBatch,
) -> tuple[torch.Tensor, torch.Tensor, float | None, float | None]:
    """Run one TabPFN forward pass.

    Calling convention matches TabPFN's canonical signature
    (``tfm-library/repositories/TabPFN .txt`` and the live 2.x package):

        forward(
            x: (train_rows + test_rows, batch, n_features),  # concatenated
            y: (train_rows, batch, 1),                       # train labels only
            *,
            only_return_standard_out=True,
            categorical_inds: list[list[int]] | None,        # one inner list per batch item
        ) -> (test_rows, batch, n_classes_or_bardist_buckets)

    The model deduces ``single_eval_pos = y.shape[0]`` and predicts the
    remaining rows of x.

    Returns ``(pred_logits, y_target, znorm_mean, znorm_std)``. The
    last two are non-None only for regression (where we z-normalise
    the context y, mirroring LennartPurucker's reference pipeline at
    `tfm-library/repositories/TabPFN V2 Finetuning.txt`).
    """
    train_x = batch.X_context       # (n_ctx, 1, F)
    train_y = batch.y_context.float()
    test_x = batch.X_query          # (n_qry, 1, F)
    raw_cat = batch.categorical_idx
    # TabPFN's assertion: categorical_inds[0] must itself be a list.
    # Our dataloader produces list[int] per chunk; wrap in a length-1
    # outer list to match the batch_size=1 we always run with.
    cat_inds: list[list[int]] | None = (
        [list(raw_cat)] if raw_cat else None
    )

    znorm_mean = znorm_std = None
    if batch.task_type == "regression":
        mean = train_y.mean(dim=0, keepdim=True)
        # ``unbiased=False`` divides by N (not N-1), so an N=1 chunk
        # yields std=0 rather than NaN. ``clamp_min`` then floors to
        # 1e-8 so the subsequent division is numerically safe.
        # ``clamp_min`` alone cannot rescue a NaN, so the unbiased=False
        # is the defensive bit here. Floor matches the ensemble path
        # (tabpfn_preprocessing.py) so the monitor and training z-norm a
        # near-constant target identically.
        std = train_y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
        train_y = (train_y - mean) / std
        y_target = (batch.y_query.float() - mean) / std
        znorm_mean = float(mean.detach().cpu().item())
        znorm_std = float(std.detach().cpu().item())
    else:
        y_target = batch.y_query

    # Concat context + query along the row/seq dimension; model sees one
    # tensor and derives the train/test split from len(y).
    combined_x = torch.cat([train_x, test_x], dim=0)

    pred_logits = model(
        combined_x,
        train_y,
        only_return_standard_out=True,
        categorical_inds=cat_inds,
    )
    return pred_logits, y_target, znorm_mean, znorm_std


def _forward_one_member(
    model: torch.nn.Module,
    *,
    X_ctx: torch.Tensor,           # (n_ctx, 1, F)  ALREADY PREPROCESSED
    y_ctx: torch.Tensor,           # (n_ctx, 1, 1)  class-permuted (classifier) / z-normed (regressor)
    X_qry: torch.Tensor,           # (n_qry, 1, F)  ALREADY PREPROCESSED
    cat_idx: list[int],
    outlier_removal_std: float | None,
) -> torch.Tensor:
    """One forward pass through the live training model for ONE preprocessed
    ensemble member.

    The input tensors here have already gone through TabPFN's CPU
    preprocessing pipeline (squashing scaler / quantile / SVD /
    fingerprint / class permutation). The remaining work before the model
    forward is the GPU soft-clip outlier removal (TabPFN's
    ``TorchSoftClipOutliersStep``, ``TabPFN .txt``) — we
    apply it here on the combined (context+query) tensor.

    Returns the raw model output logits, shape ``(n_qry, 1, L)`` where
    ``L`` is the per-row output dimensionality (= ``MAX_NUMBER_OF_CLASSES=10``
    for classifier, = bar-distribution buckets for regressor).
    """
    # Apply GPU soft-clip on numerical columns. Done on combined tensor
    # so the column-wise μ/σ uses both context AND query rows
    # (matching TabPFN's GPU pipeline which sees the concatenated tensor
    # inside `_call_model`).
    from src.train.tabpfn_preprocessing import apply_outlier_clip

    combined_x = torch.cat([X_ctx, X_qry], dim=0)
    if outlier_removal_std is not None:
        combined_x = apply_outlier_clip(
            combined_x, n_sigma=outlier_removal_std,
            categorical_idx=cat_idx,
        )

    cat_inds: list[list[int]] | None = (
        [list(cat_idx)] if cat_idx else None
    )

    pred_logits = model(
        combined_x,
        y_ctx.float(),
        only_return_standard_out=True,
        categorical_inds=cat_inds,
    )
    return pred_logits


def _classification_loss(
    pred_logits: torch.Tensor, targets: torch.Tensor,
    *, n_classes: int, criterion: torch.nn.Module,
) -> torch.Tensor:
    """CrossEntropyLoss on TabPFN's full ``MAX_NUMBER_OF_CLASSES`` (=10)
    logit columns.

    **CHANGE 2026-05-27** — previously we sliced the logits to the first
    K=n_classes columns before calling cross_entropy. That was a
    methodological bug: TabPFN's classifier head emits 10 logits
    (the pretraining max-classes; ``tfm-library/repositories/TabPFN .txt``),
    and the official `FinetunedTabPFNClassifier` computes CE over ALL
    10 columns so the softmax denominator regularises every column
    every step (gradient on z_k for k ≥ K is proportional to that
    column's softmax probability — i.e. a push-down signal).

    Slicing meant columns K..9 received zero gradient signal during
    training and were free to drift to arbitrary values. At inference
    (which softmaxes over all 10 columns then keeps the first K),
    those drifted columns stole probability mass from the K active
    columns — the calibration-collapse failure mode that produces
    high log-loss while ROC-AUC stays reasonable. See chat 2026-05-27
    and `_audit_2026-05-27_methodology.md` for the full derivation.

    The `n_classes` parameter is still required for downstream code
    (per-epoch eval, metric reporting) so we accept it but no longer
    slice with it. We do, however, sanity-check that targets are in
    `[0, n_classes)` — out-of-range targets would silently push the
    K..9 columns up (the wrong direction).
    """
    logits = pred_logits.float()
    logits = logits.reshape(-1, logits.shape[-1])
    target = targets.long().flatten()
    if __debug__:
        # Cheap assertion; bypassed under `python -O`. Catches a
        # mis-encoded label early instead of letting CE silently
        # propagate it.
        max_t = int(target.max().item()) if target.numel() else -1
        assert max_t < int(n_classes), (
            f"target label {max_t} >= n_classes={n_classes}; "
            "labels must be in [0, n_classes)"
        )
    return criterion(logits, target)


def _regression_loss(
    pred_logits: torch.Tensor, targets: torch.Tensor, *, criterion,
) -> torch.Tensor:
    """Bar-distribution NLL on the z-normalised targets."""
    # ``FullSupportBarDistribution`` calls ``torch.searchsorted`` on ``y``.
    # A strided slice is non-contiguous and makes PyTorch allocate/copy it on
    # every call (and emit a once-per-process performance warning).
    return criterion(
        logits=pred_logits,
        y=targets[:, :, 0].contiguous(),
    ).mean()


def _ensemble_step_loss(
    model: torch.nn.Module,
    batch,             # TabPFNEnsembleBatch — annotated as Any to avoid
                       # circular import-time pull of tabpfn_preprocessing.
    *,
    criterion,
) -> torch.Tensor:
    """One training-step loss for the N-estimator preprocessed batch.

    Mirrors ``FinetunedTabPFNClassifier._forward_with_loss``
    (``TabPFN .txt``):

      1. For each ensemble member i:
            * forward the model with member i's (X_ctx, y_ctx_permuted, X_qry)
            * if classifier and class_permutation is non-None, unscramble
              the logit columns by ``logits[..., perm]`` so they land back
              in canonical class order
      2. Stack the per-member logits into ``(Q, B, E, L)``.
      3. CE classifier loss: reshape to ``(B*E, L, Q)``, targets to
         ``(B*E, Q)`` via ``y_query.repeat(B*E, 1)``, single call to
         ``cross_entropy``. CE then averages over ``E*Q`` samples — exactly
         the official behaviour.
      4. Regression NLL: stack to ``(B*E, Q, L)``, ``criterion(logits, y)``
         then ``.mean()``.

    Returns a scalar tensor (the loss). Caller divides by accumulation
    before backward.
    """
    members = batch.members
    is_classification = batch.task_type == "classification"

    per_member_logits: list[torch.Tensor] = []
    for m in members:
        pred_logits = _forward_one_member(
            model,
            X_ctx=m.X_context,
            y_ctx=m.y_context,
            X_qry=m.X_query,
            cat_idx=m.categorical_idx,
            outlier_removal_std=m.outlier_removal_std,
        )                                      # (n_qry, 1, L)

        # Unscramble class permutation if any (classifier only).
        if is_classification and m.class_permutation is not None:
            # `class_permutation` is a positional permutation array, e.g.
            # [1, 0] for binary-flipped. The official inference path at
            # `TabPFN .txt` does `logits[..., perm]` to reorder
            # the output columns back into canonical class order. We do
            # the same here so the CE loss sees logits already aligned
            # with `y_query` (which stays in canonical order).
            perm = m.class_permutation
            L = pred_logits.shape[-1]
            if len(perm) < L:
                # Pad permutation to full L=10 by leaving extra columns
                # in place — they receive gradient via the softmax
                # denominator but don't get swapped.
                use_perm = np.arange(L)
                use_perm[: len(perm)] = perm
            else:
                use_perm = np.asarray(perm[:L])
            use_perm_t = torch.as_tensor(
                use_perm, device=pred_logits.device, dtype=torch.long,
            )
            pred_logits = pred_logits.index_select(-1, use_perm_t)

        per_member_logits.append(pred_logits)

    # Stack along a new E dim: (n_qry, 1, E, L)
    logits_QBEL = torch.stack(per_member_logits, dim=2)
    Q, B, E, L = logits_QBEL.shape
    assert B == 1, f"expected batch_size=1, got B={B}"

    if is_classification:
        # Reshape to (B*E, L, Q) — PyTorch CE wants class dim at axis 1.
        # `permute(1, 2, 3, 0)` → (B, E, L, Q); reshape to (B*E, L, Q).
        logits_BLQ = logits_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)
        targets_BQ = batch.y_query.reshape(B, Q).repeat(B * E, 1)
        return _classification_loss_BE_LQ(
            logits_BLQ, targets_BQ,
            n_classes=int(batch.n_classes or 2), criterion=criterion,
        )
    # Regression: stack to (B*E, Q, L) for the bar-distribution criterion.
    logits_BQL = logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L)
    targets_BQ_reg = batch.y_query.reshape(B, Q).repeat(B * E, 1).float()
    # criterion's `__call__(logits=..., y=...)` expects logits shape
    # (Q, batch, L) for `FullSupportBarDistribution.__call__`; pass with
    # the batch dim as B*E and Q on axis 0.
    return criterion(
        logits=logits_BQL.permute(1, 0, 2),     # (Q, B*E, L)
        y=targets_BQ_reg.transpose(0, 1).contiguous(),  # (Q, B*E)
    ).mean()


def _classification_loss_BE_LQ(
    logits_BLQ: torch.Tensor, targets_BQ: torch.Tensor,
    *, n_classes: int, criterion: torch.nn.Module,
) -> torch.Tensor:
    """CE on the (B*E, L, Q) / (B*E, Q) shape — matches official
    ``F.cross_entropy(input, target)`` where the class dim is at axis 1
    of `input`. See `_compute_classification_loss` at
    ``TabPFN .txt``.
    """
    if __debug__:
        max_t = int(targets_BQ.max().item()) if targets_BQ.numel() else -1
        assert max_t < int(n_classes), (
            f"target label {max_t} >= n_classes={n_classes}"
        )
    return criterion(logits_BLQ.float(), targets_BQ.long())


def _n_classes(batch: TabPFNBatch) -> int:
    """Max class index in this chunk, +1 → number of classes seen."""
    K = int(batch.y_context.flatten().max().item()) + 1
    K = max(K, int(batch.y_query.flatten().max().item()) + 1)
    return max(K, 2)            # binary at minimum


def _query_missing_context_class(batch) -> bool:
    """Return True iff the query split contains a class index that the
    context split does NOT contain.

    Mirrors the official guard at ``tfm-library/repositories/TabPFN .txt``
    (``FinetunedTabPFNClassifier._should_skip_batch``). Without it, a
    stratified PD subsample that happens to put both positives in the
    query split leaves the context with only one class — the CE loss
    is then ill-defined on the positive query row(s) because the
    in-context examples never demonstrate what "class 1" looks like.

    Works for both the legacy :class:`TabPFNBatch` (single y_context /
    y_query tensors) and the new :class:`TabPFNEnsembleBatch` (list of
    per-member y_context tensors). Regression batches always return
    False — the check is classifier-only.
    """
    if getattr(batch, "task_type", "") != "classification":
        return False

    # Ensemble batch: y_context is per-member (each member sees a
    # potentially class-permuted view), but the y_query is shared in
    # canonical class order. Concatenate across members for the union.
    if hasattr(batch, "members"):
        ctx_uniques = []
        for m in batch.members:
            ctx_uniques.append(torch.unique(m.y_context.reshape(-1)))
        ctx_unique = torch.unique(torch.cat(ctx_uniques))
        qry_unique = torch.unique(batch.y_query.reshape(-1))
    elif hasattr(batch, "y_train"):
        # TabICLTrainBatch: per-member class shuffles are BIJECTIVE remaps
        # applied consistently to y_train and y_query, so a set-membership
        # mismatch in member 0's space ⇔ a mismatch in every member's space.
        # (Mirrors tabicl's own `_task_skip_batch`.)
        ctx_unique = torch.unique(batch.y_train[0].reshape(-1).long())
        qry_unique = torch.unique(batch.y_query[0].reshape(-1).long())
    else:
        ctx_unique = torch.unique(batch.y_context.reshape(-1))
        qry_unique = torch.unique(batch.y_query.reshape(-1))

    # Check: every class in query must also be in context.
    in_ctx = torch.isin(qry_unique, ctx_unique)
    return not bool(in_ctx.all().item())


# --------------------------------------------------------------------------- #
# Ensemble per-epoch eval (n_estimators=32 via TabPFNClassifier/Regressor)
# --------------------------------------------------------------------------- #


def _save_eval_snapshot(
    model: torch.nn.Module,
    architecture_config,
    snapshot_path: Path,
    *,
    criterion: torch.nn.Module | None = None,
    inference_config=None,
) -> None:
    """Persist the live model's state_dict to a Prior-Labs-format .ckpt
    so ``TabPFNClassifier(model_path=...)`` can load it.

    **Non-destructive.** The live model and (if LoRA-wrapped) its
    PEFT adapter are left exactly as they were on entry — we operate
    on a ``copy.deepcopy`` of LoRA-wrapped models before calling
    ``merge_and_unload``. This is the whole reason this function exists
    instead of just calling ``save_finetuned``: the production save path
    mutates the live model, which would terminate training.

    **Format — matched verbatim to ``save_tabpfn_model`` at
    ``tfm-library/repositories/TabPFN .txt``.** Critical: we MUST write
    the 4 keys ``{state_dict, config, architecture_name, inference_config}``.
    Skipping ``architecture_name`` and ``inference_config`` makes
    ``load_model`` (TabPFN .txt) fall back to V2 architecture
    inference, producing the "Missing key(s) in state_dict" error on V3
    weights — observed in every snapshot-load attempt in the
    2026-05-27 PD/LGD logs.
    """
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull TabPFN's helpers lazily — they require tabpfn import.
    try:
        from tabpfn.model_loading import _resolve_architecture_name
    except ImportError:                                                # pragma: no cover
        _resolve_architecture_name = None

    from dataclasses import asdict, is_dataclass

    # The architecture_config is a dataclass instance (TabPFNV3Config /
    # TabPFNV2p6Config / …). Use `asdict` per the canonical save path
    # (TabPFN .txt). Fall back to __dict__ for non-dataclass cfgs.
    if is_dataclass(architecture_config):
        config_payload = asdict(architecture_config)
    elif hasattr(architecture_config, "__dict__"):
        config_payload = dict(architecture_config.__dict__)
    else:
        config_payload = architecture_config

    # LoRA case: clone first, merge into the clone, throw the clone away.
    # Costs one transient deep-copy of the model (~213 MB for v3) but
    # keeps the training trajectory exactly as it was.
    is_peft = (
        hasattr(model, "merge_and_unload")
        and callable(getattr(model, "merge_and_unload", None))
    )
    if is_peft:
        import copy as _copy
        cloned = _copy.deepcopy(model)
        merged = cloned.merge_and_unload()                              # type: ignore[attr-defined]
        state_dict = merged.state_dict()
        del cloned, merged
    else:
        state_dict = (
            model.module.state_dict()
            if hasattr(model, "module") else model.state_dict()
        )

    # Regressor: merge bar-distribution criterion params into the state_dict
    # under the `criterion.` prefix the TabPFN loader expects.
    if criterion is not None and hasattr(criterion, "state_dict"):
        crit_state = criterion.state_dict()
        if crit_state:
            for k, v in crit_state.items():
                state_dict[f"criterion.{k}"] = v

    # Architecture name — tells the loader which architecture class to
    # instantiate. Without this key, load_model defaults to
    # ``ARCHITECTURES["base"]`` (V2). Critical for V3 / V2.6.
    if _resolve_architecture_name is not None:
        architecture_name = _resolve_architecture_name(architecture_config)
    else:
        # Conservative fallback if private helper moves: try to identify
        # by class name (TabPFNV3Config → tabpfn_v3, etc.).
        cls_name = type(architecture_config).__name__
        if "V3" in cls_name:
            architecture_name = "tabpfn_v3"
        elif "V2p6" in cls_name or "V2_6" in cls_name:
            architecture_name = "tabpfn_v2_6"
        elif "V2p5" in cls_name or "V2_5" in cls_name:
            architecture_name = "tabpfn_v2_5"
        else:
            architecture_name = "base"

    checkpoint: dict = {
        "state_dict": state_dict,
        "config": config_payload,
        "architecture_name": architecture_name,
    }

    # Inference config — required for V2.6 and V3 (these checkpoints
    # always embed their own; the loader at TabPFN .txt
    # reads this key directly for self-loss models).
    if inference_config is not None:
        if is_dataclass(inference_config):
            checkpoint["inference_config"] = asdict(inference_config)
        else:
            checkpoint["inference_config"] = inference_config

    torch.save(checkpoint, str(snapshot_path))


def evaluate_ensemble_on_split(
    ckpt_path: Path | str,
    refs: list[DatasetRef],
    *,
    n_estimators: int,
    n_subsample: int,
    query_fraction: float,
    seed: int,
    device: str,
    task_type: str,
    metric_names: tuple[str, ...],
    family: str = "tabpfn",
) -> dict[str, float]:
    """Evaluate one TabPFN or TabICLv2 checkpoint via its sklearn API with
    full ensemble inference (``n_estimators`` forward passes per
    fit/predict, averaged with the package's standard ensembling strategy).

    This is what ``scripts/eval_pipeline.py`` does at the end of
    training, just on a smaller per-epoch sample. Reusing the same code
    path guarantees per-epoch and final-eval numbers are directly
    comparable (same model, same context/query geometry, same
    n_estimators).

    Returns a ``{metric_name: mean_over_datasets}`` dict. NaN-skips
    datasets where a metric is undefined (single-class query,
    ill-conditioned predictions, etc.) so a single degenerate dataset
    doesn't contaminate the mean.
    """
    # Local imports — the function is called once per epoch from the
    # training loop, so we can afford the lazy-load overhead in exchange
    # for keeping `loop.py`'s module-level import cost low.
    from src.eval.benchmark import _classification_metrics, _regression_metrics
    from src.train.dataloader import _load_processed_csv
    from src.train.dataloader import _stratified_subsample_indices  # type: ignore[attr-defined]
    from src.model.tabpfn_models import _make_tabpfn
    from src.train.metrics import mean_ignore_nan

    if not refs or n_subsample <= 0:
        return {m: float("nan") for m in metric_names}

    per_dataset: dict[str, list[float]] = {m: [] for m in metric_names}

    for i, ref in enumerate(refs):
        loaded = _load_processed_csv(ref)
        rng = np.random.default_rng(seed + i)
        n = len(loaded.X)

        if 0 < n_subsample < n:
            if task_type == "classification":
                keep = _stratified_subsample_indices(loaded.y, n_subsample, rng)
            else:
                keep = rng.choice(n, size=n_subsample, replace=False)
            X_sub = loaded.X.iloc[keep].reset_index(drop=True)
            y_sub = loaded.y[keep]
        else:
            X_sub = loaded.X.reset_index(drop=True)
            y_sub = loaded.y

        n_total = len(X_sub)
        n_query = max(1, int(round(n_total * float(query_fraction))))
        n_query = min(n_query, n_total - 1)
        n_ctx = n_total - n_query

        X_ctx = X_sub.iloc[:n_ctx].values
        y_ctx = y_sub[:n_ctx]
        X_qry = X_sub.iloc[n_ctx:].values
        y_qry = y_sub[n_ctx:]

        # Categorical feature INDICES (positional) into the dataframe.
        cat_idx = [
            X_sub.columns.get_loc(c)
            for c in loaded.cat_columns if c in X_sub.columns
        ]

        try:
            if family == "tabicl":
                # TabICLv2 sklearn wrappers do their own preprocessing on raw
                # features (no categorical-indices parameter exists).
                # allow_auto_download=False: VSC compute nodes have no
                # outbound network — a missing checkpoint must fail loudly,
                # not attempt an HF download.
                from src.train.tabicl_compat import import_tabicl_sklearn
                ticl_clf, ticl_reg = import_tabicl_sklearn()
                wrapper_cls = ticl_clf if task_type == "classification" else ticl_reg
                tabpfn = wrapper_cls(
                    model_path=str(ckpt_path),
                    device=device,
                    n_estimators=int(n_estimators),
                    allow_auto_download=False,
                    random_state=int(seed),
                )
            else:
                tabpfn = _make_tabpfn(
                    task_type, ckpt_path,
                    device=device, n_estimators=int(n_estimators),
                    categorical_features_indices=(cat_idx or None),
                )
            tabpfn.fit(X_ctx, y_ctx)
            if task_type == "classification":
                proba = tabpfn.predict_proba(X_qry)
                # Note: passing proba twice (test + "val") means the
                # F1-tuned classification metrics (f1/accuracy/...) use
                # an in-sample threshold here, biased toward optimism.
                # That's fine for a monitor — the unbiased threshold
                # comes from the full eval pipeline. We DO get unbiased
                # threshold-free metrics: roc_auc, log_loss, pr_auc,
                # brier_score.
                metrics = _classification_metrics(
                    proba_test=proba, y_test=y_qry,
                    proba_val=proba,  y_val=y_qry,
                    n_classes_seen=int(len(np.unique(y_ctx))),
                )
            else:
                preds = tabpfn.predict(X_qry)
                metrics = _regression_metrics(
                    pred_test=preds, y_test=y_qry, neg_nll=None,
                )
        except Exception as exc:                                       # noqa: BLE001
            LOGGER.warning(
                "ensemble eval failed for dataset=%s (n_est=%d): %s — emitting NaN",
                ref.dataset_id, n_estimators, exc,
            )
            metrics = {m: float("nan") for m in metric_names}

        for m in metric_names:
            per_dataset[m].append(float(metrics.get(m, float("nan"))))

    return {m: mean_ignore_nan(per_dataset[m]) for m in metric_names}


# --------------------------------------------------------------------------- #
# Test-set evaluation (called ONCE at end of training)
# --------------------------------------------------------------------------- #


def evaluate_on_split(
    model: torch.nn.Module,
    refs: list[DatasetRef],
    *,
    criterion,
    device: str,
    metric_name: str | tuple[str, ...] | list[str],
    n_inference_subsample_samples: int,
    seed: int = 0,
    query_fraction: float = 0.20,
) -> float | dict[str, float]:
    """Mean primary metric over a list of datasets (end-of-epoch monitor).

    Used by the training loop for per-epoch monitoring only — the proper
    eval is a separate pipeline (``scripts/eval_pipeline.py``).

    ``metric_name`` may be a single string (back-compat — returns a
    float) or a sequence of strings (returns a ``dict[str, float]``
    keyed by metric name). The multi-metric path shares the model's
    forward pass across all listed metrics, so adding R² alongside RMSE
    costs only the cheap post-processing.
    """
    multi = not isinstance(metric_name, str)
    metric_names: tuple[str, ...] = (
        tuple(metric_name) if multi else (metric_name,)  # type: ignore[arg-type]
    )

    if not refs or n_inference_subsample_samples <= 0:
        nan_result = {m: float("nan") for m in metric_names}
        return nan_result if multi else float("nan")

    was_training = model.training
    model.eval()
    is_classification = refs[0].task_type == "classification"
    per_chunk: dict[str, list[float]] = {m: [] for m in metric_names}

    try:
        with torch.no_grad():
            for i, ref in enumerate(refs):
                batch = prepare_eval_chunk(
                    ref,
                    n_inference_subsample_samples=n_inference_subsample_samples,
                    seed=seed + i,
                    query_fraction=query_fraction,
                ).to(device)
                pred_logits, y_target, zmean, zstd = _forward(model, batch)
                for m in metric_names:
                    if is_classification:
                        K = _n_classes(batch)
                        logits = pred_logits[:, :, :K]
                        value = classification_metric(
                            logits=logits, targets=y_target,
                            metric=m, n_classes=K,
                        )
                    else:
                        value = regression_metric(
                            logits=pred_logits, targets=y_target,
                            criterion=criterion, metric=m,
                            znorm_mean=zmean, znorm_std=zstd,
                        )
                    per_chunk[m].append(value)
    finally:
        # Restore prior train/eval state so the outer loop's optimizer
        # step continues against a training-mode model (matters for
        # dropout / batchnorm if the architecture grows them later).
        if was_training:
            model.train()

    means = {m: mean_ignore_nan(per_chunk[m]) for m in metric_names}
    return means if multi else means[metric_names[0]]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def train_one_config(
    cfg,
    *,
    track: str | None = None,
    base_checkpoint: str | None = None,
    learning_rate: float | None = None,
    use_lora: bool | None = None,
    query_fraction: float | None = None,
    accumulate_grad_batches: int | None = None,
    pass_mode: str | None = None,
    #: Swept from run-9. Overrides `cfg.finetuning.l2sp_lambda` for this trial. Same
    #: contract as `min_train_rows`: the per-trial value wins, `None` falls back to the
    #: config, so a run that does not sweep it behaves exactly as before.
    l2sp_lambda: float | None = None,
    #: Swept in run-8. Overrides `cfg.corpus.min_train_rows` for this trial, so the
    #: whole grid can share one config while each trial trains on its own corpus.
    min_train_rows: int | None = None,
    save_path: Path | str | None = None,
    on_epoch_end: Callable[[EpochRecord], None] | None = None,
) -> TrainingResult:
    """Run continued pretraining for one fixed (config, HP-tuple).

    The four arguments ``track``, ``base_checkpoint``, ``learning_rate``,
    ``use_lora`` are the ONLY things the script expects to vary per
    run — see ``cfg.tunable`` in ``config/train.yaml``. Each defaults
    to either the explicit ``cfg.<...>`` field if set, or the first
    value of the corresponding tunable list.

    Each parent dataset contributes EXACTLY ONE training step per epoch
    (no chunking — see 2026-05-20 refactor in `src/train/corpus.py`).

    Parameters
    ----------
    cfg
        OmegaConf config (typically ``OmegaConf.load("config/train.yaml")``).
    track
        Override ``cfg.track``. ``None`` → use the value from ``cfg``.
    base_checkpoint
        Override the base weights path. ``None`` → use
        ``cfg.tunable.<classifier|regressor>_base_paths[0]``.
    learning_rate
        Override AdamW LR. ``None`` → ``cfg.tunable.learning_rates[0]``.
    use_lora
        Override the LoRA flag. ``None`` → ``bool(cfg.tunable.use_lora[0])``
        if that list exists, else ``False``. When True the base weights
        are frozen and only the LoRA A/B matrices receive gradients;
        the adapter is merged back into the base weights at save time.
    save_path
        Where to write the final-epoch checkpoint. ``None`` →
        ``cfg.checkpoint.trained_dir / descriptive_name(...)``.
    on_epoch_end
        Optional hook called after each epoch with the
        :class:`EpochRecord` (live progress logging in a script).

    Returns
    -------
    TrainingResult
        Includes the final checkpoint path, per-epoch train loss
        history. Scoring on the held-out test set is the eval
        pipeline's job, not this loop's.
    """
    # ---- resolve every tunable parameter ---------------------------------- #
    track = track or cfg.track
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    if base_checkpoint is None:
        bases = (cfg.tunable.classifier_base_paths if track == "pd"
                 else cfg.tunable.regressor_base_paths)
        base_checkpoint = str(bases[0])
    base_checkpoint_config = str(base_checkpoint)
    base_checkpoint_path = resolve_staging_path(base_checkpoint_config)
    if learning_rate is None:
        learning_rate = float(cfg.tunable.learning_rates[0])
    if use_lora is None:
        # cfg.tunable.use_lora is a list (e.g. [false, true]) the script
        # iterates over. When this function is invoked without an explicit
        # `use_lora` argument, default to the head of that list — same
        # convention as the other tunable axes.
        tunable_lora = getattr(cfg.tunable, "use_lora", None)
        if tunable_lora is None:
            use_lora = False
        elif isinstance(tunable_lora, bool):
            use_lora = bool(tunable_lora)
        else:
            use_lora = bool(list(tunable_lora)[0])
    if query_fraction is None:
        # cfg.tunable.query_fractions is a list (e.g. [0.20, 0.30, 0.40])
        # the script iterates over; default to the head of that list.
        tunable_qf = getattr(cfg.tunable, "query_fractions", None)
        if tunable_qf is None:
            query_fraction = 0.20  # TabPFN documented default
        elif isinstance(tunable_qf, (int, float)):
            query_fraction = float(tunable_qf)
        else:
            query_fraction = float(list(tunable_qf)[0])

    # Inject the resolved choices back into cfg so downstream helpers
    # (corpus split, optimizer factory) read them via the usual path.
    cfg.optimizer.lr = float(learning_rate)

    _seed_everything(int(cfg.seed))
    device = _resolve_device(cfg)
    LOGGER.info(
        "Training track=%s on device=%s | base=%s | lr=%g | lora=%s | qf=%.2f | seed=%d",
        track, device, Path(base_checkpoint_config).name, learning_rate,
        use_lora, query_fraction, int(cfg.seed),
    )
    LOGGER.info(
        "Resolved base checkpoint path: %s -> %s",
        base_checkpoint_config, base_checkpoint_path,
    )

    # ---- 1) corpus split --------------------------------------------------- #
    # `min_train_rows` is a SWEPT axis, so the per-trial value wins over the config
    # list. Resolved by writing it into a copy of cfg.corpus rather than threading a
    # parameter through split_from_cfg, so every other caller keeps its behaviour.
    if min_train_rows is None:
        raw = getattr(cfg.corpus, "min_train_rows", 0) if hasattr(cfg, "corpus") else 0
        # A config list means "sweep this"; a bare trial resolves to its first value.
        min_train_rows = int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw or 0)
    split: CorpusSplit = split_corpus(
        track=track,
        train_fraction=float(cfg.corpus.train_fraction),
        test_fraction=float(cfg.corpus.test_fraction),
        train_dataset_ids=resolve_ids_for_track(
            cfg.corpus.get("train_dataset_ids", None), track),
        test_dataset_ids=resolve_ids_for_track(
            cfg.corpus.get("test_dataset_ids", None), track),
        seed=int(cfg.seed),
        min_train_rows=int(min_train_rows),
    )
    LOGGER.info("Corpus split: %s", split.summary)
    train_ids = sorted({c.dataset_id for c in split.train})
    test_ids  = sorted({c.dataset_id for c in split.test})
    # WITH ROW COUNTS, and the totals. The corpus is an experimental variable now
    # (min_train_rows) and will keep changing as datasets are added, so a log that
    # records only the ids cannot be read back six months later: "17 datasets" in
    # 08-2026 and "17 datasets" in 2027 will not be the same 17.
    LOGGER.info(
        "Training datasets (n=%d, %s rows total, min_train_rows=%d): %s",
        len(train_ids), f"{sum(c.n_rows for c in split.train):,}", int(min_train_rows),
        ", ".join(f"{c.dataset_id}({c.n_rows:,})" for c in split.train) or "<none>",
    )
    LOGGER.info(
        "Held-out test datasets (n=%d, %s rows total): %s",
        len(test_ids), f"{sum(c.n_rows for c in split.test):,}",
        ", ".join(f"{c.dataset_id}({c.n_rows:,})" for c in split.test) or "<none>",
    )
    if not split.train:
        raise RuntimeError(
            "Corpus split contains no training chunks. Run the data "
            "pipeline (`python scripts/data_pipeline.py`) first."
        )

    # ---- 2) base model + criterion ---------------------------------------- #
    # Two model families (2026-08-04). The grid's `use_lora` axis means:
    #   tabpfn → PEFT LoRA adapters (base frozen, adapters train);
    #   tabicl → FREEZE-BACKBONE / train-ICL-head-only (upstream stage-3;
    #            full SFT collapsed TabICLv2 in two independent reports —
    #            see src/train/tabicl_compat.py). Checkpoint names tag the
    #            difference honestly (`_lora` vs `_iclhead`).
    family = model_family(base_checkpoint_config)
    if family == "tabicl":
        from src.train.tabicl_model import load_tabicl_for_training
        model, tabicl_model_config = load_tabicl_for_training(
            base_checkpoint_path, track=track, device=device,
            freeze_backbone=bool(use_lora),
        )
        criterion = None                    # CE / pinball are functional losses
        architecture_config = None
        inference_config = None             # dataloader uses the tabicl path
    else:
        tabicl_model_config = None
        lora_cfg_dict = (
            dict(cfg.lora) if (use_lora and hasattr(cfg, "lora")) else None
        )
        model, criterion, architecture_config, inference_config = (
            load_tabpfn_for_training(
                base_checkpoint_path, track=track, device=device,
                lora_config=lora_cfg_dict,
            )
        )

    # L2-SP anchor (Li et al. 2018): snapshot the pretrained weights w0 right
    # after load, so the loss can penalise drift away from them (see
    # _l2sp_penalty). λ=0.003 matches Real-TabPFN's corpus-CPT regularizer
    # exactly (Garg et al. 2025 §method — verified against the paper
    # 2026-07-10). Configurable via cfg.optimizer.l2sp_lambda (0.0 = off).
    # Applicability per family/mode:
    #   tabpfn + LoRA        → inert (base frozen, adapters start at 0): skip.
    #   tabpfn + full-FT     → anchor all trainable weights.
    #   tabicl (both modes)  → anchor the TRAINABLE weights (full-FT: all;
    #                          icl-head mode: the head only — it starts at the
    #                          pretrained weights and can drift, unlike LoRA).
    # SWEPT AXIS from run-9: the per-trial value wins over the config, exactly as
    # `min_train_rows` does. Shadowing the parameter name is deliberate — everything
    # below this line already reads `l2sp_lambda`.
    if l2sp_lambda is None:
        l2sp_lambda = float(getattr(cfg.optimizer, "l2sp_lambda", 0.0) or 0.0)
    else:
        l2sp_lambda = float(l2sp_lambda)
    l2sp_applicable = (family == "tabicl") or (not use_lora)
    l2sp_anchor: dict[str, torch.Tensor] | None = None
    l2sp_w0_norm: float = 0.0
    l2sp_stage_names: dict[str, list[str]] = {}
    l2sp_stage_w0: dict[str, float] = {}
    if l2sp_lambda > 0.0 and l2sp_applicable:
        l2sp_anchor = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        # ‖w0‖ over the anchored tensors, captured once. It turns the L2-SP
        # penalty into a HUMAN-READABLE drift figure: the penalty is
        #     l2sp = 0.5 * lambda * ‖w - w0‖²
        # so   ‖w - w0‖ = sqrt(2 * l2sp / lambda)
        # and the relative drift is ‖w - w0‖ / ‖w0‖. Logged per epoch below.
        # Added 08-08-2026 — the single most useful diagnostic we have for
        # "did continued pretraining actually change the model?", which took
        # manual arithmetic on the log to answer for runs 5 and 6.
        l2sp_w0_norm = float(torch.sqrt(sum(
            (t.double() ** 2).sum() for t in l2sp_anchor.values()
        )).item())
        # Group anchored tensors by TOP-LEVEL module so per-stage drift can be
        # reported. Grouping on the first dotted component is deliberately
        # architecture-agnostic: TabICLv2 yields col_embedder / row_interactor /
        # icl_predictor, TabPFN yields whatever its own top-level modules are,
        # and neither is hardcoded here.
        l2sp_stage_names: dict[str, list[str]] = {}
        for n in l2sp_anchor:
            l2sp_stage_names.setdefault(n.split(".", 1)[0], []).append(n)
        l2sp_stage_w0: dict[str, float] = {
            stage: float(torch.sqrt(sum(
                (l2sp_anchor[n].double() ** 2).sum() for n in names
            )).item())
            for stage, names in l2sp_stage_names.items()
        }
        LOGGER.info(
            "L2-SP enabled: lambda=%.2e, anchoring %d trainable tensors to w0 "
            "(‖w0‖=%.3f). Per-epoch lines report drift=‖w-w0‖/‖w0‖.",
            l2sp_lambda, len(l2sp_anchor), l2sp_w0_norm,
        )
    elif l2sp_lambda > 0.0:
        LOGGER.info(
            "L2-SP lambda=%.2e configured but use_lora=True (TabPFN) — base "
            "weights are frozen, so L2-SP is inert; skipping it.", l2sp_lambda,
        )

    # ---- 3) DataLoader + optimiser / scheduler ---------------------------- #
    # The per-step subsample size is `finetuning.max_rows_per_epoch` in
    # `config/data.yaml`. As of the 2026-05-20 PD run it became clear
    # that v2.6 OOMs at the v3-safe 10_000 rows (alternating row ×
    # feature attention × 24 layers is much more memory-hungry than
    # v3's three-stage design). So `max_rows_per_epoch` is now a
    # per-version map; we look it up by the base checkpoint's leading
    # `v<MAJOR>` segment.
    from omegaconf import OmegaConf
    _data_cfg = OmegaConf.load("config/data.yaml")
    max_rows_per_epoch = _resolve_max_rows_per_epoch(
        base_checkpoint_config, _data_cfg.finetuning.max_rows_per_epoch,
    )
    # Optional per-architecture cell budget (rows × features). Off (null)
    # for both bases by default → pure row cap. Appropriate for v3 (whose
    # capacity is a cell-budget frontier), NOT v2.6 (quadratic in rows).
    # Reuses the same per-version mapping resolver; a null value or an
    # absent key resolves to None (no cell budget for that base).
    max_cells_per_epoch = None
    # Context-construction strategy for the per-step subsample. Shared by
    # BOTH families so a cross-family comparison never confounds this axis —
    # the axis Tanna et al. 2026 measure as worth more AUC than model choice.
    context_sampling = str(
        _data_cfg.finetuning.get("context_sampling", "stratified"))
    _max_cells_cfg = _data_cfg.finetuning.get("max_cells_per_epoch", None)
    if _max_cells_cfg is not None:
        try:
            _mc = _resolve_max_rows_per_epoch(base_checkpoint_config, _max_cells_cfg)
            max_cells_per_epoch = int(_mc) if _mc else None
        except Exception:                                          # null / no key
            max_cells_per_epoch = None
    # `query_fraction` is now a per-trial argument coming from the
    # sweep — defaulted above to the head of cfg.tunable.query_fractions
    # if the caller didn't pass it. The old single-value
    # `data_cfg.finetuning.query_fraction` is preserved only as a
    # back-compat fallback when no per-trial value was resolved.
    if query_fraction is None:
        query_fraction = float(_data_cfg.finetuning.query_fraction)

    # Resolve `n_estimators_finetune` (number of preprocessed ensemble
    # members per training step). Accepts EITHER a scalar int (applied to
    # both tracks) OR a per-track mapping {pd: 2, lgd: 8, default: 2} so the
    # regressor can use more members (lower per-step gradient noise), matching
    # the official FinetunedTabPFNClassifier (=2) / FinetunedTabPFNRegressor
    # (=8) defaults.
    _raw_ne = getattr(cfg.train, "n_estimators_finetune", 2)
    if isinstance(_raw_ne, (int, float)):
        n_estimators_finetune = int(_raw_ne)
    else:  # per-track mapping (OmegaConf DictConfig or plain dict)
        _ne = getattr(_raw_ne, track, None)
        if _ne is None and hasattr(_raw_ne, "get"):
            _ne = _raw_ne.get(track, None)
        if _ne is None:
            _ne = getattr(_raw_ne, "default", None)
            if _ne is None and hasattr(_raw_ne, "get"):
                _ne = _raw_ne.get("default", 2)
        n_estimators_finetune = int(_ne if _ne is not None else 2)
    n_estimators_finetune = max(1, n_estimators_finetune)

    # TabICLv2 family: upstream's finetuning uses n_estimators=2 for BOTH the
    # classifier and the regressor (tabicl._finetune defaults) — override the
    # TabPFN-derived per-track mapping (pd=2/lgd=8) unless the config sets an
    # explicit tabicl value.
    if family == "tabicl":
        n_estimators_finetune = max(
            1, int(getattr(cfg.train, "n_estimators_finetune_tabicl", 2)),
        )

    # ---- member-aware row-cap scaling (GPU-memory safety) ---------------- #
    # A training step forwards ALL `n_estimators_finetune` preprocessed
    # ensemble members and holds every member's activation graph
    # simultaneously for a SINGLE backward (see `_ensemble_step_loss`), so
    # per-step GPU memory ≈ n_estimators × rows × per-member-per-row cost.
    # Measured on the B200 (scripts/probe_row_cap.py, 2026-07-08, 64 feats,
    # fwd+bwd, single member): v3 ≈ 2.5 GB and v2.6 ≈ 5.7 GB per 1 000 rows,
    # ~linear (intercept ≈ 0). The `max_rows_per_epoch` config values are
    # calibrated for the 2-member reference (PD); we scale INVERSELY with the
    # actual member count so an 8-member LGD step holds ~the same memory as a
    # 2-member PD step instead of 4× more (which OOMs the 192 GiB card).
    # (TabPFN-measured constants — B200 probe 2026-07-08. NOT applied to the
    # tabicl family: its memory slope is unmeasured and its member count is
    # fixed at 2 anyway; the `tabicl` row-cap key in config/data.yaml is the
    # sizing knob there.)
    _REFERENCE_MEMBERS = 2
    if family == "tabpfn" and n_estimators_finetune > _REFERENCE_MEMBERS:
        _scaled = max(1000, max_rows_per_epoch * _REFERENCE_MEMBERS // n_estimators_finetune)
        if _scaled < max_rows_per_epoch:
            LOGGER.info(
                "Row cap scaled %d → %d rows/step for %d ensemble members "
                "(per-step GPU memory ≈ n_estimators × rows; see probe_row_cap).",
                max_rows_per_epoch, _scaled, n_estimators_finetune,
            )
            max_rows_per_epoch = _scaled

    # Resolve the per-epoch step plan. None → head of the sweep list
    # (default "one_sample" = one step per dataset per epoch).
    if pass_mode is None:
        raw_pm = getattr(cfg.tunable, "epoch_pass_modes", None)
        if raw_pm is None:
            pass_mode = "one_sample"
        elif isinstance(raw_pm, str):
            pass_mode = raw_pm
        else:
            pass_mode = str(list(raw_pm)[0])
    train_ds = ProcessedDatasetLoader(
        split.train,
        max_rows_per_epoch=max_rows_per_epoch,
        query_fraction=query_fraction,
        seed=int(cfg.seed),
        inference_config=inference_config,
        n_estimators_finetune=n_estimators_finetune,
        pass_mode=pass_mode,
        max_cells_per_epoch=max_cells_per_epoch,
        model_family=family,
        context_sampling=context_sampling,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=int(cfg.train.dataloader_workers),
        collate_fn=identity_collate,
        pin_memory=device == "cuda",
    )

    epochs = int(cfg.train.epochs)
    # `accumulate_grad_batches` is a tunable as of 2026-05-27. Caller
    # may pass the trial's value via the kwarg; falls back to the
    # legacy `cfg.train.accumulate_grad_batches` (or the first value of
    # `cfg.tunable.accumulate_grad_batches`) for back-compat.
    if accumulate_grad_batches is not None:
        accumulate = max(1, int(accumulate_grad_batches))
    else:
        legacy = getattr(cfg.train, "accumulate_grad_batches", None)
        if legacy is None:
            raw_acc = getattr(cfg.tunable, "accumulate_grad_batches", [1])
            if isinstance(raw_acc, int):
                accumulate = max(1, int(raw_acc))
            else:
                accumulate = max(1, int(list(raw_acc)[0]))
        else:
            accumulate = max(1, int(legacy))
    # Use ``ceil`` (not floor) so this matches what the loop actually
    # does: the inner block fires `floor(L/A)` optimizer steps, and the
    # end-of-epoch flush adds one more when `L % A != 0` — i.e.
    # `ceil(L/A)` optimizer/scheduler steps per epoch. Floor here would
    # under-size ``total_steps`` and the cosine schedule would reach LR=0
    # before training ends.
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulate))

    # EQUALISE THE TRAINING BUDGET ACROSS ARCHITECTURES (added 08-08-2026).
    # Under full_pass, steps/epoch = sum(ceil(n_rows / row_cap)) — so a base
    # with a SMALLER memory-driven row cap silently gets MORE optimizer steps
    # for the same `epochs`. In the 07-08 run that gave v2.6 20 300 steps and
    # v3/tabicl 9 100 at identical `epochs: 100`: v2.6 was trained 2.2x longer
    # purely because its row cap is 11k instead of 26k. The drift showed it —
    # v2.6 @3e-5 reached l2sp 0.61 against v3's 0.0045 — which confounds every
    # cross-architecture comparison with "who got more gradient steps".
    # With `target_total_steps` set, epochs are trimmed so each base runs the
    # SAME number of steps; `train.epochs` stays the upper bound.
    # The budget is a TARGET, not a ceiling: epochs are trimmed when a base would
    # overshoot it and RAISED when it would undershoot.
    #
    # WHY BOTH DIRECTIONS (measured, 10/11-08-2026 run). Trimming alone silently left
    # LGD at a fraction of the budget, because steps/epoch is
    # `sum_over_datasets(ceil(rows_i / row_cap))` and LGD has 6 small training tables:
    # at 100 epochs tabicl got 800 steps, v3 1 600 and v2.6 3 200 against a 9 100-step
    # target. So LGD was 3-11x undertrained AND still confounded across bases — exactly
    # the problem the equalisation was added to remove, in the track nobody checked.
    # Every LGD trial in that run was worse than its untuned baseline; 800 steps is not
    # a fair test of whether continued pretraining works.
    target_steps = getattr(cfg.train, "target_total_steps", None)
    max_epochs = getattr(cfg.train, "max_epochs_for_step_budget", None)
    if target_steps:
        wanted = max(1, math.ceil(int(target_steps) / steps_per_epoch))
        ceiling = int(max_epochs) if max_epochs else max(epochs, wanted)
        capped = min(wanted, ceiling)
        if capped != epochs:
            LOGGER.info(
                "Step-budget equalisation: %d steps/epoch x %d configured epochs = "
                "%d steps; %s to %d epochs (~%d steps) to hit the %d-step target "
                "shared by every base.%s",
                steps_per_epoch, epochs, steps_per_epoch * epochs,
                "trimming" if capped < epochs else "EXTENDING",
                capped, steps_per_epoch * capped, int(target_steps),
                "" if capped == wanted else
                f" Capped at max_epochs_for_step_budget={ceiling}, so this trial "
                f"runs {steps_per_epoch * capped} steps, SHORT of the target.",
            )
            epochs = capped
        else:
            LOGGER.info(
                "Step-budget equalisation: %d steps/epoch x %d epochs = %d steps, "
                "already the %d-step target — keeping all epochs.",
                steps_per_epoch, epochs, steps_per_epoch * epochs, int(target_steps),
            )
    total_steps = max(1, steps_per_epoch * epochs)
    optimizer, scheduler = _make_optimizer_and_scheduler(
        model, cfg, total_steps=total_steps,
    )

    use_amp, amp_dtype = _resolve_amp_dtype(cfg, device)
    # Dynamic loss scaling is useful for FP16 but unnecessary for BF16, whose
    # exponent range matches FP32. Keeping it disabled for BF16 also makes an
    # optimizer step's success/failure unambiguous.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(use_amp and amp_dtype == torch.float16),
    )

    # ---- 4) checkpoint name + path ---------------------------------------- #
    # resolve_writable_staging_path PROBES writability here — before the
    # baseline eval and all training compute — and falls back to $VSC_DATA
    # with a loud warning if staging can't be written from this node (the
    # failure mode that killed all 32 PD trials on 2026-07-03).
    save_path = Path(save_path) if save_path is not None else (
        resolve_writable_staging_path(cfg.checkpoint.trained_dir) / track / descriptive_name(
            run_name=str(cfg.run_name),
            track=track,
            base_path=base_checkpoint_config,
            learning_rate=float(learning_rate),
            seed=int(cfg.seed),
            use_lora=bool(use_lora),
            query_fraction=float(query_fraction),
            accumulate_grad_batches=int(accumulate),
            epoch_pass_mode=pass_mode,
            min_train_rows=int(min_train_rows or 0),
        )
    )

    # ---- 5) training loop -------------------------------------------------- #
    raw_grad_clip = cfg.train.grad_clip_norm
    grad_clip = None if raw_grad_clip in (None, "null") else float(raw_grad_clip)

    history: list[EpochRecord] = []
    t0 = time.monotonic()

    # Comprehensive debug banner — logged BEFORE the first forward pass so the
    # log retains full context even if the trial is later OOM-killed / SIGKILLed
    # / diverges. Captures env, cluster, hardware, versions, storage roots and
    # the complete hyperparameter set. (User request 2026-06-23.)
    _log_debug_banner(
        track=track, device=device, base_checkpoint=base_checkpoint_path,
        save_path=save_path, learning_rate=float(learning_rate),
        use_lora=bool(use_lora), query_fraction=float(query_fraction),
        accumulate=int(accumulate), pass_mode=pass_mode,
        n_estimators_finetune=int(n_estimators_finetune),
        max_rows_per_epoch=int(max_rows_per_epoch),
        max_cells_per_epoch=max_cells_per_epoch,
        context_sampling=context_sampling,
        epochs=int(epochs), total_steps=int(total_steps),
        steps_per_epoch=int(steps_per_epoch),
        weight_decay=float(cfg.optimizer.weight_decay),
        l2sp_lambda=(l2sp_lambda if l2sp_anchor is not None else 0.0),
        warmup_fraction=float(cfg.scheduler.warmup_fraction),
        seed=int(cfg.seed), n_train_ds=len(split.train),
        n_test_ds=len(split.test), use_amp=bool(use_amp),
        amp_dtype=(
            str(amp_dtype).removeprefix("torch.")
            if amp_dtype is not None else "disabled"
        ),
    )
    LOGGER.info(
        "Starting %d epochs | %d train steps/epoch | accumulate=%d | "
        "total_steps=%d | lr=%.1e | base=%s | seed=%d | device=%s | "
        "max_rows_per_epoch=%d | query_fraction=%.2f",
        epochs, len(train_loader), accumulate, total_steps, float(learning_rate),
        Path(base_checkpoint_config).name, int(cfg.seed), device,
        max_rows_per_epoch, query_fraction,
    )
    LOGGER.info("Save target   : %s", save_path)

    # ---- 5a) BASELINE eval — pre-finetuning snapshot ----------------------- #
    # This is the reference point against which every finetuned epoch must
    # beat. We emit it as ``epoch=-1`` in the per-epoch CSV / on_epoch_end
    # callback. If the final epoch's metrics are NOT clearly above this row,
    # the finetuning has not improved over the unmodified base — likely a
    # sign that the LR is too high, the trial diverged, or the corpus is
    # too small to move the prior.
    epoch_eval_n0 = int(getattr(cfg.train, "epoch_eval_subsample_samples", 0))
    epoch_eval_ne = int(getattr(cfg.train, "epoch_eval_n_estimators", 1))
    if family == "tabicl":
        # Use the family's own inference-ensemble size (8, upstream's default)
        # rather than TabPFN's 32. The monitor exists so its curves are directly
        # comparable to the final eval numbers, and config/eval.yaml scores
        # tabicl with 8 — running the monitor at 32 broke that comparability
        # and cost 4x the monitor time. (2026-08-06.)
        epoch_eval_ne = int(getattr(cfg.train, "epoch_eval_n_estimators_tabicl",
                                    min(epoch_eval_ne, 8)))
    # Monitor CADENCE: run the (expensive, 32-estimator) per-epoch eval only
    # every k-th epoch (+ always the first and last). The 2026-07-03 run's
    # timing lines showed the monitor dominating one_sample epochs ~3:1 —
    # config/train.yaml sets 5; code default 1 preserves legacy behaviour.
    epoch_eval_every = max(1, int(getattr(cfg.train, "epoch_eval_every", 1)))
    # TabICLv2 has no cheap single-forward monitor path (evaluate_on_split's
    # prepare_eval_chunk/_forward are TabPFN-specific), so its monitor always
    # goes through the sklearn ensemble path regardless of epoch_eval_ne.
    use_ensemble_eval = epoch_eval_ne > 1 or family == "tabicl"
    snapshot_path = Path(str(save_path) + ".epoch_eval.ckpt") if use_ensemble_eval else None
    # (test_metric, train_metric) pairs from epochs where the monitor RAN —
    # the divergence detector's metric window must only look at these, else
    # the by-design NaN metrics of skipped epochs would fake a collapse.
    monitored_metrics: list[tuple[float, float]] = []

    # Hold the monitoring sample and context/query split FIXED across the
    # unmodified baseline and every monitored epoch.  The previous code added
    # ``10_000 * (epoch + 1)`` to these seeds, so each point used different
    # rows.  On only 2,000 rows per dataset that sampling noise created apparent
    # AUC/RMSE "lift" which disappeared in the full K-fold evaluation.  A
    # learning curve must change only the checkpoint, not its evaluation set.
    monitor_train_seed = int(cfg.seed) + 10_000
    monitor_test_seed = int(cfg.seed) + 20_000

    # Picks the per-track primary + secondary metric names. For PD we
    # add brier_score as the calibration-collapse early-warning metric
    # (see chat 2026-05-21: loss-vs-AUC divergence diagnosed as
    # over-confidence). For LGD we keep R² as the rank/scale secondary.
    if split.train and split.train[0].task_type == "classification":
        track_primary_metric = "roc_auc"
        track_secondary_metric = "brier_score" if use_ensemble_eval else ""
        track_task_type = "classification"
    else:
        track_primary_metric = "rmse"
        track_secondary_metric = "r2"
        track_task_type = "regression"
    track_metric_names: tuple[str, ...] = (
        (track_primary_metric,) if not track_secondary_metric
        else (track_primary_metric, track_secondary_metric)
    )

    def _do_eval(
        ckpt_path: Path | str, refs: list[DatasetRef], *, seed: int,
    ) -> dict[str, float]:
        """Dispatcher: ensemble eval (sklearn API, n_estimators>1) or the
        cheap single-forward path."""
        if use_ensemble_eval:
            return evaluate_ensemble_on_split(
                ckpt_path=ckpt_path,
                refs=refs,
                n_estimators=epoch_eval_ne,
                n_subsample=epoch_eval_n0,
                query_fraction=query_fraction,
                seed=seed,
                device=device,
                task_type=track_task_type,
                metric_names=track_metric_names,
                family=family,
            )
        result = evaluate_on_split(
            model, refs, criterion=criterion, device=device,
            metric_name=track_metric_names,
            n_inference_subsample_samples=epoch_eval_n0,
            seed=seed,
            query_fraction=query_fraction,
        )
        # evaluate_on_split returns dict[str, float] when given a tuple.
        return result if isinstance(result, dict) else {track_primary_metric: float(result)}

    if epoch_eval_n0 > 0:
        LOGGER.info(
            "Baseline eval (epoch=-1, model = unmodified base checkpoint, "
            "n_estimators=%d, qf=%.2f) — this is the score every finetuned "
            "epoch must beat. Ensemble path: %s.",
            epoch_eval_ne, query_fraction,
            "TabPFNClassifier/Regressor sklearn API" if use_ensemble_eval
            else "single forward pass (cheap)",
        )
        # For the baseline we evaluate the UNMODIFIED base checkpoint — no
        # snapshot needed. We feed the base_checkpoint path straight to
        # the ensemble loader, mirroring what tabpfn-untuned does in the
        # full eval pipeline.
        baseline_ckpt = (
            str(base_checkpoint_path) if use_ensemble_eval
            else save_path  # ignored on the cheap path; the live model is used
        )
        baseline_train_d = _do_eval(
            baseline_ckpt, split.train, seed=monitor_train_seed,
        )
        baseline_test_d = _do_eval(
            baseline_ckpt, split.test, seed=monitor_test_seed,
        )
        baseline_train_p = float(baseline_train_d.get(track_primary_metric, float("nan")))
        baseline_test_p  = float(baseline_test_d.get(track_primary_metric, float("nan")))
        baseline_train_s = (
            float(baseline_train_d.get(track_secondary_metric, float("nan")))
            if track_secondary_metric else float("nan")
        )
        baseline_test_s = (
            float(baseline_test_d.get(track_secondary_metric, float("nan")))
            if track_secondary_metric else float("nan")
        )
        baseline_record = EpochRecord(
            epoch=-1,
            train_loss=float("nan"),       # no training has happened yet
            elapsed_sec=0.0,
            lr=0.0,
            train_metric=baseline_train_p,
            test_metric=baseline_test_p,
            metric_name=track_primary_metric,
            secondary_train_metric=baseline_train_s,
            secondary_test_metric=baseline_test_s,
            secondary_metric_name=track_secondary_metric,
            epoch_time_sec=0.0,
        )
        history.append(baseline_record)
        monitored_metrics.append((baseline_test_p, baseline_train_p))
        if on_epoch_end is not None:
            on_epoch_end(baseline_record)
        if track_secondary_metric:
            LOGGER.info(
                "epoch=-1 BASELINE  %s(train)=%.4f  %s(test)=%.4f  "
                "%s(train)=%.4f  %s(test)=%.4f",
                track_primary_metric, baseline_train_p,
                track_primary_metric, baseline_test_p,
                track_secondary_metric, baseline_train_s,
                track_secondary_metric, baseline_test_s,
            )
        else:
            LOGGER.info(
                "epoch=-1 BASELINE  %s(train)=%.4f  %s(test)=%.4f",
                track_primary_metric, baseline_train_p,
                track_primary_metric, baseline_test_p,
            )

    for epoch in range(epochs):
        model.train()
        # NOTE (2026-08-06): do NOT snap TabICLv2's frozen stages back to eval()
        # here. `.training` picks the ALGORITHM in TabICLv2 (train forward vs the
        # no_grad, KV-cached inference forward that writes into its input in
        # place), so eval-ing them crashed every `_iclhead` trial in the
        # 2026-08-05 run. Freezing is requires_grad=False only — see
        # load_tabicl_for_training. The whole model therefore stays on the
        # train forward path, exactly as in full-FT.
        # Per-epoch reshuffle: a fresh random subsample is drawn from each
        # dataset's full processed CSV (see ProcessedDatasetLoader.set_epoch).
        train_ds.set_epoch(epoch)
        running_loss = 0.0
        n_batches = 0
        # Number of real (.backward()-ed) micro-batches accumulated since the
        # last optimizer step. This — NOT the `enumerate` step counter — drives
        # the accumulation boundary, because steps can be SKIPPED (missing
        # context class, non-finite loss) before any backward, and using the
        # raw step index would mis-scale gradients and desync the end-of-epoch
        # flush whenever a skip occurs. (Bug fixed 2026-06-23.)
        micro_since_step = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_t0 = time.monotonic()

        # Per-epoch debug accumulators — used to compose the
        # end-of-epoch INFO line that gives gradient-noise visibility
        # (pre-clip grad-norm max/mean) and per-dataset loss spread
        # (so a single misbehaving dataset shows up clearly).
        epoch_grad_norms: list[float] = []
        epoch_clipped_count = 0
        epoch_step_losses: list[tuple[str, float]] = []   # (dataset_id, loss)
        # Realised positive rate of the CONTEXT each step actually saw.
        # This is the only direct evidence that `context_sampling` did what
        # it claims: proportional sampling of a 1 %-default dataset leaves
        # ~1 % positives in context, balanced sampling leaves far more. The
        # 07-08 run switched to balanced and the logs could not confirm it.
        epoch_ctx_pos_rate: list[float] = []
        stage_drift: dict[str, float] = {}
        epoch_skipped_steps = 0
        epoch_optimizer_steps = 0
        epoch_amp_skipped_steps = 0
        # Timing + regularization accumulators — let a single shared log line
        # answer the two recurring questions after a run: (1) WHERE is the
        # epoch's wall-clock going — forward/backward compute, data loading
        # (Lustre/GPFS I/O), or the per-epoch monitoring eval? and (2) is the
        # L2-SP anchor actually contributing to the loss?
        epoch_compute_s = 0.0                 # Σ forward+backward+step time
        epoch_l2sp: list[float] = []          # per-step L2-SP penalty values

        for step, batch in enumerate(train_loader, start=1):
            step_t0 = time.monotonic()
            batch = batch.to(device)
            # Skip-on-missing-class check — mirrors the official
            # `FinetunedTabPFNClassifier._should_skip_batch` at
            # `TabPFN .txt`. If a stratified subsample
            # happens to draw a context split that's missing one of
            # the labels present in the query split, the CE loss is
            # ill-defined for those query rows (no positive softmax
            # target). We skip the entire step, the dataloader will
            # serve a different dataset next step. Important on PD
            # with strong class imbalance (default rate ~1-3 %).
            if (batch.task_type == "classification"
                    and _query_missing_context_class(batch)):
                LOGGER.warning(
                    "epoch=%d step=%d dataset=%s — query labels not subset of "
                    "context labels; skipped step.",
                    epoch, step, batch.dataset_id,
                )
                epoch_skipped_steps += 1
                continue
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype,
            ):
                # Branch on batch type. The new TabPFNEnsembleBatch (path
                # taken when `inference_config` is non-None, i.e. every
                # real training run) carries N preprocessed views; we
                # forward each one, stack logits as (Q,B,E,L), and let
                # CE / NLL average across the E*Q query positions —
                # mirroring `FinetunedTabPFNClassifier._forward_with_loss`
                # at `TabPFN .txt`. The legacy TabPFNBatch
                # path (E=1, no preprocessing) is kept ONLY for the
                # mocked smoke test in tests/test_train.py.
                from src.train.tabpfn_preprocessing import TabPFNEnsembleBatch
                from src.train.dataloader import TabICLTrainBatch
                if isinstance(batch, TabICLTrainBatch):
                    # TabICLv2 family (2026-08-04): one forward over all E
                    # ensemble members — `TabICL._train_forward` routes on
                    # model.train(). Losses are verbatim tabicl's own
                    # finetuning objectives (`tabicl._finetune.{classifier,
                    # regressor}._compute_batch_loss`): CE over the first
                    # n_classes of the 10 logit columns; mean pinball over
                    # the 999-quantile head on z-normed targets.
                    out = model(batch.X, batch.y_train)   # (E, test, out_dim)
                    if batch.task_type == "classification":
                        n_cls = int(batch.y_train.max().item()) + 1
                        loss = torch.nn.functional.cross_entropy(
                            out[..., :n_cls].reshape(-1, n_cls),
                            batch.y_query.long().reshape(-1),
                        )
                    else:
                        from src.train.tabicl_model import tabicl_pinball_loss
                        loss = tabicl_pinball_loss(out, batch.y_query)
                elif isinstance(batch, TabPFNEnsembleBatch):
                    loss = _ensemble_step_loss(
                        model, batch, criterion=criterion,
                    )
                else:
                    pred_logits, y_target, _, _ = _forward(model, batch)
                    if batch.task_type == "classification":
                        loss = _classification_loss(
                            pred_logits, batch.y_query,
                            n_classes=_n_classes(batch), criterion=criterion,
                        )
                    else:
                        loss = _regression_loss(
                            pred_logits, y_target, criterion=criterion,
                        )
                # L2-SP penalty (full-FT only; anchor is None under LoRA).
                # Added to the back-prop loss ONLY — `loss` stays the pure
                # data loss (CE / NLL) for logging, curves, and the
                # non-finite / divergence checks below.
                if l2sp_anchor is not None:
                    _pen = _l2sp_penalty(model, l2sp_anchor, l2sp_lambda)
                    if _pen is not None:
                        epoch_l2sp.append(float(_pen.detach().cpu().item()))
                    loss_to_backprop = (
                        (loss + _pen) if _pen is not None else loss
                    ) / accumulate
                else:
                    loss_to_backprop = loss / accumulate

            if torch.isnan(loss).item() or torch.isinf(loss).item():
                LOGGER.warning(
                    "epoch=%d step=%d dataset=%s — non-finite loss; skipped",
                    epoch, step, batch.dataset_id,
                )
                optimizer.zero_grad(set_to_none=True)
                epoch_skipped_steps += 1
                continue

            scaler.scale(loss_to_backprop).backward()
            micro_since_step += 1

            stepped = False
            pre_clip_norm: float | None = None
            if micro_since_step >= accumulate:
                # We always unscale here (with or without grad_clip) so we
                # can MEASURE the pre-clip gradient norm. This is the
                # single most useful number for diagnosing the loss
                # explosion: if pre-clip norm hits 100s of × the
                # grad_clip threshold (= 1.0 in our cfg), the LR is too
                # high for the current gradient noise.
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(grad_clip if grad_clip is not None else float("inf")),
                )
                pre_clip_norm = float(total_norm.detach().cpu().item())
                epoch_grad_norms.append(pre_clip_norm)
                if grad_clip is not None and pre_clip_norm > grad_clip:
                    epoch_clipped_count += 1

                # Inspect the AMP scaler's internal state BEFORE step:
                # `scaler.step()` returns the optimizer's return value
                # when the step ran, and None when it was skipped due to
                # inf/NaN. We mirror this into `stepped` and only advance
                # the LR scheduler when the optimizer actually stepped —
                # otherwise the schedule drifts ahead of the real
                # optimization trajectory (real bug found in pipeline
                # review 2026-05-21).
                epoch_optimizer_steps += 1
                _ = scaler.step(optimizer)
                stepped = not _amp_step_was_skipped(scaler)
                scaler.update()
                if stepped:
                    scheduler.step()
                else:
                    epoch_amp_skipped_steps += 1
                    LOGGER.warning(
                        "epoch=%d step=%d: AMP scaler skipped optimizer step "
                        "(inf/NaN grads). Scheduler NOT advanced this step.",
                        epoch, step,
                    )
                optimizer.zero_grad(set_to_none=True)
                micro_since_step = 0

            loss_val = float(loss.detach().cpu().item())
            running_loss += loss_val
            n_batches += 1
            epoch_step_losses.append((batch.dataset_id, loss_val))
            # Every batch type carries `ctx_pos_rate`, measured in its builder
            # from the CANONICAL labels — never recompute it from the tensors
            # here, because both families class-permute per ensemble member and
            # the permuted labels give a meaningless rate.
            _cpr = getattr(batch, "ctx_pos_rate", float("nan"))
            if _cpr == _cpr:                      # not NaN → classification
                epoch_ctx_pos_rate.append(float(_cpr))

            step_dt = time.monotonic() - step_t0
            epoch_compute_s += step_dt
            cur_lr = float(scheduler.get_last_lr()[0])
            gpu_mb = ""
            if device == "cuda" and torch.cuda.is_available():
                gpu_mb = f" gpu_mem_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB"

            # Keep every step visible for short one-sample epochs.  Full-pass
            # epochs can exceed 200 steps, so sample their progress while the
            # end-of-epoch debug line retains the complete min/max/std and
            # worst-dataset diagnostics. This cuts ~9,000 low-signal lines per
            # large PD trial without hiding failures or outliers.
            grad_str = (
                f" grad_norm={pre_clip_norm:.3f}" if pre_clip_norm is not None
                else " grad_norm=    -    "
            )
            step_log_interval = max(
                1, int(getattr(cfg.train, "step_log_interval", 10)),
            )
            n_epoch_steps = len(train_loader)
            if (
                n_epoch_steps <= 20
                or step in (1, n_epoch_steps)
                or step % step_log_interval == 0
            ):
                LOGGER.info(
                    "  step=%3d/%d ds=%-22s loss=%.4f lr=%.2e%s %.2fs/step%s",
                    step, n_epoch_steps, batch.dataset_id,
                    loss_val, cur_lr, grad_str, step_dt, gpu_mb,
                )

        # Flush any pending gradients from a partial accumulation window
        # at the end of the epoch — otherwise the trailing micro-batches'
        # gradients are computed but never applied. Driven by the real
        # micro-batch counter (`micro_since_step > 0`), NOT the step index,
        # so it stays correct in the presence of skipped steps. No-op when
        # `accumulate == 1` (every backward already triggered a full step,
        # leaving the counter at 0).
        if micro_since_step > 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(grad_clip if grad_clip is not None else float("inf")),
            )
            pre_clip_flush = float(total_norm.detach().cpu().item())
            epoch_grad_norms.append(pre_clip_flush)
            if grad_clip is not None and pre_clip_flush > grad_clip:
                epoch_clipped_count += 1
            epoch_optimizer_steps += 1
            _ = scaler.step(optimizer)
            stepped_flush = not _amp_step_was_skipped(scaler)
            scaler.update()
            if stepped_flush:
                scheduler.step()
            else:
                epoch_amp_skipped_steps += 1
                LOGGER.warning(
                    "epoch=%d (flush): AMP scaler skipped optimizer step "
                    "(inf/NaN grads). Scheduler NOT advanced.", epoch,
                )
            optimizer.zero_grad(set_to_none=True)

        # Pure training time for this epoch (data loading + forward/backward
        # + optimizer steps), measured BEFORE the monitoring eval so the two
        # phases can be separated in the log below. eval_phase_s = epoch_dt -
        # train_phase_s then reveals whether the per-epoch monitor (a 32-member
        # ensemble inference, by default) dominates the epoch wall-clock.
        train_phase_dt = time.monotonic() - epoch_t0

        # End-of-epoch monitoring eval: score the model on a small
        # subsample of each train- and test-dataset and record the
        # primary metric (ROC-AUC for PD, RMSE for LGD). Both end up in
        # the per-epoch CSV so it's easy to see whether the model is
        # still improving, has plateaued, or has started overfitting.
        # Skipped when `cfg.train.epoch_eval_subsample_samples == 0`.
        # End-of-epoch eval — runs via the same dispatcher
        # (_do_eval) as the baseline (epoch=-1). For the ensemble path
        # we save a snapshot of the live model's state_dict here so the
        # sklearn-API loader has a checkpoint file to mmap. The snapshot
        # is overwritten on every MONITORED epoch (every `epoch_eval_every`-th;
        # non-monitored epochs write nothing), keeping disk usage bounded at
        # one .ckpt-worth (~213 MB v3 / ~43 MB v2.6) per trial.
        # Track-level metric names already resolved before the loop —
        # `track_primary_metric` / `track_secondary_metric` /
        # `track_metric_names`. Keep local aliases for the EpochRecord
        # construction below to mirror the previous variable names.
        metric_name = track_primary_metric
        secondary_metric_name = track_secondary_metric

        # Cadence: evaluate on the first epoch, every `epoch_eval_every`-th
        # epoch, and always the final one. Skipped epochs record NaN metrics
        # (loss is always recorded) and skip the snapshot write too.
        monitor_this_epoch = (
            epoch_eval_n0 > 0
            and (epoch % epoch_eval_every == 0 or epoch == epochs - 1)
        )
        if monitor_this_epoch:
            if use_ensemble_eval and snapshot_path is not None:
                if family == "tabicl":
                    # Non-destructive by construction: state_dict tensors are
                    # detach().cpu() copies, no adapter merge exists for this
                    # family. Written in upstream's {config, state_dict}
                    # schema so TabICLClassifier/Regressor(model_path=...)
                    # loads it directly.
                    from src.train.tabicl_model import save_finetuned_tabicl
                    assert tabicl_model_config is not None
                    save_finetuned_tabicl(
                        model, tabicl_model_config, snapshot_path,
                    )
                else:
                    assert architecture_config is not None
                    _save_eval_snapshot(
                        model, architecture_config, snapshot_path,
                        criterion=criterion,
                        inference_config=inference_config,
                    )
                eval_ckpt_path: Path | str = snapshot_path
            else:
                eval_ckpt_path = save_path    # ignored on cheap path

            train_metrics = _do_eval(
                eval_ckpt_path, split.train,
                seed=monitor_train_seed,
            )
            test_metrics = _do_eval(
                eval_ckpt_path, split.test,
                seed=monitor_test_seed,
            )
            train_metric = float(train_metrics.get(metric_name, float("nan")))
            test_metric  = float(test_metrics.get(metric_name,  float("nan")))
            secondary_train = (
                float(train_metrics.get(secondary_metric_name, float("nan")))
                if secondary_metric_name else float("nan")
            )
            secondary_test = (
                float(test_metrics.get(secondary_metric_name, float("nan")))
                if secondary_metric_name else float("nan")
            )
            monitored_metrics.append((test_metric, train_metric))
            # Per-stage drift: only on monitored epochs (needs a full pass over
            # the anchored parameters). Empty under LoRA, where no anchor exists.
            if l2sp_anchor is not None:
                stage_drift = _stage_drift(
                    model, l2sp_anchor, l2sp_stage_names, l2sp_stage_w0)
                if stage_drift:
                    LOGGER.info(
                        "  drift by stage: %s",
                        "  ".join(f"{k}={100 * v:.3f}%" for k, v in
                                  sorted(stage_drift.items(), key=lambda kv: -kv[1])),
                    )
        else:
            train_metric = test_metric = float("nan")
            secondary_train = secondary_test = float("nan")

        train_loss = running_loss / max(1, n_batches)
        epoch_dt = time.monotonic() - epoch_t0
        elapsed = time.monotonic() - t0

        # Per-epoch GRADIENT-NOISE summary — these three numbers are
        # the smoking gun for the loss-explosion diagnosis. With the
        # cfg grad_clip_norm=1.0:
        #   * grad_norm_max ≫ 1   ⇒  optimizer constantly clipping
        #   * clipped_frac ≈ 1.0  ⇒  LR is too high for the noise level
        #   * loss_std large      ⇒  per-dataset gradients disagree wildly
        if epoch_grad_norms:
            gnorm_arr = np.asarray(epoch_grad_norms)
            gnorm_mean = float(gnorm_arr.mean())
            gnorm_max  = float(gnorm_arr.max())
            clipped_frac = (
                float(epoch_clipped_count) / max(1, len(epoch_grad_norms))
            )
        else:
            gnorm_mean = gnorm_max = float("nan")
            clipped_frac = float("nan")
        step_losses = [v for _, v in epoch_step_losses]
        if step_losses:
            loss_arr = np.asarray(step_losses)
            loss_min = float(loss_arr.min())
            loss_max = float(loss_arr.max())
            loss_std = float(loss_arr.std())
            # Identify the single worst (highest-loss) dataset of the epoch.
            worst_ds, worst_loss = max(epoch_step_losses, key=lambda t: t[1])
        else:
            loss_min = loss_max = loss_std = worst_loss = float("nan")
            worst_ds = "?"
        # Peak GPU memory this epoch — the single most useful number for
        # debugging OOM / tuning max_rows_per_epoch on a new GPU (e.g. the
        # B200's 192 GiB). Reset after reading so each epoch reports its own
        # peak rather than a running max.
        gpu_peak = ""
        if device == "cuda" and torch.cuda.is_available():
            try:
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                resv_gb = torch.cuda.max_memory_reserved() / 1e9
                total_gb = _gpu_total_mem_gb(device) or 0.0
                pct = (100.0 * peak_gb / total_gb) if total_gb else float("nan")
                # peak/total headroom makes the max_rows_per_epoch tuning
                # decision a one-glance read on the new GPU (e.g. B200 192 GiB).
                gpu_peak = (
                    f"  gpu_peak_alloc={peak_gb:.2f}GB gpu_peak_reserved={resv_gb:.2f}GB"
                    f" gpu_total={total_gb:.0f}GB ({pct:.0f}% of VRAM)"
                )
                torch.cuda.reset_peak_memory_stats()
            except Exception:                                      # pragma: no cover
                gpu_peak = ""
        # Timing decomposition: where did the epoch's wall-clock go?
        eval_phase_dt = max(0.0, epoch_dt - train_phase_dt)
        data_io_s = max(0.0, train_phase_dt - epoch_compute_s)
        steps_per_s = (n_batches / train_phase_dt) if train_phase_dt > 0 else float("nan")
        # Scientific notation: at conservative LRs the penalty is ~1e-6..1e-9
        # (‖w−w0‖² after tiny steps), which a %.4f rendered as a useless
        # "0.0000" in every Jul-10 log line.
        l2sp_str = (
            (f"  ctx_pos={100 * float(np.mean(epoch_ctx_pos_rate)):.2f}%"
             if epoch_ctx_pos_rate else "")
            + (
                f"  l2sp={float(np.mean(epoch_l2sp)):.3e}"
                f" drift={_drift_pct(float(np.mean(epoch_l2sp)), l2sp_lambda, l2sp_w0_norm)}"
                if epoch_l2sp else ""
            )
        )
        # Mean loss per source dataset this epoch (see EpochRecord docstring).
        _per_ds: dict[str, list[float]] = {}
        for _ds, _lv in epoch_step_losses:
            _per_ds.setdefault(_ds, []).append(_lv)
        per_dataset_loss = {
            k: float(np.mean(v)) for k, v in sorted(_per_ds.items())
        }
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            elapsed_sec=elapsed,
            lr=float(scheduler.get_last_lr()[0]),
            train_metric=train_metric,
            test_metric=test_metric,
            metric_name=metric_name,
            secondary_train_metric=secondary_train,
            secondary_test_metric=secondary_test,
            secondary_metric_name=secondary_metric_name,
            epoch_time_sec=epoch_dt,
            per_dataset_loss=per_dataset_loss,
            stage_drift=stage_drift,
            optimizer_steps=epoch_optimizer_steps,
            amp_skipped_steps=epoch_amp_skipped_steps,
            data_skipped_steps=epoch_skipped_steps,
        )
        history.append(record)
        if on_epoch_end is not None:
            on_epoch_end(record)

        # ONE comprehensive line per epoch (user request 2026-07-11): every
        # number needed to diagnose a run lives on a single greppable line —
        # loss stats, LR, monitor metrics, gradient health, skip counters,
        # L2-SP, GPU peak, worst dataset, and the full timing decomposition.
        # `|`-separated key=value groups; fixed field order for easy awk/grep.
        _sec = (
            f"  {secondary_metric_name}(tr)={secondary_train:.4f}"
            f" {secondary_metric_name}(te)={secondary_test:.4f}"
            if secondary_metric_name else ""
        )
        _mon = "" if monitor_this_epoch else " [no-monitor]"
        LOGGER.info(
            "epoch=%2d/%d | loss=%.4f (min=%.4f max=%.4f std=%.4f) lr=%.2e | "
            "%s(tr)=%.4f %s(te)=%.4f%s%s | "
            "grad: mean=%.3f max=%.3f clip=%.2f | "
            "steps=%d amp_skip=%d data_skip=%d | worst=%s(%.4f)%s%s | "
            "t: epoch=%.1fs compute=%.1fs io=%.1fs monitor=%.1fs %.2fst/s | "
            "total=%.1fmin",
            epoch, epochs - 1, train_loss, loss_min, loss_max, loss_std,
            record.lr,
            metric_name, train_metric, metric_name, test_metric, _sec, _mon,
            gnorm_mean, gnorm_max, clipped_frac,
            epoch_optimizer_steps, epoch_amp_skipped_steps,
            epoch_skipped_steps,
            worst_ds, worst_loss, l2sp_str, gpu_peak,
            epoch_dt, epoch_compute_s, data_io_s, eval_phase_dt, steps_per_s,
            elapsed / 60.0,
        )

        # ---- Divergence detection — early abort on collapse ------------ #
        # We watch the LAST ``divergence_patience`` epoch records and
        # trip if any of these heuristics matches the published
        # collapse signature from the 2026-05-28 PD run (trials a0/a1):
        #   * loss CONSTANT to ~6 sig figs across all watched epochs
        #     (the model is dead — outputs deterministic garbage)
        #   * train AND test metric == 0.5 (PD) or NaN for both
        #     (classifier predicts random; regressor produces NaN)
        #   * fraction of AMP scaler skips in the watched epochs > 50 %
        #     (most steps are being thrown away)
        # When tripped, we break out of the epoch loop, set
        # diverged=True on the TrainingResult, and STILL save the
        # checkpoint (so the user can inspect what went wrong). The
        # caller writes a status="DIVERGED" row to the manifest.
        diverge_patience = int(getattr(cfg.train, "divergence_patience", 5))
        recent = [r for r in history if r.epoch >= 0][-diverge_patience:]
        if len(recent) == diverge_patience:
            losses = [r.train_loss for r in recent if not math.isnan(r.train_loss)]
            # Metric-based collapse signals must only look at epochs where the
            # monitor actually RAN — with `epoch_eval_every > 1` the skipped
            # epochs hold NaN by design and would fake a `metric_nan` collapse.
            monitored_recent = monitored_metrics[-diverge_patience:]
            metrics_window_full = len(monitored_recent) == diverge_patience
            test_metrics_recent = [t for t, _ in monitored_recent]
            train_metrics_recent = [tr for _, tr in monitored_recent]

            # A FLAT LOSS IS NOT A DEAD MODEL. This rule alone killed a perfectly healthy
            # trial in run-8: v2.6 @3e-7 full-FT held loss at 0.4689-0.4690 for five
            # epochs — inside the 1e-4 window — while its weight drift rose monotonically
            # (0.042 % -> 0.085 %), its held-out AUC sat at 0.7151, and its gradients were
            # normal. It was training, slowly, exactly as the lowest learning rate in the
            # sweep is supposed to. The abort wasted the trial AND removed the one
            # configuration the run existed to test (Garg's 3e-7).
            #
            # A model that has actually died does not move: its weights stop changing.
            # So require BOTH a flat loss and flat drift. ||w - w0|| is monotone by
            # construction, so "not growing" is the signal that nothing is being learnt.
            recent_drift = [max(r.stage_drift.values()) for r in recent
                            if getattr(r, "stage_drift", None)]
            drift_flat = (
                len(recent_drift) >= 2
                and (max(recent_drift) - min(recent_drift)) < 1e-6
            )
            loss_constant = (
                len(losses) == diverge_patience
                and max(losses) - min(losses) < 1e-4
                # No drift record (monitor off) -> fall back to the loss-only rule rather
                # than never tripping at all.
                and (drift_flat or not recent_drift)
            )
            # For PD (ROC-AUC primary), AUC=0.5 means random — exact
            # equality after the dead-model collapse.
            # For LGD (RMSE primary), the analogue is NaN train/test.
            auc_random = (
                track_primary_metric == "roc_auc"
                and metrics_window_full
                and all(
                    not math.isnan(t) and not math.isnan(tr)
                    and abs(t - 0.5) < 1e-4 and abs(tr - 0.5) < 1e-4
                    for t, tr in zip(test_metrics_recent, train_metrics_recent)
                )
            )
            # `metric_nan` is a collapse signal ONLY when the per-epoch
            # monitor is actually running. When it's disabled
            # (epoch_eval_subsample_samples == 0) every metric is NaN BY
            # DESIGN — not because the model died — so guarding on
            # ``epoch_eval_n0 > 0`` prevents a spurious DIVERGED abort that
            # would otherwise truncate every monitor-disabled run after
            # `patience` epochs. (Bug fixed 2026-06-23.)
            metric_nan = (
                epoch_eval_n0 > 0
                and metrics_window_full
                and all(
                    math.isnan(t) and math.isnan(tr)
                    for t, tr in zip(test_metrics_recent, train_metrics_recent)
                )
            )
            attempted_steps = sum(r.optimizer_steps for r in recent)
            amp_skips = sum(r.amp_skipped_steps for r in recent)
            amp_skip_storm = (
                attempted_steps > 0
                and amp_skips / attempted_steps > 0.50
            )
            if loss_constant or auc_random or metric_nan or amp_skip_storm:
                reason = (
                    "loss_const" if loss_constant
                    else "auc_random" if auc_random
                    else "metric_nan" if metric_nan
                    else "amp_skip_storm"
                )
                LOGGER.error(
                    "DIVERGED at epoch=%d after %d-epoch patience window "
                    "(reason=%s). Aborting early. Last good epoch=%d.",
                    epoch, diverge_patience, reason,
                    epoch - diverge_patience,
                )
                diverged = True
                diverged_at_epoch = max(0, epoch - diverge_patience)
                diverge_reason = reason
                break

    # ---- 5b) gather summary metrics from the history ------------------ #
    # Pulled out into local vars so the TrainingResult constructor below
    # has the per-trial baseline + last-good numbers in one place. These
    # also get written through to the manifest CSV.
    diverged = locals().get("diverged", False)
    diverged_at_epoch = locals().get("diverged_at_epoch", None)
    diverge_reason = locals().get("diverge_reason", "")
    # Baseline row (epoch=-1) is always the first entry in history when
    # the per-epoch monitor is enabled.
    baseline_row = next(
        (r for r in history if r.epoch == -1), None,
    )
    # Last good epoch = last non-divergent epoch. When diverged=True,
    # we skip the tail of NaN/constant rows.
    last_good = None
    if diverged and diverged_at_epoch is not None:
        for r in reversed(history):
            if r.epoch >= 0 and r.epoch <= diverged_at_epoch:
                last_good = r
                break
    if last_good is None:
        last_good = next((r for r in reversed(history) if r.epoch >= 0), None)

    # ---- 6) save final weights + permanent provenance --------------------- #
    train_dataset_ids = train_ids        # already computed at step (1)
    test_dataset_ids  = test_ids
    training_seconds = time.monotonic() - t0
    gpu_name = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:                                       # pragma: no cover
            gpu_name = "cuda"
    try:
        import tabpfn as _tabpfn
        tabpfn_version = getattr(_tabpfn, "__version__", None)
    except ImportError:                                         # pragma: no cover
        tabpfn_version = None
    try:
        from importlib.metadata import version as _pkg_version
        tabicl_version = _pkg_version("tabicl")
    except Exception:                                           # pragma: no cover
        tabicl_version = None
    provenance = {
        "schema_version":      1,
        "run_name":            str(cfg.run_name),
        "track":               track,
        "model_family":        family,
        # What the grid's use_lora axis actually did for this family —
        # eval-side interpretation must not assume LoRA semantics.
        "adaptation_mode": (
            ("iclhead_only" if use_lora else "full_ft") if family == "tabicl"
            else ("lora" if use_lora else "full_ft")
        ),
        "task_type":           "classification" if track == "pd" else "regression",
        "saved_at":            time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hyperparameters": {
            "base_checkpoint":     base_checkpoint_config,
            "base_checkpoint_resolved": str(base_checkpoint_path),
            "learning_rate":       float(learning_rate),
            "weight_decay":        float(cfg.optimizer.weight_decay),
            # Effective L2-SP strength actually applied this trial (0.0 when
            # off, or when use_lora makes it inert). See _l2sp_penalty.
            "l2sp_lambda":         (l2sp_lambda if l2sp_anchor is not None else 0.0),
            "betas":               [0.9, 0.999],          # hardcoded AdamW betas
            "scheduler_type":      "warmup_cosine",       # hardcoded schedule family
            "warmup_fraction":     float(cfg.scheduler.warmup_fraction),
            "epochs":              int(cfg.train.epochs),
            "accumulate_grad_batches": int(accumulate),
            "grad_clip_norm":      grad_clip,
            "amp":                 bool(cfg.train.amp),
            "amp_dtype":           (
                str(amp_dtype).removeprefix("torch.")
                if use_amp and amp_dtype is not None else "disabled"
            ),
            "max_rows_per_epoch":  max_rows_per_epoch,
            "max_cells_per_epoch": max_cells_per_epoch,
            # Context-construction strategy — the axis Tanna et al. 2026
            # measure as worth more AUC than the choice of model, so a
            # checkpoint that does not record it is not reproducible.
            "context_sampling": context_sampling,
            # Per-step ensemble members actually used. Varies by track AND
            # family (TabPFN pd=2 / lgd=8; TabICLv2 2 for both), and it drives
            # both the gradient noise and the member-aware row scaling — so
            # a checkpoint that doesn't record it can't be reproduced from
            # its own provenance. (Added 2026-08-04.)
            "n_estimators_finetune": int(n_estimators_finetune),
            "query_fraction":      query_fraction,
            "epoch_pass_mode":     pass_mode,
            # Swept since run-8. Without it in the provenance the eval cannot tell the
            # two corpus arms apart and writes both to one results directory.
            "min_train_rows":      int(min_train_rows or 0),
            "seed":                int(cfg.seed),
            "use_lora":            bool(use_lora),
            "lora": (
                {
                    "r":              int(cfg.lora.r),
                    "alpha":          int(cfg.lora.alpha),
                    "dropout":        float(cfg.lora.dropout),
                    "target_modules": list(cfg.lora.target_modules),
                }
                if (use_lora and hasattr(cfg, "lora")) else None
            ),
        },
        "training_datasets":   train_dataset_ids,
        "test_datasets":       test_dataset_ids,
        "n_train_datasets_meta": len(split.train),
        "n_test_datasets_meta":  len(split.test),
        "training_time_seconds": float(training_seconds),
        "device":              device,
        "gpu":                 gpu_name,
        # str() is deliberate: torch.__version__ is a torch.torch_version.
        # TorchVersion (a str subclass), which torch.load(weights_only=True)
        # — PyTorch >=2.6's default — refuses to unpickle. Embedding the raw
        # object made every trained checkpoint unloadable at eval time
        # (the 2026-05-31 run: 1475 UnpicklingError fails). Cast to a plain
        # str so the .ckpt is weights_only-safe regardless of the loader.
        "torch_version":       str(torch.__version__),
        "tabpfn_version":      (str(tabpfn_version) if tabpfn_version is not None else None),
        "tabicl_version":      (str(tabicl_version) if tabicl_version is not None else None),
    }
    if family == "tabicl":
        from src.train.tabicl_model import save_finetuned_tabicl
        assert tabicl_model_config is not None
        save_finetuned_tabicl(
            model, tabicl_model_config, save_path,
            provenance=provenance,
        )
    else:
        # Pass the criterion only for regression — the LGD bar-distribution
        # state must round-trip through the checkpoint (`criterion.*` keys);
        # for PD the criterion is a stateless CrossEntropyLoss.
        save_criterion = criterion if track == "lgd" else None
        save_finetuned(
            model, architecture_config, save_path,
            criterion=save_criterion,
            inference_config=inference_config,
            provenance=provenance,
        )
    LOGGER.info(
        "Saved final-epoch checkpoint: %s "
        "(provenance.json next to the .ckpt records HPs, datasets, GPU=%s, "
        "training_time=%.1fs)",
        save_path, gpu_name, training_seconds,
    )

    # Clean up the rolling per-epoch eval snapshot — kept as a single
    # file overwritten each epoch, so on success there's exactly one
    # file to remove. Best-effort: a failure here doesn't fail the trial.
    if snapshot_path is not None and snapshot_path.exists():
        try:
            snapshot_path.unlink()
        except OSError as exc:                                         # pragma: no cover
            LOGGER.warning(
                "Failed to remove eval snapshot %s (continuing): %s",
                snapshot_path, exc,
            )

    # NOTE: the training pipeline does NOT score the model on the test
    # split. Evaluation of trained checkpoints belongs to the eval
    # pipeline (`scripts/eval_pipeline.py` / `config/eval.yaml`). The
    # test_dataset_ids are recorded inside the checkpoint's provenance
    # ONLY as metadata so the eval can identify which test datasets
    # correspond to this checkpoint without re-running the splitter.

    # ---- 7) end-of-trial summary table -------------------------------- #
    # A compact text table the user sees AT the bottom of every log
    # file. Carries the same numbers that go into the manifest CSV row,
    # but human-readable. Critical when scanning a sweep of 48+ logs
    # to find which trials actually improved over baseline.
    if baseline_row is not None and last_good is not None:
        pm = baseline_row.metric_name or "metric"
        sm = baseline_row.secondary_metric_name
        delta_train = last_good.train_metric - baseline_row.train_metric
        delta_test  = last_good.test_metric  - baseline_row.test_metric
        # Sign meaning for the primary metric: higher = better for roc_auc, lower = better for rmse.
        higher_better = (pm == "roc_auc")
        def _trend(d: float) -> str:
            if math.isnan(d):
                return "  ?  "
            sign = "+" if d >= 0 else "−"
            magnitude = abs(d)
            good = (d > 0) == higher_better
            marker = "↑" if good else "↓"
            return f" {marker}{sign}{magnitude:.4f}"
        LOGGER.info("")
        LOGGER.info("─" * 78)
        LOGGER.info("  TRIAL SUMMARY  (%s)", save_path.name)
        LOGGER.info("─" * 78)
        LOGGER.info(
            "  %-22s %-12s %-12s %-12s",
            "stage", f"{pm}(train)", f"{pm}(test)", "train_loss",
        )
        LOGGER.info(
            "  %-22s %-12.4f %-12.4f %-12s",
            "epoch=-1 baseline",
            baseline_row.train_metric, baseline_row.test_metric, "n/a",
        )
        last_epoch_tag = (
            f"epoch={last_good.epoch} (DIVERGED@{diverged_at_epoch})"
            if diverged else f"epoch={last_good.epoch} (final)"
        )
        LOGGER.info(
            "  %-22s %-12.4f %-12.4f %-12.4f",
            last_epoch_tag,
            last_good.train_metric, last_good.test_metric, last_good.train_loss,
        )
        LOGGER.info(
            "  %-22s %-12s %-12s",
            "Δ (lift over baseline)",
            _trend(delta_train).strip(), _trend(delta_test).strip(),
        )
        if sm:
            LOGGER.info(
                "  %-22s %-12.4f %-12.4f",
                f"baseline {sm}",
                baseline_row.secondary_train_metric,
                baseline_row.secondary_test_metric,
            )
            LOGGER.info(
                "  %-22s %-12.4f %-12.4f",
                f"final {sm}",
                last_good.secondary_train_metric,
                last_good.secondary_test_metric,
            )
        LOGGER.info(
            "  %-22s %s",
            "status",
            f"DIVERGED ({diverge_reason})" if diverged else "OK",
        )
        LOGGER.info("─" * 78)
        LOGGER.info("")

    elapsed = time.monotonic() - t0
    return TrainingResult(
        final_ckpt_path=save_path,
        history=history,
        n_train_datasets=len(split.train),
        n_test_datasets=len(split.test),
        elapsed_sec=elapsed,
        descriptive_name=save_path.name,
        diverged=bool(diverged),
        diverged_at_epoch=diverged_at_epoch,
        diverge_reason=str(diverge_reason),
        baseline_train_metric=(
            float(baseline_row.train_metric) if baseline_row is not None else float("nan")
        ),
        baseline_test_metric=(
            float(baseline_row.test_metric) if baseline_row is not None else float("nan")
        ),
        final_train_metric=(
            float(last_good.train_metric) if last_good is not None else float("nan")
        ),
        final_test_metric=(
            float(last_good.test_metric) if last_good is not None else float("nan")
        ),
        final_train_loss=(
            float(last_good.train_loss) if last_good is not None else float("nan")
        ),
        final_secondary_train=(
            float(last_good.secondary_train_metric) if last_good is not None else float("nan")
        ),
        final_secondary_test=(
            float(last_good.secondary_test_metric) if last_good is not None else float("nan")
        ),
        primary_metric_name=(
            str(last_good.metric_name) if last_good is not None
            else (str(baseline_row.metric_name) if baseline_row is not None else "")
        ),
        secondary_metric_name=(
            str(last_good.secondary_metric_name) if last_good is not None
            else (str(baseline_row.secondary_metric_name) if baseline_row is not None else "")
        ),
        # What actually ran, for the manifest and for docs/RESULTS.md later.
        total_optimizer_steps=int(sum(r.optimizer_steps for r in history)),
        epochs_run=int(epochs),
        steps_per_epoch=int(steps_per_epoch),
        min_train_rows=int(min_train_rows or 0),
        train_dataset_ids=tuple(c.dataset_id for c in split.train),
        test_dataset_ids=tuple(c.dataset_id for c in split.test),
        train_rows_total=int(sum(c.n_rows for c in split.train)),
        test_rows_total=int(sum(c.n_rows for c in split.test)),
        final_drift=float(
            max((v for v in (last_good.stage_drift or {}).values()), default=float("nan"))
            if last_good is not None else float("nan")
        ),
    )
