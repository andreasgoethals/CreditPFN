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

from src.train.corpus import DatasetRef, CorpusSplit, split_from_cfg
from src.train.dataloader import (
    ProcessedDatasetLoader, TabPFNBatch, identity_collate, prepare_eval_chunk,
)
from src.train.metrics import (
    classification_metric, regression_metric,
    mean_ignore_nan,
)
from src.train.model import load_tabpfn_for_training, save_finetuned
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
    lora_tag = "_lora" if use_lora else ""
    return (
        f"{run_name}_{track}_{base_stem}_lr{lr_tag}_seed{seed}"
        f"{qf_tag}{acc_tag}{pass_tag}{lora_tag}.ckpt"
    )


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #


def make_warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
    schedule_type: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear-warmup → cosine-decay LR multiplier.

    Matches HuggingFace's ``get_cosine_schedule_with_warmup``
    (which is what TabPFN's ``FinetunedTabPFNClassifier`` uses
    internally; see ``repositories/TabPFN .txt``):

      * step 0           → multiplier = 0
      * step warmup_steps → multiplier = 1
      * step total_steps  → multiplier = 0  (cosine "warmup_cosine" only)

    ``schedule_type``:
        - ``"constant"``      — multiplier = 1 throughout
        - ``"warmup_only"``   — linear warmup, then constant 1
        - ``"warmup_cosine"`` — linear warmup, then cosine to 0
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
            return 0.5 * (1.0 + math.cos(math.pi * progress))
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
    warmup_fraction, seed, n_train_ds, n_test_ds, use_amp,
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
        f"max_rows_per_epoch={max_rows_per_epoch} max_cells_per_epoch={max_cells_per_epoch} "
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
        f"numpy={np.__version__} tabpfn={tabpfn_ver} platform={platform.platform()}",
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
    (``repositories/TabPFN .txt`` and the live 2.x package):

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
    `repositories/TabPFN V2 Finetuning.txt`).
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
    (the pretraining max-classes; ``repositories/TabPFN .txt``),
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
    return criterion(logits=pred_logits, y=targets[:, :, 0]).mean()


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
        y=targets_BQ_reg.transpose(0, 1),       # (Q, B*E)
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

    Mirrors the official guard at ``repositories/TabPFN .txt``
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
    ``repositories/TabPFN .txt``.** Critical: we MUST write
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
) -> dict[str, float]:
    """Evaluate one TabPFN checkpoint via the sklearn API with full
    ensemble inference (``n_estimators`` forward passes per fit/predict,
    averaged with the package's standard feature-permutation strategy).

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
    split: CorpusSplit = split_from_cfg(cfg, track=track)
    LOGGER.info("Corpus split: %s", split.summary)
    train_ids = sorted({c.dataset_id for c in split.train})
    test_ids  = sorted({c.dataset_id for c in split.test})
    LOGGER.info(
        "Training datasets (n=%d): %s",
        len(train_ids), ", ".join(train_ids) if train_ids else "<none>",
    )
    LOGGER.info(
        "Held-out test datasets (n=%d): %s",
        len(test_ids), ", ".join(test_ids) if test_ids else "<none>",
    )
    if not split.train:
        raise RuntimeError(
            "Corpus split contains no training chunks. Run the data "
            "pipeline (`python scripts/data_pipeline.py`) first."
        )

    # ---- 2) base model + criterion ---------------------------------------- #
    lora_cfg_dict = (
        dict(cfg.lora) if (use_lora and hasattr(cfg, "lora")) else None
    )
    model, criterion, architecture_config, inference_config = (
        load_tabpfn_for_training(
            base_checkpoint_path, track=track, device=device,
            lora_config=lora_cfg_dict,
        )
    )

    # L2-SP anchor (Li et al. 2018): snapshot the synthetic-prior weights
    # w0 right after load, so the loss can penalise drift away from them
    # (see _l2sp_penalty). This is OUR deliberate addition for the
    # multi-dataset corpus continued-pretraining setting — the Real-TabPFN
    # experiments themselves use weight_decay=0.0 and NO anchor (their runs
    # are single-dataset). Configurable via cfg.optimizer.l2sp_lambda
    # (0.0 = off). FULL-FT only — with LoRA the base is frozen and cannot
    # drift, so L2-SP is inert and skipped.
    l2sp_lambda = float(getattr(cfg.optimizer, "l2sp_lambda", 0.0) or 0.0)
    l2sp_anchor: dict[str, torch.Tensor] | None = None
    if l2sp_lambda > 0.0 and not use_lora:
        l2sp_anchor = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        LOGGER.info(
            "L2-SP enabled: lambda=%.2e, anchoring %d trainable tensors to w0.",
            l2sp_lambda, len(l2sp_anchor),
        )
    elif l2sp_lambda > 0.0 and use_lora:
        LOGGER.info(
            "L2-SP lambda=%.2e configured but use_lora=True — base weights are "
            "frozen, so L2-SP is inert; skipping it.", l2sp_lambda,
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
    _REFERENCE_MEMBERS = 2
    if n_estimators_finetune > _REFERENCE_MEMBERS:
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
    total_steps = max(1, steps_per_epoch * epochs)
    optimizer, scheduler = _make_optimizer_and_scheduler(
        model, cfg, total_steps=total_steps,
    )

    use_amp = bool(cfg.train.amp) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
        epochs=int(epochs), total_steps=int(total_steps),
        steps_per_epoch=int(steps_per_epoch),
        weight_decay=float(cfg.optimizer.weight_decay),
        l2sp_lambda=(l2sp_lambda if l2sp_anchor is not None else 0.0),
        warmup_fraction=float(cfg.scheduler.warmup_fraction),
        seed=int(cfg.seed), n_train_ds=len(split.train),
        n_test_ds=len(split.test), use_amp=bool(use_amp),
    )
    LOGGER.info(
        "Starting %d epochs | %d train datasets/epoch | accumulate=%d | "
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
    # Monitor CADENCE: run the (expensive, 32-estimator) per-epoch eval only
    # every k-th epoch (+ always the first and last). The 2026-07-03 run's
    # timing lines showed the monitor dominating one_sample epochs ~3:1 —
    # config/train.yaml sets 5; code default 1 preserves legacy behaviour.
    epoch_eval_every = max(1, int(getattr(cfg.train, "epoch_eval_every", 1)))
    use_ensemble_eval = epoch_eval_ne > 1
    snapshot_path = Path(str(save_path) + ".epoch_eval.ckpt") if use_ensemble_eval else None
    # (test_metric, train_metric) pairs from epochs where the monitor RAN —
    # the divergence detector's metric window must only look at these, else
    # the by-design NaN metrics of skipped epochs would fake a collapse.
    monitored_metrics: list[tuple[float, float]] = []

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
            baseline_ckpt, split.train, seed=int(cfg.seed) + 10_000 * 0,
        )
        baseline_test_d = _do_eval(
            baseline_ckpt, split.test, seed=int(cfg.seed) + 20_000 * 0,
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
        epoch_skipped_steps = 0
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
            with torch.amp.autocast("cuda", enabled=use_amp):
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
                if isinstance(batch, TabPFNEnsembleBatch):
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
                _ = scaler.step(optimizer)
                stepped = not _amp_step_was_skipped(scaler)
                scaler.update()
                if stepped:
                    scheduler.step()
                else:
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

            step_dt = time.monotonic() - step_t0
            epoch_compute_s += step_dt
            cur_lr = float(scheduler.get_last_lr()[0])
            gpu_mb = ""
            if device == "cuda" and torch.cuda.is_available():
                gpu_mb = f" gpu_mem_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB"

            # Promoted from DEBUG to INFO on 2026-05-21 — without this
            # the user can't see per-dataset gradient behaviour and the
            # diagnosis of loss explosions is blind. One line per step
            # at 12-17 steps per epoch and 100 epochs gives ~1500 lines
            # per trial which is still very tractable.
            grad_str = (
                f" grad_norm={pre_clip_norm:.3f}" if pre_clip_norm is not None
                else " grad_norm=    -    "
            )
            LOGGER.info(
                "  step=%3d/%d ds=%-22s loss=%.4f lr=%.2e%s %.2fs/step%s",
                step, len(train_loader), batch.dataset_id,
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
            _ = scaler.step(optimizer)
            stepped_flush = not _amp_step_was_skipped(scaler)
            scaler.update()
            if stepped_flush:
                scheduler.step()
            else:
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
                seed=int(cfg.seed) + 10_000 * (epoch + 1),
            )
            test_metrics = _do_eval(
                eval_ckpt_path, split.test,
                seed=int(cfg.seed) + 20_000 * (epoch + 1),
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
        l2sp_str = (
            f"  l2sp_pen_mean={float(np.mean(epoch_l2sp)):.4f}" if epoch_l2sp else ""
        )
        LOGGER.info(
            "  ↳ debug: grad_norm mean=%.3f max=%.3f clipped_frac=%.2f  "
            "per_step_loss min=%.4f max=%.4f std=%.4f  "
            "worst_ds=%s (loss=%.4f)  skipped_steps=%d%s%s",
            gnorm_mean, gnorm_max, clipped_frac,
            loss_min, loss_max, loss_std,
            worst_ds, worst_loss, epoch_skipped_steps, l2sp_str, gpu_peak,
        )
        # Timing line — kept separate so it's easy to grep across a run to see
        # whether compute, data I/O, or the monitoring eval is the bottleneck.
        LOGGER.info(
            "  ↳ timing: epoch=%.1fs  train_phase=%.1fs (compute=%.1fs data_io=%.1fs "
            "%.2f steps/s)  monitor_eval=%.1fs",
            epoch_dt, train_phase_dt, epoch_compute_s, data_io_s,
            steps_per_s, eval_phase_dt,
        )

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
        )
        history.append(record)
        if on_epoch_end is not None:
            on_epoch_end(record)

        if secondary_metric_name:
            LOGGER.info(
                "epoch=%2d/%d  loss=%.4f  lr=%.2e  "
                "%s(train)=%.4f  %s(test)=%.4f  "
                "%s(train)=%.4f  %s(test)=%.4f  "
                "epoch_dt=%.1fs  elapsed=%.1fs",
                epoch, epochs - 1, train_loss, record.lr,
                metric_name, train_metric, metric_name, test_metric,
                secondary_metric_name, secondary_train,
                secondary_metric_name, secondary_test,
                epoch_dt, elapsed,
            )
        else:
            LOGGER.info(
                "epoch=%2d/%d  loss=%.4f  lr=%.2e  "
                "%s(train)=%.4f  %s(test)=%.4f  "
                "epoch_dt=%.1fs  elapsed=%.1fs",
                epoch, epochs - 1, train_loss, record.lr,
                metric_name, train_metric, metric_name, test_metric,
                epoch_dt, elapsed,
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
        recent = history[-diverge_patience:]
        if len(recent) == diverge_patience:
            losses = [r.train_loss for r in recent if not math.isnan(r.train_loss)]
            # Metric-based collapse signals must only look at epochs where the
            # monitor actually RAN — with `epoch_eval_every > 1` the skipped
            # epochs hold NaN by design and would fake a `metric_nan` collapse.
            monitored_recent = monitored_metrics[-diverge_patience:]
            metrics_window_full = len(monitored_recent) == diverge_patience
            test_metrics_recent = [t for t, _ in monitored_recent]
            train_metrics_recent = [tr for _, tr in monitored_recent]

            loss_constant = (
                len(losses) == diverge_patience
                and max(losses) - min(losses) < 1e-4
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
            if loss_constant or auc_random or metric_nan:
                reason = (
                    "loss_const" if loss_constant
                    else "auc_random" if auc_random
                    else "metric_nan"
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
    provenance = {
        "schema_version":      1,
        "run_name":            str(cfg.run_name),
        "track":               track,
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
            "max_rows_per_epoch":  max_rows_per_epoch,
            "max_cells_per_epoch": max_cells_per_epoch,
            "query_fraction":      query_fraction,
            "epoch_pass_mode":     pass_mode,
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
    }
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
    )
