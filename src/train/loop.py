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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.train.corpus import ChunkRef, CorpusSplit, split_from_cfg
from src.train.dataloader import (
    ChunkDataset, TabPFNBatch, identity_collate, prepare_eval_chunk,
)
from src.train.metrics import (
    classification_metric, regression_metric,
    improvement_direction, mean_ignore_nan,
)
from src.train.model import load_tabpfn_for_training, save_finetuned

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class EpochRecord:
    """One row of the training history."""
    epoch: int
    train_loss: float
    elapsed_sec: float
    lr: float


@dataclass
class TrainingResult:
    """Returned by :func:`train_one_config`."""
    final_ckpt_path: Path
    test_metric_raw: float | None        # None if test split is empty
    test_metric_name: str
    history: list[EpochRecord] = field(default_factory=list)
    n_train_chunks: int = 0
    n_test_chunks: int = 0
    n_train_datasets: int = 0
    n_test_datasets: int = 0
    elapsed_sec: float = 0.0
    descriptive_name: str = ""           # the basename of final_ckpt_path


# --------------------------------------------------------------------------- #
# Public utility: descriptive checkpoint name
# --------------------------------------------------------------------------- #


def descriptive_name(
    *, run_name: str, track: str, base_path: str | Path,
    learning_rate: float, multi_chunk_policy: str, seed: int,
) -> str:
    """Build the on-disk filename encoding all tunable HPs.

    Schema:
        <run_name>_<track>_<base-stem>_lr<lr>_<policy_short>_seed<seed>.ckpt

    where ``base-stem`` is ``Path(base_path).stem`` (everything but the
    extension; the ``tabpfn-`` prefix kept for at-a-glance clarity)
    and ``policy_short`` collapses the verbose policy name to a tag:

        all_chunks_as_separate_datasets  →  allchunks
        first_chunk_only                 →  firstchunk
    """
    base_stem = Path(str(base_path)).stem
    policy_short = {
        "all_chunks_as_separate_datasets": "allchunks",
        "first_chunk_only":                "firstchunk",
    }.get(multi_chunk_policy, multi_chunk_policy)
    lr_tag = f"{learning_rate:.0e}".replace("+", "")
    return (
        f"{run_name}_{track}_{base_stem}_lr{lr_tag}_"
        f"{policy_short}_seed{seed}.ckpt"
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
    internally; see ``repositories/TabPFN .txt:18696``):

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


def _make_optimizer_and_scheduler(
    model: torch.nn.Module, cfg, *, total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW + warmup-cosine schedule, both driven by ``cfg``."""
    o = cfg.optimizer
    if o.type != "AdamW":
        raise ValueError(f"only optimizer.type='AdamW' supported (got {o.type!r})")

    lr = float(cfg.optimizer.lr) if hasattr(cfg.optimizer, "lr") else None
    if lr is None:
        # The script will set this from `cfg.tunable.learning_rates[i]`;
        # if absent, fall back to the first value of the tunable list.
        lr = float(cfg.tunable.learning_rates[0])

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(o.weight_decay),
        betas=tuple(o.betas),
    )
    sched = make_warmup_cosine_schedule(
        optim,
        total_steps=total_steps,
        warmup_fraction=float(cfg.scheduler.warmup_fraction),
        schedule_type=str(cfg.scheduler.type),
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


# --------------------------------------------------------------------------- #
# Forward pass + loss
# --------------------------------------------------------------------------- #


def _forward(
    model: torch.nn.Module,
    batch: TabPFNBatch,
) -> tuple[torch.Tensor, torch.Tensor, float | None, float | None]:
    """Run one TabPFN forward pass.

    Returns ``(pred_logits, y_target, znorm_mean, znorm_std)``. The
    last two are non-None only for regression (where we z-normalise
    the context y, mirroring LennartPurucker's reference pipeline at
    `repositories/TabPFN V2 Finetuning.txt:1463-1469`).
    """
    train_x = batch.X_context
    train_y = batch.y_context.float()
    test_x = batch.X_query
    cat_inds = batch.categorical_idx if batch.categorical_idx else None

    znorm_mean = znorm_std = None
    if batch.task_type == "regression":
        mean = train_y.mean(dim=0, keepdim=True)
        std = train_y.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_y = (train_y - mean) / std
        y_target = (batch.y_query.float() - mean) / std
        znorm_mean = float(mean.detach().cpu().item())
        znorm_std = float(std.detach().cpu().item())
    else:
        y_target = batch.y_query

    pred_logits = model(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        categorical_inds=cat_inds,
    )
    return pred_logits, y_target, znorm_mean, znorm_std


def _classification_loss(
    pred_logits: torch.Tensor, targets: torch.Tensor,
    *, n_classes: int, criterion: torch.nn.Module,
) -> torch.Tensor:
    """CrossEntropyLoss on the K-class slice of TabPFN's output."""
    logits = pred_logits[:, :, :n_classes].float()
    logits = logits.reshape(-1, logits.shape[-1])
    target = targets.long().flatten()
    return criterion(logits, target)


def _regression_loss(
    pred_logits: torch.Tensor, targets: torch.Tensor, *, criterion,
) -> torch.Tensor:
    """Bar-distribution NLL on the z-normalised targets."""
    return criterion(logits=pred_logits, y=targets[:, :, 0]).mean()


def _n_classes(batch: TabPFNBatch) -> int:
    """Max class index in this chunk, +1 → number of classes seen."""
    K = int(batch.y_context.flatten().max().item()) + 1
    K = max(K, int(batch.y_query.flatten().max().item()) + 1)
    return max(K, 2)            # binary at minimum


# --------------------------------------------------------------------------- #
# Test-set evaluation (called ONCE at end of training)
# --------------------------------------------------------------------------- #


def evaluate_on_split(
    model: torch.nn.Module,
    chunks: list[ChunkRef],
    *,
    cfg,
    criterion,
    device: str,
    metric_name: str,
) -> float:
    """Mean primary metric over a list of chunks (test-time inference).

    Used at the end of training to report the held-out test metric;
    can also be called from a future ``src/eval/`` to evaluate any
    saved checkpoint on the same chunks. Higher = better when paired
    with :func:`improvement_direction` (the caller multiplies).
    """
    if not chunks:
        return float("nan")
    model.eval()
    is_classification = chunks[0].task_type == "classification"
    per_chunk: list[float] = []
    seed = int(cfg.seed)
    n_inf = int(cfg.eval.n_inference_subsample_samples)

    with torch.no_grad():
        for i, ref in enumerate(chunks):
            batch = prepare_eval_chunk(
                ref, n_inference_subsample_samples=n_inf, seed=seed + i,
            ).to(device)
            pred_logits, y_target, zmean, zstd = _forward(model, batch)
            if is_classification:
                K = _n_classes(batch)
                logits = pred_logits[:, :, :K]
                value = classification_metric(
                    logits=logits, targets=y_target,
                    metric=metric_name, n_classes=K,
                )
            else:
                value = regression_metric(
                    logits=pred_logits, targets=y_target,
                    criterion=criterion, metric=metric_name,
                    znorm_mean=zmean, znorm_std=zstd,
                )
            per_chunk.append(value)

    return mean_ignore_nan(per_chunk)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def train_one_config(
    cfg,
    *,
    track: str | None = None,
    base_checkpoint: str | None = None,
    learning_rate: float | None = None,
    multi_chunk_policy: str | None = None,
    save_path: Path | str | None = None,
    on_epoch_end: Callable[[EpochRecord], None] | None = None,
) -> TrainingResult:
    """Run continued pretraining for one fixed (config, HP-tuple).

    The four arguments ``track``, ``base_checkpoint``, ``learning_rate``,
    ``multi_chunk_policy`` are the ONLY things the script expects to
    vary per run — see ``cfg.tunable`` in ``config/training.yaml``.
    All four default to either the explicit ``cfg.<...>`` field if
    set, or the first value of the corresponding tunable list.

    Parameters
    ----------
    cfg
        OmegaConf config (typically ``OmegaConf.load("config/training.yaml")``).
    track
        Override ``cfg.track``. ``None`` → use the value from ``cfg``.
    base_checkpoint
        Override the base weights path. ``None`` → use
        ``cfg.tunable.<classifier|regressor>_base_paths[0]``.
    learning_rate
        Override AdamW LR. ``None`` → ``cfg.tunable.learning_rates[0]``.
    multi_chunk_policy
        Override the multi-chunk policy. ``None`` →
        ``cfg.tunable.multi_chunk_policies[0]``.
    save_path
        Where to write the final-epoch checkpoint. ``None`` →
        ``cfg.checkpoint.trained_dir / descriptive_name(...)``.
    on_epoch_end
        Optional hook called after each epoch with the
        :class:`EpochRecord` (live progress logging in a script).

    Returns
    -------
    TrainingResult
        Includes the final checkpoint path, test metric (or NaN if
        the test bucket is empty), per-epoch train loss history.
    """
    # ---- resolve every tunable parameter ---------------------------------- #
    track = track or cfg.track
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    if base_checkpoint is None:
        bases = (cfg.tunable.classifier_base_paths if track == "pd"
                 else cfg.tunable.regressor_base_paths)
        base_checkpoint = str(bases[0])
    if learning_rate is None:
        learning_rate = float(cfg.tunable.learning_rates[0])
    if multi_chunk_policy is None:
        multi_chunk_policy = str(cfg.tunable.multi_chunk_policies[0])

    # Inject the resolved choices back into cfg so downstream helpers
    # (corpus split, optimizer factory) read them via the usual path.
    cfg.optimizer.lr = float(learning_rate)
    cfg.corpus.multi_chunk_policy = multi_chunk_policy

    _seed_everything(int(cfg.seed))
    device = _resolve_device(cfg)
    LOGGER.info(
        "Training track=%s on device=%s | base=%s | lr=%g | policy=%s | seed=%d",
        track, device, Path(base_checkpoint).name, learning_rate,
        multi_chunk_policy, int(cfg.seed),
    )

    # ---- 1) corpus split --------------------------------------------------- #
    split: CorpusSplit = split_from_cfg(cfg, track=track)
    LOGGER.info("Corpus split: %s", split.summary)
    if not split.train:
        raise RuntimeError(
            "Corpus split contains no training chunks. Run the data "
            "pipeline (`python scripts/data_pipeline.py`) first."
        )

    # ---- 2) base model + criterion ---------------------------------------- #
    model, criterion, architecture_config = load_tabpfn_for_training(
        base_checkpoint, track=track, device=device,
    )

    # ---- 3) DataLoader + optimiser / scheduler ---------------------------- #
    train_ds = ChunkDataset(
        split.train,
        n_total_target=int(cfg.train.n_finetune_ctx_plus_query_samples),
        query_fraction=float(cfg.train.finetune_ctx_query_split_ratio),
        seed=int(cfg.seed),
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
    accumulate = max(1, int(cfg.train.accumulate_grad_batches))
    steps_per_epoch = max(1, len(train_loader) // accumulate)
    total_steps = max(1, steps_per_epoch * epochs)
    optimizer, scheduler = _make_optimizer_and_scheduler(
        model, cfg, total_steps=total_steps,
    )

    use_amp = bool(cfg.train.amp) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- 4) checkpoint name + path ---------------------------------------- #
    save_path = Path(save_path) if save_path is not None else (
        Path(cfg.checkpoint.trained_dir) / track / descriptive_name(
            run_name=str(cfg.run_name),
            track=track,
            base_path=base_checkpoint,
            learning_rate=float(learning_rate),
            multi_chunk_policy=multi_chunk_policy,
            seed=int(cfg.seed),
        )
    )

    # ---- 5) training loop -------------------------------------------------- #
    raw_grad_clip = cfg.train.grad_clip_norm
    grad_clip = None if raw_grad_clip in (None, "null") else float(raw_grad_clip)

    history: list[EpochRecord] = []
    t0 = time.monotonic()
    LOGGER.info(
        "Starting %d epochs (%d train chunks/epoch, accumulate=%d, "
        "total_steps=%d, lr=%.1e)",
        epochs, len(train_loader), accumulate, total_steps, float(learning_rate),
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
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
                loss_to_backprop = loss / accumulate

            if torch.isnan(loss).item() or torch.isinf(loss).item():
                LOGGER.warning(
                    "epoch=%d step=%d dataset=%s — non-finite loss; skipped",
                    epoch, step, batch.dataset_id,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss_to_backprop).backward()

            if step % accumulate == 0:
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().cpu().item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        elapsed = time.monotonic() - t0
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            elapsed_sec=elapsed,
            lr=float(scheduler.get_last_lr()[0]),
        )
        history.append(record)
        if on_epoch_end is not None:
            on_epoch_end(record)

        LOGGER.info(
            "epoch=%2d  train_loss=%.4f  lr=%.2e  elapsed=%.1fs",
            epoch, train_loss, record.lr, elapsed,
        )

    # ---- 6) save final weights -------------------------------------------- #
    save_finetuned(model, architecture_config, save_path)
    LOGGER.info("Saved final-epoch checkpoint: %s", save_path)

    # ---- 7) test eval (single pass, no leak) ------------------------------ #
    metric_name = (
        cfg.eval.classification_metric if track == "pd"
        else cfg.eval.regression_metric
    )
    test_raw: float | None = None
    if split.test:
        test_raw = evaluate_on_split(
            model, split.test,
            cfg=cfg, criterion=criterion, device=device,
            metric_name=metric_name,
        )
        LOGGER.info("Test  %s=%.4f  (n_test_chunks=%d)",
                    metric_name, test_raw, len(split.test))
    else:
        LOGGER.warning("Test bucket is empty — no test metric reported.")

    elapsed = time.monotonic() - t0
    return TrainingResult(
        final_ckpt_path=save_path,
        test_metric_raw=test_raw,
        test_metric_name=metric_name,
        history=history,
        n_train_chunks=len(split.train),
        n_test_chunks=len(split.test),
        n_train_datasets=len({c.dataset_id for c in split.train}),
        n_test_datasets=len({c.dataset_id for c in split.test}),
        elapsed_sec=elapsed,
        descriptive_name=save_path.name,
    )
