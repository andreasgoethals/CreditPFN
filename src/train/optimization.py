"""Optimizers and the mean-microbatch gradient step."""
from __future__ import annotations

import math

import torch


def make_optimizer(model, params, *, name: str, lr: float, weight_decay: float):
    """The study's optimizer: AdamW (betas 0.9/0.999). ``model`` is kept for the call contract."""
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    raise ValueError(f"Unknown optimizer: {name!r}")


def step_mean_gradient(model, optimizer, scaler, *, microbatches: int,
                       max_norm: float | None) -> tuple[float, bool]:
    if microbatches < 1:
        raise ValueError("Cannot step an empty gradient window")
    scaler.unscale_(optimizer)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    for p in parameters:
        if p.grad is not None:
            p.grad.div_(microbatches)
    norm = float(torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=max_norm if max_norm is not None else float('inf'),
    ).item())
    # BF16 uses a disabled GradScaler: scaler.step alone does NOT reject NaN gradients.
    finite = math.isfinite(norm)
    if finite:
        scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if hasattr(model, '_creditpfn_adapter'):
        model.zero_grad(set_to_none=True)
    return norm, finite
