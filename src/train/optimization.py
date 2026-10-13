"""Apply a mean microbatch gradient, clipping only after averaging."""
from __future__ import annotations

import math
import torch


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
