"""Optimizers and the mean-microbatch gradient step."""
from __future__ import annotations

import math
import torch

#: Parameter-name fragments that stay on AdamW under the Muon option: input/output maps,
#: embeddings and learned tokens are not hidden-layer matrices (Jordan et al. convention).
MUON_EXCLUDED = ("encoder", "decoder", "embed", "head", "thinking", "token", "border")


def muon_partition(named_parameters):
    """Split trainable parameters into (Muon matrices, AdamW rest)."""
    matrices, rest = [], []
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        lowered = name.lower()
        if p.ndim == 2 and not any(part in lowered for part in MUON_EXCLUDED):
            matrices.append(p)
        else:
            rest.append(p)
    return matrices, rest


class MuonAdamW(torch.optim.Optimizer):
    """Muon on hidden 2-D weight matrices, AdamW on everything else.

    Muon's update RMS is matched to AdamW (``adjust_lr_fn="match_rms_adamw"``), so the
    reference learning rate keeps the same meaning; this is the one-factor Muon arm.
    Both inner optimizers share their ``param_groups`` dictionaries with this wrapper,
    so a LambdaLR scheduler and gradient clipping act on all parameters together.
    """

    def __init__(self, named_parameters, *, lr, weight_decay, betas=(0.9, 0.999), momentum=0.95):
        matrices, rest = muon_partition(named_parameters)
        if not matrices:
            raise ValueError("Muon option found no hidden weight matrices to optimize")
        self.muon = torch.optim.Muon(matrices, lr=lr, weight_decay=weight_decay, momentum=momentum,
                                     adjust_lr_fn="match_rms_adamw")
        self.adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=weight_decay, betas=betas) if rest else None
        self.optimizers = [o for o in (self.muon, self.adamw) if o is not None]
        self.param_groups = [group for o in self.optimizers for group in o.param_groups]
        self.defaults = dict(lr=lr)

    @property
    def state(self):
        """Per-parameter optimizer state of both inner optimizers (read-only view)."""
        merged = {}
        for optimizer in self.optimizers:
            merged.update(optimizer.state)
        return merged

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MuonAdamW does not support closures")
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(),
                "adamw": self.adamw.state_dict() if self.adamw is not None else None}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None:
            self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = [group for o in self.optimizers for group in o.param_groups]


def make_optimizer(model, params, *, name: str, lr: float, weight_decay: float):
    """The study's optimizers: ``adamw`` (reference) or ``muon_adamw`` (one-factor arm)."""
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    if name == "muon_adamw":
        wanted = {id(p) for p in params}
        named = [(n, p) for n, p in model.named_parameters() if id(p) in wanted]
        return MuonAdamW(named, lr=lr, weight_decay=weight_decay)
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
