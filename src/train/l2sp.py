"""L2-SP for adapter scopes, measured on the change of the adapted weights.

L2-SP (Li, Grandvalet & Davoine 2018) penalises ``0.5 * lam * ||W - W0||^2``: the distance of
the weights from their pretrained values. Full and frozen-backbone CPT apply it directly to the
trainable tensors (``loop._l2sp_penalty``). Adapters leave ``W0`` untouched and train new
parameters, so the same quantity has to be expressed through them:

* **LoRA** adds ``dW = (alpha / r) * B @ A`` to each adapted query/value slice. The penalty is
  ``0.5 * lam * sum ||dW||_F^2`` — exactly L2-SP on the effective weights. ``B`` starts at zero,
  so the penalty and its gradient are exactly zero at initialisation, like full-weight L2-SP.
  (Plain weight decay on ``A`` and ``B`` is a different regulariser: a nuclear-norm penalty on
  ``dW`` that is already non-zero at initialisation because ``A`` is random.)
* **DiagScale** multiplies each projected query dimension ``j`` by ``1 + delta_j``. The weight
  change that produces the same scaling is ``dW_q[j, :] = delta_j * W_q[j, :]`` (and
  ``db_j = delta_j * b_j``), so the penalty is ``0.5 * lam * sum_j delta_j^2 (||W_q[j]||^2 + b_j^2)``.
  This is exact for TabPFN v2/v2.6 (the scaling is applied to the weight rows themselves) and
  for any query transform that is elementwise and query-independent. TabPFN v3 and TabICLv2
  scale queries by a query-dependent factor before the correction (and TabICL applies rotary
  embeddings first), so there it is the weight-space change of the same multiplier rather than
  an exact equivalence.

With these definitions the same ``lam`` means the same thing in every scope of the scope
experiment: "how far are the adapted weights allowed to move from their pretrained values".
"""
from __future__ import annotations

import torch
from torch.nn.utils import parametrize


def _lora_modules(model):
    from src.model.lora import LowRankWeight
    found = []
    for owner in model.modules():
        if not parametrize.is_parametrized(owner):
            continue
        for name, plist in owner.parametrizations.items():
            for p in plist:
                if isinstance(p, LowRankWeight):
                    found.append((owner, name, p))
    return found


def _diagscale_row_weights(model) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """[(delta parameter, squared row norm per corrected dimension, row weights W0)]."""
    from src.model.diagscale import QueryScale, QueryWeightScale
    terms = []
    names = dict(model.named_modules())
    for owner in model.modules():
        if not parametrize.is_parametrized(owner):
            continue
        for name, plist in owner.parametrizations.items():
            for p in plist:
                if isinstance(p, QueryWeightScale):
                    original = owner.parametrizations[name].original.detach()
                    rows = original.reshape(-1, original.shape[-1])[p.start:p.stop].float()
                    terms.append((p.delta, rows.square().sum(-1), rows))
    for path, module in names.items():
        if not isinstance(module, QueryScale) or isinstance(module, QueryWeightScale):
            continue
        parent = names[path.rpartition(".")[0]]
        q = getattr(parent, "q_projection", None)
        if q is not None:
            weight, bias = q.weight.detach().float(), (q.bias.detach().float() if q.bias is not None else None)
        elif hasattr(parent, "in_proj_weight"):
            dim = parent.in_proj_weight.shape[-1]
            weight = parent.in_proj_weight.detach().float()[:dim]
            bias = parent.in_proj_bias.detach().float()[:dim] if parent.in_proj_bias is not None else None
        else:
            raise ValueError(f"DiagScale target {path}: no query projection found for L2-SP")
        squared = weight.square().sum(-1) + (bias.square() if bias is not None else 0)
        if squared.numel() != module.delta.numel():
            raise ValueError(f"DiagScale target {path}: {module.delta.numel()} corrections "
                             f"but {squared.numel()} query rows")
        terms.append((module.delta, squared.reshape(module.delta.shape),
                      weight if bias is None else torch.cat([weight, bias[:, None]], dim=1)))
    return terms


class EffectiveWeightL2SP:
    """Callable penalty ``0.5 * lam * ||W - W0||^2`` over the weights an adapter changes."""

    def __init__(self, model, mode: str):
        if mode not in ("lora", "diagscale"):
            raise ValueError(f"Effective-weight L2-SP supports lora/diagscale, not {mode!r}")
        self.mode = mode
        if mode == "lora":
            self.lora = _lora_modules(model)
            if not self.lora:
                raise ValueError("No LoRA modules found for L2-SP")
            norm = 0.0
            for owner, name, p in self.lora:
                original = owner.parametrizations[name].original.detach().float()
                rows = original.reshape(-1, original.shape[-1])
                for start, stop in p.slices:
                    norm += float(rows[start:stop].square().sum())
            self.reference_norm = norm ** 0.5
        else:
            self.terms = _diagscale_row_weights(model)
            if not self.terms:
                raise ValueError("No DiagScale corrections found for L2-SP")
            self.terms = [(delta, weight.to(delta.device), None) for delta, weight, _ in self.terms]
            self.reference_norm = float(sum(w.sum() for _, w, _ in self.terms)) ** 0.5

    def squared_change(self) -> torch.Tensor:
        """``sum ||dW||^2`` in FP32, differentiable with respect to the adapter parameters."""
        if self.mode == "lora":
            total = None
            for _, _, p in self.lora:
                term = p.delta().float().square().sum()
                total = term if total is None else total + term
            return total
        total = None
        for delta, weight, _ in self.terms:
            term = (delta.float().square() * weight).sum()
            total = term if total is None else total + term
        return total

    def __call__(self, lam: float) -> torch.Tensor:
        return 0.5 * float(lam) * self.squared_change()
