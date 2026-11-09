"""Regression CPT objectives over TabPFN's bar-distribution output.

``window_nll`` is the reference LGD objective for TabPFN (config/design.yaml): the negative
log density of the target at the resolution of a window of ``WINDOW_BUCKETS`` neighbouring
buckets, ``-log(sum_{j in W} p_j / sum_{j in W} width_j)``. With 5 of 5,000 buckets the
resolution loss is negligible, and for smooth densities (v2, v3) it is numerically close to the
bar NLL. A comb-shaped density (released TabPFN v2.6 regressor) averages out inside the window,
so training is not rewarded for filling empty teeth of the comb, which moves point
predictions under the bar NLL. Buckets are treated as bars (piecewise uniform); targets beyond the outer
edges are scored at the outermost window, unlike upstream's half-normal tails.

``bar_nll`` is upstream's pretraining loss (Real-TabPFN). ``bucket_crps`` is the
continuous ranked probability score of the bucket CDF: it depends on cumulative mass, not on
the density inside individual buckets, so a comb-shaped density (TabPFN v2.6 regressor,
see ``density_checks``) is not penalised for mass that is merely redistributed within a
neighbourhood. It is a candidate objective, not the reference.
"""
from __future__ import annotations

import torch

OBJECTIVES = ("window_nll", "bar_nll", "bucket_crps")
WINDOW_BUCKETS = 5


def window_nll(logits: torch.Tensor, borders: torch.Tensor, y: torch.Tensor,
               *, window: int = WINDOW_BUCKETS) -> torch.Tensor:
    """Negative log density of ``y`` over a centred window of ``window`` buckets. (...)->(...)."""
    if window < 1 or window % 2 == 0:
        raise ValueError("The window must be a positive odd number of buckets")
    log_p = logits.float().log_softmax(-1)
    edges = borders.to(log_p)
    k = log_p.shape[-1]
    if edges.numel() != k + 1:
        raise ValueError("Borders must have one more entry than buckets")
    if window > k:
        raise ValueError(f"Window of {window} buckets exceeds the {k} available")
    half = window // 2
    index = (torch.searchsorted(edges.contiguous(), y.to(log_p).contiguous().unsqueeze(-1),
                                right=True).squeeze(-1) - 1).clamp(0, k - 1)
    start = (index - half).clamp(0, k - window)
    offsets = torch.arange(window, device=log_p.device)
    window_index = start.unsqueeze(-1) + offsets
    log_mass = log_p.gather(-1, window_index).logsumexp(-1)
    width = edges[start + window] - edges[start]
    return width.log() - log_mass


def bucket_crps(logits: torch.Tensor, borders: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CRPS of a piecewise-constant bucket distribution, evaluated at bucket edges.

    logits: (..., K) unnormalised bucket scores; borders: (K + 1,) increasing edges;
    y: (...) targets in the same (z-normalised) space. Returns (...) losses.
    Uses sum_k width_k * (F(right_k) - 1[y <= right_k])^2, exact up to within-bucket
    resolution; targets beyond the outer edges contribute their distance to the edge.
    """
    probabilities = logits.float().softmax(-1)
    cdf = probabilities.cumsum(-1)                          # F at each right edge
    right = borders[1:].to(cdf)
    widths = (borders[1:] - borders[:-1]).to(cdf)
    step = (y.unsqueeze(-1).to(cdf) <= right).to(cdf)       # empirical CDF of the target
    inside = ((cdf - step).square() * widths).sum(-1)
    low, high = borders[0].to(cdf), borders[-1].to(cdf)
    outside = (low - y.to(cdf)).clamp_min(0) + (y.to(cdf) - high).clamp_min(0)
    return inside + outside


def regression_loss(logits: torch.Tensor, y: torch.Tensor, *, criterion, objective: str) -> torch.Tensor:
    """Mean loss over query rows for the chosen objective. logits (Q, B, K), y (Q, B)."""
    if objective == "window_nll":
        finite = torch.isfinite(y)   # upstream bar NLL ignores missing targets; so does the window
        losses = window_nll(logits, criterion.borders, torch.where(finite, y, torch.zeros_like(y)))
        return losses[finite].mean()
    if objective == "bar_nll":
        return criterion(logits=logits, y=y.contiguous()).mean()
    if objective == "bucket_crps":
        return bucket_crps(logits, criterion.borders, y).mean()
    raise ValueError(f"Unknown regression objective: {objective!r}; expected one of {OBJECTIVES}")
