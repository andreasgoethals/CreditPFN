"""Regression CPT objectives over TabPFN's bar-distribution output.

``bar_nll`` is upstream's pretraining loss (the reference). ``bucket_crps`` is the
continuous ranked probability score of the bucket CDF: it depends on cumulative mass, not on
the density inside individual buckets, so a comb-shaped density (TabPFN v2.6 regressor,
see ``density_checks``) is not penalised for mass that is merely redistributed within a
neighbourhood. It is a candidate objective, not the reference.
"""
from __future__ import annotations

import torch

OBJECTIVES = ("bar_nll", "bucket_crps")


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
    if objective == "bar_nll":
        return criterion(logits=logits, y=y.contiguous()).mean()
    if objective == "bucket_crps":
        return bucket_crps(logits, criterion.borders, y).mean()
    raise ValueError(f"Unknown regression objective: {objective!r}; expected one of {OBJECTIVES}")
