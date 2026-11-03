"""Checks that a regression base model's bucket density suits bar-distribution NLL training.

Diagnosed 03-11-2026: the released TabPFN v2.6 regressor predicts a "comb" over the 5,000
output buckets. Neighbouring bucket probabilities differ by orders of magnitude, so its
means and quantiles look normal while the likelihood of the exact target bucket is close to
random (initial NLL about 3 nats above v2/v3 on the same episode, also through upstream's own
TabPFNRegressor). NLL continued pretraining then mainly smooths that comb and moves point
predictions. These helpers measure the roughness so an unsuitable objective/base pairing is
refused instead of silently trained.
"""
from __future__ import annotations

import torch

#: Median absolute log-ratio between adjacent buckets, over buckets carrying the central
#: 90% of predicted mass. Measured 03-11-2026 on three LGD tables (600 context rows):
#: v2 0.07-0.21, v3 0.40-1.61, v2.6 3.12-3.87. Experiment 0 records it for every table.
ROUGHNESS_LIMIT = 2.5


def bucket_roughness(probabilities: torch.Tensor, *, central_mass: float = 0.9) -> torch.Tensor:
    """Per-row roughness of bucket probabilities, shape (..., buckets) -> (...)."""
    p = probabilities.float().clamp_min(1e-30)
    p = p / p.sum(-1, keepdim=True)
    cdf = p.cumsum(-1)
    tail = (1 - central_mass) / 2
    inside = (cdf >= tail) & (cdf <= 1 - tail)
    jumps = (p[..., 1:].log() - p[..., :-1].log()).abs()
    mask = inside[..., 1:] & inside[..., :-1]
    jumps = torch.where(mask, jumps, torch.nan)
    return torch.nanmedian(jumps, dim=-1).values


def check_regression_objective(logits: torch.Tensor, *, objective: str, base: str,
                               allow_rough_density: bool = False) -> float:
    """Raise when bar-NLL training is requested for a base whose density is a comb."""
    roughness = float(torch.nanmedian(bucket_roughness(logits.softmax(-1))))
    if objective == "bar_nll" and roughness > ROUGHNESS_LIMIT and not allow_rough_density:
        raise ValueError(
            f"{base}: bucket density roughness {roughness:.2f} exceeds {ROUGHNESS_LIMIT}; "
            "bar-distribution NLL would mostly smooth the output comb and distort point "
            "predictions. Use a CDF-based objective or declare allow_rough_density.")
    return roughness
