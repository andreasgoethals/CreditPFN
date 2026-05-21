"""Validation metrics for the two tracks.

Single source of truth for how the loop converts model output into a
scalar that early-stopping watches.

Public surface
--------------
* :func:`classification_metric` — ``roc_auc`` (default) or ``log_loss``
  on PD logits. ROC-AUC handles binary and multiclass (one-vs-rest).
* :func:`regression_metric` — ``neg_nll`` (default; bar-distribution
  NLL flipped so that higher = better, matching ``roc_auc`` semantics)
  or ``rmse``.
* :func:`improvement_direction` — returns ``+1`` if the metric should
  be maximised (improving = larger), ``-1`` if minimised. Lets the
  loop write ``best = metric * direction`` and use the same comparison
  everywhere.

All metrics are computed in numpy / sklearn from detached CPU tensors —
no autograd, no GPU memory pinning.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

CLASSIFICATION_CHOICES = ("roc_auc", "log_loss")
REGRESSION_CHOICES = ("neg_nll", "rmse")


def improvement_direction(metric: str) -> int:
    """Return ``+1`` if ``metric`` is maximised, ``-1`` if minimised.

    Used by the loop to fold "higher = better" / "lower = better" into
    a single ``score = metric * direction`` so early stopping just
    asks "did score increase?".
    """
    if metric in ("roc_auc", "neg_nll"):
        return +1
    if metric in ("log_loss", "rmse"):
        return -1
    raise ValueError(f"unknown metric {metric!r}")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def classification_metric(
    logits: torch.Tensor,           # (n_query, 1, n_classes) or (n_query, n_classes)
    targets: torch.Tensor,          # (n_query, 1, 1) or (n_query,)
    *,
    metric: Literal["roc_auc", "log_loss"],
    n_classes: int | None = None,
) -> float:
    """Single PD validation metric for one chunk.

    For binary tasks ROC-AUC uses the positive-class probability;
    for multiclass it uses the one-vs-rest macro-average. Falls back
    to NaN if the chunk's query split has only one class (which would
    make ROC-AUC undefined) — the caller filters NaNs before averaging.
    """
    from sklearn.metrics import roc_auc_score, log_loss

    logits_np = _flatten_logits(logits)               # (n_query, K)
    targets_np = _flatten_targets(targets).astype(np.int64)

    K = logits_np.shape[-1] if n_classes is None else int(n_classes)
    probs = _softmax(logits_np[:, :K])

    if metric == "roc_auc":
        unique = np.unique(targets_np)
        if len(unique) < 2:
            return float("nan")
        try:
            if K == 2:
                return float(roc_auc_score(targets_np, probs[:, 1]))
            return float(roc_auc_score(
                targets_np, probs, multi_class="ovr", average="macro",
            ))
        except ValueError:
            return float("nan")

    if metric == "log_loss":
        return float(log_loss(
            targets_np, probs,
            labels=list(range(K)),
        ))

    raise ValueError(f"unsupported classification metric {metric!r}")


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def regression_metric(
    logits: torch.Tensor,         # (n_query, 1, n_buckets)
    targets: torch.Tensor,        # (n_query, 1, 1)
    criterion,                    # FullSupportBarDistribution
    *,
    metric: Literal["neg_nll", "rmse"],
    znorm_mean: float | None = None,
    znorm_std: float | None = None,
) -> float:
    """Single LGD validation metric for one chunk.

    ``neg_nll``
        ``-criterion(logits, y).mean()`` — TabPFN's training objective,
        flipped so larger = better.

    ``rmse``
        Root mean squared error using the bar distribution's expected
        mean as the point prediction. If the loss was computed in
        z-normalised space (see :mod:`src.train.loop`), pass
        ``znorm_mean`` / ``znorm_std`` so we can invert the transform.
    """
    from tabpfn.architectures.base.bar_distribution import (
        FullSupportBarDistribution,
    )
    if not isinstance(criterion, FullSupportBarDistribution):
        raise TypeError(
            "regression_metric expects a FullSupportBarDistribution criterion"
        )

    if metric == "neg_nll":
        # criterion(logits, y) returns per-sample NLL; .mean() is positive.
        # We negate so "improvement" means "increase" everywhere.
        with torch.no_grad():
            nll = criterion(logits=logits, y=targets[:, :, 0]).mean()
        return float(-nll.detach().cpu().item())

    if metric == "rmse":
        with torch.no_grad():
            # Expected value under the bar distribution. The criterion's
            # `mean` op operates on a flat (B, n_buckets) tensor.
            flat_logits = logits.reshape(-1, logits.shape[-1])
            preds = criterion.mean(flat_logits)            # (n_query,)
            # Both `preds` (the bar-dist expectation) AND `targets`
            # (the z-normalised tensor passed in by `src.train.loop._forward`)
            # live in the internal z-normalised space when the caller
            # supplies znorm. We must undo the transform on BOTH so the
            # RMSE comes out in raw target units. Reported as a bug by
            # Codex on 2026-05-21 — previously only `preds` got inverted,
            # which left the per-epoch LGD RMSE comparing raw against
            # z-normalised values.
            if znorm_mean is not None and znorm_std is not None:
                preds = preds * znorm_std + znorm_mean
            preds_np = preds.detach().cpu().numpy()
            targets_np = _flatten_targets(targets)
            if znorm_mean is not None and znorm_std is not None:
                targets_np = targets_np * znorm_std + znorm_mean
        return float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))

    raise ValueError(f"unsupported regression metric {metric!r}")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def mean_ignore_nan(values: list[float]) -> float:
    """Mean of a list of metric values, ignoring NaNs.

    Returns NaN if every value is NaN (e.g. every val chunk had a
    single-class query split).
    """
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _flatten_logits(logits: torch.Tensor) -> np.ndarray:
    arr = logits.detach().cpu().float().numpy()
    if arr.ndim == 3:
        # (n_query, batch=1, K) → (n_query, K)
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    elif arr.ndim != 2:
        raise ValueError(f"unexpected logits ndim {arr.ndim}")
    return arr


def _flatten_targets(targets: torch.Tensor) -> np.ndarray:
    arr = targets.detach().cpu().float().numpy()
    return arr.reshape(-1)


def _softmax(x: np.ndarray) -> np.ndarray:
    # Numerically stable softmax along last axis.
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)
