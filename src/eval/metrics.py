"""Shared monitoring and held-out scoring metrics; no benchmark orchestration."""
from __future__ import annotations

import json

import numpy as np


def calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> str:
    """Sufficient statistics for pooled reliability plots; identical bins for all models."""
    index = np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    return json.dumps([{"bin": b, "count": int((index == b).sum()),
        "probability_sum": float(np.asarray(p)[index == b].sum()),
        "positive_count": int((np.asarray(y)[index == b] == 1).sum())}
        for b in range(n_bins)], separators=(",", ":"))


#: Probabilities are clipped to this range before taking log-odds for the
#: logistic-recalibration diagnostics (calibration slope / intercept / in-the-large).
CALIBRATION_LOGIT_CLIP = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), CALIBRATION_LOGIT_CLIP, 1 - CALIBRATION_LOGIT_CLIP)
    return np.log(p) - np.log1p(-p)


def _logistic_irls(
    design: np.ndarray, y: np.ndarray, offset: np.ndarray, start: np.ndarray,
    *, max_iter: int = 100, tol: float = 1e-10,
) -> np.ndarray | None:
    """Unpenalized maximum-likelihood logistic regression by damped Newton (IRLS).

    Maximises Σ y·η − log(1 + e^η) with η = offset + design @ β. Returns None when
    the fit does not converge or the estimate diverges (complete separation,
    a constant predictor, a one-class target), so callers record NaN rather than
    an arbitrary large number.
    """
    y = np.asarray(y, dtype=np.float64)

    def loglik(beta):
        eta = offset + design @ beta
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    beta = np.asarray(start, dtype=np.float64).copy()
    current = loglik(beta)
    for _ in range(max_iter):
        eta = offset + design @ beta
        mu = 0.5 * (1.0 + np.tanh(0.5 * eta))           # overflow-free sigmoid
        weight = mu * (1.0 - mu)
        gradient = design.T @ (y - mu)
        hessian = design.T @ (design * weight[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(step).all():
            return None
        scale = 1.0
        while scale > 1e-8:                               # step halving keeps ascent monotone
            candidate = beta + scale * step
            value = loglik(candidate)
            if np.isfinite(value) and value >= current - 1e-12:
                break
            scale *= 0.5
        else:
            return None
        beta, current = candidate, value
        if np.max(np.abs(scale * step)) < tol * (1.0 + np.max(np.abs(beta))):
            # A slowly drifting estimate under separation also takes tiny steps;
            # treat an implausibly large coefficient as divergence.
            return beta if np.max(np.abs(beta)) < 1e3 else None
    return None


def logistic_recalibration(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Logistic-recalibration diagnostics for binary probabilities (Cox 1958).

    With z = logit(clip(p, 1e-6, 1 − 1e-6)) and y ∈ {0, 1}, by unpenalized
    maximum likelihood:

    * ``calibration_slope`` = b and ``calibration_intercept`` = a from the joint fit
      logit P(y = 1) = a + b·z. A perfectly calibrated model has b = 1, a = 0;
      b < 1 indicates over-confident (too extreme) predictions, b > 1 under-confident.
    * ``calibration_in_the_large`` = a from logit P(y = 1) = a + 1·z (slope fixed at
      one; z enters as an offset). a > 0 means risk is under-predicted on average,
      a < 0 over-predicted. Note the joint-fit intercept is NOT this quantity
      unless b = 1.

    Every value is NaN for a one-class target, and the slope/intercept are NaN
    when z is constant or the outcome is completely separated by z (the MLE then
    does not exist).
    """
    out = {"calibration_slope": float("nan"), "calibration_intercept": float("nan"),
           "calibration_in_the_large": float("nan")}
    y = (np.asarray(y).reshape(-1) == 1).astype(np.float64)
    z = _logit(np.asarray(p).reshape(-1))
    if y.size == 0 or y.min() == y.max() or not np.isfinite(z).all():
        return out
    ones = np.ones((y.size, 1))
    joint = _logistic_irls(np.column_stack([ones[:, 0], z]), y, np.zeros_like(z),
                           np.array([0.0, 1.0]))
    if joint is not None:
        out["calibration_intercept"], out["calibration_slope"] = float(joint[0]), float(joint[1])
    citl = _logistic_irls(ones, y, z, np.array([0.0]))
    if citl is not None:
        out["calibration_in_the_large"] = float(citl[0])
    return out


# --------------------------------------------------------------------------- #
# Continuous ranked probability score (regression)
# --------------------------------------------------------------------------- #

#: Dense uniform midpoint grid τ_i = (i − ½)/m, m = 100, for the quantile-based CRPS
#: approximation. The midpoint rule makes the approximation exact for a point mass.
CRPS_QUANTILE_GRID_SIZE = 100
CRPS_QUANTILE_LEVELS: tuple[float, ...] = tuple(
    (i + 0.5) / CRPS_QUANTILE_GRID_SIZE for i in range(CRPS_QUANTILE_GRID_SIZE))
#: Fewer midpoint levels than this are too sparse to call a CRPS estimate.
CRPS_MIN_QUANTILE_LEVELS = 19


def _segment_squared_integral(width, f0, f1, h):
    """∫ (F(x) − h)² dx over a segment where F is linear from f0 to f1 and h is constant."""
    a, b = f0 - h, f1 - h
    return width * (a * a + a * b + b * b) / 3.0


def crps_piecewise_uniform(
    probabilities: np.ndarray, borders: np.ndarray, y: np.ndarray, *, chunk_rows: int = 512,
) -> np.ndarray:
    """Exact per-row CRPS of a histogram (piecewise-uniform) predictive distribution.

    ``probabilities`` has shape (n, K) with rows summing to one; ``borders`` has
    shape (K + 1,) (shared) or (n, K + 1), strictly non-decreasing. Bucket k carries
    mass p_k spread uniformly over [b_k, b_{k+1}], so the CDF F is piecewise linear and

        CRPS(F, y) = ∫ (F(x) − 1{x ≥ y})² dx

    is a sum of closed-form segment integrals (the bucket containing y is split at y),
    plus |y − b_0| when y < b_0 and |y − b_K| when y > b_K. Zero-width buckets are
    treated as point masses at their border.

    Approximation note: TabPFN's ``FullSupportBarDistribution`` uses half-normal tails
    in its two outermost buckets; this function treats every bucket, including the
    outer two, as uniform between its borders, so for TabPFN it is exact for the
    histogram approximation of the bar distribution, not for its half-normal tails.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    borders = np.asarray(borders, dtype=np.float64)
    n, k = probabilities.shape
    if y.size != n:
        raise ValueError("CRPS: probabilities and targets differ in length")
    if borders.ndim == 1:
        borders = np.broadcast_to(borders, (n, k + 1))
    if borders.shape != (n, k + 1):
        raise ValueError("CRPS: borders must have one more entry than buckets")
    if (not np.isfinite(probabilities).all() or not np.isfinite(borders).all()
            or (probabilities < 0).any() or (np.diff(borders, axis=1) < 0).any()):
        raise ValueError("CRPS: invalid bucket probabilities or borders")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    result = np.empty(n)
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        p, b, t = probabilities[rows], borders[rows], y[rows, None]
        lo, hi = b[:, :-1], b[:, 1:]
        width = hi - lo
        cdf_lo = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)[:, :-1]], axis=1)
        cdf_hi = cdf_lo + p
        split = np.clip(t, lo, hi)                            # y clamped into each bucket
        with np.errstate(invalid="ignore", divide="ignore"):
            cdf_split = np.where(width > 0, cdf_lo + p * (split - lo) / width, cdf_lo)
        below = _segment_squared_integral(split - lo, cdf_lo, cdf_split, 0.0)
        above = _segment_squared_integral(hi - split, cdf_split, cdf_hi, 1.0)
        outside = np.maximum(b[:, 0] - t[:, 0], 0.0) + np.maximum(t[:, 0] - b[:, -1], 0.0)
        result[rows] = (below + above).sum(axis=1) + outside
    return result


def crps_from_quantiles(quantiles: np.ndarray, levels, y: np.ndarray) -> np.ndarray:
    """Per-row CRPS approximation from predicted quantiles on a uniform midpoint grid.

    Uses CRPS(F, y) = 2 ∫₀¹ ρ_τ(y − F⁻¹(τ)) dτ, with ρ_τ the pinball loss, evaluated by
    the midpoint rule: 2 × mean_i ρ_{τ_i}(y − q_i) over τ_i = (i − ½)/m. Requires the
    levels to be exactly such a grid with m ≥ ``CRPS_MIN_QUANTILE_LEVELS``.
    """
    levels = np.asarray(levels, dtype=np.float64)
    if not is_uniform_midpoint_grid(levels):
        raise ValueError("CRPS from quantiles needs a dense uniform midpoint grid")
    q = np.asarray(quantiles, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if q.shape != (y.size, levels.size) or not np.isfinite(q).all():
        raise ValueError("Invalid quantile matrix for CRPS")
    residual = y[:, None] - q
    return 2.0 * np.maximum(levels * residual, (levels - 1.0) * residual).mean(axis=1)


def is_uniform_midpoint_grid(levels) -> bool:
    levels = np.asarray(levels, dtype=np.float64).reshape(-1)
    m = levels.size
    return bool(m >= CRPS_MIN_QUANTILE_LEVELS
                and np.allclose(levels, (np.arange(m) + 0.5) / m, rtol=0, atol=1e-9))


def _best_f1_threshold(
    proba_val_pos: np.ndarray, y_val: np.ndarray,
) -> float:
    """Return the threshold τ ∈ [0,1] that maximises F1 on the val set.

    Uses ``sklearn.metrics.precision_recall_curve`` so we evaluate F1
    only at the O(n) breakpoints sklearn returns after sorting predicted scores.
    """
    # A one-class validation fold has no meaningful ranking threshold and
    # sklearn warns for the all-negative case. Use the documented neutral
    # fallback directly; the outer metric code already handles one-class folds.
    if np.unique(np.asarray(y_val)).size < 2:
        return 0.5

    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(
        y_val, proba_val_pos,
    )
    # precision_recall_curve returns one fewer threshold than (p, r);
    # match them by dropping the final p/r pair (which has threshold = ∞).
    p, r = precisions[:-1], recalls[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / (p + r)
        f1 = np.nan_to_num(f1, nan=0.0)
    if len(thresholds) == 0 or f1.max() <= 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def _binary_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """10-bin expected calibration error for binary positive-class probs."""
    try:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(p, bins[1:-1], right=True)
        N = len(p)
        ece = 0.0
        for b in range(n_bins):
            mask = idx == b
            nb = int(mask.sum())
            if nb == 0:
                continue
            ece += (nb / N) * abs(float((y[mask] == 1).mean()) - float(p[mask].mean()))
        return float(ece)
    except (ValueError, IndexError):                              # pragma: no cover
        return float("nan")


def _posthoc_calibrated(
    proba_test: np.ndarray, proba_val: np.ndarray, y_val: np.ndarray,
    method: str,
) -> np.ndarray | None:
    """Fit a post-hoc probability calibrator on the INNER-VAL split and
    apply it to the test probabilities. Binary only; returns None if it
    cannot be fitted.

    The model itself is never refitted or retrained — only its output
    probabilities are remapped, which is what makes this "post-hoc" and
    cheap. Fitting on inner-val (never on test) keeps the test fold clean.

    * ``platt``    — logistic regression on the log-odds (Platt scaling).
      A smooth, 2-parameter monotone squash; robust on small validation
      splits, but can only fix a systematic over/under-confidence.
    * ``isotonic`` — non-parametric monotone fit. Strictly more flexible,
      so it can correct odd calibration curves, but it overfits a small
      validation split and produces a step function.

    """
    if proba_test.shape[1] != 2 or len(np.unique(y_val)) < 2:
        return None
    pv = np.clip(proba_val[:, 1], 1e-6, 1 - 1e-6)
    pt = np.clip(proba_test[:, 1], 1e-6, 1 - 1e-6)
    try:
        if method == "platt":
            from sklearn.linear_model import LogisticRegression
            z = np.log(pv / (1 - pv)).reshape(-1, 1)
            lr = LogisticRegression(C=1e10, solver="lbfgs")
            lr.fit(z, y_val)
            zt = np.log(pt / (1 - pt)).reshape(-1, 1)
            return lr.predict_proba(zt)[:, 1]
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pv, y_val)
            return np.clip(iso.predict(pt), 0.0, 1.0)
    except (ValueError, RuntimeError):                            # pragma: no cover
        return None
    return None


def _classification_metrics(
    proba_test: np.ndarray, y_test: np.ndarray,
    proba_val: np.ndarray, y_val: np.ndarray,
    n_classes_seen: int,
) -> dict:
    """All classification metrics in one place. F1 / accuracy /
    precision / recall use the threshold that MAXIMISES F1 on the
    inner-validation split (binary only); multiclass returns NaN
    for those four columns.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    out: dict[str, float] = {}
    K = proba_test.shape[1]
    if K == 2:
        # In float32, 1 - 1e-15 rounds to one and entropy evaluates 0 * log(0).
        # Promote before clipping; keep model probabilities unchanged for scoring.
        p = np.clip(np.asarray(proba_test[:, 1], dtype=np.float64), 1e-15, 1 - 1e-15)
        out.update(f1_at_05=float(f1_score(y_test, proba_test[:, 1] >= .5, zero_division=0)),
            prediction_mean=float(p.mean()), prediction_std=float(p.std()),
            prediction_entropy=float(-(p * np.log(p) + (1-p) * np.log1p(-p)).mean()),
            target_prevalence=float(np.mean(np.asarray(y_test) == 1)),
            calibration_bins=calibration_bins(proba_test[:, 1], y_test),
            # Calibration-in-the-large on the probability scale: mean(p) − mean(y),
            # with the unclipped probabilities. Positive = risk over-predicted.
            bias=float(np.mean(np.asarray(proba_test[:, 1], dtype=np.float64))
                       - np.mean(np.asarray(y_test) == 1)),
            **logistic_recalibration(proba_test[:, 1], y_test))
    else:
        out.update(bias=float("nan"), calibration_slope=float("nan"),
                   calibration_intercept=float("nan"), calibration_in_the_large=float("nan"))

    # Threshold-free metrics on test fold.
    try:
        if K == 2:
            out["roc_auc"] = float(roc_auc_score(y_test, proba_test[:, 1]))
            out["pr_auc"]  = float(average_precision_score(y_test, proba_test[:, 1]))
        elif n_classes_seen >= 2:
            out["roc_auc"] = float(
                roc_auc_score(y_test, proba_test, multi_class="ovr", average="macro")
            )
            out["pr_auc"]  = float("nan")
        else:
            out["roc_auc"] = float("nan")
            out["pr_auc"]  = float("nan")
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"]  = float("nan")
    try:
        out["log_loss"] = float(
            log_loss(y_test, proba_test, labels=list(range(K)))
        )
    except ValueError:
        out["log_loss"] = float("nan")

    # Brier score on positive-class probability (binary only). Proper
    # score, lower = better; combines calibration + sharpness.
    if K == 2:
        try:
            out["brier_score"] = float(brier_score_loss(y_test, proba_test[:, 1]))
        except ValueError:
            out["brier_score"] = float("nan")
    else:
        out["brier_score"] = float("nan")

    # Expected Calibration Error (binary): 10 equal-width confidence bins on
    # the positive-class probability. ECE = Σ_b (n_b/N)·|acc_b − conf_b|;
    # lower = better calibrated. Distinct from Brier (which mixes calibration
    # and sharpness) — ECE isolates calibration, which Basel-III PD models
    # require and which Tanna et al. 2026 show fine-tuning can silently
    # degrade even when ROC-AUC holds.
    if K == 2:
        try:
            p = np.asarray(proba_test[:, 1], dtype=float)
            yt = np.asarray(y_test).astype(int)
            n_bins = 10
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
            N = len(p)
            ece = 0.0
            for b in range(n_bins):
                mask = bin_idx == b
                nb = int(mask.sum())
                if nb == 0:
                    continue
                conf = float(p[mask].mean())
                acc = float((yt[mask] == 1).mean())
                ece += (nb / N) * abs(acc - conf)
            out["ece"] = float(ece)
        except (ValueError, IndexError):
            out["ece"] = float("nan")
    else:
        out["ece"] = float("nan")

    # POST-HOC CALIBRATION (binary only). Recorded ALONGSIDE the raw metrics,
    # never replacing them, so every row carries the before/after pair and the
    # ablation is a column subtraction rather than a second run.
    if K == 2:
        from sklearn.metrics import brier_score_loss
        from sklearn.metrics import log_loss as _ll
        for method in ("platt", "isotonic"):
            out[f"optimal_threshold_{method}"] = float("nan")
            out[f"calibration_bins_{method}"] = "[]"
            out[f"roc_auc_{method}"] = float("nan")
            cal = _posthoc_calibrated(proba_test, proba_val, y_val, method)
            if cal is None:
                out[f"ece_{method}"] = float("nan")
                out[f"brier_score_{method}"] = float("nan")
                out[f"log_loss_{method}"] = float("nan")
                out[f"f1_{method}"] = float("nan")
                if method == "isotonic":
                    out["roc_auc_isotonic"] = float("nan")
                continue
            out[f"ece_{method}"] = _binary_ece(cal, np.asarray(y_test))
            out[f"calibration_bins_{method}"] = calibration_bins(cal, y_test)
            try:
                out[f"brier_score_{method}"] = float(
                    brier_score_loss(y_test, cal))
            except ValueError:                                    # pragma: no cover
                out[f"brier_score_{method}"] = float("nan")
            try:
                out[f"log_loss_{method}"] = float(
                    _ll(y_test, np.column_stack([1 - cal, cal]), labels=[0, 1]))
            except ValueError:                                    # pragma: no cover
                out[f"log_loss_{method}"] = float("nan")

            # F1 AFTER recalibration, with the threshold RE-TUNED on the recalibrated
            # validation probabilities. Reusing the raw threshold would measure the
            # calibrator against a cut-off chosen for a different probability scale, which
            # flatters or punishes it arbitrarily.
            cal_val = _posthoc_calibrated(proba_val, proba_val, y_val, method)
            try:
                if cal_val is not None and len(np.unique(y_val)) >= 2:
                    th = _best_f1_threshold(cal_val, np.asarray(y_val))
                    out[f"optimal_threshold_{method}"] = th
                    from sklearn.metrics import f1_score
                    out[f"f1_{method}"] = float(
                        f1_score(y_test, (cal >= th).astype(int), zero_division=0))
                else:
                    out[f"f1_{method}"] = float("nan")
            except Exception:                                     # pragma: no cover
                out[f"f1_{method}"] = float("nan")

            # Platt's fitted slope may be negative; isotonic may create ties.
            try:
                out[f"roc_auc_{method}"] = float(roc_auc_score(y_test, cal))
            except ValueError:                                    # pragma: no cover
                pass
    else:
        for method in ("platt", "isotonic"):
            out[f"ece_{method}"] = float("nan")
            out[f"brier_score_{method}"] = float("nan")
            out[f"log_loss_{method}"] = float("nan")
            out[f"f1_{method}"] = float("nan")
        out["roc_auc_isotonic"] = float("nan")

    # Threshold-tuned metrics — binary only.
    if K == 2 and len(np.unique(y_val)) >= 2:
        best_th = _best_f1_threshold(proba_val[:, 1], y_val)
        out["optimal_threshold"] = best_th
        preds_t = (proba_test[:, 1] >= best_th).astype(int)
        out["f1"]        = float(f1_score(y_test, preds_t, zero_division=0))
        out["accuracy"]  = float(accuracy_score(y_test, preds_t))
        out["precision"] = float(precision_score(y_test, preds_t, zero_division=0))
        out["recall"]    = float(recall_score(y_test, preds_t, zero_division=0))

        # Specificity = TN / (TN + FP). Companion to recall (TPR).
        try:
            cm = confusion_matrix(y_test, preds_t, labels=[0, 1])
            tn, fp = float(cm[0, 0]), float(cm[0, 1])
            out.update(true_negative=tn, false_positive=fp,
                       false_negative=float(cm[1, 0]), true_positive=float(cm[1, 1]))
            out["specificity"] = (
                float("nan") if (tn + fp) == 0.0 else tn / (tn + fp)
            )
        except ValueError:
            out["specificity"] = float("nan")

        out["balanced_accuracy"] = float(balanced_accuracy_score(y_test, preds_t))

        # MCC — handles imbalance gracefully; undefined when ANY of
        # TP/TN/FP/FN combinations make the denominator zero, in which
        # case sklearn returns 0 and emits a runtime warning. We accept
        # the sklearn behaviour (returns 0 rather than NaN) since the
        # warning is suppressed at the loop level.
        out["mcc"] = float(matthews_corrcoef(y_test, preds_t))

        out["cohen_kappa"] = float(cohen_kappa_score(y_test, preds_t))
    else:
        for k in ("optimal_threshold", "f1", "accuracy", "precision", "recall",
                  "specificity", "balanced_accuracy", "mcc", "cohen_kappa"):
            out[k] = float("nan")

    return out


def _regression_metrics(
    pred_test: np.ndarray, y_test: np.ndarray,
    *, neg_nll: float | None, quantiles: np.ndarray | None = None,
    quantile_levels: tuple[float, ...] = (),
    bucket_probabilities: np.ndarray | None = None,
    bucket_borders: np.ndarray | None = None,
    crps_quantiles: np.ndarray | None = None,
    crps_quantile_levels: tuple[float, ...] = CRPS_QUANTILE_LEVELS,
) -> dict[str, float]:
    """Full regression-metric block.

    Sign convention of the density columns. ``neg_nll`` is the name the
    campaign has always used for the mean LOG-LIKELIHOOD of the targets
    under the predictive density, i.e. −NLL (higher = better; per-row NLL
    is clamped to [−100, 100] nats in ``src.model.tabpfn_models``). The
    name is kept for downstream compatibility. ``nll`` = −``neg_nll`` is
    the same quantity with the conventional sign (lower = better).

    Additional calibration / scale-free / probabilistic columns:

    * ``bias``  — mean(prediction) − mean(y). Positive = LGD over-predicted.
    * ``nrmse`` — RMSE / std(y) with the population std (ddof = 0), so it is
      comparable across tables with different target scales; nrmse² = 1 − R².
      NaN when y is constant.
    * ``crps`` / ``crps_method`` — mean continuous ranked probability score
      (lower = better, in target units; equals MAE for a point forecast).
      Methods, in order of preference:
        - ``bar_distribution_piecewise_uniform``: exact for a histogram density,
          from ``bucket_probabilities`` (n, K) and ``bucket_borders`` (K + 1,)
          — see :func:`crps_piecewise_uniform`;
        - ``quantile_midpoint_grid_<m>``: 2 × mean pinball loss over a dense
          uniform midpoint grid (``crps_quantiles`` at ``crps_quantile_levels``,
          or ``quantiles`` when ``quantile_levels`` itself is such a grid) —
          see :func:`crps_from_quantiles`;
        - ``none``: no predictive distribution; ``crps`` is NaN. The sparse
          nine-level reporting grid is deliberately not used for CRPS.

    On top of RMSE / MAE / R² / neg-NLL we emit:

    * ``median_ae``           — outlier-robust point error.
    * ``mape``                — mean absolute percentage error in
      decimal units (NOT multiplied by 100; multiply downstream if you
      want %). NaN when any target equals zero — divide-by-zero would
      poison the mean.
    * ``explained_variance``  — sklearn's ``explained_variance_score``;
      equals R² when prediction is unbiased.
    * ``pearson_r``           — linear correlation. NaN on a constant
      target or constant prediction.
    * ``spearman_r``          — rank correlation; robust to monotone
      nonlinearities. NaN under the same degenerate cases.
    """
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    out: dict[str, float] = {
        "rmse":               float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "mae":                float(mean_absolute_error(y_test, pred_test)),
        "median_ae":          float(median_absolute_error(y_test, pred_test)),
        "r2":                 float(r2_score(y_test, pred_test)),
        "explained_variance": float(explained_variance_score(y_test, pred_test)),
        "neg_nll":            float("nan") if neg_nll is None else float(neg_nll),
        "nll":                float("nan") if neg_nll is None else -float(neg_nll),
    }
    y_arr = np.asarray(y_test, dtype=np.float64).reshape(-1)
    pred_arr = np.asarray(pred_test, dtype=np.float64).reshape(-1)
    out["bias"] = float(pred_arr.mean() - y_arr.mean())
    # Spread of the predictions relative to the targets: a collapsed regressor predicts
    # nearly one value (prediction_std -> 0, std_ratio -> 0) while its RMSE can stay plausible.
    out["prediction_mean"] = float(pred_arr.mean())
    out["prediction_std"] = float(pred_arr.std())
    y_std = float(y_arr.std())
    out["prediction_std_ratio"] = out["prediction_std"] / y_std if y_std > 0 else float("nan")
    out["nrmse"] = out["rmse"] / y_std if y_std > 0 else float("nan")

    out["crps"], out["crps_method"] = float("nan"), "none"
    if bucket_probabilities is not None and bucket_borders is not None:
        out["crps"] = float(crps_piecewise_uniform(bucket_probabilities, bucket_borders, y_arr).mean())
        out["crps_method"] = "bar_distribution_piecewise_uniform"
    else:
        dense = ((crps_quantiles, crps_quantile_levels) if crps_quantiles is not None
                 else (quantiles, quantile_levels) if quantiles is not None
                 and is_uniform_midpoint_grid(quantile_levels) else None)
        if dense is not None:
            out["crps"] = float(crps_from_quantiles(dense[0], dense[1], y_arr).mean())
            out["crps_method"] = f"quantile_midpoint_grid_{len(dense[1])}"

    # MAPE — undefined where any target is zero; we emit NaN rather
    # than +inf so the column aggregates cleanly across folds.
    if np.any(y_test == 0):
        out["mape"] = float("nan")
    else:
        out["mape"] = float(
            np.mean(np.abs((y_test - pred_test) / y_test))
        )

    # Pearson / Spearman — guard against constant-vector inputs which
    # make the correlation undefined (the denominator is zero).
    if np.ptp(y_test) == 0 or np.ptp(pred_test) == 0:
        out["pearson_r"] = float("nan")
    else:
        out["pearson_r"] = float(np.corrcoef(y_test, pred_test)[0, 1])

    try:
        from scipy.stats import spearmanr
        if np.ptp(y_test) == 0 or np.ptp(pred_test) == 0:
            out["spearman_r"] = float("nan")
        else:
            rho, _ = spearmanr(y_test, pred_test)
            out["spearman_r"] = float(rho) if not np.isnan(rho) else float("nan")
    except ImportError:                                                # pragma: no cover
        out["spearman_r"] = float("nan")

    if quantiles is not None:
        q = np.asarray(quantiles)
        levels = np.asarray(quantile_levels)
        if q.shape != (len(y_test), len(levels)) or not np.isfinite(q).all():
            raise ValueError("Invalid predictive quantile matrix")
        residual = np.asarray(y_test)[:, None] - q
        out["mean_pinball_loss"] = float(np.maximum(levels * residual, (levels - 1) * residual).mean())
        out["quantile_crossing_fraction"] = float(np.any(np.diff(q, axis=1) < 0, axis=1).mean())
        for coverage, lo, hi in ((80, .10, .90), (90, .05, .95)):
            if lo in quantile_levels and hi in quantile_levels:
                lower, upper = q[:, quantile_levels.index(lo)], q[:, quantile_levels.index(hi)]
                out[f"interval_coverage_{coverage}"] = float(((y_test >= lower) & (y_test <= upper)).mean())
                out[f"interval_width_{coverage}"] = float((upper - lower).mean())

    return out
