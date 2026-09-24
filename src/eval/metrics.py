"""Shared monitoring and held-out scoring metrics; no benchmark orchestration."""
from __future__ import annotations

import numpy as np
import json


def calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> str:
    """Sufficient statistics for pooled reliability plots; identical bins for all models."""
    index = np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    return json.dumps([{"bin": b, "count": int((index == b).sum()),
        "probability_sum": float(np.asarray(p)[index == b].sum()),
        "positive_count": int((np.asarray(y)[index == b] == 1).sum())}
        for b in range(n_bins)], separators=(",", ":"))


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
        roc_auc_score, log_loss, average_precision_score,
        f1_score, accuracy_score, precision_score, recall_score,
        brier_score_loss, matthews_corrcoef, balanced_accuracy_score,
        cohen_kappa_score, confusion_matrix,
    )
    out: dict[str, float] = {}
    K = proba_test.shape[1]
    if K == 2:
        p = np.clip(proba_test[:, 1], 1e-15, 1 - 1e-15)
        out.update(f1_at_05=float(f1_score(y_test, proba_test[:, 1] >= .5, zero_division=0)),
            prediction_mean=float(p.mean()), prediction_std=float(p.std()),
            prediction_entropy=float(-(p * np.log(p) + (1-p) * np.log1p(-p)).mean()),
            target_prevalence=float(np.mean(np.asarray(y_test) == 1)),
            calibration_bins=calibration_bins(proba_test[:, 1], y_test))

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
        from sklearn.metrics import brier_score_loss, log_loss as _ll
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
) -> dict[str, float]:
    """Full regression-metric block.

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
        mean_squared_error, mean_absolute_error, r2_score,
        median_absolute_error, explained_variance_score,
    )
    out: dict[str, float] = {
        "rmse":               float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "mae":                float(mean_absolute_error(y_test, pred_test)),
        "median_ae":          float(median_absolute_error(y_test, pred_test)),
        "r2":                 float(r2_score(y_test, pred_test)),
        "explained_variance": float(explained_variance_score(y_test, pred_test)),
        "neg_nll":            float("nan") if neg_nll is None else float(neg_nll),
    }

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
