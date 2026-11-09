"""Hand-checked expectations for the calibration, bias, NRMSE, NLL and CRPS columns."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    CRPS_QUANTILE_LEVELS,
    _classification_metrics,
    _regression_metrics,
    crps_from_quantiles,
    crps_piecewise_uniform,
    is_uniform_midpoint_grid,
    logistic_recalibration,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrated_sample(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(-1.5, 1.2, n)
    p = _sigmoid(z)
    y = (rng.random(n) < p).astype(int)
    return z, p, y


# --------------------------------------------------------------------------- #
# PD: logistic recalibration and bias
# --------------------------------------------------------------------------- #


def test_perfectly_calibrated_probabilities_give_identity_recalibration():
    _, p, y = _calibrated_sample()
    result = logistic_recalibration(p, y)
    assert result["calibration_slope"] == pytest.approx(1.0, abs=0.02)
    assert result["calibration_intercept"] == pytest.approx(0.0, abs=0.02)
    assert result["calibration_in_the_large"] == pytest.approx(0.0, abs=0.02)


def test_constant_logit_shift_is_recovered_as_intercept():
    # Truth: logit P(y=1) = z.  Model reports logit(p) = z − 0.7 (risk under-predicted).
    z, _, y = _calibrated_sample(seed=1)
    p_model = _sigmoid(z - 0.7)
    result = logistic_recalibration(p_model, y)
    assert result["calibration_slope"] == pytest.approx(1.0, abs=0.02)
    assert result["calibration_intercept"] == pytest.approx(0.7, abs=0.03)
    assert result["calibration_in_the_large"] == pytest.approx(0.7, abs=0.03)


def test_overconfident_model_has_slope_below_one():
    # Model logit = 2·z → true relation logit P = 0.5·logit(p): slope 0.5, intercept 0.
    z, _, y = _calibrated_sample(seed=2)
    result = logistic_recalibration(_sigmoid(2 * z), y)
    assert result["calibration_slope"] == pytest.approx(0.5, abs=0.02)
    assert result["calibration_intercept"] == pytest.approx(0.0, abs=0.03)


def test_recalibration_matches_unpenalized_sklearn_fit():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.6, 3000)
    y = (rng.random(3000) < np.clip(1.3 * p - 0.05, 0, 1)).astype(int)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    ref = LogisticRegression(penalty=None, tol=1e-12, max_iter=10_000).fit(z, y)
    result = logistic_recalibration(p, y)
    assert result["calibration_slope"] == pytest.approx(ref.coef_[0, 0], abs=1e-5)
    assert result["calibration_intercept"] == pytest.approx(ref.intercept_[0], abs=1e-5)


def test_calibration_in_the_large_score_equation():
    # With slope fixed at 1 the MLE satisfies Σ sigmoid(a + z_i) = Σ y_i exactly.
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.5, 500)
    y = (rng.random(500) < 0.3).astype(int)
    a = logistic_recalibration(p, y)["calibration_in_the_large"]
    z = np.log(p / (1 - p))
    assert _sigmoid(a + z).sum() == pytest.approx(y.sum(), rel=1e-9)


def test_constant_prediction_recovers_prevalence_and_leaves_slope_undefined():
    y = np.array([1] * 30 + [0] * 70)
    result = logistic_recalibration(np.full(100, 0.2), y)
    assert np.isnan(result["calibration_slope"]) and np.isnan(result["calibration_intercept"])
    expected = np.log(0.3 / 0.7) - np.log(0.2 / 0.8)
    assert result["calibration_in_the_large"] == pytest.approx(expected, abs=1e-9)


def test_degenerate_targets_and_separation_give_nan():
    assert all(np.isnan(v) for v in logistic_recalibration(np.linspace(.1, .9, 10), np.zeros(10)).values())
    separated = logistic_recalibration(np.array([.1, .2, .3, .7, .8, .9]), np.array([0, 0, 0, 1, 1, 1]))
    assert np.isnan(separated["calibration_slope"])


def test_extreme_probabilities_are_clipped_not_infinite():
    p = np.array([0.0, 1.0, 0.3, 0.6, 0.2, 0.9, 0.4, 0.5])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    result = logistic_recalibration(p, y)
    assert np.isfinite(result["calibration_in_the_large"])


def test_classification_block_emits_bias_and_recalibration_columns():
    _, p, y = _calibrated_sample(n=20_000, seed=5)
    proba = np.column_stack([1 - p, p])
    metrics = _classification_metrics(proba, y, proba, y, 2)
    assert metrics["bias"] == pytest.approx(p.mean() - y.mean(), abs=1e-12)
    assert abs(metrics["bias"]) < 0.01
    assert metrics["calibration_slope"] == pytest.approx(1.0, abs=0.06)
    assert {"calibration_intercept", "calibration_in_the_large"} <= metrics.keys()
    shifted = _classification_metrics(np.column_stack([0.9 - p * 0.9, p * 0.9 + 0.1]), y, proba, y, 2)
    assert shifted["bias"] == pytest.approx(0.1 * (1 - p.mean()) + p.mean() - y.mean(), abs=1e-12)


def test_multiclass_block_reports_nan_calibration():
    rng = np.random.default_rng(6)
    proba = rng.dirichlet(np.ones(3), 60)
    y = rng.integers(0, 3, 60)
    metrics = _classification_metrics(proba, y, proba, y, 3)
    for key in ("bias", "calibration_slope", "calibration_intercept", "calibration_in_the_large"):
        assert np.isnan(metrics[key])


# --------------------------------------------------------------------------- #
# LGD: bias, NRMSE, NLL sign
# --------------------------------------------------------------------------- #


def test_regression_bias_nrmse_and_nll_by_hand():
    y = np.array([0.0, 0.5, 1.0, 0.5])
    pred = np.array([0.1, 0.6, 0.9, 0.8])
    m = _regression_metrics(pred, y, neg_nll=-0.25)
    assert m["bias"] == pytest.approx(pred.mean() - y.mean())                    # 0.1
    rmse = np.sqrt(np.mean((pred - y) ** 2))
    assert m["nrmse"] == pytest.approx(rmse / np.sqrt(0.125))                    # pop. std = √0.125
    assert m["nrmse"] ** 2 == pytest.approx(1 - m["r2"])
    assert m["neg_nll"] == -0.25 and m["nll"] == 0.25
    assert np.isnan(_regression_metrics(pred, y, neg_nll=None)["nll"])


def test_nrmse_of_perfect_prediction_is_zero_and_constant_target_is_nan():
    y = np.array([0.1, 0.4, 0.9])
    m = _regression_metrics(y.copy(), y, neg_nll=None)
    assert m["nrmse"] == 0.0 and m["bias"] == 0.0
    assert np.isnan(_regression_metrics(np.array([.1, .2]), np.array([.5, .5]), neg_nll=None)["nrmse"])


# --------------------------------------------------------------------------- #
# CRPS
# --------------------------------------------------------------------------- #


def _uniform_crps(a, b, y):
    """Closed form for U(a, b): (b−a)·[u³ + (1−u)³]/3 inside, plus the distance outside."""
    w = b - a
    u = np.clip((y - a) / w, 0.0, 1.0)
    inside = w * (u ** 3 + (1 - u) ** 3) / 3.0
    return inside + np.maximum(a - y, 0) + np.maximum(y - b, 0)


def test_crps_uniform_single_bucket_matches_closed_form():
    y = np.array([-1.0, 0.0, 0.2, 0.5, 0.75, 1.0, 3.0])
    got = crps_piecewise_uniform(np.ones((y.size, 1)), np.array([0.0, 1.0]), y)
    np.testing.assert_allclose(got, _uniform_crps(0.0, 1.0, y), atol=1e-12)
    assert got[3] == pytest.approx(1 / 12)                     # centre of U(0,1)
    assert got[0] == pytest.approx(1 + 1 / 3)                  # (a − y) + w/3


def test_crps_uniform_split_into_equal_buckets_is_unchanged():
    y = np.linspace(-0.5, 2.5, 31)
    borders = np.linspace(0.0, 2.0, 9)
    probs = np.full((y.size, 8), 1 / 8)
    np.testing.assert_allclose(crps_piecewise_uniform(probs, borders, y),
                               _uniform_crps(0.0, 2.0, y), atol=1e-12)


def test_crps_point_mass_equals_absolute_error():
    # A zero-width bucket is a point mass at its border: CRPS = |y − m|.
    y = np.array([-2.0, 0.3, 0.3, 5.0])
    borders = np.array([[-1.0, 0.3, 0.3, 4.0]] * 4)
    probs = np.array([[0.0, 1.0, 0.0]] * 4)
    np.testing.assert_allclose(crps_piecewise_uniform(probs, borders, y), np.abs(y - 0.3), atol=1e-12)


def test_crps_histogram_matches_numerical_integral():
    rng = np.random.default_rng(7)
    borders = np.sort(rng.uniform(-1, 3, 12))
    probs = rng.dirichlet(np.ones(11), 5)
    y = rng.uniform(-2, 4, 5)
    grid = np.linspace(-3, 5, 400_001)
    for i in range(5):
        cdf = np.interp(grid, borders, np.concatenate([[0], np.cumsum(probs[i])]))
        integrand = (cdf - (grid >= y[i])) ** 2
        expected = np.trapezoid(integrand, grid)
        got = crps_piecewise_uniform(probs[i:i + 1], borders, y[i:i + 1])[0]
        assert got == pytest.approx(expected, abs=1e-4)


def test_crps_quantile_grid_point_mass_is_exact_mae_and_uniform_is_close():
    levels = np.asarray(CRPS_QUANTILE_LEVELS)
    assert is_uniform_midpoint_grid(levels) and not is_uniform_midpoint_grid((.05, .1, .5, .9, .95))
    y = np.array([-1.0, 0.25, 0.5, 2.0])
    point = np.full((y.size, levels.size), 0.5)
    np.testing.assert_allclose(crps_from_quantiles(point, levels, y), np.abs(y - 0.5), atol=1e-12)
    uniform = np.tile(levels, (y.size, 1))                      # quantiles of U(0, 1)
    np.testing.assert_allclose(crps_from_quantiles(uniform, levels, y), _uniform_crps(0, 1, y), atol=5e-5)


def test_regression_block_records_crps_method():
    y = np.array([0.2, 0.5, 0.9])
    pred = np.array([0.3, 0.5, 0.7])
    levels = np.asarray(CRPS_QUANTILE_LEVELS)

    # No distribution → NaN, method "none"; the sparse reporting grid is not used.
    sparse = (.05, .10, .25, .40, .50, .60, .75, .90, .95)
    m = _regression_metrics(pred, y, neg_nll=None,
                            quantiles=np.tile(sparse, (3, 1)), quantile_levels=sparse)
    assert np.isnan(m["crps"]) and m["crps_method"] == "none"

    # Dense quantiles of a point mass at the prediction → CRPS = MAE.
    m = _regression_metrics(pred, y, neg_nll=None, crps_quantiles=np.repeat(pred[:, None], levels.size, 1))
    assert m["crps_method"] == "quantile_midpoint_grid_100"
    assert m["crps"] == pytest.approx(m["mae"])

    # The reporting grid itself is used when it is a dense midpoint grid.
    m = _regression_metrics(pred, y, neg_nll=None,
                            quantiles=np.tile(levels, (3, 1)), quantile_levels=tuple(levels))
    assert m["crps_method"] == "quantile_midpoint_grid_100"

    # Bucket probabilities win over quantiles.
    m = _regression_metrics(pred, y, neg_nll=None, crps_quantiles=np.tile(levels, (3, 1)),
                            bucket_probabilities=np.ones((3, 1)), bucket_borders=np.array([0., 1.]))
    assert m["crps_method"] == "bar_distribution_piecewise_uniform"
    assert m["crps"] == pytest.approx(_uniform_crps(0, 1, y).mean())


def test_crps_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        crps_piecewise_uniform(np.ones((2, 2)) / 2, np.array([0., 1.]), np.zeros(2))
    with pytest.raises(ValueError):
        crps_piecewise_uniform(np.ones((1, 2)) / 2, np.array([1., 0., 2.]), np.zeros(1))
    with pytest.raises(ValueError):
        crps_from_quantiles(np.zeros((1, 9)), (.05, .1, .25, .4, .5, .6, .75, .9, .95), np.zeros(1))
