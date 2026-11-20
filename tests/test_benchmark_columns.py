"""Every metric the evaluation computes has a column in the five-fold benchmark rows.

The benchmark copies a metric into ``EvalRow`` only when a field of that name exists, so a metric
added to ``src.eval.metrics`` without a field was silently dropped (gini, ks, h_measure, emp,
kendall_tau, prediction_std_ratio until 18-11-2026)."""
import numpy as np
import pytest

from src.eval import metrics as m
from src.eval.benchmark import EvalRow


def test_classification_rejects_unnormalized_validation_probabilities():
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="row totals"):
        m._classification_metrics(np.eye(2), y, np.full((2, 2), 0.8), y, 2)


def test_density_counts_include_finite_clipping_and_nonfinite_rows():
    import torch

    from src.model.tabpfn_models import _tabpfn_regression_neg_nll
    values = torch.tensor([1.0, 153.0, -101.0, float("nan"), float("inf")])
    output = {"criterion": lambda logits, target: values, "logits": values}
    counts = {}
    mean = _tabpfn_regression_neg_nll(None, None, np.zeros(5), output=output, diagnostics=counts)
    assert mean == pytest.approx(-40.2)
    assert counts == {"nll_rows": 5, "nll_clipped_rows": 4, "nll_nonfinite_rows": 2}
    assert set(counts) <= set(EvalRow.__dataclass_fields__)


def test_every_classification_and_regression_metric_has_a_benchmark_column():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(rng.uniform(size=400) * 0.5 + y * 0.3, 0.01, 0.99)
    proba = np.stack([1 - p, p], axis=1)
    cls = m._classification_metrics(proba_test=proba, y_test=y, proba_val=proba, y_val=y, n_classes_seen=2)
    target = rng.uniform(size=400)
    reg = m._regression_metrics(pred_test=target + rng.normal(0, 0.1, 400), y_test=target, neg_nll=None)
    fields = set(EvalRow.__dataclass_fields__)
    assert set(cls) <= fields, sorted(set(cls) - fields)
    assert set(reg) <= fields, sorted(set(reg) - fields)
    for name in ("gini", "ks", "h_measure", "emp", "emp_rejection_rate"):
        assert np.isfinite(cls[name])
    for name in ("kendall_tau", "prediction_std_ratio"):
        assert np.isfinite(reg[name])
