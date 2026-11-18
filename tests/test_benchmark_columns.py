"""Every metric the evaluation computes has a column in the five-fold benchmark rows.

The benchmark copies a metric into ``EvalRow`` only when a field of that name exists, so a metric
added to ``src.eval.metrics`` without a field was silently dropped (gini, ks, h_measure, emp,
kendall_tau, prediction_std_ratio until 18-11-2026)."""
import numpy as np

from src.eval import metrics as m
from src.eval.benchmark import EvalRow


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
