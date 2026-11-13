"""E5 target fine-tuning: upstream fine-tuners run end to end and never see test labels."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow
CASES = [("pd", "tabpfn-v2-classifier-v2_default.ckpt", "tabpfn"),
         ("lgd", "tabpfn-v3-regressor-v3_default.ckpt", "tabpfn"),
         ("pd", "tabicl-classifier-v2-20260212.ckpt", "tabicl"),
         ("lgd", "tabicl-regressor-v2-20260212.ckpt", "tabicl")]


def _smallest(track):
    from src.eval.dataset_loader import load_processed_dataset
    from src.train.corpus import build_dataset_pool
    pool = build_dataset_pool(track)
    if not pool:
        pytest.skip("processed data unavailable")
    ref = min(pool, key=lambda r: r.n_rows)
    return load_processed_dataset(track, ref.dataset_id)


@pytest.mark.parametrize("track, name, family", CASES)
def test_finetune_cell_runs_and_returns_finite_scores(track, name, family):
    from src.eval.target_finetune import finetune_cell
    path = ROOT / "checkpoints" / name
    if not path.is_file():
        pytest.skip("checkpoint unavailable")
    data = _smallest(track)
    idx = np.random.default_rng(0).permutation(len(data.y))[:500]
    context, validation, test = idx[:300], idx[300:380], idx[380:]
    metrics, predictions = finetune_cell(data, context, validation, test, str(path), family,
                                         recipe={"epochs": 1}, device="cpu", views=2)
    key = "roc_auc" if track == "pd" else "r2"
    assert np.isfinite(metrics[key])
    assert len(predictions["prediction"]) == len(test)
    assert metrics["context_rows"] == 300 and metrics["test_rows"] == len(test)


def test_test_labels_are_not_passed_to_fit(monkeypatch):
    """The fine-tuner sees context rows (fit) and validation rows (early stopping) only."""
    import src.eval.target_finetune as tf
    seen = {}

    class Spy:
        def fit(self, X, y, X_val=None, y_val=None, output_dir=None):
            seen["fit_rows"], seen["val_rows"] = len(y), len(y_val)

        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

    monkeypatch.setattr(tf, "_tabpfn_finetuner", lambda *a, **k: Spy())
    from types import SimpleNamespace

    import pandas as pd
    data = SimpleNamespace(X=pd.DataFrame({"a": np.arange(100.0)}), y=np.arange(100) % 2,
                           categorical_columns=[], task_type="classification")
    tf.finetune_cell(data, np.arange(60), np.arange(60, 80), np.arange(80, 100), "x", "tabpfn", device="cpu")
    assert seen == {"fit_rows": 60, "val_rows": 20}


def test_unknown_family_is_rejected():
    from types import SimpleNamespace

    import pandas as pd

    from src.eval.target_finetune import finetune_cell
    data = SimpleNamespace(X=pd.DataFrame({"a": [1.0, 2.0, 3.0]}), y=np.array([0, 1, 0]),
                           categorical_columns=[], task_type="classification")
    with pytest.raises(ValueError):
        finetune_cell(data, np.array([0]), np.array([1]), np.array([2]), "x", "xgboost")


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_tabicl_finetuned_inference_is_pinned_to_fp32(task_type):
    """``amp=False`` covers fine-tuning only; the inner inference estimator must not resolve AMP/FA3."""
    import torch

    from src.eval.target_finetune import _tabicl_finetuner
    est = _tabicl_finetuner(task_type, "unused.ckpt", recipe={"epochs": 1, "learning_rate": 1e-5,
                            "early_stopping_patience": 1}, device="cpu", views=2, seed=0, context_rows=20000)
    inner = est._create_inner_estimator(n_estimators=2, device=torch.device("cpu"))
    assert (inner.use_amp, inner.use_fa3) == (False, False)
