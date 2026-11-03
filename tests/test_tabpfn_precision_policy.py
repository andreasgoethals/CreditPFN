"""One switch sets the evaluation precision of every TabPFN estimator, original and CPT alike."""
import pytest
import torch

from src.model import tabpfn_models as tm


def test_default_keeps_the_package_policy(monkeypatch):
    monkeypatch.delenv(tm.PRECISION_ENV, raising=False)
    assert tm.inference_precision() == "auto"


def test_float32_policy_is_passed_to_the_estimator(monkeypatch, tmp_path):
    monkeypatch.setenv(tm.PRECISION_ENV, "float32")
    seen = {}

    class Fake:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    import tabpfn
    monkeypatch.setattr(tabpfn, "TabPFNClassifier", Fake)
    tm._make_tabpfn("classification", tmp_path / "tabpfn-v3-classifier-v3_default.ckpt")
    assert seen["inference_precision"] is torch.float32


def test_unknown_policy_is_refused(monkeypatch):
    monkeypatch.setenv(tm.PRECISION_ENV, "fp16")
    with pytest.raises(ValueError, match="auto"):
        tm.inference_precision()
