"""Failed probes must identify the path without weakening production validation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils import lora_checks as checks, lora_diagnostics as diagnostic
from src.utils.atomic import write_json


@pytest.mark.parametrize("change", ["repeat", "chunks", "nan", "shape"])
def test_probe_names_the_checkpoint_and_specific_failed_check(monkeypatch, capsys, change):
    class Estimator:
        calls = 0
        def fit(self, X, y):
            return self
        def predict(self, X):
            self.calls += 1
            values = X[:, 0].copy()
            if change == "repeat" and self.calls == 2:
                values += .01
            if change == "chunks" and len(X) == 16:
                values += .01
            if change == "nan" and self.calls == 1:
                values[0] = np.nan
            if change == "shape" and self.calls == 2:
                values = values[:, None]
            return values
    monkeypatch.setattr(checks, "_estimator", lambda *a, **k: Estimator())
    records = []
    with pytest.raises(AssertionError, match="checkpoint=original.ckpt.*track=lgd.*precision=float32"):
        checks.prediction_probe([Path("original.ckpt")], "lgd", "tabpfn", diagnostics=records)
    assert len(records) == 1
    record = records[0]
    if change == "chunks":
        assert record["repeat"]["passed"] and not record["query_chunks"]["passed"]
        assert record["query_chunks"]["max_abs_error"] > .009
    else:
        assert not record["repeat"]["passed"]
    json.dumps(records, allow_nan=False)
    assert "LoRA prediction comparison:" in capsys.readouterr().out


def test_diagnostic_collects_all_checkpoints_without_passing_a_failed_check(monkeypatch):
    class Estimator:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return X[:, 0] + (.1 if len(X) == 16 else 0)
    monkeypatch.setattr(checks, "_estimator", lambda *a, **k: Estimator())
    records = []
    predictions = checks.prediction_probe([Path("base"), Path("zero"), Path("changed")],
        "lgd", "tabpfn", strict=False, diagnostics=records)
    assert len(predictions) == len(records) == 3
    assert all(not row["query_chunks"]["passed"] for row in records)
    assert all(row["repeat"]["passed"] for row in records)


def test_comparison_rejects_broadcastable_shapes_and_reports_nonfinite():
    kwargs = dict(atol=1e-5, rtol=1e-4)
    assert not diagnostic.compare_predictions(np.ones(2), np.ones((2, 1)), **kwargs)["passed"]
    result = diagnostic.compare_predictions([np.nan, np.inf], [0, 1], **kwargs)
    assert result["reference_nonfinite"] == 2 and result["max_abs_error"] is None
    json.dumps(result, allow_nan=False)
    assert diagnostic.compare_predictions([1, 2], [1, 2], **kwargs)["passed"]


def test_tf32_arm_restores_process_flags_after_exception():
    import torch
    original = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(RuntimeError):
            with diagnostic.tf32_disabled(True):
                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
                raise RuntimeError("intentional")
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


@pytest.fixture
def historical_probe(tmp_path, monkeypatch):
    import torch
    from src.utils import paths
    root = tmp_path / "output" / "experiment0/lora/failed1"
    model_root = tmp_path / "checkpoints"
    base = model_root / "tabpfn-v3-regressor-v3_default.ckpt"
    base.parent.mkdir()
    base.write_bytes(b"base")
    probe = model_root / "trained/experiment0/lora/failed1/probes"
    probe.mkdir(parents=True)
    for state in ("zero", "changed"):
        (probe / f"{state}_lgd_{base.name}").write_bytes(state.encode())
    marker = dict(schema=checks.SCHEMA, id="failed1", kind="creditpfn_lora", prepared=True,
                  fingerprint="historical", versions={"torch": "old"}, inputs=[checks._binding(base)])
    write_json(root / "workflow.json", marker)
    write_json(root / "receipts/smoke_2.json", {"passed": False})
    monkeypatch.setattr(checks, "folder", lambda identifier: root)
    monkeypatch.setattr(checks, "_base", lambda *args: base)
    monkeypatch.setattr(checks, "fingerprint", lambda: "new-diagnostic")
    monkeypatch.setattr(checks, "environment_versions", lambda **kw: {"torch": "new"})
    monkeypatch.setattr(paths, "checkpoints_dir", lambda *parts: model_root.joinpath(*parts))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA B200")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return root, base


def test_historical_diagnosis_preserves_receipts_weights_and_cannot_authorize(historical_probe, monkeypatch):
    root, base = historical_probe
    original_files = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    policies = []
    def predict(paths, track, family, **kwargs):
        assert not kwargs["strict"] and track == "lgd" and family == "tabpfn"
        policies.append(kwargs["precision"])
        kwargs["diagnostics"].append({"query_chunks": {"passed": False}})
        return [np.array([0., 1.]), np.array([0., 1.]), np.array([.1, .9])]
    monkeypatch.setattr(checks, "prediction_probe", predict)
    monkeypatch.setattr(checks, "_run_training", lambda *a, **k: pytest.fail("diagnostic trained"))
    destination = diagnostic.diagnose("failed1", 2)
    result = json.loads(destination.read_text())
    assert result["complete"] and result["authorizes_production"] is False
    assert result["historical_fingerprint"] != result["current_fingerprint"]
    assert policies == ["float32", "float32", "automatic"]
    assert all(arm["original_vs_zero_export"]["passed"] for arm in result["arms"])
    assert all(p.read_bytes() == content for p, content in original_files.items())
    assert not (root / "inspection.json").exists()
    assert base.read_bytes() == b"base"


def test_diagnostic_refuses_a_replaced_base_before_inference(historical_probe, monkeypatch):
    _, base = historical_probe
    base.write_bytes(b"replaced weights")
    monkeypatch.setattr(checks, "prediction_probe", lambda *a, **k: pytest.fail("unexpected inference"))
    with pytest.raises(RuntimeError, match="base differs"):
        diagnostic.diagnose("failed1", 2)


def test_diagnostic_keeps_failed_arm_report_and_continues_other_policies(historical_probe, monkeypatch):
    def predict(paths, *args, **kwargs):
        if kwargs["precision"] == "automatic":
            raise RuntimeError("inference failed")
        return [np.zeros(2)] * 3
    monkeypatch.setattr(checks, "prediction_probe", predict)
    result = json.loads(diagnostic.diagnose("failed1", 2).read_text())
    assert not result["complete"] and not result["authorizes_production"]
    assert result["arms"][-1]["error"]["message"] == "inference failed"
    assert result["arms"][0]["complete"]


def test_diagnostic_entry_requires_gpu_allocation_before_reading_workflow(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="submitted Mindwell B200"):
        checks.main(["diagnose", "--id", "failed1", "--task", "2"])
