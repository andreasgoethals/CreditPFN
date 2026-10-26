"""Fail closed on a dead Q/V path; the audit must not modify training gradients."""
import copy
import json
import sys

import pytest
import torch

from src.model import lora
from src.utils import lora_training_probe as probe
from tests.test_lora import _v3, _inject_v3, _tiny_tabicl


def test_audit_observes_without_changing_updates_and_allows_initial_zero_a():
    model = _v3()
    _inject_v3(model, rank=2)
    control = copy.deepcopy(model)
    audit = probe.AdapterAudit(model)
    x = torch.randn(2, 6, 8)
    for candidate in (model, control):
        optimizer = torch.optim.SGD(lora.adapter_parameters(candidate), lr=.1)
        candidate(x).square().mean().backward()
        optimizer.step()
    assert all(torch.equal(value, control.state_dict()[key]) for key, value in model.state_dict().items())
    report = audit.report()
    assert report["passed"] and report["projection_count"] == 6
    assert all(r["A"]["calls"] == 1 and r["A"]["nonzero_calls"] == 0 for r in report["projections"])
    assert all(r["B"]["nonzero_calls"] == 1 for r in report["projections"])
    audit.close()
    model(x).square().mean().backward()
    assert all(r["B"]["calls"] == 1 for r in audit.report()["projections"])


@pytest.mark.parametrize("failure", ["unused_block", "no_step", "nonfinite"])
def test_failed_gradient_or_effective_update_is_identified(failure):
    model = _v3()
    _inject_v3(model, rank=2)
    audit = probe.AdapterAudit(model)
    optimizer = torch.optim.SGD(lora.adapter_parameters(model), lr=.1)
    x = torch.randn(2, 6, 8)
    if failure == "unused_block":
        # Leave the installed block registered but bypass its forward path.
        model.column_aggregator.forward = lambda data: data
    loss = model(x).square().mean()
    (loss * float("nan") if failure == "nonfinite" else loss).backward()
    if failure != "no_step":
        optimizer.step()
    report = audit.report()
    assert not report["passed"]
    failed = [r for r in report["projections"] if not r["passed"]]
    if failure == "unused_block":
        assert len(failed) == 2 and all("column_aggregator" in r["target"] for r in failed)
    elif failure == "nonfinite":
        assert any(r["B"]["nonfinite_calls"] for r in failed)
    else:
        assert all(r["effective_update_norm"] == 0 for r in failed)
    audit.close()


@pytest.mark.parametrize("regressor", [False, True])
def test_real_tabicl_recomputation_reaches_each_q_and_v(regressor):
    model = _tiny_tabicl(regressor, recompute=True)
    lora.inject(model, family="tabicl", rank=2, alpha=4)
    audit = probe.AdapterAudit(model)
    x = torch.randn(1, 12, 3)
    y = torch.randn(1, 8) if regressor else torch.randint(0, 2, (1, 8)).float()
    optimizer = torch.optim.SGD(lora.adapter_parameters(model), lr=.01)
    model(x, y).square().mean().backward()
    optimizer.step()
    report = audit.report()
    assert report["passed"], report
    assert all({"q", "v"} == {r["target"].rsplit(":", 1)[-1] for r in report["projections"]
                              if r["target"].rsplit(":", 1)[0] == target["module"] + "." + target["weight"]}
               for target in lora.metadata(model)["targets"])
    audit.close()


def test_one_dead_slice_of_packed_qkv_cannot_hide_behind_changed_matrix():
    model = _tiny_tabicl()
    lora.inject(model, family="tabicl", rank=2)
    audit = probe.AdapterAudit(model)
    # All matrices move, but the V factor in the final matrix gets no gradient.
    optimizer = torch.optim.SGD(lora.adapter_parameters(model), lr=.01)
    parameters = list(lora.adapter_parameters(model))
    sum(p.sum() for p in parameters[:-1]).backward()
    optimizer.step()
    result = audit.report()
    assert not result["passed"]
    assert len([r for r in result["projections"] if not r["passed"]]) == 1
    assert result["projections"][-1]["target"].endswith(":v")
    audit.close()


@pytest.mark.parametrize("exit_code", [0, 75])
def test_wrapper_injects_before_actual_pipeline_import_and_preserves_interrupt(monkeypatch, tmp_path, exit_code):
    original = lora.inject
    previous_argv = sys.argv
    def pipeline(*args, **kwargs):
        # Same late import as train.loop; the real injector is still used.
        from src.model.lora import inject
        model = _v3()
        params = inject(model, family="tabpfn", base_path="tabpfn-v3-classifier-v3_default.ckpt", rank=2)
        opt = torch.optim.SGD(params, lr=.1)
        model(torch.randn(2, 6, 8)).square().mean().backward()
        opt.step()
        raise SystemExit(exit_code)
    monkeypatch.setattr(probe.runpy, "run_path", pipeline)
    path = tmp_path / "audit.json"
    assert probe.run_observed(["--config", "scoped.yaml"], path) == exit_code
    report = json.loads(path.read_text())
    assert report["training_exit_code"] == exit_code
    assert report["passed"] == (exit_code == 0)
    assert lora.inject is original and sys.argv is previous_argv


def test_missing_injection_cannot_pass_wrapper(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.runpy, "run_path", lambda *args, **kwargs: None)
    with pytest.raises(AssertionError, match="missing model"):
        probe.run_observed([], tmp_path / "missing.json")
    assert not json.loads((tmp_path / "missing.json").read_text())["passed"]
