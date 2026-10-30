"""Observe optimized training models without treating monitor clones as learners."""
import copy
import json

import pytest
import torch
from torch.nn import functional as F

from src.model import diagscale, lora
from src.train import loop
from src.utils import legacy_design, legacy_probe
from tests.test_legacy_extensions import PerFeatureTransformer, TabPFNV2p6


@pytest.mark.parametrize("mode", ["diagscale", "lora"])
@pytest.mark.parametrize("version", ["v2", "v2p6"])
@pytest.mark.parametrize("fault", [None, "no_backward", "changed_backbone"])
def test_training_probe_binds_optimizer_model(mode, version, fault, monkeypatch, tmp_path):
    torch.manual_seed(518)
    cfg = legacy_design.configurations("auditprobe")[f"e{5 if mode == 'diagscale' else 6}_pd_{version}"]
    cfg.optimizer.lr = .01
    path = tmp_path / "config.yaml"
    from omegaconf import OmegaConf
    OmegaConf.save(cfg, path)
    adaptation = diagscale if mode == "diagscale" else lora
    model = PerFeatureTransformer() if version == "v2" else TabPFNV2p6()
    pristine = copy.deepcopy(model)
    observed = {}

    def pipeline(*args, **kwargs):
        adaptation.inject(model, family="tabpfn")
        optimizer, scheduler = loop._make_optimizer_and_scheduler(model, cfg, total_steps=6)
        x, y = torch.randn(2, 12, 8), torch.randn(2, 12, 2)
        for _ in range(3):
            if fault != "no_backward":
                F.mse_loss(model(x), y).backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        if fault == "changed_backbone":
            with torch.no_grad():
                next(p for p in model.parameters() if not p.requires_grad).add_(1)
        # Monitoring installs and restores DiagScale bundles, but never creates
        # a training optimizer or computes a backward pass on these models.
        for _ in range(2):
            monitor = copy.deepcopy(pristine)
            adaptation.inject(monitor, family="tabpfn")
            if mode == "diagscale":
                diagscale.restore(monitor, diagscale.adapter_state(model))
            with torch.no_grad():
                monitor(x)
        observed["optimizer"] = optimizer

    monkeypatch.setattr(legacy_probe.runpy, "run_path", pipeline)
    destination = tmp_path / "audit.json"
    if fault:
        with pytest.raises(AssertionError, match="finite updates"):
            legacy_probe.run_observed(["--config", str(path)], destination)
    else:
        assert legacy_probe.run_observed(["--config", str(path)], destination) == 0
    report = json.loads(destination.read_text())
    assert len(report["models"]) == 1
    assert report["passed"] is (fault is None)
    assert report["base_unchanged"] is (fault != "changed_backbone")
    assert {id(p) for group in observed["optimizer"].param_groups for p in group["params"]} == {
        id(p) for p in adaptation.adapter_parameters(model)}


@pytest.mark.parametrize("selection", ["missing_adapter", "extra_backbone"])
def test_training_probe_rejects_incorrect_optimizer_selection(selection, monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    cfg = legacy_design.configurations("auditprobe")["e5_pd_v2"]
    config = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config)
    model = PerFeatureTransformer()

    def wrong_optimizer(model, *args, **kwargs):
        parameters = diagscale.adapter_parameters(model)
        if selection == "missing_adapter":
            parameters = parameters[:-1]
        else:
            parameters += [next(p for p in model.parameters() if not p.requires_grad)]
        return torch.optim.AdamW(parameters, lr=.01), None

    def pipeline(*args, **kwargs):
        diagscale.inject(model, family="tabpfn")
        loop._make_optimizer_and_scheduler(model, cfg, total_steps=6)

    monkeypatch.setattr(loop, "_make_optimizer_and_scheduler", wrong_optimizer)
    monkeypatch.setattr(legacy_probe.runpy, "run_path", pipeline)
    destination = tmp_path / "audit.json"
    with pytest.raises(AssertionError, match="exactly the declared adapters"):
        legacy_probe.run_observed(["--config", str(config)], destination)
    assert json.loads(destination.read_text())["passed"] is False
