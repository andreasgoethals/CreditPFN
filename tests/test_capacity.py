"""Capacity diagnostics must measure the requested model mode and expose failures."""
from omegaconf import OmegaConf
import pytest
import torch

from src.train import capacity


@pytest.mark.parametrize("family", ["tabpfn", "tabicl"])
@pytest.mark.parametrize("frozen", [False, True])
def test_probe_passes_adaptation_to_loader(tmp_path, monkeypatch, family, frozen):
    from src.train import model as tabpfn_model, tabicl_model
    ckpt = tmp_path / "base.ckpt"
    ckpt.write_bytes(b"synthetic")
    monkeypatch.setattr(capacity, "_checkpoint", lambda *args: ckpt)
    seen = []
    fake_model = torch.nn.Linear(1, 1)

    def load(path, **kwargs):
        seen.append(kwargs)
        return (fake_model, None, None, None) if family == "tabpfn" else (fake_model, None)

    module = tabpfn_model if family == "tabpfn" else tabicl_model
    monkeypatch.setattr(module, f"load_{family}_for_training", load)
    monkeypatch.setattr(capacity, "_measure_rows", lambda *args: None)
    fn = capacity.probe_base if family == "tabpfn" else capacity.probe_tabicl_base
    fn(str(ckpt), "pd", [10], "cpu", freeze_backbone=frozen)
    assert seen[0]["freeze_backbone"] is frozen


def test_probe_dispatch_uses_track_and_family_members_and_never_falls_back(monkeypatch):
    cfg = OmegaConf.create({"train": {"n_estimators_finetune": {"pd": 2, "lgd": 8},
                                     "n_estimators_finetune_tabicl": 3},
                           "tunable": {"query_fractions": [.4],
                                       "classifier_base_paths": ["tabpfn-v2.ckpt", "tabicl-base.ckpt"],
                                       "regressor_base_paths": ["tabpfn-v3.ckpt", "tabicl-base.ckpt"]}})
    calls = []
    monkeypatch.setattr(capacity, "probe_base", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(capacity, "probe_tabicl_base", lambda *a, **kw: calls.append((a, kw)))
    capacity.probe_configured_bases(cfg, ["pd", "lgd"], [10])
    assert [kw["n_estimators"] for _, kw in calls] == [2, 2, 3, 3, 8, 8, 3, 3]
    assert [kw["freeze_backbone"] for _, kw in calls] == [False, True] * 4
    assert all(kw["query_fraction"] == .4 for _, kw in calls)
    calls.clear()

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("real diagnostic failure")

    monkeypatch.setattr(capacity, "probe_base", fail)
    with pytest.raises(TypeError, match="real diagnostic failure"):
        capacity.probe_configured_bases(cfg, ["pd"], [10])
    assert len(calls) == 1


def test_failed_probe_releases_gradients_and_only_oom_is_recoverable(capsys):
    model = torch.nn.Linear(1, 1)

    def loss(rows):
        if rows == 10:
            model.weight.grad = torch.ones_like(model.weight)
            raise torch.cuda.OutOfMemoryError("synthetic capacity limit")
        assert model.weight.grad is None
        return model(torch.ones(1, 1)).sum()

    capacity._measure_rows(model, [10, 5], "cpu", loss)
    assert model.weight.grad is None and "OOM" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="unexpected"):
        capacity._measure_rows(model, [10], "cpu",
                               lambda rows: (_ for _ in ()).throw(RuntimeError("unexpected")))
