"""Exercise recomputation semantics, gating and extension-only configuration."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.train import activation, loop
from src.utils.legacy_design import configurations
from src.utils.recompute_check import compare_recomputation


class NativeLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 12), nn.GELU(), nn.Dropout(.2))])
        self.head = nn.Linear(12, 2)

    def forward(self, x, y, *, only_return_standard_out, categorical_inds):
        x = self.embedding(x)
        out = self.blocks[0](x)
        return self.head(out)[len(y):]


def batch():
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(20, 1, 4, generator=generator)
    y = (x[:, 0, 0] > 0).long()
    members = [SimpleNamespace(X_context=x[:12], X_query=x[12:], y_context=y[:12, None, None],
                class_permutation=None if i == 0 else [1, 0], categorical_idx=[],
                outlier_removal_std=None) for i in range(2)]
    return SimpleNamespace(members=members, n_classes=2, task_type="classification", y_query=y[12:])


@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
def test_native_recomputation_preserves_predictions_gradients_updates_and_reload(dtype):
    torch.manual_seed(11)
    model = NativeLikeModel()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    # This Windows CPU's oneDNN build has no BF16 backward kernel. Exercise
    # eager CPU math here; the acceptance job separately tests native CUDA BF16.
    with torch.backends.mkldnn.flags(enabled=False):
        result = compare_recomputation(model, batch(), nn.CrossEntropyLoss(), device="cpu", dtypes=[dtype])
    assert result["passed"]
    assert result["comparisons"][0]["block_calls"] == {"False": 6, "True": 12}
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())


@pytest.mark.parametrize("fault", ["ignore", "corrupt"])
def test_recomputation_gate_rejects_ignored_option_or_numerical_change(monkeypatch, fault):
    original_forward = activation.forward

    def broken(model, *args, enabled=False, **kwargs):
        out = original_forward(model, *args, enabled=enabled and fault != "ignore", **kwargs)
        return out * 2 if enabled and fault == "corrupt" else out

    monkeypatch.setattr(activation, "forward", broken)
    model = NativeLikeModel()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(AssertionError):
        compare_recomputation(model, batch(), nn.CrossEntropyLoss(), device="cpu", dtypes=[None])
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())


def test_default_does_not_checkpoint_or_modify_native_forward(monkeypatch):
    def unexpected(*args):
        raise AssertionError("Disabled recomputation must not call checkpoint")
    monkeypatch.setattr(activation, "checkpoint", unexpected)
    model = NativeLikeModel()
    loss = loop._ensemble_step_loss(model, batch(), criterion=nn.CrossEntropyLoss())
    loss.backward()
    assert model.head.weight.grad is not None


@pytest.mark.parametrize("invalid", [None, "false", 1])
def test_recomputation_flag_is_explicit_boolean(invalid):
    with pytest.raises(ValueError, match="boolean"):
        loop._ensemble_step_loss(NativeLikeModel(), batch(), criterion=nn.CrossEntropyLoss(),
                                 activation_checkpointing=invalid)


def test_checkpoint_options_are_explicit(monkeypatch):
    observed = {}
    def capture(model, *args, **kwargs):
        observed.update(kwargs)
        return None
    monkeypatch.setattr(activation, "checkpoint", capture)
    activation.forward(NativeLikeModel(), enabled=True)
    assert observed == {"use_reentrant": False, "preserve_rng_state": True}


def test_v2_size_extensions_and_shared_anchor_use_identical_recomputation():
    configs = configurations("recomputetest")
    enabled = [k for k, c in configs.items() if c.get("train", {}).get("activation_checkpointing", False)]
    assert set(enabled) == {"e8_pd_v2", "e9_pd_v2_rows2048", "e9_pd_v2_rows10000"}
    cfg = configs["e9_pd_v2_rows10000"]
    assert cfg.train.max_rows_per_step == 10000
    assert cfg.train.n_estimators_finetune.pd == 2
    assert cfg.optimizer.l2sp_lambda == .003
    assert cfg.tunable.learning_rates == [3e-7]
