"""Exercise recomputation semantics, gating and extension-only configuration."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.train import activation, loop


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
