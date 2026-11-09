"""LoRA and DiagScale injection on separate-projection TabPFN layouts, without checkpoints."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.model import diagscale, lora


class AlongColumnAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for p in ("q", "k", "v", "out"):
            setattr(self, p + "_projection", nn.Linear(8, 8, bias=False))

    def forward(self, x):
        return self.out_projection(F.scaled_dot_product_attention(
            self.q_projection(x), self.k_projection(x), self.v_projection(x)))


class AlongRowAttention(AlongColumnAttention):
    pass


class TabPFNV2p6(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(AlongRowAttention(), AlongColumnAttention()) for _ in range(2)])
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


@pytest.mark.parametrize("mode", ["lora", "diagscale"])
def test_real_separate_adapters(mode):
    torch.manual_seed(41)
    model = TabPFNV2p6()
    pristine = copy.deepcopy(model)
    x, y = torch.randn(2, 12, 8), torch.randn(2, 12, 2)
    before = model(x).detach()
    originals = [(p, p.detach().clone()) for p in model.parameters()]
    module = lora if mode == "lora" else diagscale
    parameters = module.inject(model, family="tabpfn")
    assert torch.equal(before, model(x))
    assert set(map(id, parameters)) == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=0)
    for _ in range(3):
        F.mse_loss(model(x), y).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert all(torch.equal(p, value) and p.grad is None for p, value in originals)
    assert not torch.equal(before, model(x))
    if mode == "lora":
        merged = lora.merged_state_dict(model)
        pristine.load_state_dict(merged, strict=True)
        assert torch.allclose(model(x), pristine(x), atol=1e-7, rtol=1e-6)
        for name, value in merged.items():
            prior = dict(pristine.state_dict())[name]
            assert value.shape == prior.shape
        assert lora.weight_movement(model) > 0
        # Packed K rows have exactly zero additive change.
        for _, owner, seq in lora._installed(model):
            adapter = seq[0]
            if adapter.projections == ("q", "v"):
                matrix = adapter.delta().reshape(-1, adapter.shape[-1])
                assert torch.count_nonzero(matrix[adapter.slices[0][1]:adapter.slices[1][0]]) == 0
    else:
        module.inject(pristine, family="tabpfn")
        module.restore(pristine, module.adapter_state(model))
        assert torch.equal(model(x), pristine(x))
    values = [p.detach().clone() for p in parameters]
    model.to(torch.bfloat16)
    assert all(p.dtype == torch.float32 and torch.equal(p, value) for p, value in zip(parameters, values))


def test_legacy_query_scale_is_query_multiplier_only():
    weight = torch.randn(3, 2, 4, 8)
    adapter = diagscale.QueryWeightScale(weight, 0, 8)
    with torch.no_grad():
        adapter.delta.copy_(torch.linspace(-.2, .3, 8))
    x = torch.randn(7, 8)
    scaled = F.linear(x, adapter(weight).flatten(0, 2))
    original = F.linear(x, weight.flatten(0, 2))
    assert torch.allclose(scaled[:, :8], original[:, :8] * (1 + adapter.delta), atol=1e-6)
    assert torch.equal(scaled[:, 8:], original[:, 8:])


def test_unknown_layout_rejected_before_mutation():
    model = TabPFNV2p6()
    model.blocks[1][0].q_projection = nn.Identity()
    before = copy.deepcopy(model.state_dict())
    for mode in (lora, diagscale):
        with pytest.raises(ValueError, match="Incomplete"):
            mode.inject(model, family="tabpfn")
        assert all(p.requires_grad for p in model.parameters())
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
