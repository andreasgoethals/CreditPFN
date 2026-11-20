from contextlib import nullcontext

import torch
from tabpfn.architectures.tabpfn_v2 import TabPFNV2, TabPFNV2Config

from src.utils.v2_precision import bounded_fp32_reference, fp32_normalizations


def test_v2_bounded_reference_preserves_loss_and_every_parameter_gradient():
    generator = torch.Generator().manual_seed(91)
    model = TabPFNV2(config=TabPFNV2Config(emsize=32, nhead=2, nlayers=2), n_out=2)
    x = torch.randn(36, 1, 12, generator=generator)
    y = torch.randint(2, (24, 1), generator=generator).float()
    results = []
    for mode in (nullcontext(), bounded_fp32_reference(model)):
        model.zero_grad(set_to_none=True)
        with mode:
            logits = model(x, y)
            loss = logits.square().mean()
            loss.backward()
        results.append((loss.item(), [p.grad.detach().clone() for p in model.parameters()]))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-5, atol=1e-6)
    for base, bounded in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(base, bounded, rtol=2e-4, atol=2e-6)


def test_v2_normalization_candidate_is_scoped_and_preserves_amp_projections():
    model = TabPFNV2(config=TabPFNV2Config(emsize=32, nhead=2, nlayers=1), n_out=2)
    layer = model.blocks[0].layernorm_mha1
    features = torch.randn(3, 32).half()
    assert layer(features).dtype == torch.float16
    with fp32_normalizations(model), torch.autocast("cpu", dtype=torch.float16):
        assert layer(features).dtype == torch.float32
        assert model.output_projection(layer(features)).dtype == torch.float16
    assert layer(features).dtype == torch.float16
