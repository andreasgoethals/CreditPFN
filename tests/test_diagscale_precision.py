"""DiagScale under BF16: small query corrections must survive (second campaign, E7)."""
import torch

from src.model.diagscale import QueryScale


def _scale(delta, cast=True):
    module = QueryScale(None, (4,), "cpu", cast_multiplier=cast)
    with torch.no_grad():
        module.delta.fill_(delta)
    return module


def test_small_bf16_correction_is_not_rounded_away():
    query = torch.full((2, 4), 3.0, dtype=torch.bfloat16)
    out = _scale(0.003)(query, None)
    assert out.dtype == torch.bfloat16
    # 3 * 1.003 = 3.009 is representable to BF16 precision; casting 1.003 first would give 3.0.
    assert float(out[0, 0]) != 3.0


def test_fp32_path_is_unchanged():
    query = torch.randn(3, 4)
    module = _scale(0.25)
    assert torch.equal(module(query, None), query * (1 + module.delta).to(query.dtype))


def test_gradient_reaches_delta_in_bf16():
    module = _scale(0.0)
    query = torch.randn(5, 4).to(torch.bfloat16)
    module(query, None).float().sum().backward()
    assert module.delta.grad is not None and torch.isfinite(module.delta.grad).all()
    assert module.delta.dtype == torch.float32
