from contextlib import ExitStack, nullcontext

import pytest
import torch
from tabpfn.architectures.tabpfn_v2 import LowerPrecisionLayerNorm, TabPFNV2, TabPFNV2Config

from src.train.precision import training_precision
from src.utils.v2_precision import bounded_fp32_reference


V2 = "tabpfn-v2-classifier-v2_default.ckpt"
V26 = "tabpfn-v2.6-classifier-v2.6_default.ckpt"


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


def test_v2_training_normalization_is_scoped_and_preserves_amp_projections():
    model = TabPFNV2(config=TabPFNV2Config(emsize=32, nhead=2, nlayers=1), n_out=2)
    policy = training_precision(model, base=V2, family="tabpfn", track="pd", amp_dtype=torch.float16)
    assert policy.label == "fp16_norm32" and len(policy.normalizations) == 3
    layer = model.blocks[0].layernorm_mha1
    features = torch.randn(3, 32).half()
    assert layer(features).dtype == torch.float16
    with policy.scope(), torch.autocast("cpu", dtype=torch.float16):
        assert layer(features).dtype == torch.float32
        assert model.output_projection(layer(features)).dtype == torch.float16
    assert layer(features).dtype == torch.float16


@pytest.mark.parametrize(("base", "family", "track", "dtype", "label"), [
    (V2, "tabpfn", "pd", None, "fp32"),
    (V2, "tabpfn", "pd", torch.bfloat16, "bf16"),
    ("tabpfn-v2-regressor.ckpt", "tabpfn", "lgd", torch.float16, "fp16"),
    ("tabicl-classifier.ckpt", "tabicl", "pd", torch.float16, "fp16"),
    (V26, "tabpfn", "pd", torch.float16, "fp16"),
])
def test_other_training_recipes_keep_their_native_policy(base, family, track, dtype, label):
    policy = training_precision(torch.nn.Linear(4, 2), base=base, family=family, track=track, amp_dtype=dtype)
    assert policy.label == label and not policy.normalizations


def test_inconsistent_or_unavailable_v2_backbone_cannot_silently_change_policy():
    def resolve(model, base=V2):
        return training_precision(model, base=base, family="tabpfn", track="pd", amp_dtype=torch.float16)
    model = TabPFNV2(config=TabPFNV2Config(emsize=32, nhead=2, nlayers=1), n_out=2)
    with pytest.raises(ValueError, match="Missing"):
        resolve(None)
    with pytest.raises(ValueError, match="architecture differs"):
        resolve(torch.nn.Linear(4, 2))
    with pytest.raises(ValueError, match="architecture differs"):
        resolve(model, V26)
    assert resolve(torch.nn.Sequential(model)).label == "fp16_norm32"  # adaptation wrapper
    model.blocks[0].layernorm_mha1 = torch.nn.LayerNorm(32)
    with pytest.raises(ValueError, match="normalization layout"):
        resolve(model)


def test_deployed_policy_matches_original_e0_candidate_through_recomputed_backward():
    """Use the exact pre-promotion candidate as an independent implementation oracle."""
    from torch.utils.checkpoint import checkpoint
    generator = torch.Generator().manual_seed(71)
    model = TabPFNV2(config=TabPFNV2Config(emsize=32, nhead=2, nlayers=2), n_out=2)
    x = torch.randn(30, 1, 9, generator=generator)
    y = torch.randint(2, (20, 1), generator=generator).float()
    policy = training_precision(model, base=V2, family="tabpfn", track="pd", amp_dtype=torch.float16)
    results = []
    # The CPU's optimized FP16 attention backend is unavailable on some development CPUs.
    with torch.backends.mkldnn.flags(enabled=False), torch.random.fork_rng(devices=[]):
        for deployed in (False, True):
            torch.manual_seed(17)
            model.zero_grad(set_to_none=True)
            with ExitStack() as scope:
                if deployed:
                    scope.enter_context(policy.scope())
                else:
                    for layer in model.modules():
                        if isinstance(layer, LowerPrecisionLayerNorm):
                            scope.enter_context(layer.register_forward_pre_hook(lambda m, args: (args[0].float(),)))
                with torch.autocast("cpu", dtype=torch.float16):
                    output = checkpoint(model, x, y, use_reentrant=False)
                    loss = output.float().square().mean()
                loss.backward()
            results.append((output.detach(), [p.grad.detach().clone() for p in model.parameters()]))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for proposed, deployed in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(proposed, deployed, rtol=0, atol=0)
    assert all(not layer._forward_pre_hooks for layer in policy.normalizations)


def test_policy_change_invalidates_training_and_evaluation_identities(tmp_path):
    from src.utils.experiment import code_identity
    path = tmp_path / "src" / "train" / "precision.py"
    path.parent.mkdir(parents=True)
    path.write_text("policy = 1\n")
    before = {stage: code_identity(tmp_path, stage=stage) for stage in ("train", "eval")}
    path.write_text("policy = 22\n")
    assert all(code_identity(tmp_path, stage=stage) != value for stage, value in before.items())
