from contextlib import nullcontext

import pytest
import torch
from tabpfn.architectures.tabpfn_v2 import TabPFNV2, TabPFNV2Config

from src.train.precision import bounded_fp32_reference, training_precision
from src.utils.v2_precision import fp32_normalizations


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


def test_legacy_mixed_comparator_is_scoped_and_preserves_amp_projections():
    model = TabPFNV2(config=TabPFNV2Config(emsize=32, nhead=2, nlayers=1), n_out=2)
    layer = model.blocks[0].layernorm_mha1
    features = torch.randn(3, 32).half()
    assert layer(features).dtype == torch.float16
    with fp32_normalizations(model), torch.autocast("cpu", dtype=torch.float16):
        assert layer(features).dtype == torch.float32
        assert model.output_projection(layer(features)).dtype == torch.float16
    assert layer(features).dtype == torch.float16


@pytest.mark.parametrize(("base", "family", "track", "dtype", "label"), [
    (V26, "tabpfn", "pd", None, "fp32"),
    (V26, "tabpfn", "pd", torch.bfloat16, "bf16"),
    ("tabpfn-v2-regressor.ckpt", "tabpfn", "lgd", torch.float16, "fp16"),
    ("tabicl-classifier.ckpt", "tabicl", "pd", torch.float16, "fp16"),
    (V26, "tabpfn", "pd", torch.float16, "fp16"),
])
def test_other_training_recipes_keep_their_native_policy(base, family, track, dtype, label):
    policy = training_precision(torch.nn.Linear(4, 2), base=base, family=family, track=track, amp_dtype=dtype)
    assert policy.label == label and policy.amp_dtype == dtype and policy.bounded_backbone is None
    assert policy.scaler("cpu").is_enabled() == (dtype == torch.float16)


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
    assert resolve(torch.nn.Sequential(model)).bounded_backbone is model
    with pytest.raises(ValueError, match="FP32 model parameters"):
        resolve(model.half())
    model.float()
    model.blocks[0].per_column_attention_between_cells = None
    with pytest.raises(ValueError, match="column-attention layout"):
        resolve(model)


@pytest.mark.parametrize("requested", [torch.float16, torch.bfloat16, None])
def test_bounded_training_disables_outer_amp_scaler_and_tf32_and_restores_state(requested):
    model = TabPFNV2(config=TabPFNV2Config(emsize=32, nhead=2, nlayers=1), n_out=2)
    policy = training_precision(model, base=V2, family="tabpfn", track="pd", amp_dtype=requested)
    assert policy.label == "fp32_bounded" and policy.amp_dtype is None
    assert not policy.scaler("cpu").is_enabled()
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with pytest.raises(RuntimeError, match="deliberate"):
                with policy.scope():
                    assert not torch.is_autocast_enabled("cpu")
                    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                    assert model.output_projection(torch.ones(2, 32)).dtype == torch.float32
                    raise RuntimeError("deliberate failure in a scoped step")
            assert torch.is_autocast_enabled("cpu") and torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32 and "forward" not in model.blocks[0].__dict__
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def test_deployed_bounded_policy_matches_native_adamw_update_under_outer_checkpoint_and_amp():
    from torch.utils.checkpoint import checkpoint
    from src.train.precision import strict_fp32
    import copy

    generator = torch.Generator().manual_seed(71)
    model = TabPFNV2(config=TabPFNV2Config(emsize=32, nhead=2, nlayers=2), n_out=2)
    x = torch.randn(30, 1, 9, generator=generator)
    y = torch.randint(2, (20, 1), generator=generator).float()
    target = torch.randint(2, (10,), generator=generator)
    models = [model, copy.deepcopy(model)]
    results = []
    with torch.random.fork_rng(devices=[]):
        for index, candidate in enumerate(models):
            policy = training_precision(candidate, base=V2, family="tabpfn", track="pd", amp_dtype=torch.float16)
            optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-4)
            torch.manual_seed(17)
            # The training loop can wrap a whole ensemble member in another
            # checkpoint. Exercise both nested checkpoint levels during backward.
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(index)):
                with policy.scope() if index else strict_fp32():
                    output = checkpoint(candidate, x, y, use_reentrant=False)
                    loss = torch.nn.functional.cross_entropy(output.squeeze(1), target)
                    loss.backward()
            rng = torch.random.get_rng_state().clone()
            optimizer.step()
            results.append((output.detach(), rng, [p.grad.clone() for p in candidate.parameters()]))
    assert results[0][0].shape == results[1][0].shape == (10, 1, 2)
    torch.testing.assert_close(results[0][0], results[1][0], rtol=2e-5, atol=1e-6)
    assert torch.equal(results[0][1], results[1][1])
    for native, bounded in zip(results[0][2], results[1][2], strict=True):
        torch.testing.assert_close(native, bounded, rtol=2e-4, atol=2e-6)
    for native, bounded in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(native, bounded, rtol=2e-4, atol=2e-6)


def test_policy_change_invalidates_training_and_evaluation_identities(tmp_path):
    from src.utils.experiment import code_identity
    path = tmp_path / "src" / "train" / "precision.py"
    path.parent.mkdir(parents=True)
    path.write_text("policy = 1\n")
    before = {stage: code_identity(tmp_path, stage=stage) for stage in ("train", "eval")}
    path.write_text("policy = 22\n")
    assert all(code_identity(tmp_path, stage=stage) != value for stage, value in before.items())
