"""Adapter L2-SP equals 0.5 * lam * ||W - W0||^2 of the weights the adapter changes.

Checked against independently merged weights on the real base checkpoints (CPU), for LoRA on
all four models and DiagScale on all four models; zero (with zero gradient) at initialisation.
"""
from pathlib import Path

import pytest
import torch
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[1]
LAM = 0.003
MODELS = {"v2": "tabpfn-v2-classifier-v2_default.ckpt",
          "v2.6": "tabpfn-v2.6-regressor-v2.6_default.ckpt",
          "v3": "tabpfn-v3-classifier-v3_default.ckpt",
          "tabicl": "tabicl-classifier-v2-20260212.ckpt"}
pytestmark = pytest.mark.slow


def _load(key):
    path = ROOT / "checkpoints" / MODELS[key]
    if not path.is_file():
        pytest.skip(f"checkpoint unavailable: {path.name}")
    track = "lgd" if "regressor" in path.name else "pd"
    if key == "tabicl":
        from src.train.tabicl_model import load_tabicl_for_training
        model, _ = load_tabicl_for_training(str(path), track=track, device="cpu")
        return model, "tabicl", path
    from src.train.model import load_tabpfn_for_training
    model, *_ = load_tabpfn_for_training(str(path), track=track, device="cpu")
    return model, "tabpfn", path


def _parametrized_weights(model):
    """{(owner id, name): (owner, name)} for every parametrized tensor."""
    out = {}
    for owner in model.modules():
        if parametrize.is_parametrized(owner):
            for name in owner.parametrizations:
                out[(id(owner), name)] = (owner, name)
    return out


@pytest.mark.parametrize("key", list(MODELS))
def test_lora_penalty_is_zero_with_zero_gradient_at_initialisation(key):
    from src.model.lora import inject
    from src.train.l2sp import EffectiveWeightL2SP
    model, family, path = _load(key)
    inject(model, family=family, base_path=f"checkpoints/{path.name}", base_sha256="0" * 64,
           rank=16, alpha=32, dropout=0.0, target_modules="attention_qv")
    penalty = EffectiveWeightL2SP(model, "lora")(LAM)
    assert float(penalty) == 0.0
    penalty.backward()
    from src.model.lora import adapter_parameters
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in adapter_parameters(model))


@pytest.mark.parametrize("key", list(MODELS))
def test_lora_penalty_equals_merged_weight_distance(key):
    from src.model.lora import adapter_parameters, inject
    from src.train.l2sp import EffectiveWeightL2SP
    model, family, path = _load(key)
    inject(model, family=family, base_path=f"checkpoints/{path.name}", base_sha256="0" * 64,
           rank=16, alpha=32, dropout=0.0, target_modules="attention_qv")
    torch.manual_seed(0)
    with torch.no_grad():
        for p in adapter_parameters(model):
            p.normal_(0, 0.01)
    expected = 0.0
    for owner, name in _parametrized_weights(model).values():
        original = owner.parametrizations[name].original.detach().double()
        current = getattr(owner, name).detach().double()
        expected += float((current - original).square().sum())
    actual = float(EffectiveWeightL2SP(model, "lora")(LAM))
    assert actual == pytest.approx(0.5 * LAM * expected, rel=1e-4)
    assert actual > 0


@pytest.mark.parametrize("key", list(MODELS))
def test_diagscale_penalty_is_zero_at_initialisation(key):
    from src.model.diagscale import adapter_parameters, inject
    from src.train.l2sp import EffectiveWeightL2SP
    model, family, _ = _load(key)
    inject(model, family=family)
    penalty = EffectiveWeightL2SP(model, "diagscale")(LAM)
    assert float(penalty) == 0.0
    penalty.backward()
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in adapter_parameters(model))


@pytest.mark.parametrize("key", ["v2", "v2.6"])
def test_diagscale_penalty_equals_actual_weight_change_on_legacy_tabpfn(key):
    """v2/v2.6 DiagScale scales the query weight rows themselves: the equality is exact."""
    from src.model.diagscale import adapter_parameters, inject
    from src.train.l2sp import EffectiveWeightL2SP
    model, family, _ = _load(key)
    inject(model, family=family)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in adapter_parameters(model):
            p.normal_(0, 0.05)
    expected = 0.0
    for owner, name in _parametrized_weights(model).values():
        original = owner.parametrizations[name].original.detach().double()
        current = getattr(owner, name).detach().double()
        expected += float((current - original).square().sum())
    actual = float(EffectiveWeightL2SP(model, "diagscale")(LAM))
    assert actual == pytest.approx(0.5 * LAM * expected, rel=1e-4)


@pytest.mark.parametrize("key", ["v3", "tabicl"])
def test_diagscale_penalty_equals_scaled_query_projection_change(key):
    """v3/TabICL: penalty = 0.5 lam sum ||diag(delta) [W_q | b_q]||^2 per query projection."""
    from src.model.diagscale import QueryScale, inject
    from src.train.l2sp import EffectiveWeightL2SP
    model, family, _ = _load(key)
    inject(model, family=family)
    torch.manual_seed(2)
    expected = 0.0
    modules = dict(model.named_modules())
    with torch.no_grad():
        for path, module in modules.items():
            if type(module) is not QueryScale:
                continue
            module.delta.normal_(0, 0.05)
            parent = modules[path.rpartition(".")[0]]
            if hasattr(parent, "q_projection"):
                w = parent.q_projection.weight.double()
                b = parent.q_projection.bias
            else:
                dim = parent.in_proj_weight.shape[-1]
                w = parent.in_proj_weight.double()[:dim]
                b = parent.in_proj_bias[:dim] if parent.in_proj_bias is not None else None
            full = w if b is None else torch.cat([w, b.double()[:, None]], 1)
            delta = module.delta.double().reshape(-1)
            expected += float((delta[:, None] * full).square().sum())
    actual = float(EffectiveWeightL2SP(model, "diagscale")(LAM))
    assert actual == pytest.approx(0.5 * LAM * expected, rel=1e-4)


def test_unknown_mode_is_rejected():
    from src.train.l2sp import EffectiveWeightL2SP
    with pytest.raises(ValueError):
        EffectiveWeightL2SP(torch.nn.Linear(2, 2), "full")
