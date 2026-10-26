"""LoRA invariants and real TabICLv2 CPU probes; no weights or downloads needed."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.model import lora


class Attention(nn.Module):
    """Structural v3 projection fixture; the installed local package predates v3."""

    def __init__(self, width=8):
        super().__init__()
        for name in ("q", "k", "v", "out"):
            setattr(self, name + "_projection", nn.Linear(width, width, bias=False))

    def forward(self, x):
        q, k, v = (getattr(self, p + "_projection")(x) for p in ("q", "k", "v"))
        return x + self.out_projection(F.scaled_dot_product_attention(q, k, v))


class TabPFNV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_distribution_embedder = nn.Sequential(Attention())
        self.column_aggregator = nn.Sequential(Attention())
        self.icl_blocks = nn.Sequential(Attention())
        self.decoder = nn.Module()
        # Retrieval decoder has no learned values and must remain untouched.
        self.decoder.q_projection = nn.Linear(8, 8)
        self.decoder.k_projection = nn.Linear(8, 8)
        self.output = nn.Linear(8, 2)

    def forward(self, x):
        return self.output(self.icl_blocks(self.column_aggregator(self.feature_distribution_embedder(x))))


def _v3():
    torch.manual_seed(21)
    return TabPFNV3()


def _inject_v3(model, **kwargs):
    return lora.inject(model, family="tabpfn", base_path="tabpfn-v3-classifier-v3_default.ckpt",
                       base_sha256="a" * 64, **kwargs)


def _tiny_tabicl(regressor=False, *, recompute=False):
    pytest.importorskip("tabicl")
    from src.train.tabicl_compat import import_tabicl_core
    torch.manual_seed(11)
    model = import_tabicl_core()(
        max_classes=0 if regressor else 2, num_quantiles=9, embed_dim=16,
        col_num_blocks=1, col_nhead=2, col_num_inds=4,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=1, dropout=0., recompute=recompute)
    # A fresh upstream model initializes residual outputs to zero. Give this
    # synthetic fixture nonzero pretrained-like projections so all stages can
    # propagate adapter gradients; the GPU gate uses actual released weights.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.out_proj.weight)
    return model.train()


def test_v3_identity_and_complete_coverage_only_adapters_train():
    model = _v3().train()
    x = torch.randn(2, 6, 8)
    before = model(x).detach()
    original = [(p, p.detach().clone()) for p in model.parameters()]
    params = _inject_v3(model)
    assert torch.equal(before, model(x))
    assert model.training and all(module.training for module in model.modules())
    assert all(not p.requires_grad and torch.equal(p, value) for p, value in original)
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    spec = lora.metadata(model)
    assert (spec["rank"], spec["alpha"], spec["dropout"]) == (16, 32, 0)
    assert len(spec["targets"]) == 6
    assert not any("decoder" in t["module"] for t in spec["targets"])
    assert lora.weight_movement(model) == 0
    spec["targets"].clear()
    assert len(lora.metadata(model)["targets"]) == 6


@pytest.mark.parametrize("options", [dict(rank=0), dict(rank=True), dict(alpha=float("nan")),
                                    dict(dropout=.1), dict(target_modules=["q_projection"])])
def test_invalid_intervention_never_mutates_model(options):
    model = _v3()
    initial = copy.deepcopy(model.state_dict())
    flags = [p.requires_grad for p in model.parameters()]
    with pytest.raises(ValueError):
        _inject_v3(model, **options)
    assert list(initial) == list(model.state_dict())
    assert flags == [p.requires_grad for p in model.parameters()]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())


def test_rejects_unsupported_model_version_missing_stage_and_duplicate_injection():
    model = _v3()
    with pytest.raises(ValueError, match="versioned"):
        lora.inject(model, family="tabpfn", base_path="tabpfn-v3.5-classifier.ckpt")
    with pytest.raises(ValueError, match="architecture"):
        lora.inject(nn.Linear(3, 3), family="tabpfn")
    model.column_aggregator = nn.Identity()
    with pytest.raises(ValueError, match="stages"):
        _inject_v3(model)
    assert all(p.requires_grad for p in model.parameters())
    model = _v3()
    _inject_v3(model)
    with pytest.raises(ValueError, match="already"):
        _inject_v3(model)


def test_real_adapter_learning_merges_without_mutating_live_model():
    model = _v3()
    original = [(p, p.detach().clone()) for p in model.parameters()]
    params = _inject_v3(model, rank=2, alpha=4)
    opt = torch.optim.Adam(params, lr=.01)
    x = torch.randn(2, 5, 8)
    y = torch.randn(2, 5, 2)
    before = model(x).detach()
    F.mse_loss(model(x), y).backward()
    assert all(torch.isfinite(p.grad).all() for p in params)
    # B=0 gives zero dL/dA on the first step, a correct LoRA identity initialization.
    assert all(torch.count_nonzero(p.grad) for name, p in model.named_parameters() if ".B_" in name)
    opt.step()
    opt.zero_grad(set_to_none=True)
    after = model(x).detach()
    assert not torch.equal(before, after)
    assert all(torch.equal(p, value) for p, value in original)
    state_before = copy.deepcopy(model.state_dict())
    ids_before = [id(p) for p in params]
    native = _v3()
    merged = lora.merged_state_dict(model)
    assert set(merged) == set(native.state_dict())
    native.load_state_dict(merged, strict=True)
    torch.testing.assert_close(native(x), after, rtol=0, atol=0)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state_before.items())
    assert ids_before == [id(p) for p in lora.adapter_parameters(model)]
    assert model.training
    F.mse_loss(model(x), y).backward()
    assert all(torch.count_nonzero(p.grad) for p in params)
    opt.step()
    assert lora.weight_movement(model) > 0


def test_native_export_rejects_nonfinite_updates():
    model = _v3()
    _inject_v3(model)
    with torch.no_grad():
        lora.adapter_parameters(model)[1].fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        lora.merged_state_dict(model)


def test_adapter_master_precision_survives_parent_cast_and_autocast():
    model = _v3()
    _inject_v3(model, rank=2)
    with torch.no_grad():
        for p in lora.adapter_parameters(model):
            p.add_(.000012345)
    initial = lora.adapter_state(model)
    model.to(dtype=torch.bfloat16)
    assert all(p.dtype == torch.float32 for p in lora.adapter_parameters(model))
    assert all(torch.equal(value, lora.adapter_state(model)[key]) for key, value in initial.items())
    model.float()
    adapter = next(m for m in model.modules() if isinstance(m, lora.LowRankWeight))
    ref = adapter.delta()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.equal(ref, adapter.delta())


def test_movement_is_effective_weight_norm_and_factor_scale_invariant():
    model = _v3()
    _inject_v3(model, rank=2)
    for module in model.modules():
        if isinstance(module, lora.LowRankWeight):
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if name.startswith("B_"):
                        p.fill_(.03)
    before = lora.weight_movement(model)
    assert before > 0
    for module in model.modules():
        if isinstance(module, lora.LowRankWeight):
            with torch.no_grad():
                for projection in module.projections:
                    getattr(module, "A_" + projection).mul_(2)
                    getattr(module, "B_" + projection).div_(2)
    assert lora.weight_movement(model) == pytest.approx(before, rel=1e-7)


@pytest.mark.parametrize("regressor", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_actual_tabicl_training_and_native_roundtrip(regressor, recompute):
    model = _tiny_tabicl(regressor, recompute=recompute)
    native = copy.deepcopy(model)
    x = torch.randn(1, 14, 4)
    y = torch.linspace(0, 1, 8)[None] if regressor else torch.tensor([[0., 1.] * 4])
    original = [(p, p.detach().clone()) for p in model.parameters()]
    # Freezing alone changes native CPU numerical paths by ~1e-7 even under
    # no_grad. Test both exact equality with matching flags and bounded FP32
    # equality with the fully trainable original; do not hide a weight change.
    with torch.no_grad():
        before = model(x, y).detach()
        frozen_before = copy.deepcopy(model).requires_grad_(False)(x, y).detach()
    params = lora.inject(model, family="tabicl", rank=2, alpha=4)
    with torch.no_grad():
        torch.testing.assert_close(model(x, y), frozen_before, rtol=0, atol=0)
        torch.testing.assert_close(model(x, y), before, rtol=1e-5, atol=1e-6)
    assert all(torch.equal(lora.merged_state_dict(model)[key], value)
               for key, value in native.state_dict().items())
    assert {t["module"].split(".")[0] for t in lora.metadata(model)["targets"]} == {
        "col_embedder", "row_interactor", "icl_predictor"}
    opt = torch.optim.Adam(params, lr=.02)
    model(x, y).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert any(torch.count_nonzero(p.grad) for p in params)
    opt.step()
    assert not torch.equal(before, model(x, y))
    assert all(not p.requires_grad and torch.equal(p, value) for p, value in original)
    native.load_state_dict(lora.merged_state_dict(model), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(native(x, y), model(x, y), rtol=1e-5, atol=1e-6)
        native.requires_grad_(False)
        torch.testing.assert_close(native(x, y), model(x, y), rtol=0, atol=0)


def test_actual_tabicl_packed_keys_unchanged_and_cache_predictions_match():
    from tabicl._model.kv_cache import KVCacheEntry
    model = _tiny_tabicl()
    lora.inject(model, family="tabicl", rank=2, alpha=4)
    target = lora.metadata(model)["targets"][-1]
    attention = model.get_submodule(target["module"])
    dim = attention.embed_dim
    original = attention.parametrizations.in_proj_weight.original.detach().clone()
    with torch.no_grad():
        for p in attention.parametrizations.in_proj_weight[0].parameters():
            p.add_(.03)
    weight = attention.in_proj_weight
    assert torch.equal(weight[dim:2 * dim], original[dim:2 * dim])
    assert not torch.equal(weight[:dim], original[:dim])
    assert not torch.equal(weight[2 * dim:], original[2 * dim:])
    query, context = torch.randn(1, 3, dim), torch.randn(1, 7, dim)
    direct, keys, values = attention(query, key=context, value=context, need_kv=True)
    cached = attention(query, cached_kv=KVCacheEntry(key=keys, value=values))
    torch.testing.assert_close(cached, direct, rtol=1e-6, atol=1e-6)
    copied = copy.deepcopy(model)
    assert lora.is_lora(copied)
    torch.testing.assert_close(copied.get_submodule(target["module"])(query, key=context, value=context),
                               direct, rtol=0, atol=0)


def test_recovery_preserves_unmerged_factors_optimizer_and_exact_next_update(tmp_path: Path):
    from src.train.recovery import save_recovery, load_recovery
    model = _v3()
    _inject_v3(model, rank=2)
    x, y = torch.randn(2, 5, 8), torch.randn(2, 5, 2)
    optimizer = torch.optim.Adam(lora.adapter_parameters(model), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    F.mse_loss(model(x), y).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    path = tmp_path / "resume.pt"
    save_recovery(path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                  identity={"rank": 2, "targets": "attention_qv"}, progress={"updates": 1})
    resumed = _v3()
    _inject_v3(resumed, rank=2)
    optimizer2 = torch.optim.Adam(lora.adapter_parameters(resumed), lr=.01)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=.9)
    assert load_recovery(path, model=resumed, optimizer=optimizer2, scheduler=scheduler2,
                         scaler=scaler, identity={"rank": 2, "targets": "attention_qv"}) == {"updates": 1}
    for candidate, opt, sched in ((model, optimizer, scheduler), (resumed, optimizer2, scheduler2)):
        F.mse_loss(candidate(x), y).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in model.state_dict().items())


def test_production_optimizer_selects_only_lora_parameters():
    from omegaconf import OmegaConf
    from src.train.loop import _make_optimizer_and_scheduler
    from src.train.optimization import step_mean_gradient
    model = _v3()
    original = [(p, p.detach().clone()) for p in model.parameters()]
    params = _inject_v3(model, rank=2)
    cfg = OmegaConf.create({"optimizer": {"lr": .01, "weight_decay": 0},
                            "scheduler": {"warmup_fraction": .1, "min_lr_fraction": .05}})
    optimizer, _ = _make_optimizer_and_scheduler(model, cfg, total_steps=10)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in params}
    optimizer.param_groups[0]["lr"] = .01
    model(torch.randn(1, 6, 8)).square().mean().backward()
    norm, success = step_mean_gradient(model, optimizer, torch.amp.GradScaler("cpu", enabled=False),
                                       microbatches=1, max_norm=1)
    assert success and norm > 0
    assert all(torch.equal(p, before) for p, before in original)
    assert all(p.grad is None for p in model.parameters())


def test_actual_tabicl_production_saver_native_reload_and_live_state(tmp_path):
    from src.train.tabicl_model import save_finetuned_tabicl, load_tabicl_for_training
    model = _tiny_tabicl()
    lora.inject(model, family="tabicl", rank=2, alpha=4)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, lora.LowRankWeight):
                for name in module.projections:
                    getattr(module, "B_" + name).fill_(.02)
    cfg = dict(max_classes=2, num_quantiles=9, embed_dim=16,
               col_num_blocks=1, col_nhead=2, col_num_inds=4,
               row_num_blocks=1, row_nhead=2, row_num_cls=2,
               icl_num_blocks=1, icl_nhead=2, ff_factor=1, dropout=0., recompute=False)
    path = tmp_path / "tabicl-classifier-v2-test.ckpt"
    state = copy.deepcopy(model.state_dict())
    save_finetuned_tabicl(model, cfg, path, provenance={"adaptation_mode": "lora"})
    saved = torch.load(path, weights_only=True)
    assert not any("parametrizations" in key for key in saved["state_dict"])
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    loaded, _ = load_tabicl_for_training(path, track="pd", device="cpu")
    x, y = torch.randn(1, 14, 4), torch.tensor([[0., 1.] * 4])
    with torch.no_grad():
        torch.testing.assert_close(loaded(x, y), model(x, y), rtol=1e-5, atol=1e-6)
    model(x, y).square().mean().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in lora.adapter_parameters(model))

