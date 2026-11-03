"""The one-factor Muon arm: partition, scheduling, clipping, recovery and AdamW parity."""
import copy
import math

import pytest
import torch

from src.train.optimization import MuonAdamW, make_optimizer, muon_partition, step_mean_gradient
from src.train.loop import make_warmup_cosine_schedule


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 8)        # input map -> AdamW
        self.block = torch.nn.Linear(8, 8)          # hidden matrix -> Muon
        self.norm = torch.nn.LayerNorm(8)           # 1-D -> AdamW
        self.decoder = torch.nn.Linear(8, 3)        # output map -> AdamW

    def forward(self, x):
        return self.decoder(self.norm(torch.relu(self.block(self.encoder(x)))))


def _loss(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    x, y = torch.randn(32, 4, generator=g), torch.randint(0, 3, (32,), generator=g)
    return torch.nn.functional.cross_entropy(model(x), y)


def test_partition_puts_only_hidden_matrices_on_muon():
    model = Toy()
    matrices, rest = muon_partition(model.named_parameters())
    assert [p.shape for p in matrices] == [torch.Size([8, 8])]
    assert sum(p.numel() for p in matrices) + sum(p.numel() for p in rest) == \
        sum(p.numel() for p in model.parameters())


def test_frozen_parameters_are_not_optimized():
    model = Toy()
    model.encoder.weight.requires_grad_(False)
    _, rest = muon_partition(model.named_parameters())
    assert all(p is not model.encoder.weight for p in rest)


def test_scheduler_and_clipping_act_on_both_inner_optimizers():
    model = Toy()
    opt = make_optimizer(model, list(model.parameters()), name="muon_adamw", lr=1e-3, weight_decay=0.0)
    sched = make_warmup_cosine_schedule(opt, total_steps=10, warmup_fraction=0.1,
                                        schedule_type="warmup_cosine", min_lr_fraction=0.05)
    lrs = {g["lr"] for g in opt.param_groups}
    assert len(opt.param_groups) == len(opt.muon.param_groups) + len(opt.adamw.param_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sched.step()  # warm-up starts at zero; move to the first nonzero rate
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _loss(model).backward()
    norm, finite = step_mean_gradient(model, opt, scaler, microbatches=1, max_norm=1.0)
    assert finite and math.isfinite(norm)
    assert all(not torch.equal(before[n], p) for n, p in model.named_parameters())
    assert {g["lr"] for g in opt.param_groups} != lrs  # warm-up moved every group together


def test_recovery_state_round_trip_reproduces_the_next_update():
    torch.manual_seed(0)
    a = Toy()
    b = Toy()
    b.load_state_dict(a.state_dict())
    oa = MuonAdamW(list(a.named_parameters()), lr=1e-3, weight_decay=0.0)
    for step in range(3):
        _loss(a, step).backward(); oa.step(); oa.zero_grad()
    ob = MuonAdamW(list(b.named_parameters()), lr=1e-3, weight_decay=0.0)
    b.load_state_dict(a.state_dict())
    # Optimizer.load_state_dict may share tensors with its source; copy like a recovery file.
    ob.load_state_dict(copy.deepcopy(oa.state_dict()))
    for model, opt in ((a, oa), (b, ob)):
        _loss(model, 99).backward(); opt.step(); opt.zero_grad()
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(p, q, atol=0, rtol=0), n


def test_zero_learning_rate_leaves_weights_unchanged():
    model = Toy()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = make_optimizer(model, list(model.parameters()), name="muon_adamw", lr=0.0, weight_decay=0.0)
    _loss(model).backward(); opt.step()
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        make_optimizer(Toy(), [], name="sgd", lr=1e-3, weight_decay=0.0)
