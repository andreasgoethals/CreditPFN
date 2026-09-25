"""The GPU diagnostic must isolate repeated calculations without training or writing."""
from pathlib import Path

import pytest
import torch

from src.utils import gpu_repeatability as probe


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.dropout = torch.nn.Dropout(.4)
        self.register_buffer("visits", torch.zeros(()))

    def forward(self, x):
        self.visits += 1
        return self.linear(self.dropout(x)) + self.visits


def test_repeated_backward_restores_weights_buffers_auxiliary_and_rng():
    model, auxiliary = Toy(), Toy()
    x = torch.ones(8, 4)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()

    def loss():
        auxiliary.visits += 1
        return (model(x) + auxiliary.visits).square().mean()

    result = probe.repeated_backward(model, loss, device="cpu", auxiliary=(auxiliary,))
    assert result["repeatable"] and result["gradient_tensors"] == 2
    assert len(result["comparisons"]) == 2
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, initial[k]) for k, value in model.state_dict().items())
    assert auxiliary.visits == 0
    assert all(p.grad is None for p in model.parameters())


def test_repeated_backward_detects_uncontrolled_variation():
    model = torch.nn.Linear(1, 1, bias=False)
    calls = []

    def loss():
        calls.append(1)
        return model(torch.ones(2, 1)).sum() * len(calls)

    result = probe.repeated_backward(model, loss, device="cpu")
    assert not result["repeatable"]
    assert result["comparisons"][0]["changed_gradient_tensors"] == 1
    assert result["comparisons"][0]["max_gradient_absolute_difference"] == 2


def test_repeated_backward_restores_state_after_failure():
    model = Toy()
    rng = torch.get_rng_state().clone()

    def fail():
        model(torch.ones(2, 4))
        raise RuntimeError("kernel refusal")

    with pytest.raises(RuntimeError, match="kernel refusal"):
        probe.repeated_backward(model, fail, device="cpu")
    assert model.visits == 0 and torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("loss_kind", ["nonfinite", "no_parameter_gradients"])
def test_bad_measurements_cannot_be_reported_repeatable(loss_kind):
    model = torch.nn.Linear(1, 1)

    def loss():
        if loss_kind == "nonfinite":
            return model(torch.ones(1, 1)).sum() * float("nan")
        return torch.ones((), requires_grad=True)

    with pytest.raises(RuntimeError, match="[Nn]onfinite|[Mm]issing"):
        probe.repeated_backward(model, loss, device="cpu")


@pytest.mark.parametrize("name", probe.PROFILES)
def test_kernel_profile_restores_process_settings_even_on_error(name):
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.backends.cudnn.benchmark, torch.backends.cuda.flash_sdp_enabled())
    with pytest.raises(RuntimeError, match="test exit"):
        with probe.kernel_profile(name):
            assert torch.are_deterministic_algorithms_enabled() == (name != "default_kernels")
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            if name == "deterministic_math":
                assert not torch.backends.cuda.flash_sdp_enabled()
                assert not torch.backends.cuda.matmul.allow_tf32
            raise RuntimeError("test exit")
    after = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cuda.flash_sdp_enabled())
    assert after == before


def test_cpu_cli_refuses_before_loading_a_model(monkeypatch):
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(probe, "probe_loss", lambda *a, **kw: pytest.fail("No GPU was allocated"))
    with pytest.raises(SystemExit) as error:
        probe.main([])
    assert error.value.code == 2


@pytest.mark.parametrize("family", ["tabpfn", "tabicl"])
@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_synthetic_batch_uses_native_axis_order_and_training_loss(monkeypatch, family, track):
    from src.train import model as pfn, tabicl_model as icl, loop

    class Network(torch.nn.Module):
        def forward(self, x, y):
            assert x.shape == (2, 20, 4) and y.shape == (2, 12)
            if track == "pd":
                assert (y[:, 0] == 0).all() and (y[:, 1] == 1).all()
            return torch.ones(2, 8, 10)

    model = Network()
    class RegressionLoss(torch.nn.Module):
        def forward(self, *, logits, y):
            assert logits.shape == (8, 2, 10) and y.shape == (8, 2)
            return logits.mean(-1) - y

    criterion = torch.nn.CrossEntropyLoss() if track == "pd" else RegressionLoss()
    monkeypatch.setattr(icl, "load_tabicl_for_training", lambda *a, **kw: (model, {}))
    monkeypatch.setattr(pfn, "load_tabpfn_for_training", lambda *a, **kw: (model, criterion, {}, {}))
    monkeypatch.setattr(icl, "tabicl_pinball_loss", lambda out, target: target.mean())

    forwarded = []
    def forward(network, *, X_ctx, y_ctx, X_qry, **kwargs):
        assert X_ctx.shape == (12, 1, 4)
        assert X_qry.shape == (8, 1, 4)
        if track == "pd":
            assert (y_ctx[0] == 0).all() and (y_ctx[1] == 1).all()
        forwarded.append(X_ctx)
        return torch.ones(8, 1, 10)

    monkeypatch.setattr(loop, "_forward_one_member", forward)
    loaded, make_loss, auxiliary = probe.probe_loss(Path(f"{family}-v2.ckpt"), track,
        rows=20, features=4, members=2, query_fraction=.4, device="cpu")
    assert loaded is model and torch.isfinite(make_loss())
    assert len(forwarded) == (2 if family == "tabpfn" else 0)
    assert auxiliary == ((criterion,) if family == "tabpfn" else ())
