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


def test_table_comparison_preserves_state_and_reports_nonfinite_profiles(monkeypatch):
    from src.train import tabpfn_preprocessing as preprocessing

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 10)
            self.register_buffer("visits", torch.zeros(()))

        def forward(self, x, y, **kwargs):
            self.visits += 1
            return self.linear(x[len(y):])

    model = Classifier()
    batch = preprocessing.TabPFNEnsembleBatch(members=[preprocessing._PerEstimatorView(
        X_context=torch.ones(4, 1, 4), y_context=torch.arange(4).remainder(2).reshape(-1, 1, 1),
        X_query=torch.ones(3, 1, 4), categorical_idx=[], class_permutation=None,
        outlier_removal_std=12.)], y_query=torch.tensor([0, 1, 0]).reshape(-1, 1, 1),
        task_type="classification", dataset_id="synthetic", n_classes=2)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    original = preprocessing.apply_outlier_clip
    monkeypatch.setattr(probe, "upstream_clip", lambda x, **kwargs: x * float("nan"))

    result = probe.compare_table_batch(model, torch.nn.CrossEntropyLoss(), batch, device="cpu")
    assert result["profiles"]["current_bf16"]["status"] == "measured"
    assert result["profiles"]["current_fp32"]["status"] == "measured"
    for name in ("upstream_bf16", "upstream_fp32"):
        assert result["profiles"][name]["status"] == "error"
        assert result["profiles"][name]["clipped_inputs"][0]["nan"] == 28
    assert preprocessing.apply_outlier_clip is original
    assert all(torch.equal(value, initial[key]) for key, value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in model.parameters())


def test_table_probe_keeps_corpus_index_seed_and_bounds_rows(monkeypatch):
    from types import SimpleNamespace
    from src.train import corpus, dataloader, model as loading

    refs = [SimpleNamespace(dataset_id=name) for name in ("0002.synthetic", "0011.synthetic")]
    cfg = SimpleNamespace(track="pd", seed=42, train=SimpleNamespace(context_sampling="balanced"),
                          tunable=SimpleNamespace(query_fractions=[.4]))
    monkeypatch.setattr(corpus, "split_from_cfg", lambda cfg: SimpleNamespace(train=refs))
    network = torch.nn.Linear(1, 1)
    monkeypatch.setattr(loading, "load_tabpfn_for_training",
        lambda *args, **kwargs: (network, torch.nn.CrossEntropyLoss(), {}, "inference"))

    class Loader:
        def __init__(self, actual_refs, **kwargs):
            assert actual_refs is refs
            assert kwargs["seed"] == 42 and kwargs["max_rows_per_epoch"] == 2048
            assert kwargs["inference_config"] == "inference"
            assert kwargs["context_sampling"] == "balanced"

        def __getitem__(self, key):
            assert key == (0, 1)
            return SimpleNamespace(to=lambda device: "batch")

    monkeypatch.setattr(dataloader, "ProcessedDatasetLoader", Loader)
    monkeypatch.setattr(probe, "compare_table_batch", lambda model, criterion, batch, **kw: {"batch": batch})
    assert probe.table_probe(Path("unused"), cfg, table_number=11,
        rows=2048, members=2, device="cpu") == {"batch": "batch"}
    monkeypatch.setattr(loading, "load_tabpfn_for_training", lambda *a, **kw: pytest.fail("invalid table"))
    with pytest.raises(ValueError, match="training table"):
        probe.table_probe(Path("unused"), cfg, table_number=1, rows=2048, members=2, device="cpu")


def test_upstream_clip_fits_context_and_preserves_categoricals(monkeypatch):
    import sys
    from types import SimpleNamespace

    class Clip:
        def __init__(self, n_sigma):
            assert n_sigma == 12

        def __call__(self, x, num_train_rows):
            assert x.shape == (6, 1, 2) and num_train_rows == 4
            return x / 2

    monkeypatch.setitem(sys.modules, "tabpfn.preprocessing.torch.torch_soft_clip_outliers",
                        SimpleNamespace(TorchSoftClipOutliers=Clip))
    x = torch.ones(6, 1, 3)
    actual = probe.upstream_clip(x, n_sigma=12, categorical_idx=[1], context_rows=4)
    assert torch.equal(actual[..., 1], x[..., 1])
    assert (actual[..., [0, 2]] == .5).all() and (x == 1).all()


@pytest.mark.parametrize("args", [["--table-number", "0"], ["--table-number", "11", "--base", "tabicl"]])
def test_invalid_real_table_request_refuses_before_cuda(args, monkeypatch):
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: pytest.fail("invalid arguments"))
    with pytest.raises(SystemExit) as error:
        probe.main(args)
    assert error.value.code == 2


def test_tensor_summary_handles_missing_values_without_dumping_rows():
    summary = probe.tensor_summary(torch.tensor([float("nan"), float("inf"), -3.]))
    assert summary == {"shape": [3], "nan": 1, "inf": 1, "max_finite_absolute": 3.}
    assert probe.tensor_summary(torch.tensor([float("nan")]))["max_finite_absolute"] is None


@pytest.mark.parametrize("amp", [False, True])
def test_precision_comparison_actually_switches_autocast(amp, monkeypatch):
    from contextlib import nullcontext

    flags = []
    def autocast(device_type, *, dtype, enabled: bool):
        flags.append(enabled)
        return nullcontext()
    monkeypatch.setattr(probe.torch.amp, "autocast", autocast)
    monkeypatch.setattr(probe.torch.cuda, "synchronize", lambda: None)
    model = torch.nn.Linear(1, 1)
    probe.repeated_backward(model, lambda: model(torch.ones(2, 1)).square().mean(),
                            device="cuda", amp=amp, repeats=2)
    assert flags == [amp, amp]
