"""Isolate GPU forward/backward failures without optimizer updates or output files.

The default uses a synthetic batch. A numbered training table instead compares
production and upstream clipping (including float64 clipping calculations),
with BF16 and FP32 model arithmetic.
Neither diagnostic can grant a passing experiment-0 receipt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.train.recovery import capture_rng, restore_rng


PROFILES = ("default_kernels", "deterministic_kernels", "deterministic_math")


@contextmanager
def kernel_profile(name: str):
    if name not in PROFILES:
        raise ValueError(f"Unknown kernel profile: {name}")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(name != "default_kernels", warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = name != "deterministic_math"
        torch.backends.cudnn.allow_tf32 = name != "deterministic_math"
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention = sdpa_kernel(SDPBackend.MATH) if name == "deterministic_math" else nullcontext()
        with attention:
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def repeated_backward(model, make_loss, *, device: str, auxiliary=(), repeats: int = 3,
                      amp: bool = True) -> dict:
    """Restore parameters, buffers and RNG before each backward; compare CPU gradients."""
    if repeats < 2:
        raise ValueError("At least two repeats are needed")
    modules = (model, *auxiliary)
    states = [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()} for module in modules]
    modes = [(module, module.training) for parent in modules for module in parent.modules()]
    rng = capture_rng()
    cuda = torch.device(device).type == "cuda"
    reference = None
    first_loss = None
    comparisons = []
    started = time.monotonic()

    def reset():
        for module, state in zip(modules, states):
            module.load_state_dict(state, strict=True)
            module.zero_grad(set_to_none=True)
        for module, training in modes:
            module.training = training
        restore_rng(rng)

    try:
        for index in range(repeats):
            reset()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cuda and amp):
                loss = make_loss()
            if loss.numel() != 1 or not torch.isfinite(loss).all():
                raise RuntimeError("Nonfinite or nonscalar probe loss")
            loss.backward()
            if cuda:
                torch.cuda.synchronize()
            scalar_loss = float(loss.detach())
            gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
                         if p.grad is not None}
            del loss
            if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
                raise RuntimeError("Missing or nonfinite probe gradients")
            if reference is None:
                reference, first_loss = gradients, scalar_loss
                continue
            if gradients.keys() != reference.keys():
                raise RuntimeError("Different parameters received gradients between repeats")
            changed = []
            largest, largest_name = 0.0, None
            for name, gradient in gradients.items():
                if not torch.equal(gradient, reference[name]):
                    changed.append(name)
                    delta = float((gradient.double() - reference[name].double()).abs().max())
                    if delta > largest:
                        largest, largest_name = delta, name
            comparisons.append({"repeat": index + 1, "loss_absolute_difference": abs(scalar_loss - first_loss),
                "gradient_bitwise_equal": not changed, "changed_gradient_tensors": len(changed),
                "max_gradient_absolute_difference": largest, "max_gradient_tensor": largest_name})
        return {"status": "measured", "loss": first_loss, "gradient_tensors": len(reference),
                "repeatable": all(c["gradient_bitwise_equal"] and c["loss_absolute_difference"] == 0
                                  for c in comparisons),
                "comparisons": comparisons, "seconds": time.monotonic() - started}
    finally:
        reset()


def probe_loss(checkpoint: Path, track: str, *, rows: int, features: int, members: int,
               query_fraction: float, device: str):
    """Use the actual model loader and training loss on one fixed synthetic batch."""
    from src.train.tabicl_compat import model_family
    n_context = rows - max(1, min(rows - 2, int(rows * query_fraction)))
    generator = torch.Generator().manual_seed(31415)
    x = torch.randn(rows, members, features, generator=generator).to(device)
    y = torch.rand(rows, members, 1, generator=generator)
    if track == "pd":
        y = (y < .15).long()
        y[0], y[1] = 0, 1  # Every member's context contains both classes.
    y = y.to(device)
    if model_family(str(checkpoint)) == "tabicl":
        from src.train.tabicl_model import load_tabicl_for_training, tabicl_pinball_loss
        model, _ = load_tabicl_for_training(checkpoint, track=track, device=device)
        inputs = x.permute(1, 0, 2).contiguous()
        targets = y[..., 0].T.contiguous().float()

        def make_loss():
            output = model(inputs, targets[:, :n_context])
            if track == "pd":
                return torch.nn.functional.cross_entropy(output[..., :2].reshape(-1, 2),
                                                         targets[:, n_context:].long().reshape(-1))
            return tabicl_pinball_loss(output, targets[:, n_context:])

        auxiliary = ()
    else:
        from src.train.model import load_tabpfn_for_training
        from src.train.tabpfn_preprocessing import TabPFNEnsembleBatch, _PerEstimatorView
        from src.train.loop import _ensemble_step_loss
        model, criterion, _, _ = load_tabpfn_for_training(str(checkpoint), track=track, device=device)
        # Synthetic views bypass CPU preprocessing but use the production
        # ensemble forward/loss, including its member/query reduction shape.
        views = [_PerEstimatorView(X_context=x[:n_context, i:i+1],
            y_context=y[:n_context, :1], X_query=x[n_context:, i:i+1],
            categorical_idx=[], class_permutation=None, outlier_removal_std=None)
            for i in range(members)]
        batch = TabPFNEnsembleBatch(members=views, y_query=y[n_context:, :1],
            task_type="classification" if track == "pd" else "regression",
            dataset_id="synthetic_repeatability", n_classes=2 if track == "pd" else None)

        def make_loss():
            return _ensemble_step_loss(model, batch, criterion=criterion)

        auxiliary = (criterion,)
    model.train()
    return model, make_loss, auxiliary


def tensor_summary(value: torch.Tensor) -> dict:
    """Aggregate numerical diagnostics only; never emit feature values or rows."""
    finite = torch.isfinite(value)
    return {"shape": list(value.shape), "nan": int(torch.isnan(value).sum()),
            "inf": int(torch.isinf(value).sum()),
            "max_finite_absolute": float(value[finite].abs().max()) if finite.any() else None}


def upstream_clip(x, *, n_sigma, categorical_idx, context_rows, wide_stats=False):
    """Test upstream clipping; optionally widen its calculations, not model inputs."""
    if n_sigma is None:
        return x
    from tabpfn.preprocessing.torch.torch_soft_clip_outliers import TorchSoftClipOutliers

    numerical = [i for i in range(x.shape[-1]) if i not in categorical_idx]
    result = x.clone()
    if numerical:
        values = x[..., numerical]
        # Squaring extreme finite float32 values can overflow during bound fitting.
        # This diagnostic tests wider clipping math without changing model precision.
        if wide_stats:
            values = values.double()
        clipped = TorchSoftClipOutliers(n_sigma=n_sigma)(values, num_train_rows=context_rows)
        result[..., numerical] = clipped.to(x.dtype)
    return result


def compare_table_batch(model, criterion, batch, *, device: str) -> dict:
    """Compare the same batch/RNG/state; only clipping and precision may differ."""
    from unittest.mock import patch
    from src.train import loop, tabpfn_preprocessing

    context_rows = batch.members[0].X_context.shape[0]
    inputs = [{"context": tensor_summary(m.X_context), "query": tensor_summary(m.X_query),
               "context_labels": tensor_summary(m.y_context),
               "categorical_columns": len(m.categorical_idx),
               "outlier_removal_std": m.outlier_removal_std} for m in batch.members]
    report = {"inputs": inputs, "query_labels": tensor_summary(batch.y_query), "profiles": {}}
    print(json.dumps({"event": "table_inputs", "inputs": inputs,
                      "query_labels": report["query_labels"]}, allow_nan=False), flush=True)
    original_clip = tabpfn_preprocessing.apply_outlier_clip
    for clipping in ("current", "upstream", "upstream_float64"):
        for amp in (True, False):
            name = f"{clipping}_{'bf16' if amp else 'fp32'}"
            clipped_inputs = []

            def clip(x, *, n_sigma, categorical_idx):
                result = (original_clip(x, n_sigma=n_sigma, categorical_idx=categorical_idx)
                          if clipping == "current" else upstream_clip(x, n_sigma=n_sigma,
                              categorical_idx=categorical_idx, context_rows=context_rows,
                              wide_stats=clipping == "upstream_float64"))
                # Only the first forward's member summaries are needed.
                if len(clipped_inputs) < len(batch.members):
                    clipped_inputs.append(tensor_summary(result))
                return result

            print(f"Checking real table: {name}", flush=True)
            try:
                with patch.object(tabpfn_preprocessing, "apply_outlier_clip", clip):
                    result = repeated_backward(model,
                        lambda: loop._ensemble_step_loss(model, batch, criterion=criterion),
                        device=device, auxiliary=(criterion,), repeats=2, amp=amp)
            except (RuntimeError, ValueError, ImportError) as exc:
                # TabPFN reports encoded-input NaNs as ValueError, not RuntimeError.
                # Retain the failed measurement and continue the other settings.
                result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:1200]}
            result["clipped_inputs"] = clipped_inputs
            report["profiles"][name] = result
            print(json.dumps({"profile": name, **result}, allow_nan=False), flush=True)
            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    return report


def table_probe(checkpoint, cfg, *, table_number: int, rows: int, members: int, device: str) -> dict:
    from src.train.corpus import split_from_cfg
    from src.train.dataloader import ProcessedDatasetLoader
    from src.train.model import load_tabpfn_for_training

    # Keep the entire ordered training corpus: its index participates in each
    # table's sampling/preprocessing seed. Filtering it first changes the batch.
    refs = split_from_cfg(cfg).train
    indices = [i for i, ref in enumerate(refs) if ref.dataset_id.split('.', 1)[0] == f"{table_number:04d}"]
    if len(indices) != 1:
        raise ValueError("Table number must identify exactly one training table in this partition")
    model, criterion, _, inference_config = load_tabpfn_for_training(
        str(checkpoint), track=str(cfg.track), device=device)
    loader = ProcessedDatasetLoader(refs, max_rows_per_epoch=rows,
        query_fraction=float(cfg.tunable.query_fractions[0]), seed=int(cfg.seed),
        inference_config=inference_config, n_estimators_finetune=members,
        context_sampling=str(cfg.train.context_sampling))
    batch = loader[(0, indices[0])].to(device)
    model.train()
    return compare_table_batch(model, criterion, batch, device=device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("pd", "lgd"), default="pd")
    parser.add_argument("--base", choices=("v2", "v2.6", "v3", "tabicl"), default="v2")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--table-number", type=int,
        help="Use this numbered credit training table instead of synthetic inputs (TabPFN only)")
    args = parser.parse_args(argv)
    if not 4 <= args.rows <= 2048:
        parser.error("Diagnostic row count must be between 4 and 2048; this is not a capacity probe")
    if args.table_number is not None and (args.table_number < 1 or args.base == "tabicl"):
        parser.error("Real-table diagnosis requires a positive table number and a TabPFN base")
    # Set before creating a CUDA/cuBLAS context. This workspace applies to all profiles.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        parser.error("This diagnostic requires an allocated CUDA GPU; no model was loaded")
    if not torch.cuda.is_bf16_supported():
        parser.error("The campaign's BF16 diagnostic requires BF16 GPU support")
    from src.utils.logging_setup import configure_warning_filters
    from src.train.config import load_train_config, training_members
    from src.utils.paths import resolve_base_checkpoint
    configure_warning_filters()
    cfg = load_train_config(config_path=f"config/experiment0/recovery_{args.track}.yaml")
    prefix = "tabicl-" if args.base == "tabicl" else f"tabpfn-{args.base}-"
    bases = cfg.tunable.classifier_base_paths if args.track == "pd" else cfg.tunable.regressor_base_paths
    matches = [str(p) for p in bases if Path(str(p)).name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one configured {args.base} base, found {len(matches)}")
    checkpoint = resolve_base_checkpoint(matches[0])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Base checkpoint absent: {checkpoint}")
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    members = training_members(cfg, args.track, "tabicl" if args.base == "tabicl" else "tabpfn")
    if args.table_number is not None:
        with kernel_profile("default_kernels"):
            report = table_probe(checkpoint, cfg, table_number=args.table_number,
                rows=args.rows, members=members, device="cuda")
        report.update(action="table_loss_diagnostic", base=checkpoint.name, track=args.track,
            table_number=args.table_number, epoch=0, row_cap=args.rows, members=members, seed=seed,
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            optimizer_updates=0, checkpoint_writes=0, workflow_modified=False)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        # A complete diagnostic is useful even when every profile exposes a failure.
        # The individual statuses are measurements, never a passing training receipt.
        return 0
    model, make_loss, auxiliary = probe_loss(checkpoint, args.track, rows=args.rows,
        features=16, members=members, query_fraction=float(cfg.tunable.query_fractions[0]), device="cuda")
    report = {"action": "gpu_repeatability", "base": checkpoint.name, "track": args.track,
        "batch": "synthetic", "rows": args.rows, "features": 16, "members": members, "seed": seed,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "precision": "BF16 autocast", "optimizer_updates": 0, "checkpoint_writes": 0, "profiles": {}}
    for name in PROFILES:
        print(f"Checking {name}", flush=True)
        try:
            with kernel_profile(name):
                result = repeated_backward(model, make_loss, device="cuda", auxiliary=auxiliary)
        except RuntimeError as exc:
            # A deterministic-kernel refusal is useful evidence, never a passing result.
            result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:1200]}
        report["profiles"][name] = result
        print(f"{name}: {result['status']}, repeatable={result.get('repeatable', 'unmeasured')}", flush=True)
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    report["note"] = "Synthetic kernel diagnostic only; it does not certify experiment-0 recovery equivalence."
    print(json.dumps(report, indent=2), flush=True)
    return 0 if any(p["status"] == "measured" for p in report["profiles"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
