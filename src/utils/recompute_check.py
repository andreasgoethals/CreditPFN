"""Numerical and backward-execution checks for native activation recomputation."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from src.train.loop import _ensemble_step_loss


def compare_recomputation(model, batch, criterion, *, device, dtypes):
    """Compare two-member outputs, losses, gradients and AdamW updates.

    The recomputed arm also reloads model/optimizer state after its first update.
    No rows, ensemble members or precision settings differ between the arms.
    Block-entry observations reject APIs that silently ignore the request.
    Only synthetic inputs and compact numeric diagnostics are returned.
    """
    if len(batch.members) != 2 or not getattr(model, "blocks", None):
        raise AssertionError("Recomputation acceptance needs two members and native blocks")
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    original_training = model.training
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    results = []
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        for dtype in dtypes:
            arms, block_calls = {}, {}
            for enabled in (False, True):
                model.load_state_dict(initial)
                model.train()
                model.zero_grad(set_to_none=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-7, weight_decay=0)
                count, outputs, snapshots, observing = [0], [], [], [False]

                def enter(*args):
                    count[0] += 1

                def output(module, args, value):
                    if observing[0]:
                        outputs.append(value.detach().float().cpu().clone())

                block_hook = model.blocks[0].register_forward_pre_hook(enter)
                output_hook = model.register_forward_hook(output)
                try:
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(518)
                        for step in range(3):
                            outputs.clear()
                            optimizer.zero_grad(set_to_none=True)
                            observing[0] = True
                            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                                loss = _ensemble_step_loss(model, batch, criterion=criterion,
                                                           activation_checkpointing=enabled)
                            observing[0] = False
                            loss.backward()
                            gradients = {name: p.grad.detach().float().cpu().clone()
                                         for name, p in model.named_parameters() if p.grad is not None}
                            if not gradients or not any(torch.count_nonzero(g) for g in gradients.values()):
                                raise AssertionError("Recomputation probe did not produce nonzero gradients")
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            weights = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
                            snapshots.append(dict(loss=loss.detach().float().cpu(),
                                                  outputs=list(outputs), gradients=gradients, weights=weights))
                            if enabled and step == 0:
                                # Model and optimizer reload must preserve the next two updates.
                                restored = copy.deepcopy(optimizer.state_dict())
                                model.load_state_dict(weights)
                                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-7, weight_decay=0)
                                optimizer.load_state_dict(restored)
                    arms[enabled], block_calls[enabled] = snapshots, count[0]
                finally:
                    output_hook.remove()
                    block_hook.remove()
            if block_calls[False] != 6 or block_calls[True] <= block_calls[False]:
                raise AssertionError("Native layers were not actually recomputed during backward")
            errors = dict(loss=0.0, outputs=0.0, gradients=0.0, weights=0.0)
            for left, right in zip(arms[False], arms[True], strict=True):
                for key in errors:
                    a, b = left[key], right[key]
                    if isinstance(a, dict):
                        if a.keys() != b.keys():
                            raise AssertionError("Recomputed parameter/gradient identities differ")
                        pairs = [(a[k], b[k]) for k in a]
                    elif isinstance(a, list):
                        pairs = list(zip(a, b, strict=True))
                    else:
                        pairs = [(a, b)]
                    for x, y in pairs:
                        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                            raise AssertionError("Nonfinite recomputation evidence")
                        torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)
                        errors[key] = max(errors[key], float((x - y).abs().max()))
            results.append(dict(dtype=str(dtype or torch.float32), passed=True, updates=3,
                                reloaded_after_update=1, block_calls={str(k): v for k, v in block_calls.items()},
                                max_abs_error=errors))
    finally:
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        model.train(original_training)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return dict(passed=True, model_class=type(model).__name__, members=2, comparisons=results)


def native_probe(cfg):
    """Execute the accepted VSC TabPFN code, in both FP32 and training BF16."""
    import gc
    from src.train.model import load_tabpfn_for_training
    from src.utils.paths import resolve_base_checkpoint
    if cfg.track != "pd" or not torch.cuda.is_available():
        raise RuntimeError("The enabled recomputation extension requires a native PD GPU probe")
    model, criterion, _, _ = load_tabpfn_for_training(
        resolve_base_checkpoint(cfg.tunable.classifier_base_paths[0]), track="pd", device="cuda")
    rng = torch.Generator(device="cuda").manual_seed(519)
    x = torch.randn(128, 1, 12, generator=rng, device="cuda")
    y = (x[:, 0, 0] > 0).long()
    members = [SimpleNamespace(X_context=x[:80], X_query=x[80:],
                y_context=(y[:80] if i == 0 else 1 - y[:80]).reshape(80, 1, 1),
                class_permutation=None if i == 0 else [1, 0],
                categorical_idx=[], outlier_removal_std=None) for i in range(2)]
    batch = SimpleNamespace(members=members, task_type="classification", n_classes=2, y_query=y[80:])
    try:
        return compare_recomputation(model, batch, criterion, device="cuda", dtypes=[None, torch.bfloat16])
    finally:
        del model, criterion, x, y, batch, members
        gc.collect()
        torch.cuda.empty_cache()
