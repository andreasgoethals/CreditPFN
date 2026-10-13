"""DiagScale query multipliers, independent of upstream PEFT forks.

Implements Ning et al., Appendix B.1 and ``inject_diagscale`` in the pinned
TTC snapshot. TabICL's existing SSMax slot is the final query transform before
SDPA, including cached-KV calls; wrapping that slot avoids patching a package.
Adapters stay FP32. TabICL backbone autograd flags are retained for kernel
parity, but only deltas enter the optimizer. Original weights never step.
"""
from __future__ import annotations

from pathlib import Path
from types import MethodType

import torch
from torch import nn

SCHEMA = "creditpfn-diagscale-v1"


class QueryScale(nn.Module):
    def __init__(self, base, shape, device, *, cast_multiplier=True):
        super().__init__()
        self.base = base
        self.cast_multiplier = cast_multiplier
        self.delta = nn.Parameter(torch.zeros(shape, dtype=torch.float32, device=device))

    def forward(self, query, n):
        transformed = self.base(query, n) if self.base is not None else query
        multiplier = 1 + self.delta
        # Match the two released implementations: TabPFN uses normal tensor
        # promotion; TabICL explicitly casts its multiplier to the query dtype.
        if self.cast_multiplier:
            multiplier = multiplier.to(transformed.dtype)
        return transformed * multiplier

    def _apply(self, fn, recurse=True):
        # Inference engines may cast the entire model to BF16 after loading.
        # Move delta with the model but preserve its original FP32 values; doing
        # BF16 -> FP32 afterwards would already have lost small corrections.
        delta = self._parameters.pop('delta')
        try:
            super()._apply(fn, recurse=recurse)
            destination = fn(delta).device
            with torch.no_grad():
                delta.data = delta.data.to(device=destination, dtype=torch.float32)
                if delta.grad is not None:
                    delta.grad.data = delta.grad.data.to(device=destination, dtype=torch.float32)
        finally:
            self._parameters['delta'] = delta
        return self


def adapter_parameters(model):
    return [m.delta for m in model.modules() if isinstance(m, QueryScale)]


def adapter_state(model):
    return {name + ".delta": m.delta.detach().cpu().clone()
            for name, m in model.named_modules() if isinstance(m, QueryScale)}


def inject(model, *, family, base_path=None, base_sha256=None):
    if family not in {"tabpfn", "tabicl"}:
        raise ValueError("DiagScale supports TabPFN v3 and TabICLv2 only")
    if adapter_parameters(model):
        raise ValueError("DiagScale already injected")
    targets = []
    for name, module in model.named_modules():
        if family == "tabpfn" and type(module).__name__ == "SoftmaxScalingMLP":
            ref = next(module.parameters())
            targets.append((name, module, (module.num_heads, module.head_dim), ref.device))
        elif (family == "tabicl" and type(module).__name__ == "MultiheadAttention"
              and hasattr(module, "ssmax_layer")):
            targets.append((name + ".ssmax_layer", module.ssmax_layer,
                            (module.num_heads, 1, module.head_dim), module.in_proj_weight.device))
    if not targets:
        raise ValueError("No compatible query-scaling modules; unsupported architecture/package")
    # Enumerate and validate before mutation; never wrap a descendant twice.
    if family == "tabpfn":
        model.requires_grad_(False)
    for name, base, shape, device in targets:
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent), attr,
                QueryScale(base, shape, device, cast_multiplier=family == "tabicl"))
    model._creditpfn_adapter = dict(schema=SCHEMA, family=family,
                                   base_path=str(base_path) if base_path else None,
                                   base_sha256=base_sha256,
                                   modules=[t[0] for t in targets])
    selected = {id(p) for p in adapter_parameters(model)}
    total = sum(p.numel() for p in model.parameters())
    optimized = sum(p.numel() for p in adapter_parameters(model))
    # Autograd-enabled is not the same as optimizer-selected for TabICL.
    # Replace any previous full/frozen metadata rather than misreporting its
    # untouched backbone as an optimized part of the adapter experiment.
    model._creditpfn_freeze_info = dict(
        mode="diagscale", frozen_modules=[], trainable_params=optimized,
        trainable_fraction=optimized / max(1, total),
        trainable_parameter_names=[name for name, p in model.named_parameters() if id(p) in selected],
        original_weights_updated=False, backbone_autograd_preserved=family == "tabicl",
        autograd_enabled_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return adapter_parameters(model)


def restore(model, state):
    expected = {name + ".delta": m.delta for name, m in model.named_modules()
                if isinstance(m, QueryScale)}
    if set(state) != set(expected):
        raise ValueError("Adapter module names differ from the saved checkpoint")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid adapter shape or values: {name}")
    with torch.no_grad():
        for name, tensor in state.items():
            expected[name].copy_(tensor)


def save_if_adapter(model, path, provenance=None):
    spec = getattr(model, "_creditpfn_adapter", None)
    if spec is None:
        return False
    if not spec.get("base_sha256") or not spec.get("base_path"):
        raise ValueError("Adapter checkpoint requires an immutable original-base reference")
    from src.train.checkpoint_io import atomic_save
    # Deliberately not a native full-weight checkpoint: stock loaders must fail
    # rather than silently discard the adapter and evaluate the base model.
    atomic_save({"creditpfn_adapter": spec, "adapter_state": adapter_state(model),
                 "provenance": provenance}, Path(path), provenance)
    return True


def read_bundle(path, *, family):
    from src.utils.experiment import file_digest
    from src.utils.paths import resolve_base_checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = payload.get("creditpfn_adapter", {})
    if spec.get("schema") != SCHEMA or spec.get("family") != family:
        raise ValueError("Not a compatible DiagScale checkpoint")
    base = resolve_base_checkpoint(spec["base_path"])
    if file_digest(base) != spec["base_sha256"]:
        raise ValueError("Original base checkpoint hash differs; refusing adapter load")
    return payload, base


def make_estimator(task_type, path, *, family, **kwargs):
    """Install adapters immediately after strict base loading, before fitting caches."""
    payload, base = read_bundle(path, family=family)
    if family == "tabpfn":
        from src.model.tabpfn_models import _make_tabpfn
        est = _make_tabpfn(task_type, base, **kwargs)
        method = "_initialize_model_variables"
    else:
        from src.train.tabicl_compat import import_tabicl_sklearn
        cls, reg = import_tabicl_sklearn()
        # Match the paper's explicit TabICL inference precision, at all row counts.
        kwargs.update(use_amp=False, use_fa3=False, allow_auto_download=False)
        if kwargs.get("device") == "auto":
            kwargs["device"] = None
        est = (cls if task_type == "classification" else reg)(model_path=str(base), **kwargs)
        method = "_load_model"
    if not hasattr(est, method):
        raise RuntimeError(f"Upstream adapter initialization hook unavailable: {method}")
    original = getattr(est, method)

    def initialize(instance, *args, **options):
        result = original(*args, **options)
        models = instance.models_ if family == "tabpfn" else [instance.model_]
        for model in models:
            if not adapter_parameters(model):
                inject(model, family=family, base_path=str(base),
                       base_sha256=payload["creditpfn_adapter"]["base_sha256"])
            restore(model, payload["adapter_state"])
        return result

    setattr(est, method, MethodType(initialize, est))
    return est
