"""Native Q/V LoRA for explicit TabPFN v3 and TabICLv2 checkpoints.

The intervention adapts the query and value projections in every transformer
attention stage. Keys, attention output projections, feed-forward layers,
embeddings and prediction heads remain fixed. Retrieval decoders whose values
are labels have no learned value projection and are deliberately excluded.

Weight parametrizations also cover TabICL's packed QKV functional access; a
``Linear.forward`` wrapper would not cover that path. Native merged exports use
the original state-dict schema without mutating the live model. Recovery must
instead retain the ordinary, unmerged ``model.state_dict()``.
"""
from __future__ import annotations

import copy
import math
import re
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize

SCHEMA = "creditpfn-lora-qv-v1"
TARGET_MODULES = "attention_qv"


class LowRankWeight(nn.Module):
    """Add independent rank-r updates to named row slices of a frozen matrix."""

    def __init__(self, weight, *, projections, slices, rank, alpha):
        super().__init__()
        self.projections = tuple(projections)
        self.slices = tuple(tuple(x) for x in slices)
        self.rank = rank
        self.scaling = alpha / rank
        self.shape = tuple(weight.shape)
        for name, (start, stop) in zip(self.projections, self.slices):
            a = nn.Parameter(torch.empty(rank, weight.shape[1], device=weight.device,
                                         dtype=torch.float32))
            b = nn.Parameter(torch.zeros(stop - start, rank, device=weight.device,
                                         dtype=torch.float32))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.register_parameter("A_" + name, a)
            self.register_parameter("B_" + name, b)

    def delta(self):
        """Compute the effective update in FP32, including under autocast."""
        first = next(self.parameters())
        parts = []
        cursor = 0
        with torch.autocast(device_type=first.device.type, enabled=False):
            for name, (start, stop) in zip(self.projections, self.slices):
                if start > cursor:
                    parts.append(first.new_zeros(start - cursor, self.shape[1]))
                parts.append((getattr(self, "B_" + name) @ getattr(self, "A_" + name))
                             * self.scaling)
                cursor = stop
            if cursor < self.shape[0]:
                parts.append(first.new_zeros(self.shape[0] - cursor, self.shape[1]))
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def forward(self, original):
        return original + self.delta().to(dtype=original.dtype)

    def _apply(self, fn, recurse=True):
        # Keep master adapter values in FP32 when a parent is cast to BF16/FP16.
        # Casting and then casting back would irreversibly round small updates.
        parameters = dict(self._parameters)
        self._parameters.clear()
        try:
            super()._apply(fn, recurse=recurse)
            for parameter in parameters.values():
                device = fn(parameter).device
                with torch.no_grad():
                    parameter.data = parameter.data.to(device=device, dtype=torch.float32)
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.data.to(device=device, dtype=torch.float32)
        finally:
            self._parameters.update(parameters)
        return self


def is_lora(model):
    return getattr(model, "_creditpfn_lora", {}).get("schema") == SCHEMA


def metadata(model):
    if not is_lora(model):
        raise ValueError("Model has no CreditPFN LoRA intervention")
    return copy.deepcopy(model._creditpfn_lora)


def adapter_parameters(model):
    return [p for module in model.modules() if isinstance(module, LowRankWeight)
            for p in module.parameters(recurse=False)]


def adapter_state(model):
    selected = {id(p) for p in adapter_parameters(model)}
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()
            if id(p) in selected}


def _targets(model, family):
    if family == "tabpfn":
        if type(model).__name__ != "TabPFNV3":
            raise ValueError("LoRA requires the explicit TabPFN v3 architecture")
        stages = ("feature_distribution_embedder", "column_aggregator", "icl_blocks")
    elif family == "tabicl":
        if type(model).__name__ != "TabICL" or not hasattr(model, "icl_predictor"):
            raise ValueError("LoRA requires the explicit TabICLv2 architecture")
        stages = ("col_embedder", "row_interactor", "icl_predictor")
    else:
        raise ValueError("LoRA supports TabPFN v3 and TabICLv2 only")
    if any(not hasattr(model, stage) for stage in stages):
        raise ValueError("Unsupported model: missing expected transformer stages")
    targets = []
    covered = set()
    for path, module in model.named_modules():
        stage = path.split(".", 1)[0]
        if stage not in stages:
            continue
        if family == "tabpfn":
            has_projections = hasattr(module, "q_projection") or hasattr(module, "v_projection")
            if not has_projections:
                continue
            if type(module).__name__ not in {"Attention", "CrossAttention", "ICLAttention"}:
                raise ValueError(f"Unsupported TabPFN attention module: {path}")
            if any(not isinstance(getattr(module, f"{p}_projection", None), nn.Linear)
                   for p in ("q", "k", "v", "out")):
                raise ValueError(f"Incomplete separate attention projections: {path}")
            for p in ("q", "v"):
                linear = getattr(module, f"{p}_projection")
                targets.append(dict(module=f"{path}.{p}_projection", weight="weight",
                                    shape=list(linear.weight.shape), projections=[p],
                                    slices=[[0, linear.weight.shape[0]]]))
        else:
            if not isinstance(module, nn.MultiheadAttention):
                continue
            weight = module.in_proj_weight
            if (type(module).__name__ != "MultiheadAttention" or not hasattr(module, "ssmax_layer")
                    or weight is None or tuple(weight.shape) != (3 * module.embed_dim, module.embed_dim)):
                raise ValueError(f"Unsupported TabICL packed attention: {path}")
            dim = module.embed_dim
            targets.append(dict(module=path, weight="in_proj_weight", shape=list(weight.shape),
                                projections=["q", "v"], slices=[[0, dim], [2 * dim, 3 * dim]]))
        covered.add(stage)
    if set(stages) != covered:
        raise ValueError(f"LoRA targets missing from transformer stages: {sorted(set(stages) - covered)}")
    for target in targets:
        owner = model.get_submodule(target["module"])
        name = target["weight"]
        if parametrize.is_parametrized(owner, name):
            raise ValueError(f"Weight already parametrized: {target['module']}.{name}")
        weight = getattr(owner, name)
        if not isinstance(weight, nn.Parameter) or not weight.is_floating_point() or weight.ndim != 2:
            raise ValueError(f"Unsupported LoRA target weight: {target['module']}.{name}")
    return targets


def inject(model, *, family, base_path=None, base_sha256=None, rank=16, alpha=32,
           dropout=0.0, target_modules=TARGET_MODULES):
    """Freeze original parameters and install zero-initialized Q/V updates.

    Validation is completed before mutation. ``base_path`` must identify the
    released v3/v2 family when supplied. Tiny architectural tests may omit it;
    production provenance should always include its immutable hash.
    """
    if is_lora(model) or adapter_parameters(model) or hasattr(model, "_creditpfn_adapter"):
        raise ValueError("An adaptation intervention is already installed")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0 or float(dropout) != 0:
        raise ValueError("LoRA requires finite positive alpha and dropout=0")
    if target_modules not in (TARGET_MODULES, [TARGET_MODULES], (TARGET_MODULES,)):
        raise ValueError("LoRA target_modules must be attention_qv")
    if base_path is not None:
        pattern = (r"tabpfn-v3-(?:classifier|regressor)-.+\.ckpt" if family == "tabpfn"
                   else r"tabicl-(?:classifier|regressor)-v2-.+\.ckpt")
        if re.fullmatch(pattern, Path(base_path).name) is None:
            raise ValueError("LoRA requires explicitly versioned TabPFN v3 or TabICLv2 base weights")
    if base_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", base_sha256) is None:
        raise ValueError("Invalid immutable base SHA256")
    targets = _targets(model, family)
    # Build all factors first, so allocation failure does not partly freeze a model.
    adapters = [LowRankWeight(getattr(model.get_submodule(t["module"]), t["weight"]),
                              projections=t["projections"], slices=t["slices"],
                              rank=rank, alpha=float(alpha)) for t in targets]
    original_count = sum(p.numel() for p in model.parameters())
    model.requires_grad_(False)
    for target, adapter in zip(targets, adapters):
        owner = model.get_submodule(target["module"])
        parametrize.register_parametrization(owner, target["weight"], adapter)
    model._creditpfn_lora = dict(schema=SCHEMA, family=family, rank=rank, alpha=float(alpha),
                                dropout=0.0, target_modules=TARGET_MODULES,
                                base_path=str(base_path) if base_path is not None else None,
                                base_sha256=base_sha256, targets=targets,
                                original_parameter_count=original_count,
                                original_weights_updated=False)
    selected = {id(p) for p in adapter_parameters(model)}
    optimized = sum(p.numel() for p in adapter_parameters(model))
    model._creditpfn_freeze_info = dict(
        mode="lora", frozen_modules=["all original parameters"],
        trainable_params=optimized, trainable_fraction=optimized / (original_count + optimized),
        original_weights_updated=False,
        trainable_parameter_names=[name for name, p in model.named_parameters() if id(p) in selected],
        autograd_enabled_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return adapter_parameters(model)


def _installed(model):
    for target in metadata(model)["targets"]:
        owner = model.get_submodule(target["module"])
        sequence = getattr(owner.parametrizations, target["weight"])
        if len(sequence) != 1 or not isinstance(sequence[0], LowRankWeight):
            raise ValueError("LoRA target parametrization changed after injection")
        yield target, owner, sequence


def merged_state_dict(model):
    """Return native weights without changing adapter objects, modes or gradients."""
    state = model.state_dict()
    if not is_lora(model):
        return state
    merged = OrderedDict(state)
    for target, owner, sequence in _installed(model):
        prefix = f"{target['module']}.parametrizations.{target['weight']}."
        for key in list(merged):
            if key.startswith(prefix):
                del merged[key]
        with torch.no_grad():
            value = getattr(owner, target["weight"]).detach().clone()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite merged LoRA weight: {target['module']}")
        merged[f"{target['module']}.{target['weight']}"] = value
    if any(".parametrizations." in key for key in merged):
        raise ValueError("Unexpected parametrization remains in native LoRA export")
    # Native upstream loaders do not require module-version metadata, but retain
    # non-parametrization entries for compatible PyTorch load_state_dict users.
    if hasattr(state, "_metadata"):
        merged._metadata = {k: v for k, v in state._metadata.items()
                            if ".parametrizations" not in k}
    return merged


def weight_movement(model):
    """Relative Frobenius norm of effective Q/V updates versus original Q/V weights.

    This is invariant to reciprocal rescaling of LoRA factors. Unchanged packed
    key slices do not dilute the denominator; zero original norm is an error.
    """
    numerator = denominator = 0.0
    with torch.no_grad():
        for target, _, sequence in _installed(model):
            delta = sequence[0].delta()
            original = sequence.original
            for start, stop in target["slices"]:
                numerator += float(delta[start:stop].double().square().sum())
                denominator += float(original[start:stop].double().square().sum())
    if denominator <= 0 or not math.isfinite(numerator + denominator):
        raise ValueError("LoRA movement requires finite updates and nonzero original weights")
    return math.sqrt(numerator / denominator)
