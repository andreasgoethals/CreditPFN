"""Explicit Q/V projection maps for released TabPFN v2 and v2.6 architectures.

Both the older packed ``MultiHeadAttention`` and newer separate-projection
implementations occur in supported upstream releases. Slices refer to the
matrix obtained by flattening all output axes, never the input-feature axis.
Unknown attention layouts fail before any model is mutated.
"""
from __future__ import annotations

import math
from torch import nn
from torch.nn.utils import parametrize


def supported(model):
    return type(model).__name__ in {"PerFeatureTransformer", "TabPFNV2", "TabPFNV2p6"}


def targets(model, *, query_only=False):
    if not supported(model):
        raise ValueError("Unsupported legacy TabPFN architecture")
    stack = "blocks" if hasattr(model, "blocks") else "transformer_encoder.layers"
    try:
        blocks = model.get_submodule(stack)
    except AttributeError as exc:
        raise ValueError("Missing legacy transformer stack") from exc
    result, covered = [], {}

    def add(path, weight, projections, slices):
        owner = model.get_submodule(path)
        value = getattr(owner, weight, None)
        if (not isinstance(value, nn.Parameter) or value.ndim not in (2, 3, 4)
                or not value.is_floating_point() or parametrize.is_parametrized(owner, weight)):
            raise ValueError(f"Unsupported or already adapted legacy weight: {path}.{weight}")
        rows = math.prod(value.shape[:-1])
        if any(not 0 <= a < b <= rows for a, b in slices):
            raise ValueError("Invalid legacy projection slices")
        result.append(dict(module=path, weight=weight, shape=list(value.shape),
                           projections=projections, slices=slices))

    for path, module in model.named_modules():
        if not path.startswith(stack + "."):
            if any(hasattr(module, key) for key in ("q_projection", "_w_qkv")):
                raise ValueError(f"Legacy attention outside the validated transformer stack: {path}")
            continue
        name = type(module).__name__
        if name in {"AlongRowAttention", "AlongColumnAttention"}:
            if any(hasattr(module, attr) for attr in ("q_norm", "norm_q", "softmax_scaling")):
                raise ValueError("Query weight scaling is not validated through a nonlinear query transform")
            if any(not isinstance(getattr(module, p + "_projection", None), nn.Linear)
                   for p in ("q", "k", "v", "out")):
                raise ValueError(f"Incomplete legacy projections at {path}")
            for p in ("q",) if query_only else ("q", "v"):
                linear = getattr(module, p + "_projection")
                if linear.bias is not None:
                    raise ValueError("Legacy query scaling requires bias-free projections")
                add(path + "." + p + "_projection", "weight", [p], [[0, linear.out_features]])
        elif name == "MultiHeadAttention" and hasattr(module, "_w_qkv"):
            packed = module._w_qkv
            if packed is not None:
                if packed.ndim != 4 or packed.shape[0] != 3:
                    raise ValueError("Unexpected legacy packed QKV layout")
                width = math.prod(packed.shape[1:-1])
                add(path, "_w_qkv", ["q"] if query_only else ["q", "v"],
                    [[0, width]] if query_only else [[0, width], [2 * width, 3 * width]])
            else:
                q = module._w_q
                if q is None or q.ndim != 4 or q.shape[0] != 1:
                    raise ValueError("Unexpected legacy query layout")
                add(path, "_w_q", ["q"], [[0, math.prod(q.shape[:-1])]])
                if not query_only:
                    kv = module._w_kv
                    if kv is not None:
                        if kv.ndim != 4 or kv.shape[0] != 2:
                            raise ValueError("Unexpected legacy packed KV layout")
                        width = math.prod(kv.shape[1:-1])
                        add(path, "_w_kv", ["v"], [[width, 2 * width]])
                    else:
                        v = module._w_v
                        if v is None or v.ndim != 3:
                            raise ValueError("Unexpected legacy value layout")
                        add(path, "_w_v", ["v"], [[0, math.prod(v.shape[:-1])]])
        elif any(hasattr(module, key) for key in ("q_projection", "_w_qkv")):
            raise ValueError(f"Unrecognized legacy attention: {path} ({name})")
        else:
            continue
        block = path[len(stack) + 1:].split(".", 1)[0]
        covered[block] = covered.get(block, 0) + 1
    if not result or set(covered) != {str(i) for i in range(len(blocks))} or any(n != 2 for n in covered.values()):
        raise ValueError("Legacy adapter mapping does not cover every transformer block")
    return result
