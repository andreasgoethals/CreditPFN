"""Freeze the repeated-block transformer stack holding the most parameters.

The rule is shared across model families. Everything outside the selected stack
stays trainable, including TabICL's column embedder and row interactor. This is
not head-only training or an exact reproduction of a literature PEFT recipe;
see docs/RESEARCH_BRIEF.md and docs/LITERATURE.md for that distinction.

Freezing sets requires_grad=False, never eval(): TabICL's inference branch uses
no_grad and in-place updates that are incompatible with this training path.
Trainable input embeddings still require gradients through the frozen stack,
so its activation memory does not disappear. Measure memory and throughput.
Actual selected modules and trainable parameter counts are saved in provenance.
"""

from __future__ import annotations

import collections
import logging

import torch

LOGGER = logging.getLogger(__name__)

#: Below this many blocks, warn — a real backbone is 12 (TabICLv2) or 24 (TabPFN) deep, so a
#: shallower winner usually means the architecture is not what we think. It is a WARNING and not
#: a gate: selection is by parameter count, which is the property that actually defines "the
#: backbone", and a hard floor only broke small models for no benefit.
MIN_BACKBONE_BLOCKS = 8


def find_backbone_stack(
    model: torch.nn.Module, *, min_blocks: int = MIN_BACKBONE_BLOCKS,
) -> tuple[str | None, int]:
    """Return ``(dotted_module_path, n_blocks)`` for the repeated-block stack holding the most
    parameters.

    Detected from parameter names, so it needs no per-family knowledge: a stack of repeated
    blocks shows up as sibling parameters whose paths carry an integer index at the same
    position (``icl_blocks.0.…``, ``icl_blocks.1.…``).

    Selection is by PARAMETER COUNT, because that is what "the backbone" means — the bulk of the
    pretrained weights. Depth alone would be ambiguous (TabICLv2 has three block stacks) and a
    depth FLOOR would reject small models outright. On the six shipped checkpoints the two
    criteria agree anyway: the deepest stack is also the largest.

    Nested stacks are handled by preferring the more specific path when counts tie, so
    ``icl_predictor.tf_icl.blocks`` wins over a parent that merely contains it.
    """
    counts: dict[str, int] = collections.defaultdict(int)
    depth: dict[str, set[int]] = collections.defaultdict(set)
    for name, param in model.named_parameters():
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part.isdigit():
                prefix = ".".join(parts[:i])
                counts[prefix] += param.numel()
                depth[prefix].add(int(part))
    if not counts:
        return None, 0
    best = max(counts, key=lambda k: (counts[k], len(k)))
    n_blocks = max(depth[best]) + 1
    if n_blocks < min_blocks:
        LOGGER.warning(
            "backbone detection picked %r with only %d blocks (expected >= %d). Fine for a "
            "test fixture; on a real checkpoint check this against the architecture.",
            best, n_blocks, min_blocks,
        )
    return best, n_blocks


def freeze_backbone(
    model: torch.nn.Module, *, modules: tuple[str, ...] | None = None,
    min_blocks: int = MIN_BACKBONE_BLOCKS, family: str = "",
) -> dict:
    """Freeze the transformer backbone; leave embedders, label encoder and head trainable.

    Parameters
    ----------
    modules
        Explicit dotted module paths to freeze, bypassing detection. Use this to reproduce a
        specific published regime — e.g. TabICLv2 upstream's stage-3
        ``("col_embedder", "row_interactor")``, which freezes the front end instead and is a
        different scheme, not this one.
    min_blocks
        Warning threshold for the selected stack's depth.
    family
        Only used in the log line.

    Returns a dict describing what happened, which the caller records with the trial so the
    frozen fraction is in the results rather than inferred later.
    """
    if modules:
        targets = tuple(modules)
        n_blocks = 0
        known = {n for n, _ in model.named_modules()}
        missing = [m for m in targets if m not in known]
        if missing:
            raise ValueError(
                f"freeze targets not on the model: {missing}. "
                f"Top-level children: {[n for n, _ in model.named_children()]}"
            )
    else:
        stack, n_blocks = find_backbone_stack(model, min_blocks=min_blocks)
        if stack is None:
            raise ValueError(
                f"no repeated-block stack found in {family or type(model).__name__} — the "
                f"model has no indexed submodules at all, so there is nothing identifiable as "
                f"a backbone; pass `modules=` explicitly. Top-level children: "
                f"{[n for n, _ in model.named_children()]}"
            )
        targets = (stack,)

    prefixes = tuple(t + "." for t in targets)
    frozen = trainable = 0
    for name, param in model.named_parameters():
        if name.startswith(prefixes) or name in targets:
            param.requires_grad = False
        if not param.requires_grad:
            frozen += param.numel()
        else:
            trainable += param.numel()

    total = frozen + trainable
    info = {
        "frozen_modules": targets,
        "backbone_blocks": n_blocks,
        "frozen_params": frozen,
        "trainable_params": trainable,
        "trainable_fraction": trainable / max(1, total),
        "trainable_parameter_names": [n for n, p in model.named_parameters() if p.requires_grad],
    }
    model._creditpfn_freeze_info = info
    LOGGER.info(
        "freeze-backbone (%s): froze %s%s — %.2fM of %.2fM params (%.1f%%); "
        "%.2fM trainable (%.1f%%: embedders + label encoder + head). "
        "requires_grad=False only, no .eval(), no adapters.",
        family or type(model).__name__, ", ".join(targets),
        f" [{n_blocks} blocks]" if n_blocks else "",
        frozen / 1e6, total / 1e6, 100 * frozen / max(1, total),
        trainable / 1e6, 100 * info["trainable_fraction"],
    )
    if info["trainable_fraction"] > 0.5:
        LOGGER.warning(
            "freeze-backbone (%s) left %.1f%% of parameters trainable — that is more than "
            "half, so the detected stack is probably not the backbone. Check the module path "
            "above against the architecture.",
            family or type(model).__name__, 100 * info["trainable_fraction"],
        )
    return info
