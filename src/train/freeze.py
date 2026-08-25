"""ONE frozen-backbone implementation, shared by every model family.

WHY THIS FILE EXISTS
--------------------
`frozen_backbone` is a sweep axis in experiment 1, so the arm has to mean the same thing for
TabPFN v2 / v2.6 / v3 and for TabICLv2. Two earlier attempts did not:

  1. TabPFN got LoRA while TabICLv2 got a real freeze. LoRA is not a freeze — its adapters sit
     INSIDE the transformer, so gradients still traverse the whole network and every activation
     is still retained. That is why the LoRA arm measured as a no-op AND saved no memory in
     runs 4, 6 and 7.
  2. Both then got `requires_grad=False`, but on modules chosen per family by name. TabPFN froze
     `icl_blocks`/`blocks` (88-99 % of parameters) while TabICLv2 froze
     `col_embedder`+`row_interactor` (4.6 %) — opposite operations under one column name,
     because TabICLv2's parameters live in its LAST stage, not its first.

So the rule here is stated once, structurally, and derived from the model itself rather than from
a per-family list of module names:

    Freeze the repeated-block transformer stack holding the most parameters.
    Train everything outside it: embedders, label encoder, prediction head.

Applied to the shipped checkpoints that resolves to, with no family-specific code:

    v3       icl_blocks                    24 blocks   ->  96.6 % frozen /  3.4 % trainable
    v2.6     blocks                        24 blocks   ->  99.1 % frozen /  0.9 % trainable
    v2       transformer_encoder.*         (cluster)   ->  ~97.7 % / ~2.3 %
    tabicl   icl_predictor.tf_icl.blocks   12 blocks   ->  93.4 % frozen /  6.6 % trainable

and for the regressors 88.1 / 82.6 / 90.1 % frozen respectively — the regressor heads are much
larger, which is a real architectural fact rather than an inconsistency in this code.

Note what the rule does NOT pick: TabICLv2's `col_embedder.tf_col.blocks` (0.88M) and
`row_interactor.tf_row.blocks` (0.40M) are also repeated-block stacks, but they hold far
fewer parameters than `icl_predictor.tf_icl` (25.7M), so selection by parameter count
excludes them. That is the intent — they are the input-embedding stages, the analogue of
TabPFN's `feature_distribution_embedder`, and they stay trainable in both families.

WHAT THE LITERATURE CALLS THIS
------------------------------
Rubachev et al., "On Finetuning Tabular Foundation Models" (the only *tuned* finetuning search
for TabPFNv2) re-evaluates four partial strategies against full fine-tuning:

  * LoRA
  * "Last layers - finetuning only the upper layer ... a popular partial finetuning method"
  * "LayerNorm, Head and Embeddings - finetuning only the feature and target linear embedding
    layers, MLP prediction head and the affine layer normalization parameters"
  * learned numerical feature embeddings

What this file implements is their third strategy MINUS the LayerNorm affines, and the omission
is deliberate — see below. Their headline finding is worth knowing before we run it: "the
difference between full finetuning and all considered PEFT variations is minimal", and full
fine-tuning converged roughly twice as fast. So we should expect the frozen arm to land near the
full arm rather than beat it; the interesting outcome is whether that holds on credit data, and
whether it holds equally across four model generations.

WHY THE LAYERNORM AFFINES STAY FROZEN (a deliberate deviation)
--------------------------------------------------------------
Rubachev unfreezes the affine LayerNorm parameters inside the backbone. Doing that here would
forfeit the entire operational reason we run a frozen arm. To compute a gradient for a LayerNorm
scale in block 0, autograd has to build a graph from the loss all the way back to block 0 — so
every intermediate activation in the stack is retained, exactly as in full fine-tuning. Since
LayerNorms appear in every block, unfreezing them means the frozen arm costs what full
fine-tuning costs while updating ~1 % of the weights: the same trap LoRA fell into here.

Freezing the stack completely is what makes the arm cheap: no parameter inside it requires grad,
so no graph is built through it and its activations are never kept. That is also why the frozen
row cap can be higher than the full-FT one.

NEVER CALL .eval() HERE
-----------------------
Freezing is `requires_grad=False` and nothing else. In TabICLv2 `self.training` selects the
ALGORITHM, not just dropout: `ColEmbedder.forward` and `RowInteractor.forward` branch
`if self.training: _train_forward else: _inference_forward`, and the inference branch runs under
`no_grad` and writes CLS tokens into its input in place. Putting a frozen module in eval mode
therefore raised "A view was created in no_grad mode and is being modified inplace with grad
mode enabled" and killed all 16 frozen trials in the 05-08-2026 run. Keeping every module on the
train forward path also makes the frozen arm a clean ablation of the full arm: identical
computation, different gradients.
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
        Depth floor for automatic detection.
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
    }
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
