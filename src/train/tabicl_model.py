"""TabICLv2 v2 load / save / loss for continued pretraining.

Mirrors ``src/train/model.py`` (the TabPFN loader/saver) for the second model
family. Everything here follows tabicl's OWN finetuning code
(``tabicl._finetune/{base,classifier,regressor}.py``, dump:
``tfm-library/repositories/TabICL.txt``) so that the training-time behaviour
matches what the upstream wrappers would do:

* **Load**: ``ckpt = torch.load(path, weights_only=True)`` →
  ``TabICL(**ckpt["config"])`` + ``load_state_dict`` — exactly the sklearn
  wrapper's ``_load_model``. We additionally force ``recompute=True``
  (gradient checkpointing in all three stages, a config key the checkpoints
  carry) because the ICL stage's attention is O(rows²) and checkpointing is
  the main training-time VRAM lever.
* **Adaptation modes**: full-FT, or ``freeze_backbone=True`` = freeze the
  column embedder + row interactor and train only the ICL module — TabICLv2's
  own pretraining stage-3 regime and the literature-sanctioned adaptation
  (full SFT collapsed TabICLv2 in two independent reports; see tabicl_compat).
* **Loss**: classification = CE over the first ``n_classes`` of the 10 logit
  columns (their ``_compute_batch_loss``); regression = mean pinball loss
  over 999 quantile levels ``linspace(0,1,Q+2)[1:-1]`` on z-normalized
  targets (their ``_pinball_loss`` — identical to the pretraining objective).
* **Save**: their pretraining/finetune checkpoint schema
  ``{"config", "state_dict"}`` (+ our provenance), loadable by
  ``TabICLClassifier(model_path=...)`` / ``TabICLRegressor(model_path=...)``
  — the exact analogue of TabPFN's ``model_path`` round-trip. Contents stay
  tensors/primitives because upstream loads with ``weights_only=True``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import torch

from src.train.tabicl_compat import import_tabicl_core

LOGGER = logging.getLogger(__name__)


def relax_attention_backend() -> str:
    """Steer PyTorch away from cuDNN's fused attention kernel.

    MEASURED (probe j11521064 and j11523173): TabICLv2 trains at 26 000 rows using 27 GB of a
    183 GB card, and at 40 000 rows it does not run out of memory — it raises

        Expected mha_graph.execute(...).is_good() to be true, but got false

    which is cuDNN's fused multi-head-attention graph refusing the shape. So the 26 000 ceiling
    is a KERNEL limit, not a capacity one, on the family whose entire design point is large
    context. Disabling that one backend leaves the math and mem-efficient paths, which have no
    such shape restriction; they are slower per step but allow a far larger context.

    Returns a short description of what is enabled, for the trial log.
    """
    try:
        import torch
        if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            return "unchanged (torch too old to select an SDPA backend)"
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return "cudnn_sdp=off, flash/mem_efficient/math=on"
    except Exception as exc:                                   # pragma: no cover
        return f"unchanged ({type(exc).__name__})"


#: Which TabICLv2 modules ``freeze_backbone=True`` freezes, and WHY THIS ONE.
#: `icl_predictor` is the 12-block in-context transformer holding 26.28M of TabICLv2's 27.6M
#: parameters (95.4 %). It is the direct analogue of TabPFN's `icl_blocks` (24 blocks, 96.6 % of
#: 53.2M), so freezing it makes `frozen_backbone` the same intervention in both families:
#: freeze the deep pretrained representation, adapt the embedding/head interface around it.
#:
#: Upstream's stage-3 regime is the OPPOSITE - it freezes `col_embedder` + `row_interactor`
#: (1.28M, 4.6 %) and trains `icl_predictor`. That is a valid scheme, just not the same one, and
#: reporting the two under one "frozen" label would confound the axis. Pass
#: ``freeze_modules=("col_embedder", "row_interactor")`` to get it.
_TABICL_BACKBONE_MODULES: tuple[str, ...] = ("icl_predictor",)


def load_tabicl_for_training(
    checkpoint_path: str | Path,
    *,
    track: Literal["pd", "lgd"],
    device: str = "cuda",
    freeze_backbone: bool = False,
    freeze_modules: tuple[str, ...] | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load a TabICLv2 v2 checkpoint as a bare trainable ``TabICLv2`` module.

    Returns ``(model, model_config)`` — the config dict is needed again at
    save time (the checkpoint schema stores it). No criterion object exists
    for this family (CE / pinball are functional losses; see
    :func:`tabicl_pinball_loss`).

    ``freeze_backbone=True`` freezes ``_TABICL_BACKBONE_MODULES`` — by default
    ``icl_predictor``, the 12-block in-context transformer that is 95.4 % of the
    parameters and the analogue of TabPFN's ``icl_blocks``. This keeps the frozen arm
    the SAME intervention in both families. ``requires_grad=False`` only, never
    ``.eval()`` (see the comment below for why that matters here). Pass
    ``freeze_modules`` to select a different set, e.g. upstream's stage-3 regime
    ``("col_embedder", "row_interactor")``.
    """
    TabICL = import_tabicl_core()                               # noqa: N806

    # Do this BEFORE the model is built: the backend choice is global and the fused
    # kernel is what caps TabICLv2's context at 26k rows (see relax_attention_backend).
    LOGGER.info("TabICLv2 attention backends: %s", relax_attention_backend())
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"TabICLv2 base checkpoint not found: {ckpt_path}. Download it once "
            f"from https://huggingface.co/jingang/TabICL into the staging "
            f"checkpoints/ dir (see docs/METHOD.md)."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(
            f"{ckpt_path} lacks the TabICLv2 checkpoint schema "
            f"(keys: {sorted(ckpt.keys())}); expected 'config' + 'state_dict'."
        )
    model_config = dict(ckpt["config"])
    # Gradient checkpointing in all three stages — the training VRAM lever
    # (the ICL stage is quadratic in rows). Checkpoint-config key, state dict
    # unaffected.
    model_config["recompute"] = True
    model = TabICL(**model_config)
    model.load_state_dict(ckpt["state_dict"])

    # Sanity: the regressor config carries num_quantiles (999) and
    # max_classes=0; the classifier max_classes=10. Guard against loading the
    # wrong head for the track — a silent mixup would train garbage.
    is_regressor = int(model_config.get("max_classes", 10)) == 0
    if track == "lgd" and not is_regressor:
        raise ValueError(
            f"{ckpt_path.name} is a TabICLv2 CLASSIFIER checkpoint but "
            f"track='lgd' needs the regressor (max_classes==0)."
        )
    if track == "pd" and is_regressor:
        raise ValueError(
            f"{ckpt_path.name} is a TabICLv2 REGRESSOR checkpoint but "
            f"track='pd' needs the classifier."
        )

    if freeze_backbone:
        n_frozen = 0
        targets = tuple(freeze_modules or _TABICL_BACKBONE_MODULES)
        missing = [m for m in targets if not hasattr(model, m)]
        if missing:
            raise ValueError(
                f"TabICLv2 freeze targets not on the model: {missing}. "
                f"Available: {[n for n, _ in model.named_children()]}"
            )
        for module_name in targets:
            module = getattr(model, module_name)
            # DO NOT call module.eval() here. In TabICLv2, `.training` selects the
            # ALGORITHM, not just dropout/BN: both `ColEmbedder.forward` and
            # `RowInteractor.forward` branch `if self.training: _train_forward
            # else: _inference_forward`. The inference branch runs through
            # InferenceManager (chunked, KV-cached, wrapped in torch.no_grad)
            # and, at interaction.py `_inference_forward`, writes the CLS tokens
            # into the incoming embeddings IN PLACE:
            #     embeddings[:, :, : self.num_cls] = cls_tokens
            # With the col_embedder also in eval mode that tensor is a view
            # produced under no_grad, so the write raises
            #     "A view was created in no_grad mode and is being modified
            #      inplace with grad mode enabled"
            # a few steps into training. That killed ALL 16 `_iclhead` trials
            # (both tracks) in the 2026-08-05 run while every full-FT trial
            # passed — the tell that the freeze, not TabICLv2, was at fault.
            #
            # Freezing is `requires_grad=False` alone. The module then runs the
            # SAME `_train_forward` as in full-FT, which makes freeze-backbone a
            # clean ablation of it (identical computation, gradients differ).
            # Safe for regularisation too: TabICLv2's `dropout` defaults to 0.0 and
            # the architecture uses LayerNorm (no running statistics), so train
            # vs eval mode is behaviourally identical for the frozen stages —
            # we warn below if a checkpoint ever ships dropout > 0.
            for p in module.parameters():
                p.requires_grad = False
                n_frozen += 1
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        drop = float(model_config.get("dropout", 0.0) or 0.0)
        if drop > 0:
            LOGGER.warning(
                "TabICLv2 freeze-backbone: checkpoint has dropout=%.3g, so the "
                "FROZEN stages still apply dropout (they stay in train mode by "
                "design — see the comment in load_tabicl_for_training). Their "
                "weights are still fixed; only the forward noise differs.", drop,
            )
        n_frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOGGER.info(
            "TabICLv2 freeze-backbone: froze %s (%d tensors, %.2fM params); %.2fM params "
            "in %d tensors remain trainable (%.1f%%). requires_grad=False only, so the "
            "frozen modules stay on the TRAIN forward path and this is a clean ablation "
            "of full fine-tuning.",
            ", ".join(targets), n_frozen, n_frozen_p / 1e6, n_train_p / 1e6, n_train,
            100 * n_train_p / max(1, n_frozen_p + n_train_p),
        )

    model.to(device)
    return model, model_config


def tabicl_pinball_loss(quantiles: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean pinball (quantile) loss — verbatim tabicl's ``_pinball_loss``.

    ``quantiles``: ``(E, test, Q)`` raw quantile outputs of the regressor
    forward; ``y``: ``(E, test)`` z-normalized targets. Levels are the fixed
    ``linspace(0, 1, Q+2)[1:-1]`` grid the head was pretrained on.
    """
    n_q = quantiles.shape[-1]
    alpha = torch.linspace(0, 1, n_q + 2, device=quantiles.device,
                           dtype=quantiles.dtype)[1:-1]
    diff = y.unsqueeze(-1) - quantiles                     # (E, test, Q)
    return torch.max(alpha * diff, (alpha - 1.0) * diff).mean()


def save_finetuned_tabicl(
    model: torch.nn.Module,
    model_config: dict,
    save_path: str | Path,
    *,
    provenance: dict | None = None,
) -> Path:
    """Persist a finetuned TabICLv2 model in upstream's checkpoint schema.

    Written keys: ``config`` (TabICLv2 init kwargs — ``recompute`` reset to
    False so inference doesn't pay the checkpointing overhead) and
    ``state_dict`` (CPU tensors), plus our ``provenance``. Loadable directly
    via ``TabICLClassifier(model_path=...)`` / ``TabICLRegressor(...)``
    (their loader reads exactly these two keys with ``weights_only=True``,
    ignoring extras). A ``<save_path>.provenance.json`` sidecar mirrors
    the TabPFN convention so the eval roster can read test-set pins without
    ``torch.load``.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config_out = dict(model_config)
    config_out["recompute"] = False
    payload = {
        "config": config_out,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if provenance is not None:
        payload["provenance"] = provenance
    torch.save(payload, save_path)

    if provenance is not None:
        sidecar = save_path.with_suffix(save_path.suffix + ".provenance.json")
        sidecar.write_text(json.dumps(provenance, indent=2, default=str),
                           encoding="utf-8")
    LOGGER.info("Saved finetuned TabICLv2 checkpoint: %s", save_path)
    return save_path
