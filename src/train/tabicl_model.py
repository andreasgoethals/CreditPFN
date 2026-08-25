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

    ``freeze_backbone=True`` delegates to :func:`src.train.freeze.freeze_backbone`, the
    SINGLE implementation shared with TabPFN. It freezes the deepest repeated-block stack,
    which here is ``icl_predictor.tf_icl.blocks`` (12 blocks, 93.3 % of parameters), leaving
    the column embedder, row interactor, label encoder and ``decoder`` head trainable.
    ``requires_grad=False`` only, never ``.eval()``. Pass ``freeze_modules`` for a different
    regime, e.g. upstream's stage-3 ``("col_embedder", "row_interactor")``.
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
        # ONE implementation, shared with TabPFN — see src/train/freeze.py for the rule, the
        # literature it maps to, and why the LayerNorm affines stay frozen. The rule is
        # structural, so neither family names modules here: it resolves to
        # `icl_predictor.tf_icl.blocks` (12 blocks) for TabICLv2 and to
        # `icl_blocks` / `blocks` / `transformer_encoder` for TabPFN v3 / v2.6 / v2.
        #
        # TabICLv2's `col_embedder.tf_col` and `row_interactor.tf_row` are also repeated-block
        # stacks but only 3 deep, so the depth floor excludes them — on purpose. They are the
        # input-embedding stages, the analogue of TabPFN's feature embedder, and stay trainable
        # in both families.
        #
        # This is NOT upstream's `freeze_icl`, which freezes ALL of `icl_predictor` including
        # its `decoder` head. That would freeze the head TabPFN keeps trainable, leaving the
        # two arms different interventions again (4.6 % vs 3.4 % trainable instead of 6.7 %).
        from src.train.freeze import freeze_backbone as _freeze_backbone
        _freeze_backbone(model, modules=freeze_modules, family="tabicl-v2")

        drop = float(model_config.get("dropout", 0.0) or 0.0)
        if drop > 0:
            LOGGER.warning(
                "TabICLv2 freeze-backbone: checkpoint has dropout=%.3g, so the FROZEN stack "
                "still applies dropout — it stays on the TRAIN forward path by design (see "
                "src/train/freeze.py). Its weights are fixed; only forward noise differs.",
                drop,
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
