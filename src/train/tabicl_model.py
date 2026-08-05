"""TabICL v2 load / save / loss for continued pretraining.

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
  column embedder + row interactor and train only the ICL module — TabICL's
  own pretraining stage-3 regime and the literature-sanctioned adaptation
  (full SFT collapsed TabICL in two independent reports; see tabicl_compat).
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


def load_tabicl_for_training(
    checkpoint_path: str | Path,
    *,
    track: Literal["pd", "lgd"],
    device: str = "cuda",
    freeze_backbone: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """Load a TabICL v2 checkpoint as a bare trainable ``TabICL`` module.

    Returns ``(model, model_config)`` — the config dict is needed again at
    save time (the checkpoint schema stores it). No criterion object exists
    for this family (CE / pinball are functional losses; see
    :func:`tabicl_pinball_loss`).

    ``freeze_backbone=True`` freezes ``col_embedder`` + ``row_interactor``
    (requires_grad=False AND ``.eval()`` so dropout/norm statistics stay
    inference-mode, matching tabicl's own ``freeze_col/freeze_row`` knobs)
    and leaves only ``icl_predictor`` trainable — upstream's stage-3 regime.
    """
    TabICL = import_tabicl_core()                               # noqa: N806

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"TabICL base checkpoint not found: {ckpt_path}. Download it once "
            f"from https://huggingface.co/jingang/TabICL into the staging "
            f"checkpoints/ dir (see docs/CHECKPOINTS.md)."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(
            f"{ckpt_path} lacks the TabICL checkpoint schema "
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
            f"{ckpt_path.name} is a TabICL CLASSIFIER checkpoint but "
            f"track='lgd' needs the regressor (max_classes==0)."
        )
    if track == "pd" and is_regressor:
        raise ValueError(
            f"{ckpt_path.name} is a TabICL REGRESSOR checkpoint but "
            f"track='pd' needs the classifier."
        )

    if freeze_backbone:
        n_frozen = 0
        for module_name in ("col_embedder", "row_interactor"):
            module = getattr(model, module_name)
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
                n_frozen += 1
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        LOGGER.info(
            "TabICL freeze-backbone mode: col_embedder + row_interactor frozen "
            "(%d tensors); training ICL module only (%d trainable tensors) — "
            "upstream stage-3 regime.", n_frozen, n_train,
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
    """Persist a finetuned TabICL model in upstream's checkpoint schema.

    Written keys: ``config`` (TabICL init kwargs — ``recompute`` reset to
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
    LOGGER.info("Saved finetuned TabICL checkpoint: %s", save_path)
    return save_path
