"""Model loading + saving for continued pretraining.

Wraps ``tabpfn.base.load_model_criterion_config`` so the rest of the
training pipeline can stay agnostic to TabPFN's internal API.

Key behaviour
-------------
* ``load_tabpfn_for_training`` infers the model **version** from the
  checkpoint filename (``tabpfn-v2.6-…ckpt`` → ``"v2.6"``,
  ``tabpfn-v2.5-…ckpt`` → ``"v2.5"``) and the **task** (classifier /
  regressor) from the user-supplied ``track`` argument. This avoids
  having to put yet another knob in ``train.yaml`` — the filename
  already carries the information.

* The function returns a triple ``(model, criterion, architecture_config)``
  ready for forward passes:

    - For PD: ``criterion`` is ``CrossEntropyLoss`` (classifier loss).
    - For LGD: ``criterion`` is ``FullSupportBarDistribution`` —
      TabPFN's regression criterion. Its ``.borders`` tensor lives
      on the model's device after ``model.to(device)``.

* ``save_finetuned`` writes ``{state_dict, config}`` in the same
  format as the base checkpoints, so the saved file can be loaded
  back via ``TabPFNClassifier(model_path=...)`` /
  ``TabPFNRegressor(model_path=...)`` for downstream evaluation —
  exactly as Real-TabPFN does.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import torch

LOGGER = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"tabpfn-v(2\.5|2\.6)-")


def _infer_version(ckpt_path: Path) -> Literal["v2.5", "v2.6"]:
    m = _VERSION_RE.search(ckpt_path.name)
    if not m:
        raise ValueError(
            f"Could not infer TabPFN version from filename {ckpt_path.name!r}. "
            "Expected name to start with 'tabpfn-v2.5-' or 'tabpfn-v2.6-'."
        )
    return m.group(1)  # type: ignore[return-value]


def load_tabpfn_for_training(
    checkpoint_path: Path | str,
    *,
    track: Literal["pd", "lgd"],
    device: str = "cpu",
) -> tuple[torch.nn.Module, torch.nn.Module, object]:
    """Load a TabPFN base checkpoint, ready to be trained on.

    Parameters
    ----------
    checkpoint_path
        Local path to a ``.ckpt`` saved by Prior Labs (or by
        :func:`save_finetuned`). Must include version in name.
    track
        ``"pd"`` → load a TabPFN classifier; ``"lgd"`` → regressor.
    device
        Where to move the model after loading. The criterion (in the
        regressor case) is moved with the model.

    Returns
    -------
    model
        ``PerFeatureTransformer`` in train-ready state.
    criterion
        Loss criterion suitable for ``track``:
        - PD: ``torch.nn.CrossEntropyLoss``
        - LGD: ``FullSupportBarDistribution`` (with
          ``ignore_nan_targets=True`` set explicitly)
    architecture_config
        The ``ArchitectureConfig`` returned by ``load_model_criterion_config``;
        re-saved alongside the trained ``state_dict`` so the file
        round-trips through Prior Labs' loaders.
    """
    from tabpfn.base import load_model_criterion_config

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt}")
    version = _infer_version(ckpt)
    which: Literal["classifier", "regressor"] = (
        "classifier" if track == "pd" else "regressor"
    )

    models, criterion, architecture_configs, _inference_config = (
        load_model_criterion_config(
            model_path=ckpt,
            check_bar_distribution_criterion=False,  # we re-check below
            cache_trainset_representation=False,
            which=which,
            version=version,
            download_if_not_exists=False,
        )
    )
    model = models[0]
    architecture_config = architecture_configs[0]

    # Build the *training* criterion. The criterion returned by
    # ``load_model_criterion_config`` is fine for inference but the
    # finetuning loop wants extra knobs (e.g. ignore_nan_targets).
    if track == "pd":
        train_criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()
    else:
        from tabpfn.architectures.base.bar_distribution import (
            FullSupportBarDistribution,
        )
        if not isinstance(criterion, FullSupportBarDistribution):
            raise TypeError(
                f"Regressor checkpoint did not yield a "
                f"FullSupportBarDistribution criterion (got {type(criterion).__name__})"
            )
        train_criterion = FullSupportBarDistribution(
            borders=criterion.borders,
            ignore_nan_targets=True,
        )

    model.to(device)
    train_criterion.to(device)
    return model, train_criterion, architecture_config


def save_finetuned(
    model: torch.nn.Module,
    architecture_config,
    save_path: Path | str,
) -> Path:
    """Persist a finetuned model in Prior Labs' on-disk format.

    Format mirrors the base checkpoints (``state_dict`` + ``config``)
    so the saved file can be loaded later via
    ``TabPFNClassifier(model_path=save_path)`` or
    ``TabPFNRegressor(model_path=save_path)``.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ArchitectureConfig is a dataclass; `__dict__` serialises cleanly.
    config_payload = (
        architecture_config.__dict__
        if hasattr(architecture_config, "__dict__")
        else architecture_config
    )

    # Strip a possible DataParallel wrapper (TabPFN finetuning uses
    # DDP/DataParallel in the multi-GPU paths; ours doesn't, but be safe).
    state_dict = (
        model.module.state_dict()
        if hasattr(model, "module") else model.state_dict()
    )

    torch.save(
        {"state_dict": state_dict, "config": config_payload},
        str(save_path),
    )
    return save_path
