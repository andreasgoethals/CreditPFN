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
    """Return the version string TabPFN's loader expects (with leading 'v').

    The regex captures bare "2.5" / "2.6"; we prepend "v" to match the
    ``version: Literal["v2", "v2.5", "v2.6", "v3"]`` contract used by
    ``load_model_criterion_config`` (see ``repositories/TabPFN .txt:11712``).
    """
    m = _VERSION_RE.search(ckpt_path.name)
    if not m:
        raise ValueError(
            f"Could not infer TabPFN version from filename {ckpt_path.name!r}. "
            "Expected name to start with 'tabpfn-v2.5-' or 'tabpfn-v2.6-'."
        )
    return f"v{m.group(1)}"  # type: ignore[return-value]


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
    *,
    criterion: torch.nn.Module | None = None,
    provenance: dict | None = None,
) -> Path:
    """Persist a finetuned model in Prior Labs' on-disk format, with
    full provenance metadata.

    Format mirrors the base checkpoints (``state_dict`` + ``config``)
    so the saved file can be loaded later via
    ``TabPFNClassifier(model_path=save_path)`` /
    ``TabPFNRegressor(model_path=save_path)``. We add a third key
    ``provenance`` containing the training-time HPs, dataset list,
    walltime, and GPU info — a permanent record of *how* this
    checkpoint was produced.

    The same provenance is also written to ``<save_path>.provenance.json``
    next to the .ckpt so it can be inspected without loading torch.

    Regressor checkpoints (LGD): pass the ``criterion`` (a
    :class:`FullSupportBarDistribution`) so its parameters get merged
    into the state-dict under the ``criterion.*`` prefix. This mirrors
    TabPFN's own ``save_tabpfn_model`` (see ``.venv/.../model_loading.py``
    ``save_tabpfn_model``) — its loader pops these keys out and calls
    ``criterion.load_state_dict(...)``. Without them, reloading a
    trained LGD checkpoint would raise ``Missing key(s) in state_dict``.
    """
    import json

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = (
        architecture_config.__dict__
        if hasattr(architecture_config, "__dict__")
        else architecture_config
    )
    state_dict = (
        model.module.state_dict()
        if hasattr(model, "module") else model.state_dict()
    )

    # Regressor: merge bar-distribution criterion params into the state_dict
    # with the `criterion.` prefix the loader expects. We probe by attribute
    # rather than isinstance() to avoid importing FullSupportBarDistribution
    # at module top-level (saves a heavy tabpfn import on the save path).
    if criterion is not None and hasattr(criterion, "state_dict"):
        crit_state = criterion.state_dict()
        if crit_state:                                            # non-empty (bar dist has buffers/params)
            for k, v in crit_state.items():
                state_dict[f"criterion.{k}"] = v

    payload: dict = {"state_dict": state_dict, "config": config_payload}
    if provenance is not None:
        payload["provenance"] = provenance
        # Sidecar JSON — always written next to the .ckpt for at-a-
        # glance inspection (no torch.load needed).
        sidecar = save_path.with_suffix(save_path.suffix + ".provenance.json")
        sidecar.write_text(
            json.dumps(provenance, indent=2, default=str), encoding="utf-8",
        )

    torch.save(payload, str(save_path))
    return save_path


def load_provenance(ckpt_path: Path | str) -> dict | None:
    """Read just the ``provenance`` block from a checkpoint without
    loading the model. Falls back to the JSON sidecar if present.
    Returns ``None`` if neither has provenance recorded.
    """
    import json

    ckpt_path = Path(ckpt_path)
    sidecar = ckpt_path.with_suffix(ckpt_path.suffix + ".provenance.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    if ckpt_path.exists():
        try:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:                                       # pragma: no cover
            return None
        return blob.get("provenance")
    return None
