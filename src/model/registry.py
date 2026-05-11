"""Model registry: build the list of models the eval pipeline scores.

Two layers:

1. **Classical baselines** built from cfg knobs (which baselines to
   include, their per-model params if any). These are deterministic
   given the cfg — same cfg → same baselines on every machine.

2. **TabPFN-untuned** built from the same paths the training pipeline
   reads (``cfg.tunable.<track>_base_paths``), one entry per base
   checkpoint. Lets the eval cleanly compare "the base weights"
   against "the continued-pretrained weights".

3. **TabPFN-trained** is NOT built here — those come from the
   training manifest CSV (``logs/runs/<run_name>_<track>.csv``),
   one row per trained checkpoint. The eval pipeline pulls them
   in separately because the manifest is the canonical record of
   what was actually trained.
"""

from __future__ import annotations

import logging
from typing import Iterable, Literal

from src.model.base import ModelHandle
from src.model.boosting import CatBoostModel, XGBoostModel
from src.model.linear import LinRegModel, LogRegModel
from src.model.tabpfn_models import TabPFNUntuned

LOGGER = logging.getLogger(__name__)


# Default baseline list per track.
DEFAULT_BASELINES_PD = ("xgboost", "catboost", "logreg", "tabpfn-untuned")
DEFAULT_BASELINES_LGD = ("xgboost", "catboost", "linreg", "tabpfn-untuned")


def build_baselines(
    *,
    track: Literal["pd", "lgd"],
    base_paths_for_tabpfn_untuned: Iterable[str] = (),
    enabled: Iterable[str] | None = None,
    device: str = "auto",
    n_estimators_tabpfn: int = 4,
    seed: int = 42,
    hpo_xgboost: dict | None = None,    # {"n_trials": int, "timeout_seconds": float}
    hpo_catboost: dict | None = None,
) -> list[tuple[ModelHandle, object]]:
    """Yield ``(handle, model_instance)`` pairs for every enabled baseline.

    Parameters
    ----------
    track
        ``"pd"`` (classification) or ``"lgd"`` (regression).
    base_paths_for_tabpfn_untuned
        One ``TabPFNUntuned`` instance is created per path. Typically
        these are ``cfg.tunable.<classifier|regressor>_base_paths``,
        because those are the same checkpoints the training pipeline
        starts from — comparing them at "untuned" vs. "trained" shows
        whether continued pretraining helped.
    enabled
        Subset of ``{"xgboost", "catboost", "logreg", "linreg",
        "tabpfn-untuned"}`` to include. ``None`` → use the per-track
        default.
    device, n_estimators_tabpfn
        Forwarded to the TabPFN-untuned constructors.
    seed
        Forwarded to the boosting / linear constructors via
        ``random_state``.

    Returns
    -------
    list of (ModelHandle, model_instance) — the handle is the eval
    CSV row identity, the model_instance has the
    :class:`src.model.base.BaselineModel` interface.
    """
    if track == "pd":
        task_type = "classification"
        defaults = DEFAULT_BASELINES_PD
    elif track == "lgd":
        task_type = "regression"
        defaults = DEFAULT_BASELINES_LGD
    else:
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    enabled = set(enabled) if enabled is not None else set(defaults)
    out: list[tuple[ModelHandle, object]] = []

    hpo_xgb  = hpo_xgboost  or {}
    hpo_cb   = hpo_catboost or {}

    if "xgboost" in enabled:
        m = XGBoostModel(
            task_type=task_type, random_state=seed,
            hpo_trials=int(hpo_xgb.get("n_trials", 0)),
            hpo_timeout_seconds=hpo_xgb.get("timeout_seconds"),
            hpo_max_rows=hpo_xgb.get("max_rows"),
        )
        out.append((ModelHandle(
            name=m.name, track=track, task_type=task_type, source="baseline",
        ), m))

    if "catboost" in enabled:
        m = CatBoostModel(
            task_type=task_type, random_state=seed,
            hpo_trials=int(hpo_cb.get("n_trials", 0)),
            hpo_timeout_seconds=hpo_cb.get("timeout_seconds"),
            hpo_max_rows=hpo_cb.get("max_rows"),
        )
        out.append((ModelHandle(
            name=m.name, track=track, task_type=task_type, source="baseline",
        ), m))

    if "logreg" in enabled and track == "pd":
        m = LogRegModel(random_state=seed)
        out.append((ModelHandle(
            name=m.name, track=track, task_type=task_type, source="baseline",
        ), m))

    if "linreg" in enabled and track == "lgd":
        m = LinRegModel(random_state=seed)
        out.append((ModelHandle(
            name=m.name, track=track, task_type=task_type, source="baseline",
        ), m))

    if "tabpfn-untuned" in enabled:
        for base_path in base_paths_for_tabpfn_untuned or ():
            m = TabPFNUntuned(
                task_type=task_type, base_path=base_path,
                device=device, n_estimators=n_estimators_tabpfn,
            )
            out.append((ModelHandle(
                name=m.name, track=track, task_type=task_type,
                source="tabpfn-untuned", base_path=str(base_path),
            ), m))

    return out
