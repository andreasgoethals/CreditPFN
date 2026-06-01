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
from pathlib import Path
from typing import Iterable, Literal

from src.model.base import ModelHandle
from src.model.boosting import CatBoostModel, XGBoostModel
from src.model.linear import LinRegModel, LogRegModel
from src.model.tabpfn_models import TabPFNUntuned
from src.utils.paths import resolve_output_path

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

    # Each baseline is constructed inside a try/except so the eval roster
    # is robust: a missing optional package (e.g. catboost not installed)
    # or any constructor failure SKIPS that one baseline with a warning
    # rather than crashing the whole benchmark. The eval always runs with
    # whatever subset of {xgboost, catboost, logreg/linreg, tabpfn-untuned}
    # can actually be built. (Added 2026-05-29.)
    def _try_add(name: str, factory) -> None:
        try:
            m = factory()
        except Exception as exc:                                # noqa: BLE001
            LOGGER.warning(
                "baseline %r could not be constructed (%s: %s) — skipping it "
                "from the eval roster.", name, type(exc).__name__, exc,
            )
            return
        out.append((ModelHandle(
            name=m.name, track=track, task_type=task_type, source="baseline",
        ), m))

    if "xgboost" in enabled:
        _try_add("xgboost", lambda: XGBoostModel(
            task_type=task_type, random_state=seed,
            hpo_trials=int(hpo_xgb.get("n_trials", 0)),
            hpo_timeout_seconds=hpo_xgb.get("timeout_seconds"),
            hpo_max_rows=hpo_xgb.get("max_rows"),
        ))

    if "catboost" in enabled:
        _try_add("catboost", lambda: CatBoostModel(
            task_type=task_type, random_state=seed,
            hpo_trials=int(hpo_cb.get("n_trials", 0)),
            hpo_timeout_seconds=hpo_cb.get("timeout_seconds"),
            hpo_max_rows=hpo_cb.get("max_rows"),
        ))

    if "logreg" in enabled and track == "pd":
        _try_add("logreg", lambda: LogRegModel(random_state=seed))

    if "linreg" in enabled and track == "lgd":
        _try_add("linreg", lambda: LinRegModel(random_state=seed))

    if "tabpfn-untuned" in enabled:
        for base_path in base_paths_for_tabpfn_untuned or ():
            # Skip base checkpoints that aren't on disk — otherwise the
            # untuned model would FAIL every CV fold (polluting the
            # results with FAIL rows). The checkpoint path is resolved
            # against the output root so it works on the cluster too.
            resolved = resolve_output_path(str(base_path))
            if not Path(resolved).exists() and not Path(str(base_path)).exists():
                LOGGER.warning(
                    "tabpfn-untuned base checkpoint not on disk: %s — skipping "
                    "it from the eval roster.", base_path,
                )
                continue
            m = TabPFNUntuned(
                task_type=task_type, base_path=base_path,
                device=device, n_estimators=n_estimators_tabpfn,
            )
            out.append((ModelHandle(
                name=m.name, track=track, task_type=task_type,
                source="tabpfn-untuned", base_path=str(base_path),
            ), m))

    if not out:
        LOGGER.warning(
            "build_baselines produced an EMPTY roster for track=%s "
            "(enabled=%s). Check that at least one baseline package is "
            "installed and at least one base checkpoint is on disk.",
            track, sorted(enabled),
        )
    return out
