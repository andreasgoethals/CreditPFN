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
from src.utils.paths import resolve_output_path, resolve_staging_path

LOGGER = logging.getLogger(__name__)


# Default baseline list per track.
DEFAULT_BASELINES_PD = ("xgboost", "catboost", "logreg",
                        "tabpfn-untuned", "tabicl-untuned")
DEFAULT_BASELINES_LGD = ("xgboost", "catboost", "linreg",
                         "tabpfn-untuned", "tabicl-untuned")


def build_baselines(
    *,
    track: Literal["pd", "lgd"],
    base_paths_for_tabpfn_untuned: Iterable[str] = (),
    enabled: Iterable[str] | None = None,
    device: str = "auto",
    n_estimators_tabpfn: int = 4,
    n_estimators_tabicl: int | None = None,
    seed: int = 42,
    hpo_xgboost: dict | None = None,    # {"n_trials": int, "timeout_seconds": float}
    hpo_catboost: dict | None = None,
    hpo_logreg: dict | None = None,     # {"n_trials": int, "timeout_seconds": float}
    hpo_linreg: dict | None = None,
) -> list[tuple[ModelHandle, object]]:
    """Yield ``(handle, model_instance)`` pairs for every enabled baseline.

    Parameters
    ----------
    track
        ``"pd"`` (classification) or ``"lgd"`` (regression).
    base_paths_for_tabpfn_untuned
        One untuned instance is created per path — ``TabPFNUntuned`` or
        ``TabICLUntuned`` depending on the path's family. Typically these
        are ``cfg.tunable.<classifier|regressor>_base_paths``, because
        those are the same checkpoints the training pipeline starts from
        — comparing them at "untuned" vs. "trained" shows whether
        continued pretraining helped.
    enabled
        Subset of ``{"xgboost", "catboost", "logreg", "linreg",
        "tabpfn-untuned", "tabicl-untuned"}`` to include. ``None`` → use
        the per-track default.
    device, n_estimators_tabpfn
        Forwarded to the TabPFN-untuned constructors.
    n_estimators_tabicl
        Inference-ensemble size for TabICLv2-untuned entries. TabICLv2's ICL
        attention is quadratic in rows, so its useful ensemble size is
        much smaller than TabPFN's (upstream default 8 vs our 32);
        ``None`` → fall back to ``n_estimators_tabpfn``.
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
    hpo_lr   = hpo_logreg   or {}
    hpo_lin  = hpo_linreg   or {}

    # Each baseline is constructed inside a try/except so the eval roster
    # is robust: a missing optional package (e.g. catboost not installed)
    # or any constructor failure SKIPS that one baseline with a warning
    # rather than crashing the whole benchmark. The eval always runs with
    # whatever subset of {xgboost, catboost, logreg/linreg, <family>-untuned}
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
        _try_add("logreg", lambda: LogRegModel(
            random_state=seed,
            hpo_trials=int(hpo_lr.get("n_trials", 0)),
            hpo_timeout_seconds=hpo_lr.get("timeout_seconds"),
        ))

    if "linreg" in enabled and track == "lgd":
        _try_add("linreg", lambda: LinRegModel(
            random_state=seed,
            hpo_trials=int(hpo_lin.get("n_trials", 0)),
            hpo_timeout_seconds=hpo_lin.get("timeout_seconds"),
        ))

    # Untuned foundation-model controls, one per base checkpoint in the
    # training grid. The grid may mix families, and each family is enabled
    # independently ("tabpfn-untuned" / "tabicl-untuned") so a family can be
    # dropped from the eval roster without editing the training grid.
    from src.train.tabicl_compat import model_family
    untuned_families = {
        fam for fam, flag in (("tabpfn", "tabpfn-untuned"),
                              ("tabicl", "tabicl-untuned"))
        if flag in enabled
    }
    if untuned_families:
        for base_path in base_paths_for_tabpfn_untuned or ():
            family = model_family(base_path)
            if family not in untuned_families:
                continue
            # Skip base checkpoints that aren't on disk — otherwise the
            # untuned model would FAIL every CV fold (polluting the
            # results with FAIL rows). The checkpoint path is resolved
            # against the output root so it works on the cluster too.
            resolved = resolve_staging_path(str(base_path))
            if not Path(resolved).exists() and not Path(str(base_path)).exists():
                LOGGER.warning(
                    "untuned base checkpoint not on disk: %s — skipping "
                    "it from the eval roster.", base_path,
                )
                continue
            # Each base gets its OWN family's untuned control, so the
            # trained-vs-untuned comparison is always within-family.
            if family == "tabicl":
                from src.model.tabicl_models import TabICLUntuned
                m = TabICLUntuned(
                    task_type=task_type, base_path=resolved,
                    device=device,
                    n_estimators=int(
                        n_estimators_tabicl if n_estimators_tabicl is not None
                        else n_estimators_tabpfn
                    ),
                )
                source = "tabicl-untuned"
            else:
                m = TabPFNUntuned(
                    task_type=task_type, base_path=resolved,
                    device=device, n_estimators=n_estimators_tabpfn,
                )
                source = "tabpfn-untuned"
            out.append((ModelHandle(
                name=m.name, track=track, task_type=task_type,
                source=source, base_path=str(resolved),
            ), m))

    if not out:
        LOGGER.warning(
            "build_baselines produced an EMPTY roster for track=%s "
            "(enabled=%s). Check that at least one baseline package is "
            "installed and at least one base checkpoint is on disk.",
            track, sorted(enabled),
        )
    return out
