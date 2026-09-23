"""Resolve phase configurations and the shared training trial grid.

Phase YAML files override ``config/train.yaml``; explicit CLI overrides win last.
Planning, training and evaluation all use these rules.
"""
from __future__ import annotations

from itertools import product

from omegaconf import ListConfig, OmegaConf

Trial = tuple[str, float, bool, float, int, str, int, float | None]


def load_train_config(overrides: list[str] | None = None,
                      config_path: str | None = None, *,
                      base_path: str = "config/train.yaml"):
    """Merge common settings, a phase delta, and explicit overrides, in that order."""
    cfg = OmegaConf.load(base_path)
    if config_path and str(config_path) != str(base_path):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(str(config_path)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def training_members(cfg, track: str, family: str) -> int:
    """Resolve the training ensemble size, including TabICL's separate setting."""
    if family == "tabicl":
        return max(1, int(getattr(cfg.train, "n_estimators_finetune_tabicl", 2)))
    value = getattr(cfg.train, "n_estimators_finetune", 2)
    if not isinstance(value, (int, float)):
        selected = getattr(value, track, None)
        if selected is None and hasattr(value, "get"):
            selected = value.get(track)
        if selected is None:
            selected = getattr(value, "default", None)
            if selected is None and hasattr(value, "get"):
                selected = value.get("default")
        value = 2 if selected is None else selected
    return max(1, int(value))


def _axis(value, convert, name: str) -> list:
    values = list(value) if isinstance(value, (list, tuple, ListConfig)) else [value]
    if not values:
        raise SystemExit(f"config error: {name} must contain at least one value")
    return [convert(item) for item in values]


def resolve_grid(cfg, *, single: bool) -> list[Trial]:
    """Return (base, lr, frozen, query fraction, accumulation, pass, min rows, L2-SP).

    The three scientific axes LR, frozen backbone and L2-SP must be explicit.
    A null L2-SP axis uses ``optimizer.l2sp_lambda``. Scalars are one-value axes.
    ``single`` selects the first trial *after* filtering eligible frozen families.
    """
    track = str(cfg.track)
    if track not in {"pd", "lgd"}:
        raise SystemExit(f"config error: unknown track {track!r}")

    missing = [key for key in ("learning_rates", "l2sp_lambdas", "frozen_backbone")
               if not hasattr(cfg.tunable, key)]
    missing += [key for key in ("learning_rates", "frozen_backbone")
                if hasattr(cfg.tunable, key) and getattr(cfg.tunable, key) is None]
    if missing:
        names = ", ".join(f"tunable.{key}" for key in missing)
        raise SystemExit(f"config error: {names} must be set by the experiment config")

    bases = _axis(getattr(cfg.tunable, "classifier_base_paths" if track == "pd"
                          else "regressor_base_paths"), str, "base paths")
    lrs = _axis(cfg.tunable.learning_rates, float, "tunable.learning_rates")
    frozen = _axis(cfg.tunable.frozen_backbone, bool, "tunable.frozen_backbone")
    qfs = _axis(getattr(cfg.tunable, "query_fractions", 0.20), float, "query_fractions")
    accs = _axis(getattr(cfg.tunable, "accumulate_grad_batches", 1), int,
                 "accumulate_grad_batches")
    passes = _axis(getattr(cfg.tunable, "epoch_pass_modes", "one_sample"), str,
                   "epoch_pass_modes")
    min_rows = _axis(getattr(getattr(cfg, "corpus", None), "min_train_rows", 0), int,
                     "corpus.min_train_rows")
    raw_l2 = cfg.tunable.l2sp_lambdas
    lambdas = [None] if raw_l2 is None else _axis(raw_l2, float, "tunable.l2sp_lambdas")
    families = getattr(cfg.tunable, "adapter_families", None)
    families = _axis(families, lambda item: str(item).lower(), "adapter_families") if families else []

    def allowed(base: str) -> bool:
        if not families:
            return True
        from src.train.tabicl_compat import model_family
        name = base.lower()
        return any(f in (model_family(base), name) or f in name for f in families)

    grid = [trial for trial in product(bases, lrs, frozen, qfs, accs, passes, min_rows, lambdas)
            if not trial[2] or allowed(trial[0])]
    if not grid:
        raise SystemExit("config error: no trials remain after filtering frozen-backbone families")
    return grid[:1] if single else grid
