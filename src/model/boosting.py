"""XGBoost and CatBoost wrappers (with optional Optuna HPO).

Both libraries handle NaN natively (no imputation needed) and accept
an explicit categorical-features index list, which we forward
straight from the cached chunk's ``categorical_idx`` array. So the
wrappers are mostly thin: ``fit(X, y, ...) → predict_proba(X)``.

Optional per-dataset HPO
------------------------
For a *fair* comparison against TabPFN, "default XGBoost / CatBoost"
isn't quite the right control: in practice users tune them. The
wrappers therefore accept ``hpo_trials`` and ``hpo_timeout_seconds``;
when ``hpo_trials > 0``, ``.fit()`` runs an Optuna study on a
held-out validation slice of ``X_context`` and uses the best params
to refit on the full context. ``hpo_trials = 0`` disables the study
entirely and falls back to library defaults — the original "out of
the box" baseline. Both modes are exposed as eval-cfg knobs in
``config/eval.yaml``.

For categorical handling:

* **XGBoost** — passes the ordinal-encoded cats through as numerics.
  XGBoost's standard tree-splitting handles them well and matches
  what TabPFN sees too.
* **CatBoost** — uses first-class categorical support; we stringify
  the ordinal codes (NaN → "nan") so CatBoost's native cat encoding
  kicks in.

NaNs propagate through both libraries.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #


class XGBoostModel:
    """``XGBClassifier`` / ``XGBRegressor`` with optional Optuna HPO.

    HPO knobs:
      ``hpo_trials``           — 0 disables HPO (defaults).
      ``hpo_timeout_seconds``  — wall-clock cap on the study.

    Categorical columns are passed through as numerics (ordinal-encoded
    upstream), which XGBoost handles fine via standard tree splits.
    """

    def __init__(
        self,
        *,
        task_type: Literal["classification", "regression"],
        params: dict | None = None,
        random_state: int = 42,
        hpo_trials: int = 0,
        hpo_timeout_seconds: float | None = None,
    ) -> None:
        self.task_type = task_type
        self.name = "xgboost"
        self._params = dict(params or {})
        self._params.setdefault("random_state", random_state)
        # Reasonable defaults that don't depend on dataset shape.
        self._params.setdefault("n_estimators", 200)
        self._params.setdefault("tree_method", "hist")
        self._random_state = random_state
        self._hpo_trials = int(hpo_trials)
        self._hpo_timeout = hpo_timeout_seconds
        self._model = None
        self.best_params: dict | None = None

    def _make(self, params: dict):
        import xgboost as xgb
        if self.task_type == "classification":
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)

    def _maybe_hpo(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run Optuna HPO on a 80/20 split of (X, y); return best params.

        Falls back to the wrapper's default ``self._params`` if HPO is
        disabled, the dataset is too small, or Optuna is not installed.
        """
        if self._hpo_trials <= 0 or len(X) < 50:
            return dict(self._params)
        try:
            import optuna
        except ImportError:
            LOGGER.warning("optuna not installed; skipping XGBoost HPO")
            return dict(self._params)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, mean_squared_error

        stratify = y if self.task_type == "classification" else None
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.2,
                random_state=self._random_state, stratify=stratify,
            )
        except ValueError:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.2, random_state=self._random_state,
            )

        def objective(trial: "optuna.Trial") -> float:
            params = dict(self._params)
            params.update({
                "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
                "max_depth":         trial.suggest_int("max_depth", 3, 10),
                "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            })
            model = self._make(params)
            model.fit(X_tr, y_tr)
            if self.task_type == "classification":
                proba = model.predict_proba(X_va)
                if len(np.unique(y_va)) < 2:
                    return 0.5
                if proba.shape[1] == 2:
                    return -roc_auc_score(y_va, proba[:, 1])    # minimise
                return -roc_auc_score(y_va, proba, multi_class="ovr", average="macro")
            preds = model.predict(X_va)
            return float(np.sqrt(mean_squared_error(y_va, preds)))

        sampler = optuna.samplers.TPESampler(seed=self._random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective, n_trials=self._hpo_trials,
            timeout=self._hpo_timeout, show_progress_bar=False,
        )
        merged = dict(self._params)
        merged.update(study.best_params)
        self.best_params = study.best_params
        return merged

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int]) -> None:
        del categorical_idx
        params = self._maybe_hpo(X, y)
        self._model = self._make(params)
        self._model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# --------------------------------------------------------------------------- #
# CatBoost
# --------------------------------------------------------------------------- #


class CatBoostModel:
    """``CatBoostClassifier`` / ``CatBoostRegressor`` with optional Optuna HPO.

    We forward ``categorical_idx`` to CatBoost's ``cat_features`` so
    the library uses its own categorical-handling routines (target
    encoding, ordered boosting). The cached chunks are ordinal-
    encoded; we stringify them so CatBoost's native cat encoding
    kicks in (NaN → "nan").
    """

    def __init__(
        self,
        *,
        task_type: Literal["classification", "regression"],
        params: dict | None = None,
        random_state: int = 42,
        hpo_trials: int = 0,
        hpo_timeout_seconds: float | None = None,
    ) -> None:
        self.task_type = task_type
        self.name = "catboost"
        self._params = dict(params or {})
        self._params.setdefault("random_state", random_state)
        self._params.setdefault("iterations", 500)
        self._params.setdefault("verbose", False)
        self._params.setdefault("allow_writing_files", False)
        self._random_state = random_state
        self._hpo_trials = int(hpo_trials)
        self._hpo_timeout = hpo_timeout_seconds
        self._model = None
        self._cat_features: list[int] = []
        self.best_params: dict | None = None

    def _to_catboost_pool(self, X: np.ndarray, y: np.ndarray | None = None):
        """Build a CatBoost ``Pool``.

        CatBoost wants cat columns as **strings** (its native cat
        encoding); the cached chunks store them as float32 ordinal
        codes. We stringify on the fly and pass the result through a
        pandas DataFrame so column-wise dtype is honoured.

        Floats in cat columns that are actually NaN become the string
        ``"nan"`` — CatBoost treats this as its own category, which
        is the right behaviour for a missing categorical value.

        ``y`` must be passed at fit time (CatBoost requires the label
        to live inside the Pool); leave it None for predict.
        """
        from catboost import Pool
        if not self._cat_features:
            return Pool(X, label=y)
        import pandas as pd
        df = pd.DataFrame(X)
        for ci in self._cat_features:
            # Stringify integer-valued floats so cat_features sees
            # string categories; preserves NaN as the string "nan".
            df[ci] = df[ci].apply(
                lambda v: "nan" if pd.isna(v) else str(int(v))
            )
        return Pool(df, label=y, cat_features=self._cat_features)

    def _make(self, params: dict):
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = (
            CatBoostClassifier if self.task_type == "classification"
            else CatBoostRegressor
        )
        return cls(**params)

    def _maybe_hpo(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run Optuna HPO; return best params merged onto defaults."""
        if self._hpo_trials <= 0 or len(X) < 50:
            return dict(self._params)
        try:
            import optuna
        except ImportError:
            LOGGER.warning("optuna not installed; skipping CatBoost HPO")
            return dict(self._params)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, mean_squared_error

        stratify = y if self.task_type == "classification" else None
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.2,
                random_state=self._random_state, stratify=stratify,
            )
        except ValueError:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.2, random_state=self._random_state,
            )

        def objective(trial: "optuna.Trial") -> float:
            params = dict(self._params)
            params.update({
                "iterations":     trial.suggest_int("iterations", 200, 1000),
                "depth":          trial.suggest_int("depth", 4, 10),
                "learning_rate":  trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            })
            model = self._make(params)
            pool_tr = self._to_catboost_pool(X_tr, y=y_tr)
            model.fit(pool_tr)
            pool_va = self._to_catboost_pool(X_va)
            if self.task_type == "classification":
                proba = model.predict_proba(pool_va)
                if len(np.unique(y_va)) < 2:
                    return 0.5
                if proba.shape[1] == 2:
                    return -roc_auc_score(y_va, proba[:, 1])
                return -roc_auc_score(y_va, proba, multi_class="ovr", average="macro")
            preds = np.asarray(model.predict(pool_va)).reshape(-1)
            return float(np.sqrt(mean_squared_error(y_va, preds)))

        sampler = optuna.samplers.TPESampler(seed=self._random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective, n_trials=self._hpo_trials,
            timeout=self._hpo_timeout, show_progress_bar=False,
        )
        merged = dict(self._params)
        merged.update(study.best_params)
        self.best_params = study.best_params
        return merged

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int]) -> None:
        self._cat_features = list(categorical_idx or [])
        params = self._maybe_hpo(X, y)
        self._model = self._make(params)
        pool = self._to_catboost_pool(X, y=y)
        self._model.fit(pool)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pool = self._to_catboost_pool(X)
        return self._model.predict_proba(pool)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pool = self._to_catboost_pool(X)
        return np.asarray(self._model.predict(pool)).reshape(-1)
