"""Linear baselines: logistic regression (PD), linear regression (LGD).

These are the textbook "what does plain linear modelling do on this?"
baselines. Two implementation notes worth being explicit about
(both common gotchas):

NaN handling
------------
sklearn's ``LogisticRegression`` and ``LinearRegression`` / ``Ridge``
do not accept NaN. The cached chunks may contain NaN (TabPFN handles
it natively, and the data pipeline deliberately preserves NaNs for
that reason). So the linear baselines wrap a ``SimpleImputer(strategy=
"median")`` plus a ``StandardScaler`` in a ``Pipeline``. The scaler
is mostly a numerical convenience for the LBFGS solver — without it
LogReg can fail to converge on heavily-skewed credit-risk features.

Categorical features
--------------------
The cached chunks have ordinal-encoded categoricals (so each cat
column is a single float32 column with integer-valued cells). We
pass them through unchanged — interpreting the ordinal codes as
numeric features for the linear model. This is the **canonical
"baseline" treatment** in the credit-risk literature; if a future
experiment wants one-hot encoding, that's a different baseline
(``LogReg-OHE``) and should be added as its own class, not as a
hidden flag.

Hyperparameter tuning
---------------------
The regularization strength is the single most important knob for a
linear model on collinear credit-risk features, so both baselines tune
it per fold via Optuna (``n_trials`` from ``config/eval.yaml``'s
``hpo.logreg`` / ``hpo.linreg``): LogReg tunes ``C`` and Ridge tunes
``alpha``, scored on the same inner-val split the boosting baselines
use. This keeps the classical controls credible ("TabPFN beats *tuned*
baselines", not under-tuned ones). With ``n_trials=0`` or no val split
they fall back to library defaults.
"""

from __future__ import annotations

import logging

import numpy as np

from src.model.base import replace_inf_with_nan

LOGGER = logging.getLogger(__name__)


def _make_linear_pipeline(estimator):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])


def _tune_regularization(
    *, name, param_name, low, high, build_pipeline, score_val,
    X, y, X_val, y_val, n_trials, timeout, seed,
):
    """Optuna log-uniform search over a single regularization param.

    ``score_val`` maps (fitted_pipeline) → a scalar to MINIMISE (e.g.
    ``-roc_auc`` or ``rmse``). Returns the best param value, or ``None``
    when tuning isn't possible (no optuna, no val split, degenerate val).
    """
    if n_trials <= 0 or X_val is None or y_val is None:
        return None
    try:
        import optuna
    except Exception:                                              # pragma: no cover
        LOGGER.warning("optuna not installed; skipping %s HPO", name)
        return None
    Xf, Xvf = replace_inf_with_nan(X), replace_inf_with_nan(X_val)

    def objective(trial):
        val = trial.suggest_float(param_name, low, high, log=True)
        pipe = build_pipeline(val)
        pipe.fit(Xf, y)
        return float(score_val(pipe, Xvf, y_val))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=int(n_trials), timeout=timeout,
                   show_progress_bar=False)
    return study.best_params.get(param_name)


# --------------------------------------------------------------------------- #
# Logistic regression — PD only
# --------------------------------------------------------------------------- #


class LogRegModel:
    """``LogisticRegression`` wrapped in median-imputation + StandardScaler."""

    task_type = "classification"

    def __init__(
        self,
        *,
        params: dict | None = None,
        random_state: int = 42,
        hpo_trials: int = 0,
        hpo_timeout_seconds: float | None = None,
    ) -> None:
        self.name = "logreg"
        self._params = dict(params or {})
        self._params.setdefault("random_state", random_state)
        self._params.setdefault("max_iter", 1000)
        self._random_state = random_state
        self._hpo_trials = int(hpo_trials or 0)
        self._hpo_timeout = hpo_timeout_seconds
        self.best_params: dict = {}
        self._pipeline = None

    def _build(self, C: float | None):
        from sklearn.linear_model import LogisticRegression
        p = dict(self._params)
        if C is not None:
            p["C"] = C
        return _make_linear_pipeline(LogisticRegression(**p))

    def fit(
        self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del categorical_idx                 # ordinal cats treated as numerics
        from sklearn.metrics import roc_auc_score
        best_c = None
        if X_val is not None and y_val is not None and len(np.unique(y_val)) >= 2:
            best_c = _tune_regularization(
                name="LogReg", param_name="C", low=1e-3, high=1e3,
                build_pipeline=self._build,
                score_val=lambda pipe, Xv, yv: -roc_auc_score(
                    yv, pipe.predict_proba(Xv)[:, 1]),
                X=X, y=y, X_val=X_val, y_val=y_val,
                n_trials=self._hpo_trials, timeout=self._hpo_timeout,
                seed=self._random_state,
            )
        if best_c is not None:
            self.best_params = {"C": best_c}
        self._pipeline = self._build(best_c)
        self._pipeline.fit(replace_inf_with_nan(X), y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(replace_inf_with_nan(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(replace_inf_with_nan(X))


# --------------------------------------------------------------------------- #
# Linear regression — LGD only
# --------------------------------------------------------------------------- #


class LinRegModel:
    """``Ridge`` (mild regularisation) wrapped in the same pipeline.

    We use ``Ridge(alpha=1.0)`` rather than vanilla ``LinearRegression``
    because credit-risk LGD features are often heavily collinear
    (multiple bureau-derived ratios that move together), and a plain
    OLS solve will produce wild coefficients that hurt held-out RMSE.
    Default ridge α is the smallest principled regularisation we can
    apply without taking on a tuning step.
    """

    task_type = "regression"

    def __init__(
        self,
        *,
        params: dict | None = None,
        random_state: int = 42,
        hpo_trials: int = 0,
        hpo_timeout_seconds: float | None = None,
    ) -> None:
        self.name = "linreg"
        self._params = dict(params or {})
        self._params.setdefault("random_state", random_state)
        self._params.setdefault("alpha", 1.0)
        self._random_state = random_state
        self._hpo_trials = int(hpo_trials or 0)
        self._hpo_timeout = hpo_timeout_seconds
        self.best_params: dict = {}
        self._pipeline = None

    def _build(self, alpha: float | None):
        from sklearn.linear_model import Ridge
        p = dict(self._params)
        if alpha is not None:
            p["alpha"] = alpha
        return _make_linear_pipeline(Ridge(**p))

    def fit(
        self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del categorical_idx
        from sklearn.metrics import mean_squared_error
        best_alpha = _tune_regularization(
            name="Ridge", param_name="alpha", low=1e-3, high=1e3,
            build_pipeline=self._build,
            score_val=lambda pipe, Xv, yv: float(
                np.sqrt(mean_squared_error(yv, pipe.predict(Xv)))),
            X=X, y=y, X_val=X_val, y_val=y_val,
            n_trials=self._hpo_trials, timeout=self._hpo_timeout,
            seed=self._random_state,
        )
        if best_alpha is not None:
            self.best_params = {"alpha": best_alpha}
        self._pipeline = self._build(best_alpha)
        self._pipeline.fit(replace_inf_with_nan(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(replace_inf_with_nan(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:        # pragma: no cover
        raise NotImplementedError("LinReg has no predict_proba (regression task)")
