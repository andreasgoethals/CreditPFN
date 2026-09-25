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
Categorical columns arrive as ordinal codes, but those codes do not
represent ordered quantities. Fit a sparse one-hot encoder on the
context only; unseen validation/test categories receive all-zero
indicators. Numerical columns use median imputation and scaling.

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


def _make_linear_pipeline(estimator, *, categorical_idx=(), n_features=None):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler",  StandardScaler()),
    ])
    if not categorical_idx:
        return Pipeline([*numeric.steps, ("model", estimator)])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-2, keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    features = ColumnTransformer([
        ("numeric", numeric, [i for i in range(n_features) if i not in categorical_idx]),
        ("categorical", categorical, list(categorical_idx)),
    ], sparse_threshold=1.0)
    return Pipeline([("features", features), ("model", estimator)])


def _tune_regularization(
    *, name, param_name, low, high, build_pipeline, score_val,
    X, y, X_val, y_val, n_trials, timeout, seed,
):
    """Optuna log-uniform search over a single regularization param.

    ``score_val`` maps (fitted_pipeline) → a scalar to MINIMISE (e.g.
    ``-roc_auc`` or ``rmse``). Returns (best parameter, completed trials).
    Disabled tuning or absent validation returns (None, 0). Requested tuning
    requires Optuna; silently using defaults would change the comparison.
    """
    if n_trials <= 0 or X_val is None or y_val is None:
        return None, 0
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(f"Optuna is required for requested {name} tuning") from exc
    Xf, Xvf = replace_inf_with_nan(X), replace_inf_with_nan(X_val)

    def objective(trial):
        val = trial.suggest_float(param_name, low, high, log=True)
        pipe = build_pipeline(val)
        pipe.fit(Xf, y)
        return float(score_val(pipe, Xvf, y_val))

    sampler = optuna.samplers.TPESampler(seed=seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), timeout=timeout,
                   show_progress_bar=False)
    return study.best_params.get(param_name), sum(t.state.name == "COMPLETE" for t in study.trials)


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
        return _make_linear_pipeline(LogisticRegression(**p),
            categorical_idx=self._categorical_idx, n_features=self._n_features)

    def fit(
        self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        self._categorical_idx = list(categorical_idx or [])
        self._n_features = X.shape[1]
        self.best_params = {}
        self.hpo_trials_completed = 0
        from sklearn.metrics import roc_auc_score
        best_c = None
        if X_val is not None and y_val is not None and len(np.unique(y_val)) >= 2:
            best_c, self.hpo_trials_completed = _tune_regularization(
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
    """Ridge with fold-local preprocessing and validation-tuned regularization.

    When tuning is disabled, alpha=1.0 is the sklearn default.
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
        return _make_linear_pipeline(Ridge(**p),
            categorical_idx=self._categorical_idx, n_features=self._n_features)

    def fit(
        self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        self._categorical_idx = list(categorical_idx or [])
        self._n_features = X.shape[1]
        self.best_params = {}
        self.hpo_trials_completed = 0
        from sklearn.metrics import mean_squared_error
        best_alpha, self.hpo_trials_completed = _tune_regularization(
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
