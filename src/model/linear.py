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

We use **default hyperparameters** for the same reason as the
boosting models — see ``boosting.py``'s docstring.
"""

from __future__ import annotations

import logging

import numpy as np

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
    ) -> None:
        self.name = "logreg"
        self._params = dict(params or {})
        self._params.setdefault("random_state", random_state)
        self._params.setdefault("max_iter", 1000)
        self._pipeline = None

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int]) -> None:
        del categorical_idx       # ordinal codes treated as numerics
        from sklearn.linear_model import LogisticRegression
        self._pipeline = _make_linear_pipeline(LogisticRegression(**self._params))
        self._pipeline.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)


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
    ) -> None:
        self.name = "linreg"
        self._params = dict(params or {})
        self._params.setdefault("random_state", random_state)
        self._params.setdefault("alpha", 1.0)
        self._pipeline = None

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int]) -> None:
        del categorical_idx
        from sklearn.linear_model import Ridge
        self._pipeline = _make_linear_pipeline(Ridge(**self._params))
        self._pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:        # pragma: no cover
        raise NotImplementedError("LinReg has no predict_proba (regression task)")
