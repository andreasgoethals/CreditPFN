"""Common protocol every baseline + TabPFN variant implements.

Why a protocol rather than an abstract base class
-------------------------------------------------
Several of the wrapped libraries (xgboost, catboost, sklearn,
tabpfn) expose their own classes that we don't want to subclass —
we just need a duck-typed interface the eval loop can call into.
A ``Protocol`` lets us specify the contract without forcing
inheritance, while still giving static type-checkers enough to
verify each wrapper.

The contract
------------
For each model the eval pipeline does:

    model.fit(X_context, y_context, categorical_idx)
    if model.task_type == "classification":
        proba = model.predict_proba(X_query)        # (n, K)
    else:
        pred  = model.predict(X_query)              # (n,)

All inputs are plain numpy arrays — the cached chunks have already
been ordinal-encoded and float32-cast by the data pipeline, so the
model wrappers don't need to do any preprocessing other than what
the underlying library requires (e.g. NaN imputation for sklearn).

The ``ModelHandle`` dataclass is what the eval pipeline uses to
identify a row in the comparison CSV — it carries human-readable
metadata (display name, source: "baseline"/"tabpfn-untuned"/
"tabpfn-trained") plus the underlying model object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaselineModel(Protocol):
    """Duck-typed contract for any model the eval loop scores.

    The fit signature takes an OPTIONAL validation split. Models that
    do hyperparameter optimisation (XGBoost / CatBoost with Optuna)
    use this val split as the HPO objective; models without HPO
    (LogReg, LinReg, TabPFN-untuned/trained) ignore the val args
    completely. Always passing the same val split from the eval loop
    means the HPO objective and the F1-threshold tuning target are
    drawn from THE SAME 16% of the dataset — no model gets to peek
    at extra data the others didn't see.
    """

    name: str
    task_type: str            # "classification" | "regression"

    def fit(
        self,
        X: np.ndarray,                       # (n_train, n_features)
        y: np.ndarray,                       # (n_train,)
        categorical_idx: list[int],
        X_val: np.ndarray | None = None,     # (n_val, n_features); HPO objective
        y_val: np.ndarray | None = None,
    ) -> None: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n_query, n_classes) — classification only."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(n_query,) — regression only."""
        ...


@dataclass
class ModelHandle:
    """Identity card for one (model, source) pair in the eval CSV."""
    name:        str             # human-readable, unique per row
    track:       str             # "pd" | "lgd"
    task_type:   Literal["classification", "regression"]
    source:      Literal["baseline", "tabpfn-untuned", "tabpfn-trained"]
    base_path:   str | None = None   # only for tabpfn variants
    extra:       dict | None = None  # base lr / seed for trained TabPFN


# --------------------------------------------------------------------------- #
# Shared input sanitiser for non-TabPFN baselines
# --------------------------------------------------------------------------- #


def replace_inf_with_nan(X: np.ndarray) -> np.ndarray:
    """Replace +/-inf with NaN in a float feature matrix.

    TabPFN handles ±inf natively (the architecture clips them as part of
    the feature normaliser). The classical baselines don't:

      * ``SimpleImputer`` (LogReg / LinReg pipelines) treats ``inf`` as
        a finite-but-huge value and propagates it through the scaler →
        the LBFGS solver explodes.
      * XGBoost / CatBoost treat NaN as "missing" but raise / produce
        garbage on ``inf``.

    Converting ``inf → NaN`` at the wrapper edge gives all four
    baselines the same "missing value" semantics TabPFN gets for free.
    """
    if not np.issubdtype(X.dtype, np.floating):
        return X
    mask = np.isinf(X)
    if not mask.any():
        return X
    out = X.copy()
    out[mask] = np.nan
    return out
