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
    """Duck-typed contract for any model the eval loop scores."""

    name: str
    task_type: str            # "classification" | "regression"

    def fit(
        self,
        X: np.ndarray,        # (n_ctx, n_features)
        y: np.ndarray,        # (n_ctx,)
        categorical_idx: list[int],
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
    extra:       dict | None = None  # base lr / policy / seed for trained TabPFN
