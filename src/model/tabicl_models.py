"""TabICL wrappers: untuned (the published v2 checkpoint) and trained
(any checkpoint produced by ``scripts/train_pipeline.py`` for the tabicl
family).

Mirrors ``tabpfn_models.py`` — same duck-typed BaselineModel interface
(``fit(X, y, categorical_idx, X_val=None, y_val=None)`` / ``predict`` /
``predict_proba`` / ``neg_log_likelihood`` / ``.name``) so the benchmark
treats both families identically. Inference goes through the official
sklearn-style API (``TabICLClassifier`` / ``TabICLRegressor``), i.e. the
package's full inference ensemble — exactly what a real deployment of the
checkpoint would use.

Family differences that matter here
-----------------------------------
* **No categorical-indices parameter.** TabICL's wrappers preprocess raw
  features themselves (ordinal-encode / normalize per ensemble variant);
  the ``categorical_idx`` argument is accepted for interface parity and
  ignored. Our eval matrices are already numeric-sanitized, so this is a
  no-op in practice.
* **``allow_auto_download=False`` always.** VSC compute nodes have no
  outbound network; a missing checkpoint must fail loudly rather than
  attempt an HF download mid-job (pre-stage from a login node instead —
  see docs/CHECKPOINTS.md).
* **``neg_log_likelihood`` returns None.** The regressor outputs 999
  quantiles, not a bar-distribution density, so TabPFN-style exact NLL
  does not exist. NEVER compare density metrics across families anyway
  (different output spaces); the planned cross-family density metric is
  CRPS, computable from ``predict(X, output_type="quantiles")`` — not
  wired up yet.
* **Loading is ``weights_only=True``-safe.** Upstream's loader reads only
  ``{config, state_dict}`` (+ ignores extras such as our provenance), so
  no ``_trust_local_checkpoints`` monkeypatch is needed for this family.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from src.model.base import replace_inf_with_nan
from src.train.tabicl_compat import import_tabicl_sklearn

LOGGER = logging.getLogger(__name__)


def _sanitize(X: np.ndarray, *, dead_cols: np.ndarray | None = None,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Make a feature matrix safe for TabICL's sklearn wrappers.

    Two upstream input constraints that TabPFN does NOT have (both
    MEASURED on tabicl 2.1.1, 2026-08-04, and both reachable from real
    credit data):

    1. ``±inf`` raises ``ValueError`` from sklearn's ``check_array``
       inside their preprocessing pipeline. TabPFN clips inf natively.
       → converted to NaN, i.e. "missing", matching what every other
       wrapper in this repo does (:func:`replace_inf_with_nan`).
    2. An **all-NaN column** raises ``IndexError`` deep in their
       imputer path (their ``SimpleImputer`` drops the column, then a
       downstream boolean feature mask no longer matches the width).
       `sanitize.py` drops >90 %-missing columns corpus-wide, but a
       single CV fold of a nearly-empty column can still be all-NaN.
       → filled with 0.0. Columns are never DROPPED, so the train and
       predict matrices keep identical widths.

    Returns ``(clean_X, dead_cols)``. Pass the training call's
    ``dead_cols`` back in for the predict matrices so the same columns
    are neutralised on both sides.
    """
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64)
    X = replace_inf_with_nan(X)
    all_nan = np.isnan(X).all(axis=0)
    if dead_cols is not None:
        all_nan = all_nan | dead_cols
    if all_nan.any():
        X = X.copy()
        X[:, all_nan] = 0.0
        LOGGER.debug("tabicl input: zero-filled %d all-NaN column(s)",
                     int(all_nan.sum()))
    return X, all_nan


def _make_tabicl(task_type: str, model_path: str | Path, *,
                 device: str = "auto", n_estimators: int = 8):
    """Construct ``TabICLClassifier`` or ``TabICLRegressor`` from a path."""
    clf_cls, reg_cls = import_tabicl_sklearn()
    cls = clf_cls if task_type == "classification" else reg_cls
    kwargs: dict = {
        "model_path": str(model_path),
        "n_estimators": int(n_estimators),
        "allow_auto_download": False,
    }
    # TabICL's device param has no "auto" sentinel (None = torch default
    # resolution, cuda-if-available) — translate our registry convention.
    if device and device != "auto":
        kwargs["device"] = device
    return cls(**kwargs)


class TabICLUntuned:
    """Stock TabICL v2: the published HF checkpoint, no continued
    pretraining — the control for the tabicl family, exactly parallel
    to ``TabPFNUntuned``."""

    def __init__(
        self,
        *,
        task_type: Literal["classification", "regression"],
        base_path: str | Path,
        device: str = "auto",
        n_estimators: int = 8,
    ) -> None:
        self.task_type = task_type
        self.base_path = str(base_path)
        self.name = f"tabicl-untuned[{Path(base_path).stem}]"
        self._device = device
        self._n_estimators = n_estimators
        self._tabicl = None
        self._dead_cols: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del X_val, y_val, categorical_idx   # no HPO; TabICL preprocesses raw features itself
        self._tabicl = _make_tabicl(
            self.task_type, self.base_path,
            device=self._device, n_estimators=self._n_estimators,
        )
        X_clean, self._dead_cols = _sanitize(X)
        self._tabicl.fit(X_clean, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean, _ = _sanitize(X, dead_cols=self._dead_cols)
        return self._tabicl.predict_proba(X_clean)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean, _ = _sanitize(X, dead_cols=self._dead_cols)
        return self._tabicl.predict(X_clean)

    def neg_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float | None:
        """No exact predictive density for the quantile head — see module
        docstring. Returns None so the eval records NaN."""
        return None


class TabICLTrained:
    """A continued-pretrained TabICL checkpoint (written by
    ``save_finetuned_tabicl``). ``extra`` metadata is forwarded to the
    eval CSV row, exactly parallel to ``TabPFNTrained``."""

    def __init__(
        self,
        *,
        task_type: Literal["classification", "regression"],
        ckpt_path: str | Path,
        device: str = "auto",
        n_estimators: int = 8,
        extra: dict | None = None,
    ) -> None:
        self.task_type = task_type
        self.ckpt_path = str(ckpt_path)
        self.name = f"tabicl-trained[{Path(ckpt_path).stem}]"
        self.extra = dict(extra or {})
        self._device = device
        self._n_estimators = n_estimators
        self._tabicl = None
        self._dead_cols: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del X_val, y_val, categorical_idx
        self._tabicl = _make_tabicl(
            self.task_type, self.ckpt_path,
            device=self._device, n_estimators=self._n_estimators,
        )
        X_clean, self._dead_cols = _sanitize(X)
        self._tabicl.fit(X_clean, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean, _ = _sanitize(X, dead_cols=self._dead_cols)
        return self._tabicl.predict_proba(X_clean)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean, _ = _sanitize(X, dead_cols=self._dead_cols)
        return self._tabicl.predict(X_clean)

    def neg_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float | None:
        """No exact predictive density for the quantile head — see module
        docstring. Returns None so the eval records NaN."""
        return None
