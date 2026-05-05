"""TabPFN wrappers: untuned (the base checkpoint, no continued
pretraining) and trained (any checkpoint produced by
``scripts/train_pipeline.py``).

Both use the public sklearn-style API exposed by the ``tabpfn``
package — ``TabPFNClassifier`` / ``TabPFNRegressor`` — so we get
the model's full inference ensemble for free (preprocessing
estimator, ordinal encoder, NaN handling, output post-processing,
etc.). This is exactly the path the future "real" deployment would
take, which makes the eval numbers comparable to anything a user
might measure themselves with the same checkpoint.

Why two classes rather than one
-------------------------------
At the model level there's no difference — both are just
``TabPFN<X>(model_path=...)``. But the eval pipeline's CSV uses
``source ∈ {"tabpfn-untuned", "tabpfn-trained"}`` to identify
which family a checkpoint belongs to (so the comparison plot can
group them visually), and the two classes make that distinction
self-documenting:

  * ``TabPFNUntuned(base_path=...)`` — a path under ``checkpoints/``
    (the published Prior Labs weights). One row per (track ×
    base_path) in the eval CSV.

  * ``TabPFNTrained(ckpt_path=..., **extra)`` — a path under
    ``checkpoints/trained/`` (output of the training pipeline).
    The ``extra`` kwargs (lr, base, policy, seed) are forwarded to
    the eval CSV row so it's traceable to the originating training
    run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

LOGGER = logging.getLogger(__name__)


def _make_tabpfn(task_type: str, model_path: str | Path, **extra):
    """Construct ``TabPFNClassifier`` or ``TabPFNRegressor`` from a path."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    cls = (
        TabPFNClassifier if task_type == "classification"
        else TabPFNRegressor
    )
    return cls(model_path=str(model_path), **extra)


# --------------------------------------------------------------------------- #
# TabPFN-untuned — the base checkpoint, no continued pretraining
# --------------------------------------------------------------------------- #


class TabPFNUntuned:
    """Stock TabPFN: a base checkpoint loaded straight from
    ``checkpoints/`` — what a user gets if they pip-install tabpfn
    and pass the released weights without any fine-tuning. The
    "control" against which the continued-pretrained variants are
    measured.
    """

    def __init__(
        self,
        *,
        task_type: Literal["classification", "regression"],
        base_path: str | Path,
        device: str = "auto",
        n_estimators: int = 4,
    ) -> None:
        self.task_type = task_type
        self.base_path = str(base_path)
        self.name = f"tabpfn-untuned[{Path(base_path).stem}]"
        self._device = device
        self._n_estimators = n_estimators
        self._tabpfn = None
        self._categorical_idx: list[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int]) -> None:
        self._categorical_idx = list(categorical_idx or [])
        self._tabpfn = _make_tabpfn(
            self.task_type, self.base_path,
            device=self._device,
            n_estimators=self._n_estimators,
            categorical_features_indices=self._categorical_idx or None,
        )
        self._tabpfn.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._tabpfn.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._tabpfn.predict(X)


# --------------------------------------------------------------------------- #
# TabPFN-trained — a continued-pretrained checkpoint
# --------------------------------------------------------------------------- #


class TabPFNTrained:
    """Same machinery as ``TabPFNUntuned``, with a more descriptive
    ``name`` (the checkpoint's filename) and ``extra`` metadata
    forwarded to the eval CSV.

    ``ckpt_path`` is a path under ``checkpoints/trained/<track>/``
    written by ``src.train.loop.save_finetuned``. Because that
    function writes the same Prior Labs format
    (``state_dict + config``), the file round-trips cleanly through
    ``TabPFNClassifier(model_path=...)``.
    """

    def __init__(
        self,
        *,
        task_type: Literal["classification", "regression"],
        ckpt_path: str | Path,
        device: str = "auto",
        n_estimators: int = 4,
        extra: dict | None = None,
    ) -> None:
        self.task_type = task_type
        self.ckpt_path = str(ckpt_path)
        self.name = f"tabpfn-trained[{Path(ckpt_path).stem}]"
        self.extra = dict(extra or {})
        self._device = device
        self._n_estimators = n_estimators
        self._tabpfn = None
        self._categorical_idx: list[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int]) -> None:
        self._categorical_idx = list(categorical_idx or [])
        self._tabpfn = _make_tabpfn(
            self.task_type, self.ckpt_path,
            device=self._device,
            n_estimators=self._n_estimators,
            categorical_features_indices=self._categorical_idx or None,
        )
        self._tabpfn.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._tabpfn.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._tabpfn.predict(X)
