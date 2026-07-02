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

import contextlib
import logging
from pathlib import Path
from typing import Literal

import numpy as np

LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def _trust_local_checkpoints():
    """Force ``torch.load(weights_only=False)`` while loading our own ckpts.

    PyTorch >= 2.6 flipped ``torch.load``'s default to
    ``weights_only=True``, whose safe-unpickler rejects the config objects
    embedded in a finetuned checkpoint. TabPFN's internal loader calls
    ``torch.load(..., weights_only=None)`` (-> the new ``True`` default), so
    ``TabPFNClassifier(model_path=<our trained .ckpt>)`` raises
    ``UnpicklingError`` and **every** trained-model eval row FAILs (observed
    in the 2026-05-31 run: 875 PD + 600 LGD trained rows). The published
    base checkpoints store config as plain primitives and load fine, which
    is why only the *trained* checkpoints broke.

    We trust our own files, so within this context we intercept
    ``torch.load`` and force ``weights_only=False`` (matching what TabPFN's
    own non-base load paths already do). Scope is tight — only around model
    construction / fit — and the swap is always restored. Calls that
    explicitly pass ``weights_only=False`` are left untouched.
    """
    import torch
    orig_load = torch.load

    def _patched_load(*args, **kwargs):
        if kwargs.get("weights_only", None) is not False:
            kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        yield
    finally:
        torch.load = orig_load


def _tabpfn_regression_neg_nll(tabpfn_model, X: np.ndarray, y: np.ndarray) -> float | None:
    """Mean log-density (= −NLL) of ``y`` under TabPFN's predictive
    bar-distribution on ``X``. Higher = better (matches the ``neg_*``
    convention of the ``neg_nll`` eval column).

    This is the metric that actually rewards TabPFN's *probabilistic* output
    for LGD: point metrics (RMSE/MAE/R²) only see the mean, but the bar
    distribution predicts a full density per row, and its log-likelihood is
    what the model was trained on (bar-distribution NLL).

    Implementation is best-effort and version-tolerant: it uses
    ``TabPFNRegressor.predict(X, output_type="full")`` to recover the
    predictive distribution (the bar-distribution ``criterion`` + ``logits``,
    both in the original target space). If the installed ``tabpfn`` doesn't
    expose that surface (the API has shifted across major versions), it
    returns ``None`` so the caller records NaN — never crashing the eval.
    """
    try:
        import torch
    except Exception:                                              # pragma: no cover
        return None
    try:
        with _trust_local_checkpoints():
            out = tabpfn_model.predict(X, output_type="full")
    except Exception as exc:                                       # noqa: BLE001
        LOGGER.warning("neg_nll: predict(output_type='full') unavailable (%s: %s)",
                       type(exc).__name__, exc)
        return None
    if not isinstance(out, dict):
        LOGGER.warning("neg_nll: full output is %s, expected dict — skipping.", type(out))
        return None
    crit = out.get("criterion")
    logits = out.get("logits")
    if crit is None or logits is None:
        LOGGER.warning("neg_nll: full output lacks 'criterion'/'logits' (keys=%s) — skipping.",
                       list(out.keys()))
        return None
    try:
        logits_t = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
        y_t = torch.as_tensor(np.asarray(y).reshape(-1), dtype=logits_t.dtype,
                              device=logits_t.device)
        # FullSupportBarDistribution.forward(logits[..., num_bars], y[...])
        # returns per-row NLL (−log density). Reduce over the leading dims.
        nll = crit(logits_t, y_t)
        return float((-nll).mean().item())
    except Exception as exc:                                       # noqa: BLE001
        LOGGER.warning("neg_nll: bar-distribution NLL failed (%s: %s) — skipping.",
                       type(exc).__name__, exc)
        return None


def _make_tabpfn(task_type: str, model_path: str | Path, **extra):
    """Construct ``TabPFNClassifier`` or ``TabPFNRegressor`` from a path."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    cls = (
        TabPFNClassifier if task_type == "classification"
        else TabPFNRegressor
    )
    # Some TabPFN versions load the checkpoint at construction, others at
    # .fit(); the caller also wraps .fit() in the same context, so both
    # entry points are covered.
    with _trust_local_checkpoints():
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

    def fit(
        self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del X_val, y_val      # TabPFN has no HPO; val is unused.
        self._categorical_idx = list(categorical_idx or [])
        self._tabpfn = _make_tabpfn(
            self.task_type, self.base_path,
            device=self._device,
            n_estimators=self._n_estimators,
            categorical_features_indices=self._categorical_idx or None,
        )
        with _trust_local_checkpoints():
            self._tabpfn.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._tabpfn.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._tabpfn.predict(X)

    def neg_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float | None:
        """Mean log-density (−NLL) of ``y`` under the predictive
        bar-distribution; regression only, ``None`` on any failure."""
        if self.task_type != "regression":
            return None
        return _tabpfn_regression_neg_nll(self._tabpfn, X, y)


# --------------------------------------------------------------------------- #
# TabPFN-trained — a continued-pretrained checkpoint
# --------------------------------------------------------------------------- #


class TabPFNTrained:
    """Same machinery as ``TabPFNUntuned``, with a more descriptive
    ``name`` (the checkpoint's filename) and ``extra`` metadata
    forwarded to the eval CSV.

    ``ckpt_path`` is a path under ``checkpoints/trained/<track>/``
    written by ``src.train.model.save_finetuned`` (invoked from
    ``src.train.loop``). Because that function writes the full Prior
    Labs format (``state_dict + config + architecture_name +
    inference_config``), the file round-trips cleanly through
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

    def fit(
        self, X: np.ndarray, y: np.ndarray, categorical_idx: list[int],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del X_val, y_val
        self._categorical_idx = list(categorical_idx or [])
        self._tabpfn = _make_tabpfn(
            self.task_type, self.ckpt_path,
            device=self._device,
            n_estimators=self._n_estimators,
            categorical_features_indices=self._categorical_idx or None,
        )
        with _trust_local_checkpoints():
            self._tabpfn.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._tabpfn.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._tabpfn.predict(X)

    def neg_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float | None:
        """Mean log-density (−NLL) of ``y`` under the predictive
        bar-distribution; regression only, ``None`` on any failure."""
        if self.task_type != "regression":
            return None
        return _tabpfn_regression_neg_nll(self._tabpfn, X, y)
