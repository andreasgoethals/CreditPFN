"""E5: fine-tuning on the target table, starting from the original or the CPT checkpoint.

Each cell takes one outer fold of one held-out table: the fold's training rows minus a fixed
20% validation split are both the fine-tuning data and the inference context, the validation
rows drive early stopping (and threshold/calibration choices in the metrics), and the test
rows are scored exactly as in the five-fold benchmark. The fine-tuners are upstream's own
(``tabpfn.finetuning`` and ``tabicl._finetune``), so the recipe is the libraries' reference
recipe (Rubachev et al. 2025 for TabPFN), changed only where ``recipe`` says so. CPT never sees
the target table; fine-tuning uses only the fold's own training rows.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

DEFAULT_RECIPE = {"epochs": 30, "learning_rate": 6.5e-5, "early_stopping_patience": 8}
#: TabICL inference as in the benchmark (config/eval.yaml tabicl_inference_options).
INFERENCE_OPTIONS = {"use_amp": False, "use_fa3": False}


def _tabpfn_finetuner(task_type, path, cats, *, recipe, device, views, seed, context_rows):
    import torch
    from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor
    common = dict(device=device, epochs=int(recipe["epochs"]),
                  learning_rate=float(recipe["learning_rate"]),
                  early_stopping=True, early_stopping_patience=int(recipe["early_stopping_patience"]),
                  n_finetune_ctx_plus_query_samples=int(min(10_000, context_rows)),
                  n_estimators_final_inference=int(views), random_state=int(seed),
                  save_checkpoint_interval=None)
    extra = dict(model_path=str(path), inference_precision=torch.float32,
                 categorical_features_indices=cats or None, ignore_pretraining_limits=True)
    if task_type == "classification":
        return FinetunedTabPFNClassifier(**common, extra_classifier_kwargs=extra)
    return FinetunedTabPFNRegressor(**common, extra_regressor_kwargs=extra)


def _tabicl_finetuner(task_type, path, *, recipe, device, views, seed, context_rows):
    from tabicl._finetune.classifier import FinetunedTabICLClassifier
    from tabicl._finetune.regressor import FinetunedTabICLRegressor
    classification = task_type == "classification"
    cls = FinetunedTabICLClassifier if classification else FinetunedTabICLRegressor
    # ``amp`` governs fine-tuning only; the inner inference estimator (validation and test
    # predictions) resolves AMP and FA3 itself ("auto": on above 10,240 rows). Pin it to the
    # benchmark's FP32 policy.
    inference = {"extra_classifier_kwargs" if classification else "extra_regressor_kwargs":
                 dict(INFERENCE_OPTIONS)}
    return cls(epochs=int(recipe["epochs"]), learning_rate=float(recipe["learning_rate"]),
               early_stopping=True, patience=int(recipe["early_stopping_patience"]),
               max_data_size=int(min(10_000, context_rows)), n_estimators_inference=int(views),
               model_path=str(path), allow_auto_download=False, amp=False, device=device,
               random_state=int(seed), **inference)


def _predict(estimator, X, task_type):
    if task_type == "classification":
        return np.asarray(estimator.predict_proba(X))
    out = estimator.predict(X)
    return np.asarray(out["mean"] if isinstance(out, dict) else out, dtype=float)


def finetune_cell(data, context, validation, test, path, family, *, recipe=None, device="cuda",
                  views=8, seed=0, max_context_rows=None):
    """Fine-tune on ``context`` (early stopping on ``validation``), score ``test``.

    Returns (metrics, predictions dict). Raises on non-finite or missing predictions.
    """
    from src.eval.dataset_loader import encode_for_model
    from src.eval.metrics import _classification_metrics, _regression_metrics
    recipe = {**DEFAULT_RECIPE, **(recipe or {})}
    if max_context_rows is not None and len(context) > int(max_context_rows):
        context = context[: int(max_context_rows)]
    a, b, c, cats = encode_for_model(data.X.iloc[context], data.X.iloc[validation], data.X.iloc[test],
                                     categorical_columns=data.categorical_columns)
    y_context, y_val = data.y[context], data.y[validation]
    if family == "tabpfn":
        estimator = _tabpfn_finetuner(data.task_type, path, cats, recipe=recipe, device=device,
                                      views=views, seed=seed, context_rows=len(context))
    elif family == "tabicl":
        from src.model.tabicl_models import _sanitize
        a, mask = _sanitize(a)
        b, _ = _sanitize(b, dead_cols=mask)
        c, _ = _sanitize(c, dead_cols=mask)
        # Upstream's TabICL fine-tuner rejects NaN. Impute with the column means of the
        # fine-tuning rows only (validation/test never inform it), exactly what TabICL's own
        # inference imputer does in the paired untuned cell, so both cells see identical inputs.
        a, b, c = (np.asarray(m, dtype=np.float64) for m in (a, b, c))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # an all-missing column becomes 0
            means =np.nan_to_num(np.nanmean(np.where(np.isfinite(a), a, np.nan), axis=0)
                                  if len(a) else np.zeros(a.shape[1]), nan=0.0)
        a, b, c = (np.where(np.isfinite(m), m, means) for m in (a, b, c))
        estimator = _tabicl_finetuner(data.task_type, path, recipe=recipe, device=device,
                                      views=views, seed=seed, context_rows=len(context))
    else:
        raise ValueError(f"Unknown model family {family!r}")
    started = time.monotonic()
    from src.model.tabpfn_models import _trust_local_checkpoints
    with tempfile.TemporaryDirectory(prefix="creditpfn-finetune-") as scratch, _trust_local_checkpoints():
        if family == "tabpfn":
            estimator.fit(a, y_context, X_val=b, y_val=y_val, output_dir=Path(scratch))
        else:
            estimator.fit(a, y_context, X_val=b, y_val=y_val)
        fit_seconds = time.monotonic() - started
        prediction = _predict(estimator, c, data.task_type)
        val_prediction = _predict(estimator, b, data.task_type)
    if len(prediction) != len(test) or not np.isfinite(prediction).all() or not np.isfinite(val_prediction).all():
        raise RuntimeError("Missing or non-finite fine-tuned predictions")
    metrics = (_classification_metrics(proba_test=prediction, y_test=data.y[test], proba_val=val_prediction,
                                       y_val=y_val, n_classes_seen=len(np.unique(y_context)))
               if data.task_type == "classification" else
               _regression_metrics(pred_test=prediction, y_test=data.y[test], neg_nll=None))
    metrics.update(fit_seconds=fit_seconds, context_rows=int(len(context)),
                   validation_rows=int(len(validation)), test_rows=int(len(test)),
                   recipe_epochs=int(recipe["epochs"]), recipe_learning_rate=float(recipe["learning_rate"]))
    return metrics, {"prediction": prediction, "validation_prediction": val_prediction}
