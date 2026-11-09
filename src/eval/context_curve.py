"""Generic context-curve helpers shared by second-campaign evaluation tasks.

Nested context prefixes, disjoint outer-fold validation/test rows, context budgets and one
scored cell per (context, fold). ``src.eval.campaign_tasks`` owns the task orchestration.
"""
from __future__ import annotations

import io
import time

import numpy as np


def context_order(indices, y, task_type, seed):
    """Nested random prefixes, approximately proportional, with every class present.

    Classification ordering uses context-pool labels only; query/validation labels
    never determine a selected context. No resampling with replacement.
    """
    indices = np.asarray(indices, dtype=int)
    rng = np.random.default_rng(seed)
    if task_type != "classification":
        return rng.permutation(indices)
    heads, tails, priorities = [], [], []
    for label in np.unique(y[indices]):
        group = rng.permutation(indices[y[indices] == label])
        heads.append(group[0])
        tails.extend(group[1:])
        priorities.extend((np.arange(1, len(group)) + 0.5) / len(group))
    return np.r_[rng.permutation(heads), np.asarray(tails, dtype=int)[np.argsort(priorities, kind="stable")]]


def fold_indices(y, task_type, fold, cfg):
    from src.eval.benchmark import _inner_split, _make_outer_folds
    if fold not in range(5) or len(y) < 5:
        raise ValueError("Five outer folds required")
    train, test = list(_make_outer_folds(y, task_type=task_type, n_folds=5,
                                       seed=cfg["split_seed"]))[fold]
    pool, val = _inner_split(train, y[train], task_type=task_type,
                             val_fraction=cfg["inner_validation_fraction"],
                             seed=cfg["split_seed"] + fold)
    order = context_order(pool, y, task_type, cfg["context_seed"] + fold)
    if set(order) & (set(test) | set(val)) or set(test) & set(val):
        raise AssertionError("Context/validation/test overlap")
    return order, val, test


def budgets(available, cfg):
    maximum = min(available, cfg["max_context_rows"])
    sizes = {n for n in cfg["context_rows"] if n <= maximum}
    if cfg["include_full_available"]:
        sizes.add(maximum)
    return sorted(n for n in sizes if n >= 2)


def _pack(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def predict_all(estimator, matrix, task_type, batch_rows):
    method = estimator.predict_proba if task_type == "classification" else estimator.predict
    return np.concatenate([np.asarray(method(matrix[start:start + batch_rows]))
                           for start in range(0, len(matrix), batch_rows)])


def score_cell(data, context, validation, test, path, family, cfg, *, device="cuda"):
    from src.eval.dataset_loader import encode_for_model
    from src.eval.metrics import _classification_metrics, _regression_metrics
    from src.model.tabicl_models import _sanitize
    from src.model.tabpfn_models import _make_tabpfn, _trust_local_checkpoints
    from src.train.tabicl_compat import import_tabicl_sklearn
    if len(context) > cfg["max_context_rows"]:
        raise ValueError("Context exceeds the declared, capacity-tested experiment cap")
    a, b, c, cats = encode_for_model(data.X.iloc[context], data.X.iloc[validation], data.X.iloc[test],
                                    categorical_columns=data.categorical_columns)
    kwargs = dict(device=device, n_estimators=cfg["inference_views"][family], random_state=cfg["context_seed"])
    if family == "tabpfn":
        estimator = _make_tabpfn(data.task_type, path, categorical_features_indices=cats or None,
                                 **kwargs)
    else:
        cls, reg = import_tabicl_sklearn()
        estimator = (cls if data.task_type == "classification" else reg)(
            model_path=str(path), allow_auto_download=False,
            **cfg.get("tabicl_inference_options", {}), **kwargs)
        a, mask = _sanitize(a)
        b, _ = _sanitize(b, dead_cols=mask)
        c, _ = _sanitize(c, dead_cols=mask)
    started = time.monotonic()
    with _trust_local_checkpoints():
        estimator.fit(a, data.y[context])
        if family == "tabicl" and cfg.get("tabicl_inference_options") == {"use_amp": False, "use_fa3": False}:
            if tuple(bool(v) for v in estimator._resolve_amp_fa3()) != (False, False):
                raise RuntimeError("Requested TabICL FP32 policy was not applied")
        fit_seconds = time.monotonic() - started
        prediction = predict_all(estimator, c, data.task_type, cfg["query_batch_rows"])
        val_prediction = predict_all(estimator, b, data.task_type, cfg["query_batch_rows"])
    if len(prediction) != len(test) or not np.isfinite(prediction).all() or not np.isfinite(val_prediction).all():
        raise RuntimeError("Missing or nonfinite predictions; cell not recorded as successful")
    metrics = (_classification_metrics(proba_test=prediction, y_test=data.y[test],
               proba_val=val_prediction, y_val=data.y[validation], n_classes_seen=len(np.unique(data.y[context])))
               if data.task_type == "classification" else
               _regression_metrics(pred_test=prediction, y_test=data.y[test], neg_nll=None))
    metrics.update(fit_seconds=fit_seconds, elapsed_seconds=time.monotonic()-started)
    return metrics, _pack(context=context, validation=validation, test=test, target=data.y[test],
                          prediction=prediction, validation_target=data.y[validation], validation_prediction=val_prediction)
