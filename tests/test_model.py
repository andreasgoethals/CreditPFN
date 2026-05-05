"""Smoke + unit tests for the baseline + TabPFN model wrappers.

Layout choice
-------------
One file per ``src/`` subpackage, like ``test_data.py`` and
``test_train.py``. ``test_model.py`` covers everything in
``src/model/``: the registry, the boosting / linear / TabPFN
wrappers, and the cache helper that lives in ``src/data/cache.py``
(no separate ``test_cache.py`` file because the coverage is small).

Tests that genuinely need a TabPFN checkpoint on disk are guarded
by ``pytest.importorskip`` + a path-exists check so the suite stays
runnable in a stripped-down CI image. Boosting + linear baselines
do not require an external checkpoint and run end-to-end on
synthetic data.

Coverage map
------------
    Block 1  src.data.cache          — is_cache_valid, find_uncached_datasets
    Block 2  src.model.base          — ModelHandle dataclass shape
    Block 3  src.model.boosting      — XGBoost + CatBoost on toy data
    Block 4  src.model.linear        — LogReg + LinReg + NaN handling
    Block 5  src.model.registry      — track-aware default sets
    Block 6  src.model.tabpfn_models — guarded smoke test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.cache import find_uncached_datasets, is_cache_valid
from src.model.base import ModelHandle
from src.model.boosting import CatBoostModel, XGBoostModel
from src.model.linear import LinRegModel, LogRegModel
from src.model.registry import build_baselines


# =============================================================================
# Helpers
# =============================================================================


def _write_synthetic_chunk(folder: Path, *, chunk_idx: int = 0,
                           task_type: str = "classification",
                           n_ctx: int = 50, n_qry: int = 30,
                           n_feat: int = 5, n_cat: int = 1,
                           rng: np.random.Generator | None = None) -> Path:
    rng = rng or np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    X_ctx = rng.standard_normal((n_ctx, n_feat)).astype(np.float32)
    X_qry = rng.standard_normal((n_qry, n_feat)).astype(np.float32)
    # Make first n_cat columns ordinal-encoded categoricals (small alphabet).
    for i in range(n_cat):
        X_ctx[:, i] = rng.integers(0, 4, size=n_ctx).astype(np.float32)
        X_qry[:, i] = rng.integers(0, 4, size=n_qry).astype(np.float32)
    if task_type == "classification":
        y_ctx = rng.integers(0, 2, n_ctx).astype(np.int64)
        y_qry = rng.integers(0, 2, n_qry).astype(np.int64)
    else:
        y_ctx = rng.uniform(0, 1, n_ctx).astype(np.float32)
        y_qry = rng.uniform(0, 1, n_qry).astype(np.float32)
    out = folder / f"chunk_{chunk_idx:03d}.npz"
    np.savez_compressed(
        out, X_context=X_ctx, y_context=y_ctx, X_query=X_qry, y_query=y_qry,
        categorical_idx=np.arange(n_cat, dtype=np.int32),
    )
    return out


def _write_synthetic_dataset(cached_root: Path, *, track: str, dataset_id: str,
                             n_chunks: int = 1, task_type: str | None = None,
                             rng: np.random.Generator | None = None) -> Path:
    task_type = task_type or (
        "classification" if track == "pd" else "regression"
    )
    folder = cached_root / track / dataset_id
    rng = rng or np.random.default_rng(abs(hash(dataset_id)) % (2**32))
    for ci in range(n_chunks):
        _write_synthetic_chunk(folder, chunk_idx=ci, task_type=task_type, rng=rng)
    (folder / "meta.json").write_text(
        json.dumps({"task_type": task_type, "n_chunks": n_chunks}),
        encoding="utf-8",
    )
    return folder


# =============================================================================
# Block 1 · src.data.cache
# =============================================================================


def test_is_cache_valid_true_when_meta_and_chunks_present(tmp_path: Path) -> None:
    _write_synthetic_dataset(tmp_path, track="pd", dataset_id="0001.x", n_chunks=2)
    assert is_cache_valid(tmp_path, "pd", "0001.x") is True


def test_is_cache_valid_false_when_meta_missing(tmp_path: Path) -> None:
    assert is_cache_valid(tmp_path, "pd", "0001.never_built") is False


def test_is_cache_valid_false_when_chunk_missing(tmp_path: Path) -> None:
    """Meta says n_chunks=2 but only one chunk on disk → invalid."""
    folder = tmp_path / "pd" / "0001.x"
    folder.mkdir(parents=True)
    _write_synthetic_chunk(folder, chunk_idx=0)
    (folder / "meta.json").write_text(
        json.dumps({"task_type": "classification", "n_chunks": 2}),
        encoding="utf-8",
    )
    assert is_cache_valid(tmp_path, "pd", "0001.x") is False


def test_is_cache_valid_false_when_meta_unparseable(tmp_path: Path) -> None:
    folder = tmp_path / "pd" / "0001.x"
    folder.mkdir(parents=True)
    _write_synthetic_chunk(folder, chunk_idx=0)
    (folder / "meta.json").write_text("not-json{", encoding="utf-8")
    assert is_cache_valid(tmp_path, "pd", "0001.x") is False


def test_find_uncached_returns_only_missing(tmp_path: Path) -> None:
    _write_synthetic_dataset(tmp_path, track="pd", dataset_id="0001.cached")
    _write_synthetic_dataset(tmp_path, track="lgd", dataset_id="0002.cached_lgd")
    tracks = {
        "0001.cached":      "pd",
        "0002.cached_lgd":  "lgd",
        "0003.absent":      "pd",
        "0004.also_absent": "lgd",
    }
    missing = find_uncached_datasets(
        tmp_path, dataset_ids=list(tracks), tracks=tracks,
    )
    assert missing == ["0003.absent", "0004.also_absent"]


def test_find_uncached_skips_unknown_track(tmp_path: Path, caplog) -> None:
    """A dataset_id not in the tracks dict is skipped with a warning,
    not silently treated as 'cached'."""
    out = find_uncached_datasets(tmp_path, dataset_ids=["0001.x"], tracks={})
    assert out == []          # we don't claim 0001.x is missing — we don't know


# =============================================================================
# Block 2 · src.model.base
# =============================================================================


def test_model_handle_construction() -> None:
    h = ModelHandle(
        name="xgboost", track="pd", task_type="classification",
        source="baseline",
    )
    assert h.base_path is None
    assert h.extra is None


# =============================================================================
# Block 3 · src.model.boosting
# =============================================================================


def _make_classification_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X_ctx = rng.standard_normal((100, 4)).astype(np.float32)
    y_ctx = (X_ctx[:, 0] + X_ctx[:, 1] > 0).astype(np.int64)
    X_qry = rng.standard_normal((30, 4)).astype(np.float32)
    y_qry = (X_qry[:, 0] + X_qry[:, 1] > 0).astype(np.int64)
    return X_ctx, y_ctx, X_qry, y_qry


def _make_regression_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X_ctx = rng.standard_normal((100, 4)).astype(np.float32)
    y_ctx = (X_ctx[:, 0] + 0.5 * X_ctx[:, 1]).astype(np.float32)
    X_qry = rng.standard_normal((30, 4)).astype(np.float32)
    y_qry = (X_qry[:, 0] + 0.5 * X_qry[:, 1]).astype(np.float32)
    return X_ctx, y_ctx, X_qry, y_qry


def test_xgboost_classifier_predict_proba_shape() -> None:
    pytest.importorskip("xgboost")
    X_ctx, y_ctx, X_qry, _ = _make_classification_data()
    m = XGBoostModel(task_type="classification")
    m.fit(X_ctx, y_ctx, categorical_idx=[])
    proba = m.predict_proba(X_qry)
    assert proba.shape == (30, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_xgboost_regressor_runs() -> None:
    pytest.importorskip("xgboost")
    X_ctx, y_ctx, X_qry, _ = _make_regression_data()
    m = XGBoostModel(task_type="regression")
    m.fit(X_ctx, y_ctx, categorical_idx=[])
    pred = m.predict(X_qry)
    assert pred.shape == (30,)
    assert np.all(np.isfinite(pred))


def test_catboost_classifier_runs() -> None:
    pytest.importorskip("catboost")
    X_ctx, y_ctx, X_qry, _ = _make_classification_data()
    m = CatBoostModel(task_type="classification")
    m.fit(X_ctx, y_ctx, categorical_idx=[0])         # column 0 marked cat
    proba = m.predict_proba(X_qry)
    assert proba.shape == (30, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_catboost_regressor_runs() -> None:
    pytest.importorskip("catboost")
    X_ctx, y_ctx, X_qry, _ = _make_regression_data()
    m = CatBoostModel(task_type="regression")
    m.fit(X_ctx, y_ctx, categorical_idx=[])
    pred = m.predict(X_qry)
    assert pred.shape == (30,)


# =============================================================================
# Block 4 · src.model.linear
# =============================================================================


def test_logreg_runs_with_nan_input() -> None:
    """The wrapper must impute NaN before LogisticRegression — otherwise
    sklearn raises."""
    X_ctx, y_ctx, X_qry, _ = _make_classification_data()
    X_ctx[3, 1] = np.nan          # inject a NaN that must be handled
    X_qry[5, 0] = np.nan
    m = LogRegModel()
    m.fit(X_ctx, y_ctx, categorical_idx=[])
    proba = m.predict_proba(X_qry)
    assert proba.shape == (30, 2)
    assert np.all(np.isfinite(proba))


def test_linreg_runs_with_nan_input() -> None:
    X_ctx, y_ctx, X_qry, _ = _make_regression_data()
    X_ctx[3, 1] = np.nan
    X_qry[5, 0] = np.nan
    m = LinRegModel()
    m.fit(X_ctx, y_ctx, categorical_idx=[])
    pred = m.predict(X_qry)
    assert pred.shape == (30,)
    assert np.all(np.isfinite(pred))


def test_linreg_predict_proba_raises() -> None:
    """LinReg has no predict_proba — should raise rather than silently
    returning regression predictions in proba shape."""
    m = LinRegModel()
    with pytest.raises(NotImplementedError):
        m.predict_proba(np.zeros((1, 1)))


# =============================================================================
# Block 5 · src.model.registry
# =============================================================================


def test_build_baselines_pd_default_set() -> None:
    """Default PD set: xgboost + catboost + logreg + tabpfn-untuned (one
    per base path)."""
    bases = ["checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
             "checkpoints/tabpfn-v2.5-classifier-v2.5_default.ckpt"]
    out = build_baselines(track="pd", base_paths_for_tabpfn_untuned=bases)
    handles = [h for h, _ in out]
    sources = [h.source for h in handles]
    names = [h.name for h in handles]
    # 3 classical + 2 tabpfn-untuned
    assert len(out) == 5
    assert sources.count("baseline") == 3
    assert sources.count("tabpfn-untuned") == 2
    assert "logreg" in names
    assert "xgboost" in names
    assert "catboost" in names


def test_build_baselines_lgd_default_set() -> None:
    """LGD: linreg replaces logreg."""
    out = build_baselines(track="lgd", base_paths_for_tabpfn_untuned=[])
    names = [h.name for h, _ in out]
    assert "linreg" in names
    assert "logreg" not in names
    assert "xgboost" in names
    assert "catboost" in names


def test_build_baselines_track_specific_filter() -> None:
    """logreg requested for lgd is silently dropped (and vice-versa)."""
    out = build_baselines(
        track="lgd", enabled=["logreg", "linreg"],
        base_paths_for_tabpfn_untuned=[],
    )
    names = [h.name for h, _ in out]
    assert names == ["linreg"]


def test_build_baselines_unknown_track_raises() -> None:
    with pytest.raises(ValueError, match="track"):
        build_baselines(track="xx")


def test_build_baselines_subset() -> None:
    """`enabled` restricts the set."""
    out = build_baselines(
        track="pd", enabled=["xgboost"], base_paths_for_tabpfn_untuned=[],
    )
    assert [h.name for h, _ in out] == ["xgboost"]


# =============================================================================
# Block 6 · src.model.tabpfn_models  (guarded smoke)
# =============================================================================


def test_tabpfn_untuned_constructible_without_loading() -> None:
    """The constructor of TabPFNUntuned must not actually load the
    weights — that happens lazily on `.fit()`. Otherwise the eval
    would fail at registry build time when the checkpoint is offline."""
    from src.model.tabpfn_models import TabPFNUntuned
    m = TabPFNUntuned(
        task_type="classification",
        base_path="/does/not/exist/anywhere.ckpt",
    )
    assert m.task_type == "classification"
    assert "tabpfn-untuned" in m.name
