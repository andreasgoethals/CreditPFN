"""Smoke + unit tests for the eval benchmark.

Layout choice
-------------
One file per ``src/`` subpackage. ``test_eval.py`` covers
``src/eval/benchmark.py`` end-to-end on a synthetic cache: it
spins up two PD chunks + the boosting and linear baselines (no
TabPFN — covered in ``test_model.py``) and checks the comparison
CSV is well-formed.

Coverage map
------------
    Block 1  benchmark._score          — every metric on toy inputs
    Block 2  benchmark.run_benchmark   — end-to-end on synthetic chunks
    Block 3  benchmark.load_trained_handles — manifest reading
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.benchmark import (
    EvalRow, _score, load_trained_handles, run_benchmark,
)
from src.model.base import ModelHandle
from src.model.boosting import XGBoostModel
from src.model.linear import LogRegModel
from src.train.corpus import ChunkRef


# =============================================================================
# Helpers
# =============================================================================


def _write_chunk(folder: Path, *, chunk_idx: int = 0,
                 task_type: str = "classification",
                 n_ctx: int = 80, n_qry: int = 40,
                 n_feat: int = 4) -> Path:
    """Write one ``chunk_NNN.npz`` with separable synthetic data so
    the eval metrics aren't dominated by noise."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(chunk_idx)
    X_ctx = rng.standard_normal((n_ctx, n_feat)).astype(np.float32)
    X_qry = rng.standard_normal((n_qry, n_feat)).astype(np.float32)
    if task_type == "classification":
        y_ctx = (X_ctx[:, 0] + X_ctx[:, 1] > 0).astype(np.int64)
        y_qry = (X_qry[:, 0] + X_qry[:, 1] > 0).astype(np.int64)
    else:
        y_ctx = (X_ctx[:, 0] + X_ctx[:, 1]).astype(np.float32)
        y_qry = (X_qry[:, 0] + X_qry[:, 1]).astype(np.float32)
    out = folder / f"chunk_{chunk_idx:03d}.npz"
    np.savez_compressed(
        out, X_context=X_ctx, y_context=y_ctx, X_query=X_qry, y_query=y_qry,
        categorical_idx=np.empty(0, dtype=np.int32),
    )
    return out


def _make_chunk_refs(cached_root: Path, *, track: str, dataset_id: str,
                     n_chunks: int = 1, task_type: str | None = None
                     ) -> list[ChunkRef]:
    task_type = task_type or (
        "classification" if track == "pd" else "regression"
    )
    folder = cached_root / track / dataset_id
    refs = []
    for ci in range(n_chunks):
        p = _write_chunk(folder, chunk_idx=ci, task_type=task_type)
        refs.append(ChunkRef(
            dataset_id=dataset_id, track=track, task_type=task_type,
            chunk_path=p, chunk_idx=ci,
        ))
    (folder / "meta.json").write_text(
        json.dumps({"task_type": task_type, "n_chunks": n_chunks}),
        encoding="utf-8",
    )
    return refs


# =============================================================================
# Block 1 · _score
# =============================================================================


def test_score_roc_auc_perfect_separation() -> None:
    class _PerfectClf:
        def predict_proba(self, X):                # noqa: D401
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    X_query = np.array([[0.0], [1.0], [0.0], [1.0]])
    y_query = np.array([0, 1, 0, 1])
    auc = _score(
        _PerfectClf(), task_type="classification",
        X_query=X_query, y_query=y_query, metric_name="roc_auc",
    )
    assert auc == pytest.approx(1.0)


def test_score_roc_auc_single_class_returns_nan() -> None:
    class _DummyClf:
        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)) * 0.5,
                                    np.ones(len(X)) * 0.5])

    import math
    out = _score(
        _DummyClf(), task_type="classification",
        X_query=np.zeros((5, 1)), y_query=np.zeros(5, dtype=np.int64),
        metric_name="roc_auc",
    )
    assert math.isnan(out)


def test_score_log_loss_finite() -> None:
    class _Clf:
        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.4),
                                    np.full(len(X), 0.6)])

    ll = _score(
        _Clf(), task_type="classification",
        X_query=np.zeros((10, 1)), y_query=np.array([0, 1] * 5),
        metric_name="log_loss",
    )
    assert ll > 0 and np.isfinite(ll)


def test_score_rmse_zero_on_perfect_predictions() -> None:
    class _PerfectReg:
        def predict(self, X):
            return X[:, 0]

    rmse = _score(
        _PerfectReg(), task_type="regression",
        X_query=np.array([[1.0], [2.0]]), y_query=np.array([1.0, 2.0]),
        metric_name="rmse",
    )
    assert rmse == pytest.approx(0.0, abs=1e-9)


def test_score_unsupported_metric_raises() -> None:
    class _Reg:
        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(ValueError, match="unsupported"):
        _score(
            _Reg(), task_type="regression",
            X_query=np.zeros((1, 1)), y_query=np.zeros(1),
            metric_name="banana",
        )


# =============================================================================
# Block 2 · run_benchmark end-to-end
# =============================================================================


def test_run_benchmark_writes_per_method_csv(tmp_path: Path, monkeypatch) -> None:
    """3 chunks × 2 models × 5 folds = 30 rows total. Each model gets its
    own ``results/<TRACK>/<method>/<run_name>_<timestamp>.csv``."""
    pytest.importorskip("xgboost")
    # Force results into tmp_path so the test doesn't pollute the repo.
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))

    chunks = (
        _make_chunk_refs(tmp_path, track="pd", dataset_id="0001.alpha", n_chunks=2)
        + _make_chunk_refs(tmp_path, track="pd", dataset_id="0002.bravo", n_chunks=1)
    )
    handles_and_models = [
        (ModelHandle(name="xgboost", track="pd", task_type="classification",
                     source="baseline"),
         XGBoostModel(task_type="classification")),
        (ModelHandle(name="logreg",  track="pd", task_type="classification",
                     source="baseline"),
         LogRegModel()),
    ]
    rows = run_benchmark(
        test_chunks=chunks,
        handles_and_models=handles_and_models,
        track="pd",
        metric_name="roc_auc",
        run_name="creditpfn",
        n_folds=5,
        seed=0,
        results_base_dir="results",
    )
    assert len(rows) == 3 * 2 * 5
    assert all(isinstance(r, EvalRow) for r in rows)

    # Per-method dirs exist, one CSV per model.
    pd_dir = tmp_path / "results" / "PD"
    method_dirs = sorted(p.name for p in pd_dir.iterdir() if p.is_dir())
    assert method_dirs == ["logreg", "xgboost"]
    for sub in method_dirs:
        files = list((pd_dir / sub).glob("creditpfn_*.csv"))
        assert len(files) == 1
        df = pd.read_csv(files[0])
        # Each per-method file has 3 chunks × 5 folds = 15 rows.
        assert len(df) == 15
        expected_cols = {
            "track", "task_type", "model_name", "model_source",
            "model_path", "test_dataset_id", "test_chunk_idx", "fold_idx",
            "n_train_rows", "n_test_rows", "metric_name",
            "metric_value", "elapsed_sec", "timestamp", "status", "error",
        }
        assert expected_cols.issubset(set(df.columns))
        assert (df["status"] == "OK").all()


def test_run_benchmark_records_failure_without_killing_loop(
    tmp_path: Path, monkeypatch,
) -> None:
    """A model that raises must produce FAIL rows but not stop the
    other (model × chunk × fold) cells."""
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    chunks = _make_chunk_refs(tmp_path, track="pd", dataset_id="0001.x")

    class _BoomModel:
        task_type = "classification"
        def fit(self, X, y, categorical_idx):
            raise RuntimeError("boom")
        def predict_proba(self, X):                                    # pragma: no cover
            return np.zeros((len(X), 2))

    class _OKModel:
        task_type = "classification"
        def fit(self, X, y, categorical_idx):
            self._mu = float(y.mean()) if len(y) else 0.5
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 1 - self._mu),
                                    np.full(n, self._mu)])

    handles_and_models = [
        (ModelHandle(name="boom", track="pd",
                     task_type="classification", source="baseline"),
         _BoomModel()),
        (ModelHandle(name="ok",   track="pd",
                     task_type="classification", source="baseline"),
         _OKModel()),
    ]
    rows = run_benchmark(
        test_chunks=chunks, handles_and_models=handles_and_models,
        track="pd", metric_name="roc_auc",
        run_name="creditpfn", n_folds=3, seed=0,
        results_base_dir="results",
    )
    # 1 chunk × 2 models × 3 folds = 6 rows.
    assert len(rows) == 6
    boom_rows = [r for r in rows if r.model_name == "boom"]
    ok_rows   = [r for r in rows if r.model_name == "ok"]
    assert all(r.status == "FAIL" for r in boom_rows)
    assert all(r.status == "OK"   for r in ok_rows)
    assert "RuntimeError: boom" in (boom_rows[0].error or "")


def test_run_benchmark_per_task_tag_routes_to_distinct_files(
    tmp_path: Path, monkeypatch,
) -> None:
    """Two parallel slurm tasks for the same method but different
    datasets must NEVER write to the same file. The `per_task_tag`
    arg encodes the dataset_id into the filename suffix."""
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    chunks = _make_chunk_refs(tmp_path, track="pd", dataset_id="0001.alpha")
    handles_and_models = [
        (ModelHandle(name="logreg", track="pd",
                     task_type="classification", source="baseline"),
         LogRegModel()),
    ]
    rows = run_benchmark(
        test_chunks=chunks, handles_and_models=handles_and_models,
        track="pd", metric_name="roc_auc",
        run_name="creditpfn", n_folds=2, seed=0,
        results_base_dir="results",
        per_task_tag="task7_ds-0001.alpha",
    )
    assert rows
    files = list((tmp_path / "results" / "PD" / "logreg").glob("*.csv"))
    assert len(files) == 1
    assert "task7_ds-0001.alpha" in files[0].name


def test_run_benchmark_empty_test_returns_empty_list(tmp_path: Path) -> None:
    """No test chunks → no rows, no CSV crash."""
    rows = run_benchmark(
        test_chunks=[], handles_and_models=[],
        track="pd", metric_name="roc_auc",
        run_name="creditpfn", n_folds=5,
        results_base_dir=str(tmp_path / "results"),
    )
    assert rows == []


def test_method_dirname_baseline() -> None:
    from src.eval import _method_dirname
    h = ModelHandle(name="xgboost", track="pd",
                    task_type="classification", source="baseline")
    assert _method_dirname(h) == "xgboost"


@pytest.mark.parametrize("base,expected", [
    ("checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
     "tabpfn-untuned__v2.6-default"),
    ("checkpoints/tabpfn-v2.5-regressor-v2.5_real.ckpt",
     "tabpfn-untuned__v2.5-real"),
    ("checkpoints/tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
     "tabpfn-untuned__v2.5-default-2"),
])
def test_method_dirname_tabpfn_untuned(base: str, expected: str) -> None:
    """The short-tag scheme drops the redundant 'classifier'/'regressor'
    infix (already encoded in the parent results/<TRACK>/ folder) and
    the 'tabpfn-' prefix."""
    from src.eval import _method_dirname
    h = ModelHandle(
        name="tabpfn-untuned[x]", track="pd",
        task_type="classification", source="tabpfn-untuned",
        base_path=base,
    )
    assert _method_dirname(h) == expected


def test_method_dirname_tabpfn_trained_includes_lr_and_policy() -> None:
    """A tabpfn-trained dirname must encode the THREE knobs that vary
    across training trials: short base tag, lr, multi_chunk_policy.
    Seed is intentionally not in the dirname (different seed → new
    timestamped CSV inside the same dirname)."""
    from src.eval import _method_dirname
    h = ModelHandle(
        name="tabpfn-trained[…]", track="pd",
        task_type="classification", source="tabpfn-trained",
        base_path="checkpoints/trained/pd/creditpfn_pd_some_long_name.ckpt",
        extra={
            "base_checkpoint":    "checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
            "learning_rate":      1.0e-5,
            "multi_chunk_policy": "all_chunks_as_separate_datasets",
            "seed":               42,
        },
    )
    out = _method_dirname(h)
    assert out == "tabpfn-trained__v2.6-default__lr1e-05__allchunks"


def test_method_dirname_tabpfn_trained_first_chunk_only() -> None:
    from src.eval import _method_dirname
    h = ModelHandle(
        name="tabpfn-trained[…]", track="lgd",
        task_type="regression", source="tabpfn-trained",
        base_path="anywhere.ckpt",
        extra={
            "base_checkpoint":    "checkpoints/tabpfn-v2.5-regressor-v2.5_default.ckpt",
            "learning_rate":      5.0e-5,
            "multi_chunk_policy": "first_chunk_only",
            "seed":               7,
        },
    )
    assert _method_dirname(h) == "tabpfn-trained__v2.5-default__lr5e-05__firstchunk"


# =============================================================================
# Block 3 · load_trained_handles
# =============================================================================


def _make_synthetic_manifest(path: Path, *, rows: list[dict]) -> None:
    """Write a CSV that matches the schema of `RunRow`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track", "base_checkpoint", "learning_rate", "multi_chunk_policy",
        "seed", "test_metric_name", "test_metric_raw",
        "n_train_datasets", "n_test_datasets",
        "n_train_chunks", "n_test_chunks",
        "final_ckpt_path", "elapsed_sec", "status", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def test_load_trained_handles_skips_failed_and_missing(tmp_path: Path) -> None:
    """Manifest rows with status=FAIL or non-existent ckpt are skipped."""
    ckpt_ok = tmp_path / "ok.ckpt"
    ckpt_ok.write_bytes(b"")            # touch
    manifest = tmp_path / "logs" / "runs" / "creditpfn_pd.csv"
    _make_synthetic_manifest(manifest, rows=[
        {"track": "pd", "base_checkpoint": "x", "learning_rate": "1e-5",
         "multi_chunk_policy": "allchunks", "seed": "42",
         "final_ckpt_path": str(ckpt_ok), "status": "OK"},
        {"track": "pd", "base_checkpoint": "x", "learning_rate": "1e-5",
         "multi_chunk_policy": "allchunks", "seed": "42",
         "final_ckpt_path": "", "status": "FAIL"},
        {"track": "pd", "base_checkpoint": "x", "learning_rate": "1e-5",
         "multi_chunk_policy": "allchunks", "seed": "42",
         "final_ckpt_path": str(tmp_path / "missing.ckpt"), "status": "OK"},
        {"track": "lgd", "base_checkpoint": "x", "learning_rate": "1e-5",
         "multi_chunk_policy": "allchunks", "seed": "42",
         "final_ckpt_path": str(ckpt_ok), "status": "OK"},
    ])

    handles = load_trained_handles(manifest, track="pd")
    assert len(handles) == 1
    handle, model = handles[0]
    assert handle.source == "tabpfn-trained"
    assert handle.task_type == "classification"
    assert "ok" in handle.name


def test_load_trained_handles_no_manifest_returns_empty(tmp_path: Path) -> None:
    handles = load_trained_handles(tmp_path / "does_not_exist.csv", track="pd")
    assert handles == []


# =============================================================================
# Block 4 · scripts/eval_pipeline.py — task indexing + filters
# =============================================================================


def _dummy_handles_and_chunks(tmp_path: Path):
    """Build a 2-model × 3-dataset roster for filter tests."""
    chunks = (
        _make_chunk_refs(tmp_path, track="pd", dataset_id="0001.alpha")
        + _make_chunk_refs(tmp_path, track="pd", dataset_id="0002.bravo")
        + _make_chunk_refs(tmp_path, track="pd", dataset_id="0003.charlie")
    )
    handles_and_models = [
        (ModelHandle(name="xgboost", track="pd",
                     task_type="classification", source="baseline"), object()),
        (ModelHandle(name="logreg", track="pd",
                     task_type="classification", source="baseline"), object()),
    ]
    return handles_and_models, chunks


def test_enumerate_tasks_cartesian_product(tmp_path: Path) -> None:
    """`_enumerate_tasks` returns one entry per (model_idx, dataset_id)
    pair — 2 models × 3 datasets = 6 tasks. Within each model_idx
    group the dataset IDs are sorted (deterministic ordering across
    runs)."""
    import scripts.eval_pipeline as ep
    h, c = _dummy_handles_and_chunks(tmp_path)
    pairs = ep._enumerate_tasks(h, c)
    assert len(pairs) == 6
    assert len(set(pairs)) == 6                     # no dupes
    by_model: dict[int, list[str]] = {}
    for m_idx, ds in pairs:
        by_model.setdefault(m_idx, []).append(ds)
    for ds_list in by_model.values():
        assert ds_list == sorted(ds_list)


def test_filter_roster_task_index_picks_one_pair(tmp_path: Path) -> None:
    import scripts.eval_pipeline as ep
    h, c = _dummy_handles_and_chunks(tmp_path)
    keep_models, keep_chunks = ep._filter_roster(
        h, c, method_filter=[], dataset_filter=[], task_index=3,
    )
    # Exactly one model, exactly one dataset's chunks.
    assert len(keep_models) == 1
    assert len({k.dataset_id for k in keep_chunks}) == 1


def test_filter_roster_method_filter_intersects(tmp_path: Path) -> None:
    import scripts.eval_pipeline as ep
    h, c = _dummy_handles_and_chunks(tmp_path)
    keep_models, keep_chunks = ep._filter_roster(
        h, c, method_filter=["logreg"], dataset_filter=[], task_index=None,
    )
    assert [hh.name for hh, _ in keep_models] == ["logreg"]
    # All chunks kept (no dataset filter).
    assert len(keep_chunks) == len(c)


def test_filter_roster_dataset_filter_intersects(tmp_path: Path) -> None:
    import scripts.eval_pipeline as ep
    h, c = _dummy_handles_and_chunks(tmp_path)
    keep_models, keep_chunks = ep._filter_roster(
        h, c, method_filter=[], dataset_filter=["0002.bravo"], task_index=None,
    )
    assert {k.dataset_id for k in keep_chunks} == {"0002.bravo"}
    assert len(keep_models) == len(h)


def test_filter_roster_task_index_out_of_bounds_raises(tmp_path: Path) -> None:
    import scripts.eval_pipeline as ep
    h, c = _dummy_handles_and_chunks(tmp_path)
    with pytest.raises(IndexError, match="out of bounds"):
        ep._filter_roster(h, c, method_filter=[], dataset_filter=[], task_index=999)
