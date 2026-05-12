"""Smoke + unit tests for the eval benchmark.

Layout choice
-------------
One file per ``src/`` subpackage, like ``test_data.py`` and
``test_train.py``. ``test_eval.py`` covers everything in
``src/eval/``: the processed-CSV loader, per-method subsampling,
fold-aware cat encoding, K-fold + inner-val splits, comprehensive
metrics, and the per-(method × dataset) parallelisation in
``scripts/eval_pipeline.py``.

Coverage map
------------
    Block 1  dataset_loader.py       — load_processed_dataset, subsample,
                                       encode_for_model
    Block 2  benchmark.py            — fold construction, comprehensive
                                       metrics, end-to-end on a synthetic
                                       processed CSV
    Block 3  benchmark.resolve_test_datasets — provenance > cfg fallback
    Block 4  scripts/eval_pipeline.py — task indexing + filters
    Block 5  benchmark._method_dirname — naming for the results layout
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.benchmark import (
    EvalRow, _output_path_for, _method_dirname,
    find_existing_results,
    load_trained_handles, resolve_test_datasets, run_benchmark,
)
from src.eval.dataset_loader import (
    ProcessedDataset, encode_for_model, load_processed_dataset, subsample,
)
from src.model.base import ModelHandle
from src.model.boosting import XGBoostModel
from src.model.linear import LogRegModel


# =============================================================================
# Helpers
# =============================================================================


def _write_processed_dataset(
    out_root: Path, *,
    track: str, dataset_id: str,
    n_rows: int = 200, n_features: int = 5, n_cat: int = 1,
    task_type: str | None = None,
) -> tuple[Path, Path]:
    """Write a sanitised CSV + manifest entry that the eval loader can read."""
    task_type = task_type or (
        "classification" if track == "pd" else "regression"
    )
    folder = out_root / "data" / "processed" / track
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{dataset_id}.sanitized.csv"
    rng = np.random.default_rng(abs(hash(dataset_id)) % (2**32))

    cols: dict = {}
    cat_names = []
    for j in range(n_features):
        if j < n_cat:
            cat_names.append(f"cat_{j}")
            cols[f"cat_{j}"] = rng.choice(["A", "B", "C"], size=n_rows)
        else:
            cols[f"num_{j}"] = rng.standard_normal(n_rows).astype(np.float32)
    target_col = "target"
    if task_type == "classification":
        cols[target_col] = (
            (cols.get("num_1", rng.standard_normal(n_rows)) > 0).astype(np.int64)
        )
    else:
        cols[target_col] = rng.uniform(0, 1, n_rows).astype(np.float32)
    pd.DataFrame(cols).to_csv(csv_path, index=False)

    # Manifest row.
    manifest_path = out_root / "data" / f"manifest_{track}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "dataset_id":     dataset_id,
        "track":          track,
        "task_type":      task_type,
        "target_column":  target_col,
        "n_rows":         n_rows,
        "n_cols":         n_features,
        "n_categorical":  n_cat,
        "n_numerical":    n_features - n_cat,
        "missing_rate":   "0.0",
        "minority_class_ratio": "0.5" if task_type == "classification" else "",
        "target_mean":    "0.5",
        "target_std":     "0.3",
        "categorical_columns": ";".join(cat_names),
        "source":         "synthetic",
        "shape_hash":     "deadbeef",
    }
    if manifest_path.exists():
        df = pd.read_csv(manifest_path, dtype=str).fillna("")
        df = df[df["dataset_id"] != dataset_id]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(manifest_path, index=False)
    return csv_path, manifest_path


@pytest.fixture
def env_isolated(monkeypatch, tmp_path):
    """Route both DATA_ROOT and OUTPUT_ROOT to tmp_path so a test can
    write processed CSVs + manifests without polluting the repo."""
    monkeypatch.setenv("CREDITPFN_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("VSC_HOME", raising=False)
    monkeypatch.delenv("VSC_DATA", raising=False)
    return tmp_path


# =============================================================================
# Block 1 · dataset_loader.py
# =============================================================================


def test_load_processed_dataset_round_trips(env_isolated: Path) -> None:
    _write_processed_dataset(
        env_isolated, track="pd", dataset_id="0001.alpha", n_rows=120,
    )
    ds = load_processed_dataset(track="pd", dataset_id="0001.alpha")
    assert isinstance(ds, ProcessedDataset)
    assert ds.task_type == "classification"
    assert ds.n_rows == 120
    assert ds.n_features == 5
    assert "cat_0" in ds.categorical_columns


def test_load_processed_dataset_missing_csv_raises(env_isolated: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_processed_dataset(track="pd", dataset_id="9999.nope")


def test_subsample_no_op_when_under_cap() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(50)})
    y = rng.integers(0, 2, 50)
    Xo, yo = subsample(X, y, max_rows=200, seed=0, stratify=True)
    assert len(Xo) == 50 and len(yo) == 50


def test_subsample_caps_when_over() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(1000)})
    y = rng.integers(0, 2, 1000)
    Xo, yo = subsample(X, y, max_rows=100, seed=0, stratify=True)
    # Stratified subsample is approximate; allow a small slop.
    assert 90 <= len(Xo) <= 110
    assert len(Xo) == len(yo)


def test_encode_for_model_handles_unseen_val_categories() -> None:
    """Categories that appear only in val/test must encode to the
    unknown_value sentinel (-1), matching TabPFN's inference path."""
    X_tr = pd.DataFrame({"cat": ["A", "B", "A"], "num": [1.0, 2.0, 3.0]})
    X_va = pd.DataFrame({"cat": ["A", "C"],      "num": [4.0, 5.0]})  # "C" unseen
    X_te = pd.DataFrame({"cat": ["B"],            "num": [6.0]})
    Atr, Ava, Ate, cat_idx = encode_for_model(
        X_tr, X_va, X_te, categorical_columns=["cat"],
    )
    assert cat_idx == [0]
    # The "C" in val should encode to -1.
    assert Ava[1, 0] == -1.0


# =============================================================================
# Block 2 · benchmark.py — end-to-end on synthetic processed CSV
# =============================================================================


def test_run_benchmark_writes_per_method_csv(env_isolated: Path) -> None:
    """1 dataset × 2 models × 3 folds = 6 rows total; per-method CSVs."""
    pytest.importorskip("xgboost")
    _write_processed_dataset(
        env_isolated, track="pd", dataset_id="0001.alpha", n_rows=200,
    )
    handles_and_models = [
        (ModelHandle(name="xgboost", track="pd",
                     task_type="classification", source="baseline"),
         XGBoostModel(task_type="classification")),
        (ModelHandle(name="logreg",  track="pd",
                     task_type="classification", source="baseline"),
         LogRegModel()),
    ]
    rows = run_benchmark(
        test_dataset_ids=["0001.alpha"],
        handles_and_models=handles_and_models,
        track="pd",
        run_name="creditpfn",
        n_folds=3, inner_val_fraction=0.20, seed=0,
        results_base_dir="results",
    )
    assert len(rows) == 1 * 2 * 3
    assert all(isinstance(r, EvalRow) for r in rows)
    assert all(r.status == "OK" for r in rows)

    pd_dir = env_isolated / "results" / "PD"
    method_dirs = sorted(p.name for p in pd_dir.iterdir() if p.is_dir())
    assert method_dirs == ["logreg", "xgboost"]
    for sub in method_dirs:
        files = list((pd_dir / sub).glob("creditpfn_*.csv"))
        assert len(files) == 1
        df = pd.read_csv(files[0])
        assert len(df) == 3
        # Comprehensive metric columns are present.
        for col in ("roc_auc", "log_loss", "pr_auc",
                    "optimal_threshold", "f1", "accuracy", "precision", "recall",
                    "rmse", "mae", "r2", "neg_nll",
                    "n_train_rows", "n_val_rows", "n_test_rows"):
            assert col in df.columns, f"missing metric column: {col}"


def test_run_benchmark_classification_metrics_are_finite(env_isolated: Path) -> None:
    """A working classifier on a separable synthetic dataset must produce
    finite ROC-AUC, log-loss, F1 etc."""
    pytest.importorskip("xgboost")
    _write_processed_dataset(
        env_isolated, track="pd", dataset_id="0001.alpha", n_rows=300,
    )
    handles = [(
        ModelHandle(name="xgboost", track="pd",
                    task_type="classification", source="baseline"),
        XGBoostModel(task_type="classification"),
    )]
    rows = run_benchmark(
        test_dataset_ids=["0001.alpha"],
        handles_and_models=handles,
        track="pd", run_name="r",
        n_folds=3, seed=0,
        results_base_dir="results",
    )
    assert all(r.status == "OK" for r in rows)
    for r in rows:
        assert math.isfinite(r.roc_auc)
        assert math.isfinite(r.log_loss)
        assert math.isfinite(r.pr_auc)
        # Threshold-tuned columns are finite for binary classification.
        assert 0.0 <= r.optimal_threshold <= 1.0
        assert 0.0 <= r.f1 <= 1.0
        assert 0.0 <= r.accuracy <= 1.0


def test_run_benchmark_split_sizes_match_user_contract(env_isolated: Path) -> None:
    """5-fold CV with inner_val_fraction=0.2 produces 64/16/20 splits.
    On a 200-row dataset that's ~128 / ~32 / ~40."""
    pytest.importorskip("xgboost")
    _write_processed_dataset(
        env_isolated, track="pd", dataset_id="0001.alpha", n_rows=200,
    )
    handles = [(
        ModelHandle(name="xgboost", track="pd",
                    task_type="classification", source="baseline"),
        XGBoostModel(task_type="classification"),
    )]
    rows = run_benchmark(
        test_dataset_ids=["0001.alpha"],
        handles_and_models=handles,
        track="pd", run_name="r",
        n_folds=5, inner_val_fraction=0.20, seed=0,
        results_base_dir="results",
    )
    for r in rows:
        # Outer test fold ≈ 20% of 200 = 40 (allow ±2 for stratified rounding)
        assert 38 <= r.n_test_rows <= 42
        # Inner train ≈ 64% = 128, val ≈ 16% = 32
        assert 124 <= r.n_train_rows <= 132
        assert 28 <= r.n_val_rows <= 36
        # n_train + n_val + n_test ≈ 200
        assert 195 <= r.n_train_rows + r.n_val_rows + r.n_test_rows <= 205


def test_run_benchmark_records_failure_without_killing_loop(
    env_isolated: Path,
) -> None:
    _write_processed_dataset(
        env_isolated, track="pd", dataset_id="0001.alpha", n_rows=120,
    )

    class _BoomModel:
        task_type = "classification"
        def fit(self, X, y, categorical_idx, X_val=None, y_val=None):
            raise RuntimeError("boom")
        def predict_proba(self, X):                                    # pragma: no cover
            return np.zeros((len(X), 2))

    class _OKModel:
        task_type = "classification"
        def fit(self, X, y, categorical_idx, X_val=None, y_val=None):
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
        test_dataset_ids=["0001.alpha"],
        handles_and_models=handles_and_models,
        track="pd", run_name="r", n_folds=3, seed=0,
        results_base_dir="results",
    )
    assert len(rows) == 6
    assert sum(r.status == "FAIL" for r in rows if r.model_name == "boom") == 3
    assert sum(r.status == "OK"   for r in rows if r.model_name == "ok")   == 3


def test_run_benchmark_per_task_tag_routes_to_distinct_files(
    env_isolated: Path,
) -> None:
    _write_processed_dataset(
        env_isolated, track="pd", dataset_id="0001.alpha", n_rows=120,
    )
    handles = [(
        ModelHandle(name="logreg", track="pd",
                    task_type="classification", source="baseline"),
        LogRegModel(),
    )]
    rows = run_benchmark(
        test_dataset_ids=["0001.alpha"],
        handles_and_models=handles,
        track="pd", run_name="creditpfn", n_folds=2, seed=0,
        results_base_dir="results",
        per_task_tag="task7_ds-0001.alpha",
    )
    assert rows
    files = list((env_isolated / "results" / "PD" / "logreg").glob("*.csv"))
    assert len(files) == 1
    assert "task7_ds-0001.alpha" in files[0].name


def test_run_benchmark_empty_returns_empty(env_isolated: Path) -> None:
    rows = run_benchmark(
        test_dataset_ids=[], handles_and_models=[],
        track="pd", run_name="r", n_folds=5,
        results_base_dir="results",
    )
    assert rows == []


# Regression test for Gemini's #1: the benchmark MUST pass the eval's
# (X_val, y_val) through to model.fit; the model must NOT do its own
# internal train/test split for HPO.

def test_benchmark_passes_inner_val_to_model_fit(env_isolated: Path) -> None:
    """Capture every fit call and check X_val/y_val are populated with
    the eval's inner-val split."""
    _write_processed_dataset(
        env_isolated, track="pd", dataset_id="0001.alpha", n_rows=120,
    )

    captured: list[dict] = []

    class _SpyModel:
        task_type = "classification"
        def fit(self, X, y, categorical_idx, X_val=None, y_val=None):
            captured.append({
                "n_train": len(X), "n_val": (len(X_val) if X_val is not None else None),
                "val_is_set": X_val is not None and y_val is not None,
            })
            self._mu = float(y.mean()) if len(y) else 0.5
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 1 - self._mu),
                                    np.full(n, self._mu)])

    rows = run_benchmark(
        test_dataset_ids=["0001.alpha"],
        handles_and_models=[(
            ModelHandle(name="spy", track="pd",
                        task_type="classification", source="baseline"),
            _SpyModel(),
        )],
        track="pd", run_name="r", n_folds=3, inner_val_fraction=0.20, seed=0,
        results_base_dir="results",
    )
    assert len(rows) == 3
    assert all(r.status == "OK" for r in rows)
    # Every fold's fit got a non-empty val split.
    assert len(captured) == 3
    for c in captured:
        assert c["val_is_set"] is True
        assert c["n_val"] is not None and c["n_val"] > 0
        # 3-fold CV on 120 rows: test ≈ 40, train ≈ 80;
        # inner split of train: sub-train ≈ 64, val ≈ 16
        assert 58 <= c["n_train"] <= 70
        assert 12 <= c["n_val"] <= 20


# Regression test for Gemini's #3: threshold tuning via
# precision_recall_curve should run in O(n) time on a large val set,
# not O(n²). We test for correctness (still returns the right threshold).

def test_best_f1_threshold_via_pr_curve() -> None:
    """Synthetic: separable scores, optimum threshold should be around 0.5."""
    from src.eval.benchmark import _best_f1_threshold
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.standard_normal(n) > 0).astype(int)
    # Scores correlated with y but noisy.
    proba_pos = np.clip(y * 0.7 + rng.standard_normal(n) * 0.15, 0, 1)
    th = _best_f1_threshold(proba_pos, y)
    assert 0.3 <= th <= 0.7         # in the sensible neighbourhood
    assert isinstance(th, float)


def test_best_f1_threshold_falls_back_when_no_positive() -> None:
    from src.eval.benchmark import _best_f1_threshold
    # All-zero labels → no positives → F1 always 0 → fallback 0.5.
    proba = np.linspace(0.1, 0.9, 50)
    y = np.zeros(50, dtype=int)
    assert _best_f1_threshold(proba, y) == 0.5


# =============================================================================
# Block 3 · resolve_test_datasets — provenance > cfg fallback
# =============================================================================


def test_resolve_test_datasets_uses_provenance_for_trained(tmp_path: Path) -> None:
    ckpt = tmp_path / "trained.ckpt"
    ckpt.write_bytes(b"")
    (tmp_path / "trained.ckpt.provenance.json").write_text(
        json.dumps({"test_datasets": ["0001.foo", "0002.bar"]}),
        encoding="utf-8",
    )
    handle = ModelHandle(
        name="tabpfn-trained[…]", track="pd",
        task_type="classification", source="tabpfn-trained",
        base_path=str(ckpt), extra={},
    )
    out = resolve_test_datasets(handle, cfg_test_dataset_ids=["zzz"])
    assert out == ["0001.foo", "0002.bar"]


def test_resolve_test_datasets_falls_back_to_cfg_for_others() -> None:
    handle_baseline = ModelHandle(
        name="xgboost", track="pd",
        task_type="classification", source="baseline",
    )
    handle_untuned = ModelHandle(
        name="tabpfn-untuned[v2.6]", track="pd",
        task_type="classification", source="tabpfn-untuned",
        base_path="checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
    )
    cfg_test = ["0001.alpha", "0002.bravo"]
    assert resolve_test_datasets(handle_baseline, cfg_test_dataset_ids=cfg_test) == cfg_test
    assert resolve_test_datasets(handle_untuned,  cfg_test_dataset_ids=cfg_test) == cfg_test


def test_resolve_test_datasets_falls_back_when_provenance_missing(tmp_path: Path) -> None:
    """tabpfn-trained whose ckpt has no provenance falls back to cfg."""
    handle = ModelHandle(
        name="tabpfn-trained[…]", track="pd",
        task_type="classification", source="tabpfn-trained",
        base_path=str(tmp_path / "ghost.ckpt"),  # doesn't exist
        extra={},
    )
    cfg_test = ["0001.alpha"]
    out = resolve_test_datasets(handle, cfg_test_dataset_ids=cfg_test)
    assert out == cfg_test


# =============================================================================
# Block 4 · scripts/eval_pipeline.py — task indexing + filters
# =============================================================================


def _two_baseline_handles_and_models():
    return [
        (ModelHandle(name="xgboost", track="pd",
                     task_type="classification", source="baseline"), object()),
        (ModelHandle(name="logreg", track="pd",
                     task_type="classification", source="baseline"), object()),
    ]


def test_enumerate_tasks_cartesian_product() -> None:
    """2 models × 3 cfg test datasets = 6 tasks."""
    import scripts.eval_pipeline as ep
    pairs = ep._enumerate_tasks(
        _two_baseline_handles_and_models(),
        ["0001.alpha", "0002.bravo", "0003.charlie"],
    )
    assert len(pairs) == 6
    assert len(set(pairs)) == 6
    # Within each model_idx group the dataset IDs are sorted.
    by_model: dict[int, list[str]] = {}
    for m_idx, ds in pairs:
        by_model.setdefault(m_idx, []).append(ds)
    for ds_list in by_model.values():
        assert ds_list == sorted(ds_list)


def test_filter_roster_task_index_picks_one_pair() -> None:
    import scripts.eval_pipeline as ep
    plan = ep._filter_roster(
        _two_baseline_handles_and_models(),
        ["0001.alpha", "0002.bravo", "0003.charlie"],
        method_filter=[], dataset_filter=[], task_index=3,
    )
    assert len(plan) == 1
    handle_and_model, ds_ids = plan[0]
    assert len(ds_ids) == 1


def test_filter_roster_method_filter() -> None:
    import scripts.eval_pipeline as ep
    plan = ep._filter_roster(
        _two_baseline_handles_and_models(),
        ["0001.alpha"],
        method_filter=["logreg"], dataset_filter=[], task_index=None,
    )
    assert len(plan) == 1
    assert plan[0][0][0].name == "logreg"


def test_filter_roster_dataset_filter() -> None:
    import scripts.eval_pipeline as ep
    plan = ep._filter_roster(
        _two_baseline_handles_and_models(),
        ["0001.alpha", "0002.bravo"],
        method_filter=[], dataset_filter=["0002.bravo"], task_index=None,
    )
    for _, ds_ids in plan:
        assert ds_ids == ["0002.bravo"]


def test_filter_roster_task_index_out_of_bounds_is_soft_no_op() -> None:
    """Out-of-range ``--task-index`` returns an empty plan rather than
    raising. This is the contract over-sized slurm arrays rely on: a
    surplus task should exit zero cleanly, not fail with IndexError.
    """
    import scripts.eval_pipeline as ep
    plan = ep._filter_roster(
        _two_baseline_handles_and_models(),
        ["0001.alpha"],
        method_filter=[], dataset_filter=[], task_index=999,
    )
    assert plan == []


# =============================================================================
# Block 5 · _method_dirname  (regression: still produces clean names)
# =============================================================================


def test_method_dirname_baseline() -> None:
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
    h = ModelHandle(
        name="tabpfn-untuned[x]", track="pd",
        task_type="classification", source="tabpfn-untuned",
        base_path=base,
    )
    assert _method_dirname(h) == expected


def test_method_dirname_tabpfn_trained_includes_lr() -> None:
    h = ModelHandle(
        name="tabpfn-trained[…]", track="pd",
        task_type="classification", source="tabpfn-trained",
        base_path="checkpoints/trained/pd/whatever.ckpt",
        extra={
            "base_checkpoint":    "checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
            "learning_rate":      1.0e-4,
            "seed":               42,
        },
    )
    assert _method_dirname(h) == "tabpfn-trained__v2.6-default__lr1e-04"


def test_method_dirname_tabpfn_trained_separates_lora_variants() -> None:
    """LoRA-trained and full-FT checkpoints with otherwise-identical HPs
    must land in distinct result directories so their CSVs don't mix."""
    common_extra = {
        "base_checkpoint":    "checkpoints/tabpfn-v3-classifier-v3_default.ckpt",
        "learning_rate":      1.0e-4,
        "seed":               42,
    }
    h_full = ModelHandle(
        name="full", track="pd", task_type="classification",
        source="tabpfn-trained", base_path="x",
        extra={**common_extra, "use_lora": False},
    )
    h_lora = ModelHandle(
        name="lora", track="pd", task_type="classification",
        source="tabpfn-trained", base_path="x",
        extra={**common_extra, "use_lora": True},
    )
    assert _method_dirname(h_full) == "tabpfn-trained__v3-default__lr1e-04"
    assert _method_dirname(h_lora) == "tabpfn-trained__v3-default__lr1e-04__lora"


# =============================================================================
# Block 6 · load_trained_handles — manifest reading
# =============================================================================


def _write_synthetic_manifest(path: Path, *, rows: list[dict]) -> None:
    import csv
    fieldnames = [
        "track", "base_checkpoint", "learning_rate",
        "seed", "n_train_datasets", "n_test_datasets",
        "n_train_chunks", "n_test_chunks",
        "final_ckpt_path", "elapsed_sec", "status", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def test_load_trained_handles_skips_failed_and_missing(tmp_path: Path) -> None:
    ckpt_ok = tmp_path / "ok.ckpt"
    ckpt_ok.write_bytes(b"")
    manifest = tmp_path / "logs" / "runs" / "creditpfn_pd.csv"
    _write_synthetic_manifest(manifest, rows=[
        {"track": "pd", "base_checkpoint": "x", "learning_rate": "1e-4",
         "seed": "42",
         "final_ckpt_path": str(ckpt_ok), "status": "OK"},
        {"track": "pd", "base_checkpoint": "x", "learning_rate": "1e-4",
         "seed": "42",
         "final_ckpt_path": "", "status": "FAIL"},
        {"track": "pd", "base_checkpoint": "x", "learning_rate": "1e-4",
         "seed": "42",
         "final_ckpt_path": str(tmp_path / "missing.ckpt"), "status": "OK"},
        {"track": "lgd", "base_checkpoint": "x", "learning_rate": "1e-4",
         "seed": "42",
         "final_ckpt_path": str(ckpt_ok), "status": "OK"},
    ])
    handles = load_trained_handles(manifest, track="pd")
    assert len(handles) == 1
    handle, _model = handles[0]
    assert handle.source == "tabpfn-trained"


def test_load_trained_handles_no_manifest_returns_empty(tmp_path: Path) -> None:
    handles = load_trained_handles(tmp_path / "does_not_exist.csv", track="pd")
    assert handles == []


# =============================================================================
# Block 7 · find_existing_results — rerun-skip helper
# =============================================================================


def _xgb_handle() -> ModelHandle:
    return ModelHandle(
        name="xgboost", track="pd",
        task_type="classification", source="baseline",
    )


def _write_eval_csv(path: Path, *, rows: list[dict]) -> None:
    import csv as _csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "model_source", "model_path",
                  "test_dataset_id", "fold_idx", "status"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def test_find_existing_results_missing_dir(tmp_path: Path, monkeypatch) -> None:
    """No method dir on disk → empty list (not an error)."""
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    out = find_existing_results(
        _xgb_handle(), "0001.gmsc",
        track="pd", results_base_dir="results/benchmark",
    )
    assert out == []


def test_find_existing_results_matches_per_task_csv(
    tmp_path: Path, monkeypatch,
) -> None:
    """Slurm-style per-task CSV with an OK row counts as 'already scored'."""
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    method_dir = tmp_path / "results" / "benchmark" / "PD" / "xgboost"
    _write_eval_csv(
        method_dir / "creditpfn_2026_task7_ds-0001.gmsc.csv",
        rows=[{"model_name": "xgboost", "model_source": "baseline",
               "test_dataset_id": "0001.gmsc", "fold_idx": 0,
               "status": "OK"}],
    )
    hits = find_existing_results(
        _xgb_handle(), "0001.gmsc",
        track="pd", results_base_dir="results/benchmark",
    )
    assert len(hits) == 1
    # Different dataset → no hit.
    assert find_existing_results(
        _xgb_handle(), "0002.taiwan_creditcard",
        track="pd", results_base_dir="results/benchmark",
    ) == []


def test_find_existing_results_ignores_fail_only_rows(
    tmp_path: Path, monkeypatch,
) -> None:
    """A CSV with only FAIL rows for the target dataset should NOT count
    as 'already scored' — the caller should retry."""
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    method_dir = tmp_path / "results" / "benchmark" / "PD" / "xgboost"
    _write_eval_csv(
        method_dir / "creditpfn_2026_task7_ds-0001.gmsc.csv",
        rows=[{"model_name": "xgboost", "model_source": "baseline",
               "test_dataset_id": "0001.gmsc", "fold_idx": 0,
               "status": "FAIL"}],
    )
    assert find_existing_results(
        _xgb_handle(), "0001.gmsc",
        track="pd", results_base_dir="results/benchmark",
    ) == []


def test_find_existing_results_matches_single_process_csv(
    tmp_path: Path, monkeypatch,
) -> None:
    """A non-tagged CSV (single-process run) that contains a matching
    OK row is detected via row-level inspection."""
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    method_dir = tmp_path / "results" / "benchmark" / "PD" / "xgboost"
    _write_eval_csv(
        method_dir / "creditpfn_2026.csv",
        rows=[
            {"model_name": "xgboost", "model_source": "baseline",
             "test_dataset_id": "0002.taiwan_creditcard", "fold_idx": 0,
             "status": "OK"},
            {"model_name": "xgboost", "model_source": "baseline",
             "test_dataset_id": "0001.gmsc", "fold_idx": 0,
             "status": "OK"},
        ],
    )
    hits = find_existing_results(
        _xgb_handle(), "0001.gmsc",
        track="pd", results_base_dir="results/benchmark",
    )
    assert len(hits) == 1


def test_find_existing_results_requires_all_folds_when_count_given(
    tmp_path: Path, monkeypatch,
) -> None:
    """When ``n_folds_required`` is passed, a pair with only some folds
    OK should NOT be reported as scored — so the caller re-runs and
    retries the missing folds."""
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    method_dir = tmp_path / "results" / "benchmark" / "PD" / "xgboost"
    # 1 OK fold + 4 FAIL folds for the same dataset.
    rows = [
        {"model_name": "xgboost", "model_source": "baseline",
         "test_dataset_id": "0001.gmsc", "fold_idx": 0, "status": "OK"},
    ] + [
        {"model_name": "xgboost", "model_source": "baseline",
         "test_dataset_id": "0001.gmsc", "fold_idx": k, "status": "FAIL"}
        for k in (1, 2, 3, 4)
    ]
    _write_eval_csv(
        method_dir / "creditpfn_2026_task7_ds-0001.gmsc.csv", rows=rows,
    )
    # Partial-folds scenario: with n_folds_required=5, the pair must NOT
    # be treated as already scored.
    assert find_existing_results(
        _xgb_handle(), "0001.gmsc",
        track="pd", results_base_dir="results/benchmark",
        n_folds_required=5,
    ) == []
    # Adding the four missing OK folds → pair is now complete.
    full_rows = [
        {"model_name": "xgboost", "model_source": "baseline",
         "test_dataset_id": "0001.gmsc", "fold_idx": k, "status": "OK"}
        for k in (0, 1, 2, 3, 4)
    ]
    _write_eval_csv(
        method_dir / "creditpfn_2026_task7_ds-0001.gmsc.csv", rows=full_rows,
    )
    hits = find_existing_results(
        _xgb_handle(), "0001.gmsc",
        track="pd", results_base_dir="results/benchmark",
        n_folds_required=5,
    )
    assert len(hits) == 1
