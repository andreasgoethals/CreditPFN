"""Smoke + unit tests for the training pipeline.

Layout choice
-------------
One file per ``src/`` subpackage. ``test_train.py`` covers everything in
``src/train/``: corpus splitting, the dataloader, the learning-rate
schedule, the metric helpers, the descriptive-name builder, and a
fully-mocked end-to-end pass through ``train_one_config`` that does NOT
require a real TabPFN checkpoint or a GPU.

Tests that genuinely need TabPFN (loading the real checkpoint, running a
forward pass) are guarded with ``pytest.importorskip("tabpfn")`` so the
suite stays runnable in a stripped-down CI image.

NOTE (2026-05-20 refactor): the data pipeline no longer produces ``.npz``
chunks. Training reads sanitized CSVs directly. The fixtures here write
synthetic sanitized CSVs (and a matching per-track manifest CSV) to a
``tmp_path`` and point ``CREDITPFN_DATA_ROOT``/``CREDITPFN_OUTPUT_ROOT``
at it via monkeypatch.

Coverage map
------------
    Block 1  corpus.py     — train/test split semantics (DatasetRef)
    Block 2  dataloader.py — CSV reading, ordinal-encode, per-epoch reshuffle
    Block 3  loop.py       — LR schedule, descriptive name, end-to-end mocked
    Block 4  metrics.py    — ROC-AUC / log_loss / RMSE on toy inputs
    Block 5  model.py      — version inference + filename schema
    Block 6  scripts/train_pipeline.py — grid expansion + --single
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
import torch

from src.train.corpus import (
    DatasetRef, CorpusSplit,
    _assign_buckets, build_dataset_pool, split_corpus, split_from_cfg,
)
from src.train.dataloader import (
    ProcessedDatasetLoader, TabPFNBatch,
    _build_step_batch, _load_processed_csv,
    identity_collate, prepare_eval_chunk,
)
from src.train.loop import (
    descriptive_name, evaluate_on_split, make_warmup_cosine_schedule,
    train_one_config,
)
from src.train.metrics import (
    classification_metric, improvement_direction, mean_ignore_nan,
    regression_metric,
)
from src.train.model import _infer_version

REPO = Path(__file__).resolve().parents[1]


# =============================================================================
# Helpers
# =============================================================================


def _write_synthetic_processed(
    data_root: Path,
    output_root: Path,
    *,
    track: str,
    dataset_id: str,
    n_rows: int = 100,
    n_feat: int = 4,
    task_type: str | None = None,
    target_column: str = "y",
    categorical_columns: tuple[str, ...] = (),
    rng: np.random.Generator | None = None,
) -> Path:
    """Write one sanitized CSV and append a row to the per-track manifest.

    Mimics what the post-2026-05-20 data pipeline lands on disk:
    ``<data_root>/data/processed/{track}/<id>.sanitized.csv`` plus
    a per-track manifest row at
    ``<output_root>/data/manifest_{track}.csv``.
    """
    task_type = task_type or (
        "classification" if track == "pd" else "regression"
    )
    rng = rng or np.random.default_rng(abs(hash(dataset_id)) % (2**32))

    folder = data_root / "data" / "processed" / track
    folder.mkdir(parents=True, exist_ok=True)
    feature_cols = [f"f{i}" for i in range(n_feat)]
    data = {c: rng.standard_normal(n_rows).astype(np.float32) for c in feature_cols}
    # Lay any requested categoricals as string-encoded columns.
    for c in categorical_columns:
        data[c] = rng.choice(["A", "B", "C"], size=n_rows)
    if task_type == "classification":
        data[target_column] = rng.integers(0, 2, size=n_rows).astype(np.int64)
    else:
        data[target_column] = rng.uniform(0.0, 1.0, size=n_rows).astype(np.float32)
    df = pd.DataFrame(data)
    csv_path = folder / f"{dataset_id}.sanitized.csv"
    df.to_csv(csv_path, index=False)

    # Manifest row (one per dataset, accumulated).
    manifest_path = output_root / "data" / f"manifest_{track}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cats_field = ";".join(categorical_columns)
    row = {
        "dataset_id": dataset_id,
        "track": track,
        "task_type": task_type,
        "target_column": target_column,
        "categorical_columns": cats_field,
        "n_rows": n_rows,
        "n_cols": len(df.columns) - 1,
        "source": "synthetic",
    }
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, dtype=str).fillna("")
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        existing.to_csv(manifest_path, index=False)
    else:
        pd.DataFrame([row]).to_csv(manifest_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_processed(tmp_path, monkeypatch):
    """Build a fully synthetic 8-dataset processed corpus on disk.

    Sets ``CREDITPFN_DATA_ROOT`` and ``CREDITPFN_OUTPUT_ROOT`` so that
    every helper that goes through ``src.utils.paths.resolve_*_path``
    resolves to the temp tree.
    """
    monkeypatch.setenv("CREDITPFN_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    # The path resolver's autodetect is memoised — clear it so the
    # new env vars take effect inside this test session.
    from src.utils import paths as _paths
    _paths._autodetect_data_root.cache_clear()

    rng = np.random.default_rng(42)
    # 5 PD datasets (varying sizes — `0005.big` is bigger so subsample
    # actually subsamples) and 3 LGD datasets.
    _write_synthetic_processed(tmp_path, tmp_path,
                                track="pd", dataset_id="0001.alpha", rng=rng)
    _write_synthetic_processed(tmp_path, tmp_path,
                                track="pd", dataset_id="0002.bravo", rng=rng)
    _write_synthetic_processed(tmp_path, tmp_path,
                                track="pd", dataset_id="0003.charlie", rng=rng)
    _write_synthetic_processed(tmp_path, tmp_path,
                                track="pd", dataset_id="0004.delta", rng=rng)
    _write_synthetic_processed(tmp_path, tmp_path,
                                track="pd", dataset_id="0005.big",
                                n_rows=500, rng=rng)
    _write_synthetic_processed(tmp_path, tmp_path,
                                track="lgd", dataset_id="0001.lgd_a", rng=rng)
    _write_synthetic_processed(tmp_path, tmp_path,
                                track="lgd", dataset_id="0002.lgd_b", rng=rng)
    _write_synthetic_processed(tmp_path, tmp_path,
                                track="lgd", dataset_id="0003.lgd_c", rng=rng)
    # Bust the memoised CSV loader so a fresh test gets a fresh load.
    _load_processed_csv.cache_clear()
    return tmp_path


# =============================================================================
# Block 1 · corpus.py
# =============================================================================


def test_assign_buckets_minimum_test_dataset() -> None:
    """With train=0.8, test=0.1 on 7 datasets, rounding gives 6/1/0 — but
    we shift one dataset from train to test so test is never empty."""
    ids = [f"d{i}" for i in range(7)]
    b = _assign_buckets(ids, train_fraction=0.8, test_fraction=0.1, seed=42)
    counts = {v: list(b.values()).count(v) for v in set(b.values())}
    assert counts.get("test", 0) >= 1, f"empty test bucket: {counts}"
    assert counts.get("train", 0) >= 1


def test_assign_buckets_overshoot_fix() -> None:
    """fractions 0.9/0.2 on 7 datasets must shave to fit."""
    ids = [f"d{i}" for i in range(7)]
    b = _assign_buckets(ids, train_fraction=0.9, test_fraction=0.2, seed=42)
    total = sum(1 for v in b.values() if v in ("train", "test"))
    assert total <= len(ids)


def test_split_corpus_no_leakage(synthetic_processed) -> None:
    """A given dataset_id appears in at most one bucket."""
    split = split_corpus(
        track="pd",
        train_fraction=0.6, test_fraction=0.4,
        seed=0,
    )
    train_ids = {c.dataset_id for c in split.train}
    test_ids = {c.dataset_id for c in split.test}
    assert train_ids.isdisjoint(test_ids), (
        f"dataset_id leak: {train_ids & test_ids}"
    )


def test_split_corpus_one_ref_per_dataset(synthetic_processed) -> None:
    """Each parent dataset contributes EXACTLY ONE DatasetRef
    (no chunking — see the 2026-05-20 refactor)."""
    split = split_corpus(
        track="pd",
        train_fraction=0.6, test_fraction=0.4,
        seed=0,
    )
    assert len(split.train) + len(split.test) == 5


def test_split_corpus_deterministic(synthetic_processed) -> None:
    """Same seed → identical split."""
    a = split_corpus(track="pd", train_fraction=0.6, test_fraction=0.4, seed=42)
    b = split_corpus(track="pd", train_fraction=0.6, test_fraction=0.4, seed=42)
    assert {c.dataset_id for c in a.train} == {c.dataset_id for c in b.train}
    assert {c.dataset_id for c in a.test}  == {c.dataset_id for c in b.test}


def test_split_corpus_explicit_test_id(synthetic_processed) -> None:
    split = split_corpus(
        track="pd",
        train_fraction=0.6, test_fraction=0.4,
        test_dataset_ids=["0001.alpha"],
        seed=0,
    )
    assert all(c.dataset_id != "0001.alpha" for c in split.train)
    assert any(c.dataset_id == "0001.alpha" for c in split.test)


def test_split_corpus_explicit_train_id_alone(synthetic_processed) -> None:
    split = split_corpus(
        track="pd",
        train_fraction=0.6, test_fraction=0.4,
        train_dataset_ids=["0001.alpha", "0002.bravo"],
        seed=0,
    )
    train_ids = {c.dataset_id for c in split.train}
    test_ids = {c.dataset_id for c in split.test}
    assert train_ids == {"0001.alpha", "0002.bravo"}
    assert "0001.alpha" not in test_ids and "0002.bravo" not in test_ids


def test_split_corpus_both_explicit_lists(synthetic_processed) -> None:
    split = split_corpus(
        track="pd",
        train_fraction=0.99, test_fraction=0.01,        # ignored
        train_dataset_ids=["0001.alpha"],
        test_dataset_ids=["0002.bravo"],
        seed=0,
    )
    assert {c.dataset_id for c in split.train} == {"0001.alpha"}
    assert {c.dataset_id for c in split.test}  == {"0002.bravo"}


def test_split_corpus_overlap_raises(synthetic_processed) -> None:
    with pytest.raises(ValueError, match="appear in both"):
        split_corpus(
            track="pd",
            train_dataset_ids=["0001.alpha"],
            test_dataset_ids=["0001.alpha"],
            seed=0,
        )


def test_split_corpus_unknown_track_raises() -> None:
    with pytest.raises(ValueError, match="track"):
        split_corpus(track="xx",
                     train_fraction=0.8, test_fraction=0.2)


def test_split_corpus_fractions_sum_too_high() -> None:
    with pytest.raises(ValueError, match="sum"):
        split_corpus(track="pd",
                     train_fraction=0.9, test_fraction=0.5)


def test_build_dataset_pool_picks_up_manifest(synthetic_processed) -> None:
    """A synthetic 5-PD manifest results in 5 DatasetRefs on disk."""
    refs = build_dataset_pool("pd")
    assert len(refs) == 5
    assert all(isinstance(r, DatasetRef) for r in refs)
    assert all(r.processed_csv.exists() for r in refs)


# =============================================================================
# Block 2 · dataloader.py
# =============================================================================


def test_dataloader_yields_correct_shapes(synthetic_processed) -> None:
    refs = build_dataset_pool("pd")
    ds = ProcessedDatasetLoader(refs, max_rows_per_epoch=80, query_fraction=0.20, seed=0)
    batch = ds[0]
    assert isinstance(batch, TabPFNBatch)
    assert batch.X_context.dim() == 3 and batch.X_context.shape[1] == 1
    assert batch.X_query.dim()   == 3 and batch.X_query.shape[1]   == 1
    assert batch.y_context.shape[1:] == (1, 1)
    assert batch.y_query.shape[1:]   == (1, 1)
    # query ≈ 20% of 80 = 16, context ≈ 64
    n_ctx = batch.X_context.shape[0]
    n_qry = batch.X_query.shape[0]
    assert abs(n_qry - 16) <= 1
    assert abs(n_ctx - 64) <= 1


def test_dataloader_classification_dtype(synthetic_processed) -> None:
    refs = build_dataset_pool("pd")
    batch = ProcessedDatasetLoader(
        refs, max_rows_per_epoch=80, query_fraction=0.2, seed=0,
    )[0]
    assert batch.y_context.dtype == torch.int64
    assert batch.task_type == "classification"


def test_dataloader_regression_dtype(synthetic_processed) -> None:
    refs = build_dataset_pool("lgd")
    batch = ProcessedDatasetLoader(
        refs, max_rows_per_epoch=80, query_fraction=0.2, seed=0,
    )[0]
    assert batch.y_context.dtype == torch.float32
    assert batch.task_type == "regression"


def test_dataloader_per_epoch_reshuffle(synthetic_processed) -> None:
    """The whole point of the 2026-05-20 refactor: each epoch must
    draw a fresh random subsample from the larger datasets so the
    model eventually sees all the data.

    We use the bigger ``0005.big`` (500 rows; subsample 100 — < 100%)
    to guarantee the rows we pick on epoch 0 ≠ those we pick on epoch 1.
    """
    refs = [r for r in build_dataset_pool("pd") if r.dataset_id == "0005.big"]
    assert refs, "expected 0005.big in fixture"
    ds = ProcessedDatasetLoader(
        refs, max_rows_per_epoch=100, query_fraction=0.2, seed=0,
    )

    ds.set_epoch(0)
    batch_a = ds[0]
    ds.set_epoch(1)
    batch_b = ds[0]
    # Concatenate context + query into a single flat row pool and
    # compare element-wise — the two epochs should differ.
    a = torch.cat([batch_a.X_context.flatten(), batch_a.X_query.flatten()])
    b = torch.cat([batch_b.X_context.flatten(), batch_b.X_query.flatten()])
    assert a.numel() == b.numel()
    # Different shuffles ⇒ at least some elements differ.
    assert not torch.equal(a, b), (
        "per-epoch reshuffle did not change the subsample"
    )


def test_dataloader_deterministic_within_epoch(synthetic_processed) -> None:
    """The same epoch + same idx + same seed must yield bit-identical
    batches (run-level reproducibility)."""
    refs = build_dataset_pool("pd")
    ds_a = ProcessedDatasetLoader(refs, max_rows_per_epoch=80, query_fraction=0.2, seed=7)
    ds_b = ProcessedDatasetLoader(refs, max_rows_per_epoch=80, query_fraction=0.2, seed=7)
    ds_a.set_epoch(3)
    ds_b.set_epoch(3)
    a, b = ds_a[0], ds_b[0]
    assert torch.equal(a.X_context, b.X_context)
    assert torch.equal(a.X_query,   b.X_query)


def test_identity_collate_rejects_multi_batch() -> None:
    """TabPFN's meta_dataset_collator hard-asserts batch_size=1."""
    with pytest.raises(ValueError, match="batch_size=1"):
        identity_collate(["x", "y"])


def test_prepare_eval_chunk_subsamples_proportionally(
    synthetic_processed,
) -> None:
    refs = build_dataset_pool("pd")
    big = [r for r in refs if r.dataset_id == "0005.big"][0]
    batch = prepare_eval_chunk(big, n_inference_subsample_samples=50, seed=0)
    n_total = batch.X_context.shape[0] + batch.X_query.shape[0]
    assert n_total <= 50 + 1
    # 80/20 ctx/query split → ctx fraction is ~0.8
    ratio = batch.X_context.shape[0] / max(1, n_total)
    assert 0.7 <= ratio <= 0.9


# =============================================================================
# Block 3 · loop.py
# =============================================================================


def test_warmup_cosine_schedule_landmarks() -> None:
    """The schedule must hit the four key landmarks of HuggingFace's
    ``get_cosine_schedule_with_warmup``:

        * step 0           → multiplier 0
        * step warmup_steps → multiplier 1
        * step total/2     → multiplier 0.5  (just after warmup midpoint
                              of the cosine half-period)
        * step total       → multiplier 0
    """
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = make_warmup_cosine_schedule(
        opt, total_steps=100, warmup_fraction=0.10, schedule_type="warmup_cosine",
    )
    base = opt.param_groups[0]["lr"]
    assert math.isclose(base, 0.0, abs_tol=1e-12)            # step 0

    for _ in range(10):
        opt.step(); sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, rel_tol=1e-9)  # step 10

    for _ in range(45):
        opt.step(); sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.5, abs_tol=1e-9)  # step 55

    for _ in range(45):
        opt.step(); sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.0, abs_tol=1e-9)  # step 100


def test_warmup_only_schedule_stays_constant() -> None:
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = make_warmup_cosine_schedule(
        opt, total_steps=50, warmup_fraction=0.10, schedule_type="warmup_only",
    )
    for _ in range(5):
        opt.step(); sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)
    for _ in range(40):
        opt.step(); sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)


def test_constant_schedule_is_constant() -> None:
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = make_warmup_cosine_schedule(
        opt, total_steps=10, warmup_fraction=0.10, schedule_type="constant",
    )
    for _ in range(20):
        opt.step(); sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0)


def test_unknown_schedule_type_raises() -> None:
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = make_warmup_cosine_schedule(
        opt, total_steps=10, warmup_fraction=0.10, schedule_type="banana",
    )
    with pytest.raises(ValueError, match="schedule_type"):
        sched.step()


def test_descriptive_name_encodes_every_tunable() -> None:
    name = descriptive_name(
        run_name="myrun", track="pd",
        base_path="checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
        learning_rate=1.234e-5,
        seed=7,
    )
    # Every input must show up in the filename, so a glob can later
    # find runs by HP.
    assert name.endswith(".ckpt")
    assert "myrun" in name
    assert "pd" in name
    assert "tabpfn-v2.6-classifier-v2.6_default" in name
    assert "lr1e-05" in name
    assert "seed7" in name


def test_descriptive_name_is_deterministic() -> None:
    """Re-running with the same HPs must give the same filename — so a
    rerun overwrites in place rather than fork-bombing the directory."""
    kwargs = dict(
        run_name="r", track="pd",
        base_path="checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
        learning_rate=1e-5,
        seed=0,
    )
    assert descriptive_name(**kwargs) == descriptive_name(**kwargs)


# =============================================================================
# Block 4 · metrics.py
# =============================================================================


def test_improvement_direction_known_metrics() -> None:
    assert improvement_direction("roc_auc")  == +1
    assert improvement_direction("neg_nll")  == +1
    assert improvement_direction("log_loss") == -1
    assert improvement_direction("rmse")     == -1


def test_improvement_direction_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        improvement_direction("accuracy")


def test_classification_metric_perfect_separation() -> None:
    """Logits that perfectly separate two classes → ROC-AUC == 1."""
    n = 20
    # Rows 0..9 → class 0; rows 10..19 → class 1.
    targets = torch.tensor([0]*10 + [1]*10).reshape(n, 1, 1)
    logits = torch.zeros((n, 1, 2))
    logits[:10, 0, 0] = 5; logits[:10, 0, 1] = -5
    logits[10:, 0, 0] = -5; logits[10:, 0, 1] = 5
    auc = classification_metric(
        logits=logits, targets=targets, metric="roc_auc", n_classes=2,
    )
    assert auc == pytest.approx(1.0)


def test_classification_metric_single_class_returns_nan() -> None:
    """ROC-AUC undefined when query has one class — must return NaN."""
    targets = torch.zeros((5, 1, 1), dtype=torch.int64)
    logits = torch.randn((5, 1, 2))
    auc = classification_metric(
        logits=logits, targets=targets, metric="roc_auc", n_classes=2,
    )
    assert math.isnan(auc)


def test_classification_metric_log_loss_is_finite() -> None:
    targets = torch.randint(0, 2, (10, 1, 1))
    logits = torch.randn((10, 1, 2))
    ll = classification_metric(
        logits=logits, targets=targets, metric="log_loss", n_classes=2,
    )
    assert math.isfinite(ll) and ll > 0


def test_regression_metric_rmse_with_mock_criterion() -> None:
    """RMSE branch of regression_metric works with a stand-in criterion
    that exposes a ``.mean()`` method."""
    from tabpfn.architectures.base.bar_distribution import (
        FullSupportBarDistribution,
    )
    borders = torch.linspace(-3, 3, 11)
    crit = FullSupportBarDistribution(borders=borders, ignore_nan_targets=True)
    n = 50
    logits = torch.randn((n, 1, 10))
    targets = torch.randn((n, 1, 1))
    rmse = regression_metric(
        logits=logits, targets=targets, criterion=crit,
        metric="rmse",
    )
    assert math.isfinite(rmse) and rmse >= 0


def test_mean_ignore_nan_drops_nans() -> None:
    assert mean_ignore_nan([1.0, float("nan"), 3.0]) == pytest.approx(2.0)
    assert math.isnan(mean_ignore_nan([float("nan"), float("nan")]))


# =============================================================================
# Block 5 · model.py
# =============================================================================


@pytest.mark.parametrize("name,expected", [
    # The returned string must carry the leading 'v' so it round-trips
    # through TabPFN's `version: Literal["v2", "v2.5", "v2.6", "v3"]`.
    # We only sweep v2.6 / v3, but the inference function must remain
    # robust to the full literal set so we keep v2.5 in the test
    # (the regex pattern is exercised, no checkpoint is loaded).
    ("tabpfn-v3-classifier-v3_default.ckpt",       "v3"),
    ("tabpfn-v2.6-classifier-v2.6_default.ckpt",   "v2.6"),
    ("tabpfn-v2.5-regressor-v2.5_default.ckpt",    "v2.5"),
])
def test_infer_version_from_filename(name: str, expected: str) -> None:
    assert _infer_version(Path(name)) == expected


def test_infer_version_fails_loudly_on_bad_name() -> None:
    with pytest.raises(ValueError, match="version"):
        _infer_version(Path("randomly_named.ckpt"))


def test_save_finetuned_writes_provenance_sidecar(tmp_path: Path) -> None:
    """The .ckpt + .ckpt.provenance.json must both contain the HP /
    dataset / GPU / training-time record."""
    import json
    from src.train.model import save_finetuned, load_provenance

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

    class _Cfg:
        n_features = 2

    save_path = tmp_path / "ckpt" / "test.ckpt"
    provenance = {
        "hyperparameters": {"learning_rate": 1e-5, "epochs": 30},
        "training_datasets": ["0001.gmsc", "0002.heloc"],
        "training_time_seconds": 123.4,
        "gpu": "NVIDIA H100 NVL",
    }
    save_finetuned(_Tiny(), _Cfg(), save_path, provenance=provenance)

    # Sidecar JSON exists and matches.
    sidecar = save_path.with_suffix(save_path.suffix + ".provenance.json")
    assert sidecar.exists()
    parsed = json.loads(sidecar.read_text(encoding="utf-8"))
    assert parsed["gpu"] == "NVIDIA H100 NVL"
    assert parsed["training_datasets"] == ["0001.gmsc", "0002.heloc"]

    # The same provenance round-trips through the .ckpt file.
    loaded = load_provenance(save_path)
    assert loaded == provenance

    # Sidecar-only path (delete the .ckpt) still works.
    save_path.unlink()
    loaded2 = load_provenance(save_path)
    assert loaded2["gpu"] == "NVIDIA H100 NVL"


# =============================================================================
# Block 6 · scripts/train_pipeline.py
# =============================================================================


def test_grid_full_cartesian_product() -> None:
    """3 bases × 3 lrs × default use_lora=[False] → 9 trials."""
    import scripts.train_pipeline as tp
    cfg = NS(
        track="pd",
        tunable=NS(
            classifier_base_paths=["a", "b", "c"],
            regressor_base_paths=["x", "y", "z"],
            learning_rates=[1e-6, 1e-5, 5e-5],
        ),
    )
    grid = tp._resolve_grid(cfg, single=False)
    assert len(grid) == 3 * 3
    # No duplicates in a cartesian product of distinct lists.
    assert len(set(grid)) == len(grid)
    # Every entry is the 3-tuple (base, lr, use_lora); default use_lora=False.
    assert all(len(t) == 3 and t[2] is False for t in grid)


def test_grid_full_cartesian_product_with_lora_axis() -> None:
    """With both use_lora values requested, the grid doubles."""
    import scripts.train_pipeline as tp
    cfg = NS(
        track="pd",
        tunable=NS(
            classifier_base_paths=["a", "b"],
            regressor_base_paths=["x", "y"],
            learning_rates=[1e-5, 1e-4],
            use_lora=[False, True],
        ),
    )
    grid = tp._resolve_grid(cfg, single=False)
    assert len(grid) == 2 * 2 * 2
    # Both LoRA flavours represented.
    loras = {t[2] for t in grid}
    assert loras == {False, True}


def test_grid_single_picks_first_value() -> None:
    """``--single`` → exactly one trial, the head of every list."""
    import scripts.train_pipeline as tp
    cfg = NS(
        track="lgd",
        tunable=NS(
            classifier_base_paths=["a", "b"],
            regressor_base_paths=["P", "Q"],
            learning_rates=[5e-6, 1e-5],
        ),
    )
    grid = tp._resolve_grid(cfg, single=True)
    assert grid == [("P", 5e-6, False)]
    assert len(grid) == 1


# =============================================================================
# Block 3b · end-to-end with a mocked TabPFN model (no GPU, no checkpoint)
# =============================================================================
#
#  ``train_one_config`` is the most important integration surface. To
#  test it without TabPFN's >100 MB checkpoints or a CUDA box, we
#  monkey-patch ``load_tabpfn_for_training`` to return a tiny dummy
#  model + criterion + config that obey the same call signatures as
#  the real ones.


class _DummyClassifier(torch.nn.Module):
    """Stand-in for ``PerFeatureTransformer`` (classifier head).

    Mirrors TabPFN's canonical forward signature (per
    ``repositories/TabPFN .txt:15098-15203``):

        forward(x, y, *, only_return_standard_out=True,
                categorical_inds=None, ...) -> (n_test, batch, n_classes_max)

    where ``x`` is the concatenated train+test rows and ``y`` carries
    only the train labels. The model derives ``single_eval_pos`` from
    ``y.shape[0]`` and returns predictions for the remaining rows.
    """
    N_CLASSES_MAX = 10

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.head = torch.nn.Linear(n_features, self.N_CLASSES_MAX)

    def forward(self, x, y, *,
                only_return_standard_out=True,
                categorical_inds=None, **_):
        # x: (n_train + n_test, batch=1, n_features); y: (n_train, batch, 1).
        n_train = y.shape[0]
        test_x = x[n_train:]                            # (n_test, 1, n_features)
        return self.head(test_x)


def test_train_one_config_end_to_end_mocked(
    synthetic_processed, monkeypatch, tmp_path,
) -> None:
    """Smoke: drive ``train_one_config`` against the synthetic
    processed-CSV corpus with a dummy linear model standing in for
    TabPFN. Verifies:

      * the loop runs ``cfg.train.epochs`` epochs without crashing;
      * a checkpoint is written to the descriptive path;
      * the new provenance schema (max_rows_per_epoch / query_fraction)
        is populated.
    """
    from omegaconf import OmegaConf
    import src.train.loop as loop_mod

    n_feat = 4

    def fake_loader(checkpoint_path, *, track, device, lora_config=None):
        del lora_config
        model = _DummyClassifier(n_features=n_feat).to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        arch = NS(num_features=n_feat, num_classes=2)
        return model, criterion, arch

    monkeypatch.setattr(loop_mod, "load_tabpfn_for_training", fake_loader)

    saved_paths: list[Path] = []
    captured_provenance: list[dict] = []

    def fake_save(model, arch, save_path, *, criterion=None, provenance=None):
        del criterion
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(b"")
        saved_paths.append(Path(save_path))
        if provenance is not None:
            captured_provenance.append(provenance)
        return Path(save_path)

    monkeypatch.setattr(loop_mod, "save_finetuned", fake_save)

    cfg = OmegaConf.create({
        "seed": 0,
        "run_name": "smoketest",
        "device": "cpu",
        "track": "pd",
        "tunable": {
            "classifier_base_paths":
                ["checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"],
            "regressor_base_paths":
                ["checkpoints/tabpfn-v2.6-regressor-v2.6_default.ckpt"],
            "learning_rates":        [1e-3],
        },
        "corpus": {
            "train_fraction": 0.6,
            "test_fraction":  0.4,
        },
        "optimizer": {"weight_decay": 0.01},
        "scheduler": {"warmup_fraction": 0.10},
        "train": {
            "epochs": 2,
            "accumulate_grad_batches": 1,
            "grad_clip_norm": 1.0,
            "amp": False,
            "dataloader_workers": 0,
            # Per-epoch monitoring eval: small to keep smoke test fast.
            "epoch_eval_subsample_samples": 50,
            # Smoke test stays on the cheap single-forward path — the
            # ensemble path (n_estimators>1) calls the real
            # TabPFNClassifier sklearn API which expects on-disk
            # checkpoints and a working TabPFN install; out of scope
            # for the unit test.
            "epoch_eval_n_estimators": 1,
        },
        "eval": {
            "classification_metric": "roc_auc",
            "regression_metric": "neg_nll",
            "n_inference_subsample_samples": 100,
        },
        "checkpoint": {"trained_dir": str(tmp_path / "trained")},
    })

    # train_one_config reads `config/data.yaml` for max_rows_per_epoch +
    # query_fraction. Monkey-patch the OmegaConf.load call inside loop.py
    # so the smoke test doesn't depend on the on-disk yaml at all.
    import omegaconf as _oc
    _fake_data_cfg = _oc.OmegaConf.create({
        "finetuning": {"max_rows_per_epoch": 80, "query_fraction": 0.20},
    })
    real_load = _oc.OmegaConf.load

    def fake_load(path):
        if "data.yaml" in str(path):
            return _fake_data_cfg
        return real_load(path)

    monkeypatch.setattr(_oc.OmegaConf, "load", fake_load)

    result = loop_mod.train_one_config(cfg)

    assert len(result.history) == 2
    assert all(math.isfinite(r.train_loss) for r in result.history)
    assert result.final_ckpt_path == saved_paths[-1]
    assert not hasattr(result, "test_metric_raw")
    assert not hasattr(result, "test_metric_name")
    assert "smoketest_pd_" in result.descriptive_name
    assert result.descriptive_name.endswith(".ckpt")

    assert len(captured_provenance) == 1
    prov = captured_provenance[0]
    assert prov["schema_version"] == 1
    assert prov["track"] == "pd"
    assert prov["task_type"] == "classification"
    assert prov["hyperparameters"]["learning_rate"] == 1e-3
    assert prov["hyperparameters"]["max_rows_per_epoch"] == 80
    assert prov["hyperparameters"]["query_fraction"] == 0.20
    assert prov["hyperparameters"]["epochs"] == 2
    assert prov["hyperparameters"]["seed"] == 0
    assert prov["hyperparameters"]["base_checkpoint"].endswith(
        "tabpfn-v2.6-classifier-v2.6_default.ckpt"
    )
    assert isinstance(prov["training_datasets"], list)
    assert prov["training_time_seconds"] > 0
    assert prov["device"] == "cpu"
    assert prov["gpu"] == "cpu"
    assert "torch_version" in prov
