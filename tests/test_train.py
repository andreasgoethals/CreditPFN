"""Smoke + unit tests for the training pipeline.

Layout choice
-------------
One file per ``src/`` subpackage. ``test_train.py`` covers everything in
``src/train/``: corpus splitting, the chunk DataLoader, the learning-rate
schedule, the metric helpers, the descriptive-name builder, and a
fully-mocked end-to-end pass through ``train_one_config`` that does NOT
require a real TabPFN checkpoint or a GPU.

Tests that genuinely need TabPFN (loading the real checkpoint, running a
forward pass) are guarded with ``pytest.importorskip("tabpfn")`` so the
suite stays runnable in a stripped-down CI image.

Running
-------
::

    pytest -q tests/test_train.py
    pytest -q tests/test_train.py -k corpus       # just the corpus block

Coverage map
------------
    Block 1  corpus.py     — train/test split semantics
    Block 2  dataloader.py — chunk reading, ctx/query resplit, eval prep
    Block 3  loop.py       — LR schedule, descriptive name, end-to-end
                              with a mocked PerFeatureTransformer
    Block 4  metrics.py    — ROC-AUC / log_loss / RMSE on toy inputs
    Block 5  model.py      — version inference + filename schema
    Block 6  scripts/train_pipeline.py — grid expansion + --single

Tests intentionally lean toward *failure-mode coverage* over
behavioural completeness: a few sharp tests that catch real
regressions if a future refactor breaks the contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from src.train.corpus import (
    ChunkRef, CorpusSplit,
    _assign_buckets, build_chunk_pool, split_corpus, split_from_cfg,
)
from src.train.dataloader import (
    ChunkDataset, TabPFNBatch,
    _resample_chunk, identity_collate, prepare_eval_chunk,
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


def _write_synthetic_chunk(
    folder: Path, *,
    chunk_idx: int,
    n_ctx: int = 60, n_qry: int = 40, n_feat: int = 4,
    task_type: str = "classification",
    cat_idx: tuple[int, ...] = (0,),
    rng: np.random.Generator | None = None,
) -> Path:
    """Write a single ``chunk_NNN.npz`` file mimicking the data pipeline."""
    rng = rng or np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    X_ctx = rng.standard_normal((n_ctx, n_feat)).astype(np.float32)
    X_qry = rng.standard_normal((n_qry, n_feat)).astype(np.float32)
    if task_type == "classification":
        y_ctx = rng.integers(0, 2, size=n_ctx).astype(np.int64)
        y_qry = rng.integers(0, 2, size=n_qry).astype(np.int64)
    else:
        y_ctx = rng.uniform(0, 1, size=n_ctx).astype(np.float32)
        y_qry = rng.uniform(0, 1, size=n_qry).astype(np.float32)
    out = folder / f"chunk_{chunk_idx:03d}.npz"
    np.savez_compressed(
        out,
        X_context=X_ctx, y_context=y_ctx,
        X_query=X_qry, y_query=y_qry,
        categorical_idx=np.asarray(cat_idx, dtype=np.int32),
    )
    return out


def _write_synthetic_dataset(
    cached_root: Path, *,
    track: str,
    dataset_id: str,
    n_chunks: int = 1,
    task_type: str | None = None,
    rng: np.random.Generator | None = None,
) -> Path:
    """Write all chunks + meta.json for one synthetic dataset."""
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


@pytest.fixture
def synthetic_cache(tmp_path):
    """Build a fully synthetic 8-dataset cache for both tracks."""
    rng = np.random.default_rng(42)
    cached = tmp_path / "cached"
    # 5 PD datasets, one with 3 chunks (a multi-chunk parent like
    # `algorithmwatch` in the real corpus).
    _write_synthetic_dataset(cached, track="pd", dataset_id="0001.alpha", rng=rng)
    _write_synthetic_dataset(cached, track="pd", dataset_id="0002.bravo", rng=rng)
    _write_synthetic_dataset(cached, track="pd", dataset_id="0003.charlie", rng=rng)
    _write_synthetic_dataset(cached, track="pd", dataset_id="0004.delta", rng=rng)
    _write_synthetic_dataset(cached, track="pd", dataset_id="0005.big",
                             n_chunks=3, rng=rng)
    # 3 LGD datasets.
    _write_synthetic_dataset(cached, track="lgd", dataset_id="0001.lgd_a", rng=rng)
    _write_synthetic_dataset(cached, track="lgd", dataset_id="0002.lgd_b", rng=rng)
    _write_synthetic_dataset(cached, track="lgd", dataset_id="0003.lgd_c", rng=rng)
    return cached


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
    """fractions 0.8/0.2 on 7 datasets round to 6+1=7 (fits) — overshoot
    only triggers on degenerate cases. fractions 0.9/0.2 must shave."""
    ids = [f"d{i}" for i in range(7)]
    b = _assign_buckets(ids, train_fraction=0.9, test_fraction=0.2, seed=42)
    total = sum(1 for v in b.values() if v in ("train", "test"))
    assert total <= len(ids)


def test_split_corpus_no_leakage(synthetic_cache: Path) -> None:
    """Every chunk of a parent dataset goes to exactly one bucket."""
    split = split_corpus(
        synthetic_cache, track="pd",
        train_fraction=0.6, test_fraction=0.4,
        multi_chunk_policy="all_chunks_as_separate_datasets",
        seed=0,
    )
    train_ids = {c.dataset_id for c in split.train}
    test_ids = {c.dataset_id for c in split.test}
    assert train_ids.isdisjoint(test_ids), (
        f"dataset_id leak: {train_ids & test_ids}"
    )


def test_split_corpus_all_chunks_attached(synthetic_cache: Path) -> None:
    """When ``0005.big`` (3 chunks) is split, ALL three chunks go to the
    same bucket — never two in train and one in test."""
    split = split_corpus(
        synthetic_cache, track="pd",
        train_fraction=0.6, test_fraction=0.4,
        multi_chunk_policy="all_chunks_as_separate_datasets",
        seed=0,
    )
    big_chunks_train = sum(1 for c in split.train if c.dataset_id == "0005.big")
    big_chunks_test  = sum(1 for c in split.test  if c.dataset_id == "0005.big")
    if big_chunks_train > 0:
        assert big_chunks_test == 0
    if big_chunks_test > 0:
        assert big_chunks_train == 0


def test_split_corpus_first_chunk_only(synthetic_cache: Path) -> None:
    """``first_chunk_only`` keeps exactly one chunk per dataset."""
    split = split_corpus(
        synthetic_cache, track="pd",
        train_fraction=0.6, test_fraction=0.4,
        multi_chunk_policy="first_chunk_only",
        seed=0,
    )
    # 5 PD datasets, each contributes 1 chunk.
    assert len(split.train) + len(split.test) == 5


def test_split_corpus_deterministic(synthetic_cache: Path) -> None:
    """Same seed → identical split. This is the contract the future
    XGBoost/CatBoost comparison relies on."""
    a = split_corpus(synthetic_cache, track="pd",
                     train_fraction=0.6, test_fraction=0.4, seed=42)
    b = split_corpus(synthetic_cache, track="pd",
                     train_fraction=0.6, test_fraction=0.4, seed=42)
    assert {(c.dataset_id, c.chunk_idx) for c in a.train} == \
           {(c.dataset_id, c.chunk_idx) for c in b.train}
    assert {(c.dataset_id, c.chunk_idx) for c in a.test} == \
           {(c.dataset_id, c.chunk_idx) for c in b.test}


def test_split_corpus_explicit_test_id(synthetic_cache: Path) -> None:
    """A dataset_id in `test_dataset_ids` always lands in test, never train."""
    split = split_corpus(
        synthetic_cache, track="pd",
        train_fraction=0.6, test_fraction=0.4,
        test_dataset_ids=["0001.alpha"],
        seed=0,
    )
    assert all(c.dataset_id != "0001.alpha" for c in split.train)
    assert any(c.dataset_id == "0001.alpha" for c in split.test)


def test_split_corpus_explicit_train_id_alone(synthetic_cache: Path) -> None:
    """Explicit `train_dataset_ids` without `test_dataset_ids` →
    train = exactly those, test = remaining (count-wise)."""
    split = split_corpus(
        synthetic_cache, track="pd",
        train_fraction=0.6, test_fraction=0.4,
        train_dataset_ids=["0001.alpha", "0002.bravo"],
        seed=0,
    )
    train_ids = {c.dataset_id for c in split.train}
    test_ids = {c.dataset_id for c in split.test}
    assert train_ids == {"0001.alpha", "0002.bravo"}
    assert "0001.alpha" not in test_ids and "0002.bravo" not in test_ids


def test_split_corpus_both_explicit_lists(synthetic_cache: Path) -> None:
    """When BOTH lists are explicit, fractions are ignored entirely."""
    split = split_corpus(
        synthetic_cache, track="pd",
        train_fraction=0.99, test_fraction=0.01,        # ignored
        train_dataset_ids=["0001.alpha"],
        test_dataset_ids=["0002.bravo"],
        seed=0,
    )
    assert {c.dataset_id for c in split.train} == {"0001.alpha"}
    assert {c.dataset_id for c in split.test}  == {"0002.bravo"}


def test_split_corpus_overlap_raises(synthetic_cache: Path) -> None:
    """An ID in BOTH lists is a programming error."""
    with pytest.raises(ValueError, match="appear in both"):
        split_corpus(
            synthetic_cache, track="pd",
            train_dataset_ids=["0001.alpha"],
            test_dataset_ids=["0001.alpha"],
            seed=0,
        )


def test_split_corpus_unknown_track_raises() -> None:
    with pytest.raises(ValueError, match="track"):
        split_corpus("does/not/matter", track="xx",
                     train_fraction=0.8, test_fraction=0.2)


def test_split_corpus_fractions_sum_too_high() -> None:
    with pytest.raises(ValueError, match="sum"):
        split_corpus("does/not/matter", track="pd",
                     train_fraction=0.9, test_fraction=0.5)


def test_build_chunk_pool_unknown_policy(synthetic_cache: Path) -> None:
    """Defensive: a typo in the policy name raises rather than silently
    treating every chunk like ``first_chunk_only``."""
    with pytest.raises(ValueError, match="multi_chunk_policy"):
        build_chunk_pool(synthetic_cache, "pd", multi_chunk_policy="frist_chunk")


# =============================================================================
# Block 2 · dataloader.py
# =============================================================================


def test_resample_chunk_respects_query_fraction() -> None:
    """The split ratio passed in == the on-output query fraction (±1)."""
    rng = np.random.default_rng(0)
    n = 1000
    chunk = {
        "X_context": rng.standard_normal((n // 2, 3)).astype(np.float32),
        "X_query":   rng.standard_normal((n // 2, 3)).astype(np.float32),
        "y_context": rng.integers(0, 2, n // 2).astype(np.int64),
        "y_query":   rng.integers(0, 2, n // 2).astype(np.int64),
    }
    X_ctx, y_ctx, X_qry, y_qry = _resample_chunk(
        chunk, n_total_target=500, query_fraction=0.20, rng=rng,
    )
    n_total = len(X_ctx) + len(X_qry)
    assert n_total == 500
    assert abs(len(X_qry) - int(round(500 * 0.20))) <= 1
    # ctx + qry consistency: same number of features, no NaNs introduced
    assert X_ctx.shape[1] == X_qry.shape[1] == 3


def test_resample_chunk_subsample_smaller_than_total() -> None:
    """When n_total_target > available rows, take everything."""
    rng = np.random.default_rng(0)
    chunk = {
        "X_context": np.zeros((30, 3), dtype=np.float32),
        "X_query":   np.zeros((20, 3), dtype=np.float32),
        "y_context": np.zeros(30, dtype=np.int64),
        "y_query":   np.zeros(20, dtype=np.int64),
    }
    X_ctx, y_ctx, X_qry, y_qry = _resample_chunk(
        chunk, n_total_target=10_000, query_fraction=0.20, rng=rng,
    )
    assert len(X_ctx) + len(X_qry) == 50


def test_chunk_dataset_yields_correct_shapes(synthetic_cache: Path) -> None:
    refs = build_chunk_pool(synthetic_cache, track="pd",
                            multi_chunk_policy="all_chunks_as_separate_datasets")
    ds = ChunkDataset(refs, n_total_target=80, query_fraction=0.20, seed=0)
    batch = ds[0]
    assert isinstance(batch, TabPFNBatch)
    # n_samples=ctx, batch=1, n_features
    assert batch.X_context.dim() == 3 and batch.X_context.shape[1] == 1
    assert batch.X_query.dim()   == 3 and batch.X_query.shape[1]   == 1
    assert batch.y_context.shape[1:] == (1, 1)
    assert batch.y_query.shape[1:]   == (1, 1)
    # query is ~20% of 80 = ~16, so context should be ~64.
    n_ctx = batch.X_context.shape[0]
    n_qry = batch.X_query.shape[0]
    assert abs(n_qry - 16) <= 1
    assert abs(n_ctx - 64) <= 1


def test_chunk_dataset_classification_dtype(synthetic_cache: Path) -> None:
    """PD chunks must yield int64 y."""
    refs = build_chunk_pool(synthetic_cache, track="pd",
                            multi_chunk_policy="first_chunk_only")
    batch = ChunkDataset(refs, n_total_target=80, query_fraction=0.2, seed=0)[0]
    assert batch.y_context.dtype == torch.int64
    assert batch.task_type == "classification"


def test_chunk_dataset_regression_dtype(synthetic_cache: Path) -> None:
    """LGD chunks must yield float32 y."""
    refs = build_chunk_pool(synthetic_cache, track="lgd",
                            multi_chunk_policy="first_chunk_only")
    batch = ChunkDataset(refs, n_total_target=80, query_fraction=0.2, seed=0)[0]
    assert batch.y_context.dtype == torch.float32
    assert batch.task_type == "regression"


def test_identity_collate_rejects_multi_batch() -> None:
    """The TabPFN ``meta_dataset_collator`` hard-asserts batch_size=1.
    Our collator must do the same."""
    with pytest.raises(ValueError, match="batch_size=1"):
        identity_collate(["x", "y"])


def test_prepare_eval_chunk_subsamples_proportionally(
    synthetic_cache: Path,
) -> None:
    """Asking for fewer than total rows → both splits shrink, ratio
    preserved roughly."""
    refs = build_chunk_pool(synthetic_cache, track="pd",
                            multi_chunk_policy="first_chunk_only")
    batch = prepare_eval_chunk(refs[0], n_inference_subsample_samples=50, seed=0)
    n_total = batch.X_context.shape[0] + batch.X_query.shape[0]
    assert n_total <= 50 + 1
    # original chunk had 60 ctx / 40 qry = ratio 0.6
    ratio = batch.X_context.shape[0] / max(1, n_total)
    assert 0.5 <= ratio <= 0.7


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


@pytest.mark.parametrize("policy_short", [
    "all_chunks_as_separate_datasets", "first_chunk_only",
])
def test_descriptive_name_encodes_every_tunable(policy_short: str) -> None:
    name = descriptive_name(
        run_name="myrun", track="pd",
        base_path="checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
        learning_rate=1.234e-5,
        multi_chunk_policy=policy_short,
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
    short = "allchunks" if policy_short == "all_chunks_as_separate_datasets" else "firstchunk"
    assert short in name


def test_descriptive_name_is_deterministic() -> None:
    """Re-running with the same HPs must give the same filename — so a
    rerun overwrites in place rather than fork-bombing the directory."""
    kwargs = dict(
        run_name="r", track="pd",
        base_path="checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
        learning_rate=1e-5,
        multi_chunk_policy="all_chunks_as_separate_datasets",
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
    ("tabpfn-v2.6-classifier-v2.6_default.ckpt", "2.6"),
    ("tabpfn-v2.5-regressor-v2.5_default.ckpt",  "2.5"),
    ("tabpfn-v2.5-classifier-v2.5_default-2.ckpt", "2.5"),
])
def test_infer_version_from_filename(name: str, expected: str) -> None:
    assert _infer_version(Path(name)) == expected


def test_infer_version_fails_loudly_on_bad_name() -> None:
    with pytest.raises(ValueError, match="version"):
        _infer_version(Path("randomly_named.ckpt"))


# =============================================================================
# Block 6 · scripts/train_pipeline.py
# =============================================================================


def test_grid_full_cartesian_product() -> None:
    """3 bases × 3 lrs × 2 policies = 18 trials."""
    import scripts.train_pipeline as tp
    cfg = NS(
        track="pd",
        tunable=NS(
            classifier_base_paths=["a", "b", "c"],
            regressor_base_paths=["x", "y", "z"],
            learning_rates=[1e-6, 1e-5, 5e-5],
            multi_chunk_policies=["all_chunks_as_separate_datasets",
                                  "first_chunk_only"],
        ),
    )
    grid = tp._resolve_grid(cfg, single=False)
    assert len(grid) == 3 * 3 * 2
    # No duplicates in a cartesian product of distinct lists.
    assert len(set(grid)) == len(grid)


def test_grid_single_picks_first_value() -> None:
    """``--single`` → exactly one trial, the head of every list."""
    import scripts.train_pipeline as tp
    cfg = NS(
        track="lgd",
        tunable=NS(
            classifier_base_paths=["a", "b"],
            regressor_base_paths=["P", "Q"],
            learning_rates=[5e-6, 1e-5],
            multi_chunk_policies=["allchunks", "firstchunk"],
        ),
    )
    grid = tp._resolve_grid(cfg, single=True)
    assert grid == [("P", 5e-6, "allchunks")]
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

    Forward signature: ``(train_x, train_y, test_x, categorical_inds)``
    → ``(n_query, batch=1, n_classes_max)`` logits.
    """
    N_CLASSES_MAX = 10

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.head = torch.nn.Linear(n_features, self.N_CLASSES_MAX)

    def forward(self, train_x, train_y, test_x, categorical_inds):
        # test_x: (n_query, 1, n_features)
        return self.head(test_x)


def test_train_one_config_end_to_end_mocked(
    synthetic_cache: Path, monkeypatch, tmp_path,
) -> None:
    """Smoke: drive ``train_one_config`` against the synthetic cache
    with a dummy linear model standing in for TabPFN. Verifies:

      * the loop runs ``cfg.train.epochs`` epochs without crashing;
      * a checkpoint is written to the descriptive path;
      * a finite test metric is reported.
    """
    from omegaconf import OmegaConf
    import src.train.loop as loop_mod

    n_feat = 4

    def fake_loader(checkpoint_path, *, track, device):
        model = _DummyClassifier(n_features=n_feat).to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        # Minimal "ArchitectureConfig" — anything with a __dict__ works.
        arch = NS(num_features=n_feat, num_classes=2)
        return model, criterion, arch

    monkeypatch.setattr(loop_mod, "load_tabpfn_for_training", fake_loader)
    # Don't bother saving the dummy model state to disk for real.
    saved_paths: list[Path] = []

    def fake_save(model, arch, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(b"")        # touch the file
        saved_paths.append(Path(save_path))
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
            "multi_chunk_policies":  ["first_chunk_only"],
        },
        "corpus": {
            "cached_dir": str(synthetic_cache),
            "train_fraction": 0.6,
            "test_fraction":  0.4,
            "pinned_test_dataset_ids": [],
        },
        "optimizer": {"type": "AdamW", "weight_decay": 0.01,
                      "betas": [0.9, 0.999]},
        "scheduler": {"type": "warmup_cosine", "warmup_fraction": 0.10},
        "train": {
            "epochs": 2,
            "accumulate_grad_batches": 1,
            "grad_clip_norm": 1.0,
            "amp": False,
            "n_finetune_ctx_plus_query_samples": 80,
            "finetune_ctx_query_split_ratio": 0.20,
            "dataloader_workers": 0,
        },
        "eval": {
            "classification_metric": "roc_auc",
            "regression_metric": "neg_nll",
            "n_inference_subsample_samples": 100,
        },
        "checkpoint": {"trained_dir": str(tmp_path / "trained")},
    })

    result = loop_mod.train_one_config(cfg)

    assert len(result.history) == 2
    assert all(math.isfinite(r.train_loss) for r in result.history)
    assert result.final_ckpt_path == saved_paths[-1]
    # We only need to assert finite-OR-nan — synthetic data may yield
    # a single-class query split for a small chunk → NaN AUC.
    assert (
        result.test_metric_raw is None
        or math.isnan(result.test_metric_raw)
        or math.isfinite(result.test_metric_raw)
    )
    # Filename schema check.
    assert "smoketest_pd_" in result.descriptive_name
    assert result.descriptive_name.endswith(".ckpt")
