"""Unit tests for the TabICLv2 model family (second continued-pretraining
family, added 2026-08-04).

Coverage map
------------
    Block 1  tabicl_compat.py — family detection, guarded imports
    Block 2  tabicl_model.py  — pinball loss vs. the upstream formula,
                                save schema round-trip, head/track guard
    Block 3  dataloader.py    — TabICLTrainBatch construction on synthetic
                                data (shapes, dtypes, NaN-freeness)
    Block 4  loop.py          — `_iclhead` naming, row-cap key resolution,
                                the missing-context-class guard
    Block 5  registry/eval    — per-family untuned controls, method dirnames,
                                dirname → metadata round-trip

Tests needing the `tabicl` package are guarded with
``pytest.importorskip("tabicl")``; everything else (naming, key lookup,
dirname decoding) runs in a stripped-down image. No test needs a GPU or a
real 27M-parameter checkpoint — the model-level tests build a deliberately
tiny TabICLv2 from a shrunken config.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# =========================================================================== #
# Block 1 — family detection + guarded imports
# =========================================================================== #


@pytest.mark.parametrize(
    "path, expected",
    [
        ("checkpoints/tabicl-classifier-v2-20260212.ckpt", "tabicl"),
        ("checkpoints/tabicl-regressor-v2-20260212.ckpt", "tabicl"),
        ("checkpoints/tabpfn-v3-classifier-v3_default.ckpt", "tabpfn"),
        ("checkpoints/tabpfn-v2.6-regressor-v2.6_default.ckpt", "tabpfn"),
        # Case-insensitive, and only the BASENAME counts (a parent dir
        # called .../tabicl/... must not reclassify a TabPFN checkpoint).
        ("/x/TabICLv2-Classifier-V2.ckpt", "tabicl"),
        ("/data/tabicl/tabpfn-v3-classifier-v3_default.ckpt", "tabpfn"),
    ],
)
def test_model_family_detection(path: str, expected: str) -> None:
    from src.train.tabicl_compat import model_family
    assert model_family(path) == expected


def test_tabicl_smoke_test_imports() -> None:
    """The SLURM prolog check must pass whenever tabicl is installed."""
    pytest.importorskip("tabicl")
    from src.train.tabicl_compat import smoke_test
    smoke_test("pd")
    smoke_test("lgd")


def test_missing_finetune_extra_names_the_extra_not_the_version() -> None:
    """Regression test (2026-08-05): tabicl declares `transformers` only under
    its `finetune` extra, so a plain `pip install tabicl` yields a working
    INFERENCE install that fails only when importing the finetuning
    internals. Our first error message blamed the tabicl *version* and sent a
    real debugging session down the wrong path — it must name the missing
    extra instead."""
    pytest.importorskip("tabicl")
    import sys
    from src.train.tabicl_compat import import_tabicl_finetune_data

    # Simulate "extra not installed": block transformers and evict the
    # already-imported tabicl finetune modules so the import re-runs.
    saved = {k: v for k, v in sys.modules.items()
             if k == "transformers" or k.startswith(("transformers.",
                                                     "tabicl._finetune",
                                                     "tabicl.train"))}
    for k in saved:
        del sys.modules[k]
    sys.modules["transformers"] = None                      # type: ignore[assignment]
    try:
        with pytest.raises(ImportError) as excinfo:
            import_tabicl_finetune_data()
        msg = str(excinfo.value)
        assert "transformers" in msg
        assert "tabicl[finetune]" in msg
        assert "EXTRA, not a version problem" in msg
    finally:
        sys.modules.pop("transformers", None)
        sys.modules.update(saved)

    # And the happy path still works once it is importable again.
    import_tabicl_finetune_data()


def test_finetune_internals_have_expected_signature() -> None:
    """Pin the PRIVATE upstream API we depend on. If a tabicl upgrade
    renames these, this test fails with a clear pointer instead of the
    training loop breaking at step 1 on the cluster."""
    pytest.importorskip("tabicl")
    import inspect
    from src.train.tabicl_compat import import_tabicl_finetune_data
    MetaBatch, build_meta_batch = import_tabicl_finetune_data()
    params = set(inspect.signature(build_meta_batch).parameters)
    assert {
        "X_chunk", "y_chunk", "classification", "n_estimators", "query_size",
        "epoch_seed", "chunk_idx", "norm_methods", "feat_shuffle_method",
        "class_shuffle_method", "outlier_threshold", "preprocessing_seed",
    } <= params
    assert {
        "X", "y_train", "y_query", "train_size",
        "y_scaler_mean", "y_scaler_std",
    } <= set(MetaBatch.__dataclass_fields__)


# =========================================================================== #
# Block 2 — losses, save/load
# =========================================================================== #


def test_pinball_loss_matches_upstream_formula() -> None:
    """Verbatim check against tabicl's ``_pinball_loss``: levels are
    ``linspace(0, 1, Q+2)[1:-1]`` and the reduction is a plain mean."""
    from src.train.tabicl_model import tabicl_pinball_loss
    quantiles = torch.tensor([[[0.1, 0.5, 0.9], [0.2, 0.4, 0.6]]])   # (1,2,3)
    y = torch.tensor([[0.4, 0.5]])                                   # (1,2)
    alpha = torch.linspace(0, 1, 5)[1:-1]
    diff = y.unsqueeze(-1) - quantiles
    expected = torch.max(alpha * diff, (alpha - 1.0) * diff).mean()
    assert torch.allclose(tabicl_pinball_loss(quantiles, y), expected)


def test_pinball_loss_is_zero_for_perfect_median_only_when_degenerate() -> None:
    """A single quantile level (Q=1 → alpha=0.5) reduces to half the MAE,
    so a perfect prediction gives exactly 0 loss."""
    from src.train.tabicl_model import tabicl_pinball_loss
    q = torch.tensor([[[0.3], [0.7]]])
    y = torch.tensor([[0.3, 0.7]])
    assert float(tabicl_pinball_loss(q, y)) == pytest.approx(0.0)


def _tiny_tabicl(*, regressor: bool):
    """Build a deliberately small TabICLv2 so model-level tests are fast on
    CPU. Only config keys the installed version accepts are passed."""
    import inspect
    from src.train.tabicl_compat import import_tabicl_core
    TabICL = import_tabicl_core()
    sig = inspect.signature(TabICL.__init__)
    wanted = {
        "max_classes": 0 if regressor else 10,
        "embed_dim": 32,
        "col_num_blocks": 1, "col_nhead": 2, "col_num_inds": 4,
        "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2,
        "icl_num_blocks": 1, "icl_nhead": 2,
        "ff_factor": 1, "dropout": 0.0, "recompute": False,
    }
    if regressor:
        wanted["num_quantiles"] = 99
    cfg = {k: v for k, v in wanted.items() if k in sig.parameters}
    return TabICL(**cfg), cfg


@pytest.mark.parametrize("regressor", [False, True])
def test_save_finetuned_tabicl_round_trips(tmp_path: Path, regressor: bool) -> None:
    """Our saver must write EXACTLY the schema upstream's loader reads
    (``{config, state_dict}``, weights_only-safe) plus a provenance
    sidecar, and reset ``recompute`` so inference doesn't pay for
    gradient checkpointing."""
    pytest.importorskip("tabicl")
    from src.train.tabicl_model import save_finetuned_tabicl
    model, cfg = _tiny_tabicl(regressor=regressor)
    cfg = {**cfg, "recompute": True}          # as training would leave it
    path = tmp_path / "tabicl-test-v2.ckpt"
    prov = {"schema_version": 1, "model_family": "tabicl",
            "test_datasets": ["0002.loss2"]}
    save_finetuned_tabicl(model, cfg, path, provenance=prov)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert {"config", "state_dict"} <= set(raw)
    assert raw["config"]["recompute"] is False
    assert set(raw["state_dict"]) == set(model.state_dict())
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    assert json.loads(sidecar.read_text())["test_datasets"] == ["0002.loss2"]


def test_load_tabicl_rejects_wrong_head_for_track(tmp_path: Path) -> None:
    """Loading a CLASSIFIER checkpoint for the LGD track (and vice versa)
    must fail loudly — a silent mixup would train garbage for 50 epochs."""
    pytest.importorskip("tabicl")
    from src.train.tabicl_model import load_tabicl_for_training, save_finetuned_tabicl
    model, cfg = _tiny_tabicl(regressor=False)
    path = tmp_path / "tabicl-classifier-v2-test.ckpt"
    save_finetuned_tabicl(model, cfg, path)
    with pytest.raises(ValueError, match="CLASSIFIER"):
        load_tabicl_for_training(path, track="lgd", device="cpu")
    # The matching track loads fine and returns the config for re-saving.
    loaded, out_cfg = load_tabicl_for_training(path, track="pd", device="cpu")
    assert out_cfg["recompute"] is True        # gradient checkpointing forced on
    assert sum(p.numel() for p in loaded.parameters()) > 0


def test_load_tabicl_missing_file_names_the_download_source(tmp_path: Path) -> None:
    pytest.importorskip("tabicl")
    from src.train.tabicl_model import load_tabicl_for_training
    with pytest.raises(FileNotFoundError, match="huggingface.co/jingang/TabICL"):
        load_tabicl_for_training(tmp_path / "absent.ckpt", track="pd", device="cpu")


def test_freeze_backbone_leaves_only_icl_trainable(tmp_path: Path) -> None:
    pytest.importorskip("tabicl")
    from src.train.tabicl_model import load_tabicl_for_training, save_finetuned_tabicl
    model, cfg = _tiny_tabicl(regressor=False)
    path = tmp_path / "tabicl-classifier-v2-test.ckpt"
    save_finetuned_tabicl(model, cfg, path)

    frozen, _ = load_tabicl_for_training(
        path, track="pd", device="cpu", freeze_backbone=True,
    )
    assert not any(p.requires_grad for p in frozen.col_embedder.parameters())
    assert not any(p.requires_grad for p in frozen.row_interactor.parameters())
    assert all(p.requires_grad for p in frozen.icl_predictor.parameters())

    # REGRESSION (2026-08-06): the frozen stages must stay on the TRAINING
    # forward path. TabICLv2 branches `if self.training: _train_forward else:
    # _inference_forward`, and the inference branch runs under no_grad and
    # writes CLS tokens into its input in place — which raised "A view was
    # created in no_grad mode and is being modified inplace with grad mode
    # enabled" and killed all 16 `_iclhead` trials on the cluster. Freezing is
    # requires_grad=False ONLY; never .eval().
    frozen.train()
    assert frozen.col_embedder.training, "frozen col_embedder must stay on the train path"
    assert frozen.row_interactor.training, "frozen row_interactor must stay on the train path"

    # Full-FT mode leaves the backbone trainable. One parameter
    # (`row_interactor.tf_row.rope.freqs`) ships with requires_grad=False
    # upstream — RoPE frequencies are constants stored as a Parameter — so
    # assert on the modules we control, not on every tensor.
    full, _ = load_tabicl_for_training(path, track="pd", device="cpu")
    assert any(p.requires_grad for p in full.col_embedder.parameters())
    assert any(p.requires_grad for p in full.row_interactor.parameters())
    assert all(p.requires_grad for p in full.icl_predictor.parameters())
    n_frozen = sum(1 for p in full.parameters() if not p.requires_grad)
    assert n_frozen <= 1, "unexpected frozen tensors in full-FT mode"


# =========================================================================== #
# Block 3 — batch construction
# =========================================================================== #


def _synthetic_loaded(*, task_type: str, n: int = 240, d: int = 6):
    from src.train.dataloader import _LoadedDataset
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    X.iloc[3:9, 2] = np.nan                       # must be imputed away
    X["cat"] = pd.Series(rng.integers(0, 3, size=n)).astype("category")
    if task_type == "classification":
        y = (rng.uniform(size=n) < 0.25).astype(np.int64)
    else:
        y = rng.uniform(0.0, 1.0, size=n)
    return _LoadedDataset(
        dataset_id=f"synthetic_{task_type}", X=X, y=y,
        cat_columns=["cat"], task_type=task_type,
    )


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_build_tabicl_step_batch_shapes_and_finiteness(task_type: str) -> None:
    pytest.importorskip("tabicl")
    from src.train.dataloader import _build_tabicl_step_batch, TabICLTrainBatch
    loaded = _synthetic_loaded(task_type=task_type)
    batch = _build_tabicl_step_batch(
        loaded, n_total_target=200, query_fraction=0.20,
        rng=np.random.default_rng(1),
        n_estimators=2, epoch_seed=7, replica=0, preprocessing_seed=99,
    )
    assert isinstance(batch, TabICLTrainBatch)
    n_est, n_total, n_feat = batch.X.shape
    assert n_est == 2                                  # ensemble members
    assert n_total == 200                              # ctx + query
    assert n_feat == 7                                 # 6 numeric + 1 encoded cat
    assert batch.y_train.shape == (2, batch.train_size)
    assert batch.y_query.shape == (2, n_total - batch.train_size)
    assert batch.train_size == 160                     # query_fraction=0.20
    # NaNs/infs must never reach the transformer (only tabicl's scalers are
    # NaN-aware; the attention stack is not).
    assert torch.isfinite(batch.X).all()
    assert torch.isfinite(batch.y_train).all()
    assert batch.task_type == task_type
    assert batch.dataset_id == loaded.dataset_id


def test_build_tabicl_step_batch_is_deterministic_given_seeds() -> None:
    """Same seeds → identical batch. The per-dataset preprocessing_seed is
    fixed across epochs by design, so reproducibility must not depend on
    call order."""
    pytest.importorskip("tabicl")
    from src.train.dataloader import _build_tabicl_step_batch
    loaded = _synthetic_loaded(task_type="classification")
    kwargs = dict(n_total_target=200, query_fraction=0.20, n_estimators=2,
                  epoch_seed=7, replica=0, preprocessing_seed=99)
    a = _build_tabicl_step_batch(loaded, rng=np.random.default_rng(1), **kwargs)
    b = _build_tabicl_step_batch(loaded, rng=np.random.default_rng(1), **kwargs)
    assert torch.equal(a.X, b.X)
    assert torch.equal(a.y_train, b.y_train)
    assert torch.equal(a.y_query, b.y_query)


def test_tabicl_batch_to_device_is_a_noop_on_cpu() -> None:
    pytest.importorskip("tabicl")
    from src.train.dataloader import _build_tabicl_step_batch
    loaded = _synthetic_loaded(task_type="regression")
    batch = _build_tabicl_step_batch(
        loaded, n_total_target=120, query_fraction=0.25,
        rng=np.random.default_rng(3),
        n_estimators=2, epoch_seed=1, replica=0, preprocessing_seed=5,
    )
    moved = batch.to("cpu")
    assert moved.X.device.type == "cpu"
    assert moved.train_size == batch.train_size
    assert moved.task_type == batch.task_type


@pytest.mark.parametrize("regressor", [False, True])
def test_tabicl_forward_and_loss_are_finite_and_differentiable(regressor: bool) -> None:
    """The exact forward/loss branch loop.py runs, on a tiny model: TabICLv2's
    training forward, then CE over the first n_classes logits (classification)
    or mean pinball over the quantile head (regression)."""
    pytest.importorskip("tabicl")
    from src.train.dataloader import _build_tabicl_step_batch
    from src.train.tabicl_model import tabicl_pinball_loss
    task_type = "regression" if regressor else "classification"
    loaded = _synthetic_loaded(task_type=task_type)
    batch = _build_tabicl_step_batch(
        loaded, n_total_target=200, query_fraction=0.20,
        rng=np.random.default_rng(1),
        n_estimators=2, epoch_seed=7, replica=0, preprocessing_seed=99,
    )
    model, _ = _tiny_tabicl(regressor=regressor)
    model.train()
    out = model(batch.X, batch.y_train)
    assert out.shape[:2] == batch.y_query.shape
    if regressor:
        assert out.shape[-1] == 99                     # num_quantiles
        loss = tabicl_pinball_loss(out, batch.y_query)
    else:
        assert out.shape[-1] == 10                     # max_classes logits
        n_cls = int(batch.y_train.max().item()) + 1
        loss = torch.nn.functional.cross_entropy(
            out[..., :n_cls].reshape(-1, n_cls),
            batch.y_query.long().reshape(-1),
        )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_training_batch_absorbs_inf(monkeypatch) -> None:
    """±inf must never reach the transformer. TabPFN clips inf inside its
    feature normaliser; TabICLv2 does not, and credit features reach inf via
    zero-denominator ratios."""
    pytest.importorskip("tabicl")
    from src.train.dataloader import _build_tabicl_step_batch
    loaded = _synthetic_loaded(task_type="classification")
    loaded.X.iloc[0, 0] = np.inf
    loaded.X.iloc[1, 0] = -np.inf
    batch = _build_tabicl_step_batch(
        loaded, n_total_target=200, query_fraction=0.20,
        rng=np.random.default_rng(1),
        n_estimators=2, epoch_seed=7, replica=0, preprocessing_seed=99,
    )
    assert torch.isfinite(batch.X).all()


# =========================================================================== #
# Block 4 — training-loop integration
# =========================================================================== #


def test_descriptive_name_uses_iclhead_tag_for_tabicl() -> None:
    """The grid's use_lora axis means freeze-backbone for tabicl, so the
    filename must say `_iclhead`, never `_lora`."""
    from src.train.loop import descriptive_name
    common = dict(run_name="creditpfn", track="pd", learning_rate=1e-5,
                  seed=42, use_lora=True)
    tabicl = descriptive_name(
        base_path="checkpoints/tabicl-classifier-v2-20260212.ckpt", **common)
    tabpfn = descriptive_name(
        base_path="checkpoints/tabpfn-v3-classifier-v3_default.ckpt", **common)
    assert tabicl.endswith("_iclhead.ckpt") and "_lora" not in tabicl
    assert tabpfn.endswith("_lora.ckpt") and "_iclhead" not in tabpfn
    # Full-FT trials carry neither tag, in both families.
    for base in ("checkpoints/tabicl-classifier-v2-20260212.ckpt",
                 "checkpoints/tabpfn-v3-classifier-v3_default.ckpt"):
        name = descriptive_name(**{**common, "use_lora": False}, base_path=base)
        assert "_lora" not in name and "_iclhead" not in name


def test_row_cap_resolution_prefers_the_tabicl_key() -> None:
    """A tabicl base must resolve to the `tabicl` row cap, not to a
    version-regex match or the TabPFN default."""
    from src.train.loop import _resolve_max_rows_per_epoch
    caps = {"v3": 26000, "v2.6": 11000, "tabicl": 26000, "default": 99999}
    assert _resolve_max_rows_per_epoch(
        "checkpoints/tabicl-classifier-v2-20260212.ckpt", caps) == 26000
    assert _resolve_max_rows_per_epoch(
        "checkpoints/tabpfn-v3-classifier-v3_default.ckpt", caps) == 26000
    assert _resolve_max_rows_per_epoch(
        "checkpoints/tabpfn-v2.6-regressor-v2.6_default.ckpt", caps) == 11000
    # No tabicl entry → the shared default, not a crash.
    assert _resolve_max_rows_per_epoch(
        "checkpoints/tabicl-classifier-v2-20260212.ckpt",
        {"v3": 26000, "default": 26000}) == 26000


def test_missing_context_class_guard_handles_tabicl_batches() -> None:
    """The classifier-only skip rule (query classes ⊄ context classes) must
    work on TabICLTrainBatch, whose per-member class shuffles are bijective
    remaps applied consistently to y_train and y_query."""
    from src.train.dataloader import TabICLTrainBatch
    from src.train.loop import _query_missing_context_class

    def _batch(y_train, y_query, task_type="classification"):
        return TabICLTrainBatch(
            X=torch.zeros(1, 4, 2),
            y_train=torch.tensor(y_train, dtype=torch.float32),
            y_query=torch.tensor(y_query, dtype=torch.float32),
            train_size=len(y_train[0]),
            y_scaler_mean=None, y_scaler_std=None,
            task_type=task_type, dataset_id="synthetic",
        )

    assert not _query_missing_context_class(_batch([[0, 1, 0]], [[0, 1]]))
    assert _query_missing_context_class(_batch([[0, 0, 0]], [[0, 1]]))
    # Regression batches are never skipped by this rule.
    assert not _query_missing_context_class(
        _batch([[0.2, 0.9]], [[0.5]], task_type="regression"))


# =========================================================================== #
# Block 5 — eval-side registry / naming
# =========================================================================== #


def test_build_baselines_creates_per_family_untuned_controls(tmp_path: Path) -> None:
    """A mixed base list must yield ONE untuned control per family, each of
    the right wrapper class, so trained-vs-untuned stays within-family."""
    pytest.importorskip("tabicl")
    from src.model.registry import build_baselines
    bases = []
    for name in ("tabpfn-v3-classifier-v3_default.ckpt",
                 "tabicl-classifier-v2-20260212.ckpt"):
        p = tmp_path / name
        p.write_bytes(b"not-a-real-checkpoint")   # existence is all that's checked
        bases.append(str(p))

    out = build_baselines(
        track="pd", base_paths_for_tabpfn_untuned=bases,
        enabled=["tabpfn-untuned", "tabicl-untuned"],
    )
    sources = sorted(h.source for h, _ in out)
    assert sources == ["tabicl-untuned", "tabpfn-untuned"]

    # Each family can be dropped independently.
    only_icl = build_baselines(
        track="pd", base_paths_for_tabpfn_untuned=bases,
        enabled=["tabicl-untuned"],
    )
    assert [h.source for h, _ in only_icl] == ["tabicl-untuned"]
    assert "tabicl-untuned" in only_icl[0][0].name


def test_eval_wrapper_sanitizes_inf_and_dead_columns(tmp_path: Path) -> None:
    """MEASURED upstream constraints (tabicl 2.1.1, 2026-08-04): its sklearn
    wrappers raise ValueError on ±inf and IndexError on an all-NaN column.
    Both are reachable from real credit folds, so the wrappers must absorb
    them without dropping columns (widths must match between fit/predict)."""
    pytest.importorskip("tabicl")
    from src.model.tabicl_models import TabICLUntuned, _sanitize
    from src.train.tabicl_model import save_finetuned_tabicl

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    X[3:12, 1] = np.nan          # ordinary missingness
    X[50, 4], X[51, 4] = np.inf, -np.inf
    X[:, 2] = np.nan             # all-NaN column
    y = (rng.uniform(size=200) < 0.25).astype(np.int64)

    clean, dead = _sanitize(X)
    assert np.isfinite(clean[:, 2]).all()           # dead column zero-filled
    assert clean.shape == X.shape                   # never dropped
    assert dead.tolist() == [False, False, True, False, False]
    assert np.isnan(clean[:, 1]).any()              # real NaNs are PRESERVED
    assert not np.isinf(clean).any()

    model, cfg = _tiny_tabicl(regressor=False)
    ckpt = tmp_path / "tabicl-classifier-v2-test.ckpt"
    save_finetuned_tabicl(model, cfg, ckpt)
    wrapper = TabICLUntuned(task_type="classification", base_path=ckpt,
                            device="cpu", n_estimators=1)
    wrapper.fit(X[:150], y[:150], categorical_idx=[])
    proba = wrapper.predict_proba(X[150:])
    assert proba.shape == (50, 2)
    assert np.isfinite(proba).all()


def test_method_dirname_and_decoding_round_trip() -> None:
    """`_method_dirname` (writer) and `_decode_method_dirname` (reader) must
    agree — they are the on-disk contract between eval and the notebooks."""
    from src.eval.benchmark import _method_dirname
    from src.model.base import ModelHandle
    from src.visualize.eval_viz import _decode_method_dirname

    handle = ModelHandle(
        name="tabicl-trained[…]", track="lgd", task_type="regression",
        source="tabicl-trained", base_path="x",
        extra={"base_checkpoint": "checkpoints/tabicl-regressor-v2-20260212.ckpt",
               "learning_rate": 1e-5, "use_lora": True,
               "epoch_pass_mode": "full_pass"},
    )
    dirname = _method_dirname(handle)
    assert dirname == "tabicl-trained__tabicl-v2__lr1e-05__fullpass__iclhead"
    meta = _decode_method_dirname(dirname)
    assert meta["source"] == "tabicl-trained"
    assert meta["base_short"] == "tabicl-v2"
    assert meta["lr"] == pytest.approx(1e-5)
    assert meta["use_lora"] is True
    assert meta["full_pass"] is True

    untuned = ModelHandle(
        name="tabicl-untuned[…]", track="pd", task_type="classification",
        source="tabicl-untuned",
        base_path="checkpoints/tabicl-classifier-v2-20260212.ckpt",
    )
    d2 = _method_dirname(untuned)
    assert d2 == "tabicl-untuned__tabicl-v2"
    assert _decode_method_dirname(d2)["source"] == "tabicl-untuned"

    # TabPFN dirnames must decode exactly as before this family was added.
    tabpfn = ModelHandle(
        name="tabpfn-trained[…]", track="pd", task_type="classification",
        source="tabpfn-trained", base_path="x",
        extra={"base_checkpoint": "checkpoints/tabpfn-v3-classifier-v3_default.ckpt",
               "learning_rate": 3e-7, "use_lora": False,
               "epoch_pass_mode": "one_sample"},
    )
    d3 = _method_dirname(tabpfn)
    assert d3 == "tabpfn-trained__v3-default__lr3e-07"
    m3 = _decode_method_dirname(d3)
    assert (m3["source"], m3["base_short"], m3["use_lora"], m3["full_pass"]) == (
        "tabpfn-trained", "v3-default", False, False)


def test_row_cap_for_handle_is_symmetric_between_trained_and_untuned() -> None:
    """Regression test (2026-08-04): the untuned branch used to strip a
    dirname-style prefix off `handle.name`, which never matched, so every
    untuned model silently fell through to the `default` cap while its
    trained counterpart got the architecture cap — an un-paired comparison
    on any dataset bigger than the default."""
    from src.eval.benchmark import resolve_max_rows_for_handle
    from src.model.base import ModelHandle
    caps = {"v3": 1_000_000, "v2.6": 50_000, "tabicl": 50_000, "default": 50_000}

    untuned = ModelHandle(
        name="tabpfn-untuned[tabpfn-v3-classifier-v3_default]", track="pd",
        task_type="classification", source="tabpfn-untuned",
        base_path="checkpoints/tabpfn-v3-classifier-v3_default.ckpt",
    )
    trained = ModelHandle(
        name="tabpfn-trained[…]", track="pd", task_type="classification",
        source="tabpfn-trained", base_path="y",
        extra={"base_checkpoint": "checkpoints/tabpfn-v3-classifier-v3_default.ckpt"},
    )
    assert resolve_max_rows_for_handle(untuned, max_rows_per_model=caps) == 1_000_000
    assert resolve_max_rows_for_handle(trained, max_rows_per_model=caps) == 1_000_000

    icl = ModelHandle(
        name="tabicl-untuned[tabicl-classifier-v2-20260212]", track="pd",
        task_type="classification", source="tabicl-untuned",
        base_path="checkpoints/tabicl-classifier-v2-20260212.ckpt",
    )
    assert resolve_max_rows_for_handle(icl, max_rows_per_model=caps) == 50_000

    classical = ModelHandle(
        name="xgboost", track="pd", task_type="classification",
        source="baseline",
    )
    assert resolve_max_rows_for_handle(classical, max_rows_per_model=caps) is None


# =========================================================================== #
# Block 6 — REAL end-to-end training run (no mocks)
# =========================================================================== #


def _write_tabicl_corpus(root: Path, track: str, task_type: str) -> None:
    """Five synthetic sanitized CSVs + a manifest, mirroring the layout
    ``ProcessedDatasetLoader`` expects."""
    rng = np.random.default_rng(42)
    folder = root / "data" / "processed" / track
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, n in enumerate([180, 200, 160, 220, 300]):
        dataset_id = f"000{i + 1}.ds{i + 1}"
        d = {f"f{j}": rng.normal(size=n) for j in range(5)}
        d["cat"] = rng.choice(["A", "B", "C"], size=n)
        d["f0"][0] = np.inf                     # must be absorbed, not crash
        d["target"] = (rng.integers(0, 2, size=n).astype(np.int64)
                       if task_type == "classification"
                       else rng.uniform(0, 1, size=n).astype(np.float32))
        pd.DataFrame(d).to_csv(folder / f"{dataset_id}.sanitized.csv", index=False)
        rows.append({"dataset_id": dataset_id, "track": track,
                     "task_type": task_type, "target_column": "target",
                     "categorical_columns": "cat", "n_rows": n, "n_cols": 6,
                     "source": "synthetic"})
    (root / "data").mkdir(parents=True, exist_ok=True)
    manifest_dir = root / "output" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_dir / f"manifest_{track}.csv", index=False)


def test_freeze_backbone_survives_repeated_steps(tmp_path: Path) -> None:
    """The `_iclhead` crash only appeared a few steps in, so a single forward
    is not enough to catch it. Run several optimiser steps with the real
    freeze path and `recompute=True` (gradient checkpointing), which is the
    configuration the cluster ran."""
    pytest.importorskip("tabicl")
    from src.train.tabicl_model import load_tabicl_for_training, save_finetuned_tabicl

    model, cfg = _tiny_tabicl(regressor=False)
    path = tmp_path / "tabicl-classifier-v2-test.ckpt"
    save_finetuned_tabicl(model, cfg, path)
    trained, out_cfg = load_tabicl_for_training(
        path, track="pd", device="cpu", freeze_backbone=True,
    )
    assert out_cfg["recompute"] is True
    trained.train()
    opt = torch.optim.AdamW(
        [p for p in trained.parameters() if p.requires_grad], lr=1e-6)

    rng = np.random.default_rng(0)
    for _ in range(5):
        X = torch.tensor(rng.normal(size=(2, 120, 6)), dtype=torch.float32)
        y = torch.tensor((rng.uniform(size=(2, 120)) < 0.25).astype("float32"))
        out = trained(X, y[:, :96])
        n_cls = int(y[:, :96].max().item()) + 1
        loss = torch.nn.functional.cross_entropy(
            out[..., :n_cls].reshape(-1, n_cls), y[:, 96:].long().reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        # The backbone must remain frozen AND on the train path throughout.
        assert trained.col_embedder.training and trained.row_interactor.training
        assert not any(p.requires_grad for p in trained.col_embedder.parameters())


@pytest.mark.parametrize(
    "track, freeze_backbone",
    [("pd", True), ("lgd", False)],
)
def test_train_one_config_end_to_end_tabicl(
    tmp_path: Path, monkeypatch, track: str, freeze_backbone: bool,
) -> None:
    """Drive the REAL ``train_one_config`` on a tabicl base — nothing mocked
    except the checkpoint's size (a deliberately tiny TabICLv2 so it runs on
    CPU in seconds).

    This is the one test that exercises every seam at once: family
    detection, the loader, the batch builder, the forward/loss branch, the
    L2-SP anchor, the monitor eval through TabICLv2's sklearn wrappers, the
    snapshot write + cleanup, the save schema, the provenance sidecar, and
    the `_iclhead` filename tag. The two parameter sets between them cover
    both losses (CE / pinball) and both adaptation modes.
    """
    pytest.importorskip("tabicl")
    from omegaconf import OmegaConf
    from src.train.loop import train_one_config
    from src.train.tabicl_model import save_finetuned_tabicl
    from src.utils.paths import resolve_staging_path

    monkeypatch.setenv("CREDITPFN_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(tmp_path))
    from src.utils import paths as _paths
    _paths._autodetect_data_root.cache_clear()

    is_reg = track == "lgd"
    task_type = "regression" if is_reg else "classification"
    _write_tabicl_corpus(tmp_path, track, task_type)

    model, cfg_model = _tiny_tabicl(regressor=is_reg)
    base_name = f"tabicl-{'regressor' if is_reg else 'classifier'}-v2-20260212.ckpt"
    staged = Path(resolve_staging_path(f"checkpoints/{base_name}"))
    staged.parent.mkdir(parents=True, exist_ok=True)
    save_finetuned_tabicl(model, cfg_model, staged)

    cfg = OmegaConf.create({
        "seed": 0, "run_name": "e2e", "device": "cpu", "track": track,
        "tunable": {
            "classifier_base_paths": [f"checkpoints/{base_name}"],
            "regressor_base_paths": [f"checkpoints/{base_name}"],
            "learning_rates": [1e-4],
            "use_lora": [freeze_backbone],
        },
        "corpus": {"train_fraction": 0.6, "test_fraction": 0.4},
        "optimizer": {"weight_decay": 0.0, "l2sp_lambda": 0.003},
        "scheduler": {"warmup_fraction": 0.10},
        "train": {
            "epochs": 2, "accumulate_grad_batches": 1, "grad_clip_norm": 1.0,
            "amp": False, "dataloader_workers": 0, "step_log_interval": 10,
            "epoch_eval_subsample_samples": 60,
            "epoch_eval_n_estimators": 1,
            "epoch_eval_every": 1,
            "n_estimators_finetune": {"pd": 2, "lgd": 8, "default": 2},
            "n_estimators_finetune_tabicl": 2,
            "divergence_patience": 5,
        },
        "checkpoint": {"trained_dir": "checkpoints/trained"},
        "data_cfg_path": "config/data.yaml",
    })

    records = []
    result = train_one_config(
        cfg, track=track, use_lora=freeze_backbone,
        on_epoch_end=records.append,
    )

    # epoch=-1 baseline + 2 trained epochs, all with a finite training loss.
    assert [r.epoch for r in records] == [-1, 0, 1]
    assert all(np.isfinite(r.train_loss) for r in records if r.epoch >= 0)
    assert not result.diverged

    ckpt = Path(result.final_ckpt_path)
    assert ckpt.exists()
    assert ("_iclhead" in ckpt.name) is freeze_backbone
    assert "_lora" not in ckpt.name

    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert {"config", "state_dict"} <= set(raw)
    assert raw["config"]["recompute"] is False

    prov = json.loads(
        ckpt.with_suffix(ckpt.suffix + ".provenance.json").read_text())
    assert prov["model_family"] == "tabicl"
    assert prov["adaptation_mode"] == (
        "iclhead_only" if freeze_backbone else "full_ft")
    hp = prov["hyperparameters"]
    # TabICLv2 uses 2 members on BOTH tracks — the lgd-track value of 8 must
    # NOT leak into a tabicl trial.
    assert hp["n_estimators_finetune"] == 2
    # The tabicl row-cap key resolved (26 000 = v3's cap, for cross-family
    # parity), NOT the member-scaled TabPFN value.
    assert hp["max_rows_per_epoch"] == 26_000
    assert prov["tabicl_version"] is not None

    # The rolling monitor snapshot must be cleaned up on success.
    assert list(ckpt.parent.glob("*.epoch_eval.ckpt")) == []

    # And the finished checkpoint must load in the EVAL wrapper.
    from src.model.tabicl_models import TabICLTrained
    df = pd.read_csv(tmp_path / "data" / "processed" / track / "0001.ds1.sanitized.csv")
    X = np.asarray(df.drop(columns=["target", "cat"]).values, dtype=np.float64)
    y = df["target"].values
    wrapper = TabICLTrained(task_type=task_type, ckpt_path=ckpt,
                            device="cpu", n_estimators=1)
    wrapper.fit(X[:120], y[:120], categorical_idx=[])
    pred = wrapper.predict(X[120:]) if is_reg else wrapper.predict_proba(X[120:])
    assert np.isfinite(np.asarray(pred)).all()


def test_training_grid_contains_both_families() -> None:
    """config/train.yaml must sweep TabPFN v3 + v2.6 + TabICLv2 v2 on both
    tracks, and the pipeline's grid expansion must equal the cartesian
    product of the configured axes (whatever size that currently is)."""
    from omegaconf import OmegaConf
    repo = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(repo / "config" / "train.yaml")
    from src.train.tabicl_compat import model_family
    for key in ("classifier_base_paths", "regressor_base_paths"):
        bases = list(cfg.tunable[key])
        fams = {model_family(b) for b in bases}
        assert fams == {"tabpfn", "tabicl"}, (key, bases)
        # The product is NOT plain: two rules shrink it, and both are deliberate.
        #   * `adapter_families` restricts the `use_lora: true` arm to named families
        #     (run-8: TabICLv2 only — LoRA on TabPFN was a measured no-op three times).
        #   * `corpus.min_train_rows` is a swept axis as of run-8.
        n_other = (len(cfg.tunable.learning_rates)
                   * len(cfg.tunable.query_fractions)
                   * len(cfg.tunable.accumulate_grad_batches)
                   * len(cfg.tunable.epoch_pass_modes)
                   * max(1, len(list(cfg.corpus.get("min_train_rows", [0])))))
        adapter_fams = [str(x).lower() for x in
                        (cfg.tunable.get("adapter_families", None) or [])]
        n_adapter_bases = (
            len(bases) if not adapter_fams
            else sum(1 for b in bases if any(f in str(b).lower() for f in adapter_fams))
        )
        arms = len(bases) + (n_adapter_bases if True in list(cfg.tunable.use_lora) else 0)
        expected = arms * n_other

        # Assert the STRUCTURE, not a magic number. This test previously
        # hardcoded the trial count (48, then 18, then 36) and broke on every
        # deliberate grid reshape — three false alarms that taught nothing.
        # What actually matters is that the pipeline's real grid expansion
        # equals the cartesian product of the configured axes: that catches a
        # silently dropped or duplicated axis, which is a genuine bug, while
        # staying correct however the sweep is resized.
        from omegaconf import open_dict
        track = "pd" if key == "classifier_base_paths" else "lgd"
        cfg_track = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        with open_dict(cfg_track):
            cfg_track.track = track
        from scripts.train_pipeline import _resolve_grid
        actual = len(_resolve_grid(cfg_track, single=False))
        assert actual == expected, (key, actual, expected)
        # Every base must appear, so no family can be silently skipped.
        assert {b for b, *_ in _resolve_grid(cfg_track, single=False)} == set(bases)
