"""Run-targeting in the visualization loaders (added 03-09-2026 for Experiment 1).

Experiment 1 writes **per-split** artefacts — a manifest ``exp1_s00_pd.csv`` …
``exp1_s07_pd.csv`` per split, and eval CSVs named ``exp1_s<NN>_<ts>__task…`` — into
the same ``output/`` tree that already holds run-8's single-run files. The notebooks call
``training_viz.use_run('exp1')`` / ``eval_viz.use_run('exp1')`` so every loader sees exactly
one run. These tests pin that contract: the training loader must **pool all splits** into one
frame with a ``split`` column, and the eval loader must **keep only** the selected run's files.
Without this, exp1's eight splits either vanish (single-file lookup) or silently average with
run-8's older numbers.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.visualize import eval_viz, training_viz


@pytest.fixture(autouse=True)
def _reset_run_override():
    """The overrides are module globals — restore the pool-everything default around every test
    so state never leaks between tests or into the rest of the suite."""
    training_viz.use_run(None)
    eval_viz.use_run(None)
    yield
    training_viz.use_run(None)
    eval_viz.use_run(None)


def _write_manifest(path, ckpt_name):
    pd.DataFrame([{
        "track": "pd",
        "base_checkpoint": "tabpfn-v3-classifier-v3_default.ckpt",
        "learning_rate": 1e-6,
        "use_lora": False,
        "seed": 42,
        "final_ckpt_path": f"/x/{ckpt_name}.ckpt",
        "status": "OK",
    }]).to_csv(path, index=False)


def test_training_loader_pools_per_split_manifests(tmp_path, monkeypatch):
    """``use_run('exp1')`` concatenates every ``exp1_s<NN>_pd.csv`` into one frame, tagging each
    row with its integer ``split`` and preserving the per-split trial name."""
    # ``_resolve_paths`` does ``from src.utils.paths import ... manifests_dir`` at call time, so
    # patching the source symbol is enough.
    monkeypatch.setattr("src.utils.paths.manifests_dir", lambda: tmp_path)
    for s in range(3):
        _write_manifest(tmp_path / f"exp1_s{s:02d}_pd.csv",
                        f"exp1_s{s:02d}_pd_tabpfn-v3-classifier-v3_default_lr1e-06_seed42")

    training_viz.use_run("exp1")
    df = training_viz.load_run_manifest("pd")

    assert len(df) == 3, "all three splits must be pooled"
    assert "split" in df.columns
    assert sorted(int(x) for x in df["split"].unique()) == [0, 1, 2]
    # each split keeps its own trial identity (not collapsed into one)
    assert sorted(df["trial_name"]) == [
        "exp1_s00_pd_tabpfn-v3-classifier-v3_default_lr1e-06_seed42",
        "exp1_s01_pd_tabpfn-v3-classifier-v3_default_lr1e-06_seed42",
        "exp1_s02_pd_tabpfn-v3-classifier-v3_default_lr1e-06_seed42",
    ]


def test_training_loader_single_file_layout_still_works(tmp_path, monkeypatch):
    """A run written as one ``<run>_pd.csv`` (the run-8 layout) must still load — the per-split
    branch is a fallback, not a replacement."""
    monkeypatch.setattr("src.utils.paths.manifests_dir", lambda: tmp_path)
    _write_manifest(tmp_path / "creditpfn_pd.csv", "creditpfn_pd_base_lr1e-06_seed42")

    training_viz.use_run("creditpfn")
    df = training_viz.load_run_manifest("pd")
    assert len(df) == 1


def test_training_loader_drops_stale_pre_provenance_rows(tmp_path, monkeypatch):
    """Rows with an empty ``git_commit`` are old-code noise (the pre-fix ``ckpt_path`` failures that
    accumulate in the re-appended manifest) and must be dropped; rows stamped with a real commit are
    kept. Without this the exp1 notebooks report a ~50 % failure rate that no longer exists."""
    monkeypatch.setattr("src.utils.paths.manifests_dir", lambda: tmp_path)
    rows = [
        {"track": "pd", "base_checkpoint": "tabpfn-v3-classifier-v3_default.ckpt",
         "learning_rate": 1e-6, "use_lora": True, "seed": 42, "final_ckpt_path": "",
         "status": "FAIL", "git_commit": ""},          # stale: no provenance
        {"track": "pd", "base_checkpoint": "tabpfn-v3-classifier-v3_default.ckpt",
         "learning_rate": 1e-6, "use_lora": False, "seed": 42,
         "final_ckpt_path": "/x/exp1_s00_pd_tabpfn-v3-classifier-v3_default_lr1e-06_seed42.ckpt",
         "status": "OK", "git_commit": "badacc2"},      # current: real commit
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "exp1_s00_pd.csv", index=False)

    training_viz.use_run("exp1")
    df = training_viz.load_run_manifest("pd")
    assert len(df) == 1, "the empty-commit FAIL row must be dropped"
    assert set(df["status"]) == {"OK"}


def _write_eval_csv(root, method, run_prefix):
    d = root / "PD" / method
    d.mkdir(parents=True, exist_ok=True)
    fname = f"{run_prefix}_20260901_120000__task0_ds-0001.some_dataset.csv"
    pd.DataFrame([{"dataset": "some_dataset", "fold": 0, "roc_auc": 0.8}]).to_csv(d / fname, index=False)


def test_eval_loader_filters_to_selected_run(tmp_path, monkeypatch):
    """``use_run('exp1')`` keeps only files whose name starts ``exp1_`` — excluding ``exp0_`` and
    ``creditpfn_`` sharing the tree — while ``use_run(None)`` pools them all."""
    monkeypatch.setattr(eval_viz, "_resolve_paths", lambda: {"benchmark_root": tmp_path})
    _write_eval_csv(tmp_path, "xgboost", "exp1_s00")
    _write_eval_csv(tmp_path, "xgboost", "exp1_s01")
    _write_eval_csv(tmp_path, "xgboost", "exp0_s00")     # a DIFFERENT experiment
    _write_eval_csv(tmp_path, "xgboost", "creditpfn")    # run-8

    eval_viz.use_run("exp1")
    only_exp1 = eval_viz.load_eval_results("pd")
    assert len(only_exp1) == 2, "exp0_ and creditpfn_ must be excluded; both exp1_ splits kept"

    eval_viz.use_run(None)
    everything = eval_viz.load_eval_results("pd")
    assert len(everything) == 4, "no override pools every run"


def test_eval_run_prefix_does_not_match_sibling_numbers(tmp_path, monkeypatch):
    """``exp1`` must not swallow ``exp10`` — the ``<run>_`` boundary guards against prefix bleed."""
    monkeypatch.setattr(eval_viz, "_resolve_paths", lambda: {"benchmark_root": tmp_path})
    _write_eval_csv(tmp_path, "xgboost", "exp1_s00")
    _write_eval_csv(tmp_path, "xgboost", "exp10_s00")

    eval_viz.use_run("exp1")
    df = eval_viz.load_eval_results("pd")
    assert len(df) == 1, "exp10_ shares the 'exp1' prefix but not the 'exp1_' boundary"
