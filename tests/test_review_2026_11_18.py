"""Regression tests for the findings of the 18-11-2026 external review that were confirmed and fixed."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.visualize import campaign2 as c2
from tests import synthetic_campaign2 as syn


@pytest.fixture(autouse=True)
def _scope():
    yield
    c2.use_tables("all")


def test_data_pipeline_restores_the_callers_output_group(monkeypatch):
    from scripts import data_pipeline
    monkeypatch.setenv("CREDITPFN_EXPERIMENT", "reference")

    def fake(**kwargs):
        os.environ["CREDITPFN_EXPERIMENT"] = "general"          # what activate_experiment did
        return 0
    monkeypatch.setattr(data_pipeline, "_run", fake)
    assert data_pipeline.run(datasets=["x"]) == 0
    assert os.environ["CREDITPFN_EXPERIMENT"] == "reference"


def test_group_for_run_reads_checkpoint_file_names():
    from src.utils.paths import group_for_run
    assert group_for_run("cpt2_reference_a0_s00_pd_tabpfn-v3_acc1_l2sp1.ckpt") == "reference"
    assert group_for_run("cpt2_update_size_anchoring_a3_s01") == "update_size_anchoring"
    assert group_for_run("cpt2e0_reference_pd") == "experiment0"
    assert group_for_run("unrelated_acc1.ckpt") == "general"


def test_would_have_stopped_counts_training_epochs_only():
    from src.train.loop import would_have_stopped
    # Epochs 0..4 (1-indexed positions 1..5): best at epoch 1 (position 2); patience 3 fires at 4.
    assert would_have_stopped([0.5, 0.4, 0.45, 0.46, 0.47], patience=3) == 2


def test_table_0009_removes_the_label_encoded_in_credit_score_scale():
    from src.data.preprocessing import _FIX_FUNCTIONS as DATASET_FIXES
    raw = pd.DataFrame({"Loan ID": list("abcd"), "Customer ID": list("abcd"), "Current Loan Amount": 1,
                        "Loan Status": ["Charged Off", "Fully Paid", "Charged Off", "Fully Paid"],
                        "Credit Score": [7300.0, 730.0, 6900.0, np.nan], "Term": "Short Term"})
    fix, = [fn for table, fn in DATASET_FIXES.items() if table.startswith("0009.")]
    out = fix(raw)
    assert out["Credit Score"].tolist()[:3] == [730.0, 730.0, 690.0] and out["Credit Score"].isna().iloc[3]
    assert (out["Credit Score"].dropna() <= 1000).all()


def test_primary_holm_family_is_the_declared_sixteen(monkeypatch):
    syn.install_results(monkeypatch, {"reference": 1}, missing_fold=False)
    full = c2.primary_endpoint()
    real = c2._read_results_frames
    keep_one = lambda slug, track: real(slug, track) if track == "pd" else pd.DataFrame()  # noqa: E731
    monkeypatch.setattr(c2, "_read_results_frames", keep_one)
    partial = c2.primary_endpoint()
    assert len(partial) == 16 and partial.missing.sum() == 8
    assert set(partial[partial.missing].decision) == {"missing"}
    pd_rows = partial[~partial.missing].set_index(["metric", "model"]).p_holm
    assert np.allclose(pd_rows.sort_index(), full[full.track == "pd"].set_index(["metric", "model"]).p_holm.sort_index()
                       .clip(lower=pd_rows.sort_index()))                # never smaller than with all 16


def test_the_newest_row_of_a_fold_decides(monkeypatch, tmp_path):
    from src.utils import paths
    root = tmp_path / "PD"
    root.mkdir()
    row = dict(model_name="m", test_dataset_id="t", fold_idx=0, model_source="x-untuned", roc_auc=0.7)
    pd.DataFrame([dict(row, status="OK")]).to_csv(root / "a_20261101.csv", index=False)
    pd.DataFrame([dict(row, status="FAILED")]).to_csv(root / "b_20261102.csv", index=False)
    monkeypatch.setattr(paths, "results_dir", lambda experiment: tmp_path)
    assert c2._read_results_frames("reference", "pd").empty


def test_public_scope_refuses_to_run_without_the_proprietary_mapping(monkeypatch):
    from src.data import dataset_names
    monkeypatch.setattr(dataset_names, "_PROPRIETARY", set())
    c2.use_tables("public")
    with pytest.raises(RuntimeError, match="_private_names"):
        c2.in_scope(pd.DataFrame({"test_dataset_id": ["0001.x"]}), "test_dataset_id")


def test_leaderboard_drops_a_table_missing_a_fold_for_every_method(monkeypatch):
    syn.install_results(monkeypatch, {"reference": 1})
    finetune = syn.finetune()
    table = syn.CREDIT["pd"][0]
    finetune = finetune[~((finetune.dataset_id == table) & (finetune.fold == 4))]
    real = c2._read_results_frames
    monkeypatch.setattr(c2, "_read_results_frames", lambda slug, track: (lambda f: f[~(
        (f.test_dataset_id == table) & (f.fold_idx == 4))])(real(slug, track)))
    out = c2.leaderboard("pd", finetune)
    assert table in out["dropped"] and table not in set(out["board"].pipe(lambda b: b)["method"])


def test_leaderboard_takes_untuned_models_from_e5_on_matched_context(monkeypatch):
    syn.install_results(monkeypatch, {"reference": 1})
    errors = c2.leaderboard_errors("pd", syn.finetune())
    e5 = syn.finetune()
    e5 = e5[(e5.track == "pd") & ~e5.finetuned & (e5.start == "original") & (e5.model == "v3")]
    mine = errors[errors.method == "TabPFN v3"].merge(e5, left_on=["table", "fold"], right_on=["dataset_id", "fold"])
    assert len(mine) and np.allclose(mine.error, 1 - mine.roc_auc)
    assert {"original", "CPT", "FT", "CPT+FT", "baseline"} == set(errors.variant)


def test_seed_noise_needs_all_three_seeds(monkeypatch):
    syn.install_results(monkeypatch, {"reference": 1, "seed": 1})        # E2 arm 1 (seed 44) absent
    assert c2.seed_noise("pd").empty


def test_transfer_keeps_only_runs_that_reached_the_budget(monkeypatch):
    frame = syn.transfer_trajectories("pd")
    short = frame.run.unique()[0]
    frame = frame[~((frame.run == short) & (frame.successful_updates == 2500))]
    monkeypatch.setattr(c2, "transfer_sources", lambda track: dict(enumerate(syn.CREDIT[track])))
    long = c2.transfer_long(frame, "pd")
    assert long.successful_updates.eq(2500).all()
    assert len(long.groupby(["model", "arm"])) == len(frame.groupby(["model", "arm"])) - 1


def test_full_context_with_fold_dependent_size_is_kept():
    rows = []
    for fold, full in enumerate([6129, 6130, 6129, 6130, 6129]):
        for size in (8, full):
            for start, auc in (("original", 0.70), ("cpt", 0.71)):
                rows.append(dict(track="pd", model="v2", dataset_id="t", fold=fold, context_rows=size,
                                 start=start, roc_auc=auc))
    table = c2.context_table(pd.DataFrame(rows))
    assert table.full.sum() == 1 and np.isclose(table[table.full].gain.iloc[0], 0.01)


def test_task_list_cache_round_trips_and_is_keyed(monkeypatch, tmp_path):
    from src.eval import campaign_tasks as ct
    listed = [dict(track="pd", model="v2", trial=0, partition=0, dataset_id="0010.x", n_rows=10, fold=0,
                   source="reference")]
    monkeypatch.setattr(ct, "tasks_cache", lambda kind: tmp_path / f"{kind}.json")
    monkeypatch.setattr(ct, "tasks", lambda design=None, references=None, kind="context": [dict(listed[0], fold=9)])
    ct.write_tasks_cache("finetune", listed)
    assert ct.campaign_tasks("finetune") == listed
    monkeypatch.setattr(ct, "_tasks_key", lambda: "changed")
    assert ct.campaign_tasks("finetune")[0]["fold"] == 9                    # stale cache is ignored


def test_task_results_ignore_metrics_without_a_receipt(monkeypatch, tmp_path):
    from src.utils import paths
    folder = tmp_path / "finetune"
    (folder / "completed").mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_csv(folder / "finetune_00000.csv.gz", index=False)
    pd.DataFrame({"a": [2]}).to_csv(folder / "finetune_00001.csv.gz", index=False)
    (folder / "completed" / "finetune_00001.json").write_text(json.dumps({}))
    monkeypatch.setattr(paths, "results_dir", lambda experiment: tmp_path)
    assert c2.task_results("finetune").a.tolist() == [2]
