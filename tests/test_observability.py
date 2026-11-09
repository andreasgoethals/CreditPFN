"""Data separation, held-out scoring, output routing and release-gate contracts."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf


def test_experiment_outputs_use_both_tiers_without_cross_experiment_leaks(tmp_path, monkeypatch):
    from src.utils.paths import (
        activate_experiment,
        logs_dir,
        manifests_dir,
        results_dir,
        training_dir,
    )
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(tmp_path / "project/CreditPFN"))
    for group in ("experiment0", "reference", "seed", "transfer"):
        activate_experiment(OmegaConf.create({"experiment": {"output_group": group}}))
        assert logs_dir() == tmp_path / "data/output CreditPFN" / group / "logs"
        assert manifests_dir() == tmp_path / "data/output CreditPFN" / group / "manifests"
        assert training_dir("pd") == tmp_path / "project/CreditPFN/output CreditPFN" / group / "training/pd"
        assert results_dir("PD") == tmp_path / "project/CreditPFN/output CreditPFN" / group / "results/PD"


def test_training_entry_selects_experiment_before_opening_its_first_log(tmp_path, monkeypatch):
    from scripts import train_pipeline
    from src.utils.paths import logs_dir
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_EXPERIMENT", "experiment0")
    observed = []
    class StopBeforeTraining(Exception):
        pass
    def log(*args, **kwargs):
        observed.append(logs_dir())
        raise StopBeforeTraining
    monkeypatch.setattr(train_pipeline, "resolve_run_log", log)
    cfg = OmegaConf.create(dict(track="pd", run_name="cpt2_reference_a0",
                                experiment=dict(output_group="reference")))
    with pytest.raises(StopBeforeTraining):
        train_pipeline.run(cfg=cfg)
    assert observed == [tmp_path / "output CreditPFN/reference/logs"]


def test_recovery_children_keep_the_shell_log_path(tmp_path, monkeypatch):
    from scripts import train_pipeline
    from src.train import loop
    from src.train.config import load_train_config

    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(tmp_path))
    log = tmp_path / "output CreditPFN/experiment0/logs/recovery_123_r0.log"
    log.parent.mkdir(parents=True)
    log.write_text("START task=recovery\n")
    monkeypatch.setenv("CREDITPFN_ACTIVE_LOG", str(log))
    cfg = load_train_config(config_path="config/experiments/E00_pilots/reference_pd.yaml")
    cfg.device = "cpu"
    cfg.experiment.fingerprint = cfg.experiment.require_plan = False
    cfg.train.resource_diagnostics = False
    monkeypatch.setattr(train_pipeline, "setup_logging", lambda *a: None)
    monkeypatch.setattr(train_pipeline, "_ensure_processed", lambda *a, **kw: None)
    monkeypatch.setattr(train_pipeline, "_run_provenance", lambda *a: {})

    def failed_trial(*a, **kw):
        raise RuntimeError("synthetic child failure; no model loaded")

    monkeypatch.setattr(loop, "train_one_config", failed_trial)
    # Three invocations share the parent's path, as in a recovery pair.
    for _ in range(3):
        assert train_pipeline.run(cfg=cfg, trial_index=0, log_path=log) == 1
    assert list(log.parent.iterdir()) == [log]
    assert log.read_text().startswith("START task=recovery\n")
    assert log.read_text().count("train_pipeline: status=FAIL") == 3


def test_monitor_splits_are_fixed_disjoint_and_shuffled_for_small_tables():
    from src.train.monitoring import monitor_indices
    y = np.repeat([0, 1], 50)
    parts = monitor_indices(y, "classification", .4, 12)
    assert len(np.concatenate(parts)) == len(y)
    assert len(set(np.concatenate(parts))) == len(y)
    assert all(np.unique(y[index]).size == 2 for index in parts)
    assert all(np.array_equal(a, b) for a, b in zip(parts, monitor_indices(y, "classification", .4, 12)))
    assert not np.array_equal(parts[2], np.arange(60, 100))


def test_threshold_is_independent_of_test_labels_and_bins_preserve_counts():
    from src.eval.metrics import _classification_metrics
    p = np.array([.1, .2, .4, .8, .9, .95])
    probs = np.column_stack([1-p, p])
    val_y = np.array([0, 0, 1, 1, 1, 1])
    a = _classification_metrics(probs, val_y, probs, val_y, 2)
    b = _classification_metrics(probs, 1-val_y, probs, val_y, 2)
    assert a["optimal_threshold"] == b["optimal_threshold"] == .4
    assert sum(r["count"] for r in json.loads(a["calibration_bins"])) == 6
    assert sum(r["positive_count"] for r in json.loads(a["calibration_bins"])) == 4
    assert a["true_positive"] == 4 and a["false_positive"] == 0
    assert a["f1_at_05"] < a["f1"]


def test_platt_auc_is_measured_when_validation_slope_is_negative():
    from src.eval.metrics import _classification_metrics
    p = np.array([.1, .2, .8, .9])
    probs = np.column_stack([1-p, p])
    y = np.array([1, 1, 0, 0])
    metrics = _classification_metrics(probs, y, probs, y, 2)
    assert metrics["roc_auc"] == 0
    assert metrics["roc_auc_platt"] == 1


def test_five_folds_persist_each_original_row_once_and_classification_predictions():
    from src.eval.benchmark import _bench_model_on_dataset
    from src.eval.dataset_loader import ProcessedDataset
    from src.model.base import ModelHandle
    class Classifier:
        def fit(self, *args, **kwargs):
            pass
        def predict_proba(self, X):
            p = np.where(X[:, 0] % 2, .8, .2)
            return np.column_stack([1-p, p])
    ds = ProcessedDataset(pd.DataFrame({"index": np.arange(50)}), np.arange(50) % 2,
                          [], "classification", "test", "pd")
    predictions = []
    handle = ModelHandle(name="dummy", source="baseline", base_path=None, track="pd", task_type="classification")
    rows = _bench_model_on_dataset(handle=handle, model=Classifier(), ds=ds,
        n_folds=5, inner_val_fraction=.2, seed=99, timestamp="test", pred_records=predictions)
    assert len(rows) == 5 and all(r.status == "OK" for r in rows)
    assert sorted(r["row_idx"] for r in predictions) == list(range(50))
    assert all(r.f1 == 1 and r.calibration_bins != "[]" for r in rows)


def test_regression_intervals_are_scored_even_without_raw_prediction_output():
    from src.eval.metrics import _regression_metrics
    y = np.array([1., 2., 3.])
    q = np.column_stack([y-2, y-1, y+1, y+2])
    result = _regression_metrics(y, y, neg_nll=None, quantiles=q, quantile_levels=(.05,.1,.9,.95))
    assert result["interval_coverage_80"] == result["interval_coverage_90"] == 1
    assert result["interval_width_80"] == 2
    assert result["quantile_crossing_fraction"] == 0


def test_weight_measurements_do_not_consume_rng_or_invent_frozen_drift():
    from src.train.telemetry import parameter_statistics
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    anchor = {"weight": model.weight.detach().clone()}
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        model.weight.add_(.1)
    rows = {r["parameter"]: r for r in parameter_statistics(model, anchor)}
    assert torch.equal(rng, torch.get_rng_state())
    assert rows["weight"]["absolute_change"] == pytest.approx(np.sqrt(6) * .1)
    assert not rows["bias"]["anchored"] and np.isnan(rows["bias"]["absolute_change"])


def test_public_panel_cannot_become_credit_training_data():
    from src.data.preprocessing import DATASET_METADATA
    from src.data.retention import panel_config
    public = panel_config("research")
    assert len(public) == 31
    assert {d["id"] for d in public}.isdisjoint(DATASET_METADATA)
    assert sum(d["track"] == "pd" for d in public) == 19
    assert len({d["data_id"] for d in public}) == 31


def test_recovery_check_detects_changed_weights(tmp_path):
    from src.utils.recovery_check import compare
    before, after = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    torch.save({"state_dict": {"w": torch.ones(3)}}, before)
    torch.save({"state_dict": {"w": torch.ones(3)}}, after)
    assert compare(before, after)["bitwise_equal"]
    torch.save({"state_dict": {"w": torch.zeros(3)}}, after)
    assert not compare(before, after)["passed"]


def test_recovery_differences_separate_loss_diagnostic_and_preserve_model_gate(tmp_path):
    from src.utils.recovery_check import compare
    before, after = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    state = {"weight": torch.ones(3), "criterion.borders": torch.arange(3.),
             "criterion.losses_per_bucket": torch.zeros(2)}
    torch.save({"state_dict": state}, before)
    state["criterion.losses_per_bucket"] += .1
    torch.save({"state_dict": state}, after)
    report = compare(before, after)
    assert report["passed"]  # Detached loss diagnostics are not inference/training parameters.
    assert not report["all_saved_state_close"] and not report["bitwise_equal"]
    assert report["diagnostic_buffers_different"] == ["criterion.losses_per_bucket"]
    assert report["max_non_diagnostic_difference"] == 0
    assert report["diagnostic_buffer_differences"]["criterion.losses_per_bucket"] > .09
    state["criterion.borders"] += .001
    torch.save({"state_dict": state}, after)
    report = compare(before, after)
    assert not report["passed"]  # Inference criterion borders must still be checked.
    assert report["max_non_diagnostic_tensor"] == "criterion.borders"
    assert report["max_non_diagnostic_difference"] > .0009


@pytest.mark.parametrize("change", ["shape", "dtype", "nonfinite", "missing", "added"])
def test_recovery_state_comparison_reports_invalid_tensors(tmp_path, change):
    from src.utils.recovery_check import compare
    before, after = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    torch.save({"state_dict": {"w": torch.ones(3)}}, before)
    state = {"w": torch.ones(3)}
    if change == "shape":
        state["w"] = torch.ones(2)
    elif change == "dtype":
        state["w"] = state["w"].double()
    elif change == "nonfinite":
        state["w"][0] = float("nan")
    elif change == "missing":
        state = {}
    else:
        state["added"] = torch.zeros(1)
    torch.save({"state_dict": state}, after)
    report = compare(before, after)
    assert not report["passed"] and not report["bitwise_equal"]


@pytest.mark.parametrize("step", [0, 5, 12])
def test_recovery_trajectory_identifies_when_runs_already_differ(step):
    from src.utils.recovery_check import compare_trajectories
    before = pd.DataFrame({"successful_updates": [0, 5, 12],
                           "metric__heldout": [.6, .7, .8], "score__optional": [np.nan] * 3})
    after = before.copy()
    after.loc[after.successful_updates == step, "metric__heldout"] += .001
    report = compare_trajectories(before, after, milestones=[0, 5, 12])
    assert not report["passed"]
    assert [r["successful_updates"] for r in report["by_update"] if not r["passed"]] == [step]
    assert compare_trajectories(before, before, milestones=[0, 5, 12])["passed"]


@pytest.mark.parametrize("change", ["missing_step", "duplicate_step", "missing_metric", "nan", "inf"])
def test_recovery_trajectory_rejects_incomplete_or_nonfinite_primary_metrics(change):
    from src.utils.recovery_check import compare_trajectories
    before = pd.DataFrame({"successful_updates": [0, 5, 12], "metric__heldout": [.6, .7, .8]})
    after = before.copy()
    if change == "missing_step":
        after = after.iloc[:2]
    elif change == "duplicate_step":
        after.loc[2, "successful_updates"] = 5
    elif change == "missing_metric":
        after = after.drop(columns="metric__heldout")
    else:
        before.loc[0, "metric__heldout"] = after.loc[0, "metric__heldout"] = float(change)
    assert not compare_trajectories(before, after, milestones=[0, 5, 12])["passed"]


@pytest.mark.parametrize("family", ["tabpfn", "tabicl"])
@pytest.mark.parametrize("trained", [False, True])
def test_distribution_wrappers_use_explicit_levels_in_one_forward(family, trained):
    from src.eval.benchmark import PRED_QUANTILE_LEVELS
    from src.model.tabicl_models import TabICLTrained, TabICLUntuned
    from src.model.tabpfn_models import TabPFNTrained, TabPFNUntuned
    levels = list(PRED_QUANTILE_LEVELS)
    n = len(levels)  # Square matrices must not hide a quantile/row transpose.
    expected = np.arange(n)[:, None] + np.asarray(levels)[None, :]
    calls = []

    class Native:
        def predict(self, X, **kwargs):
            calls.append(kwargs)
            if family == "tabpfn":
                assert kwargs == dict(output_type="full", quantiles=levels)
                return dict(mean=np.arange(n), quantiles=list(expected.T),
                            criterion=lambda logits, y: torch.ones(len(y)), logits=torch.ones(n, 2))
            assert kwargs == dict(output_type=["mean", "quantiles"], alphas=levels)
            return dict(mean=np.arange(n), quantiles=expected)

    cls = {(False, "tabpfn"): TabPFNUntuned, (True, "tabpfn"): TabPFNTrained,
           (False, "tabicl"): TabICLUntuned, (True, "tabicl"): TabICLTrained}[trained, family]
    model = cls(task_type="regression", **{"ckpt_path" if trained else "base_path": "unused.ckpt"})
    setattr(model, "_" + family, Native())
    result = model.predict_distribution(np.ones((n, 2)), np.arange(n), levels)
    np.testing.assert_array_equal(result["quantiles"], expected)
    assert len(calls) == 1
    assert result["neg_nll"] == (-1. if family == "tabpfn" else None)


def test_generic_quantiles_never_assume_default_probability_levels():
    from src.eval.benchmark import PRED_QUANTILE_LEVELS, _predict_quantiles
    class Native:
        def predict(self, X, output_type="mean", alphas=None):
            assert alphas == list(PRED_QUANTILE_LEVELS)
            return np.tile(alphas, (len(X), 1))
    result = _predict_quantiles(Native(), np.ones((3, 2)))
    np.testing.assert_array_equal(result[0], PRED_QUANTILE_LEVELS)


def test_monitor_keeps_complementary_metrics(monkeypatch):
    from types import SimpleNamespace as NS

    import src.model.tabpfn_models as models
    import src.train.dataloader as data
    from src.train.loop import evaluate_ensemble_on_split
    class Native:
        def fit(self, X, y):
            pass
        def predict_proba(self, X):
            p = np.where(X[:, 0] % 2, .8, .2)
            return np.column_stack([1-p, p])
    monkeypatch.setattr(models, "_make_tabpfn", lambda *a, **kw: Native())
    monkeypatch.setattr(data, "_load_processed_csv", lambda ref: NS(
        X=pd.DataFrame({"x": np.arange(100)}), y=np.arange(100) % 2, cat_columns=[]))
    metrics = evaluate_ensemble_on_split(ckpt_path="unused", refs=[NS(dataset_id="toy")],
        n_estimators=1, n_subsample=100, query_fraction=.4, seed=42, device="cpu",
        task_type="classification", metric_names=("roc_auc",), family="tabpfn")
    assert metrics["roc_auc"] == metrics["f1"] == 1
    assert metrics["brier_score"] == pytest.approx(.04)
    assert "log_loss_isotonic" in metrics
