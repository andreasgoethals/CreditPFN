"""Data separation, held-out scoring, output routing and release-gate contracts."""
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf


def test_experiment_outputs_use_both_tiers_without_cross_experiment_leaks(tmp_path, monkeypatch):
    from src.utils.paths import activate_experiment, logs_dir, manifests_dir, results_dir, training_dir, figures_dir
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(tmp_path / "project/CreditPFN"))
    for i in range(4):
        group = f"experiment{i}"
        activate_experiment(OmegaConf.create({"experiment": {"output_group": group}}))
        assert logs_dir() == tmp_path / "data/output CreditPFN" / group / "logs"
        assert manifests_dir() == tmp_path / "data/output CreditPFN" / group / "manifests"
        assert training_dir("pd") == tmp_path / "project/CreditPFN/output CreditPFN" / group / "training/pd"
        assert results_dir("PD") == tmp_path / "project/CreditPFN/output CreditPFN" / group / "results/PD"
        assert figures_dir(f"{group}/01_test") == tmp_path / "data/output CreditPFN" / group / "figures/01_test"


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
    cfg = OmegaConf.create(dict(track="pd", run_name="cpt_main_v5",
                                experiment=dict(output_group="experiment1")))
    with pytest.raises(StopBeforeTraining):
        train_pipeline.run(cfg=cfg)
    assert observed == [tmp_path / "output CreditPFN/experiment1/logs"]


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
    from src.data.retention import panel_config
    from src.data.preprocessing import DATASET_METADATA
    public = panel_config("research")
    assert len(public) == 8
    assert {d["id"] for d in public}.isdisjoint(DATASET_METADATA)
    assert sum(d["track"] == "pd" for d in public) == 4
    assert len({d["data_id"] for d in public}) == 8


def test_workflow_never_releases_on_partial_submission_or_failure(tmp_path, monkeypatch):
    from src.utils import experiment0 as flow
    submitted = []
    monkeypatch.setattr(flow, "_cpu", lambda *a: submitted.append(a) or "123")
    state = dict(id="test", phase="null", status="running", jobs=[], failed=[],
                 done=[f"{t}:{i}" for t in ("pd", "lgd") for i in range(8)], submissions_complete=False)
    path = tmp_path / "state.json"
    flow._advance(path, state)
    assert not submitted
    state.update(submissions_complete=True, failed=["pd:0"])
    flow._advance(path, state)
    assert not submitted
    state["failed"] = []
    flow._advance(path, state)
    flow._advance(path, state)
    assert len(submitted) == 1 and state["status"] == "auditing"


def test_recovery_only_workflow_prepares_no_null_or_pilot_trials(tmp_path, monkeypatch):
    from src.utils import experiment0 as flow, prepare_experiment, stage_inputs, preflight
    folder = tmp_path / "ab12cd34"
    folder.mkdir()
    state = {"part": "recovery"}

    @contextmanager
    def locked(identifier):
        assert identifier == folder.name
        yield folder / "state.json", state

    monkeypatch.setattr(flow, "locked", locked)
    monkeypatch.setattr(flow, "root", lambda: tmp_path)
    prepared, launched = [], []
    monkeypatch.setattr(prepare_experiment, "prepare", lambda p, **kw: prepared.append((p, kw)))
    monkeypatch.setattr(preflight, "main", lambda argv: 0)
    monkeypatch.setattr(stage_inputs, "stage", lambda *a, **kw: None)
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(tmp_path / "scratch"))
    monkeypatch.setattr(flow, "launch", lambda *a: launched.append(a))
    flow.prepare(folder.name)
    assert launched == [(folder.name, "recovery")]
    assert len(prepared) == 4 and all(kw == {"write": True} for p, kw in prepared)
    configs = [OmegaConf.load(p) for p, _ in prepared]
    assert all("_probe_ab12cd34_" in cfg.run_name and cfg.train.deterministic for cfg in configs)
    assert {cfg.track for cfg in configs} == {"pd", "lgd"}


def test_recovery_only_receipt_cannot_release_budget_pilots(tmp_path, monkeypatch):
    from src.utils import experiment0 as flow
    folder = tmp_path / "testflow"
    folder.mkdir()
    state = {"id": folder.name, "part": "recovery", "phase": "recovery", "status": "auditing",
             "fingerprint": "source"}

    @contextmanager
    def locked(identifier):
        yield folder / "state.json", state

    monkeypatch.setattr(flow, "locked", locked)
    monkeypatch.setattr(flow, "root", lambda: tmp_path)
    monkeypatch.setattr(flow, "fingerprint", lambda: "source")
    monkeypatch.setenv("VSC_DATA", str(tmp_path))
    monkeypatch.setattr(flow, "launch", lambda *a: pytest.fail("A diagnostic must stop here"))
    monkeypatch.setattr(flow, "_submit", lambda *a: pytest.fail("No later GPU stage may start"))
    for track in ("pd", "lgd"):
        for trial in range(4):
            (folder / f"recovery_{track}_{trial}.json").write_text(json.dumps({"passed": True}))
    flow.audit(folder.name, "recovery")
    assert state["status"] == "passed" and (tmp_path / "recovery_passed.json").is_file()
    assert not (tmp_path / "part1_passed.json").exists()
    with pytest.raises(RuntimeError, match="passing part-1 receipt"):
        flow.start("part2")


def test_recovery_check_detects_changed_weights(tmp_path):
    from src.utils.recovery_check import compare
    before, after = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    torch.save({"state_dict": {"w": torch.ones(3)}}, before)
    torch.save({"state_dict": {"w": torch.ones(3)}}, after)
    assert compare(before, after)["bitwise_equal"]
    torch.save({"state_dict": {"w": torch.zeros(3)}}, after)
    assert not compare(before, after)["passed"]


def test_recovery_differences_separate_loss_diagnostic_and_preserve_model_gate(tmp_path):
    from src.utils.recovery_check import compare, compact_report
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
    compact = compact_report(dict(report, different_tensors=[str(i) for i in range(500)]))
    assert "different_tensors" not in compact
    assert compact["different_tensor_count"] == 500
    assert len(compact["first_different_tensors"]) == 5


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
    report = compare_trajectories(before, after)
    assert not report["passed"]
    assert report["pre_interruption_equal"] == (step == 12)
    assert [r["successful_updates"] for r in report["by_update"] if not r["passed"]] == [step]
    assert compare_trajectories(before, before)["passed"]


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
    assert not compare_trajectories(before, after)["passed"]


def test_recovery_inspection_is_read_only_and_never_runs_models(tmp_path, monkeypatch, capsys):
    from src.utils import recovery_check, audit_experiment, experiment0
    folder = tmp_path / "workflow123"
    folder.mkdir()
    state_path = folder / "state.json"
    state_path.write_text('{"status": "failed"}')
    state_before = state_path.read_bytes()
    for arm in ("reference", "resumed"):
        torch.save({"state_dict": {"weight": torch.ones(2)}}, folder / f"{arm}.ckpt")
        pd.DataFrame({"successful_updates": [0, 5, 12], "metric__heldout": [.6, .7, .8]}
                     ).to_csv(folder / f"{arm}.trajectory.csv", index=False)

    def fake_audit(config):
        arm = config.stem.rsplit("_", 1)[1]
        return {"passed": False, "problems": ["other_trial: missing measurements"],
                "trials": [{"trial": arm, "path": str(folder / f"{arm}.ckpt"),
                            "status": "OK", "successful_updates": 12}]}

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only inspection must not train, run a benchmark, write a report or advance the workflow")

    monkeypatch.setattr(audit_experiment, "audit", fake_audit)
    monkeypatch.setattr(recovery_check, "training_dir", lambda track, name, **kw: folder / name)
    monkeypatch.setattr(experiment0, "root", lambda: tmp_path)
    monkeypatch.setattr(experiment0, "complete", forbidden)
    monkeypatch.setattr(recovery_check, "benchmark_smoke", forbidden)
    monkeypatch.setattr(recovery_check, "write_json", forbidden)
    monkeypatch.setattr(recovery_check.subprocess, "run", forbidden)
    assert recovery_check.main(["--inspect", "--id", folder.name, "--task", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["diagnostic_completed"] and not report["workflow_modified"]
    assert report["comparisons"][0]["arm_audits_passed"]
    assert report["comparisons"][0]["recorded_benchmark_smoke"] is None
    assert state_path.read_bytes() == state_before
    assert not (tmp_path / "part1_passed.json").exists()


def test_recovery_audit_ignores_pending_siblings_but_checks_selected_identity():
    from src.utils.recovery_check import selected_audit_problems
    audits = {"reference": {"problems": ["trial0: identity mismatch", "trial1: missing"]}}
    assert selected_audit_problems(audits, {"reference": {"trial": "trial0"}}) == {
        "reference": ["trial0: identity mismatch"]}


@pytest.mark.parametrize("family", ["tabpfn", "tabicl"])
@pytest.mark.parametrize("trained", [False, True])
def test_distribution_wrappers_use_explicit_levels_in_one_forward(family, trained):
    from src.model.tabpfn_models import TabPFNUntuned, TabPFNTrained
    from src.model.tabicl_models import TabICLUntuned, TabICLTrained
    from src.eval.benchmark import PRED_QUANTILE_LEVELS
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
    from src.eval.benchmark import _predict_quantiles, PRED_QUANTILE_LEVELS
    class Native:
        def predict(self, X, output_type="mean", alphas=None):
            assert alphas == list(PRED_QUANTILE_LEVELS)
            return np.tile(alphas, (len(X), 1))
    result = _predict_quantiles(Native(), np.ones((3, 2)))
    np.testing.assert_array_equal(result[0], PRED_QUANTILE_LEVELS)


def test_monitor_keeps_complementary_metrics(monkeypatch):
    from types import SimpleNamespace as NS
    import src.train.dataloader as data
    import src.model.tabpfn_models as models
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


def test_consolidation_separates_project_diagnostics_from_data_manifests(tmp_path):
    from src.utils.consolidate_output import consolidate, load_consolidated
    run = "cpt_main_v5"
    stem = run + "_s00_pd_base_lr1e-6_seed42"
    training, manifests, results, dest = [tmp_path / part for part in ("training", "manifests", "results", "consolidated")]
    (training / "pd").mkdir(parents=True)
    pd.DataFrame({"successful_updates": [0, 5]}).to_csv(training / "pd" / (stem + ".trajectory.csv"), index=False)
    pd.DataFrame({"successful_updates": [0, 5], "parameter": ["w", "w"]}).to_csv(
        training / "pd" / (stem + ".parameters.csv.gz"), index=False)
    pd.DataFrame({"gpu_status": ["sampled"]}).to_csv(training / "pd" / (stem + ".resources.csv"), index=False)
    kwargs = dict(manifest_root=manifests, result_root=results, training_root=training)
    report = consolidate(run, apply=True, destination=dest, **kwargs)
    assert report["rows"]["training_pd"] == report["rows"]["parameters_pd"] == 2
    assert report["rows"]["resources_pd"] == 1
    frame = load_consolidated(run, "parameters_pd", snapshot_root=dest, **kwargs)
    assert frame.trial_name.tolist() == [stem, stem]
    assert "gpu_status" not in frame


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_gpu_canary_also_gates_five_fold_benchmark_outputs(tmp_path, monkeypatch, track):
    from src.utils import recovery_check
    from src.data import retention
    from src.eval.dataset_loader import ProcessedDataset
    import src.model.tabpfn_models as models
    class Model:
        name = "toy"
        def __init__(self, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            pass
        def predict_proba(self, X):
            p = np.where(X[:, 0] % 2, .8, .2)
            return np.column_stack([1-p, p])
        def predict_distribution(self, X, y, levels):
            return dict(mean=y, quantiles=y[:, None] + np.asarray(levels)[None, :] - .5)
    ds = ProcessedDataset(pd.DataFrame({"x": np.arange(60)}),
        np.arange(60) % 2 if track == "pd" else np.arange(60, dtype=float), [],
        "classification" if track == "pd" else "regression", "synthetic", track)
    monkeypatch.setattr(models, "TabPFNTrained", Model)
    monkeypatch.setattr(retention, "processed_dataset", lambda *args: ds)
    monkeypatch.setattr(recovery_check, "results_dir", lambda *parts, **kw: tmp_path.joinpath(*parts))
    report = recovery_check.benchmark_smoke(Path("tabpfn-v3.ckpt"), track, 2, workflow="first")
    assert report["passed"] and report["predictions"] == 60
    first = tmp_path / track.upper() / "recovery_smoke/first/base_2.csv"
    assert first.is_file()
    before = first.read_bytes()
    recovery_check.benchmark_smoke(Path("tabpfn-v3.ckpt"), track, 2, workflow="second")
    assert (first.parent.parent / "second/base_2.csv").is_file()
    assert first.read_bytes() == before
