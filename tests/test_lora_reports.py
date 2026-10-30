"""Report tests use small synthetic records, never scientific campaign output."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.train.config import load_train_config
from src.visualize import campaign as cp, lora_study as ls, style
from src.visualize.reporting import NotebookReport, page_text


def lora_fixture(track="pd", cfg=None):
    cfg = cfg if cfg is not None else load_train_config(config_path=f"config/experiment8/{track}.yaml")
    trials = cp.planned_trials(cfg).assign(status="OK", n_train_datasets=2, n_test_datasets=1)
    trials["gpu_hours"] = 1 + trials.learning_rate * 100
    trials["peak_gpu_gb"] = 20 + trials.partition
    trials["trainable_params"] = 4000
    trials["total_params"] = 1000000
    histories, trajectories, evaluation = [], [], []
    rates = sorted(trials.learning_rate.unique())
    for trial in trials.itertuples():
        effect = [.012, .006, -.015][rates.index(trial.learning_rate)]
        for update in (0, 250, 1000, 2500, 5000, 10000):
            fraction = update / 10000
            trajectories.append(dict(trial_name=trial.trial_name, successful_updates=update,
                weight_drift=trial.learning_rate * fraction,
                **{f"score__test__Table {trial.partition}__{metric}": .7 + trial.partition*.01 + effect*fraction
                   for metric in ("roc_auc", "r2")},
                **{f"score__ood__Panel__{metric}": .7 - effect*fraction for metric in ("roc_auc", "r2")}))
            if update:
                histories.append(dict(trial_name=trial.trial_name, successful_updates=update,
                    train_loss=.5/(1+fraction), grad_norm_mean=trial.learning_rate+1,
                    clipped_frac=0., amp_skipped_steps=0, data_skipped_steps=0,
                    elapsed_sec=update*.4, lr_applied=trial.learning_rate*(1-fraction),
                    observed_tables=2, full_observed_tables=2, complete_pass=True, partial_epoch=False))
        short = f"{trial.base}-default" if "tabpfn" in trial.trial_name else "tabicl-v2"
        family = "tabpfn" if "tabpfn" in trial.trial_name else "tabicl"
        for domain, dataset in (("credit", f"Table {trial.partition}"), ("ood", "Panel")):
            delta = effect if domain == "credit" else -effect
            for fold in range(5):
                original = dict(roc_auc=.65+trial.partition*.01, r2=.3+trial.partition*.01,
                    brier_score=.2, log_loss=.5, rmse=.6, mae=.4)
                record = dict(method_dirname=f"{family}-trained__{short}__lr{trial.learning_rate:g}__l2sp0__lora",
                    source=family+"-trained", adaptation_mode="lora", base_short=short,
                    test_dataset_id=dataset, split=trial.partition, eval_run=f"fixture_s{trial.partition:02d}",
                    fold_idx=fold, status="OK", domain=domain,
                    **{metric: value+(delta if metric in ("roc_auc", "r2") else -delta)
                       for metric, value in original.items()})
                evaluation.append(record)
                if trial.learning_rate == rates[0]:
                    evaluation.append(dict(record, **original, method_dirname=f"{family}-untuned__{short}",
                        source=family+"-untuned", adaptation_mode="full"))
    return cp.Campaign(cfg, trials, pd.DataFrame(trajectories), pd.DataFrame(histories), pd.DataFrame(evaluation))


def test_pending_lora_campaigns_have_explicit_counts_and_no_empty_figures(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDITPFN_ANALYSIS_ROOT", str(tmp_path))
    monkeypatch.delenv("CREDITPFN_ANALYSIS_PROJECT_ROOT", raising=False)
    runs = [cp.load_campaign(8, track) for track in ("pd", "lgd")]
    assert sum(len(run.trials) for run in runs) == 48
    for run in runs:
        coverage = ls.coverage_table(run)
        assert coverage.planned.sum() == 24 and coverage.completed.sum() == 0
        assert coverage.evidence.eq("No recorded training outcome").all()
        assert "16" in ls.protocol_table(run).to_string()
        assert ls.plot_optimization(run) == [] and ls.plot_monitors(run) == []
        assert ls.plot_primary(run) == [] and ls.plot_secondary(run) == []
    assert ls.plot_schedule(runs) == [] and ls.plot_costs(runs) == []


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_lora_original_pairing_and_primary_means(track):
    run = lora_fixture(track)
    metric = "roc_auc" if track == "pd" else "r2"
    data = ls.matched_tables(run, [metric])
    assert len(data) == 24 and data.partitions.eq(1).all()
    assert np.allclose(data.loc[data.learning_rate.eq(1e-5), "effect"], .012)
    summary = ls.score_summary(data)
    assert np.allclose(summary.original, .665 if track == "pd" else .315)
    assert np.allclose(summary.lora-summary.original, summary.effect)
    assert summary.datasets.eq(4).all()
    retention = ls.matched_tables(run, [metric], "ood")
    assert len(retention) == 6 and retention.partitions.eq(4).all()
    assert np.allclose(retention.loc[retention.learning_rate.eq(1e-5), "effect"], -.012)


def test_lora_missing_fold_drops_common_rate_cohort_without_hiding_other_model():
    run = lora_fixture()
    missing = (run.evaluation.domain.eq("ood") & run.evaluation.split.eq(0) &
               run.evaluation.method_dirname.str.contains("tabicl-trained.*lr0.0001") & run.evaluation.fold_idx.eq(4))
    run.evaluation = run.evaluation[~missing]
    retention = ls.matched_tables(run, ["roc_auc"], "ood")
    assert retention.base.unique().tolist() == ["v3"]
    assert len(ls.matched_tables(run, ["roc_auc"])) == 24
    # One nonfinite secondary metric excludes that table from BOTH secondary panels and every rate.
    run.evaluation.loc[(run.evaluation.test_dataset_id.eq("Table 0") &
        run.evaluation.source.eq("tabicl-trained") & run.evaluation.fold_idx.eq(0)), "log_loss"] = np.nan
    secondary = ls.matched_tables(run, ["brier_score", "log_loss"])
    assert set(secondary.loc[secondary.base.eq("tabicl-v2"), "dataset"]) == {"Table 1", "Table 2", "Table 3"}
    assert secondary.groupby(["base", "metric"]).dataset.nunique().to_dict() == {
        ("tabicl-v2", "brier_score"): 3, ("tabicl-v2", "log_loss"): 3,
        ("v3", "brier_score"): 4, ("v3", "log_loss"): 4}


def test_lora_report_rejects_mixed_modes_and_incomplete_originals():
    run = lora_fixture()
    run.evaluation.loc[run.evaluation.source.str.endswith("-trained"), "adaptation_mode"] = "frozen_backbone"
    with pytest.raises(ValueError, match="pool other adaptation modes"):
        ls.plot_primary(run)
    run = lora_fixture()
    run.evaluation = run.evaluation[~(run.evaluation.source.str.endswith("-untuned") & run.evaluation.fold_idx.eq(4))]
    assert ls.plot_primary(run) == [] and ls.plot_secondary(run) == []


def test_missing_training_partition_is_a_gap_and_effective_movement_is_named():
    run = lora_fixture()
    missing_trial = run.trials.loc[run.trials.learning_rate.eq(1e-5) & run.trials.partition.eq(0), "trial_name"].iloc[0]
    run.histories = run.histories[~run.histories.trial_name.eq(missing_trial)]
    pages = ls.plot_optimization(run)
    assert len(pages) == 2
    base = run.trials.set_index("trial_name").loc[missing_trial, "base"]
    page = next(page for page in pages if page.name.endswith(base))
    table = page.tables["Displayed curves and partition support"]
    assert table.loc[table.learning_rate.eq(1e-5) & table.metric.eq("train_loss"), "value"].isna().all()
    assert "initial targeted query/value weight norm" in page.caption
    assert "zero-initialized adapter factors" in page.caption


def test_update_zero_alone_is_not_a_training_result_figure():
    run = lora_fixture()
    run.trajectories = run.trajectories[run.trajectories.successful_updates.eq(0)]
    assert ls.plot_monitors(run) == []


def test_protocol_audit_exposes_precision_and_rate_mismatch():
    run = lora_fixture()
    cfg = load_train_config(config_path="config/experiment1/pd.yaml")
    reference = cp.Campaign(cfg, cp.planned_trials(cfg), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    table = ls.reference_protocol_audit(run, reference).set_index("check")
    # Experiment 1 re-scores TabICLv2 in FP32 without FA3, as experiment 8 does.
    assert table.loc["evaluation.tabicl_inference_options", "matches"]
    assert not table.loc["Candidate shared rates with L2-SP zero", "matches"]
    assert table.loc["Candidate shared rates with L2-SP zero", "experiment1"] == "1e-05"
    assert "Unavailable" in table.loc["Direct cross-experiment comparison", "experiment8"]


def test_native_lora_cannot_be_mislabelled_by_legacy_reports(monkeypatch):
    from src.visualize import training_viz as tv, trajectories, optimization
    run = lora_fixture()
    name = run.trials.trial_name.iloc[0]
    trial = tv.parse_trial_name(name)
    assert trial.adaptation_mode == "lora" and not trial.lora
    assert "LoRA" in tv._style_for(trial, {}, set())["label"]
    legacy = tv.parse_trial_name(name.replace("cpt_lora_v1", "cpt_main_v5"))
    assert legacy.adaptation_mode == "legacy_adapter" and legacy.lora
    recorded = pd.DataFrame({"trial_name": [name, name], "successful_updates": [0, 2],
                             "processed_rows": [0, 100], "metric__test__example": [.7, .71]})
    effects = trajectories.trajectory_effects(recorded, "pd")
    assert effects.adaptation_mode.eq("lora").all() and not effects.frozen.any()
    monkeypatch.setattr(trajectories, "load_trajectories", lambda *args: recorded)
    with pytest.raises(ValueError, match="adapter report"):
        trajectories.plot_trajectories("pd")
    with pytest.raises(ValueError, match="raw factor statistics"):
        optimization.plot_movement(run)
    protocol = ls.protocol_table(run).set_index("Setting").Value
    assert protocol["Factor weight decay"] == 0
    assert "BF16" in protocol["Training precision"]
    assert "not per-layer" in protocol["Per-tensor diagnostics"]


def test_lora_figures_preserve_values_geometry_and_native_losses(tmp_path):
    style.apply()
    runs = [lora_fixture(track) for track in ("pd", "lgd")]
    pages = ls.plot_schedule(runs) + ls.plot_costs(runs)
    schedule = pages[0].figure
    assert len(schedule.legends) == 1 and schedule.axes[0].get_legend() is None
    assert [line.get_color() for line in schedule.legends[0].get_lines()] == list(style.LORA_LR_COLORS.values())
    for run in runs:
        pages += ls.plot_optimization(run) + ls.plot_monitors(run) + ls.plot_primary(run) + ls.plot_secondary(run)
    assert len(pages) == 12
    for index, page in enumerate(pages):
        assert page.tables and page.caption and page_text(page)
        assert page.figure.get_figwidth() == style.WIDTH_FULL
        assert page.figure.get_figheight() <= style.MAX_HEIGHT
        page.figure.savefig(tmp_path/f"{index:02}_{page.name}.pdf")
        page.figure.canvas.draw()
        assert not any("All parameters" in axis.get_title() or "DiagScale" in axis.get_title() for axis in page.figure.axes)
        report = NotebookReport("Synthetic LoRA report")
        report.add("Evidence", page_text(page))
        assert page.caption in report.summary(SimpleNamespace(summary=lambda: "Synthetic figure inventory"))
    labels = [axis.get_ylabel() for page in pages for axis in page.figure.axes]
    assert "Query cross-entropy" in labels and "Bar-distribution NLL" in labels and "Mean quantile pinball loss" in labels
    cost = next(page for page in pages if page.name == "lora_training_cost")
    assert len(cost.tables["Completed-trial source measurements"]) == 48
    assert cost.tables["Displayed cost summaries"].completed_trials.eq(4).all()


def test_lora_notebooks_are_thin_and_end_with_complete_summary():
    for stem in ("01_training", "02_results"):
        path = Path("notebooks/rq2") / (stem + ".ipynb")
        notebook = json.loads(path.read_text(encoding="utf8"))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code[-1].strip() == "print(report.summary(sink))"
        assert f"FigureSaver('rq2/{stem}')" in "\n".join(code)
        for source in code:
            tree = ast.parse("\n".join(line for line in source.splitlines() if not line.startswith("%")))
            assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
