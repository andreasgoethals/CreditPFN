"""Compact training reports retain paired cohorts and one-trial cost weights."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualize import campaign as cp, training_summary as ts, style
from tests.test_followup_comparisons import future_campaign


def test_monitor_choice_summary_requires_unchanged_table_support(monkeypatch):
    run = future_campaign(1)
    trials = run.trials
    rows = []
    for trial in trials.itertuples():
        for updates in (0, run.target):
            for dataset in ("A", "B"):
                rows.append(dict(trial=trial.trial_name, **{f: getattr(trial, f) for f in cp.FACTORS},
                    partition=trial.partition, seed=trial.seed, dataset=dataset, updates=updates,
                    effect=(.01 * bool(trial.l2sp_lambda) + .02 * trial.frozen) * bool(updates)))
    observations = pd.DataFrame(rows)
    monkeypatch.setattr(ts.db, "monitor_observations", lambda run, split: observations.copy())
    curves, tables = ts.monitor_choice_tables(run)
    target = curves[curves.updates.eq(run.target)]
    assert target.complete.all() and target.tables.eq(2).all()
    np.testing.assert_allclose(target.loc[target.factor.eq("l2sp_lambda"), "difference"], .01)
    np.testing.assert_allclose(target.loc[target.factor.eq("frozen"), "difference"], .02)
    assert not tables.empty
    # Equal counts do not authorize swapping B for C at the final milestone.
    observations.loc[(observations.updates.eq(run.target)) & observations.dataset.eq("B"), "dataset"] = "C"
    curves, tables = ts.monitor_choice_tables(run)
    target = curves[curves.updates.eq(run.target)]
    assert not target.complete.any() and target.difference.isna().all()
    assert tables.empty


def test_missing_planned_trial_cannot_appear_as_complete_monitor_effect(monkeypatch):
    run = future_campaign(1)
    original = ts.db.monitor_observations
    data = original(run)
    data = data[~data.trial.eq(run.trials.iloc[0].trial_name)]
    monkeypatch.setattr(ts.db, "monitor_observations", lambda run, split: data.copy())
    curves, _ = ts.monitor_choice_tables(run)
    affected = curves[curves.factor.isin(["l2sp_lambda", "frozen"])]
    assert not affected.complete.any()


def test_movement_requires_all_planned_trials_at_each_recorded_milestone(monkeypatch):
    run = future_campaign(2)
    rows = pd.DataFrame([
        dict(trial_name=trial.trial_name, base=trial.base, frozen=trial.frozen,
             window_end=update, weight_drift=.01 * (1+trial.partition))
        for trial in run.trials.itertuples() for update in (1000, run.target)])
    monkeypatch.setattr(ts.cp, "optimization_curves", lambda run, field: rows)
    monkeypatch.setattr(ts.dg, "load", lambda run, kind: pd.DataFrame())
    data = ts.movement_tables(run)
    assert data.complete.all()
    rows.drop(rows[(rows.trial_name.eq(run.trials.iloc[0].trial_name)) & rows.window_end.eq(1000)].index, inplace=True)
    data = ts.movement_tables(run)
    assert data.loc[data.updates.eq(1000), "median"].isna().all()
    assert data.loc[data.updates.eq(run.target), "complete"].all()


def test_cost_summary_weights_each_completed_trial_once():
    run = SimpleNamespace(trials=pd.DataFrame([
        dict(trial_name="short", base="v3", frozen=False, status="OK", gpu_hours=1., peak_gpu_gb=10.),
        dict(trial_name="long", base="v3", frozen=False, status="OK", gpu_hours=9., peak_gpu_gb=20.),
        dict(trial_name="unfinished", base="v3", frozen=False, status="INTERRUPTED", gpu_hours=100., peak_gpu_gb=99.)]))
    data, source = ts.resource_tables(run)
    assert len(source) == 2 and set(source.trial_name) == {"short", "long"}
    assert data.trials.eq(2).all() and data.expected_trials.eq(3).all()
    assert data.set_index("measure").loc["gpu_hours", "median"] == 5.
    assert data.set_index("measure").loc["peak_gpu_gb", "median"] == 15.


def test_compact_choice_page_keeps_native_dynamics_and_monitor_identity():
    run = future_campaign(1, all_bases=True)
    style.apply()
    page, = ts.plot_choices(run)
    assert page.provenance["measurement"] == "fixed_context_monitor"
    assert "Native-scale paired optimization dynamics" in page.tables
    assert len(page.figure.axes) <= 3  # Credit, retention, one colorbar.
    assert "five-fold final benchmark" in page.caption
    assert "all-parameter updates" in page.caption
    labels = [label.get_text() for axis in page.figure.axes for label in axis.get_xticklabels()]
    rate_labels = [label for label in labels if label.startswith("LR ")]
    assert rate_labels and all("vs 3e-07" in label for label in rate_labels)
    plt.close(page.figure)


def test_seed_repeat_cost_and_movement_identify_seed43(monkeypatch):
    run = future_campaign(2)
    monkeypatch.setattr(ts.dg, "load", lambda run, kind: pd.DataFrame())
    monkeypatch.setattr(ts.cp, "optimization_curves", lambda run, field: pd.DataFrame([
        dict(trial_name=trial.trial_name, base=trial.base, frozen=trial.frozen,
             window_end=run.target, weight_drift=.01)
        for trial in run.trials.itertuples()]))
    style.apply()
    cost, = ts.plot_resources(run)
    movement, = ts.plot_movement(run)
    assert cost.caption.startswith("Training seed: 43.")
    assert movement.caption.startswith("Training seed: 43.")
    assert cost.tables["Completed-trial source measurements"].seed.eq(43).all()
    assert ts.seed_caption(SimpleNamespace(trials=pd.DataFrame({"seed": [42, 43]}))) == "Training seeds: 42, 43. "
    assert "unavailable" in ts.seed_caption(SimpleNamespace(trials=pd.DataFrame()))
    plt.close(cost.figure)
    plt.close(movement.figure)


def test_adapter_cost_figure_keeps_tasks_separate_and_declares_query_only_scope():
    runs = []
    for track in ("pd", "lgd"):
        run = future_campaign(2, track)
        run.cfg.train.adaptation_mode = "diagscale"
        runs.append(run)
    style.apply()
    page, = ts.plot_adapter_costs(runs)
    assert len(page.figure.axes) == 2
    assert "Only query-scale deltas are optimized" in page.caption
    assert "PD: Training seed: 43." in page.caption and "LGD: Training seed: 43." in page.caption
    assert page.provenance["parameter_update_scope"] == "query_scale_deltas_only"
    stats = page.tables["Displayed DiagScale cost medians, interquartile ranges and trial support"]
    source = page.tables["Completed DiagScale trial source measurements"]
    assert set(stats.track) == set(source.track) == {"pd", "lgd"}
    assert "frozen" not in stats and "frozen" not in source
    assert source.parameter_update_scope.eq("DiagScale query-scale deltas only").all()
    assert stats.trials.eq(4).all()
    labels = [tick.get_text() for tick in page.figure.axes[0].get_yticklabels()]
    assert labels == ["PD / TabPFN v3", "LGD / TabPFN v3"]
    plt.close(page.figure)
