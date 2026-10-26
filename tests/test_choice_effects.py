"""Compact recipe figures must preserve paired evidence and its averaging unit."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
import pandas as pd
import pytest

from src.visualize import choice_effects as choices, style
from src.visualize.reporting import page_text
from tests.test_followup_comparisons import future_campaign


@pytest.fixture(autouse=True)
def close_figures():
    style.apply()
    yield
    plt.close("all")


def test_one_missing_metric_fold_removes_table_from_every_displayed_recipe_and_metric():
    run = future_campaign(1)
    before = choices.benchmark_grid(run)
    assert set(before.metric) == {"roc_auc", "brier_score", "log_loss"}
    broken = (run.evaluation.domain.eq("credit") & run.evaluation.test_dataset_id.eq("Dataset 00")
              & run.evaluation.source.str.endswith("-trained") & run.evaluation.lr.eq(1e-5)
              & run.evaluation.fold_idx.eq(2))
    run.evaluation.loc[broken, "log_loss"] = np.nan
    after = choices.benchmark_grid(run)
    credit = after[after.domain.eq("credit")]
    assert "Dataset 00" not in set(credit.dataset)
    assert credit.groupby("metric").dataset.nunique().eq(15).all()
    assert choices.recipe_summary(credit).tables.eq(15).all()
    assert after[after.domain.eq("ood")].groupby("metric").dataset.nunique().eq(4).all()


def test_noncredit_metric_requires_every_partition_before_common_cohort_averaging():
    run = future_campaign(1)
    broken = (run.evaluation.domain.eq("ood") & run.evaluation.test_dataset_id.eq("Panel 0")
              & run.evaluation.split.eq(2) & run.evaluation.source.str.endswith("-untuned")
              & run.evaluation.fold_idx.eq(1))
    run.evaluation.loc[broken, "brier_score"] = np.nan
    result = choices.benchmark_grid(run)
    retention = result[result.domain.eq("ood")]
    assert "Panel 0" not in set(retention.dataset)
    assert retention.groupby("metric").dataset.nunique().eq(3).all()
    assert retention.partitions.eq(4).all()
    assert result[result.domain.eq("credit")].groupby("metric").dataset.nunique().eq(16).all()


def _effect_row(dataset, rate, anchor, effect):
    return dict(domain="credit", metric="roc_auc", base="v3", dataset=dataset,
                learning_rate=rate, l2sp_lambda=anchor, frozen=False,
                sampling="one_sample", effect=effect)


def test_choice_effects_pair_settings_then_weight_tables_equally_and_preserve_harms():
    # A has two matched settings, B one. Counting pairs as independent observations
    # would give .01; equal-table averaging correctly gives (.02 - .01) / 2 = .005.
    data = pd.DataFrame([
        _effect_row("A", 3e-7, 0., -.10), _effect_row("A", 3e-7, .003, -.09),
        _effect_row("A", 1e-6, 0., -.20), _effect_row("A", 1e-6, .003, -.17),
        _effect_row("B", 3e-7, 0., -.10), _effect_row("B", 3e-7, .003, -.11),
    ])
    paired, summary, marginal = choices.choice_tables(data)
    anchor_pairs = paired[paired.factor.eq("l2sp_lambda")].set_index("dataset")
    np.testing.assert_allclose(anchor_pairs.loc[["A", "B"], "contrast"], [.02, -.01])
    anchor_summary = summary[summary.choice.str.startswith("L2-SP")].iloc[0]
    assert anchor_summary["mean"] == pytest.approx(.005)
    assert anchor_summary.tables == 2
    assert marginal.effect.lt(0).all()  # A favorable choice contrast is still negative CPT.
    page, = choices.plot_choices(data, "pd")
    assert "less deterioration" in page.caption
    assert page.provenance["aggregation_unit"] == "table"
    assert "Marginal CPT-versus-original effects" in page_text(page)


def test_unmatched_settings_cannot_create_an_anchor_contrast():
    data = pd.DataFrame([
        _effect_row("A", 3e-7, 0., -.1),
        _effect_row("A", 1e-6, .003, .1),
    ])
    paired, summary, _ = choices.choice_tables(data)
    assert paired.empty and summary.empty


def test_benchmark_probability_metrics_keep_lower_is_better_direction():
    run = future_campaign(1)
    tuned = run.evaluation.source.str.endswith("-trained")
    run.evaluation.loc[tuned, "brier_score"] -= .02
    run.evaluation.loc[tuned, "log_loss"] += .03
    values = choices.benchmark_grid(run)
    np.testing.assert_allclose(values.loc[values.metric.eq("brier_score"), "effect"], .02)
    np.testing.assert_allclose(values.loc[values.metric.eq("log_loss"), "effect"], -.03)


def test_monitor_summary_uses_tables_not_equal_partition_means_and_separate_legends():
    run = future_campaign(1)
    for column in run.trajectories:
        if column.startswith("metric__test__"):
            observed = run.trajectories[column].notna()
            run.trajectories.loc[observed, column] = .7
    for dataset in ("Dataset 01", "Dataset 02", "Dataset 03"):
        run.trajectories[f"metric__test__{dataset}"] = np.nan
    end = run.trajectories.successful_updates.eq(run.target)
    observed = run.trajectories["metric__test__Dataset 00"].notna()
    run.trajectories.loc[end & observed, "metric__test__Dataset 00"] = .8
    page, = choices.plot_monitor_changes(run)
    summary = page.tables["Equal-table monitor means"]
    assert summary.tables.eq(13).all()
    np.testing.assert_allclose(summary["mean"], .1 / 13)
    assert page.provenance["measurement"] == "fixed_context_monitor"
    assert page.provenance["aggregation_unit"] == "table"
    assert "held-out credit" in page.figure._suptitle.get_text().lower()
    visible_axes = [ax for ax in page.figure.axes if ax.get_visible() and ax.axison]
    assert len(visible_axes) == 1
    assert visible_axes[0].get_xlabel() == "Peak learning rate"
    assert len(set(t.get_text() for t in visible_axes[0].get_xticklabels())) == 4
    assert len(visible_axes[0].collections) == 16
    legends = [artist for ax in page.figure.axes for artist in ax.artists if isinstance(artist, Legend)]
    assert {legend.get_title().get_text() for legend in legends} == {
        "Peak learning rate", "L2-SP strength", "Parameters updated"}
    style.finish_figure(page.figure)
    assert all(legend.axes is not None for legend in legends)


def test_combination_plot_reports_all_recipes_and_counts_table_improvements():
    values = pd.DataFrame([
        _effect_row(dataset, rate, anchor, effect + frozen * .01) | {"frozen": frozen}
        for dataset, effect in (("A", -.02), ("B", .03))
        for rate in (3e-7, 1e-6)
        for anchor in (0., .003)
        for frozen in (False, True)
    ])
    page, = choices.plot_combinations(values, "pd", "roc_auc")
    summary = page.tables["Cell means and improvement counts"]
    assert len(summary) == 8 and summary.tables.eq(2).all()
    assert summary.improved.eq(1).all()
    assert page.figure.axes[0].images[0].get_array().shape == (2, 4)
    assert "not independently validated" in page.caption
    assert "Every paired table and recipe effect" in page_text(page)
