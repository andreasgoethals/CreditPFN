"""Second-campaign figures on SYNTHETIC evidence in the loaders' exact schemas.

Before the campaign has run, these tests are the only place the result figures are drawn: every
figure must render from realistic frames, attach the plotted values for the notebook summary,
respect the unit of analysis, and return None (never an empty figure) when evidence is missing.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualize import campaign2 as c2
from src.visualize import results2 as r2
from src.visualize import style
from src.visualize.campaign import Page
from src.visualize.reporting import page_text
from tests import synthetic_campaign2 as syn

ALL = {"reference": 1, "seed": 2, "domain_control": 1, "update_size_anchoring": 8, "scope": 6,
       "exposure": 1, "table_size": 2, "episode_size": 4}


@pytest.fixture(autouse=True)
def _style():
    style.apply()
    yield
    plt.close("all")


@pytest.fixture
def results(monkeypatch):
    syn.install_results(monkeypatch, ALL)


def _finished(fig, name="figure"):
    """Finish as a notebook page does and return its complete text equivalent."""
    assert fig is not None
    page = Page(name=name, figure=fig, caption="Synthetic.")
    text = page_text(page)
    assert "Figure: " + name in text
    return page.figure, text


# ------------------------------------------------------------------------- missing evidence
def test_every_figure_returns_none_without_evidence():
    empty = pd.DataFrame()
    assert r2.monitor_development(empty, title="t") is None
    assert r2.arm_trajectories([dict(curves=empty, x="successful_updates", ylabel="y")], empty, title="t") is None
    assert r2.table_gains(empty, track="pd", title="t") is None
    assert r2.transfer_and_retention({"pd": empty, "lgd": empty}, title="t") is None
    assert r2.domain_control({"pd": empty}, title="t") is None
    assert r2.finetune_conditions(empty, title="t") is None
    assert r2.baseline_position({"pd": empty}, title="t") is None
    assert r2.metric_profile_heatmap({"pd": empty}, title="t") is None
    assert r2.dose_response(empty, track="pd", title="t") is None
    assert r2.tradeoff(empty, track="pd", title="t") is None
    assert r2.arm_gains(empty, ["Reference"], track="pd", title="t") is None
    assert r2.episode_matched_context(empty, title="t") is None
    assert r2.quality_cost(empty, empty, title="t") is None
    assert r2.context_curves(empty, empty, track="pd", title="t") is None
    assert r2.gain_vs_properties(empty, track="pd", title="t") is None
    assert r2.transfer_heatmaps(empty, track="pd", title="t") is None


def test_loaders_stay_empty_without_evidence(monkeypatch):
    monkeypatch.setattr(c2, "_results_frames", lambda slug, track: pd.DataFrame())
    assert c2.endpoint_metrics("reference", "pd").empty
    assert c2.baseline_comparison("pd").empty
    assert c2.arms_overview(["reference", "scope"], "pd").empty
    assert c2.arm_table_gains(["reference"], "pd").empty
    assert c2.monitor_changes(pd.DataFrame()).empty
    assert c2.finetune_table(pd.DataFrame()).empty and c2.context_table(pd.DataFrame()).empty
    assert c2.select(pd.DataFrame(), slug="reference").empty
    assert c2.join(pd.DataFrame(), pd.DataFrame(), on=["slug"]).empty
    assert "No results yet for E5" in c2.evidence_note("E5", pd.DataFrame())


# ---------------------------------------------------------------------------- RQ1 results
def test_reference_tables_use_complete_tables_and_seed_rule(results):
    gains = c2.endpoint_gains("reference", "pd")
    seeds = c2.seed_noise("pd")
    fig, text = _finished(r2.table_gains(gains, seeds, track="pd", title="E1"))
    # 17 PD tables, one with a failed fold: 16 rows per model panel, one panel per model.
    assert len(fig.axes) == 4
    assert len(fig.axes[0].get_yticklabels()) == 16
    assert syn.CREDIT["pd"][-1] not in text
    assert "gain_seed44" in text


def test_metric_profile_orients_improvements(results):
    metrics = c2.endpoint_metrics("reference", "pd")
    assert set(metrics.metric) == set(c2.PROFILE_METRICS["pd"])
    row = metrics.iloc[0]
    expected = c2.improvement(row.metric, row.cpt, row.original)
    # Fold means of a target-distance metric need not equal the distance of the means, so only
    # one-directional metrics are checked exactly.
    auc = metrics[metrics.metric == "roc_auc"].iloc[0]
    assert np.isclose(auc.improvement, auc.cpt - auc.original)
    assert np.isfinite(expected)
    assert c2.improvement("log_loss", 0.3, 0.4) > 0 and c2.improvement("calibration_slope", 0.9, 1.3) > 0
    profiles = {t: c2.select(c2.metric_profile(c2.endpoint_metrics("reference", t)), arm=0) for t in ("pd", "lgd")}
    fig, text = _finished(r2.metric_profile_heatmap(profiles, title="profile"))
    assert len(fig.axes[0].texts) == 4 * 6 and "tables_improved" in text


def test_rq1_result_figures_render(results):
    gains = {t: c2.endpoint_gains("reference", t) for t in ("pd", "lgd")}
    _finished(r2.transfer_and_retention(gains, title="t"))
    controls = {t: c2.domain_control_gains(t) for t in ("pd", "lgd")}
    fig, _ = _finished(r2.domain_control(controls, title="t"))
    assert len(fig.axes) == 2
    baselines = {t: c2.baseline_comparison(t) for t in ("pd", "lgd")}
    assert (baselines["pd"].folds == 5).all() and set(baselines["pd"].model) == set(syn.BASES)
    _finished(r2.baseline_position(baselines, title="t"))
    finetune = c2.finetune_table(syn.finetune())
    assert set(finetune.model) == set(syn.BASES)
    assert np.allclose(finetune.cpt_finetune_change, finetune.cpt_finetune - finetune.original)
    fig, _ = _finished(r2.finetune_conditions(finetune, title="t"))
    assert len(fig.axes) == 8


# ---------------------------------------------------------------------------- RQ2 / RQ3
def test_arm_overview_carries_settings_and_both_table_groups(results):
    overview = c2.arms_overview(["reference", "update_size_anchoring", "scope"], "pd")
    assert len(overview) == (1 + 8 + 6) * 4
    e5 = overview[overview.slug == "update_size_anchoring"]
    assert sorted(e5.learning_rate.unique()) == [1e-7, 1e-6, 3e-6, 1e-5]
    assert set(overview.trainable) == {"all", "frozen_backbone", "diagscale", "lora_qv_r16"}
    assert (overview.credit_tables == 16).all() and (overview.retention_tables == 19).all()
    _finished(r2.dose_response(overview, track="pd", title="t"))
    fig, _ = _finished(r2.tradeoff(overview, track="pd", title="t"))
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 4


def test_training_data_arms_and_matched_context(results):
    table = c2.arm_table_gains(["reference", "exposure", "table_size", "episode_size"], "pd")
    order = list(dict.fromkeys(table.label))
    fig, _ = _finished(r2.arm_gains(table, order, track="pd", title="t"))
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == order
    matched = c2.episode_context_gains(syn.context(kind="episode_context"), syn.context())
    assert set(matched.context) == {"2048", "8192", "full"}
    assert matched[matched.episode_rows == 16384].model.isin(["TabPFN v3", "TabICLv2"]).all()
    _finished(r2.episode_matched_context(matched, title="t"))


# --------------------------------------------------------------------------- training monitor
def test_monitor_changes_are_equal_table_means_from_update_zero():
    frame = syn.trajectories("reference", "pd", seed=1)
    changes = c2.monitor_changes(frame)
    assert set(changes.scope) == {"test", "train", "ood"}
    assert np.allclose(changes[changes.successful_updates == 0].change, 0)
    curves = c2.monitor_curves(changes)
    held = curves[(curves.scope == "test") & (curves.successful_updates == 10000)]
    assert (held.tables == 17).all()          # every held-out table once over the four partitions
    retention = curves[(curves.scope == "ood") & (curves.successful_updates == 10000)]
    assert (retention.tables == 19).all()     # the panel's tables, averaged over partitions first
    fig, text = _finished(r2.monitor_development(curves, title="t"))
    assert "Held-out credit tables" in [t.get_text() for t in fig.legends[0].get_texts()]
    by_rows = c2.monitor_curves(c2.monitor_changes(frame, x="processed_rows", scopes=("test",)), x="processed_rows")
    assert set(by_rows.processed_rows) <= {1e6, 2e6, 5e6, 10e6, 20e6, 50e6}


def test_seed_band_and_arm_trajectories():
    frame = pd.concat([syn.trajectories("reference", "pd", seed=1), syn.trajectories("seed", "pd", 2, seed=2)])
    changes = c2.monitor_changes(frame)
    band = c2.seed_band(changes)
    assert (band.seeds == 3).all() and (band.high >= band.low).all()
    _finished(r2.monitor_development(c2.monitor_curves(changes).query("slug == 'reference'"), band, title="t"))
    frame = pd.concat([syn.trajectories("reference", "pd", seed=1),
                       syn.trajectories("update_size_anchoring", "pd", 8, seed=3, partitions=2)])
    curves = c2.monitor_curves(c2.monitor_changes(frame))
    settings = pd.concat([c2.arm_settings("reference", "pd"), c2.arm_settings("update_size_anchoring", "pd")])
    arms = r2.arm_styles(settings, color_by="learning_rate", linestyle_by="l2sp_lambda")
    assert arms.loc[arms.slug == "reference", "color"].iloc[0] == style.COLORS["reference"]
    assert set(arms.linestyle) == {"-", "--", ":"}
    drift = c2.drift_curves(frame, "pd")
    fig, _ = _finished(r2.arm_trajectories([dict(curves=curves[curves.scope == "test"], x="successful_updates",
                                                 ylabel="credit"),
                                            dict(curves=drift, x="successful_updates", ylabel="drift", log_y=True)],
                                           arms, title="t"))
    assert fig.axes[4].get_yscale() == "log"
    resources = c2.resource_summary(c2.run_resources(frame))
    assert set(resources.slug) == {"reference", "update_size_anchoring"} and (resources.runs > 0).all()


# --------------------------------------------------------------------------------- RQ4
def test_context_curve_fixed_cohort_and_full_pool():
    table = c2.context_table(syn.context())
    assert table.full.any() and not table[table.full].context_rows.isin([8, 16]).any()
    curve = c2.context_cohort_curve(table)
    cohort = curve[curve.fixed_cohort]
    assert (cohort.tables == cohort.all_tables).all()
    assert curve[~curve.fixed_cohort].tables.lt(curve[~curve.fixed_cohort].all_tables).all()
    fig, _ = _finished(r2.context_curves(table, curve, track="pd", title="t"))
    assert fig.axes[4].get_xscale() == "log"


def test_transfer_matrix_and_property_scatter(results):
    long = c2.transfer_long(syn.transfer_trajectories("pd"), "pd")
    assert set(long.successful_updates) == {2500}
    assert not (long.source == long.target).any()
    fig, _ = _finished(r2.transfer_heatmaps(long, track="pd", title="t"))
    assert len(fig.axes[0].images) == 1
    gains = c2.select(c2.endpoint_gains("reference", "pd"), arm=0, credit=True, complete=True)
    properties = pd.DataFrame(dict(test_dataset_id=syn.CREDIT["pd"], rows=np.geomspace(1e3, 7e5, 17),
                                   features=np.arange(17) + 5, default_rate=np.linspace(.02, .4, 17)))
    fig, text = _finished(r2.gain_vs_properties(gains.merge(properties, on="test_dataset_id"), track="pd", title="t"))
    assert "default_rate" in text and fig.axes[1].get_xscale() == "log"
