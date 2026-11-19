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
    assert r2.primary_forest(empty, title="t") is None
    assert r2.adaptation_by_size(empty, title="t") is None
    assert r2.metric_profile_heatmap({"pd": empty}, title="t") is None
    assert r2.tradeoff(empty, track="pd", title="t") is None
    assert r2.arm_gains(empty, ["Reference"], track="pd", title="t") is None
    assert r2.episode_matched_context(empty, title="t") is None
    assert r2.transfer_similarity({"pd": dict(pairs=empty)}, title="t") is None
    assert r2.context_curves(empty, empty, title="t") is None
    assert r2.gain_vs_properties(empty, track="pd", title="t") is None
    assert r2.transfer_heatmaps(empty, track="pd", title="t") is None


def test_loaders_stay_empty_without_evidence(monkeypatch):
    monkeypatch.setattr(c2, "_read_results_frames", lambda slug, track: pd.DataFrame())
    assert c2.endpoint_metrics("reference", "pd").empty
    assert c2.baseline_comparison("pd").empty
    assert c2.arms_overview(["reference", "scope"], "pd").empty
    assert c2.arm_table_gains(["reference"], "pd").empty
    assert c2.monitor_changes(pd.DataFrame()).empty
    assert c2.finetune_table(pd.DataFrame()).empty and c2.context_table(pd.DataFrame()).empty
    assert c2.select(pd.DataFrame(), slug="reference").empty
    assert c2.join(pd.DataFrame(), pd.DataFrame(), on=["slug"]).empty
    assert "No results yet for E8" in c2.evidence_note("E8", pd.DataFrame())


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
    assert len(fig.axes[0].texts) == 4 * len(c2.PROFILE_METRICS["pd"]) and "tables_improved" in text
    assert {"ks", "h_measure", "emp"} <= set(profiles["pd"].metric)


def test_rq1_result_figures_render(results):
    baselines = {t: c2.baseline_comparison(t) for t in ("pd", "lgd")}
    assert (baselines["pd"].folds == 5).all() and set(baselines["pd"].model) == set(syn.BASES)
    finetune = c2.finetune_table(syn.finetune())
    assert set(finetune.model) == set(syn.BASES)
    assert np.allclose(finetune.cpt_finetune_change, finetune.cpt_finetune - finetune.original)
    results = syn.finetune()
    table = c2.adaptation_by_size(results)
    assert np.allclose(table.cpt_after_finetune, table.cpt_finetune - table.original_finetune)
    assert set(table.best_option) <= set(c2.ADAPTATION_OPTIONS) and table.context_rows.notna().all()
    fig, _ = _finished(r2.adaptation_by_size(table, title="t"))
    assert len(fig.axes) == 6
    summary = c2.adaptation_summary(table)
    assert summary.tables.sum() == len(table)
    assert summary[[f"best_{o}" for o in c2.ADAPTATION_OPTIONS]].sum(axis=1).eq(summary.tables).all()


def test_primary_endpoint_is_holm_adjusted_without_an_equivalence_claim(monkeypatch):
    syn.install_results(monkeypatch, ALL, missing_fold=False)
    from src.utils.design import load_design
    assert "equivalence_margin" not in load_design()["analysis"]["primary"]
    primary = c2.primary_endpoint()
    assert len(primary) == 2 * 2 * 4                                   # tasks x outcomes x models
    assert (primary.p_holm >= primary.p_value).all() and (primary.p_holm <= 1).all()
    assert ((primary.low <= primary["mean"]) & (primary["mean"] <= primary.high)).all()
    assert set(primary.decision) <= {"better", "worse", "not significant"}
    assert not {"margin", "low90", "high90"} & set(primary.columns)
    brier = primary[(primary.metric == "brier_score")].iloc[0]
    assert np.isfinite(brier["mean"])                                  # oriented: positive = lower Brier
    retention = pd.concat([c2.outcome_intervals("reference", t, ["roc_auc", "brier_score", "r2", "crps"])
                           for t in ("pd", "lgd")])
    assert set(retention.credit) == {True, False}
    fig, text = _finished(r2.primary_forest(primary, retention, title="t"))
    assert len(fig.axes) == 4 and fig._suptitle is None and "p_holm" in text
    assert np.allclose(c2.holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])


def test_positive_control_compares_own_training_tables_with_held_out_tables():
    changes = c2.monitor_table_changes(syn.trajectories("reference", "pd"), "pd")
    control = c2.positive_control(changes)
    assert set(control.model) == set(syn.BASES)
    assert (control.train_mean_change > control.test_mean_change).all()   # synthetic: training tables gain 2x
    # every credit table is a training table of three partitions and held out in one; each counts once
    assert (control.train_tables == 17).all() and (control.test_tables == 17).all()


def test_bootstrap_intervals_cover_the_mean_and_respect_the_table_as_unit(results):
    frame = pd.DataFrame([dict(test_dataset_id=t, fold_idx=f, gain=0.01 * t + 0.001 * f)
                          for t in range(6) for f in range(5)])
    interval = c2.bootstrap_interval(frame, "gain")
    assert interval["tables"] == 6 and interval["low"] < interval["mean"] < interval["high"]
    assert np.isclose(interval["mean"], frame.gain.mean())
    assert c2.bootstrap_interval(frame, "gain") == interval                      # seeded
    partial = c2.bootstrap_interval(frame[~((frame.test_dataset_id == 0) & (frame.fold_idx == 4))], "gain")
    assert partial["tables"] == 5                                                # incomplete table leaves
    one = c2.bootstrap_interval(frame.assign(gain=0.5), "gain")
    assert one["low"] == one["high"] == 0.5
    intervals = {t: c2.gain_intervals("reference", t) for t in ("pd", "lgd")}
    assert set(intervals["pd"].model) == set(syn.BASES) and set(intervals["pd"].credit) == {True, False}
    assert ((intervals["pd"].low <= intervals["pd"]["mean"]) & (intervals["pd"]["mean"] <= intervals["pd"].high)).all()
    controls = {t: c2.domain_control_intervals(t) for t in ("pd", "lgd")}
    per_table = c2.domain_control_gains("pd").groupby("model").credit_minus_noncredit.mean()
    assert np.allclose(controls["pd"].set_index("model").credit_minus_noncredit.sort_index(), per_table.sort_index())
    contrast = c2.domain_contrast_intervals({"pd": ["roc_auc", "brier_score"], "lgd": ["r2", "crps"]})
    auc = contrast[(contrast.track == "pd") & (contrast.metric == "roc_auc")].set_index("model")["mean"]
    assert np.allclose(auc.sort_index(), controls["pd"].set_index("model").credit_minus_noncredit.sort_index())
    assert len(contrast) == 2 * 2 * 4
    primary = c2.primary_endpoint()
    fig, text = _finished(r2.primary_forest(primary, None, contrast, title="t"))
    assert "domain contrast" in text and len(fig.axes) == 4


def test_precision_agreement_and_cost_tables():
    summary = {"precision": {"0": {"track": "pd", "base": "tabpfn-v2.6-classifier-v2.6_default.ckpt", "tables": [
        {"table_index": 0, "rows": 8192, "loss_fp32": 0.5, "loss_bf16": 0.5, "grad_cosine_bf16": 0.72,
         "loss_fp16": 0.5, "grad_cosine_fp16": 0.98},
        {"table_index": 1, "rows": 8192, "loss_fp32": 0.4, "error": "out of memory"}]}}}
    agreement = c2.precision_agreement(summary)
    assert len(agreement) == 2 and set(agreement.precision) == set(c2.PRECISION_LABELS.values())
    assert set(agreement.model) == {"TabPFN v2.6"}          # checkpoint file name -> model label
    fig, text = _finished(r2.precision_agreement(agreement, title="t"))
    assert "below_threshold" in text and len(fig.axes) == 1
    assert c2.precision_agreement(None).empty and r2.precision_agreement(pd.DataFrame(), title="t") is None
    resources = pd.DataFrame([dict(slug="reference", arm=0, track="pd", model="tabpfn-v2", hours_without_monitor=h)
                              for h in (1.0, 3.0)])
    finetune = pd.DataFrame([dict(track="pd", model="tabpfn-v2", dataset_id=d, fold=f, start="original",
                                  finetuned=tuned, fit_seconds=360.0 if tuned else 1.0, predict_seconds=2.0,
                                  test_rows=500, device_name="NVIDIA B200")
                             for d in ("a", "b") for f in range(5) for tuned in (False, True)])
    cost = c2.cost_comparison(resources, finetune)
    assert cost.cpt_hours_total.iloc[0] == 4.0 and np.isclose(cost.finetune_hours_per_table.iloc[0], 0.5)
    assert np.isclose(cost.tables_to_break_even.iloc[0], 8.0)
    assert np.isclose(cost.inference_seconds_per_1k_rows.iloc[0], 4.0)
    assert cost.finetune_device.iloc[0] == "NVIDIA B200"
    assert c2.cost_comparison(pd.DataFrame(), finetune).empty


# ---------------------------------------------------------------------------- RQ2 / RQ3
def test_arm_overview_carries_settings_and_both_table_groups(results):
    overview = c2.arms_overview(["reference", "update_size_anchoring", "scope"], "pd")
    assert len(overview) == (1 + 8 + 6) * 4
    e5 = overview[overview.slug == "update_size_anchoring"]
    assert sorted(e5.learning_rate.unique()) == [1e-7, 1e-6, 3e-6, 1e-5]
    assert set(overview.trainable) == {"all", "frozen_backbone", "diagscale", "lora_qv_r16"}
    assert (overview.credit_tables == 16).all() and (overview.retention_tables == 19).all()
    fig, text = _finished(r2.tradeoff(overview, track="pd", title="t"))
    legend = [t.get_text() for t in fig.legends[0].texts]
    assert "Learning rate 1e-07 (smallest marker)" in legend and "Learning rate 1e-03 (largest marker)" in legend
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 4 and fig.axes[2].get_xlabel()


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


# --------------------------------------------------------------------------------- RQ2
def test_context_curve_fixed_cohort_and_full_pool():
    table = c2.context_table(syn.context())
    assert table.full.any() and not table[table.full].context_rows.isin([8, 16]).any()
    curve = c2.context_cohort_curve(table)
    cohort = curve[curve.fixed_cohort]
    assert (cohort.tables == cohort.all_tables).all()
    assert curve[~curve.fixed_cohort].tables.lt(curve[~curve.fixed_cohort].all_tables).all()
    fig, _ = _finished(r2.context_curves(table, curve, title="t"))
    assert fig.axes[0].get_xscale() == "log"


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


# --------------------------------------------------------------------- RQ2 slope and similarity tests
def _context_results(slope_credit=-0.004, slope_noncredit=-0.001):
    import numpy as np
    rows = []
    for table in ("0001.a", "0002.b", "0003.c"):
        for fold in range(5):
            for size in (8, 16, 32, 64, 128):
                d = np.log2(size)
                for start, auc in (("original", 0.70), ("cpt", 0.75 + slope_credit * d),
                                   ("noncredit_cpt", 0.72 + slope_noncredit * d)):
                    rows.append(dict(track="pd", model="v2.6", dataset_id=table, fold=fold, context_rows=size,
                                     start=start, roc_auc=auc,
                                     source="domain_control" if start == "noncredit_cpt" else "reference"))
    return pd.DataFrame(rows)


def test_context_table_pairs_credit_and_noncredit_cpt_on_identical_rows():
    table = c2.context_table(_context_results())
    assert {"gain", "noncredit_gain", "credit_minus_noncredit"} <= set(table.columns)
    row = table[table.context_rows == 8].iloc[0]
    assert row.gain == pytest.approx(0.05 - 0.004 * 3)
    assert row.credit_minus_noncredit == pytest.approx(0.03 - 0.003 * 3)
    curve = c2.context_cohort_curve(table)
    assert curve.noncredit_tables.eq(3).all()


def test_context_table_drops_noncredit_cells_without_five_folds():
    results = _context_results()
    results = results[~((results.start == "noncredit_cpt") & (results.fold == 4) & (results.dataset_id == "0001.a"))]
    table = c2.context_table(results)
    assert table[table.dataset_id == "0001.a"].noncredit_gain.isna().all()
    assert table[table.dataset_id == "0002.b"].noncredit_gain.notna().all()
    curve = c2.context_cohort_curve(table)
    assert curve.noncredit_gain.isna().all() and curve.gain.notna().all()


def test_bootstrap_excludes_nonfinite_and_unregistered_folds():
    frame = pd.DataFrame({"test_dataset_id": np.repeat(["a", "b", "c"], 5),
                          "fold_idx": [0, 1, 2, 3, 4] * 2 + [0, 1, 2, 3, 8],
                          "gain": [1.0] * 5 + [2.0] * 4 + [np.inf] + [3.0] * 5})
    interval = c2.bootstrap_interval(frame, "gain")
    assert interval["tables"] == 1 and interval["mean"] == 1.0


def test_slope_test_recovers_the_change_per_doubling():
    slopes = c2.context_slopes(c2.context_table(_context_results()))
    by = slopes.groupby("quantity").slope_per_doubling.mean()
    assert by["gain"] == pytest.approx(-0.004)
    assert by["noncredit_gain"] == pytest.approx(-0.001)
    assert by["credit_minus_noncredit"] == pytest.approx(-0.003)
    summary = c2.slope_summary(slopes)
    gain = summary[summary.quantity == "gain"].iloc[0]
    assert gain.tables == 3 and gain.tables_negative == 3 and gain.excludes_zero
    assert r2.context_slopes(summary, title="t") is not None


def test_similarity_test_finds_distance_driven_transfer_and_permutes_tables():
    import numpy as np
    rng = np.random.default_rng(1)
    tables = [f"{i:04d}.t" for i in range(1, 9)]
    properties = pd.DataFrame(dict(test_dataset_id=tables, rows=10 ** rng.uniform(3, 5, 8),
                                   features=rng.integers(5, 60, 8), categorical_share=rng.uniform(0, 1, 8)))
    long = pd.DataFrame([dict(model="v3", source=s, target=t, scope="test", change=0.0)
                         for s in tables for t in tables if s != t])
    pairs = c2.transfer_pairs(long, properties)
    pairs["change"] = -pairs.distance + rng.normal(0, 0.05, len(pairs))   # closer sources transfer better
    test = c2.transfer_similarity(pairs, permutations=300)
    assert test.iloc[0].spearman < -0.8 and test.iloc[0].p_value < 0.05
    pairs["change"] = rng.normal(0, 1, len(pairs))                         # no relation
    assert c2.transfer_similarity(pairs, permutations=300).iloc[0].p_value > 0.05
    assert not pairs.same_source.any()                                      # synthetic ids: no shared lender
    pairs.loc[:3, "same_source"] = True
    test = c2.transfer_similarity(pairs, permutations=50)
    assert test.iloc[0].same_source_pairs == 4
    assert np.isclose(test.iloc[0].same_source_mean_change, pairs.change[:4].mean())
    fig, text = _finished(r2.transfer_similarity({"pd": dict(pairs=pairs, tests=test), "lgd": dict(pairs=pairs, tests=test)},
                                                  title="t"))
    assert len(fig.axes) == 2 * len(set(pairs.model)) and "PD similarity test" in text


# --------------------------------------------------------------- leaderboard and table scope
def test_leaderboard_pool_meets_on_identical_folds_and_anchors_the_linear_baseline(results):
    finetune = syn.finetune()
    for track, anchor in (("pd", "Logistic regression"), ("lgd", "Linear regression")):
        errors = c2.leaderboard_errors(track, finetune)
        assert not errors.table.str.startswith("openml_").any()            # held-out credit tables only
        assert set(errors.variant) == {"original", "CPT", "FT", "CPT+FT", "baseline"}
        out = c2.leaderboard(track, finetune)
        board = out["board"]
        assert len(board) == 4 * 4 + 3                                      # 4 models x 4 variants + baselines
        assert board.set_index("method").loc[anchor, "elo"] == pytest.approx(1000)
        # Foundation models come from E5 (complete); the missing E1 CPT fold no longer matters.
        assert out["dropped"] == []
        assert board.tables.eq(len(syn.CREDIT[track])).all()
        assert ((board.normalized_score >= 0) & (board.normalized_score <= 1)).all()
        fig, text = _finished(r2.leaderboard(board, track=track, title="t"))
        assert len(fig.axes) == 3 and "normalized_score" in text and "improvability" in text
    assert r2.leaderboard(pd.DataFrame(), track="pd", title="t") is None


def test_leaderboard_leaves_out_a_method_scored_on_too_few_tables(results):
    finetune = syn.finetune()
    few = finetune[~(finetune.finetuned & (finetune.model == "v2") & (finetune.start == "cpt")
                     & ~finetune.dataset_id.isin(syn.CREDIT["pd"][:3]))]
    out = c2.leaderboard("pd", few)
    coverage = out["coverage"].set_index("method")
    assert not coverage.loc["TabPFN v2 CPT+FT", "entered"]
    assert "TabPFN v2 CPT+FT" not in set(out["board"].method)
    assert out["dropped"] == []                                              # not shrunk to three tables


@pytest.fixture
def private(monkeypatch):
    """Mark two synthetic PD tables and three LGD tables proprietary; restore the scope after."""
    from src.data import dataset_names
    hidden = {*syn.CREDIT["pd"][:2], *syn.CREDIT["lgd"][:3]}
    monkeypatch.setattr(dataset_names, "_PROPRIETARY", hidden)
    yield hidden
    c2.use_tables("all")


def test_public_scope_removes_proprietary_tables_from_every_loader(results, private):
    full = c2.endpoint_gains("reference", "pd")
    c2.use_tables("public")
    assert c2.table_scope() == "public"
    public = c2.endpoint_gains("reference", "pd")
    assert set(full.test_dataset_id) - set(public.test_dataset_id) == set(syn.CREDIT["pd"][:2])
    assert not set(c2.leaderboard_errors("lgd", syn.finetune()).table) & private
    primary = c2.primary_endpoint()
    assert primary[primary.track == "lgd"].tables.eq(len(syn.CREDIT["lgd"]) - 3 - 1).all()
    assert primary.decision.eq("incomplete").all() and primary.p_value.isna().all()
    changes = c2.monitor_table_changes(syn.trajectories("reference", "pd"), "pd")
    assert not set(changes.table) & private and not changes.empty
    with pytest.raises(ValueError):
        c2.use_tables("private")
