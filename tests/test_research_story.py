"""Research contrasts must retain measurement identity and matched populations."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualize import campaign as cp, dashboards as db, interactions as ix
from src.visualize import inference_review as review, context_curves as cc, diagnostics as dg, style
from src.visualize.figures import FigureSaver, read_manifest
from src.visualize.reporting import page_text
from tests.test_followup_comparisons import future_campaign
from tests.test_new_experiment_reports import context_fixture
from tests.test_publication_reports import calibrated_campaign


@pytest.fixture(autouse=True)
def close_figures():
    style.apply()
    yield
    plt.close("all")


def test_factorial_cohort_does_not_change_when_one_recipe_loses_one_fold():
    run = future_campaign(1)
    before, summary = ix.factorial_tables(run, "roc_auc")
    assert before.dataset.nunique() == 16 and summary.tables.eq(16).all()
    mask = (run.evaluation.test_dataset_id.eq("Dataset 00") & run.evaluation.source.str.endswith("-trained")
            & run.evaluation.lr.eq(1e-5) & run.evaluation.fold_idx.eq(2))
    run.evaluation.loc[mask, "status"] = "FAIL"
    after, summary = ix.factorial_tables(run, "roc_auc")
    assert "Dataset 00" not in set(after.dataset)
    assert summary.tables.eq(15).all()
    assert len(after) == 15*16  # All other recipes also exclude that table.


def test_retention_factorial_requires_all_partitions_before_equal_table_averages():
    run = future_campaign(1)
    before, summary = ix.factorial_tables(run, "roc_auc", "ood")
    assert before.dataset.nunique() == 4 and summary.tables.eq(4).all()
    mask = run.evaluation.domain.eq("ood") & run.evaluation.test_dataset_id.eq("Panel 0") & run.evaluation.split.eq(2)
    run.evaluation = run.evaluation[~mask]
    after, summary = ix.factorial_tables(run, "roc_auc", "ood")
    assert "Panel 0" not in set(after.dataset) and summary.tables.eq(3).all()
    assert after.partitions.eq(4).all()


def test_interaction_maps_retain_scope_anchor_rates_and_all_plotted_values():
    run = future_campaign(1, all_bases=True)
    page, = ix.plot_factorial(run, "roc_auc")
    stats = page.tables["Means and table interquartile ranges"]
    assert len(stats) == 4*2*2*4
    assert len(page.figure.axes) == 8
    assert "not confidence intervals" in page.caption and "five-fold final benchmark" in page.caption
    assert "Recipe-complete table effects" in page_text(page)
    assert page.figure.get_figwidth() == style.WIDTH_FULL
    for pair in zip(page.figure.axes[::2], page.figure.axes[1::2]):
        assert all(ax.get_xscale() == "log" for ax in pair)


def test_monitor_and_benchmark_caption_identity_survives_saving(isolated_output):
    run = future_campaign(1)
    # Synthetic monitors use recorded metric columns; final metrics remain separate.
    monitor, = db.plot_endpoint_factors(run)
    benchmark, = db.plot_endpoint_factors(run, benchmark=True)
    assert "milestone monitoring" in monitor.caption
    assert "five-fold final benchmark" in benchmark.caption
    assert monitor.provenance["measurement"] != benchmark.provenance["measurement"]
    sink = FigureSaver("experiment1/provenance_test")
    cp.show(sink, [monitor, benchmark])
    metadata = read_manifest(sink.notebook)
    assert [row["provenance"]["measurement"] for row in metadata] == ["fixed_context_monitor", "five_fold_final_benchmark"]
    assert metadata[1]["provenance"]["tables_by_base"] == {"v3": 16}
    # Structured metadata stays JSON serializable, without private row-level data.
    json.dumps(metadata)


def precision_fixture():
    return pd.DataFrame([dict(dataset=d, fold=f, policy=p, state=s,
        roc_auc=.6 + f*.001 + (.07 if s == "cpt" else .073 if p == "fp32_no_fa3" else 0))
        for d in ("A", "B", "C") for f in range(5)
        for p in ("auto", "fp32_no_fa3") for s in ("base", "saved", "cpt")])


def test_numerical_policy_effects_pair_folds_and_expose_gain_reversal():
    data = precision_fixture()
    effects = review.policy_deltas(data)
    assert len(effects) == 30
    np.testing.assert_allclose(effects.loc[effects.policy.eq("auto"), "effect"], .07)
    np.testing.assert_allclose(effects.loc[effects.policy.eq("fp32_no_fa3"), "effect"], -.003)
    assert effects.save_difference.eq(0).all()
    data.loc[data.fold.eq(4), "fold"] = 5
    with pytest.raises(ValueError, match="five paired folds"):
        review.policy_deltas(data)


def test_context_spaghetti_retains_each_table_in_the_fixed_cohort():
    study = context_fixture()
    page = next(p for p in cc.plot_context_curves(study) if p.name == "context_pd_through_2048")
    data = page.tables["Complete paired table values"]
    assert data.dataset.nunique() == 2
    for ax in page.figure.axes[2:]:
        # One line per table, one mean and one zero line.
        assert len(ax.lines) == 4
    assert "not uncertainty bands" in page.caption
    assert page.provenance["fixed_tables"] == ["Table 1", "Table 2"]


def test_log_loss_failure_cannot_change_the_calibration_comparison_cohort():
    run = calibrated_campaign()
    mask = (run.evaluation.test_dataset_id.eq("Dataset 00") & run.evaluation.fold_idx.eq(1)
            & run.evaluation.source.str.endswith("-untuned"))
    run.evaluation.loc[mask, "log_loss_platt"] = np.nan
    data = dg.calibration_effects(run)
    assert set(data.metric) == {"brier_score", "ece", "log_loss"}
    assert data.groupby(["metric", "calibration"]).dataset.nunique().eq(15).all()


def test_older_exports_retain_measured_calibration_scores_without_log_loss_imputation():
    run = calibrated_campaign()
    run.evaluation = run.evaluation.drop(columns=["log_loss_platt", "log_loss_isotonic"])
    data = dg.calibration_effects(run)
    assert set(data.metric) == {"brier_score", "ece"}
    assert data.groupby(["metric", "calibration"]).dataset.nunique().eq(16).all()
    page, = dg.plot_calibration_overview(run)
    assert len(page.figure.axes) == 2
