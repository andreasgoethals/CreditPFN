"""Seed/protocol pairing and publication behavior on future synthetic evidence."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualize import campaign as cp, comparisons as cmp, pilots, reporting, style
from tests.test_publication_reports import research_campaign


def future_campaign(experiment=2, track="pd", *, all_bases=False):
    bases = ("v2", "v2.6", "v3", "tabicl-v2") if all_bases else ("v3",)
    run = research_campaign(track, experiment, bases=bases)
    panels = []
    for partition, group in run.evaluation.groupby("split"):
        example = group[group.test_dataset_id.eq(group.test_dataset_id.iloc[0])]
        for index in range(4):
            panels.append(example.assign(domain="ood", test_dataset_id=f"Panel {index}"))
    run.evaluation = pd.concat([run.evaluation, *panels], ignore_index=True)
    return run


def test_seed_pairing_retains_partition_before_noncredit_average():
    main, repeat = future_campaign(1), future_campaign(2)
    pairs = cmp.seed_pairs(main, repeat, domain="ood", benchmark=True)
    assert len(pairs) == 16 and set(pairs.partition) == {0, 1, 2, 3}
    assert len(cmp.dataset_means(pairs, repeat, ["difference"])) == 4
    # A different missing partition in each seed cannot be averaged into a pair.
    main.evaluation = main.evaluation[~(main.evaluation.domain.eq("ood") & main.evaluation.split.eq(0))]
    repeat.evaluation = repeat.evaluation[~(repeat.evaluation.domain.eq("ood") & repeat.evaluation.split.eq(1))]
    pairs = cmp.seed_pairs(main, repeat, domain="ood", benchmark=True)
    assert set(pairs.partition) == {2, 3}
    assert cmp.dataset_means(pairs, repeat, ["difference"]).empty


def test_seed_pairing_rejects_different_context_policy():
    main, repeat = future_campaign(1), future_campaign(2)
    repeat.cfg.train.context_sampling = "stratified"
    with pytest.raises(ValueError, match="identical reference"):
        cmp.seed_pairs(main, repeat)


def test_seed_reference_does_not_include_other_main_recipes():
    main, repeat = future_campaign(1), future_campaign(2)
    reference = cmp.reference_campaign(main)
    assert len(reference.trials) == 4
    assert reference.trials.learning_rate.eq(3e-7).all()
    assert reference.trials.l2sp_lambda.eq(.003).all() and not reference.trials.frozen.any()
    assert set(reference.histories.trial_name) == set(reference.trials.trial_name)
    assert set(reference.trajectories.trial_name) == set(reference.trials.trial_name)
    costs = cmp.seed_costs(main, repeat)
    assert len(costs) == 4 and "gpu_hours_seed42" in costs and "gpu_hours_seed43" in costs


def test_benchmark_missing_outer_fold_excludes_that_partition_only():
    run = future_campaign(3)
    mask = (run.evaluation.domain.eq("ood") & run.evaluation.test_dataset_id.eq("Panel 0") &
            run.evaluation.split.eq(2) & run.evaluation.fold_idx.eq(4) &
            run.evaluation.source.eq("tabpfn-untuned"))
    run.evaluation.loc[mask, "status"] = "FAIL"
    pairs = cmp.sampling_matched(run, domain="ood", benchmark=True)
    assert len(pairs) == 45  # 3 protocols x (4 panels x 4 partitions - 1)
    means = cmp.dataset_means(pairs, run, ["effect"])
    assert len(means) == 9 and "Panel 0" not in set(means.dataset)


def test_sampling_never_compares_different_partitions():
    run = future_campaign(3)
    run.evaluation = run.evaluation[~(run.evaluation.method_dirname.str.endswith("__fullpass") &
                                      run.evaluation.split.eq(1))]
    pairs = cmp.sampling_matched(run, benchmark=True)
    assert 1 not in set(pairs.partition)
    contrasts = cmp.sampling_contrasts(pairs)
    assert contrasts.comparison.nunique() == 2
    assert len(contrasts) == 3*4*2


def test_sampling_costs_pair_all_modes_and_retain_absolute_values():
    run = future_campaign(3)
    run.trials.loc[run.trials.sampling.eq("accumulate"), "gpu_hours"] *= 10
    run.trials.loc[run.trials.sampling.eq("full_pass"), "rows_seen"] *= 3
    costs = cmp.sampling_costs(run)
    assert costs.loc[costs.sampling.eq("accumulate"), "gpu_hours_ratio"].eq(10).all()
    assert costs.loc[costs.sampling.eq("full_pass"), "rows_seen_ratio"].eq(3).all()
    run.trials.loc[(run.trials.partition.eq(1)) & run.trials.sampling.eq("accumulate"), "status"] = "INTERRUPTED"
    assert 1 not in set(cmp.sampling_costs(run).partition)
    assert cmp.plot_sampling_quality_cost(run) == []


def test_sampling_costs_are_unavailable_until_trials_complete():
    run = future_campaign(3)
    run.trials["status"] = "INTERRUPTED"
    assert cmp.sampling_costs(run).empty
    assert cmp.plot_sampling_costs(run) == []
    assert cmp.plot_sampling_quality_cost(run) == []


def test_missing_recorded_time_cannot_be_shown_as_zero_cost():
    run = future_campaign(3)
    run.trials["gpu_hours"] = np.nan
    assert cmp.plot_sampling_quality_cost(run) == []


def test_metric_directions_and_absolute_score_context():
    main, repeat = future_campaign(1), future_campaign(2)
    repeat.evaluation.loc[repeat.evaluation.source.eq("tabpfn-trained"), "brier_score"] += .05
    table = cmp.metric_tables(repeat, main=main)["Dataset benchmark metric contrasts"]
    assert np.allclose(table.loc[table.metric.eq("brier_score"), "difference"], -.05)
    assert {"credit", "ood"} == set(table.domain)
    scores = cmp.benchmark_scores(repeat)
    assert scores.folds.eq(5).all()
    assert scores.method_name.str.contains("untuned").any()


def test_legacy_credit_rows_cannot_appear_as_noncredit_results():
    run = future_campaign(2)
    run.evaluation = run.evaluation[run.evaluation.domain.eq("credit")].drop(columns="domain")
    assert cmp.observations(run, domain="ood", benchmark=True).empty


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_future_figures_are_distinct_bounded_and_numerically_reported(track):
    style.apply()
    main, repeat, sampling = [future_campaign(experiment, track, all_bases=True) for experiment in (1,2,3)]
    groups = [
        lambda: cmp.plot_seed_endpoints(main, repeat), lambda: cmp.plot_seed_endpoints(main, repeat, benchmark=True),
        lambda: cmp.plot_seed_curves(main, repeat),
        lambda: pilots.plot_process_overview(repeat, reference=cmp.reference_campaign(main)),
        lambda: cmp.plot_sampling_endpoints(sampling), lambda: cmp.plot_sampling_endpoints(sampling, benchmark=True),
        lambda: cmp.plot_sampling_curves(sampling), lambda: cmp.plot_sampling_curves(sampling, row_exposure=True),
        lambda: cmp.plot_sampling_costs(sampling), lambda: cmp.plot_sampling_quality_cost(sampling),
    ]
    names = []
    qa = os.environ.get("CREDITPFN_COMPARISON_QA")
    if qa:
        Path(qa).mkdir(parents=True, exist_ok=True)
    for plot in groups:
        pages = plot()
        assert pages
        for page in pages:
            names.append(page.name)
            assert page.tables and page.caption
            assert reporting.page_text(page)
            assert page.figure.get_figwidth() == pytest.approx(style.WIDTH_FULL)
            assert page.figure.get_figheight() <= style.MAX_HEIGHT
            style.finish_figure(page.figure)
            page.figure.canvas.draw()
            if qa:
                page.figure.suptitle("SYNTHETIC TEST / "+page.figure._suptitle.get_text())
                page.figure.savefig(Path(qa)/f"{track}_{page.name}.pdf")
                page.figure.savefig(Path(qa)/f"{track}_{page.name}.png")
            plt.close(page.figure)
    assert len(names) == len(set(names))


def test_unavailable_evidence_produces_no_empty_quality_figures():
    main, repeat, sampling = [future_campaign(experiment) for experiment in (1,2,3)]
    for run in (main, repeat, sampling):
        run.trajectories = pd.DataFrame()
        run.evaluation = pd.DataFrame()
    assert cmp.plot_seed_curves(main, repeat) == []
    assert cmp.plot_seed_endpoints(main, repeat, benchmark=True) == []
    assert cmp.plot_sampling_endpoints(sampling) == []
    assert cmp.plot_sampling_curves(sampling) == []
    assert cmp.benchmark_coverage(repeat).paired_five_fold_comparisons.eq(0).all()
