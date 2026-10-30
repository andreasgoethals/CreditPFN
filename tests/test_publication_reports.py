"""Publication contracts: source isolation, complete text and paired evidence."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
import json
import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.visualize import campaign as cp, diagnostics as dg, publication as pub, style
from src.visualize import inputs, reporting
from src.visualize.figures import FigureSaver
from src.visualize.eval_viz import _decode_method_dirname


@pytest.fixture(autouse=True)
def plotting():
    style.apply()
    yield
    plt.close("all")


@pytest.mark.parametrize("prefix", ["", "output CreditPFN/"])
def test_zip_and_project_inputs_are_read_without_copy_or_source_mutation(tmp_path, monkeypatch, prefix):
    data = tmp_path / "data.zip"
    project = tmp_path / "project.zip"
    with zipfile.ZipFile(data, "w") as z:
        z.writestr(prefix+"experiment1/manifests/main.csv", "status,value\nOK,0.1234567890123456\n")
    with zipfile.ZipFile(project, "w") as z:
        z.writestr(prefix+"experiment1/training/pd/main.csv", "successful_updates\n10000\n")
    monkeypatch.setenv("CREDITPFN_ANALYSIS_ROOT", str(data))
    monkeypatch.setenv("CREDITPFN_ANALYSIS_PROJECT_ROOT", str(project))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    manifests = inputs.analysis_location("experiment1", "manifests", tmp_path/"unused")
    training = inputs.analysis_location("experiment1", "training", tmp_path/"unused")
    assert inputs.read_csv(next(manifests.glob("*.csv"))).status.tolist() == ["OK"]
    assert inputs.read_csv(next(training.rglob("*.csv"))).successful_updates.tolist() == [10000]
    assert list(training.glob("*.csv")) == []
    assert str(next(training.rglob("*.csv")).relative_to(training)) == "pd/main.csv"
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    monkeypatch.delenv("CREDITPFN_ANALYSIS_ROOT")
    assert inputs.analysis_location("experiment1", "manifests", tmp_path/"local").archive == project


@pytest.mark.parametrize("members", [["../outside.csv"], ["/absolute.csv"], ["same.csv","same.csv"]])
def test_ambiguous_or_unsafe_archives_fail_explicitly(tmp_path, monkeypatch, members):
    path = tmp_path/"bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        expected_warning = pytest.warns(UserWarning, match="Duplicate name") if len(set(members)) != len(members) else nullcontext()
        with expected_warning:
            for member in members:
                z.writestr(member, "a\n1\n")
    monkeypatch.setenv("CREDITPFN_ANALYSIS_ROOT", str(path))
    with pytest.raises(ValueError):
        inputs.analysis_root()


def test_archive_member_times_select_latest_auxiliary_workflow(tmp_path, monkeypatch):
    from src.visualize import auxiliary
    path = tmp_path / "output.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, year in (("older", 2025), ("newer", 2026)):
            entry = zipfile.ZipInfo(f"experiment0/manifests/workflow/{name}/state.json",
                                    date_time=(year, 1, 1, 0, 0, 0))
            archive.writestr(entry, json.dumps(dict(part="auxiliary", id=name,
                                status="gpu_running", fingerprint="test")))
    monkeypatch.setenv("CREDITPFN_ANALYSIS_ROOT", str(path))
    monkeypatch.delenv("CREDITPFN_ANALYSIS_PROJECT_ROOT", raising=False)
    scope, checks, arms = auxiliary.evidence()
    assert scope["workflow"] == "newer" and scope["reported_tasks"] == 0
    assert checks.empty and arms.empty
    stat = (inputs.analysis_root() / "experiment0/manifests/workflow/newer/state.json").stat()
    assert stat.st_mtime * 1e9 == stat.st_mtime_ns


def test_summary_repeats_all_displayed_values_in_section_order(isolated_output):
    sink = FigureSaver("experiment1/text_contract")
    report = cp.NotebookReport("Report", sink=sink)
    frame = pd.DataFrame({"dataset": [f"Dataset {i:03}" for i in range(80)], "value": np.arange(80)})
    report.table("All datasets", frame)
    report.add("1. Inventory", "No truncation")
    fig, ax = plt.subplots(figsize=style.figsize())
    ax.imshow(np.ma.array([[1,2],[3,4]], mask=[[0,0],[0,1]]))
    page = cp.Page("matrix", fig, "Observed values; masked cells are unavailable.")
    cp.show(sink, [page])
    report.add("2. Effects", "No imputation")
    summary = report.summary(sink)
    assert summary.index("1. Inventory") < summary.index("Dataset 079") < summary.index("2. Effects")
    assert "unavailable" in summary and "..." not in summary
    assert page.caption in summary
    assert "Figure: matrix" in summary
    report.table("Unassigned", frame.head())
    with pytest.raises(ValueError, match="assigned"):
        report.summary(sink)


def test_corrected_pilot_uses_receipt_config_not_static_grid(tmp_path, monkeypatch):
    root = tmp_path/"experiment0/manifests/workflow"
    (root/"corrected").mkdir(parents=True)
    (root/"pilot_passed.json").write_text(json.dumps({"id":"corrected"}))
    (root/"corrected/pilot_pd.yaml").write_text("run_name: exact_corrected\n")
    monkeypatch.setenv("CREDITPFN_ANALYSIS_ROOT",str(tmp_path))
    monkeypatch.delenv("CREDITPFN_ANALYSIS_PROJECT_ROOT",raising=False)
    monkeypatch.setattr(cp,"load_campaign",lambda *a,**kw: kw["cfg_override"])
    assert pub.pilot_campaigns("pd").run_name == "exact_corrected"
    with pytest.raises(ValueError, match="configuration is missing"):
        pub.pilot_campaigns("lgd")


def research_campaign(track="pd", experiment=1, *, bases=("v3",)):
    """Synthetic future project evidence. Never used as research results."""
    cfg = cp.load_train_config(config_path=str(cp.config_path(experiment, track)))
    trials = cp.planned_trials(cfg)
    trials = trials[trials.base.isin(bases)].reset_index(drop=True).assign(
        status="OK", n_train_datasets=4, n_test_datasets=4, trainable_params=1000, total_params=2000,
        sec_per_step=1., elapsed_sec=10000., gpu_hours=10000./3600, peak_gpu_gb=80.,
        rows_seen=100000, final_drift=.02, baseline_test_metric=.7, final_test_metric=.71)
    trajectory, histories, evaluation = [], [], []
    bins = json.dumps([dict(bin=i, count=10, probability_sum=i+.5, positive_count=i) for i in range(10)])
    for _, trial in trials.iterrows():
        gain = .014 - .012*np.log10(trial.learning_rate/3e-7) + (.003 if trial.l2sp_lambda else 0)
        gain *= .5 if trial.frozen else 1.
        for update in cfg.train.trajectory_steps:
            item = dict(trial_name=trial.trial_name, successful_updates=update, processed_rows=update*10,
                        record_type="trajectory")
            for index in range(4):
                dataset = f"Dataset {trial.partition*4+index:02d}"
                for split in ("test","train","ood"):
                    name = f"Panel {index}" if split=="ood" else dataset
                    initial = .7 if track=="pd" else .2+index/10
                    direction = 1 if track=="pd" else -1
                    change = (update/cfg.train.target_total_steps)*(gain+index*.002)
                    item[f"metric__{split}__{name}"] = initial + direction*change*(initial if track=="lgd" else 1)
            trajectory.append(item)
        for update in np.linspace(200,cfg.train.target_total_steps,50):
            histories.append(dict(trial_name=trial.trial_name,successful_updates=update,train_loss=.5,
                l2sp_penalty=.02,grad_norm_mean=.3,clipped_frac=.1,lr_applied=trial.learning_rate,
                weight_drift=.02*update/cfg.train.target_total_steps,training_seconds=2.,compute_seconds=1.9,
                data_wait_seconds=.1,amp_skipped_steps=0,data_skipped_steps=0))
        for index in range(4):
            family = "tabicl" if trial.base.startswith("tabicl") else "tabpfn"
            for source in (f"{family}-untuned", f"{family}-trained"):
                suffix = (f"__lr{trial.learning_rate:.0e}__l2sp{trial.l2sp_lambda:g}"
                          + ("__frozen" if trial.frozen else "")
                          + (f"__{trial.sampling.replace('full_pass','fullpass')}" if trial.sampling!="one_sample" else ""))
                method = f"{source}__{trial.base}-default" + (suffix if source.endswith("-trained") else "")
                for fold in range(5):
                    evaluation.append(dict(
                        method_dirname=method,**_decode_method_dirname(method),test_dataset_id=f"Dataset {trial.partition*4+index:02d}",
                        fold_idx=fold,eval_run=f"test_s{trial.partition:02d}",split=trial.partition,status="OK",
                        domain="credit",roc_auc=.72 if source.endswith("-trained") else .7,
                        rmse=.18 if source.endswith("-trained") else .2,mae=.1,pr_auc=.6,r2=.2,
                        brier_score=.12,log_loss=.2,ece=.01,f1=.6,optimal_threshold=.4+fold*.01,
                        calibration_bins=bins,calibration_bins_platt=bins,calibration_bins_isotonic=bins,
                        method_name=method))
    evaluation = pd.DataFrame(evaluation).drop_duplicates(
        ["method_dirname","test_dataset_id","fold_idx","eval_run","split"])
    return cp.Campaign(cfg,trials,pd.DataFrame(trajectory),pd.DataFrame(histories),evaluation)


def test_planned_coverage_and_missing_milestones_cannot_be_hidden():
    run = research_campaign()
    assert 0 in run.cfg.train.trajectory_steps
    pages = cp.plot_trajectory_pages(run)
    assert pages and all(p.tables["Curve statistics and coverage"].complete.all() for p in pages)
    missing = run.trials.iloc[0].trial_name
    run.trajectories = run.trajectories[~((run.trajectories.trial_name==missing)&
                                        (run.trajectories.successful_updates==1000))]
    page = cp.plot_trajectory_pages(run)[0]
    stats = page.tables["Curve statistics and coverage"]
    row = stats[(stats.updates==1000)&(stats.learning_rate==run.trials.iloc[0].learning_rate)&
                (stats.l2sp_lambda==run.trials.iloc[0].l2sp_lambda)]
    assert row["mean"].isna().all() and not row.complete.any()
    # If the entire trial has no observations, the other three partitions do not become "complete".
    run.trajectories = run.trajectories[run.trajectories.trial_name!=missing]
    stats = cp.plot_trajectory_pages(run)[0].tables["Curve statistics and coverage"]
    arm = stats[(stats.learning_rate==run.trials.iloc[0].learning_rate)&
                (stats.l2sp_lambda==run.trials.iloc[0].l2sp_lambda)]
    assert arm["mean"].isna().all()


def test_reliability_uses_complete_identically_paired_folds():
    run = research_campaign()
    pages = dg.plot_reliability(run)
    assert len(pages)==8  # 16 datasets, two per page, three calibration columns
    for page in pages:
        bins = page.tables["Probability bins and counts"]
        assert bins.groupby("dataset").role.nunique().eq(2).all()
        assert bins["count"].eq(50).all()
    run.evaluation.loc[(run.evaluation.test_dataset_id=="Dataset 00") &
        (run.evaluation.fold_idx==1)&(run.evaluation.source=="tabpfn-untuned"),"status"] = "FAIL"
    pages = dg.plot_reliability(run)
    assert all("Dataset 00" not in p.tables["Probability bins and counts"].dataset.values for p in pages)
    run.evaluation = pd.concat([run.evaluation,run.evaluation.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate evaluation"):
        cp.benchmark_effects(run)


def test_partial_benchmark_without_calibration_reference_stays_unavailable():
    run = research_campaign()
    run.evaluation = run.evaluation[run.evaluation.source.str.endswith("-trained")
                                    & ~np.isclose(run.evaluation.lr, 3e-7, atol=0, rtol=1e-8)]
    assert not run.evaluation.empty
    assert dg.plot_reliability(run) == []
    empty = cp._complete_folds(run.evaluation.iloc[:0], "roc_auc")
    assert empty.empty and empty.columns.equals(run.evaluation.columns)
    invalid = run.evaluation.assign(roc_auc=np.nan)
    assert cp._complete_folds(invalid, "roc_auc").empty


@pytest.mark.parametrize("track", ["pd","lgd"])
def test_future_project_views_are_bounded_and_have_complete_text(track):
    run = research_campaign(track)
    endpoint = cp.endpoint_effects(run)
    evaluated = cp.benchmark_effects(run)
    assert not endpoint.empty and not evaluated.empty
    groups = [
        cp.plot_trajectory_pages(run), cp.plot_trajectory_pages(run,split="ood"),
        cp.plot_trajectory_pages(run,x="processed_rows"),
        cp.plot_response_surfaces(endpoint,run.metric),cp.plot_dataset_pages(endpoint,run.metric),
        pub.plot_recipe_distributions(endpoint,run.metric),pub.plot_retention_tradeoff(run),
        pub.plot_trainable_fraction(run),pub.plot_drift_effect(run),pub.plot_metric_profile(run),
        pub.plot_thresholds(run),cp.plot_train_test(run),
        cp.plot_secondary_tradeoff(run),cp.plot_control_context(run),cp.plot_cost_effect(run),
        cp.plot_contrasts(cp.factor_contrasts(endpoint,"l2sp_lambda",0.,.003),run.metric,"L2-SP on minus off"),
        cp.plot_optimization(run),cp.plot_optimization(run,"clipped_frac"),
    ]
    if track=="pd":
        groups.append(dg.plot_reliability(run))
    qa = os.environ.get("CREDITPFN_FIGURE_QA")
    sink = FigureSaver(f"general/qa_synthetic_{track}") if qa else None
    for pages in groups:
        for page in pages:
            assert reporting.page_text(page)
            assert page.figure.get_figwidth()==pytest.approx(style.WIDTH_FULL)
            assert page.figure.get_figheight()<=style.MAX_HEIGHT
            page.figure.canvas.draw()
            if sink:
                sink.save(page.figure,page.name,caption="SYNTHETIC SOFTWARE TEST. "+page.caption)
            plt.close(page.figure)


def test_unknown_trajectory_split_is_not_silently_empty():
    with pytest.raises(ValueError,match="split"):
        cp.effects(research_campaign(),split="retention")


def test_sampling_seed_and_resource_views_use_their_own_evidence(monkeypatch):
    main, repeat, sampling = research_campaign(), research_campaign(experiment=2), research_campaign(experiment=3)
    pairs = cp.seed_pairs(main,repeat)
    assert len(pairs)==16*6
    assert not cp.seed_pairs(main,repeat,benchmark=True).empty
    records = pd.DataFrame([
        dict(trial_name=t.trial_name,base=t.base,frozen=t.frozen,successful_updates=u,
             relative_change=(0.01+i/1000)*u/main.target,phase="training",gpu_status="ok",
             gpu_utilization_percent=70+i,power_watts=400+i,device_memory_used_mib=10000+i)
        for t in main.trials.itertuples() for u in main.cfg.train.trajectory_steps for i in range(4)])
    monkeypatch.setattr(dg,"load",lambda campaign,kind: records)
    groups = [
        cp.plot_sampling_trajectories(sampling), cp.plot_sampling_trajectories(sampling,row_exposure=True),
        pub.plot_sampling_endpoint(sampling),pub.plot_sampling_endpoint(sampling,benchmark=True),
        cp.plot_sampling_cost(sampling),cp.plot_seed_pairs(pairs,main.metric,endpoint=main.target),
        cp.plot_seed_trajectories(pairs,main.metric,milestones=main.cfg.train.trajectory_steps),
        dg.plot_parameters(main),dg.plot_resources(main),
    ]
    sink = FigureSaver("general/qa_synthetic_protocols") if os.environ.get("CREDITPFN_FIGURE_QA") else None
    for pages in groups:
        assert pages
        for page in pages:
            assert reporting.page_text(page)
            page.figure.canvas.draw()
            if sink:
                sink.save(page.figure,page.name,caption="SYNTHETIC SOFTWARE TEST. "+page.caption)
    missing = sampling.trials.iloc[0].trial_name
    sampling.trajectories = sampling.trajectories[sampling.trajectories.trial_name!=missing]
    stats = cp.plot_sampling_trajectories(sampling)[0].tables["Protocol curve statistics and coverage"]
    arm = stats[stats.sampling==sampling.trials.iloc[0].sampling]
    assert not arm.complete.any() and arm["mean"].isna().all()


def test_reliability_requires_bins_for_both_models():
    run = research_campaign()
    run.evaluation.loc[(run.evaluation.source=="tabpfn-untuned") &
                      (run.evaluation.test_dataset_id=="Dataset 00"),"calibration_bins"]="[]"
    tables = [p.tables["Probability bins and counts"] for p in dg.plot_reliability(run)]
    raw = pd.concat(tables).query("calibration == 'raw'")
    assert not raw.empty and "Dataset 00" not in raw.dataset.values


def test_reliability_cannot_pool_four_calibration_folds_as_five():
    run = research_campaign()
    selected = ((run.evaluation.source == "tabpfn-untuned") &
                (run.evaluation.test_dataset_id == "Dataset 00") & (run.evaluation.fold_idx == 1))
    run.evaluation.loc[selected, "calibration_bins_platt"] = "[]"
    pages = dg.plot_reliability(run)
    bins = pd.concat([p.tables["Probability bins and counts"] for p in pages])
    assert bins[(bins.dataset == "Dataset 00") & (bins.calibration == "Platt")].empty
    assert not bins[(bins.dataset == "Dataset 00") & (bins.calibration == "raw")].empty
    assert any("unavailable" in text.get_text() for p in pages for ax in p.figure.axes for text in ax.texts)


def test_partial_response_surface_keeps_counts_on_the_correct_cells():
    run=research_campaign()
    data=cp.endpoint_effects(run)
    data=data[~(data.learning_rate.eq(3e-7)&data.l2sp_lambda.eq(0))]
    pages=cp.plot_response_surfaces(data,run.metric)
    for page in pages:
        matrix=page.tables["Mean effect"]
        counts=page.tables["Dataset counts"]
        assert matrix.isna().to_numpy().sum()==1
        assert matrix.index.equals(counts.index) and matrix.columns.equals(counts.columns)
        assert len(page.figure.axes[0].texts)==7
        assert all(text.get_text().endswith("n=16") for text in page.figure.axes[0].texts)


def calibrated_campaign():
    run = research_campaign()
    adapted = run.evaluation.source.str.endswith("-trained")
    for metric, baseline in (("brier_score", .2), ("ece", .04), ("log_loss", .5)):
        for index, suffix in enumerate(("", "_platt", "_isotonic")):
            run.evaluation[metric + suffix] = baseline - index*.005 - adapted*.01
    return run


def test_calibration_overview_uses_equal_table_paired_effects_and_prints_every_value():
    run = calibrated_campaign()
    pages = dg.plot_calibration_overview(run)
    assert len(pages) == 1
    stats = pages[0].tables["Plotted means and dataset quartiles"]
    values = pages[0].tables["Per-dataset probability-quality changes"]
    assert len(stats) == 9 and len(values) == 16*9
    assert stats.tables.eq(16).all()
    np.testing.assert_allclose(stats[["mean", "q25", "q75"]], .01)
    np.testing.assert_allclose(values.untuned - values.trained, values.effect)
    assert "not confidence intervals" in pages[0].caption
    assert "Per-dataset probability-quality changes" in reporting.page_text(pages[0])
    assert pages[0].figure.get_size_inches()[0] == style.WIDTH_FULL


def test_calibration_overview_keeps_one_common_cohort_for_all_nine_scores():
    run = calibrated_campaign()
    selected = (run.evaluation.test_dataset_id.eq("Dataset 00")
                & run.evaluation.source.str.endswith("-untuned") & run.evaluation.fold_idx.eq(1))
    run.evaluation.loc[selected, "ece_isotonic"] = np.nan
    values = dg.calibration_effects(run)
    assert "Dataset 00" not in set(values.dataset)
    assert values.groupby(["metric", "calibration"]).size().eq(15).all()
    # The fixed recipe cannot borrow scores from other learning rates or OOD tables.
    irrelevant = run.evaluation.source.str.endswith("-trained") & ~np.isclose(run.evaluation.lr, 3e-7, atol=0)
    run.evaluation.loc[irrelevant, "brier_score"] = 0
    run.evaluation = pd.concat([run.evaluation, run.evaluation.assign(domain="noncredit", test_dataset_id="OOD")])
    pd.testing.assert_frame_equal(values, dg.calibration_effects(run))


def test_calibration_overview_reports_missing_evidence_and_rejects_duplicate_folds():
    run = research_campaign()
    assert dg.plot_calibration_overview(run) == []
    assert dg.plot_calibration_overview(research_campaign("lgd")) == []
    run = calibrated_campaign()
    run.evaluation = pd.concat([run.evaluation, run.evaluation.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate evaluation"):
        dg.plot_calibration_overview(run)


def test_reliability_examples_are_selected_by_size_not_performance():
    run = research_campaign()
    run.evaluation["n_test_rows"] = run.evaluation.test_dataset_id.str[-2:].astype(int) + 1
    # Force a very different ranking by score; size-based selection must not change.
    run.evaluation["roc_auc"] = 1 - run.evaluation.n_test_rows/100
    pages = dg.plot_reliability(run, max_datasets=style.CALIBRATION_EXAMPLE_TABLES)
    assert len(pages) == 1
    roster = pages[0].tables["Example-selection roster"]
    assert len(roster) == 16
    assert roster.loc[roster.selected, "dataset"].tolist() == ["Dataset 00", "Dataset 15"]
    assert roster.loc[roster.selected, "outer_test_rows"].tolist() == [5, 80]
    assert set(pages[0].tables["Probability bins and counts"].dataset) == {"Dataset 00", "Dataset 15"}
    assert "independently of model scores" in pages[0].caption


def test_reliability_examples_do_not_guess_unknown_sizes_or_change_full_view():
    run = research_campaign()
    assert dg.plot_reliability(run, max_datasets=2) == []
    assert len(dg.plot_reliability(run)) == 8
    with pytest.raises(ValueError, match="positive integer"):
        dg.plot_reliability(run, max_datasets=0)


def test_lgd_results_do_not_offer_classification_only_calibration_panels():
    root = Path(__file__).resolve().parents[1]
    book = json.loads((root / "notebooks/rq1/02_results.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in book["cells"])
    assert "finish.plot_lgd_distribution(main)" in text
    assert "dg.plot_reliability" not in text and "pub.plot_thresholds" not in text
