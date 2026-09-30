"""Reader-facing names, balanced pages and compact pilot evidence."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data import dataset_names
from src.visualize import campaign as cp, corpus, pilots, reporting, style


def test_public_dataset_aliases_work_without_a_private_mapping(monkeypatch):
    monkeypatch.setattr(dataset_names, "_DISPLAY", {})
    assert dataset_names.display_name("0015.credit_risk_dataset") == "Credit Risk"
    assert dataset_names.display_name("0016.bondora_peer2peer") == "Bondora Peer-to-Peer"
    assert dataset_names.display_name("0017.SBA_loans_case") == "SBA Loans Case"
    assert dataset_names.display_name("0008.SBA_loans_case") == "SBA Loans Case (LGD)"
    assert dataset_names.display_name("unknown") == "unknown"


def test_public_model_names_are_idempotent_and_do_not_change_file_ids():
    text = "v2 / v2.6 / v3 / tabicl-v2"
    labels = "TabPFN v2 / TabPFN v2.6 / TabPFN v3 / TabICLv2"
    assert style.model_label(text) == labels
    assert style.model_label(labels) == labels
    filename = "cpt_main_v5_pd_tabpfn-v3-classifier-v3_default_lr3e-7"
    assert style.model_label(filename) == filename
    assert "TabPFN v2" in reporting.table_text(pd.DataFrame({"base": ["v2"]}))


@pytest.mark.parametrize("length,limit", [(17,12), (16,12), (25,12), (5,4), (17,20), (0,12)])
def test_balanced_pages_cover_every_item_once(length, limit):
    pages = list(style.page_slices(length, limit))
    sizes = [stop-start for start, stop in pages]
    assert [i for start, stop in pages for i in range(start,stop)] == list(range(length))
    if sizes:
        assert max(sizes) <= limit and max(sizes)-min(sizes) <= 1


def test_target_bars_and_single_pd_assignment_figure():
    style.apply()
    data = pd.DataFrame({"track": "pd", "dataset_id": [f"Dataset {i}" for i in range(17)],
                         "minority_class_ratio": np.linspace(.02,.49,17), "target_mean": np.nan})
    pages = corpus.plot_target_profiles(data)
    assert len(pages) == 1
    assert len(pages[0].figure.axes[0].patches) == 17
    assert pages[0].figure.axes[0].get_xlim()[0] == 0
    partition = pd.DataFrame({"track":"pd", "dataset":data.dataset_id,"fold":np.arange(17)%4+1})
    pages = corpus.plot_partitions(partition)
    assert len(pages) == 1 and pages[0].figure.axes[0].images[0].get_array().shape == (17,4)


def test_delivery_checks_recognize_actual_sanitized_filenames(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus.exploration, "_resolve_paths", lambda: {"processed": tmp_path})
    for track, name, target in [("pd", "0008.german", "target"), ("lgd", "0008.SBA_loans_case", "lgd")]:
        (tmp_path/track).mkdir()
        (tmp_path/track/(name+".sanitized.csv")).write_text("feature,"+target+"\n", encoding="utf8")
    data = pd.DataFrame(dict(track=["pd","lgd"], rows=[1000,2000],
        dataset_id=[dataset_names.display_name("0008.german"), dataset_names.display_name("0008.SBA_loans_case")],
        target_in_raw=[False,False]))
    checks = corpus.delivery_checks(data)
    assert checks.target_in_processed.all()
    assert checks.status.eq("Expected preprocessing").all()
    (tmp_path/"pd"/"0008.german.sanitized.csv").unlink()
    checks = corpus.delivery_checks(data)
    assert checks.loc[0,"status"] == "Inspect delivery"


def _short_campaign():
    cfg = OmegaConf.create({"track":"pd", "train":{"target_total_steps":250}})
    trials = pd.DataFrame([dict(trial_name=f"trial_{i}",base=base,learning_rate=3e-7,
        frozen=False,l2sp_lambda=.003,sampling="one_sample") for i,base in enumerate(["v2","v3"])])
    updates = [*range(13,250,13),250]
    history = pd.DataFrame([dict(trial_name=name,successful_updates=u,train_loss=.7-u/1000,
        lr_applied=3e-7*(1-u/300)) for name in trials.trial_name for u in updates])
    return cp.Campaign(cfg,trials,pd.DataFrame(),history,pd.DataFrame())


def test_short_history_uses_observed_cadence_and_one_shared_schedule():
    style.apply()
    run = _short_campaign()
    assert cp.optimization_window(run) == 13
    pages = pilots.plot_loss(run)
    assert len(pages) == 1
    for ax in pages[0].figure.axes:
        assert len(ax.lines)==1 and np.isfinite(ax.lines[0].get_ydata()).all()
    schedule = cp.plot_optimization(run,"lr_applied")
    assert len(schedule)==1 and len(schedule[0].figure.axes[0].lines)==1
    assert schedule[0].figure.axes[0].get_yscale() == "log"
    assert schedule[0].tables["Observed trial windows"].trial_name.nunique() == 2
    run.histories.loc[:,"train_loss"] = np.nan
    assert pilots.plot_loss(run) == []


def test_track_scopes_export_name_title_and_caption(monkeypatch):
    import IPython.display
    monkeypatch.setattr(IPython.display,"display",lambda *args: None)
    saved=[]
    sink=SimpleNamespace(save=lambda figure,name,caption: saved.append((name,caption)))
    for track in ("pd","lgd"):
        fig,ax=plt.subplots()
        fig.suptitle("Optimizer check")
        ax.plot([1,2],[3,4])
        cp.show(sink,[cp.Page("same_name",fig,"Recorded values.")],track=track)
        assert fig._suptitle.get_text().startswith(track.upper())
    assert [name for name,_ in saved] == ["pd_same_name","lgd_same_name"]
    assert [caption for _,caption in saved] == ["PD. Recorded values.","LGD. Recorded values."]


def test_profile_bars_and_shared_geometry_keep_both_tracks():
    data = pd.DataFrame(dict(track=["pd", "pd", "lgd"], dataset_id=["A", "B", "C"],
        rows=[100, 1000, 250], features=[3, 5, 4], missing=[0, .1, .2]))
    geometry = corpus.plot_geometry(data, "Processed")[0].figure
    assert len(geometry.axes) == 1 and len(geometry.axes[0].collections) == 2
    for page in corpus.plot_profiles(data, "Processed"):
        for ax in page.figure.axes:
            assert ax.patches and ax.get_xlim()[0] == 0 and ax.get_xscale() == "linear"


def test_exposure_merges_equal_markers_and_labels_step_share_ratio():
    data = pd.DataFrame(dict(track=["pd", "pd"], dataset_id=["A", "B"], rows=[100, 90000]))
    table = corpus.planned_exposure(data, row_cap=10000)
    assert table.full_pass_share_ratio.tolist() == pytest.approx([.2, 1.8])
    page = corpus.plot_exposure(table)[0]
    ax = page.figure.axes[0]
    assert len(ax.get_legend_handles_labels()[1]) == 2
    assert {text.get_text() for text in ax.texts} == {"0.20×", "1.80×"}


def test_partial_table_traversal_does_not_create_a_loss_cliff():
    run = _short_campaign()
    for i in range(13):
        run.histories[f"loss__table{i}"] = .5
        if i >= 3:
            run.histories.loc[run.histories.successful_updates.eq(250), f"loss__table{i}"] = np.nan
    run.histories.loc[run.histories.successful_updates.eq(250), "train_loss"] = .1
    support = cp.loss_epoch_support(run)
    assert support.loc[~support.complete_pass, "observed_tables"].eq(3).all()
    tables = cp.loss_coverage_tables(run)
    counts = tables["Loss coverage by trial"]
    assert len(counts) == run.trials.trial_name.nunique()
    assert counts.partial_coverage_epochs.eq(1).all()
    assert counts.recorded_epochs.sum() == len(support)
    assert tables["Partial-coverage epoch losses"].train_loss.eq(.1).all()
    assert cp.optimization_curves(run, "train_loss").window_end.max() == 247
    page = pilots.plot_loss(run)[0]
    assert page.tables["Partial-coverage epoch losses"].train_loss.eq(.1).all()
    assert all(ax.collections for ax in page.figure.axes)


def test_equal_configured_schedules_merge_despite_different_epoch_cadences():
    import copy
    pd_run = _short_campaign()
    lgd_run = copy.deepcopy(pd_run)
    lgd_run.cfg.track = "lgd"
    lgd_run.histories = lgd_run.histories[lgd_run.histories.successful_updates.mod(26).eq(0)]
    page = pilots.plot_schedule([pd_run, lgd_run])[0]
    assert len(page.figure.axes) == 1 and len(page.figure.axes[0].lines) == 1
    schedule = page.tables["Configured applied schedules"]
    assert schedule.tracks.eq("PD + LGD").all()
    assert schedule.iloc[0].applied_lr == 0
    assert set(page.tables["Observed trial windows"].track) == {"PD", "LGD"}


def test_device_curves_give_each_trial_one_contribution(monkeypatch):
    from src.visualize import diagnostics
    run = _short_campaign()
    data = pd.DataFrame([dict(trial_name="long", base="v2", frozen=False,
        sampling="one_sample", successful_updates=13, phase="training", gpu_utilization_percent=0.)]*100 +
        [dict(trial_name="short", base="v2", frozen=False, sampling="one_sample",
              successful_updates=13, phase="training", gpu_utilization_percent=100.)])
    monkeypatch.setattr(diagnostics, "load", lambda *args: data.copy())
    page = diagnostics.plot_resource_curves(run)[0]
    shown = page.tables["Window medians and trial counts"].dropna(subset=["median"])
    assert shown.iloc[0]["median"] == 50 and shown.iloc[0]["count"] == 2
