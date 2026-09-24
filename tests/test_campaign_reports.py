"""Analysis contracts: bounded pages, matched comparisons and no cross-run leakage."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualize import campaign as cp, style
from src.train.config import load_train_config


def campaign(experiment=1, track="pd"):
    cfg = load_train_config(config_path=str(cp.config_path(experiment,track)))
    trials = cp.planned_trials(cfg).assign(status="PENDING")
    return cp.Campaign(cfg,trials,pd.DataFrame(),pd.DataFrame(),pd.DataFrame())


def dense_effects(datasets=17):
    rows = []
    for base in ("v2","v2.6","v3","tabicl"):
        for frozen in (False,True):
            for lr in (3e-7,1e-6,1e-5,3e-5):
                for lam in (0.,.003):
                    for dataset in range(datasets):
                        rows.append(dict(base=base,frozen=frozen,learning_rate=lr,l2sp_lambda=lam,
                            sampling="one_sample",dataset=f"Dataset {dataset+1:02d}",seed=42,
                            effect=(dataset-8)/1000 + lam - (0.002 if frozen else 0)))
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def close_figures():
    style.apply()
    yield
    plt.close("all")


def test_new_config_paths_preserve_grid_sizes_and_ids():
    for track in ("pd","lgd"):
        expected = [(1,None,256,"cpt_main_v5"),(2,None,16,"cpt_seeds_v5"),
                    (3,None,48,"cpt_sampling_v5"),(0,"null",8,"cpt_null_v5_check2"),
                    (0,"pilot",16,"cpt_pilot_v5_check2"),(0,"budget",4,"cpt_budget_v5_check2")]
        for experiment,phase,count,run in expected:
            cfg = load_train_config(config_path=str(cp.config_path(experiment,track,phase)))
            planned = cp.planned_trials(cfg)
            assert len(planned) == count and planned.trial_name.is_unique
            assert cfg.run_name == run


def test_dense_heatmaps_keep_every_dataset_and_recipe_on_bounded_pages():
    pages = cp.plot_dataset_pages(dense_effects(),"roc_auc")
    assert len(pages) == 4*2*2
    assert len({p.name for p in pages}) == len(pages)
    total_cells = 0
    for page in pages:
        axis = page.figure.axes[0]
        matrix = axis.images[0].get_array()
        assert matrix.shape[0] <= style.PAGE_ROWS
        assert matrix.shape[1] <= style.PAGE_COLUMNS
        assert page.figure.get_figwidth() == pytest.approx(style.WIDTH_FULL)
        assert page.figure.get_figheight() <= style.MAX_HEIGHT
        total_cells += matrix.size
        page.figure.canvas.draw()
    assert total_cells == len(dense_effects())


def test_factor_contrasts_match_other_knobs_and_do_not_confuse_missing_values():
    data = dense_effects(3)
    anchor = cp.factor_contrasts(data,"l2sp_lambda",0.,.003)
    assert len(anchor) == len(data)//2
    assert anchor.contrast.to_numpy() == pytest.approx(np.full(len(anchor),.003))
    frozen = cp.factor_contrasts(data,"frozen",False,True)
    assert frozen.contrast.to_numpy() == pytest.approx(np.full(len(frozen),-.002))
    unavailable = data[~data.l2sp_lambda.eq(.003)]
    assert cp.factor_contrasts(unavailable,"l2sp_lambda",0.,.003).empty


def test_seed_pairs_use_only_predefined_recipe_and_intersection(monkeypatch):
    main,repeat = campaign(),campaign(2)
    left = dense_effects(3).assign(updates=5000)
    right = left.copy()
    right["seed"] = 43
    right["effect"] += .01
    right = right[right.dataset.ne("Dataset 03")]
    monkeypatch.setattr(cp,"effects",lambda run: left if run is main else right)
    pairs = cp.seed_pairs(main,repeat)
    assert len(pairs) == 4*2
    assert set(pairs.dataset) == {"Dataset 01","Dataset 02"}
    assert pairs.difference.to_numpy() == pytest.approx(np.full(len(pairs),.01))
    assert set(pairs.learning_rate) == {3e-7} and set(pairs.l2sp_lambda) == {.003}


def evaluation_frame():
    rows = []
    for dataset,scale in (("Dataset 01",1.),("Dataset 02",1000.)):
        for source,value in (("tabpfn-untuned",1.),("tabpfn-trained",.8)):
            method = f"{source}__v3-default" + ("__lr3e-07__l2sp0.003" if source.endswith("-trained") else "")
            for fold in range(5):
                rows.append(dict(method_dirname=method,base_short="v3-default",source=source,
                    test_dataset_id=dataset,fold_idx=fold,eval_run="test_s00",split=0,
                    status="OK",rmse=scale*value))
    return pd.DataFrame(rows)


def test_benchmark_requires_complete_paired_folds_and_normalizes_rmse():
    run = campaign(track="lgd")
    run.evaluation = evaluation_frame()
    result = cp.benchmark_effects(run)
    assert len(result) == 2 and result.effect.tolist() == pytest.approx([.2,.2])
    mask = (run.evaluation.test_dataset_id.eq("Dataset 01") & run.evaluation.fold_idx.eq(3)
            & run.evaluation.source.eq("tabpfn-untuned"))
    run.evaluation.loc[mask,"status"] = "FAIL"
    assert cp.benchmark_effects(run).dataset.tolist() == ["Dataset 02"]


@pytest.mark.parametrize("bad_value", [np.inf, np.nan, 0.])
def test_one_invalid_reference_fold_excludes_the_entire_fractional_effect(bad_value):
    run = campaign(track="lgd")
    run.evaluation = evaluation_frame()
    mask = (run.evaluation.test_dataset_id.eq("Dataset 01") & run.evaluation.fold_idx.eq(3)
            & run.evaluation.source.eq("tabpfn-untuned"))
    run.evaluation.loc[mask, "rmse"] = bad_value
    assert cp.benchmark_effects(run).dataset.tolist() == ["Dataset 02"]


def test_analysis_rejects_a_wrong_fold_set_with_the_right_count():
    run = campaign(track="lgd")
    run.evaluation = evaluation_frame()
    run.evaluation.loc[run.evaluation.fold_idx.eq(0), "fold_idx"] = 5
    assert cp.benchmark_effects(run).empty


def test_nested_notebooks_are_discovered_and_hidden_backups_ignored(tmp_path,monkeypatch):
    from src.utils import run_notebooks as rn
    for path in ("experiment1/01_report.ipynb","experiment2/01_report.ipynb",
                 "00_general/01_data.ipynb","experiment1/.ipynb_checkpoints/01_report.ipynb"):
        p=tmp_path/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text("{}")
    monkeypatch.setattr(rn,"notebooks_dir",lambda:tmp_path)
    assert rn.discover() == ("00_general/01_data","experiment1/01_report","experiment2/01_report")
    assert rn.discover(("experiment2",)) == ("experiment2/01_report",)


def test_nested_figure_ownership_never_clears_sibling(isolated_output):
    from src.visualize.figures import FigureSaver, read_manifest
    first = FigureSaver("experiment1/01_report")
    fig,ax=plt.subplots(figsize=style.figsize())
    ax.plot([0,1],[0,1]); first.save(fig,"test",caption="A line.")
    FigureSaver("experiment2/01_report")
    assert first.last_path.exists()
    assert len(read_manifest("experiment1/01_report")) == 1
    FigureSaver("experiment1/01_report")
    assert not first.last_path.exists()


def test_external_analysis_does_not_redirect_generated_figures(tmp_path,monkeypatch,isolated_output):
    from src.visualize import eval_viz,training_viz
    from src.visualize.figures import FigureSaver
    external=tmp_path/"download"; external.mkdir()
    monkeypatch.setenv("CREDITPFN_ANALYSIS_ROOT",str(external))
    assert training_viz._resolve_paths()["manifest_dir"] == external/"general/manifests"
    assert eval_viz._resolve_paths()["benchmark_root"] == external/"general/results"
    sink=FigureSaver("experiment0/controls")
    assert not sink.folder.is_relative_to(external)
    assert list(external.iterdir()) == []


def test_downloaded_compact_snapshot_is_read_without_import(tmp_path, monkeypatch, isolated_output):
    import hashlib
    import shutil
    from src.utils.consolidate_output import consolidate
    from src.visualize.inputs import load_consolidated
    original = tmp_path / "cluster"
    manifests = original / "manifests"
    manifests.mkdir(parents=True)
    pd.DataFrame([dict(track="pd", status="OK", learning_rate=1e-6, use_lora=False,
                       base_checkpoint="base.ckpt", seed=42)]).to_csv(manifests / "test_s00_pd.csv", index=False)
    consolidate("test", apply=True, manifest_root=manifests, result_root=original/"results",
                destination=original/"consolidated")
    download = tmp_path / "download"
    shutil.copytree(original/"consolidated", download/"general/consolidated")
    monkeypatch.setenv("CREDITPFN_ANALYSIS_ROOT", str(download))
    before = {p.relative_to(download): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in download.rglob("*") if p.is_file()}
    frame = load_consolidated("test", "trials_pd")
    assert frame.status.tolist() == ["OK"]
    after = {p.relative_to(download): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in download.rglob("*") if p.is_file()}
    assert before == after


def test_all_notebooks_are_thin_ordered_and_use_owned_savers():
    notebooks=sorted(Path("notebooks").rglob("*.ipynb"))
    assert len(notebooks) == 11
    for path in notebooks:
        nb=json.loads(path.read_text(encoding="utf8"))
        code=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
        assert "print(report.summary(sink))" == code[-1].strip()
        assert f"FigureSaver('{path.relative_to('notebooks').with_suffix('').as_posix()}')" in "\n".join(code)
        for cell in code:
            tree=ast.parse("\n".join(line for line in cell.splitlines() if not line.startswith("%")))
            assert not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for n in ast.walk(tree))
            assert "figsize=" not in cell and "color=" not in cell


def test_no_result_figures_are_fabricated_for_empty_campaign():
    run=campaign()
    assert cp.plot_trajectory_pages(run) == []
    assert cp.plot_response_surfaces(cp.endpoint_effects(run),run.metric) == []
    assert cp.plot_diagnostics(run) == []
    assert cp.plot_cost_effect(run) == []
    assert cp.plot_coverage_grid(run) == []


def test_coverage_grid_distinguishes_failures_from_pending():
    run=campaign()
    run.trials.loc[0,"status"]="FAIL"
    pages=cp.plot_coverage_grid(run)
    assert len(pages)==1
    labels=[t.get_text() for t in pages[0].figure.axes[0].texts]
    assert "FAIL" in labels and "PENDING" in labels


def test_planned_exposure_distinguishes_chunks_from_table_visits():
    from src.visualize.corpus import planned_exposure
    data=pd.DataFrame(dict(dataset_id=["small","large"],track=["pd","pd"],rows=[10,90]))
    result=planned_exposure(data,row_cap=10)
    assert result.one_sample_step_share.tolist() == [.5,.5]
    assert result.accumulate_step_share.tolist() == [.5,.5]
    assert result.full_pass_step_share.tolist() == [.1,.9]


def test_optimization_bins_average_within_trial_before_summarizing_trials():
    run = campaign()
    names = run.trials.trial_name.iloc[:2].tolist()
    run.histories = pd.DataFrame([
        dict(trial_name=names[0], successful_updates=10, train_loss="1"),
        dict(trial_name=names[0], successful_updates=20, train_loss="9"),
        dict(trial_name=names[1], successful_updates=15, train_loss="4"),
    ])
    result = cp.optimization_curves(run, "train_loss")
    assert len(result) == 2
    assert result.set_index("trial_name").loc[names[0], "train_loss"] == 5.
    assert result.set_index("trial_name").loc[names[1], "train_loss"] == 4.
    assert result.window_end.nunique() <= style.CURVE_BINS
    pages = cp.plot_optimization(run)
    assert pages and all(p.figure.get_figheight() <= style.MAX_HEIGHT for p in pages)


def test_null_audit_does_not_promote_failed_log_to_pass(tmp_path,monkeypatch):
    cfg=load_train_config(config_path=str(cp.config_path(0,"pd","null")))
    run=cp.Campaign(cfg,cp.planned_trials(cfg),pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
    folder=tmp_path/"experiment0/logs"; folder.mkdir(parents=True)
    monkeypatch.setenv("CREDITPFN_ANALYSIS_ROOT",str(tmp_path))
    report=dict(run=str(cfg.run_name),track="pd",passed=True,trials=[dict(trial=run.trials.trial_name.iloc[0],null_state={"equal":True},null_monitor_equal=True,successful_updates=2)])
    (folder/"maintenance_01.log").write_text(json.dumps(report,indent=2)+"\nEND exit_code=1")
    assert not cp.null_audits([run]).audit_passed.any()
