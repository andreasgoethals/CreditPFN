"""Compatibility gates and independent scientific-evidence checks, without GPUs."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from src.utils import evaluation_evidence as ev, runtime_container as rc, runtime_validation as rv


def test_runtime_comparison_requires_matching_rows_and_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(rv,"root",lambda _:tmp_path)
    monkeypatch.setattr(rv,"identity",lambda:{"source":"fixed"})
    records=[]
    for arm in ("original","container"):
        path=tmp_path/f"{arm}.npz"
        np.savez(path,base=[.1,.9],cpt=[.2,.8],target=[0,1])
        record=dict(task=0,track="pd",model="v2",identity=rv.identity(),gpu="fixture",cuda="13",
                    rows=2,dataset="fixture",indices="fixed",checkpoints={},prediction_path=str(path),
                    prediction_sha256=rv.file_digest(path))
        (tmp_path/f"0_{arm}.json").write_text(json.dumps(record));records.append(record)
    assert rv.compare_task("version",0)["passed"]
    np.savez(tmp_path/"container.npz",base=[.2,.9],cpt=[.2,.8],target=[0,1])
    records[1]["prediction_sha256"]=rv.file_digest(tmp_path/"container.npz")
    (tmp_path/"0_container.json").write_text(json.dumps(records[1]))
    with pytest.raises(RuntimeError,match="predictions beyond"):rv.compare_task("version",0)


@pytest.mark.parametrize("track",["pd","lgd"])
def test_score_recomputation_checks_both_required_metrics(track):
    y=np.array([0.,1.,0.,1.]);p=np.array([.1,.8,.2,.9])
    record=ev.scores(y,p,track)
    assert set(record)==({"roc_auc","brier_score"} if track=="pd" else {"r2","rmse"})
    assert ev.check_scores(record,y,p,track)==record
    record[next(iter(record))]+=.02
    with pytest.raises(ValueError,match="disagree"):ev.check_scores(record,y,p,track)
    with pytest.raises(ValueError,match="nonfinite"):ev.scores(y,[np.nan,0,0,0],track)


def benchmark_fixture(root, *, shift=0.):
    folder=root/"PD/tabicl-trained__fixture";folder.mkdir(parents=True)
    model="tabicl-trained[fixture]"
    predictions=[];metrics=[]
    for fold in range(5):
        y=np.array([0.,1.]);p=np.array([.1,.9])+shift
        metrics.append(dict(model_name=model,test_dataset_id="table",fold_idx=fold,n_train_rows=6,n_val_rows=2,
                            n_test_rows=2,split_seed=99,domain="credit",status="OK",**ev.scores(y,p,"pd")))
        predictions.extend(dict(model_name=model,test_dataset_id="table",fold_idx=fold,row_idx=2*fold+i,
                                y_true=y[i],y_pred=p[i]) for i in range(2))
    file=folder/"cpt_fixture_s00_20261029_120000.csv"
    pd.DataFrame(metrics).to_csv(file,index=False)
    target=file.with_suffix(".csv.gz");pd.DataFrame(predictions).to_csv(target,index=False)
    file.with_name(file.name+".evaluation.json").write_text(json.dumps({"__predictions__":dict(file=target.name,bytes=target.stat().st_size)}))
    return file,target


def test_refresh_allows_changed_predictions_but_not_changed_test_rows(tmp_path):
    old,_=benchmark_fixture(tmp_path/"old")
    new,pred=benchmark_fixture(tmp_path/"new",shift=.01)
    assert ev.benchmark(tmp_path/"old",tmp_path/"new")["predictions"]==10
    df=pd.read_csv(pred);df.loc[0,"row_idx"]=100
    df.to_csv(pred,index=False)
    new.with_name(new.name+".evaluation.json").write_text(json.dumps({"__predictions__":dict(file=pred.name,bytes=pred.stat().st_size)}))
    with pytest.raises(AssertionError):ev.benchmark(tmp_path/"old",tmp_path/"new")


def test_refresh_rejects_missing_folds_and_mismatched_scores(tmp_path):
    old,_=benchmark_fixture(tmp_path/"old")
    new,pred=benchmark_fixture(tmp_path/"new")
    df=pd.read_csv(new);df.loc[0,"roc_auc"]-=.1;df.to_csv(new,index=False)
    with pytest.raises(ValueError,match="disagree"):ev.benchmark(tmp_path/"old",tmp_path/"new")
    df.iloc[:-1].to_csv(new,index=False)
    with pytest.raises(ValueError,match="incomplete"):ev.benchmark(tmp_path/"old",tmp_path/"new")


def test_all_output_helpers_use_shared_layout_without_double_mapping(tmp_path, monkeypatch):
    from src.utils import paths,study_layout
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT",str(tmp_path))
    monkeypatch.delenv("VSC_DATA",raising=False)
    for name in paths.STAGING_ENV_VARS:monkeypatch.delenv(name,raising=False)
    root=tmp_path/paths.OUTPUT_DIR_NAME;root.mkdir()
    (root/study_layout.MARKER).write_text(json.dumps(study_layout.manifest()))
    assert paths.results_dir(experiment="experiment1")==root/"shared/main_grid/results"
    assert paths.training_dir(experiment="experiment3")==root/"experiment7/training"
    assert paths.manifests_dir("experiment7")==root/"experiment9/manifests"
    monkeypatch.setenv("CREDITPFN_EXPERIMENT","experiment3")
    assert paths.resolve_staging_path("output CreditPFN/results")==root/"experiment7/results"
    assert paths.resolve_staging_path("output CreditPFN/experiment8/results")==root/"experiment6/results"


def test_refresh_protocol_allows_only_the_precision_change():
    from src.utils.evaluation_refresh import verify_original_protocol
    old={"evaluation_protocol":{"config":{"cv":{"n_folds":5},"tabicl_n_estimators":8}}}
    proposed={"cv":{"n_folds":5},"tabicl_n_estimators":8,"tabicl_inference_options":{"use_amp":False,"use_fa3":False}}
    assert verify_original_protocol(old,proposed)
    proposed["tabicl_n_estimators"]=4
    with pytest.raises(RuntimeError,match="beyond TabICL precision"):
        verify_original_protocol(old,proposed)
    assert not verify_original_protocol({},proposed)


def test_workflow_runtime_gates_precede_storage_changes_and_research():
    from src.utils.study_workflow import runtime_plan
    from src.utils.legacy_submit import test_plan
    plan=runtime_plan("studytest","runtime1",Path("environment.json"))
    assert [p["key"] for p in plan]==["runtime_prepare","runtime_gpu","runtime_accept","launch"]
    assert plan[1]["tasks"]==list(range(10))
    assert all(p["needs"]==[plan[i-1]["key"]] for i,p in enumerate(plan) if i)
    assert all("%" not in str(p["args"]) for p in plan+test_plan("studytest"))
    assert all("--mem" not in str(p["args"]) for p in plan+test_plan("studytest"))


def test_runtime_cache_checks_content_and_keeps_existing_corrupt_cache(tmp_path, monkeypatch):
    import sys
    from src.utils import runtime_cache as cache
    monkeypatch.setitem(sys.modules,"fcntl",NS(LOCK_EX=2,flock=lambda *args:None))
    runtime=tmp_path/"project";runtime.mkdir()
    (runtime/"image.sqfs").write_bytes(b"fixture image")
    (runtime/"python").write_bytes(b"wrapper")
    try:
        (runtime/"alias").symlink_to("python")
    except OSError:
        pytest.skip("Creating symlinks is unavailable for this local Windows account")
    manifest=dict(images=["image.sqfs"],files={name:rc.sha256(runtime/name) for name in ("image.sqfs","python")},symlinks={"alias":"python"})
    (runtime/"runtime.json").write_text(json.dumps(manifest))
    target=tmp_path/"scratch"/"runtime"
    assert cache.stage(runtime,target)==target
    assert cache.stage(runtime,target)==target
    (target/"image.sqfs").write_bytes(b"altered")
    with pytest.raises(RuntimeError,match="cache changed"):cache.stage(runtime,target)
    assert (target/"image.sqfs").read_bytes()==b"altered"
