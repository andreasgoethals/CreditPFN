"""Compare the preserved Conda runtime and sealed container on identical GPU inputs.

These are small compatibility checks, not research estimates. Complete held-out
row coverage belongs to production context tasks and is checked there separately.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np

from src.utils.atomic import write_json
from src.utils.experiment import code_identity, digest_json, environment_versions, file_digest
from src.utils.paths import outputs_dir, resolve_staging_path, OUTPUT_DIR_NAME

CASES = [(track, model) for track in ("pd", "lgd") for model in ("v2", "v2.6", "v3", "tabicl")]


def identifier(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 40 or not value.isascii() or not value.isalnum():
        raise ValueError("Runtime version must contain 1–40 ASCII letters/digits")
    return value


def root(version):
    # Runtime acceptance is shared infrastructure, never an experiment/adapter
    # sub-scope. Preserve older source-specific receipts when code is updated.
    return (outputs_dir() / "experiment0" / "manifests" / "runtime_acceptance"
            / identifier(version) / code_identity(stage="eval"))


def evidence(version):
    return (resolve_staging_path(OUTPUT_DIR_NAME) / "experiment0" / "results"
            / "runtime_acceptance" / identifier(version) / code_identity(stage="eval"))


def identity():
    return dict(code=code_identity(stage="eval"), versions=environment_versions(stage="eval"))


def sample(version, task, arm):
    import torch
    from src.eval.context_curve import policy, reference_checkpoint, fold_indices, score_cell
    from src.eval.dataset_loader import load_processed_dataset
    if task not in range(len(CASES)) or arm not in ("original", "container"):
        raise ValueError("Unknown runtime check")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU acceptance requires actual CUDA")
    contained=bool(os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"))
    if contained != (arm == "container"):
        raise RuntimeError("Runtime arm is not executing in the requested environment")
    track, model=CASES[task]
    family="tabicl" if model=="tabicl" else "tabpfn"
    cfg=policy()
    cfg.update(base_version=model, tabicl_inference_options=dict(use_amp=False,use_fa3=False))
    pair, refs=reference_checkpoint(track,family,0,cfg)
    # Deterministic smallest held-out table reduces setup; never selects by score.
    ref=min(refs, key=lambda r:(r.n_rows,r.dataset_id))
    data=load_processed_dataset(track,ref.dataset_id)
    order,val,test=fold_indices(data.y,data.task_type,0,cfg)
    order,val,test=order[:128],val[:64],test[:128]
    arrays={}; rows=[]
    for state in ("base","cpt"):
        torch.manual_seed(42); np.random.seed(42)
        metrics,blob=score_cell(data,order,val,test,pair[state],family,cfg)
        with np.load(io.BytesIO(blob),allow_pickle=False) as saved:
            arrays[state]=saved["prediction"]
            arrays["target"]=saved["target"]
        rows.append(dict(state=state,metrics=metrics))
    folder=evidence(version);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"{task}_{arm}.npz"
    np.savez_compressed(path,**arrays)
    record=dict(task=task,arm=arm,track=track,model=model,identity=identity(),
                gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
                rows=len(test), dataset=ref.dataset_id,
                indices=digest_json(dict(context=order.tolist(),validation=val.tolist(),test=test.tolist())),
                checkpoints={k:v for k,v in pair.items() if k.endswith("sha256")},
                prediction_path=str(path), prediction_sha256=file_digest(path), scores=rows)
    write_json(root(version)/f"{task}_{arm}.json",record)
    print(f"Runtime sample {task} {arm}: {track} {model}, {len(test)} finite predictions per state",flush=True)


def compare_task(version, task):
    records=[json.loads((root(version)/f"{task}_{arm}.json").read_text()) for arm in ("original","container")]
    for key in ("task","track","model","identity","gpu","cuda","rows","dataset","indices","checkpoints"):
        if records[0][key]!=records[1][key]:
            raise RuntimeError(f"Runtime comparison changed {key}")
    if records[0]["identity"] != identity():
        raise RuntimeError("Runtime comparison source/environment is stale")
    arrays=[]
    for r in records:
        if file_digest(Path(r["prediction_path"]))!=r["prediction_sha256"]:
            raise RuntimeError("Runtime prediction evidence changed")
        with np.load(r["prediction_path"],allow_pickle=False) as data:
            arrays.append({k:data[k] for k in data.files})
    if not np.array_equal(arrays[0]["target"],arrays[1]["target"]):
        raise RuntimeError("Runtime target rows changed")
    differences={}
    for state in ("base","cpt"):
        a,b=arrays[0][state],arrays[1][state]
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=1e-6,rtol=1e-5):
            raise RuntimeError(f"Container changed {state} predictions beyond declared tolerance")
        differences[state]=float(np.max(abs(a-b)))
    result=dict(task=task, passed=True, identity=identity(), differences=differences,
                records=[file_digest(root(version)/f"{task}_{arm}.json") for arm in ("original","container")])
    write_json(root(version)/f"{task}_passed.json",result)
    print(json.dumps(result),flush=True)
    return result


def cpu(version, specification):
    import torch
    from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score, mean_squared_error
    from src.utils.runtime_container import compare
    compare(specification)
    x=torch.tensor([1.,2.,3.],requires_grad=True)
    (x*x).sum().backward()
    assert torch.equal(x.grad,torch.tensor([2.,4.,6.]))
    assert roc_auc_score([0,0,1,1],[.1,.2,.8,.9])==1.
    assert np.isclose(brier_score_loss([0,0,1,1],[.1,.2,.8,.9]),.025)
    assert r2_score([0,1,2],[0,1,2])==1. and mean_squared_error([0,1,2],[0,1,2])==0.
    buffer=io.BytesIO();torch.save(x.detach(),buffer);buffer.seek(0)
    assert torch.equal(torch.load(buffer,weights_only=True),x.detach())
    receipt=dict(passed=True,identity=identity(),specification_sha256=file_digest(specification))
    write_json(root(version)/"cpu.json",receipt)
    print("PASS: exact recorded packages, CPU tensor/autograd/serialization and metric checks",flush=True)


def capacity(version, track):
    """Exercise the unchanged production context/query shape under explicit TabICL FP32."""
    import pandas as pd
    import torch
    from src.eval.context_curve import policy,reference_checkpoint,score_cell
    from src.eval.dataset_loader import ProcessedDataset
    from src.train.corpus import build_dataset_pool
    if track not in ("pd","lgd") or not torch.cuda.is_available():
        raise RuntimeError("FP32 capacity validation needs a real GPU and known track")
    cfg=policy();cfg["tabicl_inference_options"]={"use_amp":False,"use_fa3":False}
    pair,_=reference_checkpoint(track,"tabicl",0,cfg)
    cap,query=cfg["max_context_rows"],cfg["query_batch_rows"]
    features=max([64]+[len(pd.read_csv(r.processed_csv,nrows=0).columns)-1 for r in build_dataset_pool(track)])
    rng=np.random.default_rng(518)
    x=rng.normal(size=(cap+2*query,features)).astype(np.float32)
    y=(x[:,0]>0).astype(int) if track=="pd" else x[:,0]
    data=ProcessedDataset(pd.DataFrame(x),y,[],"classification" if track=="pd" else "regression","synthetic_fp32",track)
    context,validation,test=np.arange(cap),np.arange(cap,cap+query),np.arange(cap+query,len(x))
    cells=[]
    for state in ("base","cpt"):
        torch.cuda.reset_peak_memory_stats()
        metrics,blob=score_cell(data,context,validation,test,pair[state],"tabicl",cfg)
        with np.load(io.BytesIO(blob),allow_pickle=False) as saved:
            if not np.array_equal(saved["test"],test):raise RuntimeError("FP32 capacity test truncated test rows")
        cells.append(dict(state=state,peak_bytes=torch.cuda.max_memory_allocated(),**metrics))
    record=dict(passed=True,identity=identity(),track=track,context_rows=cap,query_rows=query,
                features=features,options=cfg["tabicl_inference_options"],cells=cells)
    write_json(root(version)/f"capacity_{track}.json",record)
    print(json.dumps(record),flush=True)


def accept(version, runtime):
    from src.utils.runtime_container import sha256
    runtime=Path(runtime).resolve()
    manifest=json.loads((runtime/"runtime.json").read_text())
    cpu_record=json.loads((root(version)/"cpu.json").read_text())
    if (cpu_record.get("passed") is not True or cpu_record.get("identity")!=identity()
            or cpu_record.get("specification_sha256")!=manifest["environment_sha256"]):
        raise RuntimeError("CPU acceptance is missing or stale")
    for name,expected in manifest["files"].items():
        if sha256(runtime/name)!=expected:
            raise RuntimeError("Sealed runtime contents changed")
    for name,target in manifest.get("symlinks",{}).items():
        if os.readlink(runtime/name)!=target:raise RuntimeError("Sealed runtime symlink changed")
    checks=[compare_task(version,task) for task in range(len(CASES))]
    if any(c["identity"]!=identity() for c in checks):
        raise RuntimeError("Acceptance source/environment changed")
    capacities={}
    for track in ("pd","lgd"):
        file=root(version)/f"capacity_{track}.json"
        case=json.loads(file.read_text())
        if case.get("passed") is not True or case.get("identity")!=identity():
            raise RuntimeError("Explicit-FP32 production-shape check is missing or stale")
        capacities[track]=file_digest(file)
    record=dict(version=version,runtime=str(runtime),passed=True,identity=identity(),
                cpu_sha256=file_digest(root(version)/"cpu.json"),
                capacities=capacities,
                manifest_sha256=sha256(runtime/"runtime.json"),
                files={name:dict(size=(runtime/name).stat().st_size,mtime_ns=(runtime/name).stat().st_mtime_ns)
                       for name in manifest["files"]},
                checks={str(t):file_digest(root(version)/f"{t}_passed.json") for t in range(len(CASES))})
    write_json(root(version)/"accepted.json",record)
    return record


def require(version):
    record=json.loads((root(version)/"accepted.json").read_text())
    if record.get("passed") is not True or record.get("version")!=version or record.get("identity")!=identity():
        raise RuntimeError("Container acceptance is missing or stale")
    runtime=Path(record["runtime"])
    if file_digest(root(version)/"cpu.json")!=record["cpu_sha256"]:
        raise RuntimeError("CPU acceptance evidence changed")
    if file_digest(runtime/"runtime.json")!=record["manifest_sha256"]:
        raise RuntimeError("Container manifest changed")
    manifest=json.loads((runtime/"runtime.json").read_text())
    for name,target in manifest.get("symlinks",{}).items():
        if os.readlink(runtime/name)!=target:raise RuntimeError("Accepted runtime symlink changed")
    for name,expected in record["files"].items():
        stat=(runtime/name).stat()
        if dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)!=expected:
            raise RuntimeError("Accepted runtime artifact changed")
    for task,digest in record["checks"].items():
        if file_digest(root(version)/f"{task}_passed.json")!=digest:
            raise RuntimeError("Runtime acceptance evidence changed")
    for track,digest in record["capacities"].items():
        if file_digest(root(version)/f"capacity_{track}.json")!=digest:
            raise RuntimeError("FP32 capacity evidence changed")
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("sample","compare","cpu","capacity","accept","require"))
    p.add_argument("version");p.add_argument("--task",type=int);p.add_argument("--arm")
    p.add_argument("--runtime",type=Path);p.add_argument("--specification",type=Path)
    p.add_argument("--track",choices=("pd","lgd"))
    a=p.parse_args()
    if a.action=="sample":sample(a.version,a.task,a.arm)
    elif a.action=="compare":compare_task(a.version,a.task)
    elif a.action=="cpu":cpu(a.version,a.specification)
    elif a.action=="capacity":capacity(a.version,a.track)
    elif a.action=="accept":print(json.dumps(accept(a.version,a.runtime)))
    else:print(json.dumps(require(a.version)))


if __name__=="__main__":main()
