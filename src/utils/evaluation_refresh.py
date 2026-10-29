"""Isolated, explicitly authorized FP32 re-evaluation of completed TabICL campaigns.

Historical training plans are never rewritten. A new gate binds their audited
checkpoints and inputs to the accepted container and the current evaluator. Only
TabICL precision and output placement change; tests, CV, ensembles and seeds stay fixed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from src.utils.atomic import write_json
from src.utils.experiment import apply_split_index, digest_json, file_digest, scientific_config
from src.utils.paths import REPO_ROOT, manifests_dir, results_dir, activate_experiment, resolve_base_checkpoint
from src.utils.runtime_validation import identifier, identity

CONFIGS=("config/experiment3/pd.yaml", "config/experiment3/lgd.yaml", "config/experiment6/pd.yaml",
         "config/experiment7/pd_rows2048.yaml", "config/experiment7/pd_rows26000.yaml")
OPTIONS={"use_amp":False,"use_fa3":False}


def root(value):
    return manifests_dir("experiment0")/"evaluation_refresh"/identifier(value)


def clean_eval(cfg):
    raw=OmegaConf.to_container(cfg,resolve=True)
    raw.pop("train_cfg_path",None)
    raw.get("results",{}).pop("base_dir",None)
    return raw


def expected_eval(path):
    from src.eval.config import load_eval_configs
    cfg,_=load_eval_configs([],[],config_path=str(path))
    cfg.tabicl_inference_options=OPTIONS
    if cfg.cv.n_folds!=5 or not cfg.results.save_predictions:
        raise ValueError("Refresh must retain five folds and every test prediction")
    return cfg


def verify_original_protocol(original, proposed):
    """Historical size-study plans froze the benchmark before training."""
    recorded = original.get("evaluation_protocol", {}).get("config")
    if recorded is None:
        return False  # Older sampling plans lack it; independent row audits remain mandatory.
    actual = dict(proposed)
    actual.pop("tabicl_inference_options", None)
    before = dict(recorded)
    before.pop("tabicl_inference_options", None)
    if actual != before:
        raise RuntimeError("Historical benchmark settings changed beyond TabICL precision")
    return True


def _bind(path):
    path=Path(path).resolve();stat=path.stat()
    return dict(path=str(path),sha256=file_digest(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def verify(bindings, *, full_hash=False):
    for item in bindings:
        path=Path(item["path"]);stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(item["size"],item["mtime_ns"]) or (full_hash and file_digest(path)!=item["sha256"]):
            raise RuntimeError(f"Re-evaluation input changed: {path}")


def prepare(value, runtime_version):
    from src.utils.runtime_validation import require
    from src.utils.audit_experiment import audit
    from src.utils.prepare_experiment import read_plan, plan_path, check_shared_reference
    from src.train.config import load_train_config
    from src.train.corpus import build_dataset_pool
    from src.data.retention import load_refs
    from src.eval.context_curve import prepare as prepare_context
    runtime=require(runtime_version)
    folder=root(value)
    write_json(folder/"workflow.json",dict(prepared=False,id=value),exclusive=True)
    bindings={};campaigns=[];tasks=[]
    for source in CONFIGS:
        cfg=load_train_config(config_path=source)
        activate_experiment(cfg)
        report=audit(Path(source))
        if not report["passed"] or report["diverged"]:
            raise RuntimeError(f"Original completed training failed the fresh audit: {source}")
        path=plan_path(cfg.run_name,cfg.track)
        original=read_plan(path);check_shared_reference(original)
        stored=OmegaConf.create(original["config"])
        if (scientific_config(cfg)!=scientific_config(stored)
                or OmegaConf.to_container(cfg.tunable,resolve=True)!=OmegaConf.to_container(stored.tunable,resolve=True)):
            raise RuntimeError("Training recipe differs from its original plan")
        sources={r.dataset_id:r.processed_csv for r in [*build_dataset_pool(str(cfg.track)),
                    *load_refs(str(cfg.train.retention_panel),str(cfg.track))]}
        for spec in original["identities"].values():
            for dataset in [*(d for bucket in spec["datasets"].values() for d in bucket),*spec["retention"]]:
                file=sources[dataset["id"]]
                if file_digest(file)!=dataset["sha256"]:
                    raise RuntimeError("Dataset content changed since historical training")
                bindings[str(file)]=_bind(file)
            base=resolve_base_checkpoint(spec["trial"][0])
            if file_digest(base)!=spec["base_sha256"]:
                raise RuntimeError("Original foundation checkpoint changed")
            bindings[str(base)]=_bind(base)
        for trial in report["trials"]:
            for file in (Path(trial["path"]),Path(trial["path"]+".provenance.json")):
                bindings[str(file)]=_bind(file)
        for file in (Path(source),path):bindings[str(file.resolve())]=_bind(file)
        campaign=dict(config=source,run=str(cfg.run_name),track=str(cfg.track),
                      training_plan_sha256=original["sha256"],
                      evaluation=clean_eval(expected_eval(source)))
        campaign["historical_protocol_verified"] = verify_original_protocol(original, campaign["evaluation"])
        campaigns.append(campaign)
        write_json(folder/(cfg.run_name+"_"+cfg.track+"_audit.json"),report)
        for split in range(int(cfg.corpus.n_splits)):
            file=manifests_dir()/f"{cfg.run_name}_s{split:02d}_{cfg.track}.csv"
            bindings[str(file)]=_bind(file)
            tasks.extend(dict(kind="benchmark",config=source,split=split,shard=i,shards=16) for i in range(16))
    context=OmegaConf.load(REPO_ROOT/"config/experiment4/context.yaml")
    context.update(run_name=f"context_fp32_{value}",refresh_id=value,families=["tabicl"],tabicl_inference_options=OPTIONS)
    context_path=folder/"context.yaml"
    OmegaConf.save(context,context_path)
    context_plan,n=prepare_context(context_path)
    bindings[str(context_path)]=_bind(context_path)
    bindings[str(context_plan)]=_bind(context_plan)
    tasks.extend(dict(kind="context",plan=str(context_plan),index=i) for i in range(n))
    record=dict(prepared=True,id=value,identity=identity(),runtime_version=runtime_version,
                runtime_acceptance_sha256=file_digest(root_runtime(runtime_version)/"accepted.json"),
                campaigns=campaigns,context_config=OmegaConf.to_container(context,resolve=True),
                bindings=list(bindings.values()),tasks=tasks)
    write_json(folder/"workflow.json",record)
    print(f"Prepared {len(tasks)} independent FP32 evaluation tasks; historical results retained.",flush=True)
    return record


def root_runtime(version):
    from src.utils.runtime_validation import root as runtime_root
    return runtime_root(version)


def read(value, *, full_hash=False):
    from src.utils.runtime_validation import require
    record=json.loads((root(value)/"workflow.json").read_text())
    if record.get("prepared") is not True or record.get("id")!=value or record.get("identity")!=identity():
        raise RuntimeError("Re-evaluation preparation is missing or stale")
    require(record["runtime_version"])
    if file_digest(root_runtime(record["runtime_version"])/"accepted.json")!=record["runtime_acceptance_sha256"]:
        raise RuntimeError("Re-evaluation runtime acceptance changed")
    verify(record["bindings"],full_hash=full_hash)
    return record


def validate(train_cfg, eval_cfg):
    value=os.environ.get("CREDITPFN_EVAL_REFRESH","")
    record=read(value)
    from src.train.config import load_train_config
    candidates=[c for c in record["campaigns"] if str(train_cfg.run_name).startswith(c["run"]+"_s") and train_cfg.track==c["track"]]
    if len(candidates)!=1:
        raise RuntimeError("Campaign is outside the authorized re-evaluation")
    campaign=candidates[0]
    split=int(str(train_cfg.run_name).rsplit("_s",1)[1])
    expected=apply_split_index(load_train_config(config_path=campaign["config"]),split)
    if OmegaConf.to_container(train_cfg,resolve=True)!=OmegaConf.to_container(expected,resolve=True):
        raise RuntimeError("Refresh changed the training roster or settings")
    if clean_eval(eval_cfg)!=campaign["evaluation"] or os.environ.get("CREDITPFN_EVAL_FAMILY")!="tabicl":
        raise RuntimeError("Refresh changed settings beyond the approved TabICL precision")
    destination=results_dir(experiment=str(train_cfg.experiment.output_group)).parent/"reevaluations"/value/"results"
    if Path(eval_cfg.results.base_dir).resolve()!=destination.resolve():
        raise RuntimeError("Refresh results must remain in their isolated directory")
    return record


def validate_context(cfg):
    record=read(cfg["refresh_id"])
    if cfg!=record["context_config"]:
        raise RuntimeError("Context refresh changed its fixed protocol")


def run(value,task):
    record=read(value)
    if task not in range(len(record["tasks"])):raise ValueError("Invalid refresh task")
    item=record["tasks"][task]
    os.environ["CREDITPFN_EVAL_REFRESH"]=value
    if item["kind"]=="context":
        from src.eval.context_curve import run_task
        run_task(item["plan"],item["index"])
    else:
        from src.eval.config import load_eval_configs
        from scripts.eval_pipeline import run as evaluate
        eval_cfg,train_cfg=load_eval_configs([],[],config_path=item["config"],split_index=item["split"])
        activate_experiment(train_cfg)
        eval_cfg.tabicl_inference_options=OPTIONS
        eval_cfg.results.base_dir=str(results_dir().parent/"reevaluations"/value/"results")
        os.environ.update(CREDITPFN_EVAL_FAMILY="tabicl",CREDITPFN_EVAL_KIND="foundation")
        overrides=["tabicl_inference_options.use_amp=false", "tabicl_inference_options.use_fa3=false",
                   "results.base_dir="+str(eval_cfg.results.base_dir)]
        from src.utils.paths import logs_dir
        status=evaluate(eval_overrides=overrides,config=item["config"],split_index=item["split"],
                        task_index=item["shard"],n_tasks=item["shards"],
                        log_path=logs_dir("experiment0")/f"refresh_{value}_{task:04d}.log")
        if status:raise RuntimeError(f"Refresh benchmark failed with {status}")
    write_json(root(value)/"completed"/f"task_{task:04d}.json",
               dict(passed=True,task=task,workflow_sha256=digest_json(record)))


def inspect(value):
    """Check every receipt, independently recompute metrics, then bind publication inputs."""
    from src.utils import evaluation_evidence as evidence
    record=read(value,full_hash=True)
    for task in range(len(record["tasks"])):
        expected=dict(passed=True,task=task,workflow_sha256=digest_json(record))
        if json.loads((root(value)/"completed"/f"task_{task:04d}.json").read_text())!=expected:
            raise RuntimeError("A refresh task is incomplete or belongs to another workflow")
    groups={}
    from src.train.config import load_train_config
    for campaign in record["campaigns"]:
        cfg=load_train_config(config_path=campaign["config"])
        group=str(cfg.experiment.output_group)
        groups[group]=results_dir(experiment=group)
    reports={};bindings=[]
    for group,original in groups.items():
        fresh=original.parent/"reevaluations"/value/"results"
        reports[group]=evidence.benchmark(original,fresh)
        bindings.extend(_bind(file) for file in fresh.rglob("*") if file.is_file())
    context_task=next(t for t in record["tasks"] if t["kind"]=="context")
    plan=json.loads(Path(context_task["plan"]).read_text())
    context_results=results_dir(experiment="experiment4")/plan["config"]["run_name"]
    context_receipts=manifests_dir("experiment4")/"completed"/plan["config"]["run_name"]
    reports["context"]=evidence.context(plan,context_results,context_receipts)
    bindings.extend(_bind(file) for folder in (context_results,context_receipts) for file in folder.rglob("*") if file.is_file())
    result=dict(passed=True,workflow_sha256=digest_json(record),reports=reports,bindings=bindings)
    write_json(root(value)/"inspection.json",result)
    print(json.dumps(reports,indent=2),flush=True)
    return result


def _tree(path):
    return {str(file.relative_to(path)):file_digest(file) for file in sorted(path.rglob("*")) if file.is_file()}


def promote(value):
    """Journaled rename, preserving every historical file outside the active result tree."""
    from src.utils.clean_run import check_cluster_writers
    from src.utils.paths import consolidated_dir
    from src.utils.consolidate_output import consolidate
    from src.train.config import load_train_config
    record=read(value,full_hash=True)
    checked=json.loads((root(value)/"inspection.json").read_text())
    if checked.get("passed") is not True or checked.get("workflow_sha256")!=digest_json(record):
        raise RuntimeError("Independent refresh inspection is missing or stale")
    check_cluster_writers()
    journal=root(value)/"publication.json"
    if journal.is_file():
        progress=json.loads(journal.read_text())
        if progress["workflow_sha256"]!=digest_json(record):
            raise RuntimeError("Publication journal belongs to another refresh")
        if progress.get("complete"):
            return progress
    else:
        verify(checked["bindings"],full_hash=True)
        progress=dict(workflow_sha256=digest_json(record),moves=[],complete=False)
        groups={str(load_train_config(config_path=c["config"]).experiment.output_group) for c in record["campaigns"]}
        for group in sorted(groups):
            active=results_dir(experiment=group)
            fresh=active.parent/"reevaluations"/value/"results"
            archive=active.parent/"reevaluations"/value/"superseded"/"results"
            for track in ("PD","LGD"):
                old={p.name:p for p in (active/track).glob("tabicl-*") if p.is_dir()}
                new={p.name:p for p in (fresh/track).glob("tabicl-*") if p.is_dir()}
                if old.keys()!=new.keys():raise RuntimeError("Refreshed model directory coverage changed")
                for name in sorted(old):
                    for source,target in ((old[name],archive/track/name),(new[name],active/track/name)):
                        progress["moves"].append(dict(source=str(source),target=str(target),files=_tree(source),done=False))
        write_json(journal,progress)
    for move in progress["moves"]:
        source,target=Path(move["source"]),Path(move["target"])
        if move["done"]:
            if _tree(target)!=move["files"]:raise RuntimeError("Published or archived files changed")
            continue
        if source.exists():
            if target.exists() or _tree(source)!=move["files"]:
                raise RuntimeError("Ambiguous or changed publication source")
            target.parent.mkdir(parents=True,exist_ok=True)
            source.rename(target)
        if _tree(target)!=move["files"]:raise RuntimeError("Publication rename failed verification")
        move["done"]=True;write_json(journal,progress)
    consolidated = set()
    for campaign in record["campaigns"]:
        cfg=load_train_config(config_path=campaign["config"])
        group=str(cfg.experiment.output_group)
        if (group, campaign["run"]) in consolidated:
            continue
        pointer=consolidated_dir(group)/campaign["run"]/"LATEST.json"
        saved=pointer.with_name(f"LATEST.before-{value}.json")
        if pointer.is_file() and not saved.exists():pointer.rename(saved)
        consolidate(campaign["run"],apply=True)
        consolidated.add((group, campaign["run"]))
        write_json(manifests_dir(group)/"active_evaluation.json",
                   dict(family="tabicl", options=OPTIONS, refresh_id=value,
                        inspection_sha256=file_digest(root(value)/"inspection.json"),
                        workflow_sha256=digest_json(record)))
    cfg=record["context_config"]
    plan=manifests_dir("experiment4")/"plans"/(cfg["run_name"]+".json")
    write_json(manifests_dir("experiment4")/"active_context.json",
               dict(family="tabicl",config=cfg,plan_sha256=json.loads(plan.read_text())["sha256"],
                    inspection_sha256=file_digest(root(value)/"inspection.json")))
    progress["complete"]=True;write_json(journal,progress)
    return progress


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","run","inspect","promote"));p.add_argument("id")
    p.add_argument("--runtime-version");p.add_argument("--task",type=int)
    a=p.parse_args()
    if a.action=="prepare":prepare(a.id,a.runtime_version)
    elif a.action=="run":run(a.id,a.task)
    elif a.action=="inspect":inspect(a.id)
    else:promote(a.id)


if __name__=="__main__":main()
