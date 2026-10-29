"""One resumable submission for runtime acceptance, migration and approved additions.

Every dependency is a scientific or publication prerequisite. Independent ready
tasks have no running throttle. Failed gates block dependants; existing campaigns
and the original runtime remain recoverable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.utils.atomic import write_json
from src.utils.experiment import file_digest
from src.utils.paths import outputs_dir, results_dir, manifests_dir, resolve_staging_path, OUTPUT_DIR_NAME
from src.utils.runtime_validation import identifier, require


def root(value):
    return manifests_dir("experiment0") / "study_v2" / identifier(value)


def runtime_plan(value, version, specification):
    cpu = ["scripts/slurm/runtime_cpu.slurm"]
    return [
        dict(key="runtime_prepare", needs=[], tasks=None, args=[*cpu,"prepare",version,str(specification)]),
        dict(key="runtime_gpu", needs=["runtime_prepare"], tasks=list(range(10)),
             args=["scripts/slurm/runtime_gpu.slurm",version]),
        dict(key="runtime_accept", needs=["runtime_gpu"], tasks=None,args=[*cpu,"accept",version]),
        dict(key="launch", needs=["runtime_accept"], tasks=None,
             args=["--time=08:00:00",f"--export=ALL,CREDITPFN_RUNTIME_VERSION={version}",
                   "scripts/slurm/study_cpu.slurm","launch",value]),
    ]


def submit(value, version, specification):
    from src.utils.legacy_submit import submit_plan
    from src.utils.clean_run import check_cluster_writers
    identifier(value); identifier(version)
    specification = Path(specification).resolve(strict=True)
    folder = root(value)
    source = dict(id=value,runtime_version=version,specification=str(specification),
                  specification_sha256=file_digest(specification))
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "runtime.submit.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        marker = folder/"workflow.json"
        if marker.exists():
            if json.loads(marker.read_text())!=source:raise RuntimeError("Study workflow identity changed")
        else:
            check_cluster_writers()
            write_json(marker,source,exclusive=True)
        return submit_plan(runtime_plan(value,version,specification),folder/"runtime.submission.json")
    finally:
        lock.unlink()


def settings(value):
    record=json.loads((root(value)/"workflow.json").read_text())
    if record["id"]!=value:raise RuntimeError("Study workflow ID mismatch")
    require(record["runtime_version"])
    return record


def launch(value):
    from src.utils.clean_run import check_cluster_writers
    from src.utils.study_layout import migrate
    from src.utils import evaluation_refresh as refresh, legacy_submit as legacy
    record=settings(value)
    check_cluster_writers()
    report=migrate([outputs_dir(),resolve_staging_path(OUTPUT_DIR_NAME)],apply=True,
                   workers=min(8,int(os.environ.get("SLURM_CPUS_PER_TASK","1"))))
    write_json(root(value)/"migration.json",report)
    # Prepare only after paths have reached their final, verified locations.
    if not (refresh.root(value)/"workflow.json").exists():
        prepared=refresh.prepare(value,record["runtime_version"])
    else:
        prepared=refresh.read(value,full_hash=True)
    refresh_plan=[dict(key="refresh",needs=[],tasks=list(range(len(prepared["tasks"]))),
                       args=["scripts/slurm/refresh_gpu.slurm",value])]
    legacy.submit_plan(refresh_plan,root(value)/"refresh.submission.json")
    legacy.submit(value,"tests")
    test_state=json.loads((legacy.root(value)/"tests.submission.json").read_text())
    prerequisites=[job["id"] for job in test_state["accepted"] if job["key"]=="inspect"]
    if not prerequisites:raise RuntimeError("No independent safety inspection was submitted")
    launch_plan=[dict(key="research",needs=[],tasks=None,
                      args=["--dependency=afterok:"+":".join(prerequisites),"--kill-on-invalid-dep=yes",
                            "--time=12:00:00","scripts/slurm/study_cpu.slurm","research",value])]
    legacy.submit_plan(launch_plan,root(value)/"research-controller.submission.json")


def research(value):
    from src.utils import legacy_submit as legacy
    settings(value)
    legacy.submit(value,"research")
    states=[json.loads(p.read_text()) for p in (root(value)/"refresh.submission.json",
            legacy.root(value)/"research.submission.json")]
    jobs=[r["id"] for state in states for r in state["accepted"]]
    # Even failed tasks trigger an audit; it will refuse publication. Any cancelled
    # dependency is made terminal by --kill-on-invalid-dep on dependent arrays.
    finish=[dict(key="finish",needs=[],tasks=None,args=["--dependency=afterany:"+":".join(jobs),
             "--time=12:00:00","scripts/slurm/study_cpu.slurm","finish",value])]
    legacy.submit_plan(finish,root(value)/"finish.submission.json")


def audit_extension(value):
    """Expected roster, all five folds and independently recomputed saved scores."""
    from src.utils import legacy_checks, evaluation_evidence as evidence
    from src.utils.legacy_design import scope
    from src.utils.audit_experiment import audit
    from src.eval.config import load_eval_configs
    from src.eval.benchmark import _method_dirname
    from src.eval.dataset_loader import load_processed_dataset
    from src.eval.benchmark import _make_outer_folds
    from scripts.eval_pipeline import _build_roster,_enumerate_tasks
    workflow=legacy_checks.require_gate(value)
    reports={};expected={};groups={};context_runs=[];protocols={}
    os.environ.pop("CREDITPFN_EVAL_REFRESH",None)
    os.environ.update(CREDITPFN_EVAL_FAMILY="tabpfn",CREDITPFN_EVAL_KIND="foundation")
    for key,path in sorted(workflow["production"].items()):
        if key.startswith("e4_"):
            context_runs.append(path);continue
        with scope(value):
            report=audit(Path(path))
            if not report["passed"] or report["diverged"]:raise RuntimeError(f"Incomplete extension training: {key}")
            reports[key]=dict(completed=report["completed"],passed=True)
            for split in range(4):
                evaluation,training=load_eval_configs([],[],config_path=path,split_index=split)
                group=str(training.experiment.output_group)
                with scope(value,group):
                    roster,tables,_=_build_roster(evaluation,training,training.track)
                    groups[group]=results_dir(experiment=group)
                    protocols[(group,training.track.upper(),str(training.run_name))]=dict(
                        folds=int(evaluation.cv.n_folds),seed=int(evaluation.seed))
                    for index,table in _enumerate_tasks(roster,tables):
                        handle=roster[index][0]
                        expected.setdefault(group,set()).add((training.track.upper(),_method_dirname(handle),
                            str(training.run_name),handle.name,table))
    for group,directory in groups.items():
        found=evidence.inventory(directory,family="tabpfn")
        if found.keys()!=expected[group]:
            raise RuntimeError(f"Extension {group}: incomplete or unexpected benchmark comparisons")
        predictions=0
        datasets={}
        for key,(file,metrics) in found.items():
            rows=evidence.predictions(file)
            rows=rows[rows.model_name.eq(key[3]) & rows.test_dataset_id.eq(key[4])]
            if rows.row_idx.duplicated().any():raise RuntimeError("Extension outer test folds overlap")
            dataset_key=(key[0],key[4])
            if dataset_key not in datasets:
                datasets[dataset_key]=load_processed_dataset(key[0].lower(),key[4])
            data=datasets[dataset_key]
            protocol=protocols[(group,key[0],key[2])]
            folds=list(_make_outer_folds(data.y,task_type=data.task_type,
                       n_folds=protocol["folds"],seed=protocol["seed"]))
            import numpy as np
            if set(rows.row_idx)!=set(range(len(data.y))):
                raise RuntimeError("Extension test predictions do not cover every eligible dataset row")
            for record in metrics.to_dict("records"):
                part=rows[rows.fold_idx.eq(record["fold_idx"])]
                if len(part)!=record["n_test_rows"]:raise RuntimeError("Incomplete extension predictions")
                wanted=folds[int(record["fold_idx"])][1]
                ordered=part.sort_values("row_idx")
                if (not np.array_equal(ordered.row_idx.to_numpy(),np.sort(wanted))
                        or not np.allclose(ordered.y_true.to_numpy(),data.y[np.sort(wanted)],rtol=1e-7,atol=1e-10)):
                    raise RuntimeError("Extension test membership or targets differ from the fixed five folds")
                evidence.check_scores(record,part.y_true.to_numpy(),part.y_pred.to_numpy(),key[0].lower())
                predictions+=len(part)
        reports[group]=dict(comparisons=len(found),folds=5*len(found),predictions=predictions,passed=True)
    from omegaconf import OmegaConf
    for path in context_runs:
        cfg=OmegaConf.load(path)
        with scope(value,"experiment4"):
            plan=json.loads((manifests_dir("experiment4")/"plans"/(cfg.run_name+".json")).read_text())
            reports[cfg.run_name]=evidence.context(plan,results_dir(experiment="experiment4")/cfg.run_name,
                    manifests_dir("experiment4")/"completed"/cfg.run_name)
    write_json(root(value)/"extension-audit.json",dict(passed=True,reports=reports))
    return reports


def finish(value):
    from src.utils import evaluation_refresh as refresh
    from src.utils.legacy_design import scope
    from src.utils.legacy_checks import require_gate
    from src.utils.consolidate_output import consolidate
    from src.train.config import load_train_config
    settings(value)
    audit_extension(value)
    os.environ.pop("CREDITPFN_LEGACY_ID",None)
    refresh.inspect(value)
    refresh.promote(value)
    workflow=require_gate(value)
    seen=set()
    for key,path in sorted(workflow["production"].items()):
        if key.startswith("e4_"):continue
        cfg=load_train_config(config_path=path)
        if cfg.run_name in seen:continue
        with scope(value,str(cfg.experiment.output_group)):
            consolidate(str(cfg.run_name),apply=True)
        seen.add(cfg.run_name)
    write_json(root(value)/"complete.json",dict(passed=True,
        extension_audit_sha256=file_digest(root(value)/"extension-audit.json"),
        refresh_publication_sha256=file_digest(refresh.root(value)/"publication.json")))
    print("COMPLETE: migrated storage, accepted runtime, additions audited, FP32 results published.",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("submit","launch","research","finish"));p.add_argument("id")
    p.add_argument("--runtime-version");p.add_argument("--specification",type=Path)
    a=p.parse_args()
    if a.action=="submit":submit(a.id,a.runtime_version,a.specification)
    else:globals()[a.action](a.id)


if __name__=="__main__":main()
