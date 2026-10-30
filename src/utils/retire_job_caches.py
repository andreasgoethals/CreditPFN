"""Preview or retire only terminal CreditPFN jobs' regenerable legacy caches.

Run inside Slurm on the cache's native cluster. Failed/unknown jobs are retained.
No datasets, runtime/input caches, checkpoints or scientific output are eligible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from src.utils.atomic import write_json

CHILDREN = {'nv','triton','matplotlib','torch','huggingface','pip','wandb'}


def root():
    cluster=os.environ.get('SLURM_CLUSTER_NAME')
    if not os.environ.get('SLURM_JOB_ID') or cluster not in ('mindwell','wice'):
        raise RuntimeError('Cache inspection/retirement requires a native Slurm allocation')
    key='VSC_SCRATCH_GPFS1' if cluster=='mindwell' else 'VSC_SCRATCH_LUSTRE1'
    native = Path(os.environ[key]).resolve(strict=True)
    cache = native / 'CreditPFN' / 'cache'
    for part in (cache.parent, cache):
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise ValueError('Linked legacy cache root; retain for inspection')
    if not cache.resolve().is_relative_to(native):
        raise ValueError('Legacy cache escaped native scratch')
    return cache


def terminal_jobs():
    cluster=os.environ['SLURM_CLUSTER_NAME']
    since=(datetime.now()-timedelta(days=30)).strftime('%Y-%m-%dT00:00:00')
    args=['sacct','-M',cluster,'-X','-u',os.environ['USER'],'-S',since,
          '--format=JobIDRaw,JobName%60,State','-P','-n']
    result=subprocess.run(args,text=True,capture_output=True,check=True,timeout=180)
    records={}
    for line in result.stdout.splitlines():
        if not line.strip():continue
        job,name,state,*_=line.split('|')
        records.setdefault(job,[]).append((name,state))
    return {job for job,states in records.items() if all(
        name.startswith('creditpfn-') and state=='COMPLETED' for name,state in states)}


def no_active_consumers():
    cluster=os.environ['SLURM_CLUSTER_NAME'];own=os.environ['SLURM_JOB_ID']
    queue=subprocess.run(['squeue','-M',cluster,'-u',os.environ['USER'],'-h','-o','%i|%j'],
                         text=True,capture_output=True,check=True,timeout=45).stdout
    for line in queue.splitlines():
        if not line.strip():continue
        job,name=line.split('|',1)
        if name.startswith('creditpfn-') and job.split('_')[0]!=own:
            raise RuntimeError('CreditPFN consumers still exist on this scratch cluster')


def inventory(folder):
    if folder.is_symlink() or not folder.name.isdigit() or folder.parent!=root():
        raise ValueError('Cache path is not an exact owned legacy job directory')
    if folder.stat().st_uid != os.getuid():
        raise ValueError('Cache directory belongs to another user')
    if any(p.name not in CHILDREN for p in folder.iterdir()):
        raise ValueError('Unexpected cache contents; retain for inspection')
    count=total=0;digest=hashlib.sha256()
    for p in sorted(folder.rglob('*')):
        if p.is_symlink():raise ValueError('Linked cache contents; retain for inspection')
        stat=p.stat()
        if stat.st_uid!=os.getuid():raise ValueError('Cache contains another user\'s files')
        if p.is_file():
            count+=1;total+=stat.st_size
            digest.update(json.dumps([str(p.relative_to(folder)),stat.st_size,stat.st_mtime_ns]).encode())
    return dict(files=count,bytes=total,inventory_sha256=digest.hexdigest())


def inspect():
    no_active_consumers()
    finished=terminal_jobs();eligible=[];retained=[]
    for p in sorted(root().iterdir()):
        if not p.is_dir() or p.name not in finished:
            retained.append(dict(path=str(p),reason='not a verified completed CreditPFN job'));continue
        try:entry=inventory(p)
        except ValueError as error:
            retained.append(dict(path=str(p),reason=str(error)));continue
        eligible.append(dict(path=str(p),job=p.name,**entry))
    return dict(root=str(root()),eligible=eligible,retained=retained)


def apply(plan, receipt):
    no_active_consumers()
    if plan['root']!=str(root()):raise ValueError('Cleanup plan belongs to another native cache root')
    finished=terminal_jobs()
    plan_sha256 = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    progress=json.loads(receipt.read_text()) if receipt.exists() else dict(plan_sha256=plan_sha256,removed=[])
    if progress.get('plan_sha256') != plan_sha256:
        raise ValueError('Removal receipt belongs to another cleanup plan')
    for entry in plan['eligible']:
        if entry['path'] in progress['removed']:continue
        folder=Path(entry['path'])
        if entry['job']!=folder.name or entry['job'] not in finished:
            raise ValueError('Cache job identity or terminal state changed')
        expected={k:entry[k] for k in ('files','bytes','inventory_sha256')}
        if inventory(folder)!=expected:raise ValueError('Cache changed after inventory; retain it')
        shutil.rmtree(folder)
        progress['removed'].append(str(folder));write_json(receipt,progress)
    return progress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.apply:
        result=apply(json.loads(args.plan.read_text()),args.plan.with_name(args.plan.stem+'.removed.json'))
        print('Retired',len(result['removed']),'verified completed-job caches')
    else:
        report=inspect();write_json(args.plan,report,exclusive=True)
        print('Eligible',len(report['eligible']),'bytes',sum(e['bytes'] for e in report['eligible']),
              'retained',len(report['retained']))


if __name__=='__main__':main()
