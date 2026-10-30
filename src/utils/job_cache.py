"""Owned, disposable library caches; datasets and recovery state are never stored here."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shutil

MARKER = '.creditpfn-job-cache.json'


def identity():
    job = os.environ.get('SLURM_JOB_ID', '')
    restart = os.environ.get('SLURM_RESTART_COUNT', '0')
    cluster = os.environ.get('SLURM_CLUSTER_NAME', '')
    if not job.isdigit() or not restart.isdigit() or cluster not in ('mindwell', 'wice'):
        raise ValueError('Cache ownership requires a known Slurm job/cluster')
    return dict(schema=1, job=job, restart=restart, cluster=cluster, user=getpass.getuser())


def allowed_roots():
    return [Path(os.environ[k]).resolve() for k in
            ('VSC_SCRATCH_NODE', 'VSC_SCRATCH_GPFS1', 'VSC_SCRATCH_LUSTRE1') if os.environ.get(k)]


def guard(path):
    path = Path(path)
    if path.is_symlink() or not any(path.resolve().is_relative_to(p) and path.resolve() != p
                                    for p in allowed_roots()):
        raise ValueError('Disposable cache must be strictly inside approved scratch')
    # Reject links in intermediate components too; never redirect owned cleanup.
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('Linked cache path is unsafe')
    return path.resolve()


def create():
    record = identity()
    user = record['user']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', user):
        raise ValueError('Invalid cache owner')
    override = os.environ.get('CREDITPFN_CACHE_ROOT')
    native = 'VSC_SCRATCH_GPFS1' if record['cluster'] == 'mindwell' else 'VSC_SCRATCH_LUSTRE1'
    candidates = ([Path(override)] if override else [Path(os.environ[k]) / 'CreditPFN' / 'job_cache' /
        user / (record['job'] + '-r' + record['restart']) for k in ('VSC_SCRATCH_NODE', native) if os.environ.get(k)])
    for path in candidates:
        path = guard(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            marker = path / MARKER
            if marker.exists():
                if marker.is_symlink() or json.loads(marker.read_text()) != record:
                    raise ValueError('Cache belongs to another job or restart')
            else:
                if any(path.iterdir()):
                    raise ValueError('Refusing to claim a populated cache directory')
                with marker.open('x', encoding='utf-8') as stream:
                    json.dump(record, stream)
            return path
        except PermissionError:
            if override or path == candidates[-1].resolve():
                raise
            print('Node-local cache unavailable; using owned native-scratch fallback', file=__import__('sys').stderr)
    raise RuntimeError('No writable job cache location')


def clean(path):
    path = guard(path)
    if not path.exists():
        return
    marker = path / MARKER
    if marker.is_symlink() or json.loads(marker.read_text()) != identity():
        raise ValueError('Refusing cleanup without exact job ownership')
    shutil.rmtree(path)
    print('Removed disposable job cache:', path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('create', 'clean'))
    parser.add_argument('path', nargs='?', type=Path)
    args = parser.parse_args()
    if args.action == 'create':
        print(create())
    else:
        if args.path is None:
            parser.error('clean requires the exact owned path')
        clean(args.path)


if __name__ == '__main__':
    main()
