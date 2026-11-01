"""Gated operational recovery: preserve finished science and resume only missing work."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from src.utils.atomic import write_json
from src.utils.experiment import file_digest
from src.utils.legacy_design import root as extension_root, scope, validate_id
from src.utils.paths import outputs_dir, manifests_dir, results_dir, resolve_staging_path, REPO_ROOT


def root(value):
    return outputs_dir() / 'experiment0/manifests/campaign_recovery' / validate_id(value)


@contextmanager
def writer(path):
    """Kernel-held lock: timeout/SIGKILL cannot leave an orphaned exclusive lease."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def record(value):
    return json.loads((root(value) / 'workflow.json').read_text())


def live_arrays():
    # %A is a running task's physical ID; %F retains its accepted parent array.
    output = subprocess.check_output(['squeue', '-M', 'mindwell', '-u', os.environ['USER'],
                                      '-h', '-o', '%F'], text=True)
    return set(output.split())


def bootstrap(value, original, gate, before, runtime, specification):
    from src.utils.legacy_submit import submit_plan
    from src.utils.source_transition import verify_delta
    validate_id(original); validate_id(gate)
    delta = verify_delta(REPO_ROOT, before)
    folder = root(value); folder.mkdir(parents=True, exist_ok=True)
    state = dict(id=value, original=original, gate=gate, before=before, runtime=runtime,
                 specification=str(Path(specification).resolve()), source=delta['after'])
    destination = folder / 'workflow.json'
    if destination.exists():
        if json.loads(destination.read_text()) != state:
            raise RuntimeError('Recovery preparation differs; preserve its receipt')
    else:
        write_json(destination, state, exclusive=True)
    gate_root = extension_root(gate); gate_root.mkdir(parents=True, exist_ok=True)
    marker = gate_root / '.scope.json'
    expected = dict(kind='creditpfn_legacy_scope', id=gate)
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise RuntimeError('Unknown validation scope ownership')
    else:
        write_json(marker, expected, exclusive=True)
    cpu = ['--partition=batch_graniterapids', '--cpus-per-task=4']
    plan = [
        dict(key='runtime_cpu', needs=[], tasks=None, args=[*cpu, 'scripts/slurm/runtime_cpu.slurm', 'prepare', runtime, specification]),
        dict(key='runtime_gpu', needs=['runtime_cpu'], tasks=list(range(10)), args=['scripts/slurm/runtime_gpu.slurm', runtime]),
        dict(key='runtime_accept', needs=['runtime_gpu'], tasks=None, args=[*cpu, 'scripts/slurm/runtime_cpu.slurm', 'accept', runtime]),
        dict(key='gate_prepare', needs=['runtime_accept'], tasks=None, args=[*cpu, 'scripts/slurm/legacy_cpu.slurm', 'prepare', gate]),
        dict(key='smoke', needs=['gate_prepare'], tasks=list(range(12)), args=['scripts/slurm/legacy_gpu.slurm', 'smoke', gate]),
        dict(key='capacity', needs=['smoke'], tasks=list(range(68)), args=['--time=03:00:00', 'scripts/slurm/legacy_gpu.slurm', 'capacity', gate]),
        dict(key='inspect', needs=['capacity'], tasks=None, args=[*cpu, 'scripts/slurm/legacy_cpu.slurm', 'inspect', gate]),
        dict(key='signal', needs=['runtime_accept'], tasks=None, args=['scripts/slurm/recovery_signal.slurm', value, str(folder/'signal.json')]),
        dict(key='recover', needs=['inspect', 'signal'], tasks=None, args=[*cpu, '--time=12:00:00', '--signal=B:USR1@600', '--requeue', 'scripts/slurm/recovery_cpu.slurm', 'recover', value]),
    ]
    with writer(folder / 'bootstrap.lease'):
        archive_failures(value)
        return submit_plan(plan, folder / 'bootstrap.submission.json')


def archive_failures(value):
    """Archive terminal failed logs, never checkpoints, completed cells or peers."""
    cfg = record(value); original = cfg['original']
    destination = root(value) / 'failed-output-retirement.json'
    if destination.exists():
        ledger = json.loads(destination.read_text())
        if file_digest(Path(ledger['archive'])) != ledger['sha256']:
            raise RuntimeError('Retained failure archive changed')
        for relative, digest in ledger['files'].items():
            p = outputs_dir() / relative
            if not p.resolve().is_relative_to(outputs_dir().resolve()):
                raise RuntimeError('Failure ledger escaped the output root')
            if p.exists():
                if file_digest(p) != digest:
                    raise RuntimeError('Failed log changed after archival')
                p.unlink()
                ledger['removed'].append(str(p))
                write_json(destination, ledger)
        return ledger
    scan_path = outputs_dir() / 'experiment0/manifests/study_v2' / original / 'progress20261101/audit.json'
    scan = json.loads(scan_path.read_text())
    chosen = [Path(x['path']) for x in scan['logs'] if x['counts'].get('severe')]
    # The completed-cell databases are retained. Only missing-END logs from the
    # accounting-confirmed timed-out context array are retired after cancellation.
    for item in scan['logs']:
        p = Path(item['path'])
        if '/experiment10/' in p.as_posix() and not item['counts'].get('success_end'):
            chosen.append(p)
    chosen += [outputs_dir()/'experiment0/logs'/f'study_research_{original}_11757172_r0.log',
               outputs_dir()/'experiment0/logs'/f'study_continuation_{original}_62483670.log']
    lock = extension_root(original) / 'research.submit.lock'
    if lock.exists():
        chosen.append(lock)
    chosen = list(dict.fromkeys(p for p in chosen if p.exists()))
    for p in chosen:
        if not p.resolve().is_relative_to(outputs_dir().resolve()) or p.is_symlink():
            raise RuntimeError('Failed-output cleanup escaped its owned output root')
        if p.suffix != '.log' and p != lock:
            raise RuntimeError('Recovery cleanup selected scientific output')
    # Original accepted tasks must be absent from Slurm before retiring their logs.
    original_state = json.loads((extension_root(original)/'research.submission.json').read_text())
    ids = {r['id'] for r in original_state['accepted']}
    recovery = outputs_dir()/'experiment0/manifests/study_v2'/original/'signal-recovery.submission.json'
    ids.update(r['id'] for r in json.loads(recovery.read_text())['accepted'])
    live = live_arrays()
    if ids.intersection(live):
        raise RuntimeError('Original jobs still consume their source/output')
    archive_dir = resolve_staging_path('output CreditPFN') / 'experiment0/archives' / value
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / 'failed-logs.tar.gz'
    entries = {str(p.relative_to(outputs_dir())): file_digest(p) for p in chosen}
    if not archive.exists():
        temporary = archive.with_suffix('.pending')
        # The kernel-held recovery lease excludes a live writer of this scope.
        temporary.unlink(missing_ok=True)
        with tarfile.open(temporary, 'w:gz') as tar:
            for p in chosen:
                tar.add(p, arcname=str(p.relative_to(outputs_dir())), recursive=False)
        temporary.rename(archive)
    with tarfile.open(archive) as tar:
        for name, digest in entries.items():
            content = tar.extractfile(name)
            if content is None or hashlib.sha256(content.read()).hexdigest() != digest:
                raise RuntimeError('Failure archive verification failed; originals preserved')
    ledger = dict(archive=str(archive), sha256=file_digest(archive), files=entries, removed=[])
    write_json(destination, ledger, exclusive=True)
    for p in chosen:
        if file_digest(p) != entries[str(p.relative_to(outputs_dir()))]:
            raise RuntimeError('Failed log changed during archival; preserve it')
        p.unlink()
        ledger['removed'].append(str(p))
        write_json(destination, ledger)
    return ledger


def missing_plan(value):
    """Build a new immutable roster; never reuse cancelled Slurm dependencies."""
    from src.utils import legacy_submit
    from src.utils.audit_experiment import audit
    cfg = record(value); original = cfg['original']
    workflow = json.loads((extension_root(original)/'workflow.json').read_text())
    reports = {}
    with scope(original):
        for key, path in workflow['production'].items():
            if key.startswith('e10_'):
                continue
            reports[key] = audit(Path(path))
            if reports[key]['problems'] or reports[key]['diverged']:
                raise RuntimeError(f'Existing scientific output failed its audit: {key}')
        full = legacy_submit.research_plan(original)
    training, contexts, ready, waiting = [], [], [], []
    for item in full:
        if item['key'].startswith('e10_'):
            plan = json.loads(Path(item['args'][-1]).read_text())
            with scope(original, 'experiment4'):
                receipts = manifests_dir('experiment4') / 'completed' / plan['config']['run_name']
                tasks = []
                for index in item['tasks']:
                    p = receipts / f'task_{index:04d}.json'
                    if not p.exists():
                        tasks.append(index)
                    else:
                        receipt = json.loads(p.read_text())
                        if (receipt.get('plan_sha256') != plan['sha256']
                                or receipt.get('passed') is not True or receipt.get('task') != index):
                            raise RuntimeError('Context completion belongs to another immutable plan or failed task')
            if tasks:
                contexts.append(dict(item, tasks=tasks, args=['--time=08:00:00', *item['args']]))
        elif not item['key'].endswith('_benchmark'):
            key, split = item['key'].rsplit('_s', 1)
            report = reports[key]; width = report['expected'] // 4
            rows = report['trials'][int(split)*width:(int(split)+1)*width]
            tasks = [i for i, r in enumerate(rows) if r['status'] == 'PENDING']
            if tasks:
                training.append(dict(item, tasks=tasks))
    pending = {item['key'] for item in training}
    for item in full:
        if not item['key'].endswith('_benchmark'):
            continue
        # No benchmark completed before this recovery. Refuse automatic re-run if
        # that premise changes rather than duplicating saved comparisons.
        needs = [key for key in item['needs'] if key in pending]
        (waiting if needs else ready).append(dict(item, needs=needs))
    for group in ('experiment5', 'experiment6', 'experiment8', 'experiment9'):
        directory = resolve_staging_path('output CreditPFN')/group/'extensions'/original/'results'
        if list(directory.glob('**/*.evaluation.json')):
            raise RuntimeError('Existing benchmark output requires shard-level reconciliation')
    return training + contexts + ready + waiting


def interruptible_sleep(seconds):
    from src.train.recovery import stop_requested
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if stop_requested():
            raise SystemExit(75)
        time.sleep(min(1, max(0, until-time.monotonic())))


def recover(value):
    from src.utils.source_transition import install
    from src.utils.legacy_submit import submit_plan
    cfg = record(value); folder = root(value)
    with writer(folder / 'research.lease'):
        install(cfg['original'], cfg['gate'], cfg['runtime'], cfg['before'], folder/'signal.json')
        archive_failures(value)
        roster = folder / 'research.plan.json'
        if roster.exists():
            plan = json.loads(roster.read_text())
        else:
            plan = missing_plan(value)
            write_json(roster, plan, exclusive=True)
        state = submit_plan(plan, folder/'research.submission.json', sleep=interruptible_sleep)
        ids = {r['id'] for r in state['accepted']}
        live = live_arrays()
        deps = sorted(ids.intersection(live))
        args = (['--dependency=afterany:'+':'.join(deps)] if deps else [])
        submit_plan([dict(key='finish', needs=[], tasks=None, args=[*args,
            '--partition=batch_graniterapids', '--time=00:15:00', 'scripts/slurm/recovery_cpu.slurm', 'dispatch', value])],
            folder/'finish.submission.json', sleep=interruptible_sleep)


def dispatch(value):
    """Same-cluster barrier first; only then hand native-Lustre audits to wICE."""
    from src.utils.legacy_submit import submit_plan
    state = json.loads((root(value)/'research.submission.json').read_text())
    if state['in_flight'] is not None or state['index'] != len(state['plan']):
        raise RuntimeError('Research submission is incomplete')
    live = live_arrays()
    if {r['id'] for r in state['accepted']}.intersection(live):
        raise RuntimeError('Research consumers remain; native-Lustre audit must wait')
    return submit_plan([dict(key='finish', needs=[], tasks=None, args=[
        '--partition=batch', '--time=12:00:00', 'scripts/slurm/recovery_cpu.slurm', 'finish', value])],
        root(value)/'wice-finish.submission.json', cluster='wice')


def finish(value):
    from src.utils import study_workflow
    from src.utils.source_transition import authorization
    cfg = record(value)
    # The study coordinator owns global manifests; it scopes each scientific
    # audit internally. An outer extension scope would redirect its run record.
    authorization(cfg['original'])
    study_workflow.finish(cfg['original'])
    write_json(root(value)/'complete.json', dict(passed=True, original=cfg['original'],
        final_sha256=file_digest(study_workflow.root(cfg['original'])/'complete.json')), exclusive=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['bootstrap', 'recover', 'dispatch', 'finish'])
    p.add_argument('id'); p.add_argument('--original'); p.add_argument('--gate')
    p.add_argument('--before'); p.add_argument('--runtime'); p.add_argument('--specification')
    a = p.parse_args()
    if a.action == 'bootstrap':
        bootstrap(a.id, a.original, a.gate, a.before, a.runtime, a.specification)
    else:
        globals()[a.action](a.id)


if __name__ == '__main__':
    main()
