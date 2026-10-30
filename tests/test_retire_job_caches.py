import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils import retire_job_caches as caches


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_CLUSTER_NAME', 'mindwell')
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    monkeypatch.setenv('USER', 'fixture')
    monkeypatch.setenv('VSC_SCRATCH_GPFS1', str(tmp_path))
    monkeypatch.setattr(os, 'getuid', lambda: tmp_path.stat().st_uid, raising=False)
    monkeypatch.setattr(caches, 'no_active_consumers', lambda: None)
    monkeypatch.setattr(caches, 'terminal_jobs', lambda: {'123', '124'})
    root = caches.root()
    for job in ['123', '124', '125']:
        (root / job / 'nv').mkdir(parents=True)
        (root / job / 'nv/kernel').write_bytes(b'cache')
    return root


def test_cleanup_preserves_unknown_failed_and_scientific_content(native, tmp_path):
    (native / '124/checkpoint.pt').write_bytes(b'important')
    plan = caches.inspect()
    assert [e['job'] for e in plan['eligible']] == ['123']
    receipt = tmp_path / 'removed.json'
    progress = caches.apply(plan, receipt)
    assert progress['removed'] == [str(native / '123')]
    assert not (native / '123').exists()
    assert (native / '124/checkpoint.pt').read_bytes() == b'important'
    assert (native / '125/nv/kernel').exists()
    assert caches.apply(plan, receipt) == progress


def test_changed_cache_or_terminal_state_refuses_deletion(native, tmp_path, monkeypatch):
    plan = caches.inspect()
    (native / '123/nv/kernel').write_bytes(b'new work')
    with pytest.raises(ValueError, match='changed after inventory'):
        caches.apply(plan, tmp_path / 'receipt.json')
    assert (native / '123').exists()
    monkeypatch.setattr(caches, 'terminal_jobs', lambda: set())
    with pytest.raises(ValueError, match='terminal state'):
        caches.apply(plan, tmp_path / 'receipt.json')


def test_other_plan_receipt_and_escaped_paths_are_rejected(native, tmp_path):
    plan = caches.inspect()
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(dict(plan_sha256='different', removed=[])))
    with pytest.raises(ValueError, match='another cleanup plan'):
        caches.apply(plan, receipt)
    with pytest.raises(ValueError, match='exact owned'):
        caches.inventory(tmp_path / '123')


def test_linked_root_is_rejected(native, monkeypatch):
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda p: p == native.parent or original(p))
    with pytest.raises(ValueError, match='Linked legacy'):
        caches.root()


def test_accounting_and_queue_must_identify_completed_owned_jobs(monkeypatch):
    monkeypatch.setenv('SLURM_CLUSTER_NAME', 'mindwell')
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    monkeypatch.setenv('USER', 'fixture')
    def run(args, **kwargs):
        if args[0] == 'sacct':
            return SimpleNamespace(stdout='1|creditpfn-train|COMPLETED|\n2|creditpfn-test|FAILED|\n'
                '3|other-project|COMPLETED|\n4|creditpfn-train|COMPLETED|\n4|creditpfn-train|RUNNING|\n')
        return SimpleNamespace(stdout='999|creditpfn-cache-retire\n123_[0-4]|creditpfn-gpu\n')
    monkeypatch.setattr(caches.subprocess, 'run', run)
    assert caches.terminal_jobs() == {'1'}
    with pytest.raises(RuntimeError, match='consumers still exist'):
        caches.no_active_consumers()
