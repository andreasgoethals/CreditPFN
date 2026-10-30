import json
from pathlib import Path

import pytest

from src.utils import job_cache


@pytest.fixture
def cache_environment(tmp_path, monkeypatch):
    for key,value in dict(SLURM_JOB_ID='123',SLURM_RESTART_COUNT='0',SLURM_CLUSTER_NAME='mindwell',
                         VSC_SCRATCH_NODE=str(tmp_path/'node'),VSC_SCRATCH_GPFS1=str(tmp_path/'gpfs')).items():
        monkeypatch.setenv(key,value)
    monkeypatch.delenv('CREDITPFN_CACHE_ROOT',raising=False)
    return tmp_path


def test_cache_uses_node_local_and_cleans_only_its_owner(cache_environment, monkeypatch):
    cache = job_cache.create()
    assert cache.is_relative_to(cache_environment/'node')
    (cache/'nv').mkdir(); (cache/'nv/kernel').write_bytes(b'regenerable')
    assert job_cache.create() == cache
    monkeypatch.setenv('SLURM_JOB_ID','124')
    peer = job_cache.create()
    with pytest.raises(ValueError,match='ownership'): job_cache.clean(cache)
    monkeypatch.setenv('SLURM_JOB_ID','123')
    job_cache.clean(cache)
    assert not cache.exists() and peer.is_dir()


def test_restarts_get_separate_caches_and_unknown_directories_are_not_claimed(cache_environment,monkeypatch):
    first=job_cache.create()
    monkeypatch.setenv('SLURM_RESTART_COUNT','1')
    second=job_cache.create()
    assert first != second and first.is_dir()
    unknown=cache_environment/'node/unknown';unknown.mkdir();(unknown/'keep').write_text('important')
    monkeypatch.setenv('CREDITPFN_CACHE_ROOT',str(unknown))
    with pytest.raises(ValueError,match='populated'):job_cache.create()
    assert (unknown/'keep').read_text()=='important'


def test_cache_fails_closed_outside_scratch(cache_environment,monkeypatch):
    monkeypatch.setenv('CREDITPFN_CACHE_ROOT',str(cache_environment/'project'))
    with pytest.raises(ValueError,match='approved scratch'):job_cache.create()
    with pytest.raises(ValueError,match='approved scratch'):job_cache.clean(cache_environment/'node')
    assert not (cache_environment/'project').exists()


def test_native_fallback_is_owned_when_node_local_is_unavailable(cache_environment,monkeypatch):
    original=Path.mkdir
    def mkdir(path,*args,**kwargs):
        if path.is_relative_to(cache_environment/'node'):raise PermissionError('fixture node unavailable')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'mkdir',mkdir)
    cache=job_cache.create()
    assert cache.is_relative_to(cache_environment/'gpfs')
    assert json.loads((cache/job_cache.MARKER).read_text())['job']=='123'
