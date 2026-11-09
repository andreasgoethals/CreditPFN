"""Compatibility gates and independent scientific-evidence checks, without GPUs."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.utils import runtime_container as rc
from src.utils import runtime_validation as rv


def test_runtime_acceptance_is_global_and_preserves_previous_source_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "outputs_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(rv, "resolve_staging_path", lambda _: tmp_path / "project")
    monkeypatch.setattr(rv, "code_identity", lambda **_: "source_a")
    first = rv.root("runtime1"), rv.evidence("runtime1")
    monkeypatch.setenv("CREDITPFN_EXPERIMENT", "reference")
    assert (rv.root("runtime1"), rv.evidence("runtime1")) == first
    assert first == (tmp_path / "data/experiment0/manifests/runtime_acceptance/runtime1/source_a",
                     tmp_path / "project/experiment0/results/runtime_acceptance/runtime1/source_a")
    first[0].mkdir(parents=True)
    (first[0] / "accepted.json").write_text("earlier evidence")
    monkeypatch.setattr(rv, "code_identity", lambda **_: "source_b")
    assert rv.root("runtime1") != first[0] and rv.evidence("runtime1") != first[1]
    assert (first[0] / "accepted.json").read_text() == "earlier evidence"


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


def test_runtime_cache_checks_content_and_keeps_existing_corrupt_cache(tmp_path, monkeypatch):
    import sys

    from src.utils import runtime_cache as cache
    monkeypatch.setitem(sys.modules,"fcntl",NS(LOCK_SH=1,LOCK_EX=2,LOCK_UN=8,flock=lambda *args:None))
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


def test_runtime_cache_does_not_preserve_source_timestamps(tmp_path, monkeypatch):
    """Scratch purge follows access time: staged files must not inherit old timestamps."""
    import os
    import sys
    import time

    from src.utils import runtime_cache as cache
    monkeypatch.setitem(sys.modules,"fcntl",NS(LOCK_SH=1,LOCK_EX=2,LOCK_UN=8,flock=lambda *args:None))
    runtime=tmp_path/"project";runtime.mkdir()
    (runtime/"image.sqfs").write_bytes(b"fixture image")
    (runtime/"python").write_bytes(b"wrapper")
    try:
        (runtime/"alias").symlink_to("python")
    except OSError:
        pytest.skip("Creating symlinks is unavailable for this local Windows account")
    manifest=dict(images=["image.sqfs"],files={name:rc.sha256(runtime/name) for name in ("image.sqfs","python")},symlinks={"alias":"python"})
    (runtime/"runtime.json").write_text(json.dumps(manifest))
    old=time.time()-400*86400
    for name in ("image.sqfs","python","runtime.json"):
        os.utime(runtime/name,(old,old))
    target=tmp_path/"scratch"/"runtime"
    cache.stage(runtime,target)
    for name in ("image.sqfs","python","runtime.json"):
        assert (target/name).stat().st_mtime>old+86400, name
    receipt=json.loads((target/"CACHE.json").read_text())
    assert receipt["files"]["image.sqfs"]==[len(b"fixture image"),manifest["files"]["image.sqfs"]]
    # Reuse ignores timestamps, so touching (or an access-time update) is harmless.
    os.utime(target/"image.sqfs",(old,old))
    assert cache.stage(runtime,target)==target
    # Same-size content change is still detected through the manifest hash.
    (target/"python").write_bytes(b"wrappeR")
    with pytest.raises(RuntimeError,match="cache changed"):cache.stage(runtime,target)
    (target/"python").write_bytes(b"wrapper")
    # Earlier [size, mtime_ns] receipts remain valid; content is checked against the manifest.
    receipt["files"]={name:[(target/name).stat().st_size,123] for name in manifest["files"]}
    (target/"CACHE.json").write_text(json.dumps(receipt))
    assert cache.stage(runtime,target)==target
    (target/"image.sqfs").unlink()
    with pytest.raises(RuntimeError,match="cache changed"):cache.stage(runtime,target)
