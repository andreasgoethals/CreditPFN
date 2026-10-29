"""Exact environment recreation and sealed-container build invariants."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.utils import runtime_container as rc


def test_git_packages_preserve_commit_and_installed_source(tmp_path, monkeypatch):
    source = tmp_path / "model.py"
    source.write_text("weight = 1\n")
    direct = dict(url="https://github.com/example/model", vcs_info=dict(vcs="git", commit_id="a" * 40))
    package = NS(metadata={"Name": "TALENT"}, files=[Path("model.py")],
                 locate_file=lambda p: tmp_path / p, read_text=lambda p: json.dumps(direct))
    monkeypatch.setattr(rc.metadata, "distributions", lambda: [package])
    before = rc.source_packages()
    assert before["talent"]["commit"] == "a" * 40
    source.write_text("weight = 2\n")
    assert before != rc.source_packages()
    direct["url"] = "https://secret@example.com/model"
    with pytest.raises(RuntimeError, match="safely reproduce"):
        rc.source_packages()


def test_record_uses_strict_index_versions_and_immutable_git_commits(tmp_path, monkeypatch):
    packages = [dict(name="torch", version="2.12.0", channel="pypi"),
                dict(name="talent", version="0.1.0", channel="pypi")]
    monkeypatch.setattr(rc.subprocess, "check_output", lambda args, **kw:
                        json.dumps(packages) if "--json" in args else "@EXPLICIT\n")
    monkeypatch.setattr(rc, "installed", lambda: {"torch": "2.12.0", "talent": "0.1.0"})
    monkeypatch.setattr(rc, "source_packages", lambda: {
        "talent": dict(url="https://github.com/example/TALENT", commit="a" * 40, files={"model.py": "fixed"})})
    destination = tmp_path / "specification"
    rc.record(destination, "conda")
    pins = (destination / "pip-lock.txt").read_text().splitlines()
    assert "torch===2.12.0" in pins
    assert "talent @ git+https://github.com/example/TALENT@" + "a" * 40 in pins
    with pytest.raises(FileExistsError):
        rc.record(destination, "conda")


def test_compare_refuses_precision_package_drift_or_host_execution(tmp_path, monkeypatch):
    specification = tmp_path / "environment.json"
    specification.write_text(json.dumps(dict(python=rc.sys.version, packages={"torch": "2.12.0"}, sources={})))
    monkeypatch.setattr(rc, "installed", lambda: {"torch": "2.12.0+cu130"})
    with pytest.raises(RuntimeError, match="differs"):
        rc.compare(specification)
    monkeypatch.setattr(rc, "installed", lambda: {"torch": "2.12.0"})
    monkeypatch.setattr(rc, "source_packages", lambda: {})
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    with pytest.raises(RuntimeError, match="escaped"):
        rc.compare(specification)
    monkeypatch.setenv("APPTAINER_CONTAINER", "fixture.sif")
    rc.compare(specification)


def test_template_patch_pins_installer_and_uses_the_exact_prefix():
    template = '\n'.join([
        'mamba $_EC create --name $CW_ENV_NAME $_FF',
        'conda $_EC create --name $CW_ENV_NAME $_FF',
        'conda activate $CW_ENV_NAME',
        'CW_CONDA_VERSION=$(curl latest)',
        'curl installer --output Miniforge_inst.sh',
        'follow_log _inst_miniforge.log',
    ])
    changed = rc.patch_template(template)
    assert '--prefix "$env_root"' in changed
    assert rc.MINIFORGE_SHA256 in changed
    assert 'wait "$inst_pid"' in changed
    with pytest.raises(ValueError, match="Site wrapper changed"):
        rc.patch_template(template.replace('conda activate $CW_ENV_NAME', 'different activation'))


def test_sealing_rejects_unpacked_dependencies(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").write_text("wrapper")
    (tmp_path / "environment.sqfs").write_bytes(b"image")
    specification = tmp_path / "environment.json"
    specification.write_text("{}")
    (tmp_path / "site-packages").mkdir()
    with pytest.raises(RuntimeError, match="Unpacked"):
        rc.seal(tmp_path, specification)


@pytest.mark.parametrize("changed", [None, "image", "wrapper", "specification"])
def test_sealed_runtime_is_immutable_and_repeat_sealing_never_rewrites_it(tmp_path, monkeypatch, changed):
    from src.utils import atomic
    root = tmp_path / "runtime"
    (root / "bin").mkdir(parents=True)
    wrapper = root / "bin/python"
    wrapper.write_text("wrapper", encoding="utf-8")
    image = root / "environment.sqfs"
    image.write_bytes(b"original image")
    specification = tmp_path / "environment.json"
    specification.write_text("{}", encoding="utf-8")
    rc.seal(root, specification)
    marker = root / "runtime.json"
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    def unexpected_write(*args, **kwargs):
        raise AssertionError("Existing runtime manifests must never be rewritten")
    monkeypatch.setattr(atomic, "write_json", unexpected_write)
    if changed is None:
        rc.seal(root, specification)
    else:
        {"image": image, "wrapper": wrapper, "specification": specification}[changed].write_bytes(b"different")
        with pytest.raises(RuntimeError, match="already sealed"):
            rc.seal(root, specification)
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before
