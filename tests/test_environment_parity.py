"""Local and cluster environments must run the same scientific stack.

`config/runtime/env20261029f.pip-lock.txt` is the pip lock of the accepted VSC container.
The scientific packages below must match it exactly locally, so a local test exercises the
same upstream training and inference code that the cluster runs. (The v2.6 regression
loss defect went unnoticed because the local tabpfn was 2.2.1 while VSC ran 8.0.8.)
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

import pytest

LOCK = Path(__file__).resolve().parents[1] / "config/runtime/env20261029f.pip-lock.txt"
SCIENTIFIC = ("torch", "tabpfn", "tabicl", "numpy", "scipy", "scikit-learn", "pandas",
              "omegaconf", "xgboost", "catboost", "lightgbm", "optuna", "peft", "huggingface-hub")
TABPFN_COMMIT = "27923f45287a680622012acd51a7b29f8b94415a"


def locked():
    pins = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if "===" in line:
            name, value = line.split("===")
            pins[name.strip().lower()] = value.strip()
        elif " @ " in line:
            name, value = line.split(" @ ")
            pins[name.strip().lower()] = value.strip()
    return pins


def test_lock_pins_the_tabpfn_commit_the_cluster_runs():
    assert locked()["tabpfn"].endswith("@" + TABPFN_COMMIT)


@pytest.mark.parametrize("package", SCIENTIFIC)
def test_local_scientific_versions_match_the_cluster_lock(package):
    pins = locked()
    try:
        installed = version(package)
    except PackageNotFoundError:
        pytest.fail(f"{package} is not installed; install config/runtime pins into the local venv")
    expected = pins[package.lower()]
    if package == "tabpfn":
        assert installed == "8.0.8", installed
        return
    # Local CPU wheels carry a local tag (e.g. 2.12.0+cpu); the release must agree.
    assert installed.split("+")[0] == expected, (package, installed, expected)


def test_local_python_matches_the_cluster_minor_version():
    assert sys.version_info[:2] == (3, 13)
