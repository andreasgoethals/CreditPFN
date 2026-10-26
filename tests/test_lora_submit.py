"""One command queues the gate, with no duplicate work or cross-cluster dependency."""
import json
from types import SimpleNamespace

import pytest

from src.utils import lora_submit as submitter


@pytest.fixture
def cluster(monkeypatch, tmp_path):
    monkeypatch.setenv("VSC_DATA", str(tmp_path))
    monkeypatch.setenv("SBATCH_EXPORT", "NONE")
    monkeypatch.setattr(submitter, "folder", lambda identifier: tmp_path / identifier)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"{100 + len(calls)};mindwell\n")
    monkeypatch.setattr(submitter.subprocess, "run", run)
    return tmp_path, calls


def test_submits_whole_gate_same_controller_and_retains_job_ids(cluster):
    root, calls = cluster
    result = submitter.submit("gate2")
    assert [j["stage"] for j in result["jobs"]] == ["prepare", "smoke", "capacity", "inspect"]
    assert len(calls) == 4
    for index, (command, options) in enumerate(calls):
        assert "--clusters=mindwell" in command and "SBATCH_EXPORT" not in options["env"]
        assert options["cwd"] == submitter.REPO_ROOT
        if index:
            assert f"--dependency=afterok:{100 + index}" in command
            assert "--kill-on-invalid-dep=yes" in command
        else:
            assert not any(arg.startswith("--dependency") for arg in command)
        if index in (0, 3):
            assert "--partition=batch_graniterapids" in command
        else:
            assert "--array=0-3%2" in command
        assert not any("run_experiment8" in arg for arg in command)
    assert json.loads((root / "gate2/submission.json").read_text())["status"] == "submitted"
    assert json.loads((root / "gate2/.scope.json").read_text())["kind"] == "creditpfn_lora_scope"
    with pytest.raises(FileExistsError):
        submitter.submit("gate2")
    assert len(calls) == 4


@pytest.mark.parametrize("response", ["555;wice", "Submitted batch job 555", ""])
def test_unknown_submission_outcome_is_recorded_and_not_retried(cluster, monkeypatch, response):
    root, _ = cluster
    calls = []
    def run(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(stdout="101;mindwell" if len(calls) == 1 else response)
    monkeypatch.setattr(submitter.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="inspect queue"):
        submitter.submit("gate3")
    record = json.loads((root / "gate3/submission.json").read_text())
    assert record["status"] == "incomplete_submission"
    assert len(calls) == 2 and len(record["jobs"]) == 1
    assert record["attempting"]["stage"] == "smoke"


def test_submitter_requires_cluster_and_valid_id_before_creating_scope(cluster, monkeypatch):
    root, calls = cluster
    with pytest.raises(ValueError):
        submitter.submit("../other")
    monkeypatch.delenv("VSC_DATA")
    with pytest.raises(RuntimeError, match="VSC"):
        submitter.submit("gate4")
    assert not calls and not (root / "gate4").exists()


def test_sbatch_failure_preserves_prior_submissions(cluster, monkeypatch):
    root, _ = cluster
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 3:
            raise submitter.subprocess.CalledProcessError(1, command, stderr="quota")
        return SimpleNamespace(stdout=str(100 + len(calls)))
    monkeypatch.setattr(submitter.subprocess, "run", run)
    with pytest.raises(submitter.subprocess.CalledProcessError):
        submitter.submit("gate5")
    record = json.loads((root / "gate5/submission.json").read_text())
    assert [j["id"] for j in record["jobs"]] == ["101", "102"]
    assert record["attempting"]["stage"] == "capacity" and len(calls) == 3
