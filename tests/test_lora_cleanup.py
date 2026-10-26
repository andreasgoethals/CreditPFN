"""Scope, tier and active-writer safety for the new LoRA workflow cleaner."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils import clean_run


def make_workflow(identifier):
    roots = clean_run.lora_workflow_roots(identifier)
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        (root / "result.json").write_text("output", encoding="utf-8")
    (roots[0] / "workflow.json").write_text(json.dumps({"id": identifier, "kind": "creditpfn_lora"}), encoding="utf-8")
    return roots


def test_preview_then_clean_both_tiers_preserves_siblings_and_bases(isolated_output, monkeypatch, capsys):
    from src.utils.paths import checkpoints_dir
    monkeypatch.setattr(clean_run, "check_lora_writers", lambda i: None)
    targets = make_workflow("failed1")
    siblings = make_workflow("failed10")
    base = checkpoints_dir("original.ckpt")
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("base")
    clean_run.main(["--lora-workflow", "failed1"])
    assert all((root / "result.json").exists() for root in targets)
    assert "Nothing deleted" in capsys.readouterr().out
    clean_run.main(["--lora-workflow", "failed1", "--clean"])
    assert all(not (root / "result.json").exists() for root in targets)
    assert all((root / "result.json").exists() for root in siblings)
    assert base.read_text() == "base"


def test_unknown_workflow_is_refused_without_any_delete(isolated_output, monkeypatch):
    monkeypatch.setattr(clean_run, "check_lora_writers", lambda i: None)
    roots = make_workflow("unknown1")
    (roots[0] / "workflow.json").unlink()
    with pytest.raises(ValueError, match="Unknown"):
        clean_run.main(["--lora-workflow", "unknown1", "--clean"])
    assert all((root / "result.json").exists() for root in roots)


def test_setup_failure_before_python_can_be_cleaned_from_shell_ownership(isolated_output, monkeypatch):
    monkeypatch.setattr(clean_run, "check_lora_writers", lambda i: None)
    root = clean_run.lora_workflow_roots("setupfailed")[0]
    (root / "logs").mkdir(parents=True)
    (root / "logs/lora_prepare_1_r0.log").write_text("conda setup failed")
    (root / ".scope.json").write_text(json.dumps({"kind": "creditpfn_lora_scope", "id": "setupfailed", "schema": 1}))
    clean_run.main(["--lora-workflow", "setupfailed", "--clean"])
    assert not (root / "logs/lora_prepare_1_r0.log").exists()


def test_symlink_on_later_tier_prevents_all_deletion(isolated_output, monkeypatch):
    monkeypatch.setattr(clean_run, "check_lora_writers", lambda i: None)
    roots = make_workflow("links1")
    linked = roots[-1] / "result.json"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == linked or original(p))
    with pytest.raises(ValueError, match="linked"):
        clean_run.main(["--lora-workflow", "links1", "--clean"])
    assert all((root / "result.json").exists() for root in roots)


@pytest.mark.parametrize("details,blocked", [
    ("SubmitLine=sbatch scripts/slurm/lora_gpu.slurm smoke selected1", True),
    ("SubmitLine=sbatch scripts/slurm/lora_gpu.slurm diagnose selected1", True),
    ("SubmitLine=sbatch scripts/slurm/lora_prepare.slurm selected1 inspect", True),
    ("SubmitLine=sbatch scripts/slurm/lora_gpu.slurm smoke other2", False),
    ("SubmitLine=sbatch scripts/slurm/lora_gpu.slurm diagnose other2", False),
    ("SubmitLine=sbatch --export=ALL,CREDITPFN_CONFIG=config/experiment8/pd.yaml train.slurm", True),
    ("SubmitLine=sbatch --export=ALL,CREDITPFN_CONFIG=config/experiment5/pd.yaml train.slurm", False),
    ("SubmitLine=unknown", True),
])
def test_scheduler_guard_refuses_selected_consumers_and_unknown_ownership(monkeypatch, details, blocked):
    import subprocess
    monkeypatch.setenv("VSC_DATA", "/data/example")
    monkeypatch.setenv("USER", "test")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    def run(command, **kwargs):
        return SimpleNamespace(stdout="99|creditpfn-lora-gpu\n" if command[0] == "squeue" else details)
    monkeypatch.setattr(subprocess, "run", run)
    if blocked:
        with pytest.raises(RuntimeError, match="active"):
            clean_run.check_lora_writers("selected1")
    else:
        clean_run.check_lora_writers("selected1")


def test_cleanup_active_writer_does_not_delete_any_tier(isolated_output, monkeypatch):
    roots = make_workflow("active1")
    def blocked(identifier):
        raise RuntimeError("still active")
    monkeypatch.setattr(clean_run, "check_lora_writers", blocked)
    with pytest.raises(RuntimeError, match="active"):
        clean_run.main(["--lora-workflow", "active1", "--clean"])
    assert all((root / "result.json").exists() for root in roots)


@pytest.mark.parametrize("extra", [["--experiment", "experiment1"], ["--all-experiments"], ["--processed"],
                                   ["--stages", "train"], ["--scope", "training"]])
def test_workflow_cleanup_cannot_expand_to_another_scope(extra):
    with pytest.raises(SystemExit):
        clean_run.main(["--lora-workflow", "safe1", *extra])
