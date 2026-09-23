"""Lossless migration, changed-source refusal, and legacy checkpoint resolution."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils.archive_output import create, prune, verify
from src.utils.consolidate_output import consolidate, load_consolidated


def write_manifest(root):
    root.mkdir(parents=True, exist_ok=True)
    common = dict(track="pd", base_checkpoint="base.ckpt", learning_rate=1e-6,
                  use_lora=False, seed=42, l2sp_lambda=0.003)
    pd.DataFrame([dict(common, status="FAIL"), dict(common, status="OK"),
                  dict(common, status="SKIP")]).to_csv(root / "exp1_s00_pd.csv", index=False)


def test_snapshot_preserves_attempts_and_uses_latest_measurement(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    manifests = tmp_path / "output/manifests"
    write_manifest(manifests)
    before = (manifests / "exp1_s00_pd.csv").read_bytes()
    preview = consolidate("exp1")
    assert preview["counts"]["attempts_pd"] == 1
    assert not (tmp_path / "output/consolidated").exists()
    result = consolidate("exp1", apply=True)
    snap = Path(result["snapshot"])
    assert result["rows"]["attempts_pd"] == 3
    assert result["rows"]["trials_pd"] == 1
    assert pd.read_csv(snap / "trials_pd.csv.gz").iloc[0]["status"] == "OK"
    assert (manifests / "exp1_s00_pd.csv").read_bytes() == before
    assert len(load_consolidated("exp1", "trials_pd")) == 1
    assert load_consolidated("exp1", "trials_pd", manifest_root=tmp_path / "other") is None
    with (manifests / "exp1_s00_pd.csv").open("a") as fh:
        fh.write("\n")
    assert load_consolidated("exp1", "trials_pd") is None


def test_archive_verifies_bytes_and_requires_explicit_cleanup(tmp_path):
    root = tmp_path / "live"
    log = root / "logs/job.log"
    log.parent.mkdir(parents=True)
    log.write_text("repeated warning\n" * 1000)
    preview = create(root=root, destination=tmp_path / "archive")
    assert preview["files"] == 1 and not (tmp_path / "archive").exists()
    result = create(root=root, destination=tmp_path / "archive", apply=True)
    archive = Path(result["archive"])
    assert len(verify(archive)["files"]) == 1 and log.exists()
    with pytest.raises(ValueError, match="quiescent"):
        prune(archive, root=root, quiescent=False)
    prune(archive, root=root, quiescent=True)
    assert not log.exists() and archive.exists()


def test_archive_refuses_changed_source_before_deleting_any_file(tmp_path):
    root = tmp_path / "live"
    (root / "logs").mkdir(parents=True)
    a, b = root / "logs/a.log", root / "logs/b.log"
    a.write_text("original"); b.write_text("original")
    archive = Path(create(root=root, destination=tmp_path / "archives", apply=True)["archive"])
    b.write_text("modified")
    with pytest.raises(RuntimeError, match="changed"):
        prune(archive, root=root, quiescent=True)
    assert a.exists() and b.exists()


def test_epoch_cleanup_requires_consolidated_history(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    root = tmp_path / "output"
    epochs = root / "manifests/epochs/pd"
    epochs.mkdir(parents=True)
    csv = epochs / "exp1_s00_pd_base_lr1e-06_seed42_l2sp0.003.csv"
    csv.write_text("epoch,train_loss\n0,0.5\n")
    archive = Path(create(root=root, destination=tmp_path / "archives", apply=True, include_epochs=True)["archive"])
    with pytest.raises(RuntimeError, match="Consolidate"):
        prune(archive, root=root, quiescent=True)
    consolidate("exp1", apply=True)
    prune(archive, root=root, quiescent=True)
    assert not csv.exists()
    assert len(load_consolidated("exp1", "training_pd")) == 1
    pointer = (root / "consolidated/exp1/LATEST.json").read_bytes()
    with pytest.raises(RuntimeError, match="Restore the verified raw"):
        consolidate("exp1", apply=True)
    assert (root / "consolidated/exp1/LATEST.json").read_bytes() == pointer


def test_resolve_previously_retagged_checkpoint(tmp_path, monkeypatch):
    from src.utils.checkpoint_inventory import resolve_checkpoint
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    root = tmp_path / "checkpoints/trained/pd"
    root.mkdir(parents=True)
    old = "exp1_s00_pd_base_lr1e-06_seed42.ckpt"
    new = root / old.replace(".ckpt", "_l2sp0.003.ckpt")
    new.write_bytes(b"weights")
    Path(str(new) + ".provenance.json").write_text(json.dumps({"hyperparameters": {"l2sp_lambda": .003}}))
    path, prov = resolve_checkpoint("/old/machine/" + old, "pd")
    assert path == new and prov["hyperparameters"]["l2sp_lambda"] == .003


def test_checkpoint_failed_save_preserves_previous_completed_pair(tmp_path, monkeypatch):
    import torch
    from src.train.checkpoint_io import atomic_save
    path = tmp_path / "model.ckpt"
    atomic_save({"x": torch.ones(1)}, path, {"complete": True})
    before = path.read_bytes()
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_save({}, path, {"complete": False})
    assert path.read_bytes() == before
    assert json.loads(Path(str(path) + ".provenance.json").read_text())["complete"]


def test_l2sp_migration_refuses_missing_provenance_and_collision(tmp_path):
    from src.utils.migrate_l2sp_checkpoints import plan_renames, apply_renames
    old = tmp_path / "exp1_pd_base_lr1e-06_seed42.ckpt"
    old.write_bytes(b"old")
    with pytest.raises(ValueError, match="refusing to guess"):
        plan_renames(tmp_path)
    Path(str(old) + ".provenance.json").write_text(json.dumps({"hyperparameters": {"l2sp_lambda": .003}}))
    plan = plan_renames(tmp_path)
    plan[0][1].write_bytes(b"new")
    with pytest.raises(FileExistsError):
        apply_renames(plan, apply=True)
    assert old.read_bytes() == b"old" and plan[0][1].read_bytes() == b"new"
