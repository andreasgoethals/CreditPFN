"""Lossless migration, changed-source refusal, and legacy checkpoint resolution."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils.consolidate_output import consolidate, load_consolidated


def write_manifest(root):
    root.mkdir(parents=True, exist_ok=True)
    common = dict(track="pd", base_checkpoint="base.ckpt", learning_rate=1e-6,
                  use_lora=False, seed=42, l2sp_lambda=0.003)
    pd.DataFrame([dict(common, status="FAIL"), dict(common, status="OK"),
                  dict(common, status="SKIP")]).to_csv(root / "exp1_s00_pd.csv", index=False)


def test_snapshot_preserves_attempts_and_uses_latest_measurement(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    manifests = tmp_path / "output CreditPFN/manifests"
    write_manifest(manifests)
    before = (manifests / "exp1_s00_pd.csv").read_bytes()
    preview = consolidate("exp1")
    assert preview["counts"]["attempts_pd"] == 1
    assert not (tmp_path / "output CreditPFN/consolidated").exists()
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


def test_missing_raw_history_cannot_replace_a_complete_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    root = tmp_path / "output CreditPFN"
    epochs = root / "manifests/epochs/pd"
    epochs.mkdir(parents=True)
    csv = epochs / "exp1_s00_pd_base_lr1e-06_seed42_l2sp0.003.csv"
    csv.write_text("epoch,train_loss\n0,0.5\n")
    consolidate("exp1", apply=True)
    csv.unlink()  # Simulate deliberate removal after preserving the compact tables.
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
