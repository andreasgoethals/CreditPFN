"""Salvage migration for the 22-09-2026 L2-SP bug: retagging pre-fix checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.migrate_l2sp_checkpoints import apply_renames, plan_renames, retag


def test_retag_inserts_l2sp_before_adapter_and_extension() -> None:
    # accumulate (no pass tag), no adapter → tag goes right before .ckpt
    assert (retag("exp1_s00_pd_base_lr1e-06_seed42_qf40_acc1.ckpt", 0.003)
            == "exp1_s00_pd_base_lr1e-06_seed42_qf40_acc1_l2sp0.003.ckpt")
    # full_pass + TabICL icl-head → tag goes BETWEEN _fullpass and _iclhead
    assert (retag("exp1_s00_pd_tabicl_lr1e-05_seed42_qf40_acc1_fullpass_iclhead.ckpt", 0.003)
            == "exp1_s00_pd_tabicl_lr1e-05_seed42_qf40_acc1_fullpass_l2sp0.003_iclhead.ckpt")
    # λ=0 renders as _l2sp0 (matches descriptive_name's :g formatting), before a _lora adapter
    assert (retag("r_pd_base_lr1e-06_seed7_qf40_acc1_lora.ckpt", 0.0)
            == "r_pd_base_lr1e-06_seed7_qf40_acc1_l2sp0_lora.ckpt")


def test_retag_is_idempotent() -> None:
    already = "r_pd_base_lr1e-06_seed7_qf40_acc1_l2sp0.003.ckpt"
    assert retag(already, 0.003) == already
    assert retag(already, 0.0) == already          # never double-tags


def test_plan_renames_uses_provenance_lambda_and_skips_tagged(tmp_path: Path) -> None:
    root = tmp_path / "pd"
    root.mkdir()
    # a tagless survivor whose provenance says it trained at 0.003
    ckpt = root / "exp1_s00_pd_base_lr1e-06_seed42_qf40_acc1.ckpt"
    ckpt.write_bytes(b"")
    ckpt.with_suffix(".ckpt.provenance.json").write_text(
        json.dumps({"provenance": {"hyperparameters": {"l2sp_lambda": 0.003}}}), encoding="utf-8")
    # an already-tagged checkpoint (fixed-code run) — must be left alone
    (root / "exp1_s00_pd_base_lr1e-05_seed42_qf40_acc1_l2sp0.ckpt").write_bytes(b"")
    # a transient epoch-eval snapshot — must be ignored
    (root / "exp1_s00_pd_base_lr1e-06_seed42_qf40_acc1.ckpt.epoch_eval.ckpt").write_bytes(b"")

    plan = plan_renames(root)
    assert len(plan) == 1
    old, new, lam = plan[0]
    assert old == ckpt
    assert new.name == "exp1_s00_pd_base_lr1e-06_seed42_qf40_acc1_l2sp0.003.ckpt"
    assert lam == 0.003


def test_apply_renames_moves_ckpt_and_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "pd"
    root.mkdir()
    ckpt = root / "r_pd_base_lr1e-06_seed7_qf40_acc1.ckpt"
    ckpt.write_bytes(b"weights")
    prov = Path(str(ckpt) + ".provenance.json")
    prov.write_text(json.dumps({"provenance": {"hyperparameters": {"l2sp_lambda": 0.003}}}),
                    encoding="utf-8")

    apply_renames(plan_renames(root), apply=True)

    new = root / "r_pd_base_lr1e-06_seed7_qf40_acc1_l2sp0.003.ckpt"
    assert new.exists() and not ckpt.exists()
    assert Path(str(new) + ".provenance.json").exists() and not prov.exists()
    # missing on re-run: nothing left tagless, so a second pass is a no-op
    assert plan_renames(root) == []
