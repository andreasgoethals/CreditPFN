"""Archive safety, immutable inputs, reuse identities and descriptive summaries."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf


def historical(tmp_path):
    root = tmp_path / "live" / "output"
    manifest = root / "manifests"
    epochs = manifest / "epochs" / "pd"
    epochs.mkdir(parents=True)
    pd.DataFrame([dict(track="pd", base_checkpoint="base.ckpt", learning_rate=1e-6,
        use_lora=False, seed=42, status="OK")]).to_csv(manifest / "exp1_s00_pd.csv", index=False)
    pd.DataFrame([dict(epoch=0, train_loss=1.)]).to_csv(epochs / "exp1_s00_pd_trial.csv", index=False)
    (root / "logs").mkdir()
    (root / "logs" / "old.log").write_text("old log\n" * 20)
    weights = tmp_path / "project" / "checkpoints" / "trained"
    weights.mkdir(parents=True)
    checkpoint = weights / "exp1_s00_pd_base_lr1e-6_seed42.ckpt"
    checkpoint.write_bytes(b"historical weight bytes")
    unrelated = weights / "other_s00_pd_base_lr1e-6_seed42.ckpt"
    unrelated.write_bytes(b"keep")
    kwargs = dict(manifest_root=manifest, result_root=tmp_path / "results", roots=[weights],
                  destination=tmp_path / "archives", snapshot_root=tmp_path / "consolidated")
    return root, checkpoint, unrelated, kwargs


def test_evidence_archive_prune_and_separate_weight_retirement(tmp_path):
    from src.utils.archive_experiment import create, prune_evidence, retire
    from src.utils.archive_output import verify
    root, checkpoint, unrelated, kwargs = historical(tmp_path)
    preview = create("exp1", **kwargs)
    assert preview["indexed_checkpoints"] == 1 and checkpoint.exists()
    with pytest.raises(ValueError, match="quiescent"):
        create("exp1", write=True, **kwargs)
    result = create("exp1", write=True, quiescent=True, **kwargs)
    archive = Path(result["archive"])
    inventory = verify(archive)
    assert not inventory["weights_in_archive"]
    assert not any(f["path"].endswith(".ckpt") for f in inventory["files"])
    assert not prune_evidence(archive, root=root)["deleted"]
    assert prune_evidence(archive, root=root, apply=True, quiescent=True)["files"] == 2
    assert (root / "manifests" / "exp1_s00_pd.csv").exists()
    assert checkpoint.exists() and unrelated.exists()
    assert not retire(archive, roots=kwargs["roots"])["deleted"]
    assert retire(archive, apply=True, quiescent=True, roots=kwargs["roots"])["deleted"]
    assert not checkpoint.exists() and unrelated.exists()


def test_changed_checkpoint_and_log_block_all_deletion(tmp_path):
    from src.utils.archive_experiment import create, prune_evidence, retire
    root, checkpoint, _, kwargs = historical(tmp_path)
    archive = Path(create("exp1", write=True, quiescent=True, **kwargs)["archive"])
    checkpoint.write_bytes(b"changed weights")
    with pytest.raises(RuntimeError, match="changed"):
        retire(archive, apply=True, quiescent=True, roots=kwargs["roots"])
    assert checkpoint.exists()
    (root / "logs" / "old.log").write_text("changed log")
    with pytest.raises(RuntimeError, match="changed"):
        prune_evidence(archive, root=root, apply=True, quiescent=True)
    assert (root / "manifests" / "epochs" / "pd" / "exp1_s00_pd_trial.csv").exists()


def test_scratch_stage_is_immutable_and_pointer_cannot_escape(tmp_path):
    from src.data.preprocessing import DATASET_METADATA
    from src.utils.stage_inputs import stage, resolve
    cfg = OmegaConf.load("config/train.yaml")
    names = [f"data/processed/{m['track']}/{did}.sanitized.csv" for did, m in DATASET_METADATA.items()]
    names += list(cfg.tunable.classifier_base_paths) + list(cfg.tunable.regressor_base_paths)
    source = tmp_path / "source"
    for name in names:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"test input")
    result = stage(tmp_path / "scratch", source=source, write=True)
    pointer = Path(result["pointer"])
    root = resolve(pointer)
    assert stage(tmp_path / "scratch", source=source, write=True)["root"] == result["root"]
    (root / names[0]).write_bytes(b"changed input")
    with pytest.raises(RuntimeError, match="modified"):
        stage(tmp_path / "scratch", source=source, write=True)
    data = json.loads(pointer.read_text())
    data["root"] = str(tmp_path)
    pointer.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        resolve(pointer)


def test_pool_reuses_lanes_and_cannot_resize_around_active_work():
    from src.utils.submit_bounded import reserve
    pool, chosen, deps = reserve({}, slots=4, limit=2, active=set())
    assert not deps
    for i in chosen:
        pool["lanes"][i] = "1"
    pool, chosen, deps = reserve(pool, slots=4, limit=2, active={"1"})
    assert not deps
    for i in chosen:
        pool["lanes"][i] = "2"
    pool, chosen, deps = reserve(pool, slots=4, limit=2, active={"1", "2"})
    assert deps == ["1"]
    with pytest.raises(ValueError, match="resize"):
        reserve(pool, slots=6, limit=2, active={"1", "2"})
    _, _, deps = reserve(pool, slots=4, limit=2, active={"2"})
    assert deps == ["2"]


def test_uncertain_submission_blocks_the_next_submit(tmp_path, monkeypatch):
    import sys
    import src.utils.submit_bounded as module
    monkeypatch.setitem(sys.modules, "fcntl", NS(LOCK_EX=1, flock=lambda *a: None))
    monkeypatch.setenv("USER", "test-user")
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        return NS(stdout="" if command[0] == "squeue" else "ambiguous reply", stderr="", returncode=0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    path = tmp_path / "pool.json"
    with pytest.raises(RuntimeError, match="ambiguous"):
        module.submit(["sbatch", "--array=0-3%4", "job.slurm"], slots=4, limit=4,
                      cluster="mindwell", pool_path=path)
    with pytest.raises(RuntimeError, match="Uncertain"):
        module.submit(["sbatch", "--array=0-3%4", "job.slurm"], slots=4, limit=4,
                      cluster="mindwell", pool_path=path)
    assert sum(c[0] == "sbatch" for c in commands) == 1


def test_staging_write_probes_distinct_directories_and_refuses_data_fallback(tmp_path, monkeypatch):
    from src.utils.paths import resolve_writable_staging_path
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(tmp_path / "project"))
    monkeypatch.setenv("CREDITPFN_REQUIRE_STAGING", "1")
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path / "live"))
    pd_root = resolve_writable_staging_path("checkpoints/trained/pd")
    assert pd_root.is_dir()
    blocked = pd_root.parent / "lgd"
    blocked.write_bytes(b"file blocks directory creation")
    with pytest.raises(PermissionError, match="refusing"):
        resolve_writable_staging_path("checkpoints/trained/lgd")
    assert not (tmp_path / "live/checkpoints").exists()


def test_atomic_plan_cannot_replace_existing_plan(tmp_path):
    from src.utils.atomic import write_json
    p = tmp_path / "plan.json"
    write_json(p, {"original": True}, exclusive=True)
    with pytest.raises(FileExistsError):
        write_json(p, {"original": False}, exclusive=True)
    assert json.loads(p.read_text()) == {"original": True}


def test_null_control_detects_changed_tensor_and_monitor(tmp_path):
    from src.utils.audit_experiment import compare_states, null_monitor_parity
    a, b = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    torch.save({"state_dict": {"weight": torch.tensor([1.])}}, a)
    torch.save({"state_dict": {"weight": torch.tensor([1.])}}, b)
    assert compare_states(a, b)["equal"]
    torch.save({"state_dict": {"weight": torch.tensor([2.])}}, b)
    assert not compare_states(a, b)["equal"]
    f = pd.DataFrame({"successful_updates": [0, 2], "metric__test__table": [.7, .7]})
    assert null_monitor_parity(f)
    f.loc[1, "metric__test__table"] = .6
    assert not null_monitor_parity(f)


def test_campaign_audit_finds_the_actual_trial_filename(tmp_path, monkeypatch):
    import src.utils.audit_experiment as module
    import scripts.train_pipeline as pipeline
    from src.train.loop import descriptive_name
    cfg = pipeline._load_cfg(config_path="config/experiment0_pd.yaml")
    cfg.checkpoint.trained_dir = str(tmp_path / "weights")
    trial = ("base.ckpt", 0., False, .4, 1, "one_sample", 0, 0.)
    monkeypatch.setattr(pipeline, "_load_cfg", lambda **kw: cfg)
    monkeypatch.setattr(pipeline, "_resolve_grid", lambda *a, **kw: [trial])
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"trials": {f"{cfg.run_name}_s00/0": "expected"}}))
    monkeypatch.setattr(module, "plan_path", lambda *a: plan)
    monkeypatch.setattr(module, "manifests_dir", lambda: tmp_path / "manifests")
    filename = descriptive_name(run_name=f"{cfg.run_name}_s00", track="pd", base_path=trial[0],
        learning_rate=0., seed=42, query_fraction=.4, accumulate_grad_batches=1,
        epoch_pass_mode="one_sample", l2sp_lambda=0.)
    weight = tmp_path / "weights" / "pd" / filename
    weight.parent.mkdir(parents=True)
    torch.save({"state_dict": {"weight": torch.ones(1)}}, weight)
    monkeypatch.setattr(module, "resolve_base_checkpoint", lambda *a: weight)
    Path(str(weight) + ".provenance.json").write_text(json.dumps({
        "trial_identity": {"sha256": "expected"}, "successful_updates": 2}))
    curve = tmp_path / "manifests" / "epochs" / "pd" / filename.replace(".ckpt", ".trajectory.csv")
    curve.parent.mkdir(parents=True)
    pd.DataFrame({"successful_updates": [0, 2], "metric__test__table": [.7, .7]}).to_csv(curve, index=False)
    result = module.audit(Path("unused"), null=True)
    assert result["passed"] and result["completed"] == 1


def test_lgd_trajectory_effects_are_scale_invariant_and_require_baseline():
    from src.visualize.trajectories import trajectory_effects
    name = "cpt_main_v3_s00_lgd_tabpfn-v3-regressor-v3_default_lr1e-6_seed42_l2sp0"
    f = pd.DataFrame({"trial_name": [name, name], "successful_updates": [0, 250],
                      "processed_rows": [0, 1000], "metric__test__small": [1., .8],
                      "metric__test__large": [100., 80.]})
    effects = trajectory_effects(f, "lgd")
    assert effects.loc[effects.updates.eq(250), "effect"].tolist() == pytest.approx([.2, .2])
    with pytest.raises(ValueError, match="update-zero"):
        trajectory_effects(f.iloc[1:], "lgd")


def test_evaluation_cache_rejects_incomplete_folds(tmp_path):
    from src.eval.cache import save, load
    from src.eval.benchmark import EvalRow
    row = EvalRow(track="pd", task_type="classification", model_name="control", model_source="baseline",
                  model_path=None, test_dataset_id="sample", fold_idx=0,
                  n_train_rows=10, n_val_rows=2, n_test_rows=3, roc_auc=.7)
    save(tmp_path / "results", "abc", [row], n_folds=2)
    assert load(tmp_path / "results", "abc", n_folds=2) is None
    from dataclasses import replace
    save(tmp_path / "results", "abc", [row, replace(row, fold_idx=1)], n_folds=2)
    assert len(load(tmp_path / "results", "abc", n_folds=2)) == 2
