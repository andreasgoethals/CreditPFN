"""Immutable inputs, reuse identities and descriptive summaries."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf


def test_corpus_metadata_reuses_reads_and_invalidates_file_or_registry_changes(tmp_path, monkeypatch):
    import src.train.corpus as corpus
    import src.data.preprocessing as registry
    path = tmp_path / 'synthetic.sanitized.csv'
    path.write_text('x,z,y\n1,5,0\n2,6,1\nlate-category,7,0\n', encoding='utf-8')
    meta = {'synthetic': {'track': 'pd', 'target_column': 'y', 'categorical_columns': []}}
    monkeypatch.setattr(registry, 'DATASET_METADATA', meta)
    monkeypatch.setattr(corpus, 'processed_dir', lambda *parts: path)
    original = pd.read_csv
    reads = []

    def read(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(corpus.pd, 'read_csv', read)
    first = corpus.build_dataset_pool('pd')
    assert first == corpus.build_dataset_pool('pd')
    assert first[0].n_rows == 3 and 'x' in first[0].categorical_columns
    assert len(reads) == 1
    with path.open('a', encoding='utf-8') as stream:
        stream.write('another-category,8,1\n')
    assert corpus.build_dataset_pool('pd')[0].n_rows == 4
    assert len(reads) == 2
    meta['synthetic']['target_column'] = 'absent'
    assert corpus.build_dataset_pool('pd') == []
    assert len(reads) == 3


def test_plan_reuses_split_per_filter_without_reusing_wrong_corpus(tmp_path, monkeypatch):
    import src.utils.prepare_experiment as module
    import src.train.corpus as corpus
    import src.train.config as pipeline
    cfg = pipeline.load_train_config(config_path='config/experiment0/null_pd.yaml')
    grid = [('missing-base.ckpt', lr, False, .4, 1, 'one_sample', rows, 0.)
            for lr in (0., 1e-6) for rows in (0, 10)]
    calls, identities = [], []

    def split(current, *, min_train_rows=None):
        calls.append(min_train_rows)
        return corpus.CorpusSplit([NS(dataset_id=f'train-{min_train_rows or 0}')],
                                 [NS(dataset_id='held-out')])

    def identity(current, trial, *, split):
        identities.append((trial[6], split.train[0].dataset_id))
        return {'sha256': f'{trial[1]}-{trial[6]}', 'specification': {}}

    monkeypatch.setattr(pipeline, 'load_train_config', lambda **kwargs: cfg)
    monkeypatch.setattr(pipeline, 'resolve_grid', lambda *args, **kwargs: grid)
    monkeypatch.setattr(corpus, 'split_from_cfg', split)
    monkeypatch.setattr(module, 'trial_identity', identity)
    monkeypatch.setattr(module, 'plan_path', lambda *args: tmp_path / 'plan.json')
    assert module.prepare(tmp_path / 'phase.yaml', write=True)['training_trials'] == 4
    assert calls == [None, 10]
    assert identities == [(0, 'train-0'), (10, 'train-10')] * 2


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
    import src.train.config as pipeline
    from src.train.loop import descriptive_name
    cfg = pipeline.load_train_config(config_path="config/experiment0/null_pd.yaml")
    cfg.checkpoint.trained_dir = str(tmp_path / "weights")
    trial = ("base.ckpt", 0., False, .4, 1, "one_sample", 0, 0.)
    monkeypatch.setattr(pipeline, "load_train_config", lambda **kw: cfg)
    monkeypatch.setattr(pipeline, "resolve_grid", lambda *a, **kw: [trial])
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"trials": {f"{cfg.run_name}_s00/0": "expected"}}))
    monkeypatch.setattr(module, "plan_path", lambda *a: plan)
    monkeypatch.setattr(module, "manifests_dir", lambda: tmp_path / "manifests")
    monkeypatch.setattr(module, "training_dir", lambda track: tmp_path / "manifests" / "epochs" / track)
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
    pd.DataFrame({"successful_updates": [0, 2], "metric__test__table": [.7, .7],
                  "metric__ood__package_breast_cancer": [.8, .8]}).to_csv(curve, index=False)
    pd.DataFrame({"successful_updates": [0, 2], "parameter": ["weight", "weight"],
                  "elements": [1, 1], "relative_change": [0., 0.]}).to_csv(
                      curve.with_name(curve.name.replace(".trajectory.csv", ".parameters.csv.gz")), index=False)
    pd.DataFrame({"gpu_status": ["sampled"]}).to_csv(
        curve.with_name(curve.name.replace(".trajectory.csv", ".resources.csv")), index=False)
    history = curve.with_name(curve.name.replace(".trajectory.csv", ".csv"))
    pd.DataFrame({"data_skipped_steps": [0], "amp_skipped_steps": [0]}).to_csv(history, index=False)
    result = module.audit(Path("unused"), null=True)
    assert result["passed"] and result["completed"] == 1
    for counter in ("data_skipped_steps", "amp_skipped_steps"):
        pd.DataFrame({"data_skipped_steps": [0], "amp_skipped_steps": [0], counter: [21]}).to_csv(history, index=False)
        result = module.audit(Path("unused"), null=True)
        assert not result["passed"]
        assert any("skipped training" in problem for problem in result["problems"])
    history.unlink()
    result = module.audit(Path("unused"), null=True)
    assert not result["passed"]
    assert any("measurements" in problem for problem in result["problems"])
    pd.DataFrame({"data_skipped_steps": [0], "amp_skipped_steps": [0]}).to_csv(history, index=False)
    pd.DataFrame({"gpu_status": ["unavailable"]}).to_csv(
        curve.with_name(curve.name.replace(".trajectory.csv", ".resources.csv")), index=False)
    assert not module.audit(Path("unused"), null=True)["passed"]


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
