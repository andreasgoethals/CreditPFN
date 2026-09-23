"""Catch stale plans, misleading readiness checks and lost cluster error logs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace as NS

from omegaconf import OmegaConf
import pytest

from scripts.train_pipeline import _load_cfg
from src.utils.experiment import digest_json


def test_source_identity_normalizes_line_endings_and_includes_slurm(tmp_path):
    from src.utils.experiment import code_identity
    source = tmp_path / "src/train/a.py"
    job = tmp_path / "scripts/slurm/a.slurm"
    for path in (source, job):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"initial\n")
    original = code_identity(tmp_path)
    source.write_bytes(b"initial\r\n")
    assert code_identity(tmp_path) == original
    job.write_bytes(b"different job\n")
    assert code_identity(tmp_path) != original


def test_submission_gate_rejects_changed_source_environment_config_and_corrupt_plan(tmp_path, monkeypatch):
    import src.utils.prepare_experiment as module
    cfg = _load_cfg(config_path="config/experiment0_pd.yaml")
    spec = {"code_sha256": "original", "versions": {"torch": "test"},
            "data_config": OmegaConf.to_container(OmegaConf.load("config/data.yaml"))["finetuning"]}
    key = digest_json(spec)
    payload = {"config": OmegaConf.to_container(cfg, resolve=True), "identities": {key: spec},
               "trials": {f"{cfg.run_name}_s00/0": key}}
    path = tmp_path / "plan.json"
    monkeypatch.setattr(module, "plan_path", lambda *args: path)
    monkeypatch.setattr(module, "code_identity", lambda: "original")
    monkeypatch.setattr(module, "environment_versions", lambda: {"torch": "test"})

    def save():
        path.write_text(json.dumps(dict(payload, sha256=digest_json(payload))), encoding="utf-8")

    save()
    assert module.check_prepared(Path("config/experiment0_pd.yaml"))["checked"]
    monkeypatch.setattr(module, "code_identity", lambda: "changed")
    with pytest.raises(RuntimeError, match="Code/environment"):
        module.check_prepared(Path("config/experiment0_pd.yaml"))
    monkeypatch.setattr(module, "code_identity", lambda: "original")
    monkeypatch.setattr(module, "environment_versions", lambda: {"torch": "changed"})
    with pytest.raises(RuntimeError, match="Code/environment"):
        module.check_prepared(Path("config/experiment0_pd.yaml"))
    payload["config"]["seed"] += 1
    save()
    with pytest.raises(RuntimeError, match="Configuration"):
        module.check_prepared(Path("config/experiment0_pd.yaml"))
    contents = json.loads(path.read_text())
    contents["sha256"] = "corrupt"
    path.write_text(json.dumps(contents))
    with pytest.raises(RuntimeError, match="checksum"):
        module.read_plan(path)


def test_plan_check_recomputes_input_identities_without_writing(tmp_path, monkeypatch):
    import src.utils.prepare_experiment as module
    import scripts.train_pipeline as pipeline
    import src.train.corpus as corpus
    cfg = _load_cfg(config_path="config/experiment0_pd.yaml")
    grid = [("missing.ckpt", 0., False, .4, 1, "one_sample", 0, 0.)]
    monkeypatch.setattr(pipeline, "_load_cfg", lambda **kw: cfg)
    monkeypatch.setattr(pipeline, "_resolve_grid", lambda *args, **kw: grid)
    monkeypatch.setattr(corpus, "split_from_cfg", lambda *args, **kw: corpus.CorpusSplit(
        [NS(dataset_id="train")], [NS(dataset_id="test")]))
    identity = {"sha256": "original-input", "specification": {}}
    monkeypatch.setattr(module, "trial_identity", lambda *args, **kw: dict(identity))
    path = tmp_path / "plan.json"
    monkeypatch.setattr(module, "plan_path", lambda *args: path)
    module.prepare(tmp_path / "config.yaml", write=True)
    original = path.read_bytes()
    assert module.prepare(tmp_path / "config.yaml", check=True)["checked"]
    identity["sha256"] = "changed-input"
    with pytest.raises(RuntimeError, match="no longer matches"):
        module.prepare(tmp_path / "config.yaml", check=True)
    assert path.read_bytes() == original


def test_preflight_uses_smallest_partition_and_rejects_mixed_pass_packing(monkeypatch):
    from src.utils.preflight import Report, check_step_budget, check_packing_divides
    import src.train.corpus as corpus
    cfg = _load_cfg(config_path="config/experiment1_pd.yaml")
    cfg.train.target_total_steps = 1201
    cfg.train.max_epochs_for_step_budget = 100
    monkeypatch.setattr(corpus, "split_from_cfg", lambda cfg, **kw: NS(
        train=[None] * (12 if cfg.corpus.fold == 2 else 13), test=[None]))
    report = Report()
    check_step_budget(cfg, "synthetic", report)
    assert report.n_fail == 1  # 1201/13 fits; 1201/12 does not.
    cfg = _load_cfg(config_path="config/sampling_pd.yaml")
    report = Report()
    check_packing_divides([(cfg, "synthetic")], report, trials_per_task=2)
    assert report.n_fail == 1


def test_empty_size_filter_cannot_silently_restore_the_unfiltered_corpus(monkeypatch):
    import src.train.corpus as corpus
    refs = [NS(dataset_id=f"synthetic-{i}", n_rows=10) for i in range(4)]
    monkeypatch.setattr(corpus, "build_dataset_pool", lambda *args: refs)
    with pytest.raises(ValueError, match="refusing to ignore"):
        corpus.split_corpus(track="pd", n_folds=2, min_train_rows=11)


def test_preflight_does_not_accept_a_base_from_an_unused_storage_root(tmp_path, monkeypatch):
    import src.utils.paths as paths
    import src.utils.preflight as module
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/base.ckpt").write_bytes(b"wrong location")
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(paths, "resolve_base_checkpoint", lambda p: tmp_path / "project" / p)
    report = module.Report()
    module.check_checkpoints(None, "synthetic", [("checkpoints/base.ckpt",)], report)
    assert report.n_fail == 1


@pytest.mark.parametrize("job", ["maintenance", "eval_classical", "cluster_report"])
def test_environment_failure_is_logged_and_propagated_before_python(tmp_path, job):
    bash = shutil.which("bash")
    if not bash and os.name == "nt":
        candidate = Path(os.environ["LOCALAPPDATA"]) / "Programs/Git/bin/bash.exe"
        bash = str(candidate) if candidate.is_file() else None
    if not bash:
        pytest.skip("Bash required for the batch-shell integration test")
    repo = Path(__file__).resolve().parents[1]
    node = tmp_path / "node"
    scripts = node / "CreditPFN/scripts/slurm"
    scripts.mkdir(parents=True)
    for name in (f"{job}.slurm", "_job_log.sh"):
        shutil.copyfile(repo / "scripts/slurm" / name, scripts / name)
    (scripts / "_activate_env.sh").write_text(
        'echo "synthetic activation failure" >&2\nreturn 17\n', encoding="utf-8")
    result = subprocess.run([bash, (scripts / f"{job}.slurm").as_posix(), "stage"],
        env=dict(os.environ, VSC_DATA=node.as_posix(),
                 CREDITPFN_OUTPUT_ROOT=(node / "CreditPFN").as_posix(), SLURM_JOB_ID="test-job"),
        capture_output=True, text=True)
    assert result.returncode == 17, result.stderr
    logs = list((node / "CreditPFN/output/logs").glob("*.log"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    assert "synthetic activation failure" in content and "END exit_code=17" in content
    assert not list((node / "CreditPFN").glob("*.log"))
