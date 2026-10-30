"""LoRA gate identity, evidence integrity and isolated execution, without research runs."""
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from src.utils import lora_checks as checks
from src.utils.atomic import write_json
from src.utils.experiment import digest_json, file_digest


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "folder", lambda identifier: tmp_path / "data" / checks.validate_id(identifier))
    monkeypatch.setattr(checks, "evidence_folder", lambda identifier: tmp_path / "project" / checks.validate_id(identifier))
    monkeypatch.setattr(checks, "fingerprint", lambda: "source-a")
    monkeypatch.setattr(checks, "environment_versions", lambda **kw: {"torch": "same"})
    identifier = "workflow1"
    source = tmp_path / "base.ckpt"
    config = tmp_path / "config.yaml"
    source.write_bytes(b"base weights")
    config.write_text("recipe", encoding="utf-8")
    workflow = {"schema": checks.SCHEMA, "id": identifier, "kind": "creditpfn_lora", "prepared": True,
                "fingerprint": "source-a", "versions": {"torch": "same"},
                "inputs": [checks._binding(source)], "files": [checks._binding(config)]}
    write_json(checks.folder(identifier) / "workflow.json", workflow)
    for stage in ("smoke", "capacity"):
        for task in range(4):
            artifact = tmp_path / f"{stage}_{task}.ckpt"
            artifact.write_bytes(b"verified weights")
            evidence = checks.evidence_folder(identifier) / "results" / f"{stage}_{task}.json"
            write_json(evidence, {"artifacts": [checks._binding(artifact)]})
            write_json(checks.folder(identifier) / "receipts" / f"{stage}_{task}.json", {
                "schema": checks.SCHEMA, "id": identifier, "stage": stage, "task": task, "passed": True,
                "fingerprint": "source-a", "versions": {"torch": "same"},
                "workflow_sha256": digest_json(workflow), "evidence": str(evidence),
                "evidence_sha256": file_digest(evidence),
            })
    checks.inspect_gate(identifier)
    return identifier, tmp_path


def test_all_eight_receipts_required_and_explicit_env_selects_gate(gate, monkeypatch):
    identifier, _ = gate
    monkeypatch.setenv("CREDITPFN_LORA_GATE", identifier)
    assert checks.require_gate() == identifier
    (checks.folder(identifier) / "receipts/capacity_3.json").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        checks.require_gate()


@pytest.mark.parametrize("override", ["cv.n_folds=3", "tabicl_inference_options.use_amp=true",
                                      "tabpfn_n_estimators=1", "results.save_predictions=false"])
def test_evaluation_rechecks_validated_numerical_protocol(monkeypatch, override):
    from src.eval.config import load_eval_configs
    calls = []
    monkeypatch.setattr(checks, "require_gate", lambda: calls.append("verified"))
    eval_cfg, train_cfg = load_eval_configs([], [], config_path="config/experiment6/pd.yaml", split_index=1)
    # Relocating output does not change the scientific protocol.
    eval_cfg.results.base_dir = "other/output"
    checks.validate_evaluation(train_cfg, eval_cfg)
    assert calls == ["verified"]
    changed = OmegaConf.merge(eval_cfg, OmegaConf.from_dotlist([override]))
    with pytest.raises(RuntimeError, match="benchmark settings differ"):
        checks.validate_evaluation(train_cfg, changed)


def test_actual_benchmark_entry_rechecks_gate_before_building_models(monkeypatch):
    from scripts import eval_pipeline
    def reject(*args):
        raise RuntimeError("independent LoRA inspection is stale")
    monkeypatch.setattr(checks, "require_gate", reject)
    from src.eval.config import load_eval_configs
    eval_cfg, train_cfg = load_eval_configs([], [], config_path="config/experiment6/pd.yaml", split_index=0)
    with pytest.raises(RuntimeError, match="inspection is stale"):
        eval_pipeline._build_roster(eval_cfg, train_cfg, "pd")
    monkeypatch.setattr(eval_pipeline, "_ensure_processed", lambda *a, **kw: pytest.fail("stale gate reached shared data cache"))
    with pytest.raises(RuntimeError, match="inspection is stale"):
        eval_pipeline.run(config="config/experiment6/pd.yaml", split_index=0)


@pytest.mark.parametrize("what", ["base", "config", "evidence", "artifact", "source", "environment", "failed", "wrong_task", "wrong_id", "unprepared"])
def test_gate_rejects_drift_and_cannot_reuse_other_receipts(gate, monkeypatch, what):
    identifier, root = gate
    if what in {"base", "config", "evidence", "artifact"}:
        path = {"base": root / "base.ckpt", "config": root / "config.yaml",
                "evidence": checks.evidence_folder(identifier) / "results/smoke_0.json",
                "artifact": root / "smoke_0.ckpt"}[what]
        path.write_text("changed beyond previous contents", encoding="utf-8")
    elif what == "source":
        monkeypatch.setattr(checks, "fingerprint", lambda: "source-b")
    elif what == "environment":
        monkeypatch.setattr(checks, "environment_versions", lambda **kw: {"torch": "different"})
    else:
        path = checks.folder(identifier) / ("workflow.json" if what == "unprepared" else "receipts/smoke_0.json")
        result = json.loads(path.read_text())
        key, value = {"failed": ("passed", False), "wrong_task": ("task", 1),
                      "wrong_id": ("id", "other"), "unprepared": ("prepared", False)}[what]
        result[key] = value
        write_json(path, result)
    with pytest.raises(RuntimeError):
        checks.require_gate(identifier)


@pytest.mark.parametrize("identifier", ["", "../other", "x/y", "a_b", "a-b", "ø", "x" * 81, None])
def test_workflow_ids_cannot_escape_their_own_namespace(identifier):
    with pytest.raises(ValueError):
        checks.validate_id(identifier)


def test_scope_restores_environment_after_failure(monkeypatch):
    monkeypatch.setenv("CREDITPFN_EXPERIMENT", "experiment1")
    monkeypatch.delenv("CREDITPFN_LORA_WORKFLOW", raising=False)
    with pytest.raises(RuntimeError):
        with checks.workflow_scope("abc123"):
            import os
            assert os.environ["CREDITPFN_LORA_WORKFLOW"] == "abc123"
            assert os.environ["CREDITPFN_EXPERIMENT"] == "experiment0"
            raise RuntimeError("test failure")
    import os
    assert os.environ["CREDITPFN_EXPERIMENT"] == "experiment1"
    assert "CREDITPFN_LORA_WORKFLOW" not in os.environ


@pytest.mark.parametrize("track,family", checks.CASES)
def test_smoke_changes_only_scoped_workload_capacity_keeps_all_partitions(track, family, monkeypatch):
    from src.train.config import load_train_config, resolve_grid
    kind = "classifier" if track == "pd" else "regressor"
    basename = (f"tabpfn-v3-{kind}-v3_default.ckpt" if family == "tabpfn"
                else f"tabicl-{kind}-v2-20260212.ckpt")
    monkeypatch.setattr(checks, "_base", lambda t, f: Path("checkpoints") / basename)
    source = load_train_config(config_path=f"config/experiment6/{track}.yaml")
    source_before = OmegaConf.to_container(source, resolve=True)
    limits = checks.policy()
    train = [SimpleNamespace(dataset_id="train1"), SimpleNamespace(dataset_id="train2")]
    held = [SimpleNamespace(dataset_id="held")]
    smoke = checks.make_config(source, "tiny1", family, rate=1e-4, arm="lr1_reference", limits=limits, train=train, held=held)
    capacity = checks.make_config(source, "tiny1", family, rate=1e-3, arm="capacity", limits=limits)
    assert OmegaConf.to_container(source, resolve=True) == source_before
    assert smoke.train.target_total_steps == 6 and smoke.train.max_rows_per_step == 128
    assert smoke.corpus.train_dataset_ids == ["train1", "train2"]
    assert smoke.corpus.test_dataset_ids == ["held"]
    assert smoke.experiment.output_group == capacity.experiment.output_group == "experiment0"
    assert not smoke.experiment.lora_gate and not capacity.experiment.lora_gate
    assert capacity.corpus.n_splits == source.corpus.n_splits == 4
    assert capacity.corpus.train_dataset_ids == source.corpus.train_dataset_ids
    assert capacity.train.max_rows_per_step == source.train.max_rows_per_step
    assert capacity.train.amp == source.train.amp
    assert capacity.train.context_sampling == source.train.context_sampling
    assert capacity.train.target_total_steps == 16
    assert capacity.checkpoint.trained_dir == "checkpoints/trained/experiment0/lora/tiny1"
    assert len(resolve_grid(smoke, single=False)) == len(resolve_grid(capacity, single=False)) == 1
    assert smoke.lora == source.lora and smoke.optimizer == source.optimizer


def test_pipeline_interrupt_must_return_75_and_cleans_inherited_flow(monkeypatch, tmp_path):
    monkeypatch.setattr(checks, "evidence_folder", lambda i: tmp_path)
    seen = []
    monkeypatch.setenv("CREDITPFN_FLOW_ID", "oldflow")
    monkeypatch.setenv("CREDITPFN_STOP_REQUESTED", "1")
    monkeypatch.setenv("CREDITPFN_LORA_WORKFLOW", "workflow1")
    def run(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(returncode=75)
    monkeypatch.setattr(checks.subprocess, "run", run)
    checks._run_training(Path("test.yaml"), stop=3)
    env = seen[0][1]["env"]
    assert env["CREDITPFN_STOP_AFTER_UPDATES"] == "3" and env["CREDITPFN_LORA_WORKFLOW"] == "workflow1"
    assert "CREDITPFN_STOP_REQUESTED" not in env and "CREDITPFN_FLOW_ID" not in env
    with pytest.raises(RuntimeError, match="expected 0"):
        checks._run_training(Path("test.yaml"))


def test_capacity_visits_four_partitions_not_only_first(monkeypatch, tmp_path):
    import pandas as pd
    seen = []
    monkeypatch.setattr(checks, "folder", lambda i: tmp_path)
    path = tmp_path / "configs/pd_tabpfn_capacity.yaml"
    path.parent.mkdir()
    path.write_text("corpus: {n_splits: 4}\ntrain: {amp: true, max_rows_per_step: 8192}\n")
    def train(p, split=0):
        seen.append(split)
        return path
    monkeypatch.setattr(checks, "_run_training", train)
    monkeypatch.setattr(checks, "_audited", lambda p, split=0: {"successful_updates": 16, "path": str(path)})
    monkeypatch.setattr(checks, "_curve", lambda r, t: (pd.DataFrame({"processed_rows": [0, 8000]}), path))
    result = checks.capacity("abc", "pd", "tabpfn")
    assert seen == [0, 1, 2, 3] and len(result["partitions"]) == 4


def test_failed_smoke_receipt_prevents_capacity_before_loading_gpu(gate):
    identifier, _ = gate
    path = checks.folder(identifier) / "receipts/smoke_2.json"
    result = json.loads(path.read_text())
    result["passed"] = False
    write_json(path, result)
    with pytest.raises(RuntimeError, match="receipt"):
        checks.main(["capacity", "--id", identifier, "--task", "0"])


def test_independent_inspection_is_required_and_fast_gate_does_not_hash_weights(gate, monkeypatch):
    identifier, _ = gate
    inspection = checks.folder(identifier) / "inspection.json"
    inspection.unlink()
    with pytest.raises(FileNotFoundError):
        checks.require_gate(identifier)
    checks.inspect_gate(identifier)
    original = checks.file_digest
    hashes = []
    def digest(path):
        hashes.append(str(path))
        return original(path)
    monkeypatch.setattr(checks, "file_digest", digest)
    assert checks.require_gate(identifier) == identifier
    assert not any(path.endswith(".ckpt") for path in hashes)


@pytest.mark.parametrize("problem", [None, "keys", "head", "key_slice", "missing_stage", "nonfinite"])
def test_actual_training_weight_scope_preserves_every_non_qv_tensor(problem):
    import torch
    stages = ["col_embedder", "row_interactor", "icl_predictor"]
    targets = [{"module": stage + ".attention", "weight": "in_proj_weight", "slices": [[0, 2], [4, 6]]}
               for stage in stages]
    before = {stage + ".attention.in_proj_weight": torch.ones(6, 2) for stage in stages}
    before["head.weight"] = torch.ones(2, 2)
    after = {key: value.clone() for key, value in before.items()}
    for target in targets:
        after[target["module"] + ".in_proj_weight"][0, 0] += .1
    first = stages[0] + ".attention.in_proj_weight"
    if problem == "keys":
        after["unexpected"] = torch.ones(1)
    elif problem == "head":
        after["head.weight"] += .1
    elif problem == "key_slice":
        after[first][2, 0] += .1
    elif problem == "missing_stage":
        after[first] = before[first].clone()
    elif problem == "nonfinite":
        after[first][0, 0] = float("nan")
    if problem:
        with pytest.raises(AssertionError):
            checks.weight_scope(before, after, targets, "tabicl")
    else:
        result = checks.weight_scope(before, after, targets, "tabicl")
        assert result["only_qv_changed"] and result["changed_target_count"] == 3


@pytest.mark.parametrize("precision", ["float32", "automatic"])
def test_tabpfn_probe_keeps_explicit_v3_under_both_inference_policies(monkeypatch, precision):
    import torch
    from src.model import tabpfn_models
    seen = []
    monkeypatch.setattr(tabpfn_models, "_make_tabpfn", lambda kind, path, **kwargs: seen.append((kind, path, kwargs)))
    path = Path("tabpfn-v3-classifier-v3_default.ckpt")
    checks._estimator(path, "pd", "tabpfn", precision=precision)
    kind, selected, kwargs = seen[0]
    assert selected == path and kind == "classification"
    if precision == "float32":
        assert kwargs["inference_precision"] == torch.float32
    else:
        assert "inference_precision" not in kwargs


def test_automatic_probe_compares_same_query_batch_without_cross_precision_constraint(monkeypatch):
    import numpy as np
    seen = []
    class Estimator:
        def fit(self, X, y):
            return self
        def predict_proba(self, X):
            seen.append(len(X))
            return np.tile([.25, .75], (len(X), 1))
    policies = []
    def construct(path, track, family, **kwargs):
        policies.append(kwargs["precision"])
        return Estimator()
    monkeypatch.setattr(checks, "_estimator", construct)
    predictions = checks.prediction_probe([Path("base.ckpt"), Path("zero.ckpt")], "pd", "tabpfn",
                                         precision="automatic", check_chunks=False)
    assert policies == ["automatic", "automatic"]
    assert seen == [32, 32, 32, 32]
    assert np.array_equal(predictions[0], predictions[1])
