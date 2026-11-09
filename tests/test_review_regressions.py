"""Regression tests for code identity, evaluation resume, output routing and null audits."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest
import torch


def test_evaluation_changes_do_not_change_training_identity(tmp_path):
    from src.utils.experiment import code_identity
    for name in ("src/train/loop.py", "src/eval/benchmark.py", "src/eval/campaign_tasks.py",
                 "src/eval/metrics.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    training, evaluation = code_identity(tmp_path), code_identity(tmp_path, stage="eval")
    (tmp_path / "src/eval/campaign_tasks.py").write_text("evaluation task fix\n")
    assert code_identity(tmp_path) == training
    assert code_identity(tmp_path, stage="eval") != evaluation
    # The training monitor scores through the benchmark's scorer, so it is training code too.
    (tmp_path / "src/eval/benchmark.py").write_text("shared scorer change\n")
    assert code_identity(tmp_path) != training
    training = code_identity(tmp_path)
    (tmp_path / "src/eval/metrics.py").write_text("changed shared numerical metric\n")
    assert code_identity(tmp_path) != training


def test_eval_environment_covers_classical_hpo_and_data_libraries(monkeypatch):
    import src.utils.experiment as module
    monkeypatch.setattr(module, "version", lambda package: "1")
    original = module.environment_versions(stage="eval")
    for package in ("xgboost", "catboost", "optuna", "pandas"):
        monkeypatch.setattr(module, "version", lambda name: "2" if name == package else "1")
        assert module.environment_versions(stage="eval") != original


def test_packing_is_invariant_to_manifest_order(monkeypatch):
    from scripts import eval_pipeline as module
    from src.model.base import ModelHandle
    roster = [(ModelHandle(track="pd", task_type="classification", name=f"model{i}", source="tabpfn-trained", base_path=f"/old/{i}.ckpt"), None)
              for i in range(9)]
    monkeypatch.setattr(module, "_dataset_rows", lambda _: {})
    def cells(roster):
        pairs = [(i, ds) for i in range(len(roster)) for ds in ("a", "b", "c")]
        bins = module._pack_tasks(pairs, roster, n_tasks=4, track="pd", max_rows_per_model=None)
        return [sorted((roster[pairs[j][0]][0].name, pairs[j][1]) for j in bucket) for bucket in bins]
    assert cells(roster) == cells(roster[2:] + roster[:2])


def test_failed_retry_invalidates_earlier_ok_fold(tmp_path):
    from src.eval.benchmark import find_existing_results
    from src.model.base import ModelHandle
    folder = tmp_path / "PD/xgboost"
    folder.mkdir(parents=True)
    handle = ModelHandle(track="pd", task_type="classification", name="xgboost", source="baseline")
    for attempt, failed in ((1, 4), (2, 3)):
        pd.DataFrame([dict(test_dataset_id="synthetic", fold_idx=i, status="FAIL" if i == failed else "OK")
                      for i in range(5)]).to_csv(folder / f"phase_20260924_0{attempt}.csv", index=False)
    assert find_existing_results(handle, "synthetic", track="pd", results_base_dir=tmp_path,
                                 n_folds_required=5, run_name="phase") == []
    pd.DataFrame([dict(test_dataset_id="synthetic", fold_idx=3, status="OK")]).to_csv(
        folder / "phase_20260924_03.csv", index=False)
    assert find_existing_results(handle, "synthetic", track="pd", results_base_dir=tmp_path,
                                 n_folds_required=5, run_name="phase")


def test_relative_output_paths_cannot_create_a_second_tree(tmp_path, monkeypatch):
    from src.utils.paths import resolve_output_path, resolve_staging_path
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(tmp_path / "project"))
    assert resolve_output_path("output CreditPFN/logs/test.log") == tmp_path / "output CreditPFN/general/logs/test.log"
    assert resolve_staging_path("output CreditPFN/results").parts[-3:] == ("output CreditPFN", "general", "results")


def test_null_audit_canonicalizes_renamed_keys_but_checks_weights_and_borders(tmp_path, monkeypatch):
    import tabpfn.base

    from src.utils.audit_experiment import compare_states
    before = tmp_path / "tabpfn-v2-regressor-v2_default.ckpt"
    after = tmp_path / "run_tabpfn-v2-regressor-v2_default.ckpt"
    torch.save({"state_dict": {"old.weight": torch.ones(2)}}, before)
    torch.save({"state_dict": {"new.weight": torch.ones(2)}}, after)
    changes = {}
    def loader(model_path, *, download_if_not_exists, **kwargs):
        assert download_if_not_exists is False
        state = torch.load(model_path, weights_only=True)["state_dict"]
        weight = next(iter(state.values()))
        model = NS(state_dict=lambda: {"weight": weight})
        criterion = NS(state_dict=lambda: dict(borders=changes.get("borders", torch.arange(3.))
            if model_path == after else torch.arange(3.), losses_per_bucket=torch.full((2,), float(model_path == after))))
        return [model], criterion, [], None
    monkeypatch.setattr(tabpfn.base, "load_model_criterion_config", loader)
    assert compare_states(before, after, tabpfn_track="lgd")["equal"]
    changes["borders"] = torch.arange(3.) * 2
    assert compare_states(before, after, tabpfn_track="lgd")["changed"] == ["criterion.borders"]
    changes.clear()
    torch.save({"state_dict": {"new.weight": torch.zeros(2)}}, after)
    assert compare_states(before, after, tabpfn_track="lgd")["changed"] == ["model.weight"]


def test_results_notebooks_use_the_second_campaign_loader():
    import ast
    paths = sorted(Path("notebooks").glob("rq[1-3]/02_results.ipynb"))
    assert len(paths) == 3
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        ast.parse("\n".join(line for line in source.splitlines() if not line.startswith("%")))
        # Endpoint comparisons go through the second-campaign loader, never ad hoc frames.
        assert "c2.endpoint_gains(" in source and "c2.gain_summary(" in source


def test_submission_keeps_successful_stderr_out_of_dependency_id():
    import os
    import shutil
    import subprocess
    bash = shutil.which("bash") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Git/bin/bash.exe")
    if not Path(bash).exists():
        pytest.skip("Bash is required for submission helper verification")
    source = Path("scripts/slurm/run_experiment.sh").read_text(encoding="utf-8")
    helper = source[source.index("submit_retry() {"):source.index("submit_array() {")]
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", helper + '''
fake_submit() { echo '12345;mindwell'; echo 'informational diagnostic' >&2; }
out=$(submit_retry fake_submit)
printf '%s' "$out"
'''], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and result.stdout == "12345;mindwell"
    assert "informational diagnostic" in result.stderr


def test_missing_inputs_are_prepared_before_skip_fingerprints(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    import src.eval.benchmark as benchmark
    import src.eval.cache as cache
    from scripts import eval_pipeline as module
    eval_cfg = OmegaConf.create({"results": {"base_dir": str(tmp_path)}, "cv": {"n_folds": 5}})
    train_cfg = OmegaConf.create({"track": "pd", "run_name": "phase", "seed": 42,
                                  "experiment": {"fingerprint": True}})
    monkeypatch.setattr(module, "load_eval_configs", lambda *a, **kw: (eval_cfg, train_cfg))
    monkeypatch.setattr(module, "apply_data_source_from_cfg", lambda *a: None)
    monkeypatch.setattr(module, "resolve_run_log", lambda *a, **kw: (NS(path=tmp_path / "log", write=lambda _: None), None))
    monkeypatch.setattr(module, "setup_logging", lambda *a: None)
    monkeypatch.setattr(module, "dump_resolved", lambda *a, **kw: None)
    handle = NS(name="control")
    plan = [((handle, None), ["synthetic"])]
    monkeypatch.setattr(module, "_filter_roster", lambda *a, **kw: plan)
    input_path = tmp_path / "processed.csv"
    def roster(*a):
        assert input_path.is_file()  # complete-registry partitioning precedes cache lookup
        return [], [], None
    monkeypatch.setattr(module, "_build_roster", roster)
    monkeypatch.setattr(module, "_ensure_processed", lambda *a, **kw: input_path.write_text("x,y\n1,0\n"))
    def fingerprint(*a, **kw):
        assert input_path.is_file()
        return "correct-key"
    monkeypatch.setattr(cache, "evaluation_key", fingerprint)
    monkeypatch.setattr(benchmark, "find_existing_results", lambda *a, **kw: [Path("already-scored.csv")])
    assert module.run() == 0


@pytest.mark.parametrize("family", ["tabpfn", "tabicl"])
@pytest.mark.parametrize("kind", ["all", "foundation", "classical"])
@pytest.mark.parametrize("identity", ["planned", "foreign", None])
def test_eval_roster_enforces_saved_training_identity(tmp_path, monkeypatch, family, kind, identity):
    from omegaconf import OmegaConf

    from scripts import eval_pipeline
    from src.utils import prepare_experiment

    cfg = OmegaConf.create(dict(run_name="planned_s00", device="cpu", train=dict(retention_panel="none"),
        experiment=dict(require_plan=True), tunable=dict(classifier_base_paths=[])))
    eval_cfg = OmegaConf.create(dict(baselines=dict(enabled=[]), tabpfn_n_estimators=2))
    monkeypatch.setenv("CREDITPFN_EVAL_KIND", kind)
    monkeypatch.setattr("src.train.corpus.split_from_cfg", lambda *a, **kw: NS(train=[], test=[]))
    monkeypatch.setattr("src.model.build_baselines", lambda **kw: [])
    monkeypatch.setattr(eval_pipeline, "manifests_dir", lambda: tmp_path)
    monkeypatch.setattr(prepare_experiment, "read_plan", lambda _: {
        "trials": {"planned_s00/0": "planned", "planned_s01/0": "foreign"}})
    checkpoint = tmp_path / "trained.ckpt"
    checkpoint.write_bytes(b"fixture; no model is loaded")
    provenance = {"trial_identity": {"sha256": identity}} if identity is not None else {}
    Path(str(checkpoint) + ".provenance.json").write_text(json.dumps(provenance))
    pd.DataFrame([dict(track="pd", base_checkpoint=f"{family}-classifier.ckpt", learning_rate=1e-6,
        seed=42, final_ckpt_path=str(checkpoint), status="OK")]).to_csv(
            tmp_path / "planned_s00_pd.csv", index=False)
    if identity != "planned":
        with pytest.raises(RuntimeError, match="does not belong to the prepared training plan"):
            eval_pipeline._build_roster(eval_cfg, cfg, "pd")
    else:
        roster, _, _ = eval_pipeline._build_roster(eval_cfg, cfg, "pd")
        assert len(roster) == (0 if kind == "classical" else 1)


@pytest.mark.parametrize("dry", [False, True])
def test_dry_submission_reports_failed_plan_gate(dry):
    import os
    import shutil
    import subprocess
    bash = shutil.which("bash") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Git/bin/bash.exe")
    if not Path(bash).exists():
        pytest.skip("Bash unavailable")
    source = Path("scripts/slurm/run_experiment.sh").read_text(encoding="utf-8")
    start = source.index('if [[ "${META[5]}" == True ]]')
    guard = source[start:source.index("hms()", start)]
    prefix = 'set -e; CONFIG=unused; STAGES=train; META=(0 0 0 0 0 True); DRY=' + ("1" if dry else "")
    script = prefix + '\npython() { echo "plan checksum failed" >&2; return 9; }\n' + guard + '\necho continued\n'
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", script], capture_output=True, text=True, timeout=20)
    assert "plan checksum failed" in result.stderr
    if dry:
        assert result.returncode == 0 and "NOT READY" in result.stderr and "continued" in result.stdout
    else:
        assert result.returncode != 0 and "continued" not in result.stdout
