"""Reproductions from the September review and the downloaded null-audit failures."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
import torch


def test_evaluation_changes_do_not_change_training_identity(tmp_path):
    from src.utils.experiment import code_identity
    for name in ("src/train/loop.py", "src/eval/benchmark.py", "src/utils/consolidate_output.py",
                 "src/eval/metrics.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    training, evaluation = code_identity(tmp_path), code_identity(tmp_path, stage="eval")
    (tmp_path / "src/eval/benchmark.py").write_text("evaluation fix\n")
    (tmp_path / "src/utils/consolidate_output.py").write_text("consolidation fix\n")
    assert code_identity(tmp_path) == training
    assert code_identity(tmp_path, stage="eval") != evaluation
    (tmp_path / "src/eval/metrics.py").write_text("changed shared numerical metric\n")
    assert code_identity(tmp_path) != training


def test_eval_environment_covers_classical_hpo_and_data_libraries(monkeypatch):
    import src.utils.experiment as module
    monkeypatch.setattr(module, "version", lambda package: "1")
    original = module.environment_versions(stage="eval")
    for package in ("xgboost", "catboost", "optuna", "pandas"):
        monkeypatch.setattr(module, "version", lambda name: "2" if name == package else "1")
        assert module.environment_versions(stage="eval") != original


def test_diverged_resume_does_not_replace_measured_failure():
    from src.utils.consolidate_output import latest_trials
    common = dict(source_file="trial.csv", track="pd", status="DIVERGED")
    frame = pd.DataFrame([dict(common, source_row=0, elapsed_sec=73., final_train_loss=4., diverged_at_epoch=9),
                          dict(common, source_row=1, elapsed_sec=0.)])
    result = latest_trials(frame).iloc[0]
    assert result.elapsed_sec == 73. and result.diverged_at_epoch == 9 and result.attempt_count == 2


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


@pytest.mark.parametrize("family", ["tabpfn", "tabicl"])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("mode", ["one_sample", "full_pass", "accumulate"])
@pytest.mark.parametrize("l2sp", [0., .003])
def test_writer_reader_roundtrip_keeps_all_experimental_factors(family, frozen, mode, l2sp):
    from src.eval.benchmark import _method_dirname
    from src.model.base import ModelHandle
    from src.visualize.eval_viz import _decode_method_dirname
    base = "tabpfn-v3-classifier-v3_default.ckpt" if family == "tabpfn" else "tabicl-classifier-v2-20260212.ckpt"
    handle = ModelHandle(track="pd", task_type="classification", name="trial", source=f"{family}-trained", extra=dict(
        base_checkpoint=base, learning_rate=1e-5, use_lora=frozen,
        adaptation_mode="frozen_backbone" if frozen else "full", epoch_pass_mode=mode,
        min_train_rows=5000, l2sp_lambda=l2sp))
    meta = _decode_method_dirname(_method_dirname(handle))
    assert meta["lr"] == 1e-5 and meta["l2sp_lambda"] == l2sp
    assert meta["adaptation_mode"] == ("frozen_backbone" if frozen else "full")
    assert meta["epoch_pass_mode"] == mode and meta["min_train_rows"] == 5000
    assert "__" not in meta["base_short"]


def test_scheme_labels_do_not_merge_factors():
    from src.visualize.paper_figures import _scheme_label
    names = [f"tabpfn-trained__v3-default__lr1e-05{mode}__l2sp{lam}{frozen}"
             for mode in ("", "__fullpass", "__accumulate")
             for lam in ("0", "0.003") for frozen in ("", "__frozen")]
    assert len({_scheme_label(name) for name in names}) == len(names)


def paired_frame():
    rows = []
    for dataset, scale in (("small", 1.), ("large", 1000.)):
        for source, value in (("tabpfn-untuned", 1.), ("tabpfn-trained", .8)):
            rows.append(dict(method_dirname=f"{source}__v3-default", base_short="v3-default",
                             test_dataset_id=dataset, source=source, fold_idx=0, status="OK", rmse=scale * value))
    return pd.DataFrame(rows)


def test_paired_rmse_is_scale_invariant_and_zero_control_is_undefined():
    from src.visualize.paper_figures import paired_deltas
    frame = paired_frame()
    assert paired_deltas(frame, "rmse")["delta"].tolist() == pytest.approx([.2, .2])
    frame.loc[frame.source.eq("tabpfn-untuned"), "rmse"] = 0.
    assert paired_deltas(frame, "rmse")["delta"].isna().all()


def test_private_ids_in_failure_paths_and_errors_are_redacted(monkeypatch):
    from src.data import dataset_names as names
    from src.visualize import eval_viz, training_viz
    monkeypatch.setattr(names, "_PROPRIETARY", {"secret_table"})
    monkeypatch.setattr(names, "_DISPLAY", {"secret_table": "PropX"})
    frame = pd.DataFrame([dict(status="FAIL", test_dataset_id="PropX", source_file="a_ds-0042.secret_table.csv",
                              error="failed /data/secret_table_processed.csv")])
    monkeypatch.setattr(eval_viz, "load_eval_results", lambda _: frame)
    assert "secret_table" not in eval_viz.failed_pairs("pd").to_string()
    assert "secret_table" not in names.display_frame(frame).to_string()
    assert names.redact_private_names("notsecret_tablex") == "notsecret_tablex"
    failure = dict(trial_name="trial", base_short="base", learning_rate=0., use_lora=True,
                   seed=42, elapsed_sec=3., status="FAIL", error="failed secret_table")
    monkeypatch.setattr(training_viz, "load_run_manifest", lambda *a, **kw: pd.DataFrame([failure]))
    assert "secret_table" not in training_viz.failed_trials("pd").to_string()


def test_cached_timing_stays_visible(monkeypatch):
    import matplotlib.pyplot as plt
    from src.visualize import eval_viz
    frame = pd.DataFrame([dict(status="OK", method_name="xgboost", roc_auc=.8, elapsed_sec=0.,
                               cache_hit=True, cached_elapsed_sec=12.)])
    monkeypatch.setattr(eval_viz, "load_eval_results", lambda _: frame)
    fig = eval_viz.plot_time_vs_metric("pd")
    assert fig.axes[0].collections[0].get_offsets()[0, 0] == 12.
    plt.close(fig)


def test_default_scheme_metrics_include_the_brier_score():
    import matplotlib.pyplot as plt
    from src.visualize.paper_figures import plot_scheme_metrics
    frame = paired_frame()
    for metric in ("roc_auc", "brier_score", "ece", "f1"):
        frame[metric] = np.where(frame.source.eq("tabpfn-trained"), .4, .3)
    fig = plot_scheme_metrics(frame)
    assert "brier_score" in [label.get_text() for label in fig.axes[0].get_legend().get_texts()]
    plt.close(fig)


def test_unknown_dataset_size_is_not_plotted():
    import matplotlib.pyplot as plt
    from src.visualize.paper_figures import plot_regime_effect
    frame = paired_frame()
    manifest = pd.DataFrame({"dataset_id": ["small", "large"], "n_rows": [-1, -1]})
    fig = plot_regime_effect(frame, manifest, "rmse")
    assert not fig.axes[0].collections
    plt.close(fig)


def test_legacy_relative_output_paths_cannot_create_a_second_tree(tmp_path, monkeypatch):
    from src.utils.paths import resolve_output_path, resolve_staging_path
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(tmp_path / "project"))
    assert resolve_output_path("output/logs/test.log") == tmp_path / "output/general/logs/test.log"
    assert resolve_staging_path("output/results").parts[-3:] == ("output", "general", "results")


def test_null_audit_canonicalizes_legacy_keys_but_checks_weights_and_borders(tmp_path, monkeypatch):
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


def test_results_notebooks_save_paired_figures_and_use_real_brier_column():
    paths = list(Path("notebooks/experiment1").glob("*_results_*.ipynb"))
    assert len(paths) == 2
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert "cp.show(sink, cp.plot_secondary_tradeoff(run))" in source
        assert "'brier_score'" in source and "'brier'," not in source


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
    from scripts import eval_pipeline as module
    import src.eval.benchmark as benchmark
    import src.eval.cache as cache
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


def test_failed_frozen_trial_retains_its_real_epoch_stem(tmp_path, monkeypatch):
    from src.visualize import training_viz as module
    monkeypatch.setattr(module, "_resolve_paths", lambda *a: dict(run_name="phase", manifest_dir=tmp_path))
    pd.DataFrame([dict(track="pd", base_checkpoint="tabpfn-v3-classifier-v3_default.ckpt",
        learning_rate=1e-5, seed=42, use_lora=True, adaptation_mode="frozen_backbone",
        status="FAIL", final_ckpt_path="", epoch_pass_mode="one_sample", l2sp_lambda=0.,
        query_fraction=.4, accumulate_grad_batches=1)]).to_csv(tmp_path / "phase_s00_pd.csv", index=False)
    frame = module.load_run_manifest("pd")
    assert frame.trial_name.iloc[0].endswith("_frozen")


def test_fold_stability_summary_uses_actual_fold_column(monkeypatch):
    from src.visualize import summaries as module
    frame = pd.DataFrame([dict(method_name="xgboost", method_dirname="xgboost", source="baseline",
        base_short="xgboost", test_dataset_id="synthetic", fold_idx=i, roc_auc=.5 + i / 10,
        elapsed_sec=1., status="OK") for i in range(5)])
    monkeypatch.setattr(module.ev, "load_eval_results", lambda _: frame)
    assert "most stable" in module.eval_summary("pd")


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
