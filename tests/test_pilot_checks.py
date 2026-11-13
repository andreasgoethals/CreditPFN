"""Experiment 0 (second campaign): generated pilots stay the reference plus declared deviations."""
import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils import design
from src.utils import pilot_checks as pc
from src.utils.campaign_launch import _job_ids

ROOT = Path(__file__).resolve().parents[1]


def test_task_indices_cover_every_case_once():
    cases = [pc.recovery_case(t) for t in range(pc.RECOVERY_TASKS)]
    assert len(set(cases)) == pc.RECOVERY_TASKS == 24
    assert {c[0] for c in cases} == {"adamw", "diagscale", "lora"}
    assert {c[1] for c in cases} == {"pd", "lgd"} and {c[2] for c in cases} == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        pc.recovery_case(24)
    assert len({pc.track_model_case(t, pc.CAPACITY_TASKS) for t in range(pc.CAPACITY_TASKS)}) == 8
    with pytest.raises(ValueError):
        pc.track_model_case(8, pc.CAPACITY_TASKS)


def test_every_pilot_is_the_reference_plus_its_declared_arm():
    spec = design.load_design()
    for (name, track), cfg in design.pilot_configs().items():
        if name.startswith("recovery_"):
            variant = name.removeprefix("recovery_").rsplit("_", 1)[0]
            arm = design.recovery_arm(variant, spec)
            extra = set(design.PILOT_BOOKKEEPING) | set(design.RECOVERY_BOOKKEEPING)
        else:
            arm, extra = spec["experiment0"]["pilots"][name]["arm"], set(design.PILOT_BOOKKEEPING)
        left = set(design.undeclared_differences(cfg, arm, track, spec)) - extra
        assert not left, (name, track, left)
        assert cfg["run_name"].startswith("cpt2e0_")
        assert cfg["experiment"]["output_group"] == "experiment0"
        assert cfg["corpus"]["n_splits"] == 1


def test_recovery_twins_differ_only_by_run_name_and_are_deterministic():
    configs = design.pilot_configs()
    for variant in design.load_design()["experiment0"]["recovery"]["variants"]:
        for track in ("pd", "lgd"):
            a = dict(configs[(f"recovery_{variant}_uninterrupted", track)])
            b = dict(configs[(f"recovery_{variant}_resumed", track)])
            a.pop("run_name"), b.pop("run_name")
            assert a == b
            assert a["train"]["deterministic"] is True and a["train"]["trajectory_rows"] == []


def test_committed_pilot_configs_match_the_design(tmp_path):
    design.materialize_experiment0(tmp_path)
    fresh = {p.name: OmegaConf.load(p) for p in tmp_path.glob("*.yaml")}
    committed = {p.name: OmegaConf.load(p) for p in pc.FOLDER.glob("*.yaml")}
    assert fresh.keys() == committed.keys()
    assert all(fresh[k] == committed[k] for k in fresh), "regenerate with python -m src.utils.design --experiment0"


def test_pilots_cover_every_campaign_mechanism():
    pilots = design.load_design()["experiment0"]["pilots"]
    arms = [p["arm"] for p in pilots.values()]
    keys = {k for a in arms for k in a}
    assert {"trainable", "exposure", "grad_clip_norm",
            "corpus", "episode_rows", "single_source", "regression_objective", "l2sp_lambda"} <= keys
    assert {"domain_control"} <= {a.get("corpus") for a in arms}
    corpora = {a.get("corpus") for a in arms}
    assert {"small_only", "large_matched", "tables_above_10000_rows"} <= corpora
    scopes = {(a.get("trainable"), a.get("l2sp_lambda", 0.003)) for a in arms if "trainable" in a}
    assert {("lora_qv_r16", 0.003), ("lora_qv_r16", 0.0), ("diagscale", 0.003), ("diagscale", 0.0)} <= scopes
    assert {p.get("monitor_rows") for p in pilots.values()} >= {2000, 50000}


def test_pilot_paths_exclude_recovery_configs():
    names = [p.name for p in pc.pilot_paths()]
    assert names and not any(n.startswith("recovery_") for n in names)
    assert "zero_lr_pd.yaml" in names and "bar_nll_lgd.yaml" in names


def test_zero_lr_and_bar_nll_pilots():
    configs = design.pilot_configs()
    assert configs[("zero_lr", "pd")]["tunable"]["learning_rates"] == [0.0]
    bar = configs[("bar_nll", "lgd")]
    assert bar["train"]["regression_objective"] == "bar_nll"
    assert all("tabpfn-" in p for p in bar["tunable"]["regressor_base_paths"])
    assert configs[("reference", "lgd")]["train"]["regression_objective"] == "window_nll"


def test_submission_receipt_parses_train_array_ids():
    text = "Submitted train array 123 on mindwell (pd split=0 base=x time=1:00:00)\nnoise\n" \
           "Submitted train array 456 on mindwell (pd split=0 base=y time=1:00:00)"
    assert _job_ids(text) == ["123", "456"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
@pytest.mark.parametrize("run, group", [("cpt2e0_reference", "experiment0"),
                                        ("cpt2_update_size_anchoring_a3", "update_size_anchoring"),
                                        ("cpt2_transfer_a12", "transfer"),
                                        ("cpt2_reference_a0", "reference")])
def test_launcher_routes_campaign2_run_names(run, group):
    text = (ROOT / "scripts/slurm/run_experiment.sh").read_text(encoding="utf-8")
    block = text[text.index('case "$RUN" in'):text.index("esac", text.index('case "$RUN" in')) + 4]
    script = f'RUN={run}\n{block}\necho "$CREDITPFN_EXPERIMENT"\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()
    assert out == group
    from src.utils.paths import EXPERIMENTS
    assert group in EXPERIMENTS


def test_pilot_plans_live_with_experiment0():
    from src.utils.paths import group_for_run
    assert group_for_run("cpt2e0_zero_lr") == "experiment0"
    assert group_for_run("cpt2_seed_a0") == "seed"


def test_sanity_check_requires_the_trained_table_to_improve(monkeypatch):
    import pandas as pd
    frame = pd.DataFrame({"record_type": ["trajectory"] * 2, "train_metric": [0.70, 0.75],
                          "monitor_seconds": [1.0, 1.0]})
    monkeypatch.setattr(pc, "_trajectory", lambda row, track: frame)
    cfg = OmegaConf.create({"tunable": {"l2sp_lambdas": [0.003]}, "optimizer": {"l2sp_lambda": 0.003},
                            "train": {"adaptation_mode": "full"}})
    report = {"trials": [{"status": "OK", "trial": "t"}]}
    assert pc.pilot_checks("sanity_single", "pd", report, cfg)["t"]["passed"]
    assert not pc.pilot_checks("sanity_single", "lgd", report, cfg)["t"]["passed"]  # RMSE must fall


def test_adapter_pilots_require_a_logged_penalty_only_when_lambda_is_positive(monkeypatch):
    import pandas as pd
    report = {"trials": [{"status": "OK", "trial": "t"}]}
    for lam, penalty, ok in ((0.003, 1e-6, True), (0.003, 0.0, False), (0.0, 0.0, True)):
        frame = pd.DataFrame({"record_type": ["trajectory"], "train_metric": [0.7],
                              "monitor_seconds": [1.0], "l2sp_penalty": [penalty]})
        monkeypatch.setattr(pc, "_trajectory", lambda row, track, f=frame: f)
        cfg = OmegaConf.create({"tunable": {"l2sp_lambdas": [lam]}, "optimizer": {"l2sp_lambda": lam},
                                "train": {"adaptation_mode": "lora"}})
        assert pc.pilot_checks("lora", "pd", report, cfg)["t"]["passed"] is ok


def test_shell_group_lists_match_the_python_authority():
    import re

    from src.utils.paths import CAMPAIGN2_GROUPS, EXPERIMENTS
    text = (ROOT / "scripts/slurm/_job_log.sh").read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.strip().startswith("general|experiment0|"))
    assert set(line.strip().rstrip(") ;").split("|")) == set(EXPERIMENTS)
    text = (ROOT / "scripts/slurm/_eval_cache.sh").read_text(encoding="utf-8")
    donors = re.search(r"for donor in (.*?); do", text, re.S).group(1).replace("\\", " ").split()
    assert donors == list(CAMPAIGN2_GROUPS)


def test_monitor_rows_respect_each_models_context_limit():
    from src.train.loop import monitor_rows
    v2 = "checkpoints/tabpfn-v2-classifier-v2_default.ckpt"
    v3 = "checkpoints/tabpfn-v3-classifier-v3_default.ckpt"
    assert monitor_rows(50_000, v2, 0.4) == 16_666          # 0.6 x 16,666 <= 10,000 context rows
    assert monitor_rows(10_000, v2, 0.4) == 10_000
    assert monitor_rows(50_000, v3, 0.4) == 50_000
    assert monitor_rows(0, v2, 0.4) == 0


def test_benchmark_check_requires_every_model_table_and_fold(monkeypatch):
    import pandas as pd

    from src.data import retention
    from src.visualize import campaign2
    monkeypatch.setattr(retention, "load_refs", lambda panel, track: [])
    from src.train.config import load_train_config
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    cfg = apply_split_index(load_train_config(config_path=str(design.PILOT_FOLDER / "reference_pd.yaml")), 0)
    try:
        held_out = [r.dataset_id for r in split_from_cfg(cfg, track="pd").test]
    except FileNotFoundError as exc:
        pytest.skip(f"credit registry unavailable: {exc}")
    models = ([(f"tabpfn-trained[cpt2e0_reference_s00_pd_m{i}]", "tabpfn-trained") for i in range(4)]
              + [(f"tabpfn-untuned[m{i}]", "tabpfn-untuned") for i in range(4)]
              + [("catboost", "baseline"), ("xgboost", "baseline"), ("logreg", "baseline")])
    rows = [dict(model_name=n, model_source=s, test_dataset_id=t, fold_idx=f, status="OK")
            for n, s in models for t in held_out for f in range(5)]
    monkeypatch.setattr(campaign2, "_results_frames", lambda slug, track: pd.DataFrame(rows))
    assert pc.benchmark("pd")["passed"]
    monkeypatch.setattr(campaign2, "_results_frames", lambda slug, track: pd.DataFrame(rows[:-1]))
    out = pc.benchmark("pd")
    assert not out["passed"] and "baseline" in out["problems"][0]


@pytest.mark.slow
@pytest.mark.parametrize("task", [0, 4])        # PD and LGD with TabPFN v2
def test_capacity_check_runs_the_benchmark_scoring_path(task, monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "receipts", lambda: tmp_path)
    monkeypatch.setitem(pc.CAPACITY_ROWS["pd"], "v2", 200)
    monkeypatch.setitem(pc.CAPACITY_ROWS["lgd"], "v2", 200)
    monkeypatch.setitem(pc.CAPACITY_TEST_ROWS, "pd", 300)
    monkeypatch.setitem(pc.CAPACITY_TEST_ROWS, "lgd", 300)
    monkeypatch.setitem(pc.CAPACITY_VIEWS, "tabpfn", 2)
    import torch

    from src.model import tabpfn_models
    monkeypatch.setattr(tabpfn_models, "inference_precision", lambda: torch.float32)
    record = pc.capacity(task)
    assert record["passed"], record.get("error")
    assert record["test_rows"] == 300 and record["model"] == "v2"
