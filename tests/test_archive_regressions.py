"""Regressions for defects found by re-auditing the first study's results (05-11-2026)."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
import torch


# ------------------------------------------------------------ training precision per family
def test_tabicl_trains_under_fp16_and_refuses_bf16(monkeypatch):
    from src.train.loop import _resolve_amp_dtype
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    auto = NS(train=NS(amp=True, amp_dtype="auto"))
    assert _resolve_amp_dtype(auto, "cuda", "tabpfn") == (True, torch.bfloat16)
    assert _resolve_amp_dtype(auto, "cuda", "tabicl") == (True, torch.float16)
    with pytest.raises(ValueError, match="BF16"):
        _resolve_amp_dtype(NS(train=NS(amp=True, amp_dtype="bf16")), "cuda", "tabicl")
    assert _resolve_amp_dtype(auto, "cpu", "tabicl") == (False, None)
    fp32 = NS(train=NS(amp=True, amp_dtype="auto", tabicl_precision="fp32"))
    assert _resolve_amp_dtype(fp32, "cuda", "tabicl") == (False, None)
    assert _resolve_amp_dtype(fp32, "cuda", "tabpfn") == (True, torch.bfloat16)


def test_audit_tolerates_only_bounded_fp16_scaler_skips():
    from src.utils.audit_experiment import tolerated_skips
    assert tolerated_skips({"data_skipped_steps": 0, "amp_skipped_steps": 6}, 250, fp16=True)
    assert not tolerated_skips({"data_skipped_steps": 0, "amp_skipped_steps": 13}, 250, fp16=True)
    assert not tolerated_skips({"data_skipped_steps": 0, "amp_skipped_steps": 1}, 250, fp16=False)
    assert not tolerated_skips({"data_skipped_steps": 1, "amp_skipped_steps": 0}, 250, fp16=True)


# --------------------------------------------------------------------- collapse detection
def test_collapse_flag_uses_the_last_monitor_and_the_median_table():
    from src.utils.audit_experiment import collapsed
    frame = pd.DataFrame({"score__test__a__prediction_std": [0.2, 1e-5],
                          "score__test__b__prediction_std": [0.2, 2e-5],
                          "score__ood__c__prediction_std": [0.2, 0.3]})
    assert collapsed(frame, "pd") is True
    assert collapsed(frame.iloc[:1], "pd") is False
    lgd = pd.DataFrame({"score__test__a__prediction_std_ratio": [0.6, 0.01]})
    assert collapsed(lgd, "lgd") is True
    assert collapsed(pd.DataFrame({"x": [1]}), "pd") is None


def test_regression_metrics_record_prediction_spread():
    from src.eval.metrics import _regression_metrics
    y = np.linspace(0, 1, 50)
    constant = _regression_metrics(pred_test=np.full(50, 0.5), y_test=y, neg_nll=None)
    assert constant["prediction_std"] == 0 and constant["prediction_std_ratio"] == 0
    assert constant["prediction_mean"] == pytest.approx(0.5)
    perfect = _regression_metrics(pred_test=y, y_test=y, neg_nll=None)
    assert perfect["prediction_std_ratio"] == pytest.approx(1.0)


# ------------------------------------------------------------------- label orientation
def test_auc_is_symmetric_under_label_flip_and_column_swap():
    """An AUC below 0.5 must be the model's, not a probability-column mix-up."""
    from src.eval.metrics import _classification_metrics
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.94).astype(int)               # majority-positive, like openml 46905
    p = np.clip(0.94 + 0.03 * (2 * y - 1) + rng.normal(0, 0.02, 400), 0, 1)
    proba = np.column_stack([1 - p, p])
    a = _classification_metrics(proba_test=proba, y_test=y, proba_val=proba, y_val=y, n_classes_seen=2)
    b = _classification_metrics(proba_test=proba[:, ::-1], y_test=1 - y, proba_val=proba[:, ::-1],
                                y_val=1 - y, n_classes_seen=2)
    assert a["roc_auc"] > 0.9
    assert a["roc_auc"] == pytest.approx(b["roc_auc"])


# ------------------------------------------- the monitor scores exactly like the benchmark
def _mixed_table(n=120):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"num": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)})
    X.loc[::7, "num"] = np.nan
    X.loc[::11, "cat"] = np.nan
    return X, (X["num"].fillna(0) > 0).astype(int).to_numpy()


@pytest.mark.parametrize("family", ["tabpfn", "tabicl"])
def test_monitor_feeds_wrappers_the_benchmark_encoding(monkeypatch, family):
    import src.train.dataloader as data
    from src.model import tabicl_models, tabpfn_models
    from src.train.loop import evaluate_ensemble_on_split
    X, y = _mixed_table()
    seen = {}

    class Wrapper:
        def __init__(self, **kwargs):
            seen["options"] = kwargs.get("inference_options")
        def fit(self, X_ctx, y_ctx, cat_idx, X_val=None, y_val=None):
            seen.update(dtype=X_ctx.dtype, cat_idx=cat_idx, X_ctx=X_ctx)
        def predict_proba(self, X_rows):
            p = np.full(len(X_rows), 0.5)
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(tabpfn_models, "TabPFNTrained", Wrapper)
    monkeypatch.setattr(tabicl_models, "TabICLTrained", Wrapper)
    monkeypatch.setattr(data, "_load_processed_csv", lambda ref: NS(X=X, y=y, cat_columns=("cat",)))
    options = {"use_amp": False, "use_fa3": False}
    evaluate_ensemble_on_split(ckpt_path="unused", refs=[NS(dataset_id="toy")], n_estimators=1,
                               n_subsample=len(y), query_fraction=.4, seed=3, device="cpu",
                               task_type="classification", metric_names=("roc_auc",), family=family,
                               tabicl_inference_options=options)
    assert seen["dtype"] == np.float32 and seen["cat_idx"] == [1]
    assert np.isnan(seen["X_ctx"][:, 0]).any()            # missing numerics reach the model as NaN
    assert set(np.unique(seen["X_ctx"][:, 1][~np.isnan(seen["X_ctx"][:, 1])])) <= {0.0, 1.0, 2.0}
    if family == "tabicl":
        assert seen["options"] == options                 # FP32, no FA3: same as evaluation


def test_monitor_and_benchmark_share_the_scorer():
    import inspect

    from src.train import loop
    source = inspect.getsource(loop.evaluate_ensemble_on_split)
    assert "score_fitted(" in source and "encode_for_model(" in source
    assert "_classification_metrics(" not in source and "_regression_metrics(" not in source


def test_every_tabicl_monitor_receives_the_evaluation_options():
    import inspect

    from src.train import loop
    source = inspect.getsource(loop.train_one_config)
    assert "tabicl_inference_options=monitor_tabicl_options" in source
    assert "if native_lora and family == \"tabicl\"" not in source


# ----------------------------------------------------------------- reproducible baselines
def test_classical_tuning_has_no_wall_clock_timeout():
    """A timeout makes the completed trial count depend on node speed (first study: 8-50)."""
    from omegaconf import OmegaConf

    from src.utils.paths import REPO_ROOT
    hpo = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "config/eval.yaml"))["hpo"]
    assert all(spec["timeout_seconds"] is None and spec["n_trials"] == 50 for spec in hpo.values())


def test_seeded_logistic_tuning_is_reproducible():
    pytest.importorskip("optuna")
    from src.model.linear import LogRegModel
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)); y = (X[:, 0] + rng.normal(0, 1, 300) > 0).astype(int)
    runs = []
    for _ in range(2):
        model = LogRegModel(hpo_trials=8, hpo_timeout_seconds=None)
        model.fit(X[:200], y[:200], [], X_val=X[200:250], y_val=y[200:250])
        runs.append((model.best_params, model.predict_proba(X[250:])))
    assert runs[0][0] == runs[1][0]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


# ------------------------------------------------------------- Experiment 0 coverage
def test_experiment0_declares_dose_response_noclip_and_precision():
    from src.utils import pilot_checks as pc
    from src.utils.design import load_design
    pilots = load_design()["experiment0"]["pilots"]
    rates = {pilots[n]["arm"].get("learning_rate", 3e-7) for n in ("zero_lr", "lr1e8", "reference", "lr1e5")}
    assert rates == {0.0, 1e-8, 3e-7, 1e-5}
    assert pilots["noclip"]["arm"]["grad_clip_norm"] is None
    assert pc.PRECISION_TASKS == 8
    assert len({pc.track_model_case(t, pc.PRECISION_TASKS) for t in range(pc.PRECISION_TASKS)}) == 8


def test_recovery_comparison_reads_diagscale_bundles(tmp_path):
    """DiagScale saves an adapter bundle, not full weights; the exact-resume check must read it."""
    from src.utils.recovery_check import compare
    bundle = {"creditpfn_adapter": {"schema": 1}, "adapter_state": {"blocks.0.attn.delta": torch.ones(4)}}
    a, b = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    torch.save(bundle, a); torch.save(bundle, b)
    assert compare(a, b)["passed"]
    torch.save({**bundle, "adapter_state": {"blocks.0.attn.delta": torch.ones(4) * 2}}, b)
    assert not compare(a, b)["passed"]
    full = {"state_dict": {"w": torch.zeros(2)}}
    torch.save(full, a); torch.save(full, b)
    assert compare(a, b)["passed"]


def test_training_identity_covers_the_monitor_protocol():
    from omegaconf import OmegaConf

    from src.utils import experiment
    base = OmegaConf.create({"seed": 1, "track": "pd", "train": {}, "evaluation": {
        "tabicl_inference_options": {"use_amp": False, "use_fa3": False}}})
    other = OmegaConf.create({**OmegaConf.to_container(base), "evaluation": {
        "tabicl_inference_options": {"use_amp": True, "use_fa3": False}}})
    assert experiment.scientific_config(base) != experiment.scientific_config(other)
    import inspect
    source = inspect.getsource(experiment.code_identity)
    assert all(f in source for f in ("src/eval/benchmark.py", "src/eval/dataset_loader.py",
                                     "config/domain_control.yaml"))


# ------------------------------------------------- pre-launch review (09-11-2026)
def test_numerical_stop_counts_update_attempts_not_epochs():
    from src.train.loop import no_finite_updates
    epoch = lambda steps, skipped: NS(epoch=1, optimizer_steps=steps, amp_skipped_steps=skipped)
    single_table = [epoch(1, 1)] * 5                           # five FP16 back-offs, one update each
    assert not no_finite_updates(single_table, 50)
    assert no_finite_updates([epoch(1, 1)] * 50, 50)
    assert not no_finite_updates([epoch(1, 1)] * 60 + [epoch(1, 0)], 50)   # a success resets
    assert no_finite_updates([epoch(13, 13)] * 4, 50)          # multi-table: about four epochs


def test_monitor_runs_without_tf32_and_restores_training_setting():
    from src.train.loop import strict_fp32
    torch.backends.cuda.matmul.allow_tf32 = True
    with strict_fp32():
        assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is True
    import inspect

    from src.train import loop
    assert "with strict_fp32():" in inspect.getsource(loop.train_one_config)


def test_lgd_capacity_uses_lgd_sized_contexts():
    from src.utils import pilot_checks as pc
    largest_lgd_context = int(57_932 * 0.8 * 0.8)
    assert max(pc.CAPACITY_ROWS["lgd"].values()) <= largest_lgd_context
    assert pc.CAPACITY_ROWS["pd"]["v3"] == 467_312 and pc.PRECISION_ROWS <= 10_000


def test_campaign_walltimes_are_explicit_and_validated(monkeypatch):
    from src.utils import campaign_launch as cl
    assert cl.campaign_walltimes() == cl.DEFAULT_WALLTIMES
    monkeypatch.setenv("CREDITPFN_TRAIN_WALLTIME", "06:30:00")
    assert cl.campaign_walltimes()["WALLTIME"] == "06:30:00"
    monkeypatch.setenv("CREDITPFN_EVAL_WALLTIME", "6h")
    with pytest.raises(ValueError):
        cl.campaign_walltimes()


def test_evaluation_arrays_reserve_only_their_own_lanes():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "scripts/slurm/run_experiment.sh").read_text()
    assert '(( EVAL_TASKS < elimit )) && elimit=$EVAL_TASKS' in source
    assert 'submit_array "$ECLUSTER" "$elimit"' in source
