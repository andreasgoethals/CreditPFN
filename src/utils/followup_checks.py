"""Isolated experiment-0 gates for experiments 4–7; never submits production work."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from omegaconf import OmegaConf

from src.utils.atomic import write_json
from src.utils.experiment import code_identity, digest_json, environment_versions, file_digest
from src.utils.paths import REPO_ROOT, manifests_dir, results_dir, checkpoints_dir


def fingerprint():
    files = [*sorted((REPO_ROOT / "config/experiment10").glob("*.yaml")),
             *sorted((REPO_ROOT / "config/experiment5").glob("*.yaml")),
             *sorted((REPO_ROOT / "config/experiment8").glob("*.yaml")),
             *sorted((REPO_ROOT / "config/experiment9").glob("*.yaml")),
             *(REPO_ROOT / "config" / name for name in ("train.yaml", "eval.yaml", "data.yaml", "retention.yaml")),
             REPO_ROOT / "config/experiment0/followups.yaml",
             REPO_ROOT / "scripts/context_pipeline.py"]
    files += [REPO_ROOT / "scripts/slurm" / name for name in (
        "run_experiment0_followups.sh", "followup_prepare.slurm", "followup_gpu.slurm",
        "run_experiment10.sh", "context_benchmark.slurm", "run_experiment5.sh", "run_experiment8.sh",
        "run_experiment9.sh", "run_benchmark.sh", "_benchmark_dispatch.sh",
        "benchmark_foundation.slurm", "benchmark_classical.slurm")]
    return digest_json(dict(code=code_identity(stage="eval"),
                            training_code=code_identity(stage="train"),
                            configs={p.relative_to(REPO_ROOT).as_posix(): file_digest(p) for p in files}))


def folder(identifier):
    if not identifier.isalnum():
        raise ValueError("Invalid follow-up workflow ID")
    return manifests_dir("experiment0") / "followups" / identifier


def require_gate(identifier=None):
    identifier = identifier or os.environ.get("CREDITPFN_FOLLOWUP_GATE", "")
    if not identifier:
        raise RuntimeError("Experiments 4–7 require CREDITPFN_FOLLOWUP_GATE from completed experiment-0 follow-ups")
    for stage in ("smoke", "capacity"):
        for task in range(4):
            result = json.loads((folder(identifier) / f"{stage}_{task}.json").read_text(encoding="utf-8"))
            if (result.get("passed") is not True or result.get("fingerprint") != fingerprint()
                    or result.get("stage") != stage or result.get("task") != task
                    or result.get("versions") != environment_versions(stage="eval")):
                raise RuntimeError("Missing, failed or stale follow-up validation receipt")
            if file_digest(Path(result["evidence"])) != result["evidence_sha256"]:
                raise RuntimeError("Follow-up evidence changed or was removed")
    return identifier


def prepare():
    from src.train.config import load_train_config
    from src.utils.auxiliary_check import select_tables
    from src.utils.prepare_experiment import prepare as prepare_training
    identifier = uuid.uuid4().hex
    root = folder(identifier)
    root.mkdir(parents=True, exist_ok=False)
    limits = OmegaConf.load("config/experiment0/followups.yaml")
    # Validate the complete large-table design without launching training.
    for path in [Path("config/experiment8/pd.yaml"), *sorted(Path("config/experiment9").glob("*.yaml"))]:
        prepare_training(path)
    for track in ("pd", "lgd"):
        source = load_train_config(config_path=f"config/experiment5/{track}.yaml")
        if list(source.tunable.learning_rates) != list(limits.adapter_rates):
            raise ValueError("Adapter smoke rates must cover the entire experiment-5 grid")
        train, held = select_tables(source, limits)
        for lr_index, lr in enumerate(limits.adapter_rates):
            for arm in ("reference", "resumed"):
                cfg = OmegaConf.create(OmegaConf.to_container(source))
                cfg.run_name = f"cpt_diagscale_debug_{identifier}_lr{lr_index}_{arm}"
                cfg.experiment.output_group = "experiment0"
                cfg.experiment.followup_gate = False
                cfg.corpus.n_splits = 1
                cfg.corpus.train_dataset_ids = [r.dataset_id for r in train]
                cfg.corpus.test_dataset_ids = [r.dataset_id for r in held]
                cfg.train.target_total_steps = limits.target_updates
                cfg.train.max_rows_per_step = limits.row_cap
                cfg.train.epochs = limits.target_updates
                cfg.train.max_epochs_for_step_budget = 20
                cfg.train.deterministic = True
                cfg.train.retention_panel = "smoke"
                cfg.train.trajectory_steps = [0, limits.interrupt_after, limits.target_updates]
                cfg.train.recovery_every_updates = 1
                cfg.train.epoch_eval_subsample_samples = limits.monitor_rows
                cfg.train.epoch_eval_n_estimators = limits.monitor_members
                cfg.train.epoch_eval_n_estimators_tabicl = limits.monitor_members
                cfg.tunable.learning_rates = [float(lr)]
                cfg.checkpoint.trained_dir = "checkpoints/trained/experiment0"
                path = root / f"{track}_{lr_index}_{arm}.yaml"
                OmegaConf.save(cfg, path)
                prepare_training(path, write=True)
    write_json(root / "workflow.json", dict(fingerprint=fingerprint(), id=identifier))
    print(identifier, flush=True)
    return identifier


def _base(track, family):
    from src.utils.paths import resolve_base_checkpoint
    kind = "classifier" if track == "pd" else "regressor"
    name = f"tabpfn-v3-{kind}-v3_default.ckpt" if family == "tabpfn" else f"tabicl-{kind}-v2-20260212.ckpt"
    return resolve_base_checkpoint("checkpoints/" + name)


def attention_cache_probe(attention, *, cache_factory=None):
    """Compare native fresh-KV and cached-KV paths with a nonzero adapter."""
    import torch
    from src.model.diagscale import QueryScale
    if not isinstance(attention.ssmax_layer, QueryScale) or not torch.count_nonzero(attention.ssmax_layer.delta):
        raise AssertionError("Cache check requires a nonzero query adapter")
    if cache_factory is None:
        from tabicl._model.kv_cache import KVCacheEntry
        cache_factory = KVCacheEntry
    ref = attention.in_proj_weight
    dimension = ref.shape[1]
    with torch.inference_mode():
        q = torch.randn(1, 6, dimension, device=ref.device, dtype=ref.dtype)
        kv = torch.randn(1, 10, dimension, device=ref.device, dtype=ref.dtype)
        fresh, key, value = attention(q, kv, kv, need_kv=True)
        cached = attention(q, cached_kv=cache_factory(key=key, value=value))
    if not torch.isfinite(fresh).all() or not torch.allclose(fresh, cached, atol=1e-6, rtol=1e-5):
        raise AssertionError("Fresh and cached attention disagree with nonzero adapters")
    return float((fresh - cached).abs().max())


def adapter_probe(track, family, root):
    """Zero identity, gradient isolation, base immutability and native cached inference."""
    import numpy as np
    import torch
    from src.model.diagscale import inject, adapter_parameters, save_if_adapter, make_estimator
    from src.model.tabpfn_models import _make_tabpfn, _trust_local_checkpoints
    from src.train.tabicl_compat import import_tabicl_sklearn
    from src.train.optimization import step_mean_gradient
    task_type = "classification" if track == "pd" else "regression"
    base = _base(track, family)
    if family == "tabpfn":
        from src.train.model import load_tabpfn_for_training
        model, _, _, _ = load_tabpfn_for_training(base, track=track, device="cuda")
    else:
        from src.train.tabicl_model import load_tabicl_for_training
        model, _ = load_tabicl_for_training(base, track=track, device="cuda")
    original = [(p, p.detach().cpu().clone(), p.requires_grad) for p in model.parameters()]
    params = inject(model, family=family, base_path=f"checkpoints/{base.name}", base_sha256=file_digest(base))
    path = checkpoints_dir("trained", "experiment0", "followups", root.name, f"zero_{track}_{family}_diagscale.ckpt")
    save_if_adapter(model, path)
    # An optimizer containing only adapter tensors must not step or clip a base
    # parameter, even when TabICL retains its original autograd flags.
    # Use a visible perturbation for this loader check: a sub-ULP change in a
    # low-precision inference kernel would not prove that the loader ignored it.
    optimizer = torch.optim.AdamW(params, lr=.01, weight_decay=.01)
    for p in params:
        p.grad = torch.ones_like(p)
    for p, _, enabled in original:
        if p.requires_grad:
            p.grad = torch.ones_like(p)
        if family == "tabicl" and p.requires_grad != enabled:
            raise AssertionError("TabICL autograd flag parity changed")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    _, finite = step_mean_gradient(model, optimizer, scaler, microbatches=1, max_norm=1.)
    if not finite or not all(p.grad is None for p in model.parameters()):
        raise AssertionError("Gradient flush failed")
    if any(not torch.equal(p.detach().cpu(), old) for p, old, _ in original):
        raise AssertionError("An original parameter changed")
    changed_path = path.with_name(f"changed_{track}_{family}_diagscale.ckpt")
    save_if_adapter(model, changed_path)
    cache_errors = []
    if family == "tabicl":
        model.eval()
        cache_errors = [attention_cache_probe(module) for module in model.modules()
                        if type(module).__name__ == "MultiheadAttention"]
        if not cache_errors:
            raise AssertionError("No native attention caches were checked")
    count = sum(p.numel() for p in adapter_parameters(model))
    del original, model, optimizer, params
    torch.cuda.empty_cache()
    rng = np.random.default_rng(912)
    X = rng.normal(size=(128, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) if track == "pd" else X[:, 0] + .1 * X[:, 1]
    kwargs = dict(device="cuda", n_estimators=2, random_state=19)
    if family == "tabpfn":
        original_est = _make_tabpfn(task_type, base, **kwargs)
    else:
        cls, reg = import_tabicl_sklearn()
        original_est = (cls if track == "pd" else reg)(model_path=str(base), allow_auto_download=False,
                                                       use_amp=False, use_fa3=False, **kwargs)
    predictions = []
    for est in (original_est, make_estimator(task_type, path, family=family, **kwargs),
                make_estimator(task_type, changed_path, family=family, **kwargs)):
        with _trust_local_checkpoints():
            est.fit(X[:96], y[:96])
            predict = est.predict_proba if track == "pd" else est.predict
            first = np.asarray(predict(X[96:]))
            repeat = np.asarray(predict(X[96:]))
            chunks = np.concatenate([np.asarray(predict(X[96:112])), np.asarray(predict(X[112:]))])
        if not np.isfinite(first).all() or not np.allclose(first, repeat, atol=1e-6, rtol=1e-5):
            raise AssertionError("Cached/repeated inference is not stable")
        if not np.allclose(first, chunks, atol=1e-5, rtol=1e-4):
            raise AssertionError("Query chunking changes predictions beyond tolerance")
        predictions.append(first)
    zero_error = float(np.max(np.abs(predictions[0]-predictions[1])))
    if not np.allclose(predictions[0], predictions[1], atol=1e-5, rtol=1e-5):
        raise AssertionError(f"Zero adapter changes native predictions: {zero_error}")
    changed = float(np.max(np.abs(predictions[1]-predictions[2])))
    if changed == 0:
        raise AssertionError("Native loader ignored nonzero adapters")
    return dict(adapter_parameters=count, zero_max_error=zero_error, nonzero_max_effect=changed,
                base_unchanged=True, cached_inference=True, native_attention_cache_errors=cache_errors)


def smoke(identifier, track, trial):
    import torch
    from src.utils.audit_experiment import audit
    from src.utils.recovery_check import compare_trajectories
    from src.utils.auxiliary_check import trajectory
    root = folder(identifier)
    family = "tabpfn" if trial == 0 else "tabicl"
    probe = adapter_probe(track, family, root)
    context_check = context_probe(track, family)
    outcomes = []
    limits = OmegaConf.load("config/experiment0/followups.yaml")
    for index in range(len(limits.adapter_rates)):
        rows, curves = {}, {}
        for arm in ("reference", "resumed"):
            path = root / f"{track}_{index}_{arm}.yaml"
            command = [sys.executable, "-u", "scripts/train_pipeline.py", "--config", str(path),
                       "--split-index", "0", "--trial-index", str(trial)]
            if os.environ.get("CREDITPFN_ACTIVE_LOG"):
                command += ["--log-path", os.environ["CREDITPFN_ACTIVE_LOG"]]
            env = dict(os.environ, CREDITPFN_SEGMENT_SECONDS="0", PYTHONHASHSEED="42")
            for key in ("CREDITPFN_STOP_AFTER_UPDATES", "CREDITPFN_FLOW_ID", "CREDITPFN_FLOW_PHASE"):
                env.pop(key, None)
            if arm == "resumed":
                rc = subprocess.run(command, env=dict(env, CREDITPFN_STOP_AFTER_UPDATES=str(limits.interrupt_after))).returncode
                if rc != 75:
                    raise RuntimeError(f"Expected recovery boundary exit 75, observed {rc}")
            subprocess.run(command, env=env, check=True)
            report = audit(path)
            row = report["trials"][trial]
            if row["status"] != "OK" or any(p.startswith(row["trial"]+":") for p in report["problems"]):
                raise RuntimeError("Adapter training/output audit failed")
            rows[arm] = torch.load(row["path"], map_location="cpu", weights_only=False)["adapter_state"]
            curves[arm] = trajectory(row, track)
        equal = rows["reference"].keys() == rows["resumed"].keys() and all(
            torch.allclose(v, rows["resumed"][k], atol=1e-7, rtol=1e-6) for k, v in rows["reference"].items())
        moved = any(bool(torch.count_nonzero(v)) for v in rows["reference"].values())
        monitors = compare_trajectories(curves["reference"], curves["resumed"],
                                       milestones=[0, int(limits.interrupt_after), int(limits.target_updates)])
        if not equal or not moved or not monitors["passed"]:
            raise RuntimeError("Adapter update/recovery/monitor parity check failed")
        outcomes.append(dict(lr=float(limits.adapter_rates[index]), recovery_equal=equal, moved=moved, monitors=monitors))
    return dict(probe=probe, context_check=context_check, rates=outcomes)


def context_probe(track, family):
    """Real original/CPT loaders, five folds, nested contexts and complete query folds."""
    import io
    import numpy as np
    import pandas as pd
    from src.eval.context_curve import policy, reference_checkpoint, fold_indices, score_cell
    from src.eval.dataset_loader import ProcessedDataset
    cfg = policy()
    cfg['inference_views'] = {'tabpfn': 2, 'tabicl': 2}
    pair, _ = reference_checkpoint(track, family, 0, cfg)
    rng = np.random.default_rng(518)
    X = rng.normal(size=(160, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) if track == 'pd' else X[:, 0]
    kind = 'classification' if track == 'pd' else 'regression'
    data = ProcessedDataset(pd.DataFrame(X), y, [], kind, 'synthetic_context', track)
    observed, cells = [], []
    for fold in range(5):
        order, val, test = fold_indices(y, kind, fold, cfg)
        observed.extend(test.tolist())
        for size in (16, 32):
            for state in ('base', 'cpt'):
                metrics, blob = score_cell(data, order[:size], val, test, pair[state], family, cfg)
                with np.load(io.BytesIO(blob)) as saved:
                    if not np.array_equal(saved['test'], test) or len(saved['prediction']) != len(test):
                        raise AssertionError('Context curve dropped or changed query rows')
                cells.append(dict(fold=fold, context_rows=size, state=state, test_rows=len(test), **metrics))
    if sorted(observed) != list(range(len(y))):
        raise AssertionError('Outer test folds do not cover each row exactly once')
    return dict(cells=cells, all_test_rows=True, reference_cpt_sha256=pair['cpt_sha256'])


def capacity(identifier, track, trial):
    """Production-shape checks, separated from cheap functional tests."""
    import numpy as np
    import pandas as pd
    import torch
    from src.eval.context_curve import policy, score_cell
    from src.eval.dataset_loader import ProcessedDataset
    from src.utils.prepare_experiment import prepare as prepare_training
    from src.utils.audit_experiment import audit
    from src.utils.auxiliary_check import trajectory
    from src.train.config import load_train_config
    family = "tabpfn" if trial == 0 else "tabicl"
    root = folder(identifier)
    limits = OmegaConf.load("config/experiment0/followups.yaml")
    # Use the entire eligible training corpus, exercising its real feature widths.
    cfg = load_train_config(config_path=f"config/experiment5/{track}.yaml")
    cfg.experiment.output_group = "experiment0"
    cfg.experiment.followup_gate = False
    cfg.corpus.n_splits = 1
    cfg.run_name = f"cpt_diagscale_debug_{identifier}_{family}_capacity"
    cfg.train.target_total_steps = 16
    cfg.train.trajectory_steps = [0, 16]
    # Keep the adapter phase's declared FP32 cap; the separate full-weight
    # episode-size controls below exercise the 26k BF16 recipe.
    cfg.train.retention_panel = "smoke"
    cfg.train.epoch_eval_n_estimators = 2
    cfg.train.epoch_eval_n_estimators_tabicl = 2
    cfg.tunable.learning_rates = [0.001]
    cfg.checkpoint.trained_dir = "checkpoints/trained/experiment0"
    path = root / f"capacity_{track}_{trial}.yaml"
    OmegaConf.save(cfg, path)
    prepare_training(path, write=True)
    subprocess.run([sys.executable, "-u", "scripts/train_pipeline.py", "--config", str(path),
                    "--trial-index", str(trial), "--split-index", "0",
                    "--log-path", os.environ["CREDITPFN_ACTIVE_LOG"]], check=True)
    audited = audit(path)
    selected = audited["trials"][trial]
    if selected["status"] != "OK" or any(p.startswith(selected["trial"]+":") for p in audited["problems"]):
        raise RuntimeError("Production-size adapter probe failed")
    large_reports = []
    if track == "pd":
        # Test all four distinct policies through real feature widths. Each
        # 16-update probe visits every table in its first training partition.
        cases = [("config/experiment8/pd.yaml", 2*trial),
                 ("config/experiment8/pd.yaml", 2*trial+1),
                 ("config/experiment9/pd_rows2048.yaml", trial),
                 ("config/experiment9/pd_rows26000.yaml", trial)]
        for case, (source_path, trial_index) in enumerate(cases):
            large = load_train_config(config_path=source_path)
            cap = int(large.train.max_rows_per_step)
            large.experiment.output_group = "experiment0"
            large.experiment.followup_gate = False
            large.corpus.n_splits = 1
            large.run_name = f"cpt_large_debug_{identifier}_{family}_case{case}_capacity"
            large.train.target_total_steps = 16
            large.train.trajectory_steps = [0, 16]
            large.train.retention_panel = "smoke"
            large.train.epoch_eval_n_estimators = 2
            large.train.epoch_eval_n_estimators_tabicl = 2
            large.checkpoint.trained_dir = "checkpoints/trained/experiment0"
            large_path = root / f"size_case{case}_{trial}.yaml"
            OmegaConf.save(large, large_path)
            prepare_training(large_path, write=True)
            subprocess.run([sys.executable, "-u", "scripts/train_pipeline.py", "--config", str(large_path),
                            "--trial-index", str(trial_index), "--split-index", "0", "--log-path",
                            os.environ["CREDITPFN_ACTIVE_LOG"]], check=True)
            report = audit(large_path)
            row = report["trials"][trial_index]
            if row["status"] != "OK" or any(p.startswith(row["trial"]+":") for p in report["problems"]):
                raise RuntimeError("Large-table production-loader probe failed")
            if case != 0:
                # Full episodes must actually reach the model, not just appear
                # in a YAML. No skips are permitted by the experiment-0 audit.
                frame = trajectory(row, track)
                if int(frame.iloc[-1].processed_rows) != 16 * cap:
                    raise RuntimeError("Size-study probe did not process the declared episode rows")
            large_reports.append(row)
    curve_cfg = policy()
    rng = np.random.default_rng(991)
    maximum = int(limits.context_probe_rows[-1])
    query_rows = int(curve_cfg['query_batch_rows'])
    X = rng.normal(size=(maximum+2*query_rows, 64)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) if track == "pd" else X[:, 0]
    data = ProcessedDataset(X=pd.DataFrame(X), y=y, categorical_columns=[],
                            task_type="classification" if track == "pd" else "regression",
                            dataset_id="synthetic_capacity", track=track)
    costs = []
    for size in limits.context_probe_rows:
        print(f"Context capacity: {track}/{family}, {size} rows, "
              f"{curve_cfg['inference_views'][family]} inference views", flush=True)
        torch.cuda.reset_peak_memory_stats()
        metrics, _ = score_cell(data, np.arange(size), np.arange(maximum, maximum+query_rows),
                                np.arange(maximum+query_rows, maximum+2*query_rows),
                                _base(track, family), family, curve_cfg)
        costs.append(dict(context_rows=int(size), seconds=metrics["elapsed_seconds"],
                           peak_bytes=torch.cuda.max_memory_allocated(), features=64,
                           query_batch_rows=query_rows))
        print(json.dumps(costs[-1]), flush=True)
    return dict(training_audit=selected, large_table_audits=large_reports, context_capacity=costs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "smoke", "capacity", "inspect"])
    parser.add_argument("--id")
    parser.add_argument("--task", type=int, choices=range(4))
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare()
        return 0
    if not args.id:
        parser.error("--id required")
    if args.action == "inspect":
        require_gate(args.id)
        print("All eight follow-up receipts passed for this source/environment.")
        return 0
    if args.task is None:
        parser.error("--task required")
    root = folder(args.id)
    if json.loads((root / "workflow.json").read_text())["fingerprint"] != fingerprint():
        raise RuntimeError("Workflow source/config changed; do not mix implementations")
    if args.action == "capacity":
        for task in range(4):
            report = json.loads((root / f"smoke_{task}.json").read_text())
            if not report.get("passed") or report.get("fingerprint") != fingerprint():
                raise RuntimeError("All four smoke tasks must pass before capacity checks")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This is a VSC GPU check, not a local training command")
    track, trial = ("pd" if args.task < 2 else "lgd"), args.task % 2
    receipt = root / f"{args.action}_{args.task}.json"
    if receipt.exists():
        raise RuntimeError("Task already has a receipt; inspect it or start a fresh workflow")
    result = dict(stage=args.action, task=args.task, passed=False, fingerprint=fingerprint(),
                  versions=environment_versions(stage="eval"))
    try:
        details = (smoke if args.action == "smoke" else capacity)(args.id, track, trial)
        detail_path = results_dir(experiment="experiment0") / "followups" / args.id / f"{args.action}_{args.task}.json"
        write_json(detail_path, details)
        result.update(passed=True, evidence=str(detail_path), evidence_sha256=file_digest(detail_path))
    finally:
        write_json(receipt, result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
