"""Isolated experiment-0 LoRA gates. This utility never submits production work.

Preparation hashes production plans, inputs, bases, source and environment. Four
smoke tasks test both model families/tasks; four capacity tasks cover all dataset
partitions at the production row/feature limits. An inspection passes only when
all eight immutable receipts and their underlying evidence remain intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from omegaconf import OmegaConf

from src.utils.atomic import write_json
from src.utils.experiment import code_identity, digest_json, environment_versions, file_digest
from src.utils.paths import REPO_ROOT, OUTPUT_DIR_NAME, outputs_dir, resolve_staging_path

SCHEMA = 1
CASES = (("pd", "tabpfn"), ("pd", "tabicl"), ("lgd", "tabpfn"), ("lgd", "tabicl"))


def validate_id(identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9]{1,80}", identifier):
        raise ValueError("LoRA workflow ID must contain 1–80 ASCII letters/digits")
    return identifier


def folder(identifier):
    return outputs_dir() / "experiment0" / "lora" / validate_id(identifier)


def evidence_folder(identifier):
    return resolve_staging_path(Path(OUTPUT_DIR_NAME, "experiment0", "lora", validate_id(identifier)))


@contextmanager
def workflow_scope(identifier):
    """Scope every existing pipeline writer; restore the caller's routing afterwards."""
    validate_id(identifier)
    saved = {k: os.environ.get(k) for k in ("CREDITPFN_LORA_WORKFLOW", "CREDITPFN_EXPERIMENT")}
    os.environ.update(CREDITPFN_LORA_WORKFLOW=identifier, CREDITPFN_EXPERIMENT="experiment0")
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def policy():
    limits = OmegaConf.load(REPO_ROOT / "config/experiment0/lora.yaml")
    if not 0 < limits.interrupt_after < limits.target_updates or limits.row_cap < 8:
        raise ValueError("Invalid LoRA smoke budget or row cap")
    tolerances = (limits.prediction_atol, limits.prediction_rtol, limits.regression_chunk_atol_std)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in tolerances):
        raise ValueError("LoRA prediction tolerances must be finite and positive")
    return limits


def fingerprint():
    files = [REPO_ROOT / "config" / name for name in (
        "experiment0/lora.yaml", "experiment6/pd.yaml", "experiment6/lgd.yaml",
        "train.yaml", "eval.yaml", "data.yaml", "retention.yaml")]
    files += [REPO_ROOT / "scripts/slurm" / name for name in (
        "run_experiment0_lora.sh", "lora_prepare.slurm", "lora_gpu.slurm",
        "run_experiment6.sh", "run_benchmark.sh", "_benchmark_dispatch.sh",
        "benchmark_foundation.slurm", "benchmark_classical.slurm")]
    return digest_json({"train": code_identity(stage="train"), "eval": code_identity(stage="eval"),
                        "files": {p.relative_to(REPO_ROOT).as_posix(): file_digest(p) for p in files}})


def _binding(path):
    path = Path(path).resolve()
    digest = file_digest(path)
    stat = path.stat()
    return {"path": str(path), "sha256": digest, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verify_bindings(bindings, *, full_hash=True):
    if not bindings:
        raise RuntimeError("LoRA workflow has no bound evidence")
    for item in bindings:
        path = Path(item["path"])
        stat = path.stat()
        if (("size" in item and (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]))
                or ((full_hash or "size" not in item) and file_digest(path) != item["sha256"])):
            raise RuntimeError(f"LoRA workflow input/evidence changed: {item['path']}")


def _workflow(identifier, *, full_hash=False):
    path = folder(identifier) / "workflow.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema") != SCHEMA or payload.get("id") != identifier
            or payload.get("kind") != "creditpfn_lora" or payload.get("prepared") is not True):
        raise RuntimeError("LoRA preparation is missing, incomplete or belongs to another workflow")
    if payload.get("fingerprint") != fingerprint() or payload.get("versions") != environment_versions(stage="eval"):
        raise RuntimeError("LoRA workflow source/config/environment changed; prepare a fresh workflow")
    verify_bindings(payload["inputs"], full_hash=full_hash)
    verify_bindings(payload["files"])
    return payload


def _receipt(identifier, stage, task, workflow, *, full_hash=False):
    path = folder(identifier) / "receipts" / f"{stage}_{task}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if any(result.get(key) != value for key, value in {
        "schema": SCHEMA, "id": identifier, "stage": stage, "task": task, "passed": True,
        "workflow_sha256": digest_json(workflow), "fingerprint": workflow["fingerprint"],
        "versions": workflow["versions"],
    }.items()):
        raise RuntimeError("Missing, failed, stale or mismatched LoRA validation receipt")
    expected = evidence_folder(identifier) / "results" / f"{stage}_{task}.json"
    if Path(result["evidence"]).resolve() != expected.resolve():
        raise RuntimeError("LoRA receipt points outside its workflow evidence")
    verify_bindings([{"path": result["evidence"], "sha256": result["evidence_sha256"]}])
    details = json.loads(expected.read_text(encoding="utf-8"))
    verify_bindings(details["artifacts"], full_hash=full_hash)
    return result


def _complete_gate(identifier, *, full_hash=False):
    workflow = _workflow(validate_id(identifier), full_hash=full_hash)
    receipts = {}
    for stage in ("smoke", "capacity"):
        for task in range(len(CASES)):
            _receipt(identifier, stage, task, workflow, full_hash=full_hash)
            path = folder(identifier) / "receipts" / f"{stage}_{task}.json"
            receipts[path.name] = file_digest(path)
    return workflow, receipts


def inspect_gate(identifier):
    """CPU audit: hash large artifacts once, then publish a bound inspection receipt."""
    workflow, receipts = _complete_gate(identifier, full_hash=True)
    write_json(folder(identifier) / "inspection.json", {
        "schema": SCHEMA, "id": identifier, "passed": True,
        "workflow_sha256": digest_json(workflow), "receipts": receipts,
    })
    return identifier


def require_gate(identifier=None):
    identifier = identifier or os.environ.get("CREDITPFN_LORA_GATE", "")
    if not identifier:
        raise RuntimeError("Experiment 6 requires CREDITPFN_LORA_GATE from completed experiment-0 LoRA checks")
    workflow, receipts = _complete_gate(identifier)
    inspected = json.loads((folder(identifier) / "inspection.json").read_text(encoding="utf-8"))
    if inspected != {"schema": SCHEMA, "id": identifier, "passed": True,
                     "workflow_sha256": digest_json(workflow), "receipts": receipts}:
        raise RuntimeError("LoRA independent inspection is missing, stale or failed")
    return identifier


def _source(track):
    from src.train.config import load_train_config
    return load_train_config(config_path=str(REPO_ROOT / f"config/experiment6/{track}.yaml"))


def validate_evaluation(train_cfg, eval_cfg):
    """Recheck the accepted gate and numerical benchmark settings inside each job."""
    require_gate()
    from src.train.size_studies import evaluation_protocol
    expected = _source(str(train_cfg.track))
    if evaluation_protocol(expected) != evaluation_protocol(train_cfg, eval_cfg):
        raise RuntimeError("LoRA benchmark settings differ from the validated experiment-8 protocol")


def _base(track, family):
    from src.utils.paths import resolve_base_checkpoint
    kind = "classifier" if track == "pd" else "regressor"
    name = f"tabpfn-v3-{kind}-v3_default.ckpt" if family == "tabpfn" else f"tabicl-{kind}-v2-20260212.ckpt"
    return resolve_base_checkpoint("checkpoints/" + name)


def make_config(source, identifier, family, *, rate, arm, limits, train=None, held=None):
    """Keep production numerical choices; shrink only the isolated smoke workload."""
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    track = str(cfg.track)
    cfg.run_name = f"cpt_lora_debug_{identifier}_{track}_{family}_{arm}"
    cfg.experiment.output_group = "experiment0"
    cfg.experiment.lora_gate = False
    cfg.experiment.followup_gate = False
    cfg.tunable["classifier_base_paths" if track == "pd" else "regressor_base_paths"] = [str(_base(track, family))]
    cfg.tunable.learning_rates = [float(rate)]
    cfg.checkpoint.trained_dir = f"checkpoints/trained/experiment0/lora/{validate_id(identifier)}"
    cfg.train.retention_panel = "smoke"
    cfg.train.epoch_eval_subsample_samples = limits.monitor_rows
    cfg.train.epoch_eval_n_estimators = limits.monitor_members
    cfg.train.epoch_eval_n_estimators_tabicl = limits.monitor_members
    cfg.train.recovery_every_updates = 1
    if train is not None:
        cfg.corpus.n_splits = 1
        cfg.corpus.train_dataset_ids = [r.dataset_id for r in train]
        cfg.corpus.test_dataset_ids = [r.dataset_id for r in held]
        cfg.train.deterministic = True
        cfg.train.max_rows_per_step = limits.row_cap
        cfg.train.target_total_steps = limits.target_updates
        cfg.train.trajectory_steps = [0, limits.interrupt_after, limits.target_updates]
        cfg.train.epochs = limits.target_updates
        cfg.train.max_epochs_for_step_budget = 20
    else:
        # All four original partitions, all their real feature widths and measured
        # row/cell limits; no artificial enlargement of a production limit.
        cfg.train.target_total_steps = limits.capacity_updates
        cfg.train.trajectory_steps = [0, limits.capacity_updates]
    return cfg


def prepare(identifier=None):
    from src.utils.prepare_experiment import prepare as prepare_training
    from src.utils.auxiliary_check import select_tables
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    from src.data.retention import load_refs
    identifier = validate_id(identifier or uuid.uuid4().hex)
    root = folder(identifier)
    marker = root / "workflow.json"
    if marker.exists():
        raise RuntimeError("LoRA workflow already exists; inspect it or choose a new ID")
    write_json(marker, {"schema": SCHEMA, "id": identifier, "kind": "creditpfn_lora", "prepared": False}, exclusive=True)
    limits = policy()
    inputs, files = {}, []
    # Production plans live under public experiment 6 (recorded campaign 8). Preparation records but never trains them.
    with workflow_scope(identifier):
        for track in ("pd", "lgd"):
            source = _source(track)
            if list(source.tunable.learning_rates) != list(limits.learning_rates):
                raise ValueError("LoRA smoke rates must cover the entire production grid")
            production = prepare_training(REPO_ROOT / f"config/experiment6/{track}.yaml", write=True)
            files.append(_binding(production["plan"]))
            for index in range(int(source.corpus.n_splits)):
                split = split_from_cfg(apply_split_index(OmegaConf.create(OmegaConf.to_container(source)), index))
                for ref in [*split.train, *split.test, *load_refs(str(source.train.retention_panel), track)]:
                    inputs[str(ref.processed_csv)] = _binding(ref.processed_csv)
            train, held = select_tables(source, limits)
            for family in ("tabpfn", "tabicl"):
                base = _base(track, family)
                inputs[str(base)] = _binding(base)
                cases = [(f"lr{index}_{arm}", rate, train, held)
                         for index, rate in enumerate(limits.learning_rates) for arm in ("reference", "resumed")]
                cases.append(("capacity", max(limits.learning_rates), None, None))
                for arm, rate, selected_train, selected_held in cases:
                    cfg = make_config(source, identifier, family, rate=rate, arm=arm, limits=limits,
                                      train=selected_train, held=selected_held)
                    path = root / "configs" / f"{track}_{family}_{arm}.yaml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    OmegaConf.save(cfg, path)
                    files.append(_binding(path))
                    planned = prepare_training(path, write=True)
                    files.append(_binding(planned["plan"]))
    payload = {"schema": SCHEMA, "id": identifier, "kind": "creditpfn_lora", "prepared": True,
               "fingerprint": fingerprint(), "versions": environment_versions(stage="eval"),
               "inputs": list(inputs.values()), "files": files}
    write_json(marker, payload)
    print(f"Prepared LoRA workflow {identifier}; no GPU or production work has been submitted.", flush=True)
    return identifier


def _run_training(path, *, split=0, stop=None):
    identifier = validate_id(os.environ.get("CREDITPFN_LORA_WORKFLOW", ""))
    evidence = evidence_folder(identifier) / "diagnostics" / f"gradients_{Path(path).stem}_{split}_{uuid.uuid4().hex}.json"
    command = [sys.executable, "-u", "-m", "src.utils.lora_training_probe", "--report", str(evidence),
               "--", "--config", str(path),
               "--split-index", str(split), "--trial-index", "0"]
    if os.environ.get("CREDITPFN_ACTIVE_LOG"):
        command += ["--log-path", os.environ["CREDITPFN_ACTIVE_LOG"]]
    env = dict(os.environ, CREDITPFN_SEGMENT_SECONDS="0", PYTHONHASHSEED="42")
    for key in ("CREDITPFN_STOP_AFTER_UPDATES", "CREDITPFN_STOP_REQUESTED", "CREDITPFN_FLOW_ID", "CREDITPFN_FLOW_PHASE"):
        env.pop(key, None)
    if stop is not None:
        env["CREDITPFN_STOP_AFTER_UPDATES"] = str(stop)
    result = subprocess.run(command, env=env, check=False)
    expected = 75 if stop is not None else 0
    if result.returncode != expected:
        raise RuntimeError(f"LoRA pipeline returned {result.returncode}, expected {expected}")
    if stop is None:
        report = json.loads(evidence.read_text(encoding="utf-8"))
        if report.get("kind") != "lora_training_gradient_audit" or report.get("passed") is not True:
            raise AssertionError("Missing or failed per-projection LoRA training gradient audit")
    return evidence


def _audited(path, *, split=0):
    from src.utils.audit_experiment import audit
    result = audit(path)
    row = result["trials"][split]
    if row["status"] != "OK" or any(p.startswith(row["trial"] + ":") for p in result["problems"]):
        raise RuntimeError("LoRA training/checkpoint/trajectory audit failed")
    return row


def _curve(row, track):
    import pandas as pd
    from src.utils.paths import training_dir
    path = training_dir(track, row["trial"] + ".trajectory.csv", experiment="experiment0")
    return pd.read_csv(path), path


def _load_native(path, track, family):
    if family == "tabpfn":
        from src.train.model import load_tabpfn_for_training
        return load_tabpfn_for_training(path, track=track, device="cuda")
    from src.train.tabicl_model import load_tabicl_for_training
    model, cfg = load_tabicl_for_training(path, track=track, device="cuda")
    return model, None, cfg, None


def _save_native(model, info, path, family):
    criterion, cfg, inference = info
    if family == "tabpfn":
        from src.train.model import save_finetuned
        save_finetuned(model, cfg, path, criterion=criterion, inference_config=inference)
    else:
        from src.train.tabicl_model import save_finetuned_tabicl
        save_finetuned_tabicl(model, cfg, path)


def _attention(model, family):
    expected = "ICLAttention" if family == "tabpfn" else "MultiheadAttention"
    for module in model.modules():
        if type(module).__name__ == expected:
            return module
    raise AssertionError(f"No native {expected} found for the LoRA attention probe")


def _attention_forward(attention, x, family, *, cached=False):
    if family == "tabpfn":
        fresh, cache = attention(x, single_eval_pos=10, return_kv=True)
        if not cached:
            return fresh
        hit, _ = attention(x[:, 10:], single_eval_pos=0, cached_kv=cache)
        return fresh[:, 10:], hit
    from tabicl._model.attention import flash_attn3_toggle
    from tabicl._model.kv_cache import KVCacheEntry
    with flash_attn3_toggle(False):
        fresh, key, value = attention(x[:, 10:], x[:, :10], x[:, :10], need_kv=True)
        if not cached:
            return fresh
        return fresh, attention(x[:, 10:], cached_kv=KVCacheEntry(key=key, value=value))


def _estimator(path, track, family, *, precision="float32"):
    import torch
    from src.model.tabpfn_models import _make_tabpfn
    kwargs = dict(device="cuda", n_estimators=2, random_state=19)
    if family == "tabpfn":
        # Explicit checkpoint identity, never the package's newest model default.
        if precision == "float32":
            kwargs["inference_precision"] = torch.float32
        elif precision != "automatic":
            raise ValueError("Unknown LoRA probe precision policy")
        return _make_tabpfn("classification" if track == "pd" else "regression", path, **kwargs)
    from src.train.tabicl_compat import import_tabicl_sklearn
    cls, reg = import_tabicl_sklearn()
    return (cls if track == "pd" else reg)(model_path=str(path), allow_auto_download=False,
                                          use_amp=False, use_fa3=False, **kwargs)


def prediction_probe(paths, track, family, *, precision="float32", check_chunks=True,
                     strict=True, diagnostics=None):
    """Native inference, repeated calls and query chunks with exact failure evidence."""
    import numpy as np
    import torch
    from src.model.tabpfn_models import _trust_local_checkpoints
    from src.utils.lora_diagnostics import compare_predictions, compare_query_chunks
    rng = np.random.default_rng(912)
    limits = policy()
    tolerance = dict(atol=float(limits.prediction_atol), rtol=float(limits.prediction_rtol))
    X = rng.normal(size=(128, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) if track == "pd" else X[:, 0] + .1 * X[:, 1]
    predictions = []
    for path in paths:
        est = _estimator(path, track, family, precision=precision)
        with _trust_local_checkpoints():
            est.fit(X[:96], y[:96])
            predict = est.predict_proba if track == "pd" else est.predict
            first = np.asarray(predict(X[96:]))
            repeat = np.asarray(predict(X[96:]))
            chunks = (np.concatenate([np.asarray(predict(X[96:112])), np.asarray(predict(X[112:]))])
                      if check_chunks else first)
        record = dict(checkpoint=Path(path).name, track=track, family=family, precision=precision,
                      fit_mode=str(getattr(est, "fit_mode", "unspecified")),
                      resolved_autocast=(bool(est.use_autocast_) if hasattr(est, "use_autocast_") else None),
                      forced_inference_dtype=str(getattr(est, "forced_inference_dtype_", None)),
                      tf32=dict(matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                                cudnn=bool(torch.backends.cudnn.allow_tf32)),
                      repeat=compare_predictions(first, repeat, **tolerance),
                      query_chunks=(compare_query_chunks(first, chunks, track=track, context_y=y[:96],
                                    regression_atol_std=limits.regression_chunk_atol_std, **tolerance)
                                    if check_chunks else None),
                      query_chunks_pointwise=(compare_predictions(first, chunks, **tolerance)
                                               if check_chunks else None))
        if diagnostics is not None:
            diagnostics.append(record)
        print("LoRA prediction comparison: " + json.dumps(record, allow_nan=False), flush=True)
        failed = [name for name in ("repeat", "query_chunks")
                  if record[name] is not None and not record[name]["passed"]]
        if strict and failed:
            raise AssertionError(f"LoRA native prediction parity failed: checkpoint={Path(path).name} "
                                 f"track={track} precision={precision} checks={','.join(failed)}; "
                                 "see preceding comparison for finite counts and error magnitudes")
        predictions.append(first)
        del est
    return predictions


def adapter_probe(identifier, track, family):
    """Real released model: zero identity, real gradients, base freeze, merge and KV caches."""
    import numpy as np
    import torch
    from src.model.lora import inject, adapter_parameters, metadata
    from src.utils.paths import checkpoints_dir
    base = _base(track, family)
    limits = policy()
    tolerance = dict(atol=float(limits.prediction_atol), rtol=float(limits.prediction_rtol))
    model, *info = _load_native(base, track, family)
    model.eval()
    original = [(p, p.detach().cpu().clone()) for p in model.parameters()]
    # Autograd flags can alter native attention dispatch even under no_grad.
    # Compare zero LoRA with equally frozen weights; native estimator identity
    # below separately covers the original released inference policy.
    for p, _ in original:
        p.requires_grad_(False)
    attention = _attention(model, family)
    dimension = attention.q_projection.in_features if family == "tabpfn" else attention.in_proj_weight.shape[1]
    torch.manual_seed(518)
    x = torch.randn(1, 16, dimension, device="cuda")
    with torch.no_grad():
        before = _attention_forward(attention, x, family).detach()
    params = inject(model, family=family, base_path="checkpoints/" + base.name,
                    base_sha256=file_digest(base), rank=16, alpha=32, dropout=0,
                    target_modules="attention_qv")
    with torch.no_grad():
        zero = _attention_forward(attention, x, family)
    if not torch.allclose(before, zero, atol=1e-6, rtol=1e-5):
        raise AssertionError("Zero-initialized LoRA changed native attention")
    after = _attention_forward(attention, x, family)
    root = checkpoints_dir("trained", "experiment0", "lora", identifier, "probes")
    zero_path = root / f"zero_{track}_{base.name}"
    changed_path = root / f"changed_{track}_{base.name}"
    _save_native(model, info, zero_path, family)
    after.square().mean().backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        raise AssertionError("LoRA produced a nonfinite adapter gradient")
    finite_nonzero = [p for p in params if p.grad is not None and torch.isfinite(p.grad).all()
                      and torch.count_nonzero(p.grad)]
    if len(finite_nonzero) < 2 or any(p.grad is not None for p, _ in original):
        raise AssertionError("LoRA Q/V gradients missing/nonfinite or frozen weights received gradients")
    optimizer = torch.optim.AdamW(params, lr=.01, weight_decay=0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not torch.equal(p.detach().cpu(), old) for p, old in original):
        raise AssertionError("LoRA optimizer changed a pretrained parameter")
    with torch.no_grad():
        changed = _attention_forward(attention, x, family).detach()
        fresh, cached = _attention_forward(attention, x, family, cached=True)
    if torch.equal(before, changed) or not torch.allclose(fresh, cached, **tolerance):
        raise AssertionError("Nonzero LoRA is ineffective or incompatible with native KV caching")
    _save_native(model, info, changed_path, family)
    count = sum(p.numel() for p in adapter_parameters(model))
    meta = metadata(model)
    del original, optimizer, params, finite_nonzero, model, attention, info, after
    torch.cuda.empty_cache()
    merged, *merged_info = _load_native(changed_path, track, family)
    merged.eval()
    with torch.no_grad():
        restored = _attention_forward(_attention(merged, family), x, family)
    if not torch.allclose(changed, restored, **tolerance):
        raise AssertionError("Merged/reloaded native attention disagrees with active LoRA")
    merged_error = float((changed - restored).abs().max())
    del merged, merged_info
    torch.cuda.empty_cache()
    predictions = prediction_probe([base, zero_path, changed_path], track, family)
    if not np.allclose(predictions[0], predictions[1], **tolerance):
        raise AssertionError("Zero LoRA native checkpoint predictions differ from the original")
    effect = float(np.max(np.abs(predictions[1] - predictions[2])))
    if effect == 0:
        raise AssertionError("Native loader ignored the nonzero LoRA update")
    automatic_error = None
    if family == "tabpfn":
        # Test native zero-export identity under the exact production automatic
        # precision policy as well. Do not equate predictions across precisions;
        # low-precision query chunk sensitivity is a separate scientific issue.
        automatic = prediction_probe([base, zero_path], track, family, precision="automatic", check_chunks=False)
        automatic_error = float(np.max(np.abs(automatic[0] - automatic[1])))
        if not np.allclose(automatic[0], automatic[1], **tolerance):
            raise AssertionError("Zero LoRA export changes native automatic-precision predictions")
    return {"zero_identity": True, "base_unchanged": True, "gradient_isolation": True,
            "native_cache_equal": True, "merged_max_error": merged_error,
            "automatic_zero_export_max_error": automatic_error,
            "prediction_effect": effect, "trainable_parameters": count, "adapter": meta,
            "artifacts": [_binding(zero_path), _binding(changed_path)]}


def weight_scope(before, after, targets, family):
    """Native final weights may change Q/V slices only, in all three model stages."""
    import torch
    if before.keys() != after.keys():
        raise AssertionError("Native LoRA export changed the checkpoint key schema")
    prefix = "model." if family == "tabpfn" else ""
    mapping = {prefix + target["module"] + "." + target["weight"]: target for target in targets}
    if not set(mapping).issubset(before):
        raise AssertionError("LoRA target metadata does not match native checkpoint tensors")
    changed_stages, changed_targets = set(), []
    for name, original in before.items():
        trained = after[name]
        if original.shape != trained.shape or not torch.isfinite(trained).all():
            raise AssertionError(f"Invalid final LoRA tensor: {name}")
        if name not in mapping:
            if not torch.equal(original, trained):
                raise AssertionError(f"LoRA changed a non-Q/V tensor: {name}")
            continue
        target = mapping[name]
        mask = torch.zeros(original.shape[0], dtype=torch.bool, device=original.device)
        for start, stop in target["slices"]:
            mask[start:stop] = True
        if not torch.equal(original[~mask], trained[~mask]):
            raise AssertionError(f"LoRA changed a frozen packed K slice: {name}")
        if not torch.equal(original[mask], trained[mask]):
            changed_stages.add(target["module"].split(".", 1)[0])
            changed_targets.append(name)
    expected = ({"feature_distribution_embedder", "column_aggregator", "icl_blocks"}
                if family == "tabpfn" else {"col_embedder", "row_interactor", "icl_predictor"})
    if changed_stages != expected:
        raise AssertionError(f"LoRA failed to update each intended transformer stage: {sorted(changed_stages)}")
    return {"only_qv_changed": True, "frozen_key_slices_unchanged": True,
            "changed_stages": sorted(changed_stages), "changed_target_count": len(changed_targets)}


def smoke(identifier, track, family):
    import torch
    from src.utils.recovery_check import compare, compare_trajectories
    from src.utils.audit_experiment import _inference_state
    root, limits = folder(identifier), policy()
    probe = adapter_probe(identifier, track, family)
    artifacts, rates = list(probe.pop("artifacts")), []
    native_track = track if family == "tabpfn" else None
    original = _inference_state(_base(track, family), native_track)
    for index, rate in enumerate(limits.learning_rates):
        rows, curves = {}, {}
        for arm in ("reference", "resumed"):
            path = root / "configs" / f"{track}_{family}_lr{index}_{arm}.yaml"
            if arm == "resumed":
                _run_training(path, stop=limits.interrupt_after)
            gradient_evidence = _run_training(path)
            row = rows[arm] = _audited(path)
            if row.get("successful_updates") != limits.target_updates:
                raise AssertionError("LoRA smoke did not execute its complete successful-update budget")
            curves[arm], curve_path = _curve(row, track)
            artifacts += [_binding(row["path"]), _binding(curve_path), _binding(gradient_evidence)]
        weights = compare(Path(rows["reference"]["path"]), Path(rows["resumed"]["path"]))
        monitors = compare_trajectories(curves["reference"], curves["resumed"],
                                       milestones=[0, limits.interrupt_after, limits.target_updates])
        if not weights["passed"] or not monitors["passed"]:
            raise AssertionError("Actual LoRA training uninterrupted/resumed weights or monitors differ")
        final = _inference_state(Path(rows["reference"]["path"]), native_track)
        scope = weight_scope(original, final, probe["adapter"]["targets"], family)
        rates.append({"learning_rate": float(rate), "weights": weights, "monitors": monitors, "weight_scope": scope})
        del final
    torch.cuda.empty_cache()
    return {"probe": probe, "rates": rates, "artifacts": artifacts}


def capacity(identifier, track, family):
    """Visit all original partitions; a passing first partition is insufficient."""
    from src.train.config import load_train_config
    path = folder(identifier) / "configs" / f"{track}_{family}_capacity.yaml"
    cfg = load_train_config(config_path=str(path))
    artifacts, partitions = [], []
    for split in range(int(cfg.corpus.n_splits)):
        gradient_evidence = _run_training(path, split=split)
        row = _audited(path, split=split)
        frame, curve_path = _curve(row, track)
        if row.get("successful_updates") != policy().capacity_updates:
            raise AssertionError("LoRA capacity check did not finish its production-shape update budget")
        if "processed_rows" not in frame or int(frame.iloc[-1].processed_rows) <= 0:
            raise AssertionError("LoRA capacity check processed no rows")
        partitions.append({"split": split, "audit": row, "processed_rows": int(frame.iloc[-1].processed_rows)})
        artifacts += [_binding(row["path"]), _binding(curve_path), _binding(gradient_evidence)]
    return {"partitions": partitions, "production_row_cap": OmegaConf.select(cfg, "train.max_rows_per_step"),
            "production_precision": {"amp": cfg.train.amp}, "artifacts": artifacts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "smoke", "capacity", "inspect", "diagnose"])
    parser.add_argument("--id")
    parser.add_argument("--task", type=int, choices=range(len(CASES)))
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.id)
        return 0
    if not args.id:
        parser.error("--id required")
    if args.action == "inspect":
        inspect_gate(args.id)
        print("All eight LoRA receipts passed; source, environment, plans, inputs and evidence match.")
        return 0
    if args.task is None:
        parser.error("--task required")
    # Diagnosis can inspect a failed historical workflow after instrumentation
    # changes. It cannot write or replace any of its validation receipts.
    if args.action == "diagnose":
        import torch
        if (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
                or os.environ.get("SLURM_CLUSTER_NAME") != "mindwell"
                or "B200" not in torch.cuda.get_device_name()):
            raise RuntimeError("LoRA inference diagnosis requires a submitted Mindwell B200 allocation")
        from src.utils.lora_diagnostics import diagnose
        with workflow_scope(args.id):
            diagnose(args.id, args.task)
        return 0
    workflow = _workflow(args.id)
    if args.action == "capacity":
        for task in range(len(CASES)):
            _receipt(args.id, "smoke", task, workflow)
    import torch
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("LoRA smoke/capacity checks require a submitted VSC GPU allocation")
    if os.environ.get("SLURM_CLUSTER_NAME") != "mindwell" or "B200" not in torch.cuda.get_device_name():
        raise RuntimeError("LoRA production capacity receipts require the declared Mindwell B200 GPU")
    receipt = folder(args.id) / "receipts" / f"{args.action}_{args.task}.json"
    # Atomic reservation stops duplicate tasks before either one can write checkpoints.
    result = {"schema": SCHEMA, "id": args.id, "stage": args.action, "task": args.task,
              "passed": False, "workflow_sha256": digest_json(workflow),
              "fingerprint": workflow["fingerprint"], "versions": workflow["versions"],
              "job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name()}
    write_json(receipt, result, exclusive=True)
    try:
        with workflow_scope(args.id):
            details = (smoke if args.action == "smoke" else capacity)(args.id, *CASES[args.task])
        evidence = evidence_folder(args.id) / "results" / f"{args.action}_{args.task}.json"
        write_json(evidence, details, exclusive=True)
        # A concurrent source/input edit invalidates the completed task too.
        if _workflow(args.id) != workflow:
            raise RuntimeError("LoRA workflow changed during GPU checks")
        result.update(passed=True, evidence=str(evidence), evidence_sha256=file_digest(evidence))
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        write_json(receipt, result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
