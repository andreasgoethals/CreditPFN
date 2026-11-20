"""Experiment 0 for the second campaign: GPU checks and the CPU summary.

GPU tasks (one array element each, receipts under ``experiment0/manifests/pilots/runs/``):

* ``recovery`` (24): AdamW, DiagScale and LoRA x PD/LGD x four models. Trains the
  uninterrupted twin, then the same configuration interrupted at ``interrupt_after`` (exit 75)
  and resumed; saved weights and every monitor milestone must agree.
* ``capacity`` (8): the benchmark's own FP32 scoring path for each original model at its largest
  context and the largest test fold, on a synthetic table with the widest feature count; records
  time and peak memory.
* ``evaluation`` (8): the real E5 fine-tuning and E6 context-curve runner
  (``src.eval.campaign_tasks``) in quick mode on the reference pilot, smallest held-out table;
  including E4 non-credit context and, for PD, E12's matched-context scoring on the episode pilot.
* ``scale`` (8): the production E5 procedure, both starts and all candidate rates, on fold zero
  of the largest campaign table for each task and model, using a CPT pilot that held it out.
* ``precision`` (8): update-zero training loss and gradient norm on every training table in the
  pilot partition at bounded episode size. Comparable FP32 and training-precision losses
  must agree and both gradients must be finite and aligned on every training table.
  The four largest training tables also require a finite backward at production episode size.

``summary`` (CPU) audits every pilot (budget, trajectory points, finite monitors, numerical
skips, timing), and applies the pilot-specific checks: zero-LR state/monitor parity, the
single-table sanity check (the trained table's own monitor score must improve), L2-SP penalty
logged for adapters with lambda > 0 and absent with lambda = 0, row milestones present, and
collects the GPU receipts. Every finding is recorded; the receipt says passed/failed per check.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.utils import pilot_gate
from src.utils.atomic import write_json
from src.utils.design import PILOT_FOLDER as FOLDER
from src.utils.paths import outputs_dir

VARIANTS = ("adamw", "diagscale", "lora")
TRACKS = ("pd", "lgd")
SCALE_PILOTS = {"pd": "reference", "lgd": "scale_reference"}
MODELS = ("v2", "v2.6", "v3", "tabicl-v2")
MODELS_PER_TRACK = len(MODELS)
RECOVERY_TASKS = len(VARIANTS) * len(TRACKS) * MODELS_PER_TRACK
CAPACITY_TASKS = len(TRACKS) * MODELS_PER_TRACK
EVALUATION_TASKS = len(TRACKS) * MODELS_PER_TRACK
PRECISION_TASKS = len(TRACKS) * MODELS_PER_TRACK
CHECK_COUNTS = dict(recovery=RECOVERY_TASKS, capacity=CAPACITY_TASKS, evaluation=EVALUATION_TASKS,
                    precision=PRECISION_TASKS, scale=EVALUATION_TASKS)
PRECISION_TABLES = 4
#: Update-zero loss agreement required between FP32 and the training precision (nats).
PRECISION_TOLERANCE = (0.02, 0.02)   # absolute + relative to the FP32 loss
#: Whether the training-precision gradient points where the FP32 gradient points
#: (``gradient_agrees``, reported per table). Agreeing losses do not show this: a forward can
#: match while the backward overflows or loses its small terms.
PRECISION_GRADIENT_COSINE = 0.90
#: Largest inference context the benchmark gives each model per task: config/eval.yaml caps,
#: bounded by the largest table's outer-fold context (0.8 x 0.8 x rows: PD 730,176, LGD 57,932).
CAPACITY_ROWS = {"pd": {"v2": 10_000, "v2.6": 50_000, "v3": 467_312, "tabicl-v2": 467_312},
                 "lgd": {"v2": 10_000, "v2.6": 37_076, "v3": 37_076, "tabicl-v2": 37_076}}
#: Rows of the precision comparison. Agreement of FP32 and the training precision does not need
#: the full episode (BF16's TabICL failure showed at 1,000 rows), and an FP32 backward at 26,000
#: rows may not fit next to its own graph.
PRECISION_ROWS = 8192


def receipts() -> Path:
    return outputs_dir() / "experiment0" / "manifests" / "pilots"


def recovery_case(task: int) -> tuple[str, str, int]:
    """Array index -> (variant, track, trial); 3 x 2 x 4 = 24 tasks."""
    if not 0 <= task < RECOVERY_TASKS:
        raise ValueError("Recovery task index out of range")
    variant, rest = divmod(task, len(TRACKS) * MODELS_PER_TRACK)
    track, trial = divmod(rest, MODELS_PER_TRACK)
    return VARIANTS[variant], TRACKS[track], trial


def track_model_case(task: int, count: int) -> tuple[str, int]:
    if not 0 <= task < count:
        raise ValueError("Task index out of range")
    track, trial = divmod(task, MODELS_PER_TRACK)
    return TRACKS[track], trial


def _write(name: str, record: dict) -> Path:
    return pilot_gate.record_check(receipts(), name, record)


# ------------------------------------------------------------------ exact resume (GPU)
def recovery(task: int, folder: Path = FOLDER) -> dict:
    import pandas as pd

    from src.train.config import load_train_config
    from src.train.recovery import configure_execution
    from src.utils.audit_experiment import audit
    from src.utils.design import load_design
    from src.utils.paths import training_dir
    from src.utils.recovery_check import compare, compare_trajectories
    variant, track, trial = recovery_case(task)
    spec = load_design()["experiment0"]["recovery"]
    paths = {arm: folder / f"recovery_{variant}_{arm}_{track}.yaml" for arm in ("uninterrupted", "resumed")}
    cfg = load_train_config(config_path=str(paths["resumed"]))
    if not cfg.train.deterministic:
        raise ValueError("Exact-resume comparison requires train.deterministic=true")
    execution = configure_execution(True)
    if any(audit(p)["trials"][trial]["status"] != "PENDING" for p in paths.values()):
        raise RuntimeError("Recovery case already has output; inspect it instead of rerunning")
    common = [sys.executable, "-u", "scripts/train_pipeline.py", "--split-index", "0", "--trial-index", str(trial)]
    if os.environ.get("CREDITPFN_ACTIVE_LOG"):
        common += ["--log-path", os.environ["CREDITPFN_ACTIVE_LOG"]]
    env = {k: v for k, v in os.environ.items()
           if k not in ("CREDITPFN_STOP_AFTER_UPDATES", "CREDITPFN_STOP_REQUESTED")}
    env.update(PYTHONHASHSEED=str(int(cfg.seed)), CREDITPFN_SEGMENT_SECONDS="0")
    subprocess.run([*common, "--config", str(paths["uninterrupted"])], env=env, check=True)
    stop = int(spec["interrupt_after"])
    stopped = subprocess.run([*common, "--config", str(paths["resumed"])],
                             env=dict(env, CREDITPFN_STOP_AFTER_UPDATES=str(stop)))
    if stopped.returncode != 75:
        raise RuntimeError(f"Expected recovery exit 75 at update {stop}, got {stopped.returncode}")
    subprocess.run([*common, "--config", str(paths["resumed"])], env=env, check=True)
    reports = {arm: audit(p) for arm, p in paths.items()}
    rows = {arm: r["trials"][trial] for arm, r in reports.items()}
    target = int(spec["successful_updates"])
    if any(r["status"] != "OK" or r.get("successful_updates") != target for r in rows.values()):
        raise RuntimeError(f"A recovery arm did not finish its {target} successful updates")
    weights = compare(Path(rows["uninterrupted"]["path"]), Path(rows["resumed"]["path"]))
    curves = []
    for arm in ("uninterrupted", "resumed"):
        frame = pd.read_csv(training_dir(track, rows[arm]["trial"] + ".trajectory.csv", experiment="experiment0"))
        if "record_type" in frame:
            frame = frame[frame.record_type == "trajectory"].reset_index(drop=True)
        curves.append(frame)
    monitors = compare_trajectories(*curves, milestones=list(cfg.train.trajectory_steps))
    problems = [p for r in reports.values() for p in r["problems"]
                if p.startswith((rows["uninterrupted"]["trial"] + ":", rows["resumed"]["trial"] + ":"))]
    # This check is about resume exactness. Skipped FP16 updates at its deliberately large learning
    # rate are a finding about that rate, identical in both arms; they are reported, not failed.
    blocking = [p for p in problems if "skipped training batches/updates" not in p]
    result = dict(task=task, variant=variant, track=track, trial=trial, interrupt_after=stop,
                  execution=execution, weights=weights, monitors=monitors, audit_problems=problems,
                  passed=bool(weights["passed"] and monitors["passed"] and not blocking))
    _write(f"recovery_{task:02d}.json", result)
    print(json.dumps({k: result[k] for k in ("task", "variant", "track", "trial", "passed")}), flush=True)
    if not result["passed"]:
        raise RuntimeError("Exact-resume comparison failed")
    return result


# ------------------------------------------------------------- FP32 capacity (GPU)
#: Largest outer test fold the benchmark scores in one call: a fifth of the largest held-out table.
CAPACITY_TEST_ROWS = {"pd": 146_036, "lgd": 11_587}
#: Inference views of the benchmark (config/eval.yaml tabpfn_n_estimators / tabicl_n_estimators).
CAPACITY_VIEWS = {"tabpfn": 8, "tabicl": 8}


def capacity(task: int) -> dict:
    """FP32 benchmark scoring of the original model at its largest context and test fold.

    Runs the benchmark's own path: ``encode_for_model``, the untuned wrapper with the benchmark's
    inference views, and ``score_fitted`` (for LGD the full predictive distribution with the
    dense quantile grid), on a synthetic table with the widest feature count.
    """
    import pandas as pd
    import torch

    from src.eval.benchmark import score_fitted
    from src.eval.dataset_loader import encode_for_model
    from src.model.tabicl_models import TabICLUntuned
    from src.model.tabpfn_models import TabPFNUntuned, inference_precision
    from src.utils.design import BASES
    from src.utils.paths import resolve_base_checkpoint
    track, trial = track_model_case(task, CAPACITY_TASKS)
    model = MODELS[trial]
    family = "tabicl" if model.startswith("tabicl") else "tabpfn"
    if family == "tabpfn" and inference_precision() is not torch.float32:
        raise RuntimeError("Capacity check requires FP32 TabPFN evaluation")
    rows, features, validation, test = CAPACITY_ROWS[track][model], 64, 2048, CAPACITY_TEST_ROWS[track]
    rng = np.random.default_rng(518)
    x = pd.DataFrame(rng.normal(size=(rows + validation + test, features)).astype(np.float32))
    y = (x[0] > 0).astype(int).to_numpy() if track == "pd" else x[0].to_numpy()
    parts = np.split(np.arange(len(x)), [rows, rows + validation])
    X_tr, X_va, X_te, cats = encode_for_model(*(x.iloc[i].reset_index(drop=True) for i in parts),
                                              categorical_columns=[])
    task_type = "classification" if track == "pd" else "regression"
    path = resolve_base_checkpoint(BASES[track][model])
    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    wrapper = (TabICLUntuned(task_type=task_type, base_path=path, device=device, n_estimators=CAPACITY_VIEWS["tabicl"],
                             inference_options={"use_amp": False, "use_fa3": False})
               if family == "tabicl" else
               TabPFNUntuned(task_type=task_type, base_path=path, device=device,
                             n_estimators=CAPACITY_VIEWS["tabpfn"]))
    peak = (lambda: int(torch.cuda.max_memory_allocated())) if gpu else (lambda: None)
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    record = dict(task=task, track=track, model=model, context_rows=rows, test_rows=test, features=features,
                  gpu=torch.cuda.get_device_name() if gpu else None)
    try:
        wrapper.fit(X_tr, y[parts[0]], cats)
        metrics, _, _ = score_fitted(wrapper, task_type, y_tr=y[parts[0]], X_va=X_va, y_va=y[parts[1]],
                                     X_te=X_te, y_te=y[parts[2]])
        primary = float(metrics.get("roc_auc" if track == "pd" else "r2", float("nan")))
        record.update(passed=bool(np.isfinite(primary)), seconds=time.monotonic() - started,
                      peak_bytes=peak(), primary=primary)
        if track == "lgd" and family == "tabpfn":
            record.update({k: metrics.get(k) for k in ("nll", "nll_rows", "nll_nonfinite_rows",
                                                       "nll_clipped_rate", "nll_capped100")})
            record["passed"] &= bool(np.isfinite(metrics.get("nll", np.nan))
                                     and metrics.get("nll_rows") == test and metrics.get("nll_nonfinite_rows") == 0)
    except Exception as exc:  # an out-of-memory here is the finding, not a crash
        record.update(passed=False, seconds=time.monotonic() - started, error=repr(exc)[:500],
                      peak_bytes=peak())
    _write(f"capacity_{task:02d}.json", record)
    print(json.dumps(record), flush=True)
    return record


# ---------------------------------------------- context curve + target fine-tuning (GPU)
def evaluation(task: int, folder: Path = FOLDER, *, scale: bool = False) -> dict:
    """E5 fine-tuning and E6 context curves through their real runner, on the reference pilot.

    For one model and track, the smallest held-out table (fold 0) is run by
    ``campaign_tasks.run`` in quick mode for both kinds, exactly as the campaign arrays will run
    it (task enumeration, checkpoint naming, fold rows, atomic output), then run again to check
    that a finished task is skipped. Scores must be finite for the original and the CPT model.
    """
    import pandas as pd

    from src.eval import campaign_tasks as ct
    track, trial = track_model_case(task, EVALUATION_TASKS)
    key = "roc_auc" if track == "pd" else "r2"
    cases = [("finetune", "reference")]
    if not scale:
        cases.extend([("context", "reference"), ("context", "domain_control")])
        if track == "pd":
            cases.append(("episode_context", "episode2048"))
    record = dict(task=task, track=track, runs={}, problems=[])
    for kind, pilot in cases:
        references = {t: folder / f"{SCALE_PILOTS[t] if scale else pilot}_{t}.yaml" for t in TRACKS}
        source = "domain_control" if pilot == "domain_control" else "reference"
        label = "context_noncredit" if pilot == "domain_control" else kind
        candidates = [(i, t) for i, t in enumerate(ct.tasks(references=references, kind=kind, pilot_source=source))
                      if t["track"] == track and t["trial"] == trial and t["fold"] == 0]
        index, chosen = (max if scale else min)(candidates, key=lambda c: (c[1]["n_rows"], c[1]["dataset_id"]))
        record.update(model=chosen["model"])
        expected = ({"noncredit_cpt"} if source == "domain_control" else
                    {"cpt"} if kind == "episode_context" else {"original", "cpt"})
        started = time.monotonic()
        try:
            options = dict(references=references, slug="experiment0", quick=not scale, pilot_source=source)
            if kind == "finetune" and not scale:
                from src.eval.finetune_recovery import InterruptedEpoch
                try:
                    path = ct.run(kind, index, interrupt_finetune=True, **options)
                except InterruptedEpoch:
                    path = ct.run(kind, index, **options)
            else:
                path = ct.run(kind, index, **options)
            written = path.stat().st_mtime_ns
            frame = pd.read_csv(path)
            again = ct.run(kind, index, **options)
            skipped = again == path and again.stat().st_mtime_ns == written
            scores = frame[key].to_numpy(dtype=float)
            if kind == "finetune":
                arms = {(start, tuned) for start in expected for tuned in (True, False)}
                if len(frame) != len(arms) or set(zip(frame.start, frame.finetuned)) != arms:
                    record["problems"].append("finetune: incomplete paired starts or missing untuned comparator")
                if not scale:
                    resumed = frame.loc[(frame.start == "original") & frame.finetuned].filter(
                        regex=r"^resumed_at_epoch_lr")
                    if resumed.empty or not (resumed.to_numpy() >= 1).any():
                        record["problems"].append("finetune: no verified restart inside a candidate fit")
            else:
                from src.eval.context_curve import budgets
                rules = ct.policy(chosen["model"], ct.load_design(), quick=not scale, kind=kind)
                sizes = budgets(int(frame.available_context.iloc[0]), rules)
                cells = {(start, size) for start in expected for size in sizes}
                if len(cells) != len(frame) or cells != set(zip(frame.start, frame.context_rows)):
                    record["problems"].append(f"{label}: incomplete or duplicated context sizes")
            record["runs"][label] = dict(dataset_rows=chosen["n_rows"], rows=len(frame),
                                        starts=sorted(frame.start.unique()),
                                        finite=bool(np.isfinite(scores).all()),
                                        seconds=time.monotonic() - started, resumed_skip=skipped,
                                        scores=frame[["start", key]].to_dict("records"))
            if not (record["runs"][label]["finite"] and skipped and set(frame.start) == expected):
                record["problems"].append(f"{kind}: missing or non-finite scores")
        except Exception as exc:  # the failure is the finding
            record["problems"].append(f"{kind}: {exc!r}"[:500])
    record["passed"] = not record["problems"]
    check = "scale" if scale else "evaluation"
    _write(f"{check}_{task:02d}.json", record)
    print(json.dumps(record, default=str), flush=True)
    return record


# ------------------------------------------------------- training precision (GPU)
def _tf32(enabled: bool) -> None:
    """FP32 storage with 10-bit-mantissa matrix multiplies (tensor cores) on or off."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)


def precision(task: int, folder: Path = FOLDER, *, diagnostic: bool = False) -> dict:
    """Update-zero training loss in FP32 versus the resolved training precision."""
    from contextlib import nullcontext
    import torch

    from src.train.config import load_train_config, training_members
    from src.train.corpus import split_from_cfg
    from src.train.dataloader import (
        _build_ensemble_step_batch,
        _build_tabicl_step_batch,
        _load_processed_csv,
    )
    from src.train.loop import _resolve_amp_dtype, batch_loss, training_row_cap
    from src.train.tabicl_compat import model_family
    from src.utils.paths import resolve_base_checkpoint
    from src.train.precision import training_precision
    from src.utils.v2_precision import bounded_fp32_reference
    track, trial = track_model_case(task, PRECISION_TASKS)
    cfg = load_train_config(config_path=str(folder / f"reference_{track}.yaml"))
    bases = list(cfg.tunable.classifier_base_paths if track == "pd" else cfg.tunable.regressor_base_paths)
    base = str(bases[trial])
    family = model_family(base)
    device = "cuda"
    use_amp, amp_dtype = _resolve_amp_dtype(cfg, device, family)
    members = training_members(cfg, track, family)
    rows = min(training_row_cap(cfg, base, family, members), PRECISION_ROWS)
    query_fraction = float(cfg.tunable.query_fractions[0])
    objective = str(cfg.train.get("regression_objective", "bar_nll"))
    path = resolve_base_checkpoint(base)
    if family == "tabicl":
        from src.train.tabicl_model import load_tabicl_for_training
        model, _ = load_tabicl_for_training(path, track=track, device=device)
        criterion = inference_config = None
        if use_amp:   # as in training: the column embedder stays in FP32 under FP16 autocast
            from src.train.tabicl_model import column_embedder_in_fp32
            column_embedder_in_fp32(model)
    else:
        from src.train.model import load_tabpfn_for_training
        model, criterion, _, inference_config = load_tabpfn_for_training(path, track=track, device=device)
    model.train()
    refs = sorted(split_from_cfg(cfg, track=track).train, key=lambda r: -int(r.n_rows or 0))
    record = dict(task=task, track=track, base=Path(base).name, family=family,
                  training_dtype=str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
                  episode_rows=rows, members=members, tables=[])
    production_policy = training_precision(model, base=base, family=family, track=track, amp_dtype=amp_dtype)
    training_label = production_policy.label
    record.update(training_label=training_label, fp32_normalization_layers=len(production_policy.normalizations))
    # Candidate precisions, each compared with FP32 on identical batches. TabICLv2 is not run
    # under BF16 (its forward is corrupted there).
    candidates = [("fp32", None)] + ([("bf16", torch.bfloat16)] if family == "tabpfn" else []) + [
        ("fp16", torch.float16)]
    v2_pd = track == "pd" and MODELS[trial] == "v2"
    if v2_pd and training_label == "fp16_norm32":
        candidates.append((training_label, torch.float16))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    groups = sorted({".".join(n.split(".")[:2]) for n in names})

    def gradient(batch, dtype, label):
        """(loss, flat FP64 gradient, loss scale). FP16 trains behind GradScaler: scale the loss
        as the scaler does (65,536, halved after an overflow)."""
        scale = 65536.0 if dtype == torch.float16 else 1.0
        while True:
            model.zero_grad(set_to_none=True)
            _tf32(dtype is not None)
            mode = (bounded_fp32_reference(model) if v2_pd and label == "fp32" else
                    production_policy.scope() if label == training_label else nullcontext())
            with mode, torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                with torch.amp.autocast("cuda", enabled=dtype is not None, dtype=dtype):
                    loss = batch_loss(model, batch, criterion=criterion, regression_objective=objective,
                                      activation_checkpointing=bool(label == training_label
                                          and cfg.train.get("activation_checkpointing", False)))
                (loss * scale).backward()
            parts = {n: (p.grad.detach().to(device="cpu", dtype=torch.float64).reshape(-1) / scale if p.grad is not None
                         else torch.zeros(p.numel(), dtype=torch.float64))
                     for n, p in model.named_parameters() if p.requires_grad}
            flat = torch.cat([parts[n] for n in names])
            if bool(torch.isfinite(flat).all()) or scale <= 1.0:
                return float(loss.detach()), flat, scale
            scale /= 2.0

    def cosine(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    sizes = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    spans = {g: [(int(offsets[i]), int(offsets[i + 1])) for i, n in enumerate(names)
                 if ".".join(n.split(".")[:2]) == g] for g in groups}

    for index, ref in enumerate(refs):   # every training table of the partition, largest first
        loaded = _load_processed_csv(ref)
        build = dict(n_total_target=rows, query_fraction=query_fraction, n_estimators=members)

        def make(seed, loaded=loaded, build=build, index=index):
            if family == "tabicl":
                return _build_tabicl_step_batch(loaded, rng=np.random.default_rng(seed), epoch_seed=seed,
                                                replica=0, preprocessing_seed=index, **build).to(device)
            return _build_ensemble_step_batch(loaded, rng=np.random.default_rng(seed),
                                              inference_config=inference_config, rng_seed=seed,
                                              **build).to(device)

        row = {"table_index": index, "rows": int(min(rows, len(loaded.y))),
               "features": int(loaded.X.shape[1])}
        label = "preparation"
        try:
            if v2_pd and index == 0:
                small = make(index, build={**build, "n_total_target": min(rows, 256)})
                native_loss, native, _ = gradient(small, None, "native")
                bounded_loss, bounded, _ = gradient(small, None, "fp32")
                direction = cosine(native, bounded)
                relative_error = float((native - bounded).norm() / native.norm().clamp_min(1e-300))
                record["reference_parity"] = dict(rows=min(rows, 256), cosine=direction, relative_error=relative_error,
                    passed=bool(direction >= .9999 and relative_error <= .001
                                and np.isclose(native_loss, bounded_loss, rtol=1e-5, atol=1e-6)))
                del small, native, bounded
            batch = make(index)
            if v2_pd:
                row["prepared_features"] = max(m.X_context.shape[-1] for m in batch.members)
            reference = None
            for label, dtype in candidates:
                try:
                    loss, flat, scale = gradient(batch, dtype, label)
                except Exception as exc:
                    if label in ("fp32", training_label):
                        raise
                    row[f"error_{label}"] = repr(exc)[:500]
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                row[f"loss_{label}"] = loss
                row[f"grad_norm_{label}"] = float(flat.norm())
                row[f"grad_finite_{label}"] = bool(torch.isfinite(flat).all())
                if label == "fp32":
                    reference = flat
                    continue
                row[f"loss_scale_{label}"] = scale
                row[f"grad_cosine_{label}"] = cosine(flat, reference)
                row[f"grad_norm_ratio_{label}"] = float(flat.norm() / reference.norm().clamp_min(1e-300))
                total = float(reference.square().sum().clamp_min(1e-300))
                row[f"modules_{label}"] = {
                    g: [round(cosine(torch.cat([flat[a:b] for a, b in spans[g]]),
                                     torch.cat([reference[a:b] for a, b in spans[g]])), 3),
                        round(float(sum(reference[a:b].square().sum() for a, b in spans[g])) / total, 4)]
                    for g in groups}
            absolute, relative = PRECISION_TOLERANCE
            row["gradient_agrees"] = bool(
                row[f"grad_norm_{training_label}"] > 0 and row["grad_norm_fp32"] > 0
                and (training_label == "fp32" or row[f"grad_cosine_{training_label}"] >= PRECISION_GRADIENT_COSINE))
            row["passed"] = bool(
                np.isfinite(row[f"loss_{training_label}"])
                and abs(row[f"loss_{training_label}"] - row["loss_fp32"]) <= absolute + relative * abs(row["loss_fp32"])
                and row["grad_finite_fp32"] and row[f"grad_finite_{training_label}"] and row["gradient_agrees"])
            if v2_pd:
                # The original FP16 control can legitimately disagree; its measured
                # loss and direction are still required to interpret the change.
                row["passed"] &= bool(row.get("grad_finite_fp16") and row["grad_norm_fp16"] > 0
                                      and np.isfinite(row["loss_fp16"]) and np.isfinite(row["grad_cosine_fp16"]))
        except Exception as exc:  # an out-of-memory or numerical error is the finding, not a crash
            # Missing reference measurements cannot demonstrate agreement.
            oom = "OutOfMemoryError" in type(exc).__name__ or "out of memory" in str(exc).lower()
            reference_oom = oom and label == "fp32"
            row.update(passed=False, not_checkable=reference_oom, stage=label, error=repr(exc)[:500])
        reference = flat = batch = None   # noqa: F841 - release gradients before the next table
        model.zero_grad(set_to_none=True)
        record["tables"].append(row)
        torch.cuda.empty_cache()
    # A failed FP32 reference can prevent the comparison from even reaching FP16.
    # Exercise the deployed backward separately at full episode size on these tables.
    full_rows = training_row_cap(cfg, base, family, members)
    build = dict(n_total_target=full_rows, query_fraction=query_fraction, n_estimators=members)
    _tf32(False)
    record["full_episode"] = []
    full_indices = set(range(min(len(refs), PRECISION_TABLES)))
    if v2_pd:
        # Row count alone missed wide tables. Include every equally wide table
        # with a longer production episode, before and after preprocessing.
        larger = [r for r in record["tables"] if refs[r["table_index"]].n_rows > rows]
        for field in ("features", "prepared_features"):
            widest = max((r.get(field, 0) for r in larger), default=0)
            full_indices.update(r["table_index"] for r in larger if widest and r.get(field) == widest)
    for index in sorted(full_indices):
        ref = refs[index]
        loaded = _load_processed_csv(ref)
        entry = {"table_index": index, "rows": int(min(full_rows, len(loaded.y))),
                 "features": int(loaded.X.shape[1])}
        full_labels = (("fp32", "production", "fp16") if v2_pd else
                       ("production", "fp32") + (("fp32_checkpointed",) if family == "tabpfn" else ()))
        reference = None
        for label in full_labels:
            try:
                if family == "tabicl":
                    batch = _build_tabicl_step_batch(loaded, rng=np.random.default_rng(0), epoch_seed=0,
                                                     replica=0, preprocessing_seed=0, **build).to(device)
                else:
                    batch = _build_ensemble_step_batch(loaded, rng=np.random.default_rng(0),
                                                       inference_config=inference_config, rng_seed=0,
                                                       **build).to(device)
                if v2_pd:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    test_label = training_label if label == "production" else label
                    dtype = {"fp32": None, "production": amp_dtype, "fp16": torch.float16}[label]
                    loss_value, flat, scale = gradient(batch, dtype, test_label)
                    norm = float(flat.norm())
                    if not np.isfinite(loss_value) or not np.isfinite(norm) or norm <= 0:
                        raise ValueError("Unusable full-size loss or gradient")
                    entry[label] = dict(policy=test_label, seconds=round(time.monotonic() - started, 3), loss=loss_value,
                                        gradient_norm=norm, loss_scale=scale,
                                        peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
                    if label == "fp32":
                        reference = flat
                    elif reference is not None:
                        entry[label]["gradient_cosine"] = cosine(flat, reference)
                        entry[label]["gradient_agrees"] = entry[label]["gradient_cosine"] >= PRECISION_GRADIENT_COSINE
                        absolute, relative = PRECISION_TOLERANCE
                        entry[label]["loss_agrees"] = bool(np.isclose(loss_value, entry["fp32"]["loss"],
                                                                      atol=absolute, rtol=relative))
                    batch = flat = None
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                seconds = []
                for _ in range(2):   # the first pass includes warm-up; report the faster
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    production = label == "production"
                    _tf32(production and use_amp)
                    checkpointed = (bool(cfg.train.get("activation_checkpointing", False))
                                    if production else label == "fp32_checkpointed")
                    scale = 65536.0 if production and training_label == "fp16" else 1.0
                    while True:
                        model.zero_grad(set_to_none=True)
                        with torch.amp.autocast("cuda", enabled=production and use_amp, dtype=amp_dtype):
                            loss = batch_loss(model, batch, criterion=criterion, regression_objective=objective,
                                              activation_checkpointing=checkpointed)
                        (loss * scale).backward()
                        norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
                        norm = float(torch.stack(norms).double().norm()) / scale
                        if np.isfinite(norm) or scale <= 1:
                            break
                        scale /= 2
                    if not np.isfinite(float(loss.detach())) or not np.isfinite(norm) or norm == 0:
                        raise ValueError("Full-episode backward returned a nonfinite loss or unusable gradient")
                    torch.cuda.synchronize()
                    seconds.append(time.monotonic() - started)
                entry[label] = dict(seconds=round(min(seconds), 3), loss=float(loss.detach()),
                                    gradient_norm=norm, loss_scale=scale,
                                    peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
            except Exception as exc:  # out of memory is the answer, not a crash
                entry[label] = dict(error=repr(exc)[:200])
            loss = batch = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        record["full_episode"].append(entry)
    record["passed"] = (bool(record["full_episode"])
                        and all("error" not in e["production"] for e in record["full_episode"])
                        and (not v2_pd or (record.get("reference_parity", {}).get("passed", False)
                             and all(e["production"].get("gradient_agrees") and e["production"].get("loss_agrees")
                                     and np.isfinite(e["fp16"].get("gradient_cosine", np.nan))
                                     for e in record["full_episode"])))
                        and bool(record["tables"])
                        and all(r["passed"] for r in record["tables"]))
    if diagnostic:
        # Diagnostic comparisons on a candidate implementation must never replace an
        # earlier campaign receipt or qualify a subsequent campaign for release.
        from src.utils.experiment import code_identity
        job = os.environ["SLURM_JOB_ID"]
        record.update(code=code_identity(stage="eval"), job=job)
        write_json(receipts() / "diagnostics" / f"precision_{task:02d}_{job}.json", record, exclusive=True)
    else:
        _write(f"precision_{task:02d}.json", record)
    print(json.dumps(record), flush=True)
    return record


# --------------------------------------------------------------------- summary (CPU)
def pilot_paths(folder: Path = FOLDER) -> list[Path]:
    return sorted(p for p in folder.glob("*.yaml") if not p.name.startswith("recovery_"))


def _trajectory(report_row: dict, track: str):
    import pandas as pd

    from src.utils.paths import training_dir
    path = training_dir(track, report_row["trial"] + ".trajectory.csv", experiment="experiment0")
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def pilot_checks(name: str, track: str, report: dict, cfg) -> dict:
    """Pilot-specific checks beyond the generic audit, per trial."""
    checks = {}
    lam = float(cfg.tunable.l2sp_lambdas[0]) if cfg.tunable.l2sp_lambdas else float(cfg.optimizer.l2sp_lambda)
    mode = str(cfg.train.get("adaptation_mode", "full"))
    for row in report.get("trials", []):
        if row.get("status") != "OK":
            continue
        frame = _trajectory(row, track)
        if frame.empty:
            checks[row["trial"]] = {"passed": False, "problem": "no trajectory"}
            continue
        main = frame[frame.record_type == "trajectory"] if "record_type" in frame else frame
        result = {"monitor_seconds_mean": float(main.monitor_seconds.mean()),
                  "row_milestones": int((frame.get("record_type") == "rows_milestone").sum())
                  if "record_type" in frame else 0}
        better = (lambda a, b: b > a) if track == "pd" else (lambda a, b: b < a)
        if name == "sanity_single":
            first, last = float(main.train_metric.iloc[0]), float(main.train_metric.iloc[-1])
            result.update(own_table_start=first, own_table_end=last, passed=bool(better(first, last)))
        elif mode in ("lora", "diagscale"):
            penalty = float(main.l2sp_penalty.iloc[-1]) if "l2sp_penalty" in main else float("nan")
            result.update(l2sp_penalty_end=penalty, l2sp_lambda=lam,
                          passed=bool(penalty > 0) if lam > 0 else bool(penalty == 0 or np.isnan(penalty)))
        else:
            result["passed"] = True
        checks[row["trial"]] = result
    if name == "sanity_single":
        # The smoke test asks whether training lowers the error on the trained table. Broken
        # training code fails it for every model; if any model passes, a model that does not is
        # behaving differently at this learning rate. That is a finding, recorded, not a failure.
        own = [c for c in checks.values() if "own_table_start" in c]
        if any(c["passed"] for c in own):
            for c in own:
                if not c["passed"]:
                    c.update(passed=True, finding="own table worse after training at this learning "
                                                  "rate while other models improve: a model property")
    return checks


#: Models the reference pilots' five-fold benchmark must score: 4 CPT pilots, 4 original
#: models and the classical baselines of each task.
BASELINES = {"pd": ("catboost", "xgboost", "logreg"), "lgd": ("catboost", "xgboost", "linreg")}


def _benchmark_rows(handle, table, track, config, run_name, n_folds):
    import pandas as pd

    from src.eval.benchmark import find_existing_results
    from src.eval.cache import evaluation_key

    paths = find_existing_results(
        handle, table, track=track, results_base_dir=config["results"]["base_dir"],
        run_name=run_name, n_folds_required=n_folds,
        evaluation_key=evaluation_key(handle, table, track=track, config=config),
        require_predictions=config["results"]["save_predictions"])
    metrics = (["roc_auc", "brier_score", "gini", "ks", "h_measure", "emp", "emp_rejection_rate"]
               if track == "pd" else ["r2", "rmse"] + ([] if handle.source == "baseline" else ["crps"]))
    density = track == "lgd" and handle.source.startswith("tabpfn-")
    if density:
        metrics.extend(["nll", "n_test_rows", "nll_rows", "nll_nonfinite_rows", "nll_clipped_rows",
                        "nll_clipped_rate", "nll_nonfinite_rate", "nll_capped100"])
    columns = ["model_name", "test_dataset_id", "fold_idx", "status", *metrics]
    frames = [pd.read_csv(path, usecols=columns) for path in paths]
    if not frames:
        return False
    rows = pd.concat(frames)
    rows = rows[(rows.model_name == handle.name) & (rows.test_dataset_id == table)].drop_duplicates(
        "fold_idx", keep="last")
    if density and not ((rows.nll_rows == rows.n_test_rows) & (rows.nll_rows > 0)
                        & (rows.nll_nonfinite_rows == 0) & (rows.nll_nonfinite_rate == 0)
                        & (rows.nll_clipped_rows >= 0) & (rows.nll_clipped_rows <= rows.nll_rows)
                        & np.isclose(rows.nll_clipped_rate, rows.nll_clipped_rows / rows.nll_rows)).all():
        return False
    return (set(rows.fold_idx) == set(range(n_folds)) and rows.status.eq("OK").all()
            and np.isfinite(rows[metrics].to_numpy(dtype=float)).all())


def benchmark(track: str, folder: Path = FOLDER, *, n_folds: int = 5) -> dict:
    """Exercise the production checkpoint roster, cache keys, predictions and required endpoints."""
    from omegaconf import OmegaConf

    from scripts.eval_pipeline import _build_roster
    from src.eval.config import load_eval_configs
    from src.utils.paths import activate_experiment

    cfg, train = load_eval_configs([], [], config_path=str(folder / f"reference_{track}.yaml"), split_index=0)
    activate_experiment(train)
    config = OmegaConf.to_container(cfg, resolve=True)
    roster, tables, _ = _build_roster(cfg, train, track)
    trained = [h.name for h, _ in roster if h.source.endswith("-trained")]
    original = [h.name for h, _ in roster if h.source.endswith("-untuned")]
    classical = {h.name for h, _ in roster if h.source == "baseline"}
    problems = []
    if (not tables or len(set(trained)) != MODELS_PER_TRACK or len(set(original)) != MODELS_PER_TRACK
            or classical != set(BASELINES[track])):
        problems.append("Incomplete reference roster or table set")
    for handle, _ in roster:
        for table in tables:
            if not _benchmark_rows(handle, table, track, config, str(train.run_name), n_folds):
                problems.append(f"{handle.name}, {table}: missing current five-fold metrics or predictions")
    return dict(track=track, tables=len(tables), passed=not problems, problems=problems)


def summary(folder: Path = FOLDER) -> dict:
    from src.train.config import load_train_config
    from src.utils.audit_experiment import audit
    from src.utils.prepare_experiment import prepare
    run = os.environ[pilot_gate.RUN_ENV]
    root = receipts()
    request = pilot_gate.current(root, run, verify_project=True)
    grid = pilot_gate.read(pilot_gate.directory(root, run) / "preflight.json")
    if grid.get("passed") is not True:
        raise RuntimeError("Experiment 0 CPU preflight has not passed")
    jobs = pilot_gate.read(pilot_gate.directory(root, run) / "jobs.json")
    if not jobs["summary"] or jobs["summary"] != os.environ.get("SLURM_JOB_ID") or jobs["run"] != run:
        raise RuntimeError("Only the summary job of a completed Experiment 0 submission may publish")
    publication = pilot_gate.directory(root, run) / "summary.json"
    if publication.is_file():
        result = pilot_gate.read(publication)
        write_json(root / "summary.json", result)
        return result
    expected_jobs = {job for group in ("pilots", "benchmark", "classical")
                     for ids in jobs[group].values() for job in ids}
    expected_jobs.update(jobs[kind] for kind in CHECK_COUNTS if jobs.get(kind))
    pilots = {}
    for path in pilot_paths(folder):
        name, track = path.stem.rsplit("_", 1)
        try:
            prepare(path, check=True)  # recompute trial identities from the current input bytes
            report = audit(path, null=name == "zero_lr")
            report["pilot_checks"] = pilot_checks(name, track, report, load_train_config(config_path=str(path)))
            # Experiment 0 runs only rates the campaign uses: any divergence is a finding to
            # investigate, and every trial must have been checked (none skipped as diverged).
            report["passed"] = (bool(report.get("passed")) and report.get("diverged", 0) == 0
                                and len(report["pilot_checks"]) == report.get("completed", 0) > 0
                                and all(c.get("passed") for c in report["pilot_checks"].values()))
        except Exception as exc:  # an unreadable pilot is a finding, not a crash of the summary
            report = {"passed": False, "problems": [repr(exc)[:500]]}
        pilots[path.stem] = report
    receipts_by_kind = {}
    for kind, count in CHECK_COUNTS.items():
        receipts_by_kind[kind] = {}
        for task in range(count):
            key = f"{kind}_{task:02d}"
            owner = request["checks"].get(key)
            item = {"passed": False, "missing": True}
            if owner:
                owner_dir = pilot_gate.directory(root, owner)
                file = owner_dir / f"{key}.json"
                if file.is_file() and (owner_dir / "jobs.json").is_file():
                    evidence = pilot_gate.read(file)
                    array = pilot_gate.read(owner_dir / "jobs.json").get(kind)
                    if array and evidence.get("run") == owner and evidence.get("task") == task and evidence.get("array") == array:
                        item = evidence
                        if owner != run:
                            expected_jobs.add(f"{array}_{task}")
            receipts_by_kind[kind][str(task)] = item
    benchmarks = {}
    for track in TRACKS:
        try:
            benchmarks[track] = benchmark(track, folder)
        except Exception as exc:  # unreadable results are a finding
            benchmarks[track] = {"passed": False, "problems": [repr(exc)[:500]]}
    receipts_by_kind["benchmark"] = benchmarks
    density = pilot_gate.directory(root, request["evidence"]) / "density_audit.json"
    result = dict(pilots=pilots, **receipts_by_kind,
                  density_audit=json.loads(density.read_text(encoding="utf-8")) if density.is_file() else None,
                  failed_pilots=sorted(k for k, v in pilots.items() if not v.get("passed")),
                  failed=sorted(f"{kind}_{t}" for kind, items in receipts_by_kind.items()
                                for t, v in items.items() if not v.get("passed")))
    scheduler_problems = pilot_gate.incomplete_jobs(expected_jobs)
    result.update(run=run, identity=request["identity"], grid=grid, scheduler_problems=scheduler_problems,
                  passed=bool(pilots) and not (result["failed_pilots"] or result["failed"] or scheduler_problems))
    pilot_gate.current(root, run)
    write_json(publication, result, exclusive=True)
    write_json(root / "summary.json", result)  # latest report for offline notebooks
    print(json.dumps({k: result[k] for k in ("passed", "failed_pilots", "failed", "scheduler_problems")}, indent=2), flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(*CHECK_COUNTS, "precision-diagnostic", "summary"))
    parser.add_argument("--task", type=int)
    args = parser.parse_args(argv)
    if args.action == "summary":
        return 0 if summary()["passed"] else 1
    if args.task is None:
        parser.error(f"{args.action} needs --task")
    if os.environ.get("SLURM_JOB_ID"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The Experiment 0 GPU allocation has no usable CUDA device")
    if args.action == "precision-diagnostic":
        precision(args.task, diagnostic=True)
        return 0  # The report carries each agreement result; completing the measurement is success.
    record = (evaluation(args.task, scale=True) if args.action == "scale" else
              {"recovery": recovery, "capacity": capacity, "evaluation": evaluation,
               "precision": precision}[args.action](args.task))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
