"""Experiment 0 for the second campaign: GPU checks and the CPU summary.

GPU tasks (one array element each, results in ``experiment0/manifests/campaign2/``):

* ``recovery`` (24): AdamW, DiagScale and LoRA x PD/LGD x four models. Trains the
  uninterrupted twin, then the same configuration interrupted at ``interrupt_after`` (exit 75)
  and resumed; saved weights and every monitor milestone must agree.
* ``capacity`` (8): the benchmark's own FP32 scoring path for each original model at its largest
  context and the largest test fold, on a synthetic table with the widest feature count; records
  time and peak memory.
* ``evaluation`` (8): the real E4 fine-tuning and E10 context-curve runner
  (``src.eval.campaign_tasks``) in quick mode on the reference pilot, smallest held-out table;
  for PD also E9's matched-context scoring on the ``episode2048`` pilot.
* ``precision`` (8): the update-zero training loss and gradient norm of each original model on
  its four largest training tables at the real episode size, computed by the training loop's own
  ``batch_loss`` in FP32 and in the resolved training precision. The two losses must agree
  (the first study trained TabICLv2 under BF16, which corrupted its forward) and the two
  gradients must be finite and point the same way (cosine similarity).

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

from src.utils.atomic import write_json
from src.utils.design import PILOT_FOLDER as FOLDER
from src.utils.paths import manifests_dir

VARIANTS = ("adamw", "diagscale", "lora")
TRACKS = ("pd", "lgd")
MODELS = ("v2", "v2.6", "v3", "tabicl-v2")
MODELS_PER_TRACK = len(MODELS)
RECOVERY_TASKS = len(VARIANTS) * len(TRACKS) * MODELS_PER_TRACK
CAPACITY_TASKS = len(TRACKS) * MODELS_PER_TRACK
EVALUATION_TASKS = len(TRACKS) * MODELS_PER_TRACK
PRECISION_TASKS = len(TRACKS) * MODELS_PER_TRACK
PRECISION_TABLES = 4
#: Update-zero loss agreement required between FP32 and the training precision (nats).
PRECISION_TOLERANCE = (0.02, 0.02)   # absolute + relative to the FP32 loss
#: The training-precision gradient must point where the FP32 gradient points. Agreeing losses do
#: not show this: a forward can match while the backward overflows or loses its small terms.
PRECISION_GRADIENT_COSINE = 0.90
#: Episodes summed to see whether precision noise averages out under gradient accumulation.
PRECISION_ACCUMULATION = 4
#: Largest inference context the benchmark gives each model per task: config/eval.yaml caps,
#: bounded by the largest table's outer-fold context (0.8 x 0.8 x rows: PD 730,176, LGD 57,932).
CAPACITY_ROWS = {"pd": {"v2": 10_000, "v2.6": 50_000, "v3": 467_312, "tabicl-v2": 467_312},
                 "lgd": {"v2": 10_000, "v2.6": 37_076, "v3": 37_076, "tabicl-v2": 37_076}}
#: Rows of the precision comparison. Agreement of FP32 and the training precision does not need
#: the full episode (BF16's TabICL failure showed at 1,000 rows), and an FP32 backward at 26,000
#: rows may not fit next to its own graph.
PRECISION_ROWS = 8192


def receipts() -> Path:
    return manifests_dir("experiment0") / "pilots"


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
    out = receipts() / name
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, record)
    return out


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
    result = dict(task=task, variant=variant, track=track, trial=trial, interrupt_after=stop,
                  execution=execution, weights=weights, monitors=monitors, audit_problems=problems,
                  passed=bool(weights["passed"] and monitors["passed"] and not problems))
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
        record.update(passed=True, seconds=time.monotonic() - started, peak_bytes=peak(),
                      primary=float(metrics.get("roc_auc" if track == "pd" else "r2", float("nan"))))
    except Exception as exc:  # an out-of-memory here is the finding, not a crash
        record.update(passed=False, seconds=time.monotonic() - started, error=repr(exc)[:500],
                      peak_bytes=peak())
    _write(f"capacity_{task:02d}.json", record)
    print(json.dumps(record), flush=True)
    return record


# ---------------------------------------------- context curve + target fine-tuning (GPU)
def evaluation(task: int, folder: Path = FOLDER) -> dict:
    """E4 fine-tuning and E10 context curves through their real runner, on the reference pilot.

    For one model and track, the smallest held-out table (fold 0) is run by
    ``campaign_tasks.run`` in quick mode for both kinds, exactly as the campaign arrays will run
    it (task enumeration, checkpoint naming, fold rows, atomic output), then run again to check
    that a finished task is skipped. Scores must be finite for the original and the CPT model.
    """
    import pandas as pd

    from src.eval import campaign_tasks as ct
    track, trial = track_model_case(task, EVALUATION_TASKS)
    key = "roc_auc" if track == "pd" else "r2"
    kinds = {"context": {t: folder / f"reference_{t}.yaml" for t in TRACKS}}
    kinds["finetune"] = kinds["context"]
    if track == "pd":
        kinds["episode_context"] = {"pd": folder / "episode2048_pd.yaml"}
    record = dict(task=task, track=track, runs={}, problems=[])
    for kind, references in kinds.items():
        candidates = [(i, t) for i, t in enumerate(ct.tasks(references=references, kind=kind))
                      if t["track"] == track and t["trial"] == trial and t["fold"] == 0]
        index, chosen = min(candidates, key=lambda c: (c[1]["n_rows"], c[1]["dataset_id"]))
        record.update(model=chosen["model"])
        expected = {"cpt"} if kind == "episode_context" else {"original", "cpt"}
        started = time.monotonic()
        try:
            path = ct.run(kind, index, references=references, slug="experiment0", quick=True)
            frame = pd.read_csv(path)
            again = ct.run(kind, index, references=references, slug="experiment0", quick=True)
            scores = frame[key].to_numpy(dtype=float)
            record["runs"][kind] = dict(dataset_rows=chosen["n_rows"], rows=len(frame),
                                        starts=sorted(frame.start.unique()),
                                        finite=bool(np.isfinite(scores).all()),
                                        seconds=time.monotonic() - started, resumed_skip=again == path,
                                        scores=frame[["start", key]].to_dict("records"))
            if not (record["runs"][kind]["finite"] and set(frame.start) == expected):
                record["problems"].append(f"{kind}: missing or non-finite scores")
        except Exception as exc:  # the failure is the finding
            record["problems"].append(f"{kind}: {exc!r}"[:500])
    record["passed"] = not record["problems"]
    _write(f"evaluation_{task:02d}.json", record)
    print(json.dumps(record, default=str), flush=True)
    return record


# ------------------------------------------------------- training precision (GPU)
def _tf32(enabled: bool) -> None:
    """FP32 storage with 10-bit-mantissa matrix multiplies (tensor cores) on or off."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)


def precision(task: int, folder: Path = FOLDER) -> dict:
    """Update-zero training loss in FP32 versus the resolved training precision."""
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
    else:
        from src.train.model import load_tabpfn_for_training
        model, criterion, _, inference_config = load_tabpfn_for_training(path, track=track, device=device)
    model.train()
    refs = sorted(split_from_cfg(cfg, track=track).train, key=lambda r: -int(r.n_rows or 0))
    record = dict(task=task, track=track, base=Path(base).name, family=family,
                  training_dtype=str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
                  episode_rows=rows, members=members, tables=[])
    training_label = "fp32" if not use_amp else "fp16" if amp_dtype == torch.float16 else "bf16"
    record["training_label"] = training_label
    # Candidate precisions, each compared with FP32 on identical batches. TabICLv2 is not run
    # under BF16 (its forward is corrupted there).
    candidates = [("fp32", None)] + ([("bf16", torch.bfloat16)] if family == "tabpfn" else []) + [
        ("fp16", torch.float16)]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    groups = sorted({".".join(n.split(".")[:2]) for n in names})

    def gradient(batch, dtype):
        """(loss, flat FP64 gradient, loss scale). FP16 trains behind GradScaler: scale the loss
        as the scaler does (65,536, halved after an overflow)."""
        scale = 65536.0 if dtype == torch.float16 else 1.0
        while True:
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dtype is not None, dtype=dtype):
                loss = batch_loss(model, batch, criterion=criterion, regression_objective=objective)
            (loss * scale).backward()
            parts = {n: (p.grad.detach().double().reshape(-1) / scale if p.grad is not None
                         else torch.zeros(p.numel(), dtype=torch.float64, device=p.device))
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

    for index, ref in enumerate(refs[:PRECISION_TABLES]):
        loaded = _load_processed_csv(ref)
        build = dict(n_total_target=rows, query_fraction=query_fraction, n_estimators=members)

        def make(seed, loaded=loaded, build=build, index=index):
            if family == "tabicl":
                return _build_tabicl_step_batch(loaded, rng=np.random.default_rng(seed), epoch_seed=seed,
                                                replica=0, preprocessing_seed=index, **build).to(device)
            return _build_ensemble_step_batch(loaded, rng=np.random.default_rng(seed),
                                              inference_config=inference_config, rng_seed=seed,
                                              **build).to(device)

        row = {"table_index": index, "rows": int(min(rows, len(loaded.y)))}
        try:
            sums = {}
            for draw in range(PRECISION_ACCUMULATION):   # different episodes of the same table
                batch = make(index + 1000 * draw)
                reference = None
                for label, dtype in candidates:
                    loss, flat, scale = gradient(batch, dtype)
                    sums[label] = flat if label not in sums else sums[label] + flat
                    if draw:
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
                    # Where the disagreement sits: cosine and share of the FP32 squared norm per module.
                    total = float(reference.square().sum().clamp_min(1e-300))
                    row[f"modules_{label}"] = {
                        g: [round(cosine(torch.cat([flat[a:b] for a, b in spans[g]]),
                                         torch.cat([reference[a:b] for a, b in spans[g]])), 3),
                            round(float(sum(reference[a:b].square().sum() for a, b in spans[g])) / total, 4)]
                        for g in groups}
                del batch
            # Does the disagreement average out over several episodes (gradient accumulation)?
            for label, _ in candidates[1:]:
                row[f"grad_cosine_{label}_x{PRECISION_ACCUMULATION}"] = cosine(sums[label], sums["fp32"])
            absolute, relative = PRECISION_TOLERANCE
            row["passed"] = bool(
                np.isfinite(row[f"loss_{training_label}"])
                and abs(row[f"loss_{training_label}"] - row["loss_fp32"]) <= absolute + relative * abs(row["loss_fp32"])
                and row["grad_finite_fp32"] and row[f"grad_finite_{training_label}"]
                and row.get(f"grad_cosine_{training_label}", 1.0) >= PRECISION_GRADIENT_COSINE)
        except Exception as exc:  # an out-of-memory or numerical error is the finding, not a crash
            row.update(passed=False, error=repr(exc)[:500])
        sums = reference = flat = batch = None   # noqa: F841 - release gradients before the next table
        model.zero_grad(set_to_none=True)
        record["tables"].append(row)
        torch.cuda.empty_cache()
    # TabICLv2 only: where the FP16 forward leaves the FP32 one, how that depends on the episode
    # size, and what running one component in FP32 (the rest in FP16) repairs and costs.
    if family == "tabicl":
        record["trace"] = []

        def forward(batch, dtype):
            """Loss and, per module in execution order, a fixed sample of its output."""
            seen, handles = {}, []

            def hook(name):
                def keep(_module, _inputs, output):
                    tensor = output[0] if isinstance(output, (tuple, list)) else output
                    if name in seen or not torch.is_tensor(tensor) or not tensor.is_floating_point():
                        return
                    flat = tensor.detach().reshape(-1)
                    index = torch.linspace(0, flat.numel() - 1, min(flat.numel(), 8192),
                                           device=flat.device).long()
                    seen[name] = (flat[index].double(), int((~torch.isfinite(flat)).sum()))
                return keep

            for name, module in model.named_modules():
                if name:
                    handles.append(module.register_forward_hook(hook(name)))
            torch.manual_seed(0)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=dtype is not None, dtype=dtype):
                loss = float(batch_loss(model, batch, criterion=criterion, regression_objective=objective))
            for handle in handles:
                handle.remove()
            return loss, seen

        def in_fp32(module):
            """Run one component outside autocast; returns the undo."""
            original = module.forward

            def wrapped(*args, **kwargs):
                cast = lambda v: v.float() if torch.is_tensor(v) and v.is_floating_point() else v  # noqa: E731
                with torch.amp.autocast("cuda", enabled=False):
                    return original(*[cast(a) for a in args], **{k: cast(v) for k, v in kwargs.items()})
            module.forward = wrapped
            return lambda: delattr(module, "forward")

        def timed(batch, dtype):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            loss, flat, scale = gradient(batch, dtype)
            torch.cuda.synchronize()
            return loss, flat, scale, round(time.monotonic() - started, 3), round(
                torch.cuda.max_memory_allocated() / 1e9, 1)

        full_rows = training_row_cap(cfg, base, family, members)
        for index, ref in enumerate(refs[:PRECISION_TABLES]):
            loaded = _load_processed_csv(ref)
            entry = {"table_index": index, "sizes": {}, "components": {}}

            def make(n, loaded=loaded, index=index):
                return _build_tabicl_step_batch(
                    loaded, rng=np.random.default_rng(index), epoch_seed=index, replica=0,
                    preprocessing_seed=index, n_total_target=n, query_fraction=query_fraction,
                    n_estimators=members).to(device)
            try:
                for n in (1024, 4096, PRECISION_ROWS, 16384, full_rows):
                    batch = make(n)
                    with torch.no_grad():
                        losses = []
                        for dtype in (None, torch.float16):
                            torch.manual_seed(0)
                            with torch.amp.autocast("cuda", enabled=dtype is not None, dtype=dtype):
                                losses.append(float(batch_loss(model, batch, criterion=criterion,
                                                               regression_objective=objective)))
                    entry["sizes"][str(n)] = [round(v, 4) for v in losses]
                    del batch
                batch = make(PRECISION_ROWS)
                _, exact = forward(batch, None)
                _, half = forward(batch, torch.float16)
                errors = [(name, float((half[name][0] - value).norm() / value.norm().clamp_min(1e-30)),
                           half[name][1]) for name, (value, _) in exact.items() if name in half]
                entry["first_divergent"] = [[n, round(e, 4), bad] for n, e, bad in errors
                                            if not e <= 0.05][:12]
                entry["non_finite_modules"] = [n for n, _, bad in errors if bad][:12]
                _, reference, _, seconds, peak = timed(batch, None)
                entry["components"]["none_fp32"] = dict(seconds=seconds, peak_gb=peak)
                for component in ("", "col_embedder", "row_interactor", "icl_predictor"):
                    undo = in_fp32(getattr(model, component)) if component else (lambda: None)
                    try:
                        loss, flat, scale, seconds, peak = timed(batch, torch.float16)
                    finally:
                        undo()
                    entry["components"][component or "all_fp16"] = dict(
                        loss=round(loss, 4), cosine=round(cosine(flat, reference), 4), scale=scale,
                        seconds=seconds, peak_gb=peak)
                del batch, reference, flat
                # Cost of the column embedder in FP32 at the real episode size.
                batch = make(full_rows)
                for label, component, dtype in (("fp32", "", None), ("fp16", "", torch.float16),
                                                ("col_embedder_fp32", "col_embedder", torch.float16)):
                    undo = in_fp32(getattr(model, component)) if component else (lambda: None)
                    try:
                        timed(batch, dtype)   # warm-up
                        loss, _, _, seconds, peak = timed(batch, dtype)
                    finally:
                        undo()
                    entry["components"][f"full_{label}"] = dict(loss=round(loss, 4), seconds=seconds,
                                                                peak_gb=peak)
            except Exception as exc:  # the trace is evidence, never a gate
                entry["error"] = repr(exc)[:500]
            batch = reference = flat = exact = half = None   # noqa: F841
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            record["trace"].append(entry)
    # What one FP32 update costs at the REAL episode size on each of the four tables, with and
    # (TabPFN) without activation checkpointing: the evidence for memory and walltime. Not a gate.
    full_rows = training_row_cap(cfg, base, family, members)
    build = dict(n_total_target=full_rows, query_fraction=query_fraction, n_estimators=members)
    _tf32(False)
    record["full_episode"] = []
    for index, ref in enumerate(refs[:PRECISION_TABLES]):
        loaded = _load_processed_csv(ref)
        entry = {"table_index": index, "rows": int(min(full_rows, len(loaded.y))),
                 "features": int(loaded.X.shape[1])}
        for label in ("fp32",) + (("fp32_checkpointed",) if family == "tabpfn" else ()):
            try:
                if family == "tabicl":
                    batch = _build_tabicl_step_batch(loaded, rng=np.random.default_rng(0), epoch_seed=0,
                                                     replica=0, preprocessing_seed=0, **build).to(device)
                else:
                    batch = _build_ensemble_step_batch(loaded, rng=np.random.default_rng(0),
                                                       inference_config=inference_config, rng_seed=0,
                                                       **build).to(device)
                seconds = []
                for _ in range(2):   # the first pass includes warm-up; report the faster
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    model.zero_grad(set_to_none=True)
                    loss = batch_loss(model, batch, criterion=criterion, regression_objective=objective,
                                      activation_checkpointing=label == "fp32_checkpointed")
                    loss.backward()
                    torch.cuda.synchronize()
                    seconds.append(time.monotonic() - started)
                entry[label] = dict(seconds=round(min(seconds), 3), loss=float(loss.detach()),
                                    peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
            except Exception as exc:  # out of memory is the answer, not a crash
                entry[label] = dict(error=repr(exc)[:200])
            loss = batch = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        record["full_episode"].append(entry)
    record["passed"] = bool(record["tables"]) and all(r["passed"] for r in record["tables"])
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
    return checks


#: Models the reference pilots' five-fold benchmark must score: 4 CPT pilots, 4 original
#: models and the classical baselines of each task.
BASELINES = {"pd": ("catboost", "logreg"), "lgd": ("catboost", "linreg")}


def benchmark(track: str, folder: Path = FOLDER, *, n_folds: int = 5) -> dict:
    """End-to-end five-fold benchmark of a reference pilot: every expected model scored on every
    held-out credit table and retention table, on all folds, with status OK."""
    from omegaconf import OmegaConf

    from src.data.retention import load_refs
    from src.train.config import load_train_config
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    from src.visualize.campaign2 import _results_frames
    base = load_train_config(config_path=str(folder / f"reference_{track}.yaml"))
    run_name = str(base.run_name)
    cfg = apply_split_index(base, 0)
    tables = {r.dataset_id for r in split_from_cfg(cfg, track=track).test}
    tables |= {r.dataset_id for r in load_refs(str(OmegaConf.select(cfg, "train.retention_panel")), track)}
    frame = _results_frames("experiment0", track)
    problems = []
    if frame.empty:
        return dict(track=track, passed=False, problems=["no benchmark results"])
    frame = frame[frame.test_dataset_id.isin(tables)]
    source = frame.model_source.astype(str)
    groups = {"cpt": frame[source.str.endswith("trained") & frame.model_name.str.contains(run_name, regex=False)],
              "original": frame[source.str.endswith("untuned")],
              "baseline": frame[source == "baseline"]}
    expected = {"cpt": 4, "original": 4, "baseline": len(BASELINES[track])}
    for group, rows in groups.items():
        names = sorted(rows.model_name.unique())
        if len(names) != expected[group]:
            problems.append(f"{group}: {len(names)} models scored, expected {expected[group]}")
        folds = rows.groupby(["model_name", "test_dataset_id"]).fold_idx.nunique()
        incomplete = folds[folds < n_folds]
        missing = {(m, t) for m in names for t in tables} - set(folds.index)
        if len(incomplete) or missing:
            problems.append(f"{group}: {len(incomplete) + len(missing)} model-table pairs lack all {n_folds} folds")
    if any(n not in set(groups["baseline"].model_name) for n in BASELINES[track]):
        problems.append(f"baselines missing: {BASELINES[track]}")
    return dict(track=track, tables=len(tables), passed=not problems, problems=problems)


def summary(folder: Path = FOLDER) -> dict:
    from src.train.config import load_train_config
    from src.utils.audit_experiment import audit
    pilots = {}
    for path in pilot_paths(folder):
        name, track = path.stem.rsplit("_", 1)
        try:
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
    for kind, count in (("recovery", RECOVERY_TASKS), ("capacity", CAPACITY_TASKS),
                        ("evaluation", EVALUATION_TASKS), ("precision", PRECISION_TASKS)):
        receipts_by_kind[kind] = {}
        for task in range(count):
            file = receipts() / f"{kind}_{task:02d}.json"
            receipts_by_kind[kind][str(task)] = (json.loads(file.read_text(encoding="utf-8"))
                                                 if file.is_file() else {"passed": False, "missing": True})
    benchmarks = {}
    for track in TRACKS:
        try:
            benchmarks[track] = benchmark(track, folder)
        except Exception as exc:  # unreadable results are a finding
            benchmarks[track] = {"passed": False, "problems": [repr(exc)[:500]]}
    receipts_by_kind["benchmark"] = benchmarks
    density = receipts() / "density_audit.json"
    result = dict(pilots=pilots, **receipts_by_kind,
                  density_audit=json.loads(density.read_text(encoding="utf-8")) if density.is_file() else None,
                  failed_pilots=sorted(k for k, v in pilots.items() if not v.get("passed")),
                  failed=sorted(f"{kind}_{t}" for kind, items in receipts_by_kind.items()
                                for t, v in items.items() if not v.get("passed")))
    result["passed"] = not result["failed_pilots"] and not result["failed"]
    _write("summary.json", result)
    print(json.dumps({k: result[k] for k in ("passed", "failed_pilots", "failed")}, indent=2), flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("recovery", "capacity", "evaluation", "precision", "summary"))
    parser.add_argument("--task", type=int)
    args = parser.parse_args(argv)
    if args.action == "summary":
        return 0 if summary()["passed"] else 1
    if args.task is None:
        parser.error(f"{args.action} needs --task")
    {"recovery": recovery, "capacity": capacity, "evaluation": evaluation,
     "precision": precision}[args.action](args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
