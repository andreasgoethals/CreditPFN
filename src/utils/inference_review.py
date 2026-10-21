"""Isolated experiment-0 TabICL numerical audit using finished checkpoints.

No training or overwriting scientific benchmarks. Each task journals completed
inference cells in one PROJECT SQLite file; receipts and the plan live on DATA.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf

from src.utils.atomic import write_json
from src.utils.experiment import code_identity, digest_json, environment_versions, file_digest
from src.utils.paths import REPO_ROOT, manifests_dir, results_dir, checkpoints_dir

POLICIES = {
    "auto": {"use_amp": "auto", "use_fa3": "auto"},
    "fp32_no_fa3": {"use_amp": False, "use_fa3": False},
    "amp_no_fa3": {"use_amp": True, "use_fa3": False},
}
CONFIG = REPO_ROOT / "config/experiment0/inference_review.yaml"


def folder(identifier, *, project=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Invalid numerical-review identifier")
    root = results_dir(experiment="experiment0") if project else manifests_dir("experiment0")
    return root / "inference_review" / identifier


def require_campaign_versions(recorded, current):
    """Do not mistake a package upgrade for a numerical-policy or CPT effect."""
    packages = ("tabicl", "torch", "numpy", "scikit-learn", "scipy")
    changed = [name for name in packages if recorded.get(name) != current.get(name)]
    if changed:
        raise ValueError("Restore the campaign environment before this diagnostic; changed: "+", ".join(changed))


def launcher_identity():
    names = ("inference_review_cpu.slurm", "inference_review_gpu.slurm")
    return digest_json({name: (REPO_ROOT / "scripts/slurm" / name).read_text(encoding="utf-8")
                        for name in names})


def validate_gpu_inputs(tasks):
    """Check the existing Mindwell cache before spending a GPU allocation."""
    from src.utils.stage_inputs import resolve
    pointer = os.environ.get("CREDITPFN_INPUT_POINTER")
    if not pointer:
        pointer = Path(os.environ["VSC_SCRATCH_GPFS1"]) / "CreditPFN/ACTIVE_INPUTS.json"
    root = resolve(Path(pointer))
    for task in tasks:
        path = root / "data/processed/pd" / (task["dataset_id"]+".sanitized.csv")
        if file_digest(path) != task["data_sha256"]:
            raise ValueError("Prepared GPU input cache differs from the completed campaign")
    return str(root)


def prepare(identifier):
    from src.eval.context_curve import reference_checkpoint
    from src.train.tabicl_model import load_tabicl_for_training, save_finetuned_tabicl
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    if cfg["outer_folds"] != 5 or len(set(cfg["datasets"])) != 3:
        raise ValueError("This bounded audit requires three tables and five folds")
    tasks = []
    # Reuse the existing provenance/data checks without preparing any old plan.
    for partition in range(4):
        pair, refs = reference_checkpoint("pd", "tabicl", partition, cfg)
        provenance = json.loads(Path(pair["cpt"]+".provenance.json").read_text(encoding="utf-8"))
        require_campaign_versions(provenance["trial_identity"]["specification"]["versions"],
                                  environment_versions(stage="eval"))
        for ref in refs:
            if ref.dataset_id in cfg["datasets"]:
                for fold in range(5):
                    tasks.append(dict(dataset_id=ref.dataset_id, partition=partition, fold=fold,
                                      data_sha256=file_digest(ref.processed_csv), **pair))
    if len(tasks) != 15:
        raise ValueError("Review datasets do not match the existing main-study roster")
    tasks.sort(key=lambda t: (cfg["datasets"].index(t["dataset_id"]), t["fold"]))
    gpu_inputs = validate_gpu_inputs(tasks)
    original = Path(tasks[0]["base"])
    saved = checkpoints_dir("trained", "experiment0", "inference_review", identifier, "unchanged.ckpt")
    plan_file = folder(identifier) / "plan.json"
    if not saved.exists():
        model, model_cfg = load_tabicl_for_training(original, track="pd", device="cpu")
        save_finetuned_tabicl(model, model_cfg, saved)
        del model
    import torch
    a = torch.load(original, map_location="cpu", weights_only=True)
    b = torch.load(saved, map_location="cpu", weights_only=True)
    if a["state_dict"].keys() != b["state_dict"].keys() or any(
            not torch.equal(value, b["state_dict"][key]) for key, value in a["state_dict"].items()):
        raise ValueError("Unchanged-save control changed model weights")
    expected_cfg = dict(a["config"], recompute=False)
    if b["config"] != expected_cfg:
        raise ValueError("Unchanged-save control changed inference configuration")
    plan = dict(schema=1, config=cfg, tasks=tasks, gpu_inputs=gpu_inputs, code=code_identity(stage="eval"),
                launchers=launcher_identity(), versions=environment_versions(stage="eval"),
                saved=str(saved), saved_sha256=file_digest(saved),
                library_pins=subprocess.check_output(["git", "submodule", "status"], cwd=REPO_ROOT, text=True).strip())
    plan["sha256"] = digest_json(plan)
    if plan_file.exists():
        if json.loads(plan_file.read_text(encoding="utf-8")) != plan:
            raise ValueError("Review plan changed; choose a new review ID")
    else:
        write_json(plan_file, plan, exclusive=True)
    print(f"PREPARED {identifier}: 15 inference tasks; original weights verified unchanged.", flush=True)
    return plan


def read_plan(identifier, *, current=True):
    plan = json.loads((folder(identifier) / "plan.json").read_text(encoding="utf-8"))
    if plan.get("sha256") != digest_json({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("Review plan checksum failed")
    if current and (plan["code"] != code_identity(stage="eval") or plan["launchers"] != launcher_identity()
                    or plan["versions"] != environment_versions(stage="eval")):
        raise ValueError("Review code/environment changed; restore it or choose a new review ID")
    return plan


def stage_weights(paths, task, plan):
    """One bulk copy per weight file, then all repeated inference reads are node-local."""
    scratch = Path(os.environ["VSC_SCRATCH_NODE"])
    destination = scratch / f"creditpfn-review-{os.environ['SLURM_JOB_ID']}"
    destination.mkdir(parents=True, exist_ok=True)
    staged = {}
    for state, source in paths.items():
        expected = (plan if state == "saved" else task)[state+"_sha256"]
        target = destination / f"{state}-{expected}.ckpt"
        if not target.exists():
            pending = target.with_suffix(".pending")
            shutil.copyfile(source, pending)
            if file_digest(pending) != expected:
                raise ValueError("Review weight changed during staging")
            pending.replace(target)
        elif file_digest(target) != expected:
            raise ValueError("Node-local review weight changed")
        staged[state] = target
    return staged


@contextmanager
def attention_trace():
    """Count actual backend calls and input dtypes, rather than infer them from options."""
    import tabicl._model.attention as attention
    observed = {}
    patches = [(attention.F, "scaled_dot_product_attention", "torch_SDPA_dispatch")]
    if getattr(attention, "HAS_FLASH_ATTN3", False):
        patches.append((attention, "flash_attn3", "flash_attention_3"))
    originals = []
    try:
        for module, name, label in patches:
            original = getattr(module, name)
            originals.append((module, name, original))

            def wrapped(q, *args, _fn=original, _label=label, **kwargs):
                key = f"{_label}/{q.dtype}/{q.device.type}"
                observed[key] = observed.get(key, 0)+1
                return _fn(q, *args, **kwargs)

            setattr(module, name, wrapped)
        yield observed
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def review_cells(task, cfg, context_rows):
    cells = [(context_rows, policy, state, "benchmark")
             for policy in ("auto", "fp32_no_fa3") for state in ("base", "saved", "cpt")]
    if task["dataset_id"] == cfg["datasets"][0] and task["fold"] == 0:
        cells += [(size, policy, state, "boundary")
                  for size in cfg["boundary_rows"] if size <= context_rows
                  for policy in POLICIES for state in ("base", "cpt")]
    return cells


def probe_positions(count, cfg, fold):
    """Select the small diagnostic query subset independently of target labels."""
    rng = np.random.default_rng(cfg["split_seed"]+9977+fold)
    return np.sort(rng.choice(count, size=min(count, cfg["boundary_query_rows"]), replace=False))


def run_cell(data, context, validation, test, path, policy, cfg):
    import torch
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from src.eval.dataset_loader import encode_for_model
    from src.model.tabicl_models import TabICLUntuned
    a, b, c, cats = encode_for_model(data.X.iloc[context], data.X.iloc[validation], data.X.iloc[test],
                                    categorical_columns=data.categorical_columns)
    # The campaign wrapper uses the upstream default random_state=42. Preserve it.
    model = TabICLUntuned(task_type="classification", base_path=path, device="cuda",
                          n_estimators=cfg["inference_views"], inference_options=POLICIES[policy])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with attention_trace() as trace, torch.inference_mode():
        model.fit(a, data.y[context], cats, b, data.y[validation])
        resolved = tuple(bool(x) for x in model._tabicl._resolve_amp_fa3())
        # Match the benchmark's validation-then-full-test inference call sequence.
        model.predict_proba(b)
        probabilities = np.asarray(model.predict_proba(c))
    if probabilities.shape != (len(test), 2) or not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(1), 1):
        raise ValueError("Invalid review probabilities")
    if not trace:
        raise ValueError("No attention backend calls observed; inspect installed instrumentation")
    if policy == "fp32_no_fa3" and any("float32" not in key or "flash_attention_3" in key for key in trace):
        raise ValueError("Requested FP32/no-FA3 did not execute in FP32/no-FA3")
    result = dict(roc_auc=float(roc_auc_score(data.y[test], probabilities[:, 1])),
                  brier_score=float(brier_score_loss(data.y[test], probabilities[:, 1])),
                  resolved_amp=resolved[0], resolved_fa3=resolved[1], actual_attention=trace,
                  seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  gpu=torch.cuda.get_device_name(), cuda_runtime=torch.version.cuda,
                  context_sha256=digest_json(context.tolist()), test_sha256=digest_json(test.tolist()),
                  test_rows=len(test), context_rows=len(context))
    blob = io.BytesIO()
    np.savez_compressed(blob, context=context, validation=validation, test=test,
                        target=data.y[test], prediction=probabilities)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result, blob.getvalue()


def parity_metrics(a, b, *, atol, rtol):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid parity predictions")
    return dict(passed=bool(np.allclose(a, b, atol=atol, rtol=rtol)),
                max_absolute_difference=float(np.max(abs(a-b))))


def journal_summary(path, plan, index):
    """Recompute metrics and pairing from saved predictions before declaring success."""
    from sklearn.metrics import roc_auc_score, brier_score_loss
    cfg, task = plan["config"], plan["tasks"][index]
    records, arrays = [], {}
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("Corrupt inference journal")
        if db.execute("SELECT sha FROM identity").fetchall() != [(plan["sha256"],)]:
            raise ValueError("Foreign inference journal")
        for key, text, blob in db.execute("SELECT key, metrics, predictions FROM cells ORDER BY key"):
            scope, size, policy, state = key.split("/")
            row = json.loads(text)
            if (row["scope"], row["context_rows"], row["policy"], row["state"], row["fold"]) != (
                    scope, int(size), policy, state, task["fold"]):
                raise ValueError("Cell identity mismatch")
            with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                values = {k: saved[k] for k in ("context", "validation", "test", "target", "prediction")}
            for indices in (values[k] for k in ("context", "validation", "test")):
                if indices.ndim != 1 or len(indices) != len(np.unique(indices)) or (
                        indices < 0).any() or not np.issubdtype(indices.dtype, np.integer):
                    raise ValueError("Invalid row indices")
            context, validation, test = [set(values[k]) for k in ("context", "validation", "test")]
            if context & test or validation & test or context & validation:
                raise ValueError("Context/validation/test leakage")
            p, y = values["prediction"], values["target"]
            if (len(test) != row["test_rows"] or len(context) != int(size) or len(y) != len(test)
                    or p.shape != (len(y), 2) or not np.isfinite(p).all()
                    or not np.isin(y, [0, 1]).all() or not np.allclose(p.sum(1), 1)
                    or ((p < 0) | (p > 1)).any()):
                raise ValueError("Invalid saved predictions")
            if (row["context_sha256"] != digest_json(values["context"].tolist())
                    or row["test_sha256"] != digest_json(values["test"].tolist())):
                raise ValueError("Row checksum mismatch")
            scores = dict(roc_auc=roc_auc_score(y, p[:, 1]), brier_score=brier_score_loss(y, p[:, 1]))
            if any(not np.isclose(row[k], v, atol=1e-12, rtol=1e-9) for k, v in scores.items()):
                raise ValueError("Saved metric disagrees with prediction evidence")
            if not row.get("actual_attention"):
                raise ValueError("Missing actual backend evidence")
            if policy == "fp32_no_fa3" and any(
                    "float32" not in k or "flash_attention_3" in k for k in row["actual_attention"]):
                raise ValueError("FP32 policy backend evidence mismatch")
            arrays[key], records = values, records + [row]
    native = [row for row in records if row["scope"] == "benchmark"]
    sizes = {row["context_rows"] for row in native}
    if len(sizes) != 1:
        raise ValueError("Missing or mismatched benchmark contexts")
    native_size = sizes.pop()
    expected = {f"{scope}/{size}/{policy}/{state}"
                for size, policy, state, scope in review_cells(task, cfg, native_size)}
    if set(arrays) != expected:
        raise ValueError("Missing or unexpected inference cells")
    for scope, size in {(row["scope"], row["context_rows"]) for row in records}:
        cohort = [v for k, v in arrays.items() if k.startswith(f"{scope}/{size}/")]
        if any(not np.array_equal(a[key], cohort[0][key])
               for a in cohort[1:] for key in ("context", "validation", "test", "target")):
            raise ValueError("Unpaired inference inputs")
    anchor = arrays[f"benchmark/{native_size}/auto/base"]
    previous = None
    for size in cfg["boundary_rows"]:
        probe = arrays.get(f"boundary/{size}/auto/base")
        if probe is None:
            continue
        positions = probe_positions(len(anchor["test"]), cfg, task["fold"])
        if (not set(probe["context"]) <= set(anchor["context"])
                or not np.array_equal(probe["test"], anchor["test"][positions])
                or not np.array_equal(probe["target"], anchor["target"][positions])
                or (previous is not None and not np.array_equal(previous, probe["context"][:len(previous)]))):
            raise ValueError("Boundary probe changed query cohort or nested contexts")
        previous = probe["context"]
    checks = {policy: parity_metrics(arrays[f"benchmark/{native_size}/{policy}/base"]["prediction"],
                                     arrays[f"benchmark/{native_size}/{policy}/saved"]["prediction"],
                                     atol=cfg["parity_atol"], rtol=cfg["parity_rtol"])
              for policy in ("auto", "fp32_no_fa3")}
    return records, checks


def run_task(identifier, index):
    import fcntl  # VSC/Linux only; no stale PID lock survives a terminated process.
    import torch
    from src.eval.dataset_loader import load_processed_dataset
    from src.eval.benchmark import _make_outer_folds, _inner_split
    from src.eval.context_curve import context_order
    from src.utils.paths import processed_dir
    plan = read_plan(identifier)
    if index not in range(len(plan["tasks"])):
        raise ValueError("Invalid review task")
    if not torch.cuda.is_available():
        raise RuntimeError("Run the numerical audit on a VSC GPU")
    task, cfg = plan["tasks"][index], plan["config"]
    paths = {state: Path(task[state]) for state in ("base", "cpt")}
    paths["saved"] = Path(plan["saved"])
    for state, path in paths.items():
        if file_digest(path) != (plan if state == "saved" else task)[state+"_sha256"]:
            raise ValueError("Review checkpoint changed")
    if file_digest(Path(task["cpt"]+".provenance.json")) != task["provenance_sha256"]:
        raise ValueError("Review checkpoint provenance changed")
    paths = stage_weights(paths, task, plan)
    if file_digest(processed_dir("pd", task["dataset_id"]+".sanitized.csv")) != task["data_sha256"]:
        raise ValueError("Review dataset changed")
    data = load_processed_dataset("pd", task["dataset_id"])
    pool, test = list(_make_outer_folds(data.y, task_type="classification", n_folds=5,
                                      seed=cfg["split_seed"]))[task["fold"]]
    context, validation = _inner_split(pool, data.y[pool], task_type="classification",
                                      val_fraction=cfg["inner_validation_fraction"],
                                      seed=cfg["split_seed"]+task["fold"])
    if len(context) > cfg["max_context_rows"]:
        raise ValueError("Review would exceed the measured campaign cap")
    order = context_order(context, data.y, "classification", 271828+task["fold"])
    root = folder(identifier, project=True)
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / f"task_{index:02d}.sqlite"
    with (root / f"task_{index:02d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS identity (sha TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS cells (key TEXT PRIMARY KEY, metrics TEXT, predictions BLOB)")
            old = db.execute("SELECT sha FROM identity").fetchall()
            if old and old != [(plan["sha256"],)]:
                raise ValueError("Review journal belongs to another plan")
            if not old:
                db.execute("INSERT INTO identity VALUES (?)", (plan["sha256"],))
                db.commit()
            cells = review_cells(task, cfg, len(context))
            for size, policy, state, scope in cells:
                key = f"{scope}/{size}/{policy}/{state}"
                if db.execute("SELECT 1 FROM cells WHERE key=?", (key,)).fetchone():
                    continue
                selected = context if scope == "benchmark" else order[:size]
                query = test if scope == "benchmark" else test[probe_positions(len(test), cfg, task["fold"])]
                val = validation if scope == "benchmark" else validation[
                    probe_positions(len(validation), cfg, task["fold"])]
                print(f"Task {index}: {key}; {len(query)} query rows", flush=True)
                result, predictions = run_cell(data, selected, val, query, paths[state], policy, cfg)
                result.update(scope=scope, policy=policy, state=state, fold=task["fold"])
                db.execute("INSERT INTO cells VALUES (?,?,?)", (key, json.dumps(result), predictions))
                db.commit()  # Completed cells survive a timeout; never change inputs on resume.
        records, checks = journal_summary(db_path, plan, index)
    passed = all(v["passed"] for v in checks.values())
    write_json(folder(identifier) / f"task_{index:02d}.json",
               dict(plan_sha256=plan["sha256"], task=index, passed=passed, parity=checks,
                    journal_sha256=file_digest(db_path), cells=records))
    if not passed:
        raise RuntimeError("Unchanged checkpoint failed prediction parity; inspect before interpretation")
    print(f"PASS task {index}; numerical differences between policies remain measurements.", flush=True)


def inspect(identifier):
    plan = read_plan(identifier, current=False)
    rows = []
    for index in range(len(plan["tasks"])):
        path = folder(identifier) / f"task_{index:02d}.json"
        if not path.exists():
            rows.append(dict(task=index, passed=False, reason="missing completion receipt"))
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        journal = folder(identifier, project=True) / f"task_{index:02d}.sqlite"
        valid = (receipt.get("passed") is True and receipt.get("plan_sha256") == plan["sha256"]
                 and receipt.get("task") == index and journal.exists()
                 and receipt.get("journal_sha256") == file_digest(journal))
        if valid:
            records, checks = journal_summary(journal, plan, index)
            valid = records == receipt["cells"] and checks == receipt["parity"] and all(
                check["passed"] for check in checks.values())
        rows.append(dict(task=index, passed=valid))
    result = dict(plan_sha256=plan["sha256"],
                  passed=all(row["passed"] for row in rows), tasks=rows)
    write_json(folder(identifier) / "audit.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("Review incomplete or failed; retain completed cells and inspect logs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "inspect"))
    parser.add_argument("--id", required=True)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.stage == "run":
        run_task(args.id, args.task)
    else:
        {"prepare": prepare, "inspect": inspect}[args.stage](args.id)


if __name__ == "__main__":
    main()
