"""Paired context learning curves on every row of five disjoint outer test folds.

One immutable task per dataset/family/outer fold. SQLite transactions store all
predictions and completed cells in one project-storage file per task. A killed
job resumes at the next cell, never changing the test set with the context size.
"""
from __future__ import annotations

import io
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils.experiment import digest_json, file_digest, code_identity, environment_versions
from src.utils.atomic import write_json
from src.utils.paths import manifests_dir, results_dir, checkpoints_dir, resolve_base_checkpoint


def policy(path="config/experiment4/context.yaml"):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    sizes = cfg["context_rows"]
    if (cfg["experiment"] != "experiment4" or cfg["outer_folds"] != 5
            or cfg["families"] != ["tabpfn", "tabicl"] or cfg["tracks"] != ["pd", "lgd"]
            or not sizes or any(type(n) is not int for n in sizes)
            or sizes != sorted(set(sizes)) or min(sizes) < 2
            or max(sizes) > cfg["max_context_rows"] or cfg["query_batch_rows"] < 1
            or not 0 < cfg["inner_validation_fraction"] < 1):
        raise ValueError("Invalid context-only five-fold protocol")
    if any(not isinstance(cfg["inference_views"][f], int) or cfg["inference_views"][f] < 1
           for f in cfg["families"]):
        raise ValueError("Inference views must be fixed positive integers")
    return cfg


def context_order(indices, y, task_type, seed):
    """Nested random prefixes, approximately proportional, with every class present.

    Classification ordering uses context-pool labels only; query/validation labels
    never determine a selected context. No resampling with replacement.
    """
    indices = np.asarray(indices, dtype=int)
    rng = np.random.default_rng(seed)
    if task_type != "classification":
        return rng.permutation(indices)
    heads, tails, priorities = [], [], []
    for label in np.unique(y[indices]):
        group = rng.permutation(indices[y[indices] == label])
        heads.append(group[0])
        tails.extend(group[1:])
        priorities.extend((np.arange(1, len(group)) + 0.5) / len(group))
    return np.r_[rng.permutation(heads), np.asarray(tails, dtype=int)[np.argsort(priorities, kind="stable")]]


def fold_indices(y, task_type, fold, cfg):
    from src.eval.benchmark import _make_outer_folds, _inner_split
    if fold not in range(5) or len(y) < 5:
        raise ValueError("Five outer folds required")
    train, test = list(_make_outer_folds(y, task_type=task_type, n_folds=5,
                                       seed=cfg["split_seed"]))[fold]
    pool, val = _inner_split(train, y[train], task_type=task_type,
                             val_fraction=cfg["inner_validation_fraction"],
                             seed=cfg["split_seed"] + fold)
    order = context_order(pool, y, task_type, cfg["context_seed"] + fold)
    if set(order) & (set(test) | set(val)) or set(test) & set(val):
        raise AssertionError("Context/validation/test overlap")
    return order, val, test


def budgets(available, cfg):
    maximum = min(available, cfg["max_context_rows"])
    sizes = {n for n in cfg["context_rows"] if n <= maximum}
    if cfg["include_full_available"]:
        sizes.add(maximum)
    return sorted(n for n in sizes if n >= 2)


def reference_checkpoint(track, family, partition, cfg):
    """Resolve the predeclared recipe, never an observed winning score."""
    from src.train.config import load_train_config, resolve_grid
    from src.train.loop import descriptive_name
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    source = load_train_config(config_path=f"config/experiment1/{track}.yaml")
    if str(source.run_name) != cfg["source_run"]:
        raise ValueError("Source experiment identity changed")
    source = apply_split_index(source, partition)
    recipe = cfg["source_recipe"]
    candidates = [t for t in resolve_grid(source, single=False)
                  if ("tabpfn-v3-" if family == "tabpfn" else "tabicl-") in Path(t[0]).name
                  and t[1] == recipe["learning_rate"] and t[2] == recipe["frozen_backbone"]
                  and t[7] == recipe["l2sp_lambda"] and t[5] == "one_sample" and t[6] == 0]
    if len(candidates) != 1 or int(source.seed) != recipe["seed"]:
        raise ValueError("Reference recipe must resolve to exactly one trial")
    base, lr, frozen, query, acc, mode, minimum, lam = candidates[0]
    name = descriptive_name(run_name=source.run_name, track=track, base_path=base,
                            learning_rate=lr, seed=source.seed, use_lora=frozen,
                            query_fraction=query, accumulate_grad_batches=acc,
                            epoch_pass_mode=mode, min_train_rows=minimum, l2sp_lambda=lam)
    path = checkpoints_dir("trained", "experiment1", track, name)
    prov = json.loads(Path(str(path) + ".provenance.json").read_text(encoding="utf-8"))
    ident = prov["trial_identity"]
    spec = ident["specification"]
    split = split_from_cfg(source)
    expected_train, expected_test = ({r.dataset_id for r in refs} for refs in (split.train, split.test))
    if (digest_json(spec) != ident["sha256"] or prov.get("diverged")
            or prov["successful_updates"] != recipe["successful_updates"]
            or set(prov["training_datasets"]) != expected_train
            or set(prov["test_datasets"]) != expected_test
            or expected_train & expected_test
            or spec["trial"] != [Path(base).name, lr, frozen, query, acc, mode, minimum, lam]
            or spec["config"]["seed"] != recipe["seed"]):
        raise ValueError("CPT reference provenance does not match the declared recipe/partition")
    original = resolve_base_checkpoint(base)
    if file_digest(original) != spec["base_sha256"]:
        raise ValueError("CPT and original checkpoint hashes do not match")
    recorded = {d["id"]: d["sha256"] for bucket in spec["datasets"].values() for d in bucket}
    if any(recorded.get(r.dataset_id) != file_digest(r.processed_csv) for r in split.train + split.test):
        raise ValueError("Dataset content changed since CPT")
    return dict(base=str(original), cpt=str(path), base_sha256=file_digest(original),
                cpt_sha256=file_digest(path), provenance_sha256=file_digest(Path(str(path)+".provenance.json"))), split.test


def prepare(path="config/experiment4/context.yaml"):
    cfg = policy(path)
    tasks = []
    for track in cfg["tracks"]:
        for partition in range(4):
            for family in cfg["families"]:
                pair, refs = reference_checkpoint(track, family, partition, cfg)
                for ref in refs:
                    for fold in range(5):
                        tasks.append(dict(track=track, family=family, partition=partition,
                                          dataset_id=ref.dataset_id, data_sha256=file_digest(ref.processed_csv),
                                          fold=fold, **pair))
    plan = dict(schema=1, config=cfg, tasks=tasks, code=code_identity(stage="eval"),
                versions=environment_versions(stage="eval"))
    plan["sha256"] = digest_json(plan)
    destination = manifests_dir("experiment4") / "plans" / (cfg["run_name"] + ".json")
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != plan:
            raise ValueError("Immutable context plan changed; use a fresh run_name")
    else:
        write_json(destination, plan, exclusive=True)
    return destination, len(tasks)


def read_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if digest_json({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("Context plan checksum failed")
    if plan["code"] != code_identity(stage="eval") or plan["versions"] != environment_versions(stage="eval"):
        raise ValueError("Context plan code/environment changed")
    return plan


def _pack(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def predict_all(estimator, matrix, task_type, batch_rows):
    method = estimator.predict_proba if task_type == "classification" else estimator.predict
    return np.concatenate([np.asarray(method(matrix[start:start + batch_rows]))
                           for start in range(0, len(matrix), batch_rows)])


def score_cell(data, context, validation, test, path, family, cfg, *, device="cuda"):
    from src.eval.dataset_loader import encode_for_model
    from src.eval.metrics import _classification_metrics, _regression_metrics
    from src.model.tabpfn_models import _make_tabpfn, _trust_local_checkpoints
    from src.model.tabicl_models import _sanitize
    from src.train.tabicl_compat import import_tabicl_sklearn
    if len(context) > cfg["max_context_rows"]:
        raise ValueError("Context exceeds the declared, capacity-tested experiment cap")
    a, b, c, cats = encode_for_model(data.X.iloc[context], data.X.iloc[validation], data.X.iloc[test],
                                    categorical_columns=data.categorical_columns)
    kwargs = dict(device=device, n_estimators=cfg["inference_views"][family], random_state=cfg["context_seed"])
    if family == "tabpfn":
        estimator = _make_tabpfn(data.task_type, path, categorical_features_indices=cats or None,
                                 **kwargs)
    else:
        cls, reg = import_tabicl_sklearn()
        estimator = (cls if data.task_type == "classification" else reg)(
            model_path=str(path), allow_auto_download=False, **kwargs)
        a, mask = _sanitize(a)
        b, _ = _sanitize(b, dead_cols=mask)
        c, _ = _sanitize(c, dead_cols=mask)
    started = time.monotonic()
    with _trust_local_checkpoints():
        estimator.fit(a, data.y[context])
        fit_seconds = time.monotonic() - started
        prediction = predict_all(estimator, c, data.task_type, cfg["query_batch_rows"])
        val_prediction = predict_all(estimator, b, data.task_type, cfg["query_batch_rows"])
    if len(prediction) != len(test) or not np.isfinite(prediction).all() or not np.isfinite(val_prediction).all():
        raise RuntimeError("Missing or nonfinite predictions; cell not recorded as successful")
    metrics = (_classification_metrics(proba_test=prediction, y_test=data.y[test],
               proba_val=val_prediction, y_val=data.y[validation], n_classes_seen=len(np.unique(data.y[context])))
               if data.task_type == "classification" else
               _regression_metrics(pred_test=prediction, y_test=data.y[test], neg_nll=None))
    metrics.update(fit_seconds=fit_seconds, elapsed_seconds=time.monotonic()-started)
    return metrics, _pack(context=context, validation=validation, test=test, target=data.y[test],
                          prediction=prediction, validation_target=data.y[validation], validation_prediction=val_prediction)


def run_task(plan_path, index):
    from src.utils.followup_checks import require_gate
    from src.eval.dataset_loader import load_processed_dataset
    from src.train.corpus import build_dataset_pool
    require_gate()
    plan = read_plan(plan_path)
    if index not in range(len(plan["tasks"])):
        raise ValueError("Invalid context task index")
    task, cfg = plan["tasks"][index], plan["config"]
    data = load_processed_dataset(task["track"], task["dataset_id"])
    ref = next(r for r in build_dataset_pool(task["track"]) if r.dataset_id == task["dataset_id"])
    if file_digest(ref.processed_csv) != task["data_sha256"]:
        raise ValueError("Prepared dataset changed")
    for state in ("base", "cpt"):
        if file_digest(Path(task[state])) != task[state+"_sha256"]:
            raise ValueError("Prepared checkpoint changed")
    if file_digest(Path(task["cpt"]+".provenance.json")) != task["provenance_sha256"]:
        raise ValueError("Prepared CPT provenance changed")
    context, val, test = fold_indices(data.y, data.task_type, task["fold"], cfg)
    folder = results_dir(experiment="experiment4") / cfg["run_name"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"task_{index:04d}.sqlite"
    with sqlite3.connect(path, timeout=30) as db:
        db.execute("CREATE TABLE IF NOT EXISTS identity (sha TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS cells (rows INTEGER, state TEXT, metrics TEXT, predictions BLOB, PRIMARY KEY(rows,state))")
        old = db.execute("SELECT sha FROM identity").fetchall()
        if old and old != [(plan["sha256"],)]:
            raise ValueError("Results belong to another prepared context plan")
        if not old:
            db.execute("INSERT INTO identity VALUES (?)", (plan["sha256"],))
            db.commit()
        for size in budgets(len(context), cfg):
            for state in ("base", "cpt"):
                if db.execute("SELECT 1 FROM cells WHERE rows=? AND state=?", (size, state)).fetchone():
                    continue
                print(f"Context task {index}: {size} rows, {state}, {len(test)} test rows", flush=True)
                metrics, blob = score_cell(data, context[:size], val, test, task[state], task["family"], cfg)
                metrics.update(context_rows=size, available_context_rows=len(context),
                               test_rows=len(test), validation_rows=len(val), state=state,
                               context_indices_sha256=digest_json(context[:size].tolist()),
                               test_indices_sha256=digest_json(test.tolist()), **{k: task[k] for k in
                               ("track", "family", "partition", "fold", "dataset_id")})
                db.execute("INSERT INTO cells VALUES (?,?,?,?)", (size, state, json.dumps(metrics), blob))
                db.commit()
        rows = [json.loads(r[0]) for r in db.execute("SELECT metrics FROM cells ORDER BY rows,state")]
        target = folder / f"task_{index:04d}.csv.gz"
        temporary = target.with_suffix(".pending")
        pd.DataFrame(rows).to_csv(temporary, index=False, compression="gzip")
        temporary.replace(target)
    write_json(manifests_dir("experiment4") / "completed" / cfg['run_name'] / f"task_{index:04d}.json",
               dict(plan_sha256=plan["sha256"], task=index, cells=len(rows), passed=True))
