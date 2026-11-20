"""Evaluation tasks over held-out tables, one model, table and outer fold per task:

* ``context`` (E6): original and E1 model at nested labelled-context sizes, and the E4 model (CPT
  on the matched non-credit twins, start ``noncredit_cpt``) on the same rows;
* ``finetune`` (E5): fine-tuning on the target fold from the original and from the E1 model,
  at the candidate learning rate with the best inner-validation score;
* ``episode_context`` (E12): each episode-size model at fixed contexts of 2,048 and 8,192 rows and
  the full fold, so the effect of training-episode size can be read at matched inference context
  (the original model's scores at these sizes come from ``context``, same rows).

The CPT checkpoint is always from the partition in which the table was held out, so it never saw
the table.
Context ordering, validation and test rows come from ``context_curve.fold_indices`` (nested,
class-preserving context prefixes; complete test folds). Results are one CSV per task under
``results/<slug>/`` on PROJECT, written atomically, so a resubmitted array skips finished tasks.

Experiment 0 runs the same code against the reference *pilot* (``references``) in ``quick`` mode:
three context sizes and up to eight fine-tuning epochs on the smallest held-out table.

    python -m src.eval.campaign_tasks count context
    python -m src.eval.campaign_tasks run   context --task 17
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.design import BASES, experiment_by_slug, load_design

KINDS = ("context", "finetune", "episode_context")
SLUGS = {"context": "context", "finetune": "target_finetune", "episode_context": "episode_size"}
FAMILY = {"v2": "tabpfn", "v2.6": "tabpfn", "v3": "tabpfn", "tabicl-v2": "tabicl"}
INFERENCE_VIEWS = {"tabpfn": 8, "tabicl": 8}   # config/eval.yaml: 8 views for every model
POLICY_SEEDS = {"split_seed": 99, "context_seed": 271828, "inner_validation_fraction": 0.20}
QUICK = {"context_rows": [32, 512], "epochs": 8}   # allow initial AMP scale backoff on the smallest folds


#: Start name of each E6 checkpoint source (experiment slug); only the reference tasks also score
#: the original model, which is the same for every source.
SOURCE_START = {"reference": "cpt", "domain_control": "noncredit_cpt"}


def _references(references: dict | None, source: str = "reference") -> dict:
    from src.utils.design import config_path
    return references or {track: config_path(source, track) for track in ("pd", "lgd")}


MODEL_OF_BASE = {Path(path).name: model for bases in BASES.values() for model, path in bases.items()}


def tasks(design: dict | None = None, references: dict | None = None, *, kind: str = "context",
          pilot_source: str = "reference") -> list[dict]:
    """Deterministic task list: track, model, partition, table, fold (no checkpoint I/O)."""
    from src.utils.design import config_path, experiment_by_slug
    design = design or load_design()
    if kind == "episode_context" and references is not None:   # one explicit episode-size config
        return [dict(t, arm=0) for t in _config_tasks({"pd": references["pd"]}, ("pd",))]
    if kind == "episode_context":
        out = []
        for arm in range(len(experiment_by_slug("episode_size", design)[1]["arms"])):
            out += [dict(t, arm=arm) for t in
                    _config_tasks({"pd": config_path("episode_size", "pd", arm, design)}, ("pd",))]
        return out
    if references is not None:   # Experiment 0: one explicit reference pilot
        return [dict(t, source=pilot_source) for t in _config_tasks(references, tuple(design["tasks"]))]
    sources = experiment_by_slug("context", design)[1].get("sources", ["reference"]) if kind == "context"         else ["reference"]
    return [dict(t, source=source) for source in sources
            for t in _config_tasks(_references(None, source), tuple(design["tasks"]))]


def _config_tasks(references: dict, tracks) -> list[dict]:
    from src.train.config import load_train_config, resolve_grid
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    out = []
    for track in tracks:
        base = load_train_config(config_path=str(references[track]))
        models = [MODEL_OF_BASE[Path(t[0]).name] for t in resolve_grid(base, single=False)]
        for partition in range(int(base.corpus.n_splits)):
            split = split_from_cfg(apply_split_index(base.copy(), partition))
            for ref in sorted(split.test, key=lambda r: r.dataset_id):
                for trial, model in enumerate(models):
                    for fold in range(5):
                        out.append(dict(track=track, model=model, trial=trial, partition=partition,
                                        dataset_id=ref.dataset_id, n_rows=int(ref.n_rows), fold=fold))
    return out


def cpt_checkpoint(task: dict, references: dict | None = None) -> Path:
    """The reference checkpoint of the task's partition; refuses missing or diverged runs."""
    from src.train.config import load_train_config, resolve_grid
    from src.train.loop import descriptive_name, naming_mode
    from src.utils.experiment import apply_split_index
    from src.utils.paths import resolve_staging_path
    reference = _references(references, task.get("source", "reference"))[task["track"]]
    cfg = apply_split_index(load_train_config(config_path=str(reference)), task["partition"])
    base, lr, frozen, query, accum, mode, minimum, l2sp = resolve_grid(cfg, single=False)[task["trial"]]
    name = descriptive_name(run_name=cfg.run_name, track=cfg.track, base_path=base, learning_rate=lr,
                            seed=cfg.seed, use_lora=frozen, query_fraction=query,
                            accumulate_grad_batches=accum, epoch_pass_mode=mode,
                            min_train_rows=minimum, l2sp_lambda=l2sp,
                            adaptation_mode=naming_mode(cfg, frozen))
    path = resolve_staging_path(cfg.checkpoint.trained_dir) / cfg.track / name
    provenance = Path(str(path) + ".provenance.json")
    if not path.is_file() or not provenance.is_file():
        raise FileNotFoundError(f"Reference checkpoint not available yet: {path.name}")
    saved = json.loads(provenance.read_text(encoding="utf-8"))
    if saved.get("diverged"):
        raise RuntimeError(f"Reference checkpoint diverged: {path.name}")
    from src.utils.prepare_experiment import plan_path, read_plan
    planned = read_plan(plan_path(str(cfg.run_name), str(cfg.track)))["trials"].get(
        f"{cfg.run_name}/{task['trial']}")
    if planned is None or saved.get("trial_identity", {}).get("sha256") != planned:
        raise RuntimeError(f"Checkpoint is not the prepared plan's trial: {path.name}")
    return path


def policy(model: str, design: dict, *, quick: bool = False, kind: str = "context") -> dict:
    spec = experiment_by_slug("context", design)[1]
    sizes = (list(experiment_by_slug("episode_size", design)[1]["matched_context_evaluation"][:-1])
             if kind == "episode_context" else list(spec["context_rows"]))
    return {**POLICY_SEEDS, "max_context_rows": int(spec["max_context_rows"][model]),
            "context_rows": QUICK["context_rows"] if quick else sizes,
            "include_full_available": True, "inference_views": dict(INFERENCE_VIEWS),
            "query_batch_rows": 4096, "tabicl_inference_options": {"use_amp": False, "use_fa3": False},
            "family": FAMILY[model]}


def _tasks_key() -> str:
    """Identity of a task list: the design file and the size of every processed table (the
    enumeration reads both). Cheap: one read and directory listings, no CSV parsing."""
    import hashlib

    from src.utils.design import DESIGN
    from src.utils.paths import processed_dir
    digest = hashlib.sha256(Path(DESIGN).read_bytes())
    for track in ("pd", "lgd"):
        folder = processed_dir(track)
        for path in sorted(folder.glob("*.sanitized.csv")) if folder.is_dir() else []:
            digest.update(f"{track}/{path.name}:{path.stat().st_size}".encode())
    return digest.hexdigest()


def tasks_cache(kind: str) -> Path:
    from src.utils.paths import manifests_dir
    return manifests_dir(SLUGS[kind]) / f"{kind}_tasks.json"


def write_tasks_cache(kind: str, listed: list[dict]) -> None:
    """Written by the controller before it submits an array: workers then read their task from
    it instead of parsing every credit table to enumerate the list again (~20 s per worker)."""
    from src.utils.atomic import write_json
    write_json(tasks_cache(kind), dict(key=_tasks_key(), tasks=listed))


def campaign_tasks(kind: str, design: dict | None = None) -> list[dict]:
    """The campaign task list: from the controller's cache when its key still matches."""
    path = tasks_cache(kind)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("key") == _tasks_key():
            return cached["tasks"]
    return tasks(design, kind=kind)


def run(kind: str, index: int, *, references: dict | None = None, slug: str | None = None,
        quick: bool = False, device: str | None = None, pilot_source: str = "reference",
        interrupt_finetune: bool = False) -> Path:
    from filelock import FileLock
    from src.utils.paths import results_dir
    if kind not in KINDS:
        raise ValueError(f"Unknown task kind {kind!r}")
    design = load_design()
    task = (tasks(design, references, kind=kind, pilot_source=pilot_source) if references is not None
            else campaign_tasks(kind, design))[index]
    if kind == "episode_context" and references is None:
        from src.utils.design import config_path
        references = {"pd": config_path("episode_size", "pd", task["arm"], design)}
    folder = results_dir(experiment=slug or SLUGS[kind]) / kind
    if slug == "experiment0" and references is not None:
        import os
        from src.utils import pilot_gate
        if run_id := os.environ.get(pilot_gate.RUN_ENV):
            folder = pilot_gate.directory(folder, run_id)
        folder = folder / pilot_source / ("quick" if quick else "full")
    folder.mkdir(parents=True, exist_ok=True)
    lock = nullcontext()
    if kind == "finetune":
        locks = folder / "recovery"
        locks.mkdir(exist_ok=True)
        lock = FileLock(locks / f"{index:05d}.lock")
    with lock:
        return _run_task(kind, index, task, design, references, folder, quick, device, interrupt_finetune)


def _run_task(kind, index, task, design, references, folder, quick, device, interrupt_finetune):
    from src.eval.context_curve import budgets, fold_indices, score_cell
    from src.eval.dataset_loader import load_processed_dataset
    from src.eval.target_finetune import finetune_cell
    from src.utils.atomic import write_json
    from src.utils.experiment import digest_json, file_digest
    from src.utils.paths import resolve_base_checkpoint
    target = folder / f"{kind}_{index:05d}.csv.gz"
    receipt = folder / "completed" / f"{kind}_{index:05d}.json"
    rules = policy(task["model"], design, quick=quick, kind=kind)
    family = rules["family"]
    source = task.get("source", "reference")
    starts = {SOURCE_START[source]: cpt_checkpoint(task, references)}
    if kind != "episode_context" and source == "reference":
        starts = {"original": resolve_base_checkpoint(BASES[task["track"]][task["model"]]), **starts}
    recipe = {k: experiment_by_slug("target_finetune", design)[1]["recipe"][k]
              for k in ("epochs", "learning_rates", "early_stopping_patience")}
    if quick:
        recipe["epochs"] = QUICK["epochs"]
    recipe["selection"] = "validation_roc_auc_pd_rmse_lgd"   # part of the task identity (17-11-2026)
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    identity = task_identity(kind, task, rules, recipe if kind == "finetune" else None, quick, starts)
    identity["device_name"] = _device_name(device)
    stored = target.with_name(target.name.replace(".csv.gz", ".predictions.npz"))
    work = folder / "recovery" / f"{index:05d}-{digest_json(identity)}"
    if target.exists():
        # A finished task (metrics, predictions AND receipt) is reused only for the same task,
        # checkpoints and protocol. Metrics without a receipt are an interrupted publication and
        # are recomputed; a receipt with another identity means other code or inputs: refuse.
        done = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
        if not done:
            print(f"Task {index}: metrics without a receipt (interrupted); recomputing", flush=True)
        elif done.get("identity") != identity:
            raise RuntimeError(f"{target.name} was produced under a different identity; move it aside")
        elif not stored.is_file() or (done.get("metrics_sha256"), done.get("predictions_sha256")) != (
                file_digest(target), file_digest(stored)):
            raise RuntimeError(f"{target.name} has missing or changed metrics/predictions; inspect its receipt")
        else:
            if kind == "finetune":
                _retire_work(work)
            print(f"Task {index} already complete: {target}", flush=True)
            return target
    if kind == "finetune":
        work.mkdir(exist_ok=True)
    data = load_processed_dataset(task["track"], task["dataset_id"])
    context, validation, test = fold_indices(data.y, data.task_type, task["fold"], rules)
    rows, predictions = [], {}

    def keep(row: dict, packed: dict) -> None:
        rows.append(row)
        for name, values in packed.items():
            predictions[f"{len(rows) - 1}__{name}"] = np.asarray(values)

    for start, path in starts.items():
        if kind in ("context", "episode_context"):
            for size in budgets(len(context), rules):
                metrics, packed = score_cell(data, context[:size], validation, test, path, family, rules,
                                             device=device)
                keep(dict(start=start, context_rows=size, **metrics), _unpack(packed))
        else:
            # Both starts are also scored WITHOUT fine-tuning on exactly the same rows, so the
            # effect of fine-tuning is paired within the task (same context cap and sampling).
            used = context[: rules["max_context_rows"]]
            def baseline():
                metrics, packed = score_cell(data, used, validation, test, path, family, rules, device=device)
                return metrics, _unpack(packed)
            metrics, arrays = _cached_fit(work / f"{start}-baseline.npz", baseline)
            keep(dict(start=start, finetuned=False, context_rows=len(used), **metrics), arrays)
            # One fine-tuning per candidate rate; the rate with the best inner-validation score
            # is the cell's result. Test rows never take part in the choice.
            candidates = {}
            for candidate, rate in enumerate(recipe["learning_rates"]):
                fixed = {"epochs": recipe["epochs"], "learning_rate": float(rate),
                         "early_stopping_patience": recipe["early_stopping_patience"]}
                artifact = work / f"{start}-lr{candidate}.npz"
                epoch_path = artifact.with_suffix(".pt")
                def fit():
                    return finetune_cell(data, context, validation, test, path, family, recipe=fixed,
                                         views=INFERENCE_VIEWS[family], device=device, seed=rules["context_seed"],
                                         max_context_rows=rules["max_context_rows"], recovery_path=epoch_path,
                                         stop_after_epoch=1 if interrupt_finetune and start == "original"
                                         and candidate == 1 else None)
                candidates[float(rate)] = _cached_fit(artifact, fit)
                epoch_path.unlink(missing_ok=True)
            scores = {rate: validation_score(data.task_type, data.y[validation], out["validation_prediction"])
                      for rate, (_, out) in candidates.items()}
            chosen = min(scores, key=lambda rate: (scores[rate], rate))
            metrics, out = candidates[chosen]
            primary = "roc_auc" if data.task_type == "classification" else "rmse"
            metrics = {**metrics, "selection_metric": selection_metric(data.task_type, data.y[validation]),
                       **{f"validation_score_lr{rate:g}": score for rate, score in scores.items()},
                       **{f"test_{primary}_lr{rate:g}": float(m[primary]) for rate, (m, _) in candidates.items()},
                       **{f"fit_seconds_lr{rate:g}": float(m["fit_seconds"]) for rate, (m, _) in candidates.items()},
                       **{f"optimizer_steps_lr{rate:g}": m["optimizer_steps"] for rate, (m, _) in candidates.items()},
                       **{f"resumed_at_epoch_lr{rate:g}": m["resumed_at_epoch"] for rate, (m, _) in candidates.items()},
                       # Cost of the cell = every candidate's fine-tuning (the rate search is part of it).
                       "selected_fit_seconds": float(metrics["fit_seconds"]),
                       "fit_seconds": sum(float(m["fit_seconds"]) for m, _ in candidates.values())}
            keep(dict(start=start, finetuned=True, **metrics),
                 dict(context=used, validation=validation, test=test, target=data.y[test],
                      prediction=out["prediction"], validation_target=data.y[validation],
                      validation_prediction=out["validation_prediction"]))
    frame = pd.DataFrame(rows).assign(**task, test_rows=len(test), validation_rows=len(validation),
                                      available_context=len(context), quick=quick,
                                      device_name=identity["device_name"])
    # Row i of the metrics owns the arrays "i__<name>" (indices, targets, predictions). Published in
    # order predictions -> metrics -> receipt; readers take a metrics file only with its receipt.
    pending = stored.with_suffix(".pending")
    with open(pending, "wb") as handle:
        np.savez_compressed(handle, **predictions)
    pending.replace(stored)
    pending = target.with_suffix(".pending")
    frame.to_csv(pending, index=False, compression="gzip")
    pending.replace(target)
    write_json(receipt, dict(task=task, rows=len(frame), identity=identity, predictions=stored.name,
                             metrics_sha256=file_digest(target), predictions_sha256=file_digest(stored)))
    if kind == "finetune":
        _retire_work(work)
    print(json.dumps(dict(kind=kind, index=index, rows=len(frame))), flush=True)
    return target


def _cached_fit(path, compute):
    """Atomically publish one complete start/rate, with metrics and arrays in the same ZIP."""
    import os
    if path.exists():
        with np.load(path, allow_pickle=False) as item:
            return json.loads(str(item["metrics"])), {k: item[k] for k in item.files if k != "metrics"}
    metrics, arrays = compute()
    pending = path.with_suffix(".partial")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, metrics=json.dumps(metrics, default=lambda v: v.item()), **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    return metrics, arrays


def _retire_work(work):
    """Release this task's recovery files after its final receipt has been published."""
    if work.is_dir():
        for item in work.iterdir():
            if item.is_file() and item.suffix in {".pt", ".npz", ".partial", ".pending"}:
                item.unlink()
        work.rmdir()


def _device_name(device: str) -> str:
    """The accelerator the timings were measured on (GPU hours are only comparable per device)."""
    if str(device).startswith("cuda"):
        import torch
        return torch.cuda.get_device_name(torch.device(device))
    return "cpu"


def selection_metric(task_type: str, y) -> str:
    """What chooses the E5 learning rate: validation AUC for PD, the metric the upstream
    fine-tuners also early-stop on, so stopping and rate choice optimise the same quantity
    (agreed 17-11-2026); log loss only when a one-class validation split leaves AUC undefined.
    LGD: validation RMSE (the fine-tuners early-stop on MSE)."""
    if task_type != "classification":
        return "validation_rmse"
    return "validation_roc_auc" if np.unique(np.asarray(y)).size > 1 else "validation_log_loss"


def validation_score(task_type: str, y, prediction) -> float:
    """Lower is better: minus AUC (PD; log loss if AUC is undefined) or RMSE (LGD) on the inner
    validation rows. Every candidate rate shares the rows, so all use the same criterion."""
    y = np.asarray(y)
    prediction = np.asarray(prediction, dtype=float)
    if task_type == "classification":
        positive = prediction[:, 1] if prediction.ndim == 2 else prediction
        if selection_metric(task_type, y) == "validation_roc_auc":
            from sklearn.metrics import roc_auc_score
            return -float(roc_auc_score(y, positive))
        positive = np.clip(positive, 1e-7, 1 - 1e-7)
        return float(-np.mean(np.where(y == 1, np.log(positive), np.log1p(-positive))))
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _unpack(packed: bytes) -> dict:
    import io
    with np.load(io.BytesIO(packed)) as arrays:
        return {name: arrays[name] for name in arrays.files}


def task_identity(kind, task, rules, recipe, quick, starts) -> dict:
    """Bind a result to the target bytes and evaluation implementation as well as its weights."""
    from src.utils.experiment import code_identity, environment_versions, file_digest
    from src.utils.paths import processed_dir
    dataset = processed_dir(task["track"], task["dataset_id"] + ".sanitized.csv")
    return json.loads(json.dumps(dict(kind=kind, task=task, rules=rules, recipe=recipe, quick=bool(quick),
                                      dataset_sha256=file_digest(dataset), code=code_identity(stage="eval"),
                                      versions=environment_versions(stage="eval"),
                                      checkpoints={k: file_digest(v) for k, v in starts.items()}),
                                 default=str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("count", "run"))
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("--task", type=int)
    args = parser.parse_args(argv)
    if args.action == "count":
        print(len(tasks(kind=args.kind)))
        return 0
    if args.task is None:
        parser.error("run needs --task")
    run(args.kind, args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
