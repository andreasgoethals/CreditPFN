"""Evaluation tasks over held-out tables, one model, table and outer fold per task:

* ``context`` (E10): original and E1 model at nested labelled-context sizes;
* ``finetune`` (E4): fine-tuning on the target fold from the original and from the E1 model,
  at the candidate learning rate with the best inner-validation score;
* ``episode_context`` (E9): each episode-size model at fixed contexts of 2,048 and 8,192 rows and
  the full fold, so the effect of training-episode size can be read at matched inference context
  (the original model's scores at these sizes come from ``context``, same rows).

The CPT checkpoint is always from the partition in which the table was held out, so it never saw
the table.
Context ordering, validation and test rows come from ``context_curve.fold_indices`` (nested,
class-preserving context prefixes; complete test folds). Results are one CSV per task under
``results/<slug>/`` on PROJECT, written atomically, so a resubmitted array skips finished tasks.

Experiment 0 runs the same code against the reference *pilot* (``references``) in ``quick`` mode:
three context sizes and two fine-tuning epochs on the smallest held-out table.

    python -m src.eval.campaign_tasks count context
    python -m src.eval.campaign_tasks run   context --task 17
"""
from __future__ import annotations

import argparse
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
QUICK = {"context_rows": [32, 512], "epochs": 2}


def _references(references: dict | None) -> dict:
    from src.utils.design import config_path
    return references or {track: config_path("reference", track) for track in ("pd", "lgd")}


MODEL_OF_BASE = {Path(path).name: model for bases in BASES.values() for model, path in bases.items()}


def tasks(design: dict | None = None, references: dict | None = None, *, kind: str = "context") -> list[dict]:
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
    return _config_tasks(_references(references), tuple(design["tasks"]))


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
    reference = _references(references)[task["track"]]
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


def run(kind: str, index: int, *, references: dict | None = None, slug: str | None = None,
        quick: bool = False, device: str | None = None) -> Path:
    from src.eval.context_curve import budgets, fold_indices, score_cell
    from src.eval.dataset_loader import load_processed_dataset
    from src.eval.target_finetune import finetune_cell
    from src.utils.atomic import write_json
    from src.utils.paths import resolve_base_checkpoint, results_dir
    if kind not in KINDS:
        raise ValueError(f"Unknown task kind {kind!r}")
    design = load_design()
    task = tasks(design, references, kind=kind)[index]
    if kind == "episode_context" and references is None:
        from src.utils.design import config_path
        references = {"pd": config_path("episode_size", "pd", task["arm"], design)}
    folder = results_dir(experiment=slug or SLUGS[kind]) / kind
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{kind}_{index:05d}.csv.gz"
    receipt = folder / "completed" / f"{kind}_{index:05d}.json"
    rules = policy(task["model"], design, quick=quick, kind=kind)
    family = rules["family"]
    starts = {"cpt": cpt_checkpoint(task, references)}
    if kind != "episode_context":
        starts = {"original": resolve_base_checkpoint(BASES[task["track"]][task["model"]]), **starts}
    recipe = {k: experiment_by_slug("target_finetune", design)[1]["recipe"][k]
              for k in ("epochs", "learning_rates", "early_stopping_patience")}
    if quick:
        recipe["epochs"] = QUICK["epochs"]
    identity = task_identity(kind, task, rules, recipe if kind == "finetune" else None, quick, starts)
    if target.exists():
        # A finished task is reused only for the same task, data, checkpoints and protocol.
        done = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
        if done.get("identity") != identity:
            raise RuntimeError(f"{target.name} was produced under a different identity; move it aside")
        print(f"Task {index} already complete: {target}", flush=True)
        return target
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
            metrics, packed = score_cell(data, used, validation, test, path, family, rules, device=device)
            keep(dict(start=start, finetuned=False, context_rows=len(used), **metrics), _unpack(packed))
            # One fine-tuning per candidate rate; the rate with the best inner-validation score
            # is the cell's result. Test rows never take part in the choice.
            candidates = {}
            for rate in recipe["learning_rates"]:
                fixed = {"epochs": recipe["epochs"], "learning_rate": float(rate),
                         "early_stopping_patience": recipe["early_stopping_patience"]}
                candidates[float(rate)] = finetune_cell(
                    data, context, validation, test, path, family, recipe=fixed,
                    views=INFERENCE_VIEWS[family], device=device,
                    max_context_rows=rules["max_context_rows"])
            scores = {rate: validation_score(data.task_type, data.y[validation], out["validation_prediction"])
                      for rate, (_, out) in candidates.items()}
            chosen = min(scores, key=lambda rate: (scores[rate], rate))
            metrics, out = candidates[chosen]
            primary = "roc_auc" if data.task_type == "classification" else "rmse"
            metrics = {**metrics, "selection_metric": "validation_log_loss" if data.task_type == "classification"
                       else "validation_rmse",
                       **{f"validation_score_lr{rate:g}": score for rate, score in scores.items()},
                       **{f"test_{primary}_lr{rate:g}": float(m[primary]) for rate, (m, _) in candidates.items()},
                       "fit_seconds": sum(float(m["fit_seconds"]) for m, _ in candidates.values())}
            keep(dict(start=start, finetuned=True, **metrics),
                 dict(context=used, validation=validation, test=test, target=data.y[test],
                      prediction=out["prediction"], validation_target=data.y[validation],
                      validation_prediction=out["validation_prediction"]))
    frame = pd.DataFrame(rows).assign(**task, test_rows=len(test), validation_rows=len(validation),
                                      available_context=len(context), quick=quick)
    # Row i of the metrics owns the arrays "i__<name>" (indices, targets, predictions).
    stored = target.with_name(target.name.replace(".csv.gz", ".predictions.npz"))
    pending = stored.with_suffix(".pending")
    with open(pending, "wb") as handle:
        np.savez_compressed(handle, **predictions)
    pending.replace(stored)
    pending = target.with_suffix(".pending")
    frame.to_csv(pending, index=False, compression="gzip")
    pending.replace(target)
    write_json(receipt, dict(task=task, rows=len(frame), identity=identity, predictions=stored.name))
    print(json.dumps(dict(kind=kind, index=index, rows=len(frame))), flush=True)
    return target


def validation_score(task_type: str, y, prediction) -> float:
    """Lower is better: log loss (PD) or RMSE (LGD) on the inner validation rows."""
    y = np.asarray(y)
    prediction = np.asarray(prediction, dtype=float)
    if task_type == "classification":
        positive = np.clip(prediction[:, 1] if prediction.ndim == 2 else prediction, 1e-7, 1 - 1e-7)
        return float(-np.mean(np.where(y == 1, np.log(positive), np.log1p(-positive))))
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _unpack(packed: bytes) -> dict:
    import io
    with np.load(io.BytesIO(packed)) as arrays:
        return {name: arrays[name] for name in arrays.files}


def task_identity(kind, task, rules, recipe, quick, starts) -> dict:
    """What a stored task result depends on: task, protocol, quick mode and checkpoint bytes."""
    from src.utils.experiment import file_digest
    return json.loads(json.dumps(dict(kind=kind, task=task, rules=rules, recipe=recipe, quick=bool(quick),
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
