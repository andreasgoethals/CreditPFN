"""Inspect/fingerprint an experiment without training or submitting jobs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from omegaconf import OmegaConf

from src.utils.experiment import (
    apply_split_index, code_identity, digest_json, environment_versions, trial_identity,
)
from src.utils.paths import manifests_dir


def plan_path(run_name: str, track: str) -> Path:
    return manifests_dir() / "plans" / (re.sub(r"_s\d+$", "", run_name) + f"_{track}.json")


def read_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contents = {k: v for k, v in payload.items() if k != "sha256"}
    if payload.get("sha256") != digest_json(contents):
        raise RuntimeError("Prepared plan failed its checksum; restore the original plan")
    return payload


def check_prepared(config: Path) -> dict:
    """Cheap submission gate: no CSV/weight reads, jobs, or output writes.

    Full input checks belong in CPU preparation and each training process.
    This catches source/config/environment drift before allocating any GPU.
    """
    from scripts.train_pipeline import _load_cfg
    cfg = _load_cfg(config_path=str(config))
    payload = read_plan(plan_path(str(cfg.run_name), str(cfg.track)))
    if payload["config"] != OmegaConf.to_container(cfg, resolve=True):
        raise RuntimeError("Configuration differs from the prepared plan; prepare a fresh named phase")
    code, versions = code_identity(), environment_versions()
    data_config = OmegaConf.to_container(OmegaConf.load("config/data.yaml"), resolve=True)["finetuning"]
    identities = payload["identities"]
    if not payload["trials"] or any(key not in identities for key in payload["trials"].values()):
        raise RuntimeError("Prepared plan has missing trial identities")
    for key, spec in identities.items():
        if digest_json(spec) != key:
            raise RuntimeError("Prepared trial identity failed its checksum")
        if spec["code_sha256"] != code or spec["versions"] != versions:
            raise RuntimeError("Code/environment differs from the prepared plan; prepare a fresh named phase")
        if spec["data_config"] != data_config:
            raise RuntimeError("Data settings differ from the prepared plan; prepare a fresh named phase")
    return {"run": str(cfg.run_name), "track": str(cfg.track), "checked": True,
            "training_trials": len(payload["trials"])}


def assert_prepared(cfg, trial_index: int, identity: dict) -> None:
    path = plan_path(str(cfg.run_name), str(cfg.track))
    if not path.is_file():
        raise RuntimeError("Prepare this experiment first: python -m src.utils.prepare_experiment "
                           "--config <experiment.yaml> --write")
    payload = read_plan(path)
    expected = payload["trials"].get(f"{cfg.run_name}/{trial_index}")
    if expected != identity["sha256"]:
        raise RuntimeError("The prepared plan differs from this trial. Recheck code, data, environment "
                           "and configuration; never overwrite an active experiment's plan.")


def prepare(config: Path, *, write: bool = False, check: bool = False) -> dict:
    from scripts.train_pipeline import _load_cfg, _resolve_grid
    from src.train.corpus import split_from_cfg, _scalar_min_rows
    if write and check:
        raise ValueError("Choose write or check, not both")
    cfg = _load_cfg(config_path=str(config))
    grid = _resolve_grid(cfg, single=False)
    n_splits = int(cfg.corpus.n_splits)
    folds = int(cfg.corpus.n_folds or 0)
    seeds = list(OmegaConf.select(cfg, "experiment.training_seeds", default=[cfg.seed]))
    if folds and n_splits > folds * len(seeds):
        raise ValueError("n_splits exceeds dataset folds × training seeds")
    payload = {"schema_version": 1, "run": str(cfg.run_name), "config": OmegaConf.to_container(cfg, resolve=True),
               "trials": {}, "partitions": [], "identities": {}}
    for index in range(n_splits):
        current = apply_split_index(OmegaConf.create(OmegaConf.to_container(cfg)), index)
        split = split_from_cfg(current)
        splits_by_min_rows = {_scalar_min_rows(current.corpus.get("min_train_rows", 0)): split}
        if not split.train or not split.test:
            raise ValueError("Both training and held-out dataset sets must be nonempty")
        payload["partitions"].append({"index": index, "training_seed": int(current.seed),
            "partition_seed": int(current.corpus.split_seed), "fold": int(current.corpus.fold),
            "train": [r.dataset_id for r in split.train], "test": [r.dataset_id for r in split.test]})
        if write or check:
            for i, trial in enumerate(grid):
                min_rows = int(trial[6])
                if min_rows not in splits_by_min_rows:
                    splits_by_min_rows[min_rows] = split_from_cfg(current, min_train_rows=min_rows)
                identity = trial_identity(current, trial, split=splits_by_min_rows[min_rows])
                payload["trials"][f"{current.run_name}/{i}"] = identity["sha256"]
                payload["identities"][identity["sha256"]] = identity["specification"]
    if folds and n_splits >= folds:
        for start in range(0, n_splits, folds):
            group = payload["partitions"][start:start + folds]
            if len(group) != folds:
                raise ValueError("A complete phase cannot end halfway through a dataset partition")
            tests = [d for p in group for d in p["test"]]
            universe = set(group[0]["train"] + group[0]["test"])
            if len(tests) != len(set(tests)) or set(tests) != universe:
                raise ValueError("Dataset folds must cover every dataset exactly once per training seed")
    report = {"run": str(cfg.run_name), "track": str(cfg.track), "trials_per_partition": len(grid),
              "partitions": n_splits, "training_trials": len(grid) * n_splits,
              "training_seeds": seeds, "dataset_folds": folds,
              "test_counts": [len(p["test"]) for p in payload["partitions"]], "written": write}
    from src.utils.paths import resolve_base_checkpoint
    bases = [resolve_base_checkpoint(trial[0]) for trial in grid]
    if all(p.is_file() for p in bases):
        report["estimated_final_checkpoint_bytes"] = n_splits * sum(p.stat().st_size for p in bases)
        report["storage_estimate_note"] = "Base-file-size estimate; allow overhead plus one optimizer recovery file per active trial."
    if write or check:
        path = plan_path(str(cfg.run_name), str(cfg.track))
        payload["sha256"] = digest_json(payload)
        if check:
            if read_plan(path) != payload:
                raise RuntimeError("Prepared plan no longer matches code, inputs, environment or configuration")
            report.update(checked=True, plan=str(path))
            return report
        if path.exists() and read_plan(path) != payload:
            raise RuntimeError("An immutable plan already exists with different contents. Use a new run name.")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            from src.utils.atomic import write_json
            write_json(path, payload, exclusive=True)
        report["plan"] = str(path)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true", help="verify the entire plan including input hashes; write nothing")
    parser.add_argument("--profile-workers", type=int, nargs="+")
    args = parser.parse_args(argv)
    if args.profile_workers:
        if not args.write:
            parser.error("--profile-workers creates named pilot configs and requires --write")
        from scripts.train_pipeline import _load_cfg
        reports = []
        for workers in args.profile_workers:
            if workers < 0:
                raise ValueError("Worker profiles require explicit nonnegative worker counts")
            cfg = _load_cfg(config_path=str(args.config))
            cfg.run_name = f"{cfg.run_name}_w{workers}"
            cfg.train.dataloader_workers = workers
            folder = manifests_dir() / "pilot_configs"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{cfg.run_name}_{cfg.track}.yaml"
            OmegaConf.save(cfg, path)
            reports.append(dict(prepare(path, write=args.write), config=str(path)))
        print(json.dumps(reports, indent=2))
    else:
        print(json.dumps(prepare(args.config, write=args.write, check=args.check), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
