"""Content-keyed evaluation reuse; training seeds cannot change the evaluation split."""
from __future__ import annotations

import gzip
import json
import os
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path

from omegaconf import OmegaConf

from src.utils.experiment import code_identity, digest_json, environment_versions, file_digest
from src.utils.paths import resolve_data_path, resolve_staging_path


def evaluation_key(handle, dataset_id: str, *, track: str, config: dict) -> str:
    base = getattr(handle, "base_path", None)
    base_hash = file_digest(resolve_staging_path(base)) if base else None
    data = resolve_data_path(f"data/processed/{track}/{dataset_id}.sanitized.csv")
    from src.data.retention import is_retention
    if is_retention(dataset_id):
        data = resolve_data_path(f"data/retention/{dataset_id}.csv.gz")
    clean = dict(config)
    clean.pop("train_cfg_path", None)
    clean["results"] = {"save_predictions": bool(clean.get("results", {}).get("save_predictions", False))}
    return digest_json({"schema": 1, "source": handle.source, "name": handle.name,
        "base_sha256": base_hash, "track": track, "dataset_id": dataset_id,
        "dataset_sha256": file_digest(data), "config": clean,
        "data_config": OmegaConf.to_container(OmegaConf.load("config/data.yaml"), resolve=True)["finetuning"],
        "code": code_identity(stage="eval"), "versions": environment_versions(stage="eval")})


def cache_path(base_dir, key: str) -> Path:
    return resolve_staging_path(base_dir).parent / "evaluation_cache" / f"{key}.json.gz"


def load(base_dir, key: str, *, n_folds: int):
    path = cache_path(base_dir, key)
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    rows = data.get("rows", [])
    if data.get("key") != key or len(rows) != n_folds:
        return None
    if {int(r["fold_idx"]) for r in rows if r["status"] == "OK"} != set(range(n_folds)):
        return None
    return rows


def load_predictions(base_dir, key: str):
    with gzip.open(cache_path(base_dir, key), "rt", encoding="utf-8") as stream:
        return json.load(stream).get("predictions")


def predictions_complete(rows, predictions) -> bool:
    """Require one unique prediction per test row in every successful fold."""
    if rows is None or predictions is None:
        return False
    try:
        expected = {}
        for row in rows:
            row = asdict(row) if is_dataclass(row) else row
            if row["status"] == "OK":
                expected[(row["model_name"], row["test_dataset_id"], int(row["fold_idx"]))] = int(row["n_test_rows"])
        actual = {key: set() for key in expected}
        for row in predictions:
            key = (row["model_name"], row["test_dataset_id"], int(row["fold_idx"]))
            index = int(row["row_idx"])
            if key not in actual or index in actual[key]:
                return False
            actual[key].add(index)
        return all(len(actual[key]) == count for key, count in expected.items())
    except (KeyError, TypeError, ValueError):
        return False


def save(base_dir, key: str, rows, *, n_folds: int, predictions=None) -> None:
    if len(rows) != n_folds or any(r.status != "OK" for r in rows):
        return
    if predictions is not None and not predictions_complete(rows, predictions):
        raise ValueError("Cannot cache incomplete predictions")
    path = cache_path(base_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with gzip.open(pending, "wt", encoding="utf-8") as stream:
            json.dump({"key": key, "rows": [asdict(r) for r in rows], "predictions": predictions}, stream)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)
