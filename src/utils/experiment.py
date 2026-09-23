"""Shared experiment indexing and immutable, content-based trial identity."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from omegaconf import OmegaConf

from src.train.sampling import PROTOCOL_VERSION


def apply_split_index(cfg, index: int | None):
    """Flatten training-seed × dataset-fold; keep partition randomness independent."""
    if index is None:
        return cfg
    if index < 0:
        raise ValueError("split index must be nonnegative")
    folds = OmegaConf.select(cfg, "corpus.n_folds")
    if folds:
        seeds = list(OmegaConf.select(cfg, "experiment.training_seeds", default=[int(cfg.seed)]))
        repeat, fold = divmod(index, int(folds))
        if repeat >= len(seeds):
            raise ValueError("split index exceeds training seeds × dataset folds")
        cfg.seed = int(seeds[repeat])
        cfg.corpus.fold = fold
        # Unlike legacy random draws, this seed must not change between folds.
        if OmegaConf.select(cfg, "corpus.split_seed") is None:
            raise ValueError("Balanced folds require an explicit, fixed corpus.split_seed")
    else:
        cfg.corpus.split_seed = index
    cfg.run_name = f"{cfg.run_name}_s{index:02d}"
    return cfg


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=128)
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    after = Path(path).stat()
    if (after.st_size, after.st_mtime_ns) != (size, mtime_ns):
        raise RuntimeError(f"File changed during fingerprinting: {path}")
    return h.hexdigest()


def file_digest(path: Path) -> str:
    p = path.resolve()
    stat = p.stat()
    return _file_digest(str(p), stat.st_size, stat.st_mtime_ns)


def environment_versions() -> dict:
    result = {}
    for package in ("torch", "tabpfn", "tabicl", "numpy", "scikit-learn", "scipy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "absent"
    return result


def code_identity(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parents[2]
    paths = [p for folder in ("src/train", "src/data", "src/model", "src/eval", "src/utils", "scripts")
             for p in (root / folder).rglob("*.py") if p.name != "_private_names.py"]
    return digest_json({p.relative_to(root).as_posix(): file_digest(p) for p in sorted(paths)})


def scientific_config(cfg) -> dict:
    """Exclude placement/logging settings; retain every training/evaluation choice."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    result = {k: raw[k] for k in ("seed", "track", "optimizer", "scheduler", "lora", "train", "corpus")
              if k in raw}
    result["train"] = dict(result.get("train", {}))
    for k in ("dataloader_workers", "prefetch_factor", "step_log_interval", "recovery_every_updates",
              "segment_seconds", "epoch_log_interval"):
        result["train"].pop(k, None)
    result["corpus"] = dict(result.get("corpus", {}))
    result["corpus"].pop("n_splits", None)
    result["optimizer"] = dict(result.get("optimizer", {}))
    result["optimizer"].pop("lr", None)  # the effective LR is in the trial tuple
    return result


def trial_identity(cfg, trial, *, split=None) -> dict:
    from src.train.corpus import split_from_cfg
    from src.utils.paths import resolve_base_checkpoint
    base, lr, frozen, query, accumulation, mode, min_rows, l2sp = trial
    split = split or split_from_cfg(cfg, track=str(cfg.track), min_train_rows=int(min_rows))
    datasets = {}
    for bucket in ("train", "test"):
        datasets[bucket] = [{"id": r.dataset_id, "sha256": file_digest(r.processed_csv),
                             "target": r.target_column, "categorical": list(r.categorical_columns)}
                            for r in sorted(getattr(split, bucket), key=lambda r: r.dataset_id)]
    payload = {
        "protocol": PROTOCOL_VERSION,
        "config": scientific_config(cfg),
        "trial": [Path(base).name, float(lr), bool(frozen), float(query), int(accumulation),
                  mode, int(min_rows), float(cfg.optimizer.l2sp_lambda if l2sp is None else l2sp)],
        "data_config": OmegaConf.to_container(OmegaConf.load("config/data.yaml"), resolve=True)["finetuning"],
        "base_sha256": file_digest(resolve_base_checkpoint(base)),
        "datasets": datasets,
        "code_sha256": code_identity(),
        "versions": environment_versions(),
    }
    return {"sha256": digest_json(payload), "specification": payload}
