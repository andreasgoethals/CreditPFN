"""Chunking and caching to numpy (Stage 5 of the data pipeline).

For every sanitised dataset, this stage:

  1. **Subsamples / chunks** the dataset into one or more chunks of at
     most ``cfg.dataset.max_rows_per_chunk`` rows each. Datasets larger
     than that limit become multiple chunks (and therefore multiple
     ``.npz`` files); each chunk later acts as one independent dataset
     for the multi-table fine-tuning DataLoader.

     * Classification (PD): **stratified** — each chunk preserves
       parent-class proportions via ``StratifiedKFold``.
     * Regression (LGD): **random shuffle**, no stratification.

     This matches TabPFN's own ``shuffle_and_chunk_data`` semantics in
     ``repositories/TabPFN .txt`` lines 17702–17761.

  2. **Splits each chunk** into a context / query partition,
     ``cfg.dataset.context_fraction`` of rows in context (default 0.60)
     and the remainder in query, per Garg et al. 2025 §4.

  3. **Ordinal-encodes categoricals lazily** at cache-write time using
     ``OrdinalEncoder(handle_unknown="use_encoded_value",
     unknown_value=-1, encoded_missing_value=np.nan)`` — the canonical
     idiom from TabPFN's own finetuning examples.

  4. **Writes one ``.npz`` per chunk** under
     ``data/cached/{track}/<dataset_id>/chunk_NNN.npz`` with arrays
     ``X_context``, ``y_context``, ``X_query``, ``y_query``,
     ``categorical_idx``. A small ``meta.json`` sidecar in the same
     folder records the dataset-level metadata (task type, track,
     chunk count, seed, semantic manifest-row hash, processed-file
     hash, dataset-config hash, cache fingerprint).

Public entry point
------------------
``main(cfg) -> int``

    Reads
    -----
    * ``cfg.paths.processed/{pd,lgd}/<id>.sanitized.csv``
    * ``cfg.paths.manifest_pd`` / ``manifest_lgd``

    Writes
    ------
    * ``cfg.paths.cached/{pd,lgd}/<id>/chunk_NNN.npz``
    * ``cfg.paths.cached/{pd,lgd}/<id>/meta.json``

    Returns
    -------
    int
        ``0`` on success, ``1`` if any dataset failed (logged).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import DATASET_METADATA
from src.utils.paths import resolve_data_path

LOGGER = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = 2
_MAX_RANDOM_SEED = 2**32 - 1


# =============================================================================
# Pure helpers
# =============================================================================


def _stable_int_hash(text: str, *, modulo: int = 100_000) -> int:
    """Return a process-stable integer hash for seed derivation."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % modulo


def _dataset_seed(base_seed: int, dataset_id: str) -> int:
    """Return a deterministic seed for a dataset across Python processes."""
    return (int(base_seed) + _stable_int_hash(dataset_id)) % _MAX_RANDOM_SEED


def _shuffle_and_chunk_indices(
    n_rows: int, y: np.ndarray | None,
    *,
    max_rows_per_chunk: int, min_chunk_size: int,
    equal_split_size: bool, stratify: bool, seed: int,
) -> list[np.ndarray]:
    """Return a list of integer-index arrays partitioning [0, n_rows).

    Mirrors TabPFN's ``shuffle_and_chunk_data`` semantics — see
    ``repositories/TabPFN .txt`` lines 17702–17761.

    * If ``stratify=True`` and ``y`` is supplied, uses StratifiedKFold
      so each chunk preserves class proportions.
    * Else: a single shuffle then equal-or-fixed-size partition.
    * The remainder chunk is dropped if its size is < ``min_chunk_size``.
    """
    rng = np.random.default_rng(seed)

    if n_rows <= max_rows_per_chunk:
        # One chunk; just shuffle.
        idx = np.arange(n_rows)
        rng.shuffle(idx)
        return [idx]

    n_chunks = (n_rows + max_rows_per_chunk - 1) // max_rows_per_chunk

    if stratify and y is not None and len(np.unique(y[~pd.isna(y)])) >= 2:
        from sklearn.model_selection import StratifiedKFold
        # StratifiedKFold needs integer labels and no NaNs.
        labels = y.astype(np.int64)
        skf = StratifiedKFold(
            n_splits=n_chunks, shuffle=True, random_state=seed,
        )
        chunks: list[np.ndarray] = []
        for _, idx in skf.split(np.zeros(len(labels)), labels):
            chunks.append(np.asarray(idx, dtype=np.int64))
    else:
        all_idx = np.arange(n_rows)
        rng.shuffle(all_idx)
        if equal_split_size:
            chunks = [
                np.asarray(c, dtype=np.int64)
                for c in np.array_split(all_idx, n_chunks)
            ]
        else:
            chunks = []
            for start in range(0, n_rows, max_rows_per_chunk):
                chunks.append(all_idx[start:start + max_rows_per_chunk])

    # Drop chunks that fell below min_chunk_size (matches TabPFN).
    chunks = [c for c in chunks if len(c) >= min_chunk_size]
    return chunks


def _split_context_query(
    n: int, context_fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (context_idx, query_idx). Context first, query second."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_ctx = int(round(n * context_fraction))
    n_ctx = max(1, min(n - 1, n_ctx))
    return perm[:n_ctx], perm[n_ctx:]


def _ordinal_encode_categoricals(
    X: pd.DataFrame, categorical_columns: list[str],
    *, fit_indices: np.ndarray, unknown_value: int,
    missing_value_sentinel: float,
) -> tuple[np.ndarray, list[int]]:
    """Ordinal-encode the named categorical columns; return ``(X_array,
    cat_positions)``.

    The encoder is **fit on the rows indexed by ``fit_indices``** (which
    will be the context split) and then transformed against the whole
    chunk. This faithfully reproduces TabPFN's inference scenario: any
    category that appears in the query split but never in the context
    split is encoded as ``unknown_value`` (default ``-1``), exactly like
    a real test-time row would be. Fitting on the whole chunk would leak
    query categories back into the encoder's vocabulary and over-state
    the model's ability to handle unseen categories.
    """
    from sklearn.preprocessing import OrdinalEncoder

    cols = list(X.columns)
    cat_positions = [cols.index(c) for c in categorical_columns if c in cols]
    if not cat_positions:
        return X.to_numpy(dtype=np.float32, na_value=np.nan), []

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_value,
        encoded_missing_value=missing_value_sentinel,
    )
    cat_arr = X.iloc[:, cat_positions].astype(object)
    cat_arr = cat_arr.where(cat_arr.notna(), other=np.nan)
    # Fit on context rows only, then transform the whole chunk.
    encoder.fit(cat_arr.iloc[fit_indices])
    encoded = encoder.transform(cat_arr)

    out = X.to_numpy(dtype=object).copy()
    for write_pos, src_pos in enumerate(cat_positions):
        out[:, src_pos] = encoded[:, write_pos]
    out = out.astype(np.float32)
    return out, cat_positions


def _to_jsonable(value):
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(value, (DictConfig, ListConfig)):
            return _to_jsonable(OmegaConf.to_container(value, resolve=True))
    except ImportError:
        pass

    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_to_jsonable(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return {
            str(k): _to_jsonable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return value


def _json_hash(payload) -> str:
    blob = json.dumps(
        _to_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _normalise_manifest_row(manifest_row: dict) -> dict:
    row = _to_jsonable(manifest_row)
    row.pop("date_added", None)
    return row


def _manifest_row_hash(manifest_row: dict) -> str:
    return _json_hash(_normalise_manifest_row(manifest_row))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_config_hash(cfg) -> str:
    dataset_cfg = _to_jsonable(cfg.dataset)
    dataset_cfg.pop("skip_if_cached", None)
    payload = {
        "seed": int(cfg.seed),
        "dataset": dataset_cfg,
    }
    return _json_hash(payload)


def _cache_fingerprint(
    manifest_row: dict,
    *,
    dataset_config_hash: str,
    processed_csv_sha256: str,
) -> str:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "manifest_row_hash": _manifest_row_hash(manifest_row),
        "dataset_config_hash": dataset_config_hash,
        "processed_csv_sha256": processed_csv_sha256,
    }
    return _json_hash(payload)


def _encoded_missing_value(value) -> float:
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "nan":
            return np.nan
        return float(stripped)
    return float(value)


def _resolve_categorical_encoding(cfg) -> tuple[int, float]:
    enc = cfg.dataset.categorical_encoding
    if enc.strategy != "ordinal":
        raise ValueError(
            "dataset.categorical_encoding.strategy must be 'ordinal'"
        )
    if enc.handle_unknown != "use_encoded_value":
        raise ValueError(
            "dataset.categorical_encoding.handle_unknown must be "
            "'use_encoded_value'"
        )
    return int(enc.unknown_value), _encoded_missing_value(
        enc.encoded_missing_value
    )


def _clear_chunk_files(out_dir: Path) -> None:
    for stale in out_dir.glob("chunk_*.npz"):
        stale.unlink()


# =============================================================================
# Per-dataset orchestrator
# =============================================================================


def materialise_dataset(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    manifest_row: dict,
    out_dir: Path,
    cfg,
    cache_fingerprint: str,
    processed_csv_sha256: str,
    dataset_config_hash: str,
) -> int:
    """Chunk → split → encode → write. Returns the number of chunks written."""
    target = manifest_row["target_column"]
    track = manifest_row["track"]
    task_type = manifest_row["task_type"]
    cats_hint = (manifest_row["categorical_columns"].split(";")
                 if manifest_row["categorical_columns"] else [])

    if target not in df.columns:
        raise ValueError(
            f"target {target!r} missing from sanitised {dataset_id}"
        )

    # X / y split.
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y_raw = df[target]
    if task_type == "classification":
        y = pd.to_numeric(y_raw, errors="coerce").astype(np.int64).to_numpy()
        stratify = cfg.dataset.stratify_classification
    else:
        y = pd.to_numeric(y_raw, errors="coerce").astype(np.float32).to_numpy()
        stratify = False

    # Resolve which surviving columns are categorical.
    cats_present = [c for c in cats_hint if c in feature_cols]
    dataset_seed = _dataset_seed(cfg.seed, dataset_id)
    unknown_value, missing_value_sentinel = _resolve_categorical_encoding(cfg)

    # Chunk indices.
    chunks = _shuffle_and_chunk_indices(
        n_rows=len(df), y=y if stratify else None,
        max_rows_per_chunk=cfg.dataset.max_rows_per_chunk,
        min_chunk_size=cfg.dataset.min_chunk_size,
        equal_split_size=cfg.dataset.equal_split_size,
        stratify=stratify,
        seed=dataset_seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_chunk_files(out_dir)

    if not chunks:
        LOGGER.warning("%s: no chunks survived min_chunk_size filter", dataset_id)
        return 0

    for chunk_idx, row_idx in enumerate(chunks):
        X_chunk = X.iloc[row_idx].reset_index(drop=True)
        y_chunk = y[row_idx]

        # Context / query split — done *before* categorical encoding so
        # the encoder can be fit on the context rows only. This makes
        # the cached chunks faithful to TabPFN's inference scenario:
        # categories that appear only in the query get the
        # ``unknown_value`` sentinel (default ``-1``), not a fresh ID.
        ctx_idx, qry_idx = _split_context_query(
            len(X_chunk),
            context_fraction=cfg.dataset.context_fraction,
            seed=(dataset_seed + chunk_idx) % _MAX_RANDOM_SEED,
        )

        # Ordinal-encode categoricals: fit on context, transform whole.
        X_arr, cat_positions = _ordinal_encode_categoricals(
            X_chunk, cats_present,
            fit_indices=ctx_idx,
            unknown_value=unknown_value,
            missing_value_sentinel=missing_value_sentinel,
        )

        out_path = out_dir / f"chunk_{chunk_idx:03d}.npz"
        np.savez_compressed(
            out_path,
            X_context=X_arr[ctx_idx].astype(np.float32),
            y_context=y_chunk[ctx_idx].astype(
                np.int64 if task_type == "classification" else np.float32
            ),
            X_query=X_arr[qry_idx].astype(np.float32),
            y_query=y_chunk[qry_idx].astype(
                np.int64 if task_type == "classification" else np.float32
            ),
            categorical_idx=np.asarray(cat_positions, dtype=np.int32),
        )

    # meta.json sidecar
    meta = {
        "dataset_id": dataset_id,
        "track": track,
        "task_type": task_type,
        "n_chunks": len(chunks),
        "n_rows_total": int(len(df)),
        "n_features": len(feature_cols),
        "categorical_idx": [feature_cols.index(c) for c in cats_present],
        "categorical_columns": cats_present,
        "context_fraction": cfg.dataset.context_fraction,
        "max_rows_per_chunk": cfg.dataset.max_rows_per_chunk,
        "seed": cfg.seed,
        "chunk_seed": dataset_seed,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_fingerprint": cache_fingerprint,
        "manifest_row_hash": _manifest_row_hash(manifest_row),
        "processed_csv_sha256": processed_csv_sha256,
        "dataset_config_hash": dataset_config_hash,
        "encoded_missing_value": (
            cfg.dataset.categorical_encoding.encoded_missing_value
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(_to_jsonable(meta), indent=2),
        encoding="utf-8",
    )
    return len(chunks)


# =============================================================================
# CLI
# =============================================================================


def _load_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load("config/data.yaml")


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _is_already_cached(out_dir: Path, cache_fingerprint: str) -> bool:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        return False
    if meta.get("cache_fingerprint") != cache_fingerprint:
        return False
    try:
        n_chunks = int(meta.get("n_chunks", -1))
    except (TypeError, ValueError):
        return False
    if n_chunks <= 0:
        return False
    expected = {f"chunk_{i:03d}.npz" for i in range(n_chunks)}
    actual = {p.name for p in out_dir.glob("chunk_*.npz")}
    return actual == expected


def main(cfg=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if cfg is None:
        cfg = _load_cfg()

    proc_root = resolve_data_path(cfg.paths.processed)
    cache_root = resolve_data_path(cfg.paths.cached)
    manifests = {
        "pd": _read_manifest(resolve_data_path(cfg.paths.manifest_pd)),
        "lgd": _read_manifest(resolve_data_path(cfg.paths.manifest_lgd)),
    }
    if any(m.empty for m in manifests.values()):
        LOGGER.error(
            "Manifests are empty. Run `python -m src.data.register` first."
        )
        return 1

    dataset_config_hash = _dataset_config_hash(cfg)
    failures = 0
    for dataset_id, meta in DATASET_METADATA.items():
        track = meta["track"]
        san_path = proc_root / track / f"{dataset_id}.sanitized.csv"
        if not san_path.exists():
            LOGGER.warning("missing sanitised file: %s — skipped", san_path)
            continue
        try:
            mrow = manifests[track]
            row = mrow[mrow["dataset_id"] == dataset_id]
            if row.empty:
                LOGGER.warning("%s: not in manifest, skipping", dataset_id)
                continue
            manifest_row = row.iloc[0].to_dict()

            out_dir = cache_root / track / dataset_id
            processed_csv_sha256 = _file_sha256(san_path)
            cache_fingerprint = _cache_fingerprint(
                manifest_row,
                dataset_config_hash=dataset_config_hash,
                processed_csv_sha256=processed_csv_sha256,
            )
            if (
                cfg.dataset.skip_if_cached
                and _is_already_cached(out_dir, cache_fingerprint)
            ):
                LOGGER.info("%s: cache up-to-date, skipped", dataset_id)
                continue

            df = pd.read_csv(san_path, low_memory=False)
            n_chunks = materialise_dataset(
                df,
                dataset_id=dataset_id,
                manifest_row=manifest_row,
                out_dir=out_dir,
                cfg=cfg,
                cache_fingerprint=cache_fingerprint,
                processed_csv_sha256=processed_csv_sha256,
                dataset_config_hash=dataset_config_hash,
            )
            LOGGER.info(
                "%-26s rows=%d  → %d chunks  in %s",
                dataset_id, len(df), n_chunks, out_dir,
            )
        except Exception as exc:
            LOGGER.error("%s failed: %s", dataset_id, exc, exc_info=True)
            failures += 1

    return 1 if failures else 0


def _parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Chunk and cache every sanitised dataset to .npz."
    ).parse_args()


if __name__ == "__main__":
    _parse_args()
    raise SystemExit(main())
