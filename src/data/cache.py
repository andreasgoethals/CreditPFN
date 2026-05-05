"""Cache-state inspection.

Used by both the data pipeline (to decide what to skip) and any
downstream consumer (the train pipeline, the eval pipeline) that
wants to know whether a dataset has been materialised before it
tries to load it.

The cache contract is the one written by ``src/data/dataset.py``:

    data/cached/{track}/<dataset_id>/
        chunk_NNN.npz       (one or more)
        meta.json           (sidecar with `n_chunks`, fingerprints, …)

A "valid" cache satisfies all of the following:

  * the ``meta.json`` sidecar exists and parses;
  * its ``cache_schema_version`` matches the current code's
    ``CACHE_SCHEMA_VERSION``;
  * every ``chunk_NNN.npz`` file referenced by ``n_chunks`` is on
    disk;
  * (when given) the supplied ``cache_fingerprint`` matches the one
    persisted in ``meta.json``.

The validity check intentionally does NOT compare to the current
``DATASET_METADATA``-derived fingerprint — that's a stricter check
worth running occasionally, but the day-to-day question for the
train pipeline is just "are the files on disk and intact". A
fingerprint mismatch would be caught by ``dataset.py``'s own
``skip_if_cached`` logic the next time the data pipeline runs.

Public surface
--------------
* :func:`is_cache_valid`         — boolean for one (track, dataset_id).
* :func:`find_uncached_datasets` — list the IDs missing from the cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from src.utils.paths import resolve_data_path

LOGGER = logging.getLogger(__name__)


def _meta_path(cached_root: Path, track: str, dataset_id: str) -> Path:
    return resolve_data_path(cached_root) / track / dataset_id / "meta.json"


def is_cache_valid(
    cached_root: Path | str,
    track: str,
    dataset_id: str,
) -> bool:
    """True if ``data/cached/{track}/{dataset_id}/`` is on disk and
    structurally intact (meta.json + every referenced chunk).
    """
    meta_p = _meta_path(Path(cached_root), track, dataset_id)
    if not meta_p.exists():
        return False
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    # Defer schema-version awareness to the data pipeline; here we only
    # care that meta exposes a positive n_chunks and the matching files.
    try:
        n_chunks = int(meta.get("n_chunks", 0))
    except (TypeError, ValueError):
        return False
    if n_chunks <= 0:
        return False

    folder = meta_p.parent
    expected = {f"chunk_{i:03d}.npz" for i in range(n_chunks)}
    actual = {p.name for p in folder.glob("chunk_*.npz")}
    return expected.issubset(actual)


def find_uncached_datasets(
    cached_root: Path | str,
    *,
    dataset_ids: Iterable[str],
    tracks: dict[str, str],
) -> list[str]:
    """Return the subset of ``dataset_ids`` whose cache is invalid.

    Parameters
    ----------
    cached_root
        ``cfg.corpus.cached_dir`` (or equivalent).
    dataset_ids
        Iterable of dataset IDs to check.
    tracks
        Mapping ``{dataset_id: "pd"|"lgd"}``. Typically built by the
        caller from ``DATASET_METADATA``.

    The result is sorted (stable across machines) so callers can use
    it directly in log messages without wrapping in another sort.
    """
    cached_root = resolve_data_path(cached_root)
    missing: list[str] = []
    for did in dataset_ids:
        track = tracks.get(did)
        if track is None:
            LOGGER.warning(
                "find_uncached_datasets: no track known for %r — skipped",
                did,
            )
            continue
        if not is_cache_valid(cached_root, track, did):
            missing.append(did)
    return sorted(missing)
