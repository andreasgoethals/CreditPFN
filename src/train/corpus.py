"""Corpus-level split: assign whole *datasets* to train / test.

This is the standard "leave-some-datasets-out" protocol for tabular
foundation models: every row of a given parent dataset goes to the
same bucket, so the test split never sees rows from a dataset the
model trained on.

NO VALIDATION BUCKET. We do fixed-epoch training and pick between
hyperparameter settings *post-hoc* on the test split (cf. discussion
in chat 2026-05-04 — too few datasets for a meaningful val signal).

NO `.npz` CACHE. As of 2026-05-20 the data pipeline stops at
``data/processed/{track}/<id>.sanitized.csv``. The training pipeline
loads those CSVs directly and applies the per-epoch random subsample
itself (see :mod:`src.train.dataloader`).

Future-comparison contract
--------------------------
The split is a deterministic function of:

    (manifest CSV contents, track, train_fraction, test_fraction,
     pinned_train_dataset_ids, pinned_test_dataset_ids, seed)

So every model in the future "TabPFN vs. XGBoost vs. CatBoost vs. …"
comparison must call :func:`split_corpus` with the SAME arguments to
guarantee the same train/test buckets. The convenience wrapper
:func:`split_from_cfg` reads them from a config object and is the
recommended entry point.

Public surface
--------------
* :class:`DatasetRef`   — pointer to one sanitized CSV (atomic unit).
* :class:`CorpusSplit`  — ``train`` and ``test`` lists of DatasetRef.
* :func:`build_dataset_pool` — list every (track, dataset_id) on disk.
* :func:`split_corpus`       — deterministic train/test bucket assignment.
* :func:`split_from_cfg`     — same, but reads ``cfg.corpus`` + ``cfg.seed``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.paths import resolve_data_path, resolve_output_path

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetRef:
    """One processed dataset on disk; the atomic unit consumed by the loop.

    Replaces the old `ChunkRef` (which pointed at a `.npz` chunk under
    ``data/cached/``). Every consumer that used to enumerate chunks now
    enumerates datasets, so each parent contributes EXACTLY ONE training
    step per epoch — no over-weighting of giant datasets.
    """
    dataset_id: str
    track: str               # "pd" | "lgd"
    task_type: str           # "classification" | "regression"
    target_column: str
    categorical_columns: tuple[str, ...]
    processed_csv: Path      # data/processed/{track}/{id}.sanitized.csv


@dataclass(frozen=True)
class CorpusSplit:
    """Output of :func:`split_corpus`."""
    train: list[DatasetRef]
    test:  list[DatasetRef]

    @property
    def summary(self) -> dict[str, int]:
        return {
            "train_datasets": len(self.train),
            "test_datasets":  len(self.test),
        }


# --------------------------------------------------------------------------- #
# Manifest reading
# --------------------------------------------------------------------------- #


_MANIFEST_TEMPLATE = "data/manifest_{track}.csv"
_PROCESSED_TEMPLATE = "data/processed/{track}/{dataset_id}.sanitized.csv"


def _read_manifest(track: str) -> pd.DataFrame:
    """Read ``data/manifest_{track}.csv`` from the durable output root."""
    p = resolve_output_path(_MANIFEST_TEMPLATE.format(track=track))
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str).fillna("")


def build_dataset_pool(track: str) -> list[DatasetRef]:
    """List every dataset for a track that has a sanitized CSV on disk.

    Walks the manifest CSV (the authoritative list of registered
    datasets) and only keeps rows whose sanitized CSV exists at
    ``data/processed/{track}/{id}.sanitized.csv``. Silently skips
    datasets that are in the manifest but missing their sanitized
    output — the training pipeline's `_ensure_processed` hook is
    responsible for filling those before training starts.
    """
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    df = _read_manifest(track)
    if df.empty:
        return []

    refs: list[DatasetRef] = []
    for _, row in df.iterrows():
        did = row["dataset_id"]
        csv = resolve_data_path(
            _PROCESSED_TEMPLATE.format(track=track, dataset_id=did)
        )
        if not csv.exists():
            LOGGER.warning(
                "missing sanitized CSV for %s/%s at %s — skipped",
                track, did, csv,
            )
            continue
        cats_field = row.get("categorical_columns", "") or ""
        cats = tuple(c for c in cats_field.split(";") if c)
        task_type = row.get(
            "task_type",
            "classification" if track == "pd" else "regression",
        )
        refs.append(DatasetRef(
            dataset_id=did,
            track=track,
            task_type=task_type,
            target_column=row["target_column"],
            categorical_columns=cats,
            processed_csv=csv,
        ))
    return refs


# --------------------------------------------------------------------------- #
# Bucket assignment
# --------------------------------------------------------------------------- #


def _assign_buckets(
    dataset_ids: list[str], *,
    train_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Count-based deterministic train/test split.

    Sorts the IDs (so input order doesn't matter), permutes with
    ``seed``, then slices into train / test by *count*. This
    guarantees each bucket gets ``round(N × fraction)`` datasets —
    a hash-based split can yield an empty bucket on a small corpus
    by random luck, which is unacceptable when the test bucket needs
    at least one dataset to report a metric.

    Datasets falling past ``train + test`` are mapped to the
    sentinel ``"unused"`` bucket (silently dropped from both lists).
    """
    n = len(dataset_ids)
    if n == 0:
        return {}

    # Initial counts via rounding.
    n_train = int(round(n * train_fraction))
    n_test  = int(round(n * test_fraction))

    # Guarantee at least 1 in test whenever the user asked for a
    # positive fraction — final reporting needs a test set. We shave
    # from train because train is the larger bucket.
    if test_fraction > 0 and n_test == 0 and n_train > 1:
        n_test = 1
        n_train -= 1

    # If rounding overshot, shave from train first (largest bucket),
    # then test as a last resort.
    while n_train + n_test > n and n_train > 0:
        n_train -= 1
    while n_train + n_test > n and n_test > 1:
        n_test -= 1
    n_train = max(0, n_train)
    n_test  = max(0, n_test)

    rng = np.random.default_rng(seed)
    order = sorted(dataset_ids)
    perm = rng.permutation(len(order))
    shuffled = [order[i] for i in perm]

    bucket: dict[str, str] = {}
    for did in shuffled[:n_train]:
        bucket[did] = "train"
    for did in shuffled[n_train:n_train + n_test]:
        bucket[did] = "test"
    for did in shuffled[n_train + n_test:]:
        bucket[did] = "unused"
    return bucket


# --------------------------------------------------------------------------- #
# Public splitter
# --------------------------------------------------------------------------- #


def split_corpus(
    *,
    track: str,
    train_fraction: float = 0.70,
    test_fraction: float = 0.30,
    train_dataset_ids: Sequence[str] = (),
    test_dataset_ids: Sequence[str] = (),
    seed: int = 42,
) -> CorpusSplit:
    """Build a :class:`CorpusSplit` for one track.

    Splits are by **dataset_id**, never by row — every row of a
    given parent dataset goes to the same bucket, so the test set
    never sees rows from a dataset the train set saw.

    Two modes, controlled by the explicit-list arguments:

    * **Mode A** (both lists empty) — fraction-based split.
      ``train_fraction + test_fraction`` should sum to ≤ 1.0; any
      slack is unused (lets you train on a subset of the corpus
      without code changes).

    * **Mode B** (at least one list non-empty) — explicit override.

      - ``train_dataset_ids`` non-empty → train = *exactly* these IDs.
      - ``test_dataset_ids``  non-empty → test  = *exactly* these IDs.
      - If only one list is given, the *other* bucket is filled
        count-wise from the remaining IDs using its fraction.
      - An ID may not appear in both lists.
    """
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")
    if train_fraction + test_fraction > 1.0 + 1e-9:
        raise ValueError(
            f"train+test fractions must sum ≤ 1.0; got "
            f"{train_fraction + test_fraction}"
        )

    train_dataset_ids = list(train_dataset_ids or ())
    test_dataset_ids  = list(test_dataset_ids  or ())
    overlap = set(train_dataset_ids) & set(test_dataset_ids)
    if overlap:
        raise ValueError(
            f"dataset_id(s) appear in both train_dataset_ids and "
            f"test_dataset_ids: {sorted(overlap)}"
        )

    pool = build_dataset_pool(track)
    if not pool:
        LOGGER.warning("no processed CSVs found for track=%s", track)
        return CorpusSplit(train=[], test=[])

    refs_by_id = {r.dataset_id: r for r in pool}
    unique_ids = sorted(refs_by_id.keys())
    explicit_train = set(train_dataset_ids) & set(unique_ids)
    explicit_test  = set(test_dataset_ids)  & set(unique_ids)

    # Warn loudly on typos.
    for did in train_dataset_ids:
        if did not in unique_ids:
            LOGGER.warning(
                "train_dataset_ids: %r not found on disk for track=%s — skipped",
                did, track,
            )
    for did in test_dataset_ids:
        if did not in unique_ids:
            LOGGER.warning(
                "test_dataset_ids: %r not found on disk for track=%s — skipped",
                did, track,
            )

    bucket: dict[str, str] = {}
    for did in explicit_train:
        bucket[did] = "train"
    for did in explicit_test:
        bucket[did] = "test"

    remaining = [d for d in unique_ids if d not in bucket]
    if remaining:
        need_train = not explicit_train
        need_test  = not explicit_test
        if need_train or need_test:
            count_buckets = _assign_buckets(
                remaining,
                train_fraction=train_fraction if need_train else 0.0,
                test_fraction=test_fraction   if need_test  else 0.0,
                seed=seed,
            )
            bucket.update(count_buckets)

    train: list[DatasetRef] = []
    test:  list[DatasetRef] = []
    for did in unique_ids:
        b = bucket.get(did, "unused")
        ref = refs_by_id[did]
        if b == "train":
            train.append(ref)
        elif b == "test":
            test.append(ref)
    return CorpusSplit(train=train, test=test)


def split_from_cfg(cfg, *, track: str | None = None) -> CorpusSplit:
    """Apply :func:`split_corpus` using ``cfg.corpus``, ``cfg.seed``,
    and the active ``cfg.track`` (or the supplied override)."""
    track = track or cfg.track
    corpus = cfg.corpus
    return split_corpus(
        track=track,
        train_fraction=float(corpus.train_fraction),
        test_fraction=float(corpus.test_fraction),
        train_dataset_ids=tuple(corpus.get("train_dataset_ids", []) or ()),
        test_dataset_ids=tuple(corpus.get("test_dataset_ids", []) or ()),
        seed=int(cfg.seed),
    )
