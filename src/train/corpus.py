"""Corpus-level split: assign whole *datasets* (not rows) to train / test.

This is the standard "leave-some-datasets-out" protocol for tabular
foundation models: every chunk of a given parent dataset goes to the
same bucket, so the test split never sees rows from a dataset the
model trained on.

NO VALIDATION BUCKET. We do fixed-epoch training and pick between
hyperparameter settings *post-hoc* on the test split (cf. discussion
in chat 2026-05-04 — too few datasets for a meaningful val signal).

Future-comparison contract
--------------------------
The split is a deterministic function of:

    (cached_root, track, train_fraction, test_fraction,
     multi_chunk_policy, pinned_test_dataset_ids, seed)

So every model in the future "TabPFN vs. XGBoost vs. CatBoost vs. …"
comparison must call :func:`split_corpus` with the SAME arguments to
guarantee the same train/test buckets. The convenience wrapper
:func:`split_from_cfg` reads them from a config object and is the
recommended entry point.

Public surface
--------------
* :class:`ChunkRef`     — atomic unit consumed by the loop / eval.
* :class:`CorpusSplit`  — ``train`` and ``test`` lists of ChunkRef.
* :func:`build_chunk_pool` — list every cached chunk for a track.
* :func:`split_corpus`     — deterministic train/test bucket assignment.
* :func:`split_from_cfg`   — same, but reads ``cfg.corpus`` + ``cfg.seed``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.paths import resolve_data_path

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkRef:
    """One cached chunk on disk; the atomic unit consumed by the loop."""
    dataset_id: str
    track: str               # "pd" | "lgd"
    task_type: str           # "classification" | "regression"
    chunk_path: Path
    chunk_idx: int           # 0, 1, 2, …  within parent dataset


@dataclass(frozen=True)
class CorpusSplit:
    """Output of :func:`split_corpus`."""
    train: list[ChunkRef]
    test:  list[ChunkRef]

    @property
    def summary(self) -> dict[str, int]:
        return {
            "train_chunks":   len(self.train),
            "test_chunks":    len(self.test),
            "train_datasets": len({c.dataset_id for c in self.train}),
            "test_datasets":  len({c.dataset_id for c in self.test}),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _list_chunk_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("chunk_*.npz"))


def build_chunk_pool(
    cached_root: Path | str,
    track: str,
    *,
    multi_chunk_policy: str = "all_chunks_as_separate_datasets",
) -> list[ChunkRef]:
    """Walk the cache and list every chunk for a track.

    Parameters
    ----------
    cached_root
        ``cfg.corpus.cached_dir`` (e.g. ``data/cached``).
    track
        ``"pd"`` or ``"lgd"``.
    multi_chunk_policy
        * ``"first_chunk_only"`` — keep only ``chunk_000.npz`` for each
          parent dataset; the rest are silently dropped.
        * ``"all_chunks_as_separate_datasets"`` — every chunk becomes
          its own atomic training step.

    The on-disk layout is ``{cached_root}/{track}/{dataset_id}/chunk_NNN.npz``
    plus a sibling ``meta.json`` (the latter is read for ``task_type``).
    """
    cached_root = resolve_data_path(cached_root)
    track_root = cached_root / track
    if not track_root.is_dir():
        return []

    refs: list[ChunkRef] = []
    for did_dir in sorted(track_root.iterdir()):
        if not did_dir.is_dir():
            continue
        meta_path = did_dir / "meta.json"
        if not meta_path.exists():
            LOGGER.warning("missing meta.json: %s — skipped", meta_path)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        task_type = meta.get(
            "task_type",
            "classification" if track == "pd" else "regression",
        )
        chunks = _list_chunk_files(did_dir)
        if not chunks:
            continue
        if multi_chunk_policy == "first_chunk_only":
            chunks = chunks[:1]
        elif multi_chunk_policy != "all_chunks_as_separate_datasets":
            raise ValueError(
                f"unknown multi_chunk_policy={multi_chunk_policy!r}; "
                "expected 'first_chunk_only' or "
                "'all_chunks_as_separate_datasets'"
            )
        for ci, p in enumerate(chunks):
            refs.append(ChunkRef(
                dataset_id=did_dir.name,
                track=track,
                task_type=task_type,
                chunk_path=p,
                chunk_idx=ci,
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
    cached_root: Path | str,
    *,
    track: str,
    train_fraction: float = 0.80,
    test_fraction: float = 0.20,
    multi_chunk_policy: str = "all_chunks_as_separate_datasets",
    train_dataset_ids: Sequence[str] = (),
    test_dataset_ids: Sequence[str] = (),
    seed: int = 42,
) -> CorpusSplit:
    """Build a :class:`CorpusSplit` for one track.

    Splits are by **dataset_id**, never by chunk — every chunk of a
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

    Useful for debugging (train on one dataset) and for an outer
    driver (the future split-orchestrator the user mentioned) that
    wants full control over the split.
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

    pool = build_chunk_pool(
        cached_root, track, multi_chunk_policy=multi_chunk_policy,
    )
    if not pool:
        LOGGER.warning("no cached chunks found for track=%s under %s",
                       track, cached_root)
        return CorpusSplit(train=[], test=[])

    unique_ids = sorted({r.dataset_id for r in pool})
    explicit_train = set(train_dataset_ids) & set(unique_ids)
    explicit_test  = set(test_dataset_ids)  & set(unique_ids)

    # Validate: any explicit ID that is requested but not cached is a
    # programming error; warn loudly so the user catches typos.
    for did in train_dataset_ids:
        if did not in unique_ids:
            LOGGER.warning("train_dataset_ids: %r not in cache for track=%s "
                           "— skipped", did, track)
    for did in test_dataset_ids:
        if did not in unique_ids:
            LOGGER.warning("test_dataset_ids: %r not in cache for track=%s "
                           "— skipped", did, track)

    bucket: dict[str, str] = {}
    for did in explicit_train:
        bucket[did] = "train"
    for did in explicit_test:
        bucket[did] = "test"

    # Whatever's left after the explicit overrides → split count-wise.
    remaining = [d for d in unique_ids if d not in bucket]
    if remaining:
        # Determine which buckets still need filling. If both explicit
        # lists were provided, the remaining IDs are unused.
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

    train: list[ChunkRef] = []
    test:  list[ChunkRef] = []
    for ref in pool:
        b = bucket.get(ref.dataset_id, "unused")
        if b == "train":
            train.append(ref)
        elif b == "test":
            test.append(ref)
        # "unused" → silently dropped

    return CorpusSplit(train=train, test=test)


def split_from_cfg(cfg, *, track: str | None = None) -> CorpusSplit:
    """Apply :func:`split_corpus` using ``cfg.corpus``, ``cfg.seed``,
    and the active ``cfg.track`` (or the supplied override).

    The multi-chunk policy is fixed to ``"first_chunk_only"`` (see
    ``src/train/loop.py::MULTI_CHUNK_POLICY``); callers may override
    it by setting ``cfg.corpus.multi_chunk_policy`` directly before
    calling, but no sweep axis exists for it.
    """
    track = track or cfg.track
    corpus = cfg.corpus

    policy = (str(corpus.multi_chunk_policy)
              if hasattr(corpus, "multi_chunk_policy")
              else "first_chunk_only")

    return split_corpus(
        cached_root=corpus.cached_dir,
        track=track,
        train_fraction=float(corpus.train_fraction),
        test_fraction=float(corpus.test_fraction),
        multi_chunk_policy=policy,
        train_dataset_ids=tuple(corpus.get("train_dataset_ids", []) or ()),
        test_dataset_ids=tuple(corpus.get("test_dataset_ids", []) or ()),
        seed=int(cfg.seed),
    )
