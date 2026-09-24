"""Corpus-level split: assign whole *datasets* to train / test.

This is the standard "leave-some-datasets-out" protocol for tabular
foundation models: every row of a given parent dataset goes to the
same bucket, so the test split never sees rows from a dataset the
model trained on.

NO VALIDATION BUCKET. The current study describes a prespecified grid
and update horizon. A post-hoc winner does not have an unbiased test
estimate; see docs/RESEARCH_BRIEF.md for the reporting contract.

NO `.npz` CACHE. As of 2026-05-20 the data pipeline stops at
``data/processed/{track}/<id>.sanitized.csv``. The training pipeline
loads those CSVs directly and applies the per-epoch random subsample
itself (see :mod:`src.train.dataloader`).

Future-comparison contract
--------------------------
The split is a deterministic function of:

    (registered datasets and processed metadata, track, train_fraction, test_fraction,
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
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.paths import processed_dir

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetRef:
    """One processed dataset on disk; the atomic unit consumed by the loop.

    Replaces the old `ChunkRef`. The pass mode determines whether a
    parent contributes one sampled batch or a complete set of partitions.
    """
    dataset_id: str
    track: str               # "pd" | "lgd"
    task_type: str           # "classification" | "regression"
    target_column: str
    categorical_columns: tuple[str, ...]
    processed_csv: Path      # data/processed/{track}/{id}.sanitized.csv
    #: Rows in the sanitized CSV, read from the file. Carried because corpus
    #: composition is a first-class experimental variable: `min_train_rows` filters on
    #: it, and the run log records it so a result can be read against the corpus that
    #: produced it.
    n_rows: int = 0


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
# Processed metadata inspection
# --------------------------------------------------------------------------- #


#: Processed-CSV filename. The DIRECTORY comes from `src/utils/paths.py`, the one module
#: allowed to know which storage tier it lives on. The dataset pool no longer reads a manifest
#: file (26-08-2026) — see `build_dataset_pool`.
_PROCESSED_NAME = "{dataset_id}.sanitized.csv"


@lru_cache(maxsize=256)
def _processed_metadata(csv: Path, target: str, hints: tuple[str, ...],
                        size: int, mtime_ns: int) -> tuple[tuple[str, ...], int] | None:
    """Cache only schema/counts, never full frames; invalidate changed files.

    Full CSV inspection catches categorical values appearing late in a table.
    Repeated fold/trial resolution should not repeat that expensive read.
    """
    from src.data.register import infer_categorical_numerical
    df = pd.read_csv(csv, low_memory=False)
    after = csv.stat()
    if (after.st_size, after.st_mtime_ns) != (size, mtime_ns):
        raise RuntimeError("Processed CSV changed during schema inspection")
    if target not in df.columns:
        return None
    cats, _ = infer_categorical_numerical(df, target, list(hints))
    return tuple(cats), len(df)


def build_dataset_pool(track: str) -> list[DatasetRef]:
    """Every dataset for a track that has a sanitized CSV on disk.

    Built from CODE + DATA, never from a manifest FILE. The authority is
    :data:`src.data.preprocessing.DATASET_METADATA` (track / task_type / target / categorical
    hints — always present, versioned with the code) plus the processed CSV on disk. Categorical
    columns are the code hints unioned with the string-dtype columns detected in the processed
    CSV, exactly as :func:`src.data.register.infer_categorical_numerical` computes them for the
    manifest.

    This is deliberate (26-08-2026): the registry used to live in ``output CreditPFN/manifests/`` and had
    to be rebuilt from raw (slow) or copied across whenever ``output CreditPFN/`` was wiped for a clean run.
    Now nothing under ``output CreditPFN/`` is required to START a run — that folder holds only results the
    code produces. A dataset with no processed CSV is skipped silently; the pipeline's
    ``_ensure_processed`` hook materialises missing CSVs before training.
    """
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    from src.data.preprocessing import DATASET_METADATA

    refs: list[DatasetRef] = []
    for did, meta in sorted(DATASET_METADATA.items()):
        if meta.get("track") != track:
            continue
        csv = processed_dir(track, _PROCESSED_NAME.format(dataset_id=did))
        if not csv.exists():
            LOGGER.debug("no processed CSV for %s/%s at %s — skipped", track, did, csv)
            continue
        target = meta["target_column"]
        stat = csv.stat()
        metadata = _processed_metadata(csv.resolve(), target,
            tuple(meta.get("categorical_columns", [])), stat.st_size, stat.st_mtime_ns)
        if metadata is None:
            LOGGER.warning("target %r missing from %s — skipped", target, csv)
            continue
        cats, n_rows = metadata
        refs.append(DatasetRef(
            dataset_id=did,
            track=track,
            task_type=meta.get(
                "task_type", "classification" if track == "pd" else "regression"),
            target_column=target,
            categorical_columns=tuple(cats),
            processed_csv=csv,
            n_rows=n_rows,
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


def _assign_random_split(
    dataset_ids: list[str], *, n_test: int, seed: int,
) -> dict[str, str]:
    """Draw `n_test` datasets at random as the test set; the rest train.

    One draw per `seed`, so `--split-index k` gives split k and the analysis averages over them.
    Unlike K-fold this does not guarantee every dataset is tested exactly once, but with 28 draws
    of 4 from 17 each dataset lands in a test set ~6.6 times, which is what the averaging needs.
    """
    order = sorted(dataset_ids)
    n = len(order)
    if n == 0:
        return {}
    if not 1 <= n_test < n:
        raise ValueError(f"n_test must be in [1, {n}); got {n_test}")
    rng = np.random.default_rng(seed)
    test = set(rng.choice(order, size=int(n_test), replace=False).tolist())
    return {d: ("test" if d in test else "train") for d in order}


def _assign_folds(
    dataset_ids: list[str], *,
    n_folds: int,
    fold: int,
    seed: int,
) -> dict[str, str]:
    """Dataset-level K-fold: fold `fold` is the test set, everything else trains.

    Folds rather than repeated random draws, because with random draws over a 17-dataset corpus
    some datasets never land in a test set and others land there repeatedly — the effect is then
    estimated on an arbitrary subset and "how much does this depend on the split" stays
    unanswerable. K folds cover every dataset exactly once, which is what turns n_test = 5 into an
    effect estimated on all 17.

    NO dataset is dropped: the training set is the complement of the test fold, so a K-fold run
    also trains on MORE tables than the old 70/30 draw (13 of 17 rather than 12). `seed` selects
    the partition, so R repeats at different seeds give R x K configurations.
    """
    order = sorted(dataset_ids)
    n = len(order)
    if n == 0:
        return {}
    if not 2 <= n_folds <= n:
        raise ValueError(f"n_folds must be in [2, corpus size = {n}]; got {n_folds}")
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}); got {fold}")

    rng = np.random.default_rng(seed)
    shuffled = [order[i] for i in rng.permutation(n)]
    # Contiguous blocks over the shuffled order, sizes differing by at most one.
    edges = [round(k * n / n_folds) for k in range(n_folds + 1)]
    test = set(shuffled[edges[fold]:edges[fold + 1]])
    return {did: ("test" if did in test else "train") for did in order}


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
    n_folds: int | None = None,
    fold: int = 0,
    n_test_datasets: int | None = None,
    min_train_rows: int = 0,
    require_complete_registry: bool = False,
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
    if require_complete_registry:
        from src.data.preprocessing import DATASET_METADATA
        expected = {d for d, m in DATASET_METADATA.items() if m["track"] == track}
        actual = {r.dataset_id for r in pool}
        if actual != expected:
            raise ValueError(f"Incomplete {track} corpus: {len(actual)} available, {len(expected)} registered. "
                             "Restore the confirmed dataset inventory before splitting; no silent omissions.")
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
            # K-FOLD when `n_folds` is set, otherwise the historic fraction draw. The
            # fraction path is kept so every run before run-9 stays reproducible.
            if n_test_datasets:
                count_buckets = _assign_random_split(
                    remaining, n_test=int(n_test_datasets), seed=seed,
                )
            elif n_folds:
                count_buckets = _assign_folds(
                    remaining, n_folds=int(n_folds), fold=int(fold), seed=seed,
                )
            else:
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

    # TRAIN-SIDE ONLY size filter. The test set is never touched: dropping small test
    # datasets would change what "held-out performance" means between two trials of the
    # same sweep, which is the one thing that must stay fixed.
    #
    # A size-filter ablation is motivated by Garg et al.'s corpus/context
    # sensitivity, but their classification results do not establish the
    # effect for this credit corpus or for LGD. It is inactive in the main grid.
    if min_train_rows:
        kept = [r for r in train if (r.n_rows or 0) >= int(min_train_rows)]
        dropped = [r for r in train if r not in kept]
        if dropped:
            LOGGER.info(
                "corpus.min_train_rows=%d: dropping %d of %d TRAINING datasets below the "
                "threshold (%s). Test set unchanged (%d datasets).",
                int(min_train_rows), len(dropped), len(train),
                ", ".join(f"{r.dataset_id}:{r.n_rows}" for r in dropped), len(test),
            )
        if not kept:
            raise ValueError(f"corpus.min_train_rows={min_train_rows} removed every training dataset "
                             f"for track={track}; refusing to ignore the requested filter")
        train = kept
    return CorpusSplit(train=train, test=test)


def resolve_ids_for_track(raw, track: str) -> tuple[str, ...]:
    """Resolve an explicit-ID config value (``train_dataset_ids`` /
    ``test_dataset_ids``) for one track.

    Accepts either:

    * a **flat list** — applies to whichever track is active (back-compat); or
    * a **per-track mapping** ``{"pd": [...], "lgd": [...]}`` — so the shared
      ``config/train.yaml`` can pin different IDs per track without the
      *other* track tripping the explicit-ID validation. A missing track
      key resolves to "no pins" (fall back to the fraction split).

    Returns a tuple of IDs (empty when nothing applies to this track).
    """
    if raw is None:
        return ()
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:                                          # pragma: no cover
        pass
    if isinstance(raw, dict):
        return tuple(raw.get(track, []) or ())
    return tuple(raw or ())


def _scalar_min_rows(raw) -> int:
    """0 for a swept list, the value for a scalar. See `split_from_cfg`."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:                                          # pragma: no cover
        pass
    if isinstance(raw, (list, tuple)):
        return 0
    return int(raw or 0)


def split_from_cfg(cfg, *, track: str | None = None,
                   min_train_rows: int | None = None) -> CorpusSplit:
    """Apply :func:`split_corpus` using ``cfg.corpus``, ``cfg.seed``,
    and the active ``cfg.track`` (or the supplied override).

    ``train_dataset_ids`` / ``test_dataset_ids`` may be a flat list or a
    per-track mapping — see :func:`resolve_ids_for_track`.

    ``min_train_rows`` overrides the value read from ``cfg.corpus`` — training passes the
    per-trial swept value here. This is THE single split path: ``train_one_config`` and the eval
    pipeline both call it, so they cannot disagree on the held-out datasets (they did until
    26-08-2026, when training called ``split_corpus`` directly and dropped ``n_test_datasets`` /
    ``split_seed`` / ``n_folds`` / ``fold`` — every split then trained on the same seed-42 draw
    while eval scored a different one).
    """
    track = track or cfg.track
    corpus = cfg.corpus
    return split_corpus(
        track=track,
        require_complete_registry=bool(corpus.get("require_complete_registry", False)),
        # Optional since run-9: a config that sets `n_test_datasets` or `n_folds` never uses
        # the fractions, and requiring them forces dead knobs into every experiment file.
        train_fraction=float(corpus.get("train_fraction", 0.70)),
        test_fraction=float(corpus.get("test_fraction", 0.30)),
        train_dataset_ids=resolve_ids_for_track(
            corpus.get("train_dataset_ids", None), track),
        test_dataset_ids=resolve_ids_for_track(
            corpus.get("test_dataset_ids", None), track),
        # SPLIT SEED, not the training seed. One seed used to drive both the dataset
        # partition and weight initialisation, so changing the split also changed the init
        # and the two effects were confounded. Defaults to cfg.seed so old runs reproduce.
        # `or` would be wrong here: split_seed=0 is a LEGITIMATE value (it is split 0
        # of n_splits, set by src.utils.experiment.apply_split_index) and `0 or 42` is 42 in
        # Python. That silently made split 0 draw the same datasets as a no-split run,
        # collapsing an 8-split campaign to 7 distinct draws with nothing in the output
        # to show it. Test for None explicitly.
        seed=int(cfg.seed if corpus.get("split_seed", None) is None
                 else corpus.get("split_seed")),
        n_folds=corpus.get("n_folds", None),
        n_test_datasets=corpus.get("n_test_datasets", None),
        fold=int(corpus.get("fold", 0) or 0),
        # A LIST here means "swept" (config/train.yaml since run-8). Outside a single trial
        # there is no one value, so this wrapper applies NO filter when reading from cfg — its
        # roster/task-planner callers only use the TEST split, and the filter is train-side.
        # `train_one_config` passes its trial's own value via the `min_train_rows` argument.
        min_train_rows=(int(min_train_rows) if min_train_rows is not None
                        else _scalar_min_rows(corpus.get("min_train_rows", 0))),
    )
