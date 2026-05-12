"""DataLoader: turn a list of :class:`ChunkRef` into TabPFN-shaped batches.

Per-step recipe (one "batch" = one dataset, batch_size is FIXED at 1
by TabPFN's ``meta_dataset_collator`` assertion at
``repositories/TabPFN .txt:17665-17666``):

  1. Pick one chunk (one ``.npz`` file) — that's our atomic unit.

  2. Concatenate the cached ``X_context`` and ``X_query`` rows into one
     pool. The cache's 60/40 split is ignored at training time; we
     resplit deterministically from ``(seed, chunk_idx)`` — so the
     same chunk gets the **same** ctx/query mix every epoch (a
     reproducibility choice, not the per-epoch reshuffle described
     in `repositories/TabPFN .txt:18640-18656`). Inter-epoch
     variation comes from the DataLoader's outer shuffle of the
     chunk order, not from re-resampling within a chunk.

  3. Subsample to ``cfg.train.n_finetune_ctx_plus_query_samples``
     (default 100_000 — matches the chunk cap; tuned for wICE H100 NVL).
     Random rows, without replacement.

  4. Random 80 / 20 split where the 20% is the query split — i.e.
     ``cfg.train.finetune_ctx_query_split_ratio``. Both splits are
     drawn from the same chunk so they share encoder vocabulary, etc.

  5. Cast to ``torch.Tensor`` of shape:
        - ``X``:  (n_samples, batch_size=1, n_features)
        - ``y``:  (n_samples, batch_size=1, 1)
     (this matches the ``model(train_x, train_y, test_x,
     categorical_inds)`` API in `TabPFN V2 Finetuning.txt:1413-1505`).

For *test-time evaluation*, see :func:`prepare_eval_chunk`: we
honour the cache's stable 60/40 ctx/query split (no extra
randomness so test numbers are reproducible across HP variants),
and subsample to ``cfg.eval.n_inference_subsample_samples`` if larger.
The same function is the entry point for the future `src/eval/`
module, which will use it to score XGBoost / CatBoost / TabICL on
the same chunks for an apples-to-apples comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.train.corpus import ChunkRef

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tensor batch container
# --------------------------------------------------------------------------- #


@dataclass
class TabPFNBatch:
    """One forward-pass-ready batch (batch_size=1).

    Tensor shapes follow the TabPFN convention used by the
    ``PerFeatureTransformer`` forward signature:

    * ``X_context``  — (n_ctx,   1, n_features)   float32
    * ``y_context``  — (n_ctx,   1, 1)            float32 / int64
    * ``X_query``    — (n_query, 1, n_features)   float32
    * ``y_query``    — (n_query, 1, 1)            float32 / int64
    * ``categorical_idx`` — list[int]
    """
    X_context: torch.Tensor
    y_context: torch.Tensor
    X_query:   torch.Tensor
    y_query:   torch.Tensor
    categorical_idx: list[int]
    task_type: str
    dataset_id: str

    def to(self, device: str) -> "TabPFNBatch":
        return TabPFNBatch(
            X_context=self.X_context.to(device, non_blocking=True),
            y_context=self.y_context.to(device, non_blocking=True),
            X_query=self.X_query.to(device, non_blocking=True),
            y_query=self.y_query.to(device, non_blocking=True),
            categorical_idx=self.categorical_idx,
            task_type=self.task_type,
            dataset_id=self.dataset_id,
        )


# --------------------------------------------------------------------------- #
# Numpy → torch helpers
# --------------------------------------------------------------------------- #


def _load_chunk(ref: ChunkRef) -> dict[str, np.ndarray]:
    """Read one cached ``.npz`` into a plain dict of numpy arrays."""
    with np.load(ref.chunk_path) as data:
        return {
            "X_context": data["X_context"],
            "y_context": data["y_context"],
            "X_query":   data["X_query"],
            "y_query":   data["y_query"],
            "categorical_idx": data["categorical_idx"],
        }


def _to_xy_tensor(
    X: np.ndarray, y: np.ndarray, *, task_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast numpy ``(n, F)`` and ``(n,)`` to the TabPFN tensor shapes."""
    X_t = torch.from_numpy(np.ascontiguousarray(X.astype(np.float32, copy=False)))
    X_t = X_t.unsqueeze(1)  # (n, 1, F)

    if task_type == "classification":
        y_dtype = torch.int64
    else:
        y_dtype = torch.float32
    y_t = torch.as_tensor(y, dtype=y_dtype).reshape(-1, 1, 1).contiguous()
    return X_t, y_t


def _resample_chunk(
    chunk: dict[str, np.ndarray], *,
    n_total_target: int,
    query_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concat cache's ctx/query, subsample, then resplit.

    Returns ``(X_ctx, y_ctx, X_qry, y_qry)`` numpy arrays.
    """
    X_full = np.concatenate([chunk["X_context"], chunk["X_query"]], axis=0)
    y_full = np.concatenate([chunk["y_context"], chunk["y_query"]], axis=0)

    n_total_actual = min(n_total_target, len(X_full))
    if n_total_actual <= 1:
        # Pathological — shouldn't happen with min_chunk_size=2000, but
        # be defensive: degenerate to whatever we have.
        return chunk["X_context"], chunk["y_context"], \
               chunk["X_query"], chunk["y_query"]

    if n_total_actual < len(X_full):
        sel = rng.choice(len(X_full), size=n_total_actual, replace=False)
    else:
        sel = rng.permutation(len(X_full))

    X_full = X_full[sel]
    y_full = y_full[sel]

    n_query = max(1, int(round(n_total_actual * query_fraction)))
    n_query = min(n_query, n_total_actual - 1)
    n_ctx = n_total_actual - n_query

    return X_full[:n_ctx], y_full[:n_ctx], X_full[n_ctx:], y_full[n_ctx:]


# --------------------------------------------------------------------------- #
# Public: training Dataset
# --------------------------------------------------------------------------- #


class ChunkDataset(Dataset):
    """One ``__getitem__`` call → one fully-formed :class:`TabPFNBatch`.

    Designed to be wrapped in ``torch.utils.data.DataLoader`` with
    ``batch_size=1, collate_fn=lambda batch: batch[0]`` (since our
    "batch" is already a single dataset).

    Pass training-mode ``seed`` to make subsampling deterministic
    across processes. The per-call rng is seeded from
    ``(seed, idx)`` — so the same chunk index produces the same
    resample every epoch (full run-level reproducibility, at the
    cost of no per-epoch resampling variation within a chunk).
    """

    def __init__(
        self,
        chunks: Sequence[ChunkRef],
        *,
        n_total_target: int,
        query_fraction: float,
        seed: int = 0,
    ) -> None:
        if len(chunks) == 0:
            raise ValueError("ChunkDataset received an empty chunk list")
        self.chunks = list(chunks)
        self.n_total_target = int(n_total_target)
        self.query_fraction = float(query_fraction)
        self._base_seed = int(seed)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> TabPFNBatch:
        ref = self.chunks[idx]
        # Per-call rng so parallel DataLoader workers don't clash;
        # mix in idx and base_seed so it's reproducible per (seed, idx).
        rng = np.random.default_rng(
            (self._base_seed * 1_000_003 + idx) & 0xFFFF_FFFF
        )
        chunk = _load_chunk(ref)

        X_ctx, y_ctx, X_qry, y_qry = _resample_chunk(
            chunk,
            n_total_target=self.n_total_target,
            query_fraction=self.query_fraction,
            rng=rng,
        )

        X_ctx_t, y_ctx_t = _to_xy_tensor(X_ctx, y_ctx, task_type=ref.task_type)
        X_qry_t, y_qry_t = _to_xy_tensor(X_qry, y_qry, task_type=ref.task_type)

        cat_idx = chunk["categorical_idx"].tolist()
        return TabPFNBatch(
            X_context=X_ctx_t,
            y_context=y_ctx_t,
            X_query=X_qry_t,
            y_query=y_qry_t,
            categorical_idx=cat_idx,
            task_type=ref.task_type,
            dataset_id=ref.dataset_id,
        )


def identity_collate(batch):
    """Keep TabPFN's batch_size=1 invariant.

    Our ``__getitem__`` already returns one full :class:`TabPFNBatch`
    (= one dataset). The DataLoader wraps it in a length-1 list — we
    just unwrap it.
    """
    if len(batch) != 1:
        raise ValueError(
            f"identity_collate expects batch_size=1; got {len(batch)} "
            "(TabPFN's meta_dataset_collator hard-asserts this — see "
            "repositories/TabPFN .txt:17666)"
        )
    return batch[0]


# --------------------------------------------------------------------------- #
# Public: test/eval chunk preparation (deterministic)
# --------------------------------------------------------------------------- #


def prepare_eval_chunk(
    ref: ChunkRef,
    *,
    n_inference_subsample_samples: int,
    seed: int,
) -> TabPFNBatch:
    """Build a deterministic eval batch for one chunk.

    Uses the cache's stable 60/40 context/query split (no resampling,
    so test numbers are reproducible across HP variants and across
    different models in the future TabPFN-vs-XGBoost-vs-… benchmark).
    If ``n_inference_subsample_samples`` is smaller than the chunk's
    row count, both splits are subsampled proportionally (preserves
    the ratio).
    """
    chunk = _load_chunk(ref)
    rng = np.random.default_rng(seed)

    n_ctx_full = len(chunk["X_context"])
    n_qry_full = len(chunk["X_query"])
    total = n_ctx_full + n_qry_full

    if total > n_inference_subsample_samples > 0:
        keep_ratio = n_inference_subsample_samples / total
        n_ctx_keep = max(1, int(round(n_ctx_full * keep_ratio)))
        n_qry_keep = max(1, int(round(n_qry_full * keep_ratio)))
        ctx_sel = rng.choice(n_ctx_full, size=n_ctx_keep, replace=False)
        qry_sel = rng.choice(n_qry_full, size=n_qry_keep, replace=False)
        X_ctx, y_ctx = chunk["X_context"][ctx_sel], chunk["y_context"][ctx_sel]
        X_qry, y_qry = chunk["X_query"][qry_sel], chunk["y_query"][qry_sel]
    else:
        X_ctx, y_ctx = chunk["X_context"], chunk["y_context"]
        X_qry, y_qry = chunk["X_query"], chunk["y_query"]

    X_ctx_t, y_ctx_t = _to_xy_tensor(X_ctx, y_ctx, task_type=ref.task_type)
    X_qry_t, y_qry_t = _to_xy_tensor(X_qry, y_qry, task_type=ref.task_type)

    return TabPFNBatch(
        X_context=X_ctx_t,
        y_context=y_ctx_t,
        X_query=X_qry_t,
        y_query=y_qry_t,
        categorical_idx=chunk["categorical_idx"].tolist(),
        task_type=ref.task_type,
        dataset_id=ref.dataset_id,
    )


# Backwards-compat alias kept only for the duration of the refactor;
# remove once src/eval/ lands.
prepare_validation_chunk = prepare_eval_chunk
