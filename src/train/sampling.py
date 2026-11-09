"""Epoch-aware row partitions and sampler indices, including persistent workers."""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler

PROTOCOL_VERSION = 5


def partition_rows(y, *, row_cap: int, rng: np.random.Generator,
                   classification: bool) -> list[np.ndarray]:
    """Partition every row once, with fresh membership/order for each epoch.

    Chunks differ in size by at most one and respect the cap. Distributing
    shuffled class blocks round-robin also spreads each class across chunks,
    without oversampling minority rows or dropping majority rows.
    """
    y = np.asarray(y)
    n = len(y)
    if row_cap < 2 or n < 2:
        raise ValueError("Row partitions need at least two rows and a row cap >= 2")
    count = (n + row_cap - 1) // row_cap
    if n // count < 2:
        raise ValueError("Cannot partition these rows under the cap with context and query in every chunk")
    if classification:
        classes = np.unique(y)
        ordered = np.concatenate([
            rng.permutation(np.flatnonzero(y == label))
            for label in rng.permutation(classes)
        ])
        chunks = [ordered[i::count].copy() for i in range(count)]
    else:
        chunks = list(np.array_split(rng.permutation(n), count))
    for chunk in chunks:
        rng.shuffle(chunk)
    return chunks


class EpochSampler(Sampler):
    """Send the epoch in each index; worker dataset copies need no shared mutable state."""

    def __init__(self, dataset, *, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.start_offset = 0
        self.order: list[int] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        rng = np.random.default_rng(self.seed + self.epoch * 10_007)
        self.order = rng.permutation(len(self.dataset._plan)).tolist()

    def __iter__(self):
        return iter((self.epoch, i) for i in self.order[self.start_offset:])

    def __len__(self):
        return len(self.order)
