"""Keep numerical thread pools inside the CPU allocation of a Slurm task."""
from __future__ import annotations

import os


def allocated_cpus() -> int | None:
    """Return the per-task Slurm ceiling; leave non-cluster defaults alone."""
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if not raw:
        return None
    if not raw.isdecimal() or int(raw) < 1:
        raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer")
    limit = int(raw)
    if hasattr(os, "sched_getaffinity"):
        limit = min(limit, len(os.sched_getaffinity(0)))
    return max(1, limit)


def bounded_threads(requested: int | None = None) -> int | None:
    """Resolve library 'all cores' defaults without exceeding this task's CPUs."""
    limit = allocated_cpus()
    if limit is None:
        return requested
    return min(int(requested), limit) if requested is not None and int(requested) > 0 else limit


def configure_training_threads(workers: int) -> int | None:
    """Reserve one core per data worker and cap already-loaded native pools."""
    limit = allocated_cpus()
    if limit is None:
        return None
    threads = max(1, limit - max(0, int(workers)))
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(threads)
    # Deliberately process-wide, like the worker initializer. Each trial
    # reapplies its own budget, and spawned workers reduce their pools to one.
    threadpool_limits(limits=threads)
    return threads
