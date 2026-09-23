"""Recovery at optimizer boundaries and monitoring without altering training randomness."""
from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from src.train.checkpoint_io import atomic_save


class TrainingInterrupted(RuntimeError):
    """The trial can continue from a verified recovery checkpoint (CLI exit 75)."""


def capture_rng() -> dict:
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    npstate = state["numpy"]
    np.random.set_state((npstate[0], np.asarray(npstate[1], dtype=np.uint32), *npstate[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Recovery requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def preserve_random_state(model=None):
    state = capture_rng()
    modes = [(m, m.training) for m in model.modules()] if model is not None else []
    try:
        yield
    finally:
        restore_rng(state)
        for module, training in modes:
            module.training = training


def save_recovery(path: Path, *, model, optimizer, scheduler, scaler,
                  identity: dict, progress: dict) -> None:
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Recovery must be saved after gradients have been flushed")
    atomic_save({"schema_version": 1, "identity": identity,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "rng": capture_rng(), "progress": progress}, path, None)


def load_recovery(path: Path, *, model, optimizer, scheduler, scaler, identity: dict) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1 or state.get("identity") != identity:
        raise RuntimeError("Recovery checkpoint belongs to a different trial/configuration")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state["progress"]


def batch_rows(batch) -> int:
    """Count context + query rows once, before ensemble replication."""
    if hasattr(batch, "X"):
        return int(batch.X.shape[1])
    if hasattr(batch, "X_context"):
        x = batch.X_context
        q = batch.X_query
        if isinstance(x, list):
            return int(x[0].shape[0] + q[0].shape[0])
        return int(x.shape[0] + q.shape[0])
    # TabPFNEnsembleBatch holds each member in a preprocessed tensor list.
    return int(batch.members[0].X_context.shape[0] + batch.members[0].X_query.shape[0])
