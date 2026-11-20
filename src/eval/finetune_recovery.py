"""Recover E5 at completed epochs of the two pinned upstream training loops.

Neither fine-tuner exposes a callback with its optimizer, random state and early-stopping
state. Bind the epoch hooks to this estimator instance, leaving both installed libraries
untouched. A changed upstream method requires a fresh parity check before use.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import random
import textwrap
import time
from pathlib import Path
from types import FunctionType, MethodType

import numpy as np
import torch


class InterruptedEpoch(RuntimeError):
    """Experiment 0 stopped immediately after a durable epoch checkpoint."""


class FitProgress:
    def __init__(self, path: Path | None, *, stop_after_epoch: int | None = None):
        self.path = path
        self.stop_after_epoch = stop_after_epoch
        self.optimizer_steps = 0
        self.resumed_at_epoch = 0
        self.previous_seconds = 0.
        self.started = time.monotonic()

    def count_step(self, optimizer, args, kwargs):
        if isinstance(optimizer, torch.optim.AdamW):
            self.optimizer_steps += 1

    def save(self, epoch, model, optimizer, scheduler, scaler, best, metric, patience, step,
             rng=None, schedule_shape=None):
        path = self.path
        cuda = next(model.parameters()).is_cuda
        record = dict(next_epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                      scheduler=scheduler.state_dict() if scheduler is not None else None,
                      scaler=scaler.state_dict() if scaler is not None else None,
                      best=best, metric=float(metric), patience=patience, global_step=step,
                      optimizer_steps=self.optimizer_steps, schedule_shape=schedule_shape,
                      seconds=self.previous_seconds + time.monotonic() - self.started,
                      python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                      local_rng=rng.bit_generator.state if rng is not None else None,
                      torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if cuda else None)
        pending = path.with_suffix(".pending")
        with pending.open("wb") as stream:
            torch.save(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
        if self.stop_after_epoch == epoch + 1:
            raise InterruptedEpoch(f"Saved epoch {epoch + 1}")

    def restore(self, model, optimizer, scheduler, scaler, *, make_scheduler=None, rng=None):
        # This private working file is written under the task's identity and exclusive lock.
        record = torch.load(self.path, map_location="cpu", weights_only=False)
        model.load_state_dict(record["model"])
        if scheduler is None and record["scheduler"] is not None:
            scheduler = make_scheduler(*record["schedule_shape"])
        if scheduler is not None:
            scheduler.load_state_dict(record["scheduler"])
        # Scheduler construction can change the optimizer's current LR. Restore it last.
        optimizer.load_state_dict(record["optimizer"])
        if scaler is not None:
            scaler.load_state_dict(record["scaler"])
        random.setstate(record["python_rng"])
        np.random.set_state(record["numpy_rng"])
        if rng is not None:
            rng.bit_generator.state = record["local_rng"]
        torch.set_rng_state(record["torch_rng"])
        if record["cuda_rng"] is not None:
            torch.cuda.set_rng_state(record["cuda_rng"])
        self.previous_seconds = record["seconds"]
        self.resumed_at_epoch = record["next_epoch"]
        self.optimizer_steps = record["optimizer_steps"]
        return (scheduler, record["best"], record["metric"], record["patience"],
                record["global_step"], record["schedule_shape"])


_METHODS = {
    "tabpfn": ("_fit", "7f9106e070e4c22dd513cf4a2541574b58aed238b24aa95fa4b10902e243a6fd"),
    "tabicl": ("_fit_impl", "f603d640459dd5cf166d5f5bea9f372a602f9671528d4584338a0843deb779d4"),
}


def bind_training_loop(estimator, family, progress):
    """Add explicit epoch-boundary calls to a pinned method, preserving its loop body."""
    if family == "tabicl" and progress.path is None:
        return
    name, expected = _METHODS[family]
    original = getattr(type(estimator), name, None)
    if original is None and progress.path is None:  # An estimator test double.
        return
    namespace = dict(original.__globals__, _progress=progress)
    if family == "tabpfn":
        from src.eval.finetune_scheduler import UpdateAlignedLambdaLR
        namespace["LambdaLR"] = UpdateAlignedLambdaLR
    if progress.path is None:
        method = FunctionType(original.__code__, namespace, name, original.__defaults__, original.__closure__)
        method.__kwdefaults__ = original.__kwdefaults__
    else:
        source = textwrap.dedent(inspect.getsource(original)).strip()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError(f"Changed upstream {family} fine-tuner; revalidate epoch recovery")
        tree = ast.parse(source)
        body = tree.body[0].body
        loop = next(node for node in body if isinstance(node, ast.For) and ast.unparse(node.target) == "epoch")
        if family == "tabpfn":
            restore = """
def _make_scheduler(steps, warmup):
    fn = get_cosine_schedule_with_warmup(total_steps=steps, warmup_steps=warmup, warmup_only=self.lr_warmup_only)
    return LambdaLR(optimizer, lr_lambda=fn)
if _progress.path.exists():
    scheduler, best_model_state, best_metric, patience_counter, global_step, _shape = _progress.restore(
        self.finetuned_estimator_.model_, optimizer, scheduler, scaler, make_scheduler=_make_scheduler, rng=rng)
    if _shape is not None:
        total_steps, warmup_steps = _shape
    epoch_to_start_from = _progress.resumed_at_epoch
"""
            save = """
_progress.save(epoch, self.finetuned_estimator_.model_, optimizer, scheduler, scaler, best_model_state,
               best_metric, patience_counter, global_step, rng, (total_steps, warmup_steps) if scheduler else None)
"""
        else:
            restore = """
if _progress.path.exists():
    scheduler, best_state, best_metric, patience_counter, global_step, _ = _progress.restore(
        self.model_, optimizer, scheduler, scaler)
"""
            # Keep the upstream tqdm object for its logging, skipping already committed epochs.
            loop.body[:0] = ast.parse("if epoch < _progress.resumed_at_epoch:\n    continue").body
            save = """
_progress.save(epoch, self.model_, optimizer, scheduler, scaler, best_state, best_metric, patience_counter, global_step)
"""
        pos = body.index(loop)
        body[pos:pos] = ast.parse(restore).body
        loop.body.extend(ast.parse(save).body)
        exec(compile(ast.fix_missing_locations(tree), inspect.getfile(original), "exec"), namespace)
        method = namespace[name]
    setattr(estimator, name, MethodType(method, estimator))
