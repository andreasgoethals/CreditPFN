"""Compare an interrupted-and-resumed training run with its uninterrupted twin (Experiment 0)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def saved_tensors(path: Path) -> dict:
    """The trained tensors of a saved checkpoint: full weights (``state_dict``) or a DiagScale
    bundle's ``adapter_state``. Both arms of a recovery case share one format."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "adapter_state" in payload:
        return {f"adapter.{k}": v for k, v in payload["adapter_state"].items()}
    return payload["state_dict"]


def compare(reference: Path, resumed: Path) -> dict:
    before, after = saved_tensors(reference), saved_tensors(resumed)
    keys_equal = before.keys() == after.keys()
    max_delta = 0.0
    non_diagnostic_max = 0.0
    non_diagnostic_tensor = None
    diagnostic_deltas = {}
    deltas = []
    different = []
    exact = True
    for key in sorted(before.keys() & after.keys()):
        a, b = before[key], after[key]
        exact &= a.dtype == b.dtype and torch.equal(a, b)
        if (a.shape != b.shape or a.dtype != b.dtype
                or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            different.append(key)
            continue
        delta = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
        max_delta = max(max_delta, delta)
        if key == "criterion.losses_per_bucket":
            diagnostic_deltas[key] = delta
        elif delta > non_diagnostic_max:
            non_diagnostic_max, non_diagnostic_tensor = delta, key
        if delta:
            deltas.append({"tensor": key, "max_absolute_difference": delta})
        if not torch.allclose(a, b, rtol=1e-6, atol=1e-7):
            different.append(key)
    # FullSupportBarDistribution.forward updates this detached diagnostic outside
    # the model's recovery state. It affects neither returned losses nor inference.
    # Report its difference, but use the same inference-state scope as null audits.
    model_differences = [key for key in different if key != "criterion.losses_per_bucket"]
    return {"passed": keys_equal and not model_differences,
            "all_saved_state_close": keys_equal and not different,
            "bitwise_equal": keys_equal and exact,
            "max_absolute_difference": max_delta, "different_tensors": model_differences,
            "missing_tensors": sorted(before.keys() - after.keys()),
            "added_tensors": sorted(after.keys() - before.keys()),
            "max_non_diagnostic_difference": non_diagnostic_max,
            "max_non_diagnostic_tensor": non_diagnostic_tensor,
            "diagnostic_buffer_differences": diagnostic_deltas,
            "diagnostic_buffers_different": [key for key in different if key == "criterion.losses_per_bucket"],
            "largest_differences": sorted(deltas, key=lambda d: -d["max_absolute_difference"])[:5],
            "rtol": 1e-6, "atol": 1e-7}


def compare_trajectories(reference: pd.DataFrame, resumed: pd.DataFrame, *, milestones) -> dict:
    """Compare every scheduled monitor milestone, including those before the interruption."""
    columns = sorted(c for c in reference if c.startswith(("metric__", "score__")))
    other_columns = sorted(c for c in resumed if c.startswith(("metric__", "score__")))
    milestones = list(milestones)
    problems = []
    if not columns or columns != other_columns:
        problems.append("Missing or different monitor columns")
    if any(frame.get("successful_updates", pd.Series(dtype=int)).tolist() != milestones
           for frame in (reference, resumed)):
        problems.append(f"Expected exactly one measurement at each of updates {milestones}")
    if problems:
        return {"passed": False, "problems": problems, "by_update": []}
    updates = []
    for index, step in enumerate(milestones):
        a = reference[columns].iloc[index].to_numpy(float)
        b = resumed[columns].iloc[index].to_numpy(float)
        close = np.isclose(a, b, rtol=1e-5, atol=1e-7, equal_nan=True)
        # Optional scores may be unavailable in both arms; primary metrics must be finite.
        primary = np.array([c.startswith("metric__") for c in columns])
        close[primary] &= np.isfinite(a[primary]) & np.isfinite(b[primary])
        close &= ~np.isinf(a) & ~np.isinf(b)
        finite = np.isfinite(a) & np.isfinite(b)
        delta = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None
        updates.append({"successful_updates": step, "passed": bool(close.all()),
                        "different_metrics": int((~close).sum()),
                        "first_different_metrics": [c for c, ok in zip(columns, close) if not ok][:5],
                        "max_absolute_difference": delta})
    return {"passed": all(u["passed"] for u in updates), "problems": [], "by_update": updates, "rtol": 1e-5, "atol": 1e-7}
