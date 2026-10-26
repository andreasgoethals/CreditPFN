"""Inference-only investigation of a failed LoRA probe; never grants a gate.

Read the historical workflow and its saved native probe weights without changing
them. A changed diagnostic source is allowed here precisely because this output
cannot replace smoke/capacity receipts or authorize production training.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

import numpy as np


def compare_predictions(reference, observed, *, atol, rtol):
    """Report finite values, shape and tolerance violations without broadcasting."""
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    record = dict(reference_shape=list(reference.shape), observed_shape=list(observed.shape),
                  reference_nonfinite=int((~np.isfinite(reference)).sum()),
                  observed_nonfinite=int((~np.isfinite(observed)).sum()),
                  atol=float(atol), rtol=float(rtol), passed=False,
                  max_abs_error=None, mean_abs_error=None, max_tolerance_ratio=None,
                  violations=None)
    if reference.shape != observed.shape or not reference.size:
        return record
    finite = np.isfinite(reference) & np.isfinite(observed)
    if finite.any():
        difference = np.abs(reference[finite] - observed[finite])
        allowance = float(atol) + float(rtol) * np.abs(reference[finite])
        record.update(max_abs_error=float(difference.max()), mean_abs_error=float(difference.mean()),
                      max_tolerance_ratio=float((difference / allowance).max()),
                      violations=int((difference > allowance).sum()) + int((~finite).sum()))
        record["passed"] = bool(finite.all() and record["violations"] == 0)
    return record


@contextmanager
def tf32_disabled(disable):
    """One diagnostic arm only; restore both process-level policies on failure."""
    import torch
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        if disable:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def diagnose(identifier, task):
    """Compare existing base/zero/changed exports, with no fitting of model weights."""
    import torch
    from src.utils import lora_checks as checks
    from src.utils.atomic import write_json
    from src.utils.paths import checkpoints_dir

    checks.validate_id(identifier)
    track, family = checks.CASES[task]
    historical = json.loads((checks.folder(identifier) / "workflow.json").read_text(encoding="utf-8"))
    if (historical.get("schema") != checks.SCHEMA or historical.get("id") != identifier
            or historical.get("kind") != "creditpfn_lora" or historical.get("prepared") is not True):
        raise RuntimeError("Inference diagnosis requires a complete historical LoRA preparation")
    base = checks._base(track, family)
    bindings = [item for item in historical["inputs"] if Path(item["path"]).name == base.name]
    if len(bindings) != 1 or checks.file_digest(base) != bindings[0]["sha256"]:
        raise RuntimeError("Diagnostic base differs from the historical LoRA workflow")
    root = checkpoints_dir("trained", "experiment0", "lora", identifier, "probes")
    paths = [base, root / f"zero_{track}_{base.name}", root / f"changed_{track}_{base.name}"]
    artifacts = [checks._binding(path) for path in paths]  # Fail before inference if an export is missing.
    limits = checks.policy()
    tolerance = dict(atol=float(limits.prediction_atol), rtol=float(limits.prediction_rtol))
    result = dict(kind="creditpfn_lora_inference_diagnostic", schema=1, id=identifier,
                  task=task, track=track, family=family, authorizes_production=False,
                  historical_workflow_sha256=checks.digest_json(historical),
                  historical_fingerprint=historical["fingerprint"],
                  historical_versions=historical["versions"],
                  current_fingerprint=checks.fingerprint(),
                  current_versions=checks.environment_versions(stage="eval"),
                  gpu=torch.cuda.get_device_name(), job_id=os.environ["SLURM_JOB_ID"],
                  artifacts=artifacts, arms=[])
    destination = checks.folder(identifier) / "diagnostics" / f"predictions_{task}_{uuid.uuid4().hex}.json"
    write_json(destination, result, exclusive=True)
    policies = [("float32_inherited_tf32", "float32", False),
                ("float32_tf32_disabled", "float32", True)]
    if family == "tabpfn":
        policies.append(("automatic_inherited_tf32", "automatic", False))
    for label, precision, disable in policies:
        arm = dict(policy=label, precision=precision, checks=[], complete=False)
        result["arms"].append(arm)
        try:
            with tf32_disabled(disable):
                arm["tf32"] = dict(matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                                   cudnn=bool(torch.backends.cudnn.allow_tf32))
                predictions = checks.prediction_probe(paths, track, family, precision=precision,
                                                       strict=False, diagnostics=arm["checks"])
            arm["original_vs_zero_export"] = compare_predictions(predictions[0], predictions[1], **tolerance)
            arm["zero_vs_changed_export"] = compare_predictions(predictions[1], predictions[2], **tolerance)
            arm["complete"] = True
        except Exception as error:
            arm["error"] = dict(type=type(error).__name__, message=str(error))
        finally:
            torch.cuda.empty_cache()
            write_json(destination, result)
    # Re-check content, not only timestamps, to detect a concurrent checkpoint write.
    checks.verify_bindings(artifacts, full_hash=True)
    result["complete"] = all(arm["complete"] for arm in result["arms"])
    write_json(destination, result)
    print(json.dumps(result, allow_nan=False), flush=True)
    print(f"Diagnostic only; no validation receipt granted. Report: {destination}", flush=True)
    return destination
