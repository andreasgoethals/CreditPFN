"""Observe every Q/V adapter in real debug training; no production-loop changes."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import torch

from src.model import lora
from src.utils.atomic import write_json


class AdapterAudit:
    """Gradient reachability per projection, including both slices of packed QKV.

    A has zero gradient on the first backward because B starts at zero. Require
    finite gradient paths to both factors, a nonzero B gradient, and a nonzero
    effective update for each projection across the bounded training probe.
    Export rounding is recorded separately by the existing native-weight audit.
    """

    def __init__(self, model):
        self.model = model
        self.records, self.handles, self.adapters = [], [], []
        for target, _, sequence in lora._installed(model):
            adapter = sequence[0]
            for projection in adapter.projections:
                record = dict(target=f"{target['module']}.{target['weight']}:{projection}")
                for factor in ("A", "B"):
                    observations = dict(calls=0, nonzero_calls=0, nonfinite_calls=0)
                    record[factor] = observations
                    parameter = getattr(adapter, f"{factor}_{projection}")
                    if not parameter.requires_grad:
                        raise AssertionError(f"Frozen LoRA factor: {record['target']} {factor}")
                    self.handles.append(parameter.register_hook(self._hook(observations)))
                self.records.append(record)
                self.adapters.append((adapter, projection))
        if not self.records:
            raise AssertionError("No Q/V adapters available for gradient audit")

    @staticmethod
    def _hook(observations):
        def observe(gradient):
            observations["calls"] += 1
            finite = bool(torch.isfinite(gradient).all())
            observations["nonfinite_calls"] += int(not finite)
            observations["nonzero_calls"] += int(finite and bool(torch.count_nonzero(gradient)))
            # Returning None leaves the gradient unchanged.
        return observe

    @torch.no_grad()
    def report(self):
        for record, (adapter, projection) in zip(self.records, self.adapters):
            with torch.autocast(device_type=next(adapter.parameters()).device.type, enabled=False):
                delta = (getattr(adapter, "B_" + projection) @ getattr(adapter, "A_" + projection)) * adapter.scaling
            norm = float(delta.double().norm())
            record["effective_update_norm"] = norm if math.isfinite(norm) else None
            record["passed"] = bool(
                all(record[factor]["calls"] > 0 and record[factor]["nonfinite_calls"] == 0 for factor in ("A", "B"))
                and record["B"]["nonzero_calls"] > 0 and math.isfinite(norm) and norm > 0)
        return dict(schema=1, kind="lora_training_gradient_audit", projections=self.records,
                    projection_count=len(self.records),
                    passed=all(record["passed"] for record in self.records))

    def close(self):
        for handle in self.handles:
            handle.remove()


def run_observed(argv, destination):
    """Run the unchanged training entry point with temporary observation hooks."""
    original = lora.inject
    audits = []
    def inject(*args, **kwargs):
        parameters = original(*args, **kwargs)
        audits.append(AdapterAudit(args[0] if args else kwargs["model"]))
        return parameters
    previous_argv = sys.argv
    code = 0
    try:
        with patch.object(lora, "inject", inject):
            sys.argv = ["scripts/train_pipeline.py", *argv]
            try:
                runpy.run_path("scripts/train_pipeline.py", run_name="__main__")
            except SystemExit as error:
                code = int(error.code or 0)
    except BaseException:
        code = 1
        raise
    finally:
        sys.argv = previous_argv
        results = [audit.report() for audit in audits]
        for audit in audits:
            audit.close()
        report = dict(schema=1, kind="lora_training_gradient_audit", training_exit_code=code,
                      models=results, passed=code == 0 and len(results) == 1 and results[0]["passed"])
        write_json(destination, report, exclusive=True)
        print(f"LoRA per-projection gradient audit: passed={report['passed']} report={destination}", flush=True)
    if code == 0 and not report["passed"]:
        missing = [r["target"] for model in results for r in model["projections"] if not r["passed"]]
        raise AssertionError("LoRA training did not update every Q/V adapter: " + ", ".join(missing or ["missing model"]))
    return code


def main(argv=None):
    from src.utils import lora_checks as checks
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args, training = parser.parse_known_args(argv)
    if training[:1] == ["--"]:
        training = training[1:]
    identifier = checks.validate_id(os.environ.get("CREDITPFN_LORA_WORKFLOW", ""))
    root = checks.evidence_folder(identifier).resolve()
    if not args.report.resolve().is_relative_to(root / "diagnostics"):
        raise ValueError("Gradient evidence must stay inside this LoRA debug workflow")
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME") != "mindwell"
            or not torch.cuda.is_available() or "B200" not in torch.cuda.get_device_name()):
        raise RuntimeError("LoRA training probe requires a submitted Mindwell B200 allocation")
    if "--config" not in training:
        raise ValueError("Training probe requires a scoped debug configuration")
    config = Path(training[training.index("--config") + 1]).resolve()
    if not config.is_relative_to((checks.folder(identifier) / "configs").resolve()):
        raise ValueError("Training probe may only run this workflow's debug configs")
    return run_observed(training, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
