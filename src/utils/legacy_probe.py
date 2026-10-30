"""Observe legacy adapter gradients and native export/cache behavior on VSC."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import torch

from src.model import diagscale, lora
from src.utils.atomic import write_json
from src.utils.legacy_design import limits, root, validate_id


class ScaleAudit:
    def __init__(self, model):
        self.model = model
        self.records, self.handles = [], []
        for name, module in model.named_modules():
            if isinstance(module, diagscale.QueryScale):
                record = dict(target=name, calls=0, nonzero=0, nonfinite=0)
                def observe(gradient, record=record):
                    record["calls"] += 1
                    finite = bool(torch.isfinite(gradient).all())
                    record["nonfinite"] += int(not finite)
                    record["nonzero"] += int(finite and bool(torch.count_nonzero(gradient)))
                self.handles.append(module.delta.register_hook(observe))
                self.records.append((record, module))
        self.original = [(p, p.detach().cpu().clone()) for p in model.parameters() if not p.requires_grad]

    def report(self):
        for record, module in self.records:
            record["update_norm"] = float(module.delta.detach().double().norm())
            record["passed"] = (record["calls"] > 0 and record["nonzero"] > 0 and record["nonfinite"] == 0
                                and math.isfinite(record["update_norm"]) and record["update_norm"] > 0)
        unchanged = all(torch.equal(p.detach().cpu(), value) for p, value in self.original)
        return dict(projections=[r for r, _ in self.records], base_unchanged=unchanged,
                    passed=bool(self.records) and unchanged and all(r["passed"] for r, _ in self.records))

    def close(self):
        for handle in self.handles:
            handle.remove()


def run_observed(argv, destination):
    from src.utils.lora_training_probe import AdapterAudit
    from src.train import loop
    from src.train.config import load_train_config
    cfg = load_train_config(config_path=argv[argv.index("--config") + 1])
    mode = str(cfg.train.get("adaptation_mode", "full"))
    module = lora if mode == "lora" else diagscale
    original, audits, frozen = loop._make_optimizer_and_scheduler, [], []

    def make_optimizer(model, *args, **kwargs):
        optimizer, scheduler = original(model, *args, **kwargs)
        if mode in ("lora", "diagscale"):
            expected = {id(p) for p in module.adapter_parameters(model)}
            selected = [id(p) for group in optimizer.param_groups for p in group["params"]]
            if not expected or set(selected) != expected or len(selected) != len(expected):
                raise AssertionError("Training optimizer must contain exactly the declared adapters")
            # Monitoring reloads DiagScale bundles into separate inference models.
            # Observe the model actually handed to the training optimizer, never
            # those inference-only clones. Every optimized adapter must still
            # receive finite gradients and change; frozen weights must stay equal.
            audits.append(AdapterAudit(model) if mode == "lora" else ScaleAudit(model))
            frozen.extend((p, p.detach().cpu().clone()) for p in model.parameters() if not p.requires_grad)
        return optimizer, scheduler

    previous = sys.argv
    code = 0
    try:
        with patch.object(loop, "_make_optimizer_and_scheduler", make_optimizer):
            sys.argv = ["scripts/train_pipeline.py", *argv]
            try:
                runpy.run_path("scripts/train_pipeline.py", run_name="__main__")
            except SystemExit as exc:
                code = int(exc.code or 0)
    except BaseException:
        code = 1
        raise
    finally:
        sys.argv = previous
        observations = [audit.report() for audit in audits]
        unchanged = all(torch.equal(p.detach().cpu(), initial) for p, initial in frozen)
        for audit in audits:
            audit.close()
        valid = (len(observations) == 1 and observations[0]["passed"]) if mode in ("lora", "diagscale") else not observations
        report = dict(mode=mode, exit_code=code, models=observations, base_unchanged=unchanged,
                      passed=code == 0 and valid and unchanged)
        write_json(destination, report, exclusive=True)
    if code == 0 and not report["passed"]:
        raise AssertionError("Training did not produce finite updates in every declared adapter")
    return code


def _attention(model):
    for module in model.modules():
        if type(module).__name__ == "AlongColumnAttention" or (
                "MultiHeadAttention" in type(module).__name__ and hasattr(module, "_w_qkv")):
            return module
    raise AssertionError("No supported native legacy attention found")


def attention_call(module, x, *, cached=False):
    if hasattr(module, "q_projection"):
        fresh, cache = module(x, single_eval_pos=10, return_kv=True)
        if cached:
            hit, _ = module(x[:, 10:], single_eval_pos=0, cached_kv=cache)
            return fresh[:, 10:], hit
        return fresh
    # Older attention owns its KV buffers; exercise cross attention plus its cache.
    module.empty_kv_cache()
    fresh = module(x[:, 10:], x[:, :10], cache_kv=True)
    if cached:
        return fresh, module(x[:, 10:], use_cached_kv=True)
    return fresh


def _predictions(path, track, mode, *, precision="float32"):
    import numpy as np
    from src.utils.lora_checks import prediction_probe
    if mode != "diagscale":
        return prediction_probe([path], track, "tabpfn", precision=precision,
                                check_chunks=precision == "float32")[0]
    from src.model.tabpfn_models import _trust_local_checkpoints
    from src.utils.lora_diagnostics import compare_predictions, compare_query_chunks
    rng = np.random.default_rng(912)
    x = rng.normal(size=(128, 8)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int) if track == "pd" else x[:, 0] + .1 * x[:, 1]
    options = dict(device="cuda", n_estimators=2, random_state=19)
    if precision == "float32":
        options["inference_precision"] = torch.float32
    estimator = diagscale.make_estimator("classification" if track == "pd" else "regression",
                                        path, family="tabpfn", **options)
    with _trust_local_checkpoints():
        estimator.fit(x[:96], y[:96])
        predict = estimator.predict_proba if track == "pd" else estimator.predict
        first, repeat = np.asarray(predict(x[96:])), np.asarray(predict(x[96:]))
        chunks = np.concatenate([predict(x[96:112]), predict(x[112:])])
    tolerance = dict(atol=limits().prediction_atol, rtol=limits().prediction_rtol)
    if not compare_predictions(first, repeat, **tolerance)["passed"]:
        raise AssertionError("Repeated DiagScale predictions differ")
    if precision == "float32" and not compare_query_chunks(first, chunks, track=track,
        context_y=y[:96], regression_atol_std=limits().regression_chunk_atol_std, **tolerance)["passed"]:
        raise AssertionError("DiagScale query batching changes predictions")
    return first


def adapter_probe(identifier, cfg):
    import numpy as np
    from src.utils.lora_checks import _load_native, _save_native, _binding
    from src.utils.experiment import file_digest
    from src.utils.paths import resolve_base_checkpoint, checkpoints_dir
    track, mode = str(cfg.track), str(cfg.train.adaptation_mode)
    base = resolve_base_checkpoint((cfg.tunable.classifier_base_paths if track == "pd" else cfg.tunable.regressor_base_paths)[0])
    model, *info = _load_native(base, track, "tabpfn")
    model.eval().requires_grad_(False)
    original = [(p, p.detach().cpu().clone()) for p in model.parameters()]
    attention = _attention(model)
    dimension = attention.q_projection.in_features if hasattr(attention, "q_projection") else attention._input_size
    torch.manual_seed(518)
    x = torch.randn(1, 16, dimension, device="cuda")
    with torch.no_grad():
        before = attention_call(attention, x).detach()
    module = lora if mode == "lora" else diagscale
    parameters = module.inject(model, family="tabpfn", base_path="checkpoints/" + base.name, base_sha256=file_digest(base))
    with torch.no_grad():
        zero = attention_call(attention, x).detach()
    if not torch.allclose(before, zero, atol=1e-6, rtol=1e-5):
        raise AssertionError("Zero adapter changed native legacy attention")
    directory = checkpoints_dir("trained", "experiment0", "extensions", identifier, "probes")
    zero_path = directory / f"zero_{mode}_{base.name}"
    changed_path = directory / f"changed_{mode}_{base.name}"
    _save_native(model, info, zero_path, "tabpfn")
    # Cache buffers must not retain a training graph from a preceding forward.
    after = attention_call(attention, x)
    after.square().mean().backward()
    finite = [p for p in parameters if p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)]
    if not finite or any(p.grad is not None for p, _ in original):
        raise AssertionError("Missing adapter gradient or gradient entered original weights")
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not torch.equal(p.detach().cpu(), value) for p, value in original):
        raise AssertionError("An original parameter changed")
    with torch.no_grad():
        changed = attention_call(attention, x).detach()
        fresh, cache = attention_call(attention, x, cached=True)
    if torch.equal(changed, before) or not torch.allclose(fresh, cache, atol=1e-5, rtol=1e-4):
        raise AssertionError("Nonzero adapter ignored or incompatible with native caching")
    _save_native(model, info, changed_path, "tabpfn")
    reloaded, *_ = _load_native(base if mode == "diagscale" else changed_path, track, "tabpfn")
    reloaded.eval().requires_grad_(False)
    if mode == "diagscale":
        bundle = torch.load(changed_path, map_location="cpu", weights_only=False)
        diagscale.inject(reloaded, family="tabpfn", base_path="checkpoints/" + base.name,
                         base_sha256=file_digest(base))
        diagscale.restore(reloaded, bundle["adapter_state"])
    with torch.no_grad():
        restored = attention_call(_attention(reloaded), x)
    if not torch.allclose(changed, restored, atol=1e-6, rtol=1e-5):
        raise AssertionError("Reloaded adapter attention differs from the live model")
    del reloaded, restored
    del model, attention, parameters, finite, original, optimizer, after, info
    torch.cuda.empty_cache()
    tolerance = dict(atol=limits().prediction_atol, rtol=limits().prediction_rtol)
    errors = {}
    for precision in ("float32", "automatic"):
        original_pred = _predictions(base, track, "native", precision=precision)
        zero_pred = _predictions(zero_path, track, mode, precision=precision)
        changed_pred = _predictions(changed_path, track, mode, precision=precision)
        if not np.allclose(original_pred, zero_pred, **tolerance):
            raise AssertionError(f"Zero export changes {precision} predictions")
        if not np.isfinite(changed_pred).all() or np.array_equal(zero_pred, changed_pred):
            raise AssertionError("Nonzero exported adapter is missing or nonfinite at inference")
        errors[precision] = dict(zero_error=float(np.max(np.abs(original_pred-zero_pred))),
                                 adapted_effect=float(np.max(np.abs(zero_pred-changed_pred))))
    return dict(zero_identity=True, base_unchanged=True, native_cache_equal=True, reload_equal=True,
                predictions=errors, artifacts=[_binding(zero_path), _binding(changed_path)])


def main(argv=None):
    from src.utils.paths import results_dir
    from src.utils.legacy_design import scope
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args, training = parser.parse_known_args(argv)
    if training[:1] == ["--"]:
        training = training[1:]
    identifier = validate_id(os.environ.get("CREDITPFN_LEGACY_ID", ""))
    config = Path(training[training.index("--config") + 1]).resolve()
    if not config.is_relative_to((root(identifier) / "configs/debug").resolve()):
        raise ValueError("Only this extension's debug configs may use the probe")
    with scope(identifier):
        if not args.report.resolve().is_relative_to((results_dir(experiment="experiment0") / "gradients").resolve()):
            raise ValueError("Gradient evidence must stay in the extension's output")
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME") != "mindwell"
            or not torch.cuda.is_available() or "B200" not in torch.cuda.get_device_name()):
        raise RuntimeError("Legacy runtime probe requires a submitted Mindwell B200 allocation")
    return run_observed(training, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
