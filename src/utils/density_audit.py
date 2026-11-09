"""Experiment 0: bucket-density roughness of every original TabPFN regressor on every LGD table.

Uses upstream ``TabPFNRegressor`` (not CreditPFN's training path) on one fixed random
context/query split per table, and records the roughness that
``src.train.density_checks`` uses, the bar-distribution NLL of the query targets and R2.
Descriptive evidence for the open v2.6 LGD objective decision; it changes nothing.

    python -m src.utils.density_audit --out <file.json>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MODELS = {"v2": "checkpoints/tabpfn-v2-regressor-v2_default.ckpt",
          "v2.6": "checkpoints/tabpfn-v2.6-regressor-v2.6_default.ckpt",
          "v3": "checkpoints/tabpfn-v3-regressor-v3_default.ckpt"}
CONTEXT, QUERY, SEED = 600, 400, 0


def split(n: int, seed: int = SEED):
    order = np.random.default_rng(seed).permutation(n)
    context = order[:min(CONTEXT, n // 2)]
    query = order[len(context):len(context) + min(QUERY, n - len(context))]
    return context, query


def table_row(model: str, path: Path, data, device: str) -> dict:
    import torch
    from tabpfn import TabPFNRegressor

    from src.train.density_checks import ROUGHNESS_LIMIT, bucket_roughness
    context, query = split(len(data.y))
    x, y = data.X, np.asarray(data.y, dtype=float)
    regressor = TabPFNRegressor(model_path=str(path), n_estimators=2, device=device,
                                random_state=0, ignore_pretraining_limits=True)
    regressor.fit(x.iloc[context], y[context])
    out = regressor.predict(x.iloc[query], output_type="full")
    logits = out["logits"].float().cpu()
    criterion = regressor.znorm_space_bardist_
    z = torch.as_tensor((y[query] - regressor.y_train_mean_) / regressor.y_train_std_, dtype=torch.float32)
    nll = criterion(logits.unsqueeze(1), z.reshape(-1, 1)).squeeze().float()
    finite = torch.isfinite(nll)
    roughness = bucket_roughness(logits.softmax(-1))
    mean = np.asarray(out["mean"], dtype=float)
    variance = float(np.var(y[query]))
    return {"model": model, "dataset": data.dataset_id, "context_rows": int(len(context)),
            "query_rows": int(len(query)),
            "roughness_median": float(torch.nanmedian(roughness)),
            "roughness_limit": ROUGHNESS_LIMIT,
            "exceeds_limit": bool(float(torch.nanmedian(roughness)) > ROUGHNESS_LIMIT),
            "nll_median": float(nll[finite].median()) if finite.any() else None,
            "nll_nonfinite": int((~finite).sum()),
            "r2": float(1 - np.mean((y[query] - mean) ** 2) / variance) if variance > 0 else None}


def audit(device: str = "auto") -> dict:
    import torch

    from src.eval.dataset_loader import load_processed_dataset
    from src.train.corpus import build_dataset_pool
    from src.utils.paths import resolve_base_checkpoint
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, failures = [], []
    for ref in sorted(build_dataset_pool("lgd"), key=lambda r: r.dataset_id):
        data = load_processed_dataset("lgd", ref.dataset_id)
        for model, path in MODELS.items():
            try:
                result = table_row(model, resolve_base_checkpoint(path), data, device)
                rows.append(result)
            except Exception as exc:  # record and continue: this is an inventory, not a gate
                result = {"model": model, "dataset": ref.dataset_id, "error": repr(exc)[:300]}
                failures.append(result)
            print(result, flush=True)
    return {"split_seed": SEED, "context_rows": CONTEXT, "query_rows": QUERY, "device": device,
            "rows": rows, "failures": failures, "passed": not failures}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    from src.utils.atomic import write_json
    report = audit(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
