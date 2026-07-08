#!/usr/bin/env python
"""Row-cap capacity probe: measure TRAINING peak GPU memory + step time as the
in-context row count grows, for both TabPFN bases.

WHY (2026-07-04): the Jul-3 run measured tiny training peaks on the 192 GiB
B200 (v3: 0.93 GB @ 20k rows; v2.6: 0.24 GB @ 9k rows), so the row caps in
``config/data.yaml`` were raised (v3 → 100k, v2.6 → 30k) based on the papers'
scaling laws — v3 ≈ linear in cells (row-chunked attention), v2.6 ≈ quadratic
in rows (dual attention, O(r²·min(c,500))). This probe VERIFIES those
projections empirically before a full sweep commits to them: it loads each
base, runs a real forward+backward (the training objective) on synthetic data
with the corpus feature width, and reports peak memory + step time per row
count.

Run on a GPU node (see scripts/slurm/probe_row_cap.slurm), ~10-20 min:

    python scripts/probe_row_cap.py                      # both bases, default grid
    python scripts/probe_row_cap.py --track pd --rows 20000 50000 100000 150000

Reads the same base checkpoints the training pipeline uses
(cfg.tunable.<track>_base_paths). Output: one table per base; a row that OOMs
is reported as OOM and the probe continues with the next size.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

N_FEATURES = 64          # sanitize.max_columns — the corpus feature width
QUERY_FRACTION = 0.20


def probe_base(base_path: str, track: str, rows_grid: list[int], device: str) -> None:
    import torch
    from src.train.model import load_tabpfn_for_training
    from src.train.dataloader import TabPFNBatch
    from src.train.loop import _forward, _classification_loss, _regression_loss, _n_classes
    from src.utils.paths import resolve_staging_path

    # Base checkpoints live in project STAGING, not the repo. The config lists
    # them as repo-relative paths (e.g. "checkpoints/tabpfn-v3-...ckpt"); resolve
    # through the staging root, mirroring loop.py's base_checkpoint_path.
    ckpt = resolve_staging_path(base_path)
    print(f"\n=== {Path(base_path).name}  (track={track}, {N_FEATURES} features, "
          f"qf={QUERY_FRACTION}) ===")
    if not Path(ckpt).exists():
        print(f"  SKIP: base checkpoint not found at {ckpt}")
        return
    model, criterion, _arch, _inf = load_tabpfn_for_training(
        str(ckpt), track=track, device=device, lora_config=None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    print(f"{'rows':>8} {'ctx':>8} {'query':>7} {'peak_alloc':>11} "
          f"{'peak_resv':>10} {'fwd+bwd_s':>10}  note")

    for n_rows in rows_grid:
        n_query = max(1, int(n_rows * QUERY_FRACTION))
        n_ctx = n_rows - n_query
        try:
            torch.manual_seed(0)
            X = torch.randn(n_rows, 1, N_FEATURES, dtype=torch.float32)
            if track == "pd":
                y = (torch.rand(n_rows, 1, 1) < 0.15).long()   # ~credit default rate
            else:
                y = torch.rand(n_rows, 1, 1, dtype=torch.float32)  # LGD in [0,1]
            batch = TabPFNBatch(
                X_context=X[:n_ctx], y_context=y[:n_ctx],
                X_query=X[n_ctx:], y_query=y[n_ctx:],
                categorical_idx=[], task_type=(
                    "classification" if track == "pd" else "regression"),
                dataset_id=f"probe_{n_rows}",
            ).to(device)

            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                pred_logits, y_target, _, _ = _forward(model, batch)
                if track == "pd":
                    loss = _classification_loss(
                        pred_logits, batch.y_query,
                        n_classes=_n_classes(batch), criterion=criterion,
                    )
                else:
                    loss = _regression_loss(pred_logits, y_target, criterion=criterion)
            loss.backward()
            optimizer.zero_grad(set_to_none=True)   # free grads; don't step
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.monotonic() - t0
            if device == "cuda":
                pa = torch.cuda.max_memory_allocated() / 1e9
                pr = torch.cuda.max_memory_reserved() / 1e9
                print(f"{n_rows:>8,} {n_ctx:>8,} {n_query:>7,} {pa:>10.2f}G "
                      f"{pr:>9.2f}G {dt:>10.2f}  loss={float(loss):.4f}")
            else:
                print(f"{n_rows:>8,} {n_ctx:>8,} {n_query:>7,} {'n/a':>11} "
                      f"{'n/a':>10} {dt:>10.2f}  (cpu)")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{n_rows:>8,} {'-':>8} {'-':>7} {'OOM':>11} {'OOM':>10} "
                  f"{'-':>10}  << row cap must stay below this")
        except Exception as exc:                               # noqa: BLE001
            print(f"{n_rows:>8,}  FAILED: {type(exc).__name__}: {exc}")

    del model, criterion, optimizer
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--track", choices=["pd", "lgd", "both"], default="both")
    ap.add_argument("--rows", type=int, nargs="+", default=None,
                    help="row counts to probe (default: per-base grid)")
    args = ap.parse_args()

    import torch
    from omegaconf import OmegaConf
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA device — timing only, no memory numbers.")
    cfg = OmegaConf.load("config/train.yaml")

    tracks = ["pd", "lgd"] if args.track == "both" else [args.track]
    for track in tracks:
        bases = (cfg.tunable.classifier_base_paths if track == "pd"
                 else cfg.tunable.regressor_base_paths)
        for base in bases:
            is_v26 = "v2.6" in str(base)
            grid = args.rows or (
                [9_000, 20_000, 30_000, 50_000] if is_v26         # quadratic-in-rows
                else [20_000, 50_000, 100_000, 200_000]           # ~linear-in-cells
            )
            probe_base(str(base), track, grid, device)
    print("\nRead-off: pick the largest row count whose peak_alloc leaves ~10x "
          "headroom AND whose step time you can afford x total_steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
