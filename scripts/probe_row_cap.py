#!/usr/bin/env python
"""Row-cap capacity probe: measure TRAINING peak GPU memory + step time as the
in-context row count grows, for every base in the sweep (both families).

WHY (2026-07-04): an earlier monitor-based measurement reported implausibly
tiny peaks because it was not measuring the training step. This probe instead
loads each base and runs a real forward+backward objective on synthetic data
with the corpus feature width. The 2026-07-08 B200 results were approximately
linear per member (v3 ≈2.5 GB and v2.6 ≈5.7 GB per 1 000 rows) and established
the current two-member caps of 26k / 11k; the training loop scales them down
for larger ensembles. Re-run this probe before changing caps, feature width,
hardware, or ensemble size.

Run on a GPU node (see scripts/slurm/probe_row_cap.slurm), ~10-20 min:

    python scripts/probe_row_cap.py                      # both bases, default grid
    python scripts/probe_row_cap.py --track pd --rows 20000 50000 100000 150000

Reads the same base checkpoints the training pipeline uses
(cfg.tunable.<track>_base_paths). Output: one table per base; a row that OOMs
is reported as OOM and the probe continues with the next size.

TABICL (2026-08-04): its bases go through a separate probe path because the
loader, batch layout and loss all differ. Its in-config cap (10 000 rows) is
upstream's own chunk size, NOT a measurement — this probe is how to replace it
with one. Both families are probed with the SAME per-step ensemble size the
training loop would use (TabPFN: 1 member here, then scaled; TabICL: 2), since
peak memory is roughly members x per-member cost.
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


def probe_tabicl_base(base_path: str, track: str, rows_grid: list[int],
                      device: str, n_estimators: int = 2) -> None:
    """TabICL variant of :func:`probe_base`.

    Differences that matter for the measurement: the meta-batch is
    ``(E, rows, features)`` with all E members forwarded in ONE graph before a
    single backward (so peak memory scales with E), ``recompute=True`` turns on
    gradient checkpointing in all three stages, and the loss is upstream's own
    (CE over the first n_classes logits / mean pinball over 999 quantiles).
    """
    import torch
    from src.train.tabicl_model import load_tabicl_for_training, tabicl_pinball_loss
    from src.utils.paths import resolve_staging_path

    ckpt = resolve_staging_path(base_path)
    print(f"\n=== {Path(base_path).name}  (track={track}, {N_FEATURES} features, "
          f"qf={QUERY_FRACTION}, n_estimators={n_estimators}, recompute=True) ===")
    if not Path(ckpt).exists():
        print(f"  SKIP: base checkpoint not found at {ckpt}")
        return
    model, _cfg = load_tabicl_for_training(
        str(ckpt), track=track, device=device, freeze_backbone=False,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    print(f"{'rows':>8} {'ctx':>8} {'query':>7} {'peak_alloc':>11} "
          f"{'peak_resv':>10} {'fwd+bwd_s':>10}  note")

    for n_rows in rows_grid:
        n_query = max(1, int(n_rows * QUERY_FRACTION))
        n_ctx = n_rows - n_query
        try:
            torch.manual_seed(0)
            E = int(n_estimators)
            X = torch.randn(E, n_rows, N_FEATURES, dtype=torch.float32)
            if track == "pd":
                y_all = (torch.rand(E, n_rows) < 0.15).float()   # ~default rate
            else:
                y_all = torch.randn(E, n_rows)                   # z-normed targets
            X = X.to(device)
            y_train = y_all[:, :n_ctx].to(device)
            y_query = y_all[:, n_ctx:].to(device)

            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                out = model(X, y_train)
                if track == "pd":
                    n_cls = int(y_train.max().item()) + 1
                    loss = torch.nn.functional.cross_entropy(
                        out[..., :n_cls].reshape(-1, n_cls),
                        y_query.long().reshape(-1),
                    )
                else:
                    loss = tabicl_pinball_loss(out, y_query)
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

    del model, optimizer
    if device == "cuda":
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
            from src.train.tabicl_compat import model_family
            if model_family(str(base)) == "tabicl":
                # Quadratic-in-rows ICL attention, probed at the training
                # ensemble size (2), so these numbers are directly comparable
                # to config/data.yaml's finetuning.max_rows_per_epoch.tabicl.
                grid = args.rows or [5_000, 10_000, 20_000, 30_000]
                probe_tabicl_base(str(base), track, grid, device)
                continue
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
