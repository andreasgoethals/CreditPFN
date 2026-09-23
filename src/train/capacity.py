"""Synthetic forward/backward capacity measurements for both model families.

These screens exclude AdamW state, L2-SP anchors and evaluation overhead. A real
pilot must establish the operating margin before any measured row cap is raised.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path


def _measure_rows(model, rows_grid, device, make_loss) -> None:
    import torch

    cuda = torch.device(device).type == "cuda"
    print(f"{'rows':>8} {'peak_alloc':>11} {'peak_resv':>10} {'batch+bwd_s':>12}  note")
    for n_rows in rows_grid:
        if n_rows < 2:
            raise ValueError("A probe needs at least one context and one query row")
        loss = None
        model.zero_grad(set_to_none=True)
        try:
            torch.manual_seed(0)
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = make_loss(n_rows)
            loss.backward()
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            peak = f"{torch.cuda.max_memory_allocated() / 1e9:.2f}G" if cuda else "n/a"
            reserved = f"{torch.cuda.max_memory_reserved() / 1e9:.2f}G" if cuda else "n/a"
            print(f"{n_rows:>8,} {peak:>11} {reserved:>10} {elapsed:>12.2f}  "
                  f"loss={float(loss.detach()):.4f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{n_rows:>8,} {'OOM':>11} {'OOM':>10} {'-':>12}  exceeds capacity")
        finally:
            # Release failed graphs and gradients before trying the next row count.
            loss = None
            model.zero_grad(set_to_none=True)
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()


def _checkpoint(base_path, track, n_estimators, freeze_backbone, query_fraction, n_features):
    from src.utils.paths import resolve_base_checkpoint

    if track not in {"pd", "lgd"} or n_estimators < 1 or n_features < 1:
        raise ValueError("Invalid probe track, member count or feature width")
    if not 0 < query_fraction < 1:
        raise ValueError("query_fraction must be between zero and one")
    ckpt = resolve_base_checkpoint(base_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Probe base checkpoint not found: {ckpt}")
    print(f"\n=== {Path(base_path).name} track={track} features={n_features} "
          f"query_fraction={query_fraction} members={n_estimators} "
          f"frozen_backbone={freeze_backbone} AMP=BF16 ===")
    return ckpt


def probe_base(base_path: str, track: str, rows_grid: list[int], device: str,
               n_estimators: int = 2, *, freeze_backbone: bool = False,
               query_fraction: float = 0.4, n_features: int = 64) -> None:
    import torch
    from src.train.model import load_tabpfn_for_training
    from src.train.dataloader import TabPFNBatch
    from src.train.loop import _forward, _classification_loss, _regression_loss, _n_classes

    ckpt = _checkpoint(base_path, track, n_estimators, freeze_backbone, query_fraction, n_features)
    model, criterion, _, _ = load_tabpfn_for_training(
        str(ckpt), track=track, device=device, lora_config=None, freeze_backbone=freeze_backbone)
    model.train()

    def make_loss(n_rows):
        n_query = min(n_rows - 1, max(1, int(n_rows * query_fraction)))
        n_ctx = n_rows - n_query
        X = torch.randn(n_rows, n_estimators, n_features)
        y = ((torch.rand(n_rows, n_estimators, 1) < 0.15).long() if track == "pd"
             else torch.rand(n_rows, n_estimators, 1))
        batch = TabPFNBatch(
            X_context=X[:n_ctx], y_context=y[:n_ctx],
            X_query=X[n_ctx:], y_query=y[n_ctx:], categorical_idx=[],
            task_type="classification" if track == "pd" else "regression",
            dataset_id=f"probe_{n_rows}",
        ).to(device)
        logits, target, _, _ = _forward(model, batch)
        if track == "pd":
            return _classification_loss(logits, batch.y_query, n_classes=_n_classes(batch),
                                        criterion=criterion)
        return _regression_loss(logits, target, criterion=criterion)

    _measure_rows(model, rows_grid, device, make_loss)


def probe_tabicl_base(base_path: str, track: str, rows_grid: list[int], device: str,
                      n_estimators: int = 2, *, freeze_backbone: bool = False,
                      query_fraction: float = 0.4, n_features: int = 64) -> None:
    import torch
    from src.train.tabicl_model import load_tabicl_for_training, tabicl_pinball_loss

    ckpt = _checkpoint(base_path, track, n_estimators, freeze_backbone, query_fraction, n_features)
    model, _ = load_tabicl_for_training(str(ckpt), track=track, device=device,
                                      freeze_backbone=freeze_backbone)
    model.train()

    def make_loss(n_rows):
        n_query = min(n_rows - 1, max(1, int(n_rows * query_fraction)))
        n_ctx = n_rows - n_query
        X = torch.randn(n_estimators, n_rows, n_features, device=device)
        y = ((torch.rand(n_estimators, n_rows, device=device) < 0.15).float() if track == "pd"
             else torch.randn(n_estimators, n_rows, device=device))
        out = model(X, y[:, :n_ctx])
        if track == "pd":
            n_classes = 2  # Synthetic PD is binary even if a tiny context contains one class.
            return torch.nn.functional.cross_entropy(out[..., :n_classes].reshape(-1, n_classes),
                                                      y[:, n_ctx:].long().reshape(-1))
        return tabicl_pinball_loss(out, y[:, n_ctx:])

    _measure_rows(model, rows_grid, device, make_loss)


def probe_configured_bases(cfg, tracks, rows_grid=None, *, device="cuda", frozen_modes=(False, True)):
    """Probe the configured members and query fraction with family-specific row grids."""
    from omegaconf import OmegaConf
    from src.train.config import training_members
    from src.train.tabicl_compat import model_family

    query_fraction = float(cfg.tunable.query_fractions[0])
    n_features = int(OmegaConf.load("config/data.yaml").sanitize.max_columns)
    for track in tracks:
        bases = (cfg.tunable.classifier_base_paths if track == "pd"
                 else cfg.tunable.regressor_base_paths)
        for base in bases:
            name = str(base)
            members = training_members(cfg, track, model_family(name))
            if model_family(name) == "tabicl":
                probe, grid = probe_tabicl_base, [10_000, 26_000, 40_000, 60_000]
            elif "v2.6" in name:
                probe, grid = probe_base, [9_000, 11_000, 14_000, 20_000]
            elif "v3" in name:
                probe, grid = probe_base, [20_000, 26_000, 40_000, 50_000]
            else:
                probe, grid = probe_base, [8_000, 10_000, 14_000]
            for frozen in frozen_modes:
                probe(name, track, rows_grid or grid, device, n_estimators=members,
                      freeze_backbone=frozen, query_fraction=query_fraction, n_features=n_features)
