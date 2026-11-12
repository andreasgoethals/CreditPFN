"""Synthetic second-campaign evidence in the exact schemas the loaders read.

Used only by tests (and to look at the figures before real results exist). Values are random
draws with plausible scales; they are not results and are never written to `output CreditPFN/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASES = {"TabPFN v2": "tabpfn-v2", "TabPFN v2.6": "tabpfn-v2.6", "TabPFN v3": "tabpfn-v3", "TabICLv2": "tabicl"}
DESIGN_NAMES = {"TabPFN v2": "v2", "TabPFN v2.6": "v2.6", "TabPFN v3": "v3", "TabICLv2": "tabicl-v2"}
CREDIT = {"pd": [f"synthetic_pd_{i:02d}" for i in range(17)], "lgd": [f"synthetic_lgd_{i:02d}" for i in range(8)]}
RETENTION = {"pd": [f"openml_{46900 + i}" for i in range(19)], "lgd": [f"openml_{46950 + i}" for i in range(12)]}
METRIC = {"pd": "roc_auc", "lgd": "r2"}


def _source(base: str, kind: str) -> str:
    return ("tabicl-" if base == "tabicl" else "tabpfn-") + kind


def _scores(rng, track, n, level):
    if track == "pd":
        auc = np.clip(level + rng.normal(0, 0.004, n), 0.5, 1)
        return dict(roc_auc=auc, log_loss=0.45 - 0.3 * (auc - 0.7) + rng.normal(0, .003, n),
                    brier_score=0.14 - 0.1 * (auc - 0.7) + rng.normal(0, .002, n),
                    ece=np.abs(rng.normal(0.02, 0.008, n)), calibration_slope=rng.normal(1.0, 0.08, n),
                    calibration_in_the_large=rng.normal(0.0, 0.05, n))
    r2 = level + rng.normal(0, 0.01, n)
    return dict(r2=r2, nrmse=np.sqrt(np.clip(1 - r2, 0.01, None)), mae=0.25 - 0.1 * r2 + rng.normal(0, .003, n),
                crps=0.15 - 0.05 * r2 + rng.normal(0, .002, n), bias=rng.normal(0, 0.01, n),
                prediction_std_ratio=rng.normal(0.85, 0.05, n))


def results(slug: str, track: str, arms: int = 1, seed: int = 0, *, baselines: bool = True,
            effect: float = 0.003, missing_fold: bool = True) -> pd.DataFrame:
    """Fold-level metric rows as written by the benchmark (status OK), for one experiment slug:
    the original models, every arm's CPT model and, for the reference, CatBoost and linear."""
    rng = np.random.default_rng(seed)
    rows = []
    tables = CREDIT[track] + RETENTION[track]
    levels = np.random.default_rng(99).uniform(*((0.62, 0.9) if track == "pd" else (0.05, 0.6)), len(tables))
    for table, level in zip(tables, levels):   # the same table difficulty in every experiment
        credit = not table.startswith("openml_")
        for model, base in BASES.items():
            arch = f"{base}-{'classifier' if track == 'pd' else 'regressor'}"
            shift = (effect if credit else -effect / 2) * (2.0 if model == "TabPFN v2" else 0.5)
            for fold in range(5):
                if slug == "reference":
                    rows.append(dict(model_name=f"{_source(base, 'untuned')}[{arch}]",
                                     model_source=_source(base, "untuned"), test_dataset_id=table, fold_idx=fold,
                                     status="OK", **{k: float(v[0]) for k, v in _scores(rng, track, 1, level).items()}))
                for arm in range(arms):
                    arm_shift = shift * (1 - 0.6 * arm) + rng.normal(0, abs(effect) * 0.5)
                    rows.append(dict(model_name=f"{_source(base, 'trained')}[cpt2_{slug}_a{arm}_s00_{track}_{arch}]",
                                     model_source=_source(base, "trained"), test_dataset_id=table, fold_idx=fold,
                                     status="OK",
                                     **{k: float(v[0]) for k, v in _scores(rng, track, 1, level + arm_shift).items()}))
            if slug == "reference" and baselines and model == "TabPFN v2":
                for fold in range(5):
                    for name, delta in (("catboost", 0.002), ("logreg" if track == "pd" else "linreg", -0.03)):
                        rows.append(dict(model_name=name, model_source="baseline", test_dataset_id=table,
                                         fold_idx=fold, status="OK",
                                         **{k: float(v[0]) for k, v in _scores(rng, track, 1, level + delta).items()}))
    frame = pd.DataFrame(rows)
    if missing_fold:   # one failed fold: that table must leave every summary
        frame = frame[~((frame.test_dataset_id == CREDIT[track][-1]) & (frame.fold_idx == 4)
                        & frame.model_source.str.endswith("trained"))]
    return frame


def install_results(monkeypatch, slugs_arms: dict, **kwargs) -> None:
    """Route ``campaign2._results_frames`` to synthetic frames: {slug: number of arms}."""
    from src.visualize import campaign2
    cache = {(slug, track): results(slug, track, arms, seed=i, **kwargs)
             for i, (slug, arms) in enumerate(slugs_arms.items()) for track in ("pd", "lgd")}
    monkeypatch.setattr(campaign2, "_results_frames",
                        lambda slug, track: cache.get((slug, track), pd.DataFrame()).copy())


def trajectories(slug: str, track: str, arms: int = 1, seed: int = 0, *, updates=None,
                 partitions: int = 4, speed: float = 1.0) -> pd.DataFrame:
    """Monitor records as ``load_trajectories`` returns them: one row per run and monitor point,
    with ``score__<scope>__<table>__<metric>`` columns, rows-processed milestones and diagnostics."""
    from src.utils.design import monitor_updates
    rng = np.random.default_rng(seed)
    updates = updates or monitor_updates(10000)
    metric = METRIC[track]
    credit = CREDIT[track]
    rows = []
    for arm in range(arms):
        for model in BASES:
            for partition in range(partitions):
                held = credit[partition::partitions]
                train = [t for t in credit if t not in held]
                start = {(scope, t): rng.uniform(0.6, 0.9) if track == "pd" else rng.uniform(0.1, 0.6)
                         for scope, tables in (("test", held), ("train", train), ("ood", RETENTION[track]))
                         for t in tables}
                rate = (0.004 if model == "TabPFN v2" else 0.001) * speed * (1 + arm)
                run = f"cpt2_{slug}_a{arm}_s{partition:02d}_{track}_{BASES[model]}"
                rows_per_update = 6000 * (1 + arm)
                elapsed = 0.0
                for kind, step in [("trajectory", u) for u in updates] + [
                        ("rows_milestone", int(m / rows_per_update) + 1) for m in (1e6, 2e6, 5e6, 10e6, 20e6, 50e6)
                        if m / rows_per_update < max(updates)]:
                    progress = 1 - np.exp(-step / 1500)
                    elapsed = step * 1.2 * (1 + arm)
                    record = dict(run=run, slug=slug, arm=arm, partition=partition, track=track, model=model,
                                  record_type=kind, successful_updates=step, processed_rows=step * rows_per_update,
                                  elapsed_sec=elapsed, monitor_seconds=30.0,
                                  cuda_peak_reserved_bytes=(20 + 5 * arm) * 1e9,
                                  weight_drift=1e-4 * (1 + arm) * np.sqrt(step) if step else 0.0)
                    for (scope, table), value in start.items():
                        effect = {"test": rate, "train": 2 * rate, "ood": -rate}[scope] * progress
                        noise = rng.normal(0, 0.0007) if step else 0.0
                        record[f"score__{scope}__{table}__{metric}"] = value + effect + noise
                    record["test_metric"] = np.mean([v for k, v in record.items() if k.startswith("score__test__")])
                    rows.append(record)
    return pd.DataFrame(rows)


def transfer_trajectories(track: str, seed: int = 0) -> pd.DataFrame:
    """E11: one short run per source table (arm = source index) scoring every other table."""
    rng = np.random.default_rng(seed)
    metric = METRIC[track]
    tables = CREDIT[track]
    rows = []
    for model in BASES:
        for arm, source in enumerate(tables):
            run = f"cpt2_transfer_a{arm}_s00_{track}_{BASES[model]}"
            base = {t: rng.uniform(0.6, 0.9) for t in tables if t != source}
            for step in (0, 250, 2500):
                record = dict(run=run, slug="transfer", arm=arm, partition=0, track=track, model=model,
                              record_type="trajectory", successful_updates=step, processed_rows=step * 5000)
                for target, value in base.items():
                    record[f"score__test__{target}__{metric}"] = value + (rng.normal(0.001, 0.004) if step else 0)
                rows.append(record)
    return pd.DataFrame(rows)


def finetune(seed: int = 0) -> pd.DataFrame:
    """E4 task rows: per track, model (design name), table and fold, both starts with and without
    fine-tuning, as ``task_results('finetune')`` returns them."""
    rng = np.random.default_rng(seed)
    rows = []
    for track in ("pd", "lgd"):
        for name in DESIGN_NAMES.values():
            for table in CREDIT[track]:
                level = rng.uniform(0.65, 0.85) if track == "pd" else rng.uniform(0.1, 0.5)
                for fold in range(5):
                    for start, shift in (("original", 0.0), ("cpt", 0.002)):
                        for tuned, gain in ((False, 0.0), (True, 0.006)):
                            rows.append(dict(track=track, model=name, start=start, finetuned=tuned, dataset_id=table,
                                             fold=fold, **{METRIC[track]: level + shift + gain + rng.normal(0, .002)}))
    return pd.DataFrame(rows)


def context(seed: int = 0, kind: str = "context") -> pd.DataFrame:
    """E10 (``context``) or E9 (``episode_context``) task rows with nested context sizes."""
    from src.utils.design import load_design
    design = load_design()
    rng = np.random.default_rng(seed)
    rows = []
    if kind == "context":
        grid = [int(v) for v in design["experiments"]["E10"]["context_rows"]]
        caps = design["experiments"]["E10"]["max_context_rows"]
        for track in ("pd", "lgd"):
            for name in DESIGN_NAMES.values():
                for i, table in enumerate(CREDIT[track]):
                    available = int(600 * 1.6 ** i)
                    sizes = [s for s in grid if s < min(available, caps[name])] + [min(available, caps[name])]
                    for fold in range(5):
                        for size in sizes:
                            base = (0.55 + 0.05 * np.log10(size)) if track == "pd" else (0.02 * np.log2(size))
                            for start, shift in (("original", 0.0), ("cpt", 0.01 / np.log2(size + 1))):
                                rows.append(dict(track=track, model=name, start=start, dataset_id=table, fold=fold,
                                                 context_rows=size, available_context=available,
                                                 **{METRIC[track]: base + shift + rng.normal(0, .002)}))
        return pd.DataFrame(rows)
    sizes = design["experiments"]["E9"]["arms"]
    caps = design["experiments"]["E10"]["max_context_rows"]
    for arm, spec in enumerate(sizes):
        models = spec.get("models", design["models"])
        for name in DESIGN_NAMES.values():
            if name not in models:
                continue
            for i, table in enumerate(CREDIT["pd"]):
                available = int(600 * 1.6 ** i)
                for fold in range(5):
                    for size in [s for s in (2048, 8192) if s < available] + [min(available, caps[name])]:
                        rows.append(dict(track="pd", model=name, start="cpt", arm=arm, dataset_id=table, fold=fold,
                                         context_rows=size, available_context=available,
                                         roc_auc=0.55 + 0.05 * np.log10(size) + 0.001 * arm + rng.normal(0, .002)))
    return pd.DataFrame(rows)
