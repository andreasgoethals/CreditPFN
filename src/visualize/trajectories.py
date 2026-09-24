"""Fixed-update, within-dataset effects for the descriptive experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.visualize import style
from src.visualize.training_viz import _resolve_paths, parse_trial_name, compact_base


def load_trajectories(track: str, cfg=None) -> pd.DataFrame:
    from src.utils.consolidate_output import matches_run, read_table
    from src.visualize.inputs import load_consolidated
    paths = _resolve_paths(cfg)
    frame = load_consolidated(paths["run_name"], f"training_{track}", manifest_root=paths["manifest_dir"])
    if frame is None:
        frames = []
        for path in sorted((paths["epoch_dir"] / track).glob("*.trajectory.csv")):
            if matches_run(path.name, paths["run_name"], track=track):
                item = read_table(path)
                item["trial_name"] = path.name.removesuffix(".trajectory.csv")
                frames.append(item)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty or "record_type" not in frame:
        return pd.DataFrame()
    frame = frame[frame.record_type.eq("trajectory")].copy()
    if frame.duplicated(["trial_name", "successful_updates"]).any():
        raise ValueError("Duplicate trajectory updates: reconcile the trial's progress records")
    return frame


def trajectory_effects(frame: pd.DataFrame, track: str, *, split="test") -> pd.DataFrame:
    """Pair each observation with update zero on the same trial and dataset.

    Positive effects mean improvement: AUC minus initial AUC; LGD fractional RMSE
    reduction. Missing baselines/nonfinite values remain unavailable, never zero.
    """
    from src.data.dataset_names import display_name
    if frame.empty:
        return pd.DataFrame()
    prefix = f"metric__{split}__"
    rows = []
    for name, group in frame.groupby("trial_name", sort=True):
        trial = parse_trial_name(name)
        if trial is None:
            raise ValueError(f"Unrecognized trajectory trial name: {name}")
        baseline = group[group.successful_updates.eq(0)]
        if len(baseline) != 1:
            raise ValueError("Each trajectory requires exactly one update-zero measurement")
        for column in (c for c in frame if c.startswith(prefix)):
            initial = float(baseline.iloc[0][column])
            if not np.isfinite(initial) or (track == "lgd" and initial <= 0):
                continue
            for _, row in group.iterrows():
                value = float(row[column])
                if not np.isfinite(value):
                    continue
                effect = value - initial if track == "pd" else (initial - value) / initial
                rows.append({"trial": name, "dataset": display_name(column.removeprefix(prefix)),
                    "base": compact_base(trial.base_short), "learning_rate": trial.lr,
                    "frozen": trial.lora, "l2sp_lambda": trial.l2sp_lambda,
                    "seed": trial.seed, "updates": int(row.successful_updates),
                    "processed_rows": row.processed_rows, "effect": effect,
                    "sampling": "accumulate" if "_accumulate" in name else "full_pass" if "_fullpass" in name else "one_sample"})
    return pd.DataFrame(rows)


def plot_trajectories(track: str, *, cfg=None, sampling="one_sample") -> dict:
    import matplotlib.pyplot as plt
    effects = trajectory_effects(load_trajectories(track, cfg), track)
    if effects.empty:
        return {}
    effects = effects[effects.sampling.eq(sampling)]
    figures = {}
    for base, data in effects.groupby("base", sort=True):
        fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL), sharey=True,
                                 layout="constrained")
        for ax, frozen in zip(axes, (False, True)):
            selected = data[data.frozen.eq(frozen)]
            for (lr, lam), recipe in selected.groupby(["learning_rate", "l2sp_lambda"]):
                # Average repetitions within each dataset first, then give each dataset one vote.
                per_dataset = recipe.groupby(["dataset", "updates"]).effect.mean()
                curve = per_dataset.groupby("updates").mean()
                expected = len(recipe[recipe.updates.eq(0)])
                counts = recipe.groupby("updates").size()
                curve = curve.where(counts.eq(expected))
                ax.plot(curve.index, curve, color=style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr))),
                        linestyle=style.TRAJECTORY_LINESTYLES.get(lam, ":"), label=f"{lr:.0e}, {lam:g}")
            ax.axhline(0, color=style.color("reference"), linestyle=":")
            style.title(ax, f"{base}: {'frozen backbone' if frozen else 'full updates'}")
            ax.set_xlabel("Successful optimizer updates")
            if not selected.empty:
                ax.legend(title="Peak LR, L2-SP", ncol=2)
        axes[0].set_ylabel("AUC change from update 0" if track == "pd" else "Fractional RMSE reduction from update 0")
        figures[base] = fig
    return figures
