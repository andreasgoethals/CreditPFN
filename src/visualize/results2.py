"""Second-campaign publication figures (RQ1-RQ4).

Every function takes the frames produced by ``src.visualize.campaign2`` and returns a matplotlib
figure, or ``None`` when its evidence does not exist yet, so a notebook states "no results yet"
instead of drawing an empty or partial figure. Colours, sizes and markers come from
``src.visualize.style``; each figure attaches the exact plotted values as ``_report_tables`` so the
notebook summary repeats every plotted number.

Units of analysis: a five-fold result is one value per held-out table (folds are averaged first,
and only tables with all five paired folds enter); seeds and partitions are never pooled as if
they were tables. Monitor curves are equal-table means of one fixed context/query split.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_names import display_name
from src.visualize import style
from src.visualize.campaign2 import METRIC_LABELS, MONITOR_SCOPES

TRACKS = ("pd", "lgd")
TRACK_NAME = {"pd": "PD", "lgd": "LGD"}
SCORE = {"pd": "AUC", "lgd": "R²"}


# ----------------------------------------------------------------------------- helpers
def _empty(frame) -> bool:
    return frame is None or not isinstance(frame, pd.DataFrame) or frame.empty


def _models(frame: pd.DataFrame, column: str = "model") -> list[str]:
    present = list(dict.fromkeys(frame[column].dropna()))
    return [m for m in style.MODEL_ORDER if m in present] + sorted(m for m in present
                                                                   if m not in style.MODEL_ORDER)


def _tracks(frames: dict) -> list[str]:
    return [t for t in TRACKS if not _empty(frames.get(t))]


def _report(fig, **tables):
    fig._report_tables = {name.replace("_", " "): frame.reset_index(drop=True)
                          for name, frame in tables.items() if not _empty(frame)}
    return fig


def _zero(ax, axis: str = "y") -> None:
    (ax.axhline if axis == "y" else ax.axvline)(0, **style.ZERO_LINE)


def _proxy_line(ax, label, **kwargs) -> None:
    """A legend entry without data (``finish_figure`` gathers labelled artists)."""
    ax.plot([], [], label=label, **kwargs)


def _proxy_point(ax, label, **kwargs) -> None:
    ax.scatter([], [], label=label, **kwargs)


def _jitter(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-style.JITTER_WIDTH / 2, style.JITTER_WIDTH / 2, n)


def _model_proxies(ax, models) -> None:
    for model in models:
        _proxy_point(ax, model, color=style.model_color(model), s=style.POINT_SIZE)


def _height(rows: int, per_row: float) -> float:
    return min(style.MAX_HEIGHT, style.FIGURE_BAND + rows * per_row)


def _rows_axis(ax) -> None:
    """Log axis of processed rows with decade ticks only (milestones span less than 2 decades)."""
    from matplotlib.ticker import FixedLocator, LogFormatterSciNotation, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1e6, 1e7, 1e8]))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _plain_log_axis(ax, values) -> None:
    """Log axis of hours with plain 1-2-5 tick labels inside the data range."""
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    ax.set_xscale("log")
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    ticks = [m * 10 ** e for e in range(-2, 5) for m in (1, 2, 5) if low / 1.5 <= m * 10 ** e <= high * 1.5]
    if len(ticks) > 4:
        ticks = ticks[::2]
    ax.xaxis.set_major_locator(FixedLocator(ticks or [low]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------- training monitor figures
def monitor_development(curves: pd.DataFrame, band: pd.DataFrame | None = None, *, title: str,
                        x: str = "successful_updates"):
    """Rows: task; columns: model. Change of the equal-table mean monitor score from update 0 on
    the held-out credit tables (solid), the run's own training tables (dashed) and the non-credit
    retention panel (dotted); optional band: range over three training seeds (held-out credit)."""
    if _empty(curves):
        return None
    tracks = [t for t in TRACKS if t in set(curves.track)]
    models = _models(curves)
    fig, axes = plt.subplots(len(tracks), len(models), sharex=True, sharey="row", squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(len(tracks), style.ROW_HEIGHT * 1.2)))
    for r, track in enumerate(tracks):
        for c, model in enumerate(models):
            ax = axes[r, c]
            color = style.model_color(model)
            if not _empty(band):
                b = band[(band.track == track) & (band.model == model)].sort_values(x)
                if len(b):
                    ax.fill_between(b[x], b.low, b.high, color=color, alpha=style.SEED_BAND_ALPHA,
                                    linewidth=0)
            for scope in MONITOR_SCOPES:
                s = curves[(curves.track == track) & (curves.model == model) & (curves.scope == scope)]
                s = s.sort_values(x)
                if len(s):
                    ax.plot(s[x], s.change, color=color, linestyle=style.SCOPE_LINESTYLES[scope],
                            alpha=style.SCOPE_ALPHA[scope], linewidth=style.MEAN_CURVE_LINE * 0.7)
            _zero(ax)
            if r == 0:
                ax.set_title(model)
            if c == 0:
                ax.set_ylabel(f"{TRACK_NAME[track]}: Δ {SCORE[track]}")
            if r == len(tracks) - 1:
                ax.set_xlabel("Successful updates" if x == "successful_updates" else "Rows processed")
    for scope in MONITOR_SCOPES:
        if scope in set(curves.scope):
            _proxy_line(axes[0, 0], style.SCOPE_LABELS[scope], color=style.COLORS["mean"],
                        linestyle=style.SCOPE_LINESTYLES[scope])
    if not _empty(band):
        axes[0, 0].fill_between([], [], [], color=style.COLORS["untuned"], alpha=style.SEED_BAND_ALPHA * 2,
                                linewidth=0, label="Range over seeds 42, 43, 44 (held-out credit)")
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, monitor_curves=curves, seed_range=band)


def arm_styles(settings: pd.DataFrame, *, color_by: str = "label", linestyle_by: str | None = None,
               ordered: bool = True) -> pd.DataFrame:
    """Colour and line style per arm. The reference arm is always black and solid. Other arms are
    coloured by ``color_by``: the trainable-scope colours for ``trainable``, an ordered ramp
    (``ordered``) or the arm palette, which avoids the model hues. ``linestyle_by``:
    ``l2sp_lambda`` (E5) or ``rate_within_scope`` (E6: lower rate dashed, higher solid)."""
    if _empty(settings):
        return pd.DataFrame()
    arms = settings.copy()
    is_ref = arms.slug.eq("reference")
    keys = list(dict.fromkeys(arms.loc[~is_ref, color_by]))
    if color_by == "trainable":
        palette = {k: style.SCOPE_COLORS.get(k, style.color(k)) for k in keys}
    elif ordered:
        palette = style.ordered(sorted(keys))
    else:
        palette = {k: style.ARM_PALETTE[i % len(style.ARM_PALETTE)] for i, k in enumerate(keys)}
    arms["color"] = [style.COLORS["reference"] if ref else palette[key] for ref, key in zip(is_ref, arms[color_by])]
    if linestyle_by == "l2sp_lambda":
        arms["linestyle"] = [style.LAMBDA_LINESTYLES.get(float(v), "-") for v in arms.l2sp_lambda]
    elif linestyle_by == "rate_within_scope":
        rank = arms.groupby("trainable").learning_rate.rank(method="dense").astype(int) - 1
        arms["linestyle"] = [style.RATE_LINESTYLES[min(r, 1)] for r in rank]
    else:
        arms["linestyle"] = "-"
    arms.loc[is_ref, "linestyle"] = "-"
    arms["label"] = [("Reference (E1)" if ref else label) for ref, label in zip(is_ref, arms.label)]
    return arms


def arm_trajectories(rows: list[dict], arms: pd.DataFrame, *, title: str):
    """Rows: the given panels (each a curve frame, x column and y label); columns: model. One line
    per arm, coloured and styled by ``arms`` (``arm_styles``)."""
    rows = [r for r in rows if not _empty(r.get("curves"))]
    if not rows or _empty(arms):
        return None
    models = _models(pd.concat([r["curves"] for r in rows]))
    same_x = len({r["x"] for r in rows}) == 1
    fig, axes = plt.subplots(len(rows), len(models), sharey="row", sharex=same_x, squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(len(rows), style.ROW_HEIGHT * 1.2)))
    seen = set()
    for i, row in enumerate(rows):
        x = row["x"]
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = row["curves"][row["curves"].model == model]
            for spec in arms.itertuples():
                s = sub[(sub.slug == spec.slug) & (sub.arm == spec.arm)].sort_values(x)
                if s.empty:
                    continue
                seen.add((spec.slug, spec.arm))
                ax.plot(s[x], s.change, color=spec.color, linestyle=spec.linestyle,
                        linewidth=style.MEAN_CURVE_LINE * (0.85 if spec.slug == "reference" else 0.6),
                        marker="o" if x != "successful_updates" else None, markersize=2.5)
            if row.get("log_y"):
                ax.set_yscale("log")
            else:
                _zero(ax)
            if x != "successful_updates":
                _rows_axis(ax)
            if i == 0:
                ax.set_title(model)
            if j == 0:
                ax.set_ylabel(row["ylabel"])
            if not same_x or i == len(rows) - 1:
                ax.set_xlabel(row.get("xlabel", "Successful updates" if x == "successful_updates"
                                      else "Rows processed"))
    for spec in arms.itertuples():
        if (spec.slug, spec.arm) in seen:
            _proxy_line(axes[0, 0], spec.label, color=spec.color, linestyle=spec.linestyle)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, **{f"panel {i + 1} {r['ylabel']}": r["curves"] for i, r in enumerate(rows)})


# ------------------------------------------------------------------- RQ1 endpoint figures
def table_gains(gains: pd.DataFrame, seeds: pd.DataFrame | None = None, *, track: str, title: str):
    """Columns: model; rows: held-out credit tables. Five-fold gain of CPT over the original
    model per table (seed 42, E1); with E2, a line spans the three seeds' gains and a filled
    marker means all three agree in sign and exceed their range."""
    if _empty(gains):
        return None
    g = gains[gains.credit & gains.complete & (gains.arm == 0)]
    if g.empty:
        return None
    models = _models(g)
    order = list(g.groupby("test_dataset_id").gain.mean().sort_values().index)
    ypos = {t: i for i, t in enumerate(order)}
    fig, axes = plt.subplots(1, len(models), sharey=True, squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(len(order), style.TABLE_ROW_HEIGHT)))
    has_seeds = not _empty(seeds)
    used = []
    for ax, model in zip(axes[0], models):
        color = style.model_color(model)
        sub = g[g.model == model]
        beyond = set()
        if has_seeds:
            s = seeds[(seeds.model == model) & seeds.credit]
            columns = [c for c in s.columns if c.startswith("gain_seed")]
            for row in s.itertuples():
                if row.test_dataset_id in ypos:
                    values = [getattr(row, c) for c in columns]
                    ax.hlines(ypos[row.test_dataset_id], min(values), max(values), color=color,
                              linewidth=style.TABLE_CURVE_LINE * 1.6, alpha=0.8)
            beyond = set(s[s.beyond_noise].test_dataset_id)
            used.append(s.assign(panel=model))
        y = sub.test_dataset_id.map(ypos)
        filled = sub.test_dataset_id.isin(beyond) if has_seeds else pd.Series(True, index=sub.index)
        ax.scatter(sub.gain[filled], y[filled], s=style.POINT_SIZE, color=color, zorder=3)
        ax.scatter(sub.gain[~filled], y[~filled], s=style.POINT_SIZE, facecolors=style.HOLLOW_FACE,
                   edgecolors=color, zorder=3)
        ax.axvline(sub.gain.mean(), color=color, linestyle="--", linewidth=0.9)
        _zero(ax, "x")
        ax.set_title(model)
        ax.set_xlabel(f"Δ {SCORE[track]}")
        ax.grid(axis="x", alpha=0.25)
    axes[0, 0].set_yticks(range(len(order)), [display_name(t) for t in order])
    axes[0, 0].set_ylim(-0.7, len(order) - 0.3)
    ref = style.COLORS["mean"]
    if has_seeds:
        _proxy_point(axes[0, 0], "Seed 42; all seeds exceed their range", color=ref, s=style.POINT_SIZE)
        _proxy_point(axes[0, 0], "Seed 42; within seed range", facecolors=style.HOLLOW_FACE, edgecolors=ref,
                     s=style.POINT_SIZE)
        _proxy_line(axes[0, 0], "Range over seeds 42, 43, 44", color=ref, linewidth=style.TABLE_CURVE_LINE * 1.6)
    else:
        _proxy_point(axes[0, 0], "Seed 42 (E1)", color=ref, s=style.POINT_SIZE)
    _proxy_line(axes[0, 0], "Equal-table mean (seed 42)", color=ref, linestyle="--", linewidth=0.9)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, per_table_gain=g[["model", "test_dataset_id", "gain", "original", "folds"]],
                   seed_gains=pd.concat(used) if used else None)


def transfer_and_retention(gains: dict, *, title: str):
    """One panel per task. Per model: per-table five-fold gains on held-out credit tables (filled,
    left) and on the non-credit retention panel (hollow, right), with equal-table means."""
    tracks = [t for t in _tracks(gains) if not gains[t][gains[t].complete & (gains[t].arm == 0)].empty]
    if not tracks:
        return None
    fig, axes = plt.subplots(1, len(tracks), squeeze=False, figsize=style.figsize(style.WIDTH_FULL, 0.42))
    tables = []
    for ax, track in zip(axes[0], tracks):
        g = gains[track][gains[track].complete & (gains[track].arm == 0)]
        models = _models(g)
        for i, model in enumerate(models):
            color = style.model_color(model)
            for credit, offset in ((True, -0.18), (False, 0.18)):
                v = g[(g.model == model) & (g.credit == credit)].gain.to_numpy()
                if not len(v):
                    continue
                x = i + offset + _jitter(len(v), seed=i)
                if credit:
                    ax.scatter(x, v, s=style.TABLE_MARKER_SIZE, color=color, alpha=style.POINT_ALPHA,
                               linewidths=0)
                else:
                    ax.scatter(x, v, s=style.TABLE_MARKER_SIZE, facecolors=style.HOLLOW_FACE,
                               edgecolors=color, linewidths=0.6)
                ax.scatter([i + offset], [v.mean()], marker=style.MEAN_MARKER, s=style.MEAN_MARKER_SIZE,
                           color=color, edgecolors=style.COLORS["mean"], linewidths=0.6, zorder=4)
                tables.append(dict(track=track, model=model,
                                   tables="held-out credit" if credit else "non-credit retention",
                                   n=len(v), mean_gain=v.mean(), median_gain=float(np.median(v)),
                                   improved=int((v > 0).sum())))
        _zero(ax)
        ax.set_xticks(range(len(models)), models, rotation=20, ha="right")
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_ylabel(f"Δ {SCORE[track]} (CPT − original)")
        ax.set_title(TRACK_NAME[track])
    ref = style.COLORS["mean"]
    _proxy_point(axes[0, 0], "Held-out credit table", color=ref, s=style.TABLE_MARKER_SIZE)
    _proxy_point(axes[0, 0], "Non-credit retention table", facecolors=style.HOLLOW_FACE, edgecolors=ref,
                 s=style.TABLE_MARKER_SIZE)
    _proxy_point(axes[0, 0], "Equal-table mean", marker=style.MEAN_MARKER, color=style.COLORS["untuned"],
                 edgecolors=ref, s=style.MEAN_MARKER_SIZE)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, summary=pd.DataFrame(tables),
                   per_table=pd.concat([gains[t][gains[t].complete & (gains[t].arm == 0)].assign(track=t)
                                        for t in tracks]))


def domain_control(controls: dict, *, title: str):
    """One panel per task: per held-out credit table and model, the gain of CPT on the matched
    non-credit twins (E3, vertical) against CPT on the credit tables (E1, horizontal)."""
    tracks = _tracks(controls)
    if not tracks:
        return None
    fig, axes = plt.subplots(1, len(tracks), squeeze=False, figsize=style.figsize(style.WIDTH_FULL, 0.47))
    models_seen = []
    for ax, track in zip(axes[0], tracks):
        c = controls[track]
        for model in _models(c):
            s = c[c.model == model]
            ax.scatter(s.gain_credit, s.gain_noncredit, s=style.POINT_SIZE, color=style.model_color(model),
                       alpha=style.FILLED_ALPHA, linewidths=0)
            models_seen.append(model)
        values = np.r_[c.gain_credit.to_numpy(), c.gain_noncredit.to_numpy()]
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        pad = 0.08 * max(high - low, 1e-6)
        low, high = min(low, 0) - pad, max(high, 0) + pad
        ax.plot([low, high], [low, high], color=style.COLORS["annotation"], linewidth=0.7)
        _zero(ax)
        _zero(ax, "x")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Δ {SCORE[track]}, CPT on credit tables (E1)")
        ax.set_ylabel(f"Δ {SCORE[track]}, CPT on non-credit twins (E3)")
        ax.set_title(TRACK_NAME[track])
    _model_proxies(axes[0, 0], [m for m in style.MODEL_ORDER if m in models_seen])
    _proxy_line(axes[0, 0], "Equal gain (y = x)", color=style.COLORS["annotation"], linewidth=0.7)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, **{f"{TRACK_NAME[t]} per table": controls[t][
        ["model", "test_dataset_id", "gain_credit", "gain_noncredit", "credit_minus_noncredit"]] for t in tracks})


def finetune_conditions(table: pd.DataFrame, *, title: str):
    """Rows: task; columns: model. Per held-out table, the change relative to the original model
    without fine-tuning of CPT alone, fine-tuning the original model, and fine-tuning the CPT
    model, all on the same folds; grey lines join one table, diamonds are equal-table means."""
    if _empty(table):
        return None
    tracks = [t for t in TRACKS if t in set(table.track)]
    models = _models(table)
    conditions = ("cpt", "finetune", "cpt_finetune")
    fig, axes = plt.subplots(len(tracks), len(models), sharey="row", squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(len(tracks), style.ROW_HEIGHT * 1.3)))
    summary = []
    for r, track in enumerate(tracks):
        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = table[(table.track == track) & (table.model == model)]
            color = style.model_color(model)
            values = sub[[f"{k}_change" for k in conditions]].to_numpy()
            for row in values:
                ax.plot(range(3), row, color=style.CONNECTOR_COLOR, linewidth=style.CONNECTOR_LINE, zorder=1)
            for k, condition in enumerate(conditions):
                if not len(values):
                    continue
                ax.scatter(np.full(len(values), k) + _jitter(len(values), seed=k) * 0.4, values[:, k],
                           s=style.TABLE_MARKER_SIZE, color=color, marker=style.CONDITION_MARKERS[condition],
                           alpha=style.POINT_ALPHA, linewidths=0, zorder=2)
                ax.scatter([k], [values[:, k].mean()], marker=style.MEAN_MARKER, s=style.MEAN_MARKER_SIZE,
                           color=color, edgecolors=style.COLORS["mean"], linewidths=0.6, zorder=3)
                summary.append(dict(track=track, model=model, condition=style.CONDITION_LABELS[condition],
                                    tables=len(values), mean_change=values[:, k].mean(),
                                    median_change=float(np.median(values[:, k])),
                                    improved=int((values[:, k] > 0).sum())))
            _zero(ax)
            ax.set_xticks(range(3), [style.CONDITION_TICKS[k] for k in conditions])
            ax.set_xlim(-0.5, 2.5)
            if r == 0:
                ax.set_title(model)
            if c == 0:
                ax.set_ylabel(f"{TRACK_NAME[track]}: Δ {SCORE[track]}")
    ref = style.COLORS["mean"]
    for condition in conditions:
        _proxy_point(axes[0, 0], style.CONDITION_LABELS[condition], marker=style.CONDITION_MARKERS[condition],
                     color=style.COLORS["untuned"], s=style.TABLE_MARKER_SIZE * 1.5)
    _proxy_point(axes[0, 0], "Equal-table mean", marker=style.MEAN_MARKER, color=style.COLORS["untuned"],
                 edgecolors=ref, s=style.MEAN_MARKER_SIZE)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, summary=pd.DataFrame(summary), per_table=table)


def baseline_position(frames: dict, *, title: str):
    """Rows: task; columns: model. Per held-out credit table, the original and the CPT model's
    five-fold score minus tuned CatBoost's (zero line). The linear baseline is reported in the
    attached table, not drawn: it sits far below and would compress the comparison."""
    tracks = [t for t in _tracks(frames) if not frames[t][frames[t].credit].empty]
    if not tracks:
        return None
    models = _models(pd.concat([frames[t] for t in tracks]))
    fig, axes = plt.subplots(len(tracks), len(models), sharey="row", squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(len(tracks), style.ROW_HEIGHT * 1.3)))
    summary = []
    for r, track in enumerate(tracks):
        f = frames[track][frames[track].credit]
        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = f[f.model == model]
            color = style.model_color(model)
            values = sub[["original_vs_catboost", "cpt_vs_catboost"]].to_numpy()
            for row in values:
                ax.plot([0, 1], row, color=style.CONNECTOR_COLOR, linewidth=style.CONNECTOR_LINE, zorder=1)
            if len(values):
                ax.scatter(np.zeros(len(values)), values[:, 0], s=style.TABLE_MARKER_SIZE,
                           facecolors=style.HOLLOW_FACE, edgecolors=color, linewidths=0.6, zorder=2)
                ax.scatter(np.ones(len(values)), values[:, 1], s=style.TABLE_MARKER_SIZE, color=color,
                           linewidths=0, zorder=2)
                ax.scatter([0, 1], values.mean(axis=0), marker=style.MEAN_MARKER, s=style.MEAN_MARKER_SIZE,
                           color=color, edgecolors=style.COLORS["mean"], linewidths=0.6, zorder=3)
                summary.append(dict(track=track, model=model, tables=len(values),
                                    original_minus_catboost=values[:, 0].mean(),
                                    cpt_minus_catboost=values[:, 1].mean(),
                                    linear_minus_catboost=sub.linear_vs_catboost.mean(),
                                    original_beats_catboost=int((values[:, 0] > 0).sum()),
                                    cpt_beats_catboost=int((values[:, 1] > 0).sum())))
            ax.axhline(0, color=style.COLORS["catboost"], linewidth=0.9)
            ax.set_xticks([0, 1], ["Original", "CPT"])
            ax.set_xlim(-0.4, 1.4)
            if r == 0:
                ax.set_title(model)
            if c == 0:
                ax.set_ylabel(f"{TRACK_NAME[track]}: {SCORE[track]} − CatBoost")
    _proxy_line(axes[0, 0], "Tuned CatBoost (zero)", color=style.COLORS["catboost"], linewidth=0.9)
    _proxy_point(axes[0, 0], "Original model, per table", facecolors=style.HOLLOW_FACE,
                 edgecolors=style.COLORS["mean"], s=style.TABLE_MARKER_SIZE)
    _proxy_point(axes[0, 0], "CPT model, per table", color=style.COLORS["mean"], s=style.TABLE_MARKER_SIZE)
    _proxy_point(axes[0, 0], "Equal-table mean", marker=style.MEAN_MARKER, color=style.COLORS["untuned"],
                 edgecolors=style.COLORS["mean"], s=style.MEAN_MARKER_SIZE)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, summary=pd.DataFrame(summary),
                   per_table=pd.concat([frames[t].assign(track=t) for t in tracks]))


def _signed(value: float) -> str:
    return "0" if value == 0 else f"{value:+.2g}"


def metric_profile_heatmap(profiles: dict, *, title: str):
    """One panel per task: models (rows) × metrics (columns). Colour: share of complete held-out
    tables on which CPT improved the metric; text: equal-table mean improvement (positive =
    better) and the count of improved tables."""
    tracks = _tracks(profiles)
    if not tracks:
        return None
    fig, axes = plt.subplots(len(tracks), 1, squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(len(tracks), style.ROW_HEIGHT * 1.15)))
    cmap = plt.get_cmap(style.CMAP_SHARE)
    image = None
    for ax, track in zip(axes[:, 0], tracks):
        p = profiles[track]
        models = _models(p)
        metrics = [m for m in METRIC_LABELS if m in set(p.metric)]
        share = np.full((len(models), len(metrics)), np.nan)
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics):
                row = p[(p.model == model) & (p.metric == metric)]
                if row.empty:
                    continue
                row = row.iloc[0]
                share[i, j] = row.tables_improved / row.tables
                red, green, blue, _ = cmap(share[i, j])
                ink = style.GRID_TEXT_LIGHT if .299 * red + .587 * green + .114 * blue < .5 else style.GRID_TEXT_DARK
                ax.text(j, i, f"{_signed(row.mean_improvement)}\n{int(row.tables_improved)}/{int(row.tables)}",
                        ha="center", va="center", fontsize=style.ANNOTATION_SIZE, color=ink)
        image = ax.imshow(np.ma.masked_invalid(share), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics])
        ax.set_yticks(range(len(models)), models)
        ax.tick_params(length=0)
        ax.grid(False)
        ax.set_title(TRACK_NAME[track])
    fig.colorbar(image, ax=axes[:, 0].tolist(), label="Share of tables improved", shrink=0.85)
    fig.suptitle(title)
    return _report(fig, **{f"{TRACK_NAME[t]} metric profile": profiles[t] for t in tracks})


# ------------------------------------------------------------------- RQ2 figures
def dose_response(overview: pd.DataFrame, *, track: str, title: str):
    """Rows: held-out credit tables, non-credit retention panel; columns: model. Equal-table mean
    five-fold gain against the peak learning rate of full-weight CPT, one line per L2-SP strength
    (the reference recipe is lr 3e-7 at λ 0.003)."""
    if _empty(overview):
        return None
    o = overview[(overview.trainable == "all") & overview.slug.isin(["reference", "update_size_anchoring"])]
    if o.slug.eq("update_size_anchoring").sum() == 0:
        return None
    models = _models(o)
    rows = [("credit_mean", "Held-out credit"), ("retention_mean", "Retention panel")]
    fig, axes = plt.subplots(2, len(models), sharex=True, sharey="row", squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(2, style.ROW_HEIGHT * 1.2)))
    for r, (column, name) in enumerate(rows):
        for c, model in enumerate(models):
            ax = axes[r, c]
            color = style.model_color(model)
            for lam in sorted(o.l2sp_lambda.unique()):
                s = o[(o.model == model) & (o.l2sp_lambda == lam)].sort_values("learning_rate")
                if s.empty:
                    continue
                ax.plot(s.learning_rate, s[column], color=color, linestyle=style.LAMBDA_LINESTYLES.get(lam, "-"),
                        marker=style.LAMBDA_MARKERS.get(lam, "o"), markersize=3.5, linewidth=1.0)
            _zero(ax)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(model)
            if c == 0:
                ax.set_ylabel(f"{name}: Δ {SCORE[track]}")
            if r == 1:
                ax.set_xlabel("Peak learning rate")
    for lam in sorted(o.l2sp_lambda.unique()):
        _proxy_line(axes[0, 0], f"L2-SP λ = {lam:g}", color=style.COLORS["mean"],
                    linestyle=style.LAMBDA_LINESTYLES.get(lam, "-"), marker=style.LAMBDA_MARKERS.get(lam, "o"),
                    markersize=3.5)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, arms=o[["slug", "arm", "model", "learning_rate", "l2sp_lambda", "credit_mean",
                                "credit_tables", "credit_improved", "retention_mean", "retention_tables"]])


def tradeoff(overview: pd.DataFrame, *, track: str, title: str):
    """Columns: model. Equal-table mean five-fold gain on the non-credit retention panel against
    the gain on held-out credit tables, per arm of E1, E5 and E6. Colour: trainable scope; marker:
    L2-SP strength (squares: adapter or frozen scopes); size grows with the learning rate; lines
    join the learning rates of one scope and strength."""
    if _empty(overview):
        return None
    o = overview.dropna(subset=["credit_mean", "retention_mean"])
    if o.slug.isin(["update_size_anchoring", "scope"]).sum() == 0:
        return None
    models = _models(o)
    rates = sorted(o.learning_rate.unique())
    size = dict(zip(rates, np.linspace(*style.SIZE_RANGE, len(rates)))) if len(rates) > 1 else {rates[0]: 20}
    fig, axes = plt.subplots(1, len(models), squeeze=False,
                             figsize=style.figsize(style.WIDTH_FULL, 0.42))
    for ax, model in zip(axes[0], models):
        sub = o[o.model == model]
        for (scope, lam), s in sub.groupby(["trainable", "l2sp_lambda"]):
            s = s.sort_values("learning_rate")
            color = style.SCOPE_COLORS.get(scope, style.color(scope))
            marker = style.LAMBDA_MARKERS.get(lam, "o") if scope == "all" else "s"
            ax.plot(s.credit_mean, s.retention_mean, color=color, linewidth=0.6, alpha=0.7)
            ax.scatter(s.credit_mean, s.retention_mean, s=[size[v] for v in s.learning_rate], color=color,
                       marker=marker, edgecolors="none", alpha=style.FILLED_ALPHA, zorder=3)
        _zero(ax)
        _zero(ax, "x")
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_title(model)
        ax.set_xlabel(f"Held-out credit Δ {SCORE[track]}")
    axes[0, 0].set_ylabel(f"Retention panel Δ {SCORE[track]}")
    for scope in [s for s in style.SCOPE_COLORS if s in set(o.trainable)]:
        _proxy_point(axes[0, 0], style.SCOPE_NAMES[scope], color=style.SCOPE_COLORS[scope], marker="s"
                     if scope != "all" else "o", s=20)
    for lam in sorted(o[o.trainable == "all"].l2sp_lambda.unique()):
        _proxy_point(axes[0, 0], f"L2-SP λ = {lam:g}", color=style.COLORS["untuned"],
                     marker=style.LAMBDA_MARKERS.get(lam, "o"), s=20)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, arms=o[["slug", "arm", "label", "model", "trainable", "learning_rate", "l2sp_lambda",
                                "credit_mean", "retention_mean", "credit_tables", "retention_tables"]])


# ------------------------------------------------------------------- RQ3 figures
def arm_gains(table: pd.DataFrame, order: list[str], *, track: str, title: str):
    """Rows: arms (in ``order``); panels: held-out credit tables and non-credit retention panel.
    Per model, the equal-table mean five-fold gain (marker) and the interquartile range over
    tables (line)."""
    if _empty(table):
        return None
    order = [label for label in order if label in set(table.label)]
    if len(order) < 2:
        return None
    panels = [(True, "Held-out credit tables")] + ([(False, "Non-credit retention panel")]
                                                   if (~table.credit).any() else [])
    models = _models(table)
    ypos = {label: len(order) - 1 - i for i, label in enumerate(order)}
    fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(max(len(order), 5), style.ARM_ROW_HEIGHT)))
    rows = []
    for ax, (credit, name) in zip(axes[0], panels):
        sub = table[table.credit == credit]
        for model in models:
            color = style.model_color(model)
            stats = (sub[sub.model == model].groupby("label").gain
                     .agg(mean="mean", q1=lambda g: g.quantile(.25), q3=lambda g: g.quantile(.75),
                          tables="size", improved=lambda g: int((g > 0).sum())).reset_index())
            stats = stats[stats.label.isin(ypos)]
            y = stats.label.map(ypos) - style.MODEL_DODGE.get(model, 0)   # first model on top
            ax.hlines(y, stats.q1, stats.q3, color=color, linewidth=1.0)
            ax.scatter(stats["mean"], y, color=color, s=style.POINT_SIZE * 0.8, zorder=3)
            rows.append(stats.assign(model=model, tables_group=name))
        _zero(ax, "x")
        ax.set_title(name)
        ax.set_xlabel(f"Δ {SCORE[track]} (CPT − original)")
        ax.grid(axis="x", alpha=0.25)
    axes[0, 0].set_yticks([ypos[label] for label in order], order)
    for y in range(len(order) - 1):
        for ax in axes[0]:
            ax.axhline(y + 0.5, color=style.CONNECTOR_COLOR, linewidth=0.4)
    _model_proxies(axes[0, 0], models)
    _proxy_line(axes[0, 0], "Interquartile range over tables", color=style.COLORS["untuned"], linewidth=1.0)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, arm_summary=pd.concat(rows) if rows else None)


def episode_matched_context(gains: pd.DataFrame, *, title: str):
    """Columns: model. Equal-table mean gain over the original model at the same inference context
    (2,048 rows, 8,192 rows, the full pool) against the training-episode size of E10."""
    if _empty(gains):
        return None
    models = _models(gains)
    contexts = [c for c in ("2048", "8192", "full") if c in set(gains.context)]
    palette = style.ordered(contexts)
    names = {"2048": "2,048 context rows", "8192": "8,192 context rows", "full": "Full context pool"}
    fig, axes = plt.subplots(1, len(models), sharey=True, sharex=True, squeeze=False,
                             figsize=style.figsize(style.WIDTH_FULL, 0.36))
    summary = (gains.groupby(["model", "episode_rows", "context"]).gain
               .agg(mean="mean", tables="size", improved=lambda g: int((g > 0).sum())).reset_index())
    for ax, model in zip(axes[0], models):
        for context in contexts:
            s = summary[(summary.model == model) & (summary.context == context)].sort_values("episode_rows")
            ax.plot(s.episode_rows, s["mean"], color=palette[context], marker="o", markersize=3, linewidth=1.0)
        _zero(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(gains.episode_rows.unique()), [f"{int(v):,}" for v in sorted(gains.episode_rows.unique())],
                      rotation=30)
        ax.minorticks_off()
        ax.set_title(model)
        ax.set_xlabel("Training episode rows")
    axes[0, 0].set_ylabel("Δ AUC vs original")
    for context in contexts:
        _proxy_line(axes[0, 0], names[context], color=palette[context], marker="o", markersize=3)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, summary=summary)


def quality_cost(overview: pd.DataFrame, resources: pd.DataFrame, *, title: str):
    """One panel per task: equal-table mean five-fold gain on held-out credit tables against the
    mean wall-clock hours of one training run, per arm and model (reference, full pass and the
    episode sizes; E10 sizes joined by a line)."""
    if _empty(overview) or _empty(resources):
        return None
    merged = overview.merge(resources[["slug", "arm", "track", "model", "mean_hours"]],
                            on=["slug", "arm", "track", "model"]).dropna(subset=["credit_mean", "mean_hours"])
    merged = merged[merged.slug.isin(style.SLUG_MARKERS)]
    if merged.empty or merged.slug.nunique() < 2:
        return None
    tracks = [t for t in TRACKS if t in set(merged.track)]
    fig, axes = plt.subplots(1, len(tracks), squeeze=False, figsize=style.figsize(style.WIDTH_FULL, 0.42))
    for ax, track in zip(axes[0], tracks):
        sub = merged[merged.track == track]
        for model in _models(sub):
            color = style.model_color(model)
            series = sub[(sub.model == model) & (sub.slug == "episode_size")].sort_values("episode_rows")
            if len(series) > 1:
                ax.plot(series.mean_hours, series.credit_mean, color=color, linewidth=0.6, alpha=0.7)
            for slug, s in sub[sub.model == model].groupby("slug"):
                ax.scatter(s.mean_hours, s.credit_mean, color=color, marker=style.SLUG_MARKERS[slug],
                           s=style.POINT_SIZE, edgecolors="none", zorder=3)
        _zero(ax)
        _plain_log_axis(ax, sub.mean_hours)
        ax.set_xlabel("Hours per training run")
        ax.set_ylabel(f"Held-out credit Δ {SCORE[track]}")
        ax.set_title(TRACK_NAME[track])
    names = {"reference": "Reference (E1)", "exposure": "Full pass (E8)", "episode_size": "Episode sizes (E10)",
             "class_sampling": "Balanced classes (E7)", "table_size": "Table-size corpora (E9)"}
    _model_proxies(axes[0, 0], _models(merged))
    for slug in [s for s in style.SLUG_MARKERS if s in set(merged.slug)]:
        _proxy_point(axes[0, 0], names[slug], marker=style.SLUG_MARKERS[slug], color=style.COLORS["untuned"],
                     s=style.POINT_SIZE)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, quality_and_cost=merged[["slug", "arm", "label", "track", "model", "credit_mean",
                                                 "credit_tables", "mean_hours"]])


# ------------------------------------------------------------------- RQ4 figures
def context_curves(table: pd.DataFrame, curve: pd.DataFrame, *, track: str, title: str):
    """Columns: model. Top: equal-table mean score of the original (dashed) and the CPT model
    (solid) against the labelled context rows, over the tables that reach every plotted size.
    Bottom: per-table gain (thin grey, up to each table's largest size) and the same fixed-cohort
    equal-table mean gain."""
    if _empty(table) or _empty(curve):
        return None
    t = table[(table.track == track) & ~table.full]
    cv = curve[(curve.track == track) & curve.fixed_cohort]
    if t.empty or cv.empty:
        return None
    models = _models(t)
    fig, axes = plt.subplots(2, len(models), sharex=True, sharey="row", squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(2, style.ROW_HEIGHT * 1.3)))
    for c, model in enumerate(models):
        color = style.model_color(model)
        m = cv[cv.model == model].sort_values("context_rows")
        top, bottom = axes[0, c], axes[1, c]
        for start in ("original", "cpt"):
            top.plot(m.context_rows, m[start], color=color, linestyle=style.START_LINESTYLES[start],
                     linewidth=1.1)
        for _, s in t[t.model == model].groupby("dataset_id"):
            s = s.sort_values("context_rows")
            bottom.plot(s.context_rows, s.gain, color=style.CONNECTOR_COLOR, linewidth=style.TABLE_CURVE_LINE)
        bottom.plot(m.context_rows, m.gain, color=color, linewidth=style.MEAN_CURVE_LINE)
        _zero(bottom)
        bottom.set_xscale("log")
        bottom.set_xlabel("Labelled context rows")
        top.set_title(model)
    axes[0, 0].set_ylabel(SCORE[track])
    axes[1, 0].set_ylabel(f"Δ {SCORE[track]} (CPT − original)")
    for start in ("original", "cpt"):
        _proxy_line(axes[0, 0], style.START_LABELS[start], color=style.COLORS["mean"],
                    linestyle=style.START_LINESTYLES[start])
    _proxy_line(axes[0, 0], "Single table", color=style.CONNECTOR_COLOR, linewidth=style.TABLE_CURVE_LINE)
    _proxy_line(axes[0, 0], "Equal-table mean, fixed cohort", color=style.COLORS["mean"],
                linewidth=style.MEAN_CURVE_LINE)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, fixed_cohort_curve=cv, per_table=t)


PROPERTY_LABELS = {"original": "Original model score", "rows": "Rows (log scale)", "features": "Features",
                   "categorical_share": "Categorical share", "missing_share": "Missing share",
                   "default_rate": "Default rate", "zero_share": "Share of LGD = 0",
                   "one_share": "Share of LGD = 1", "nearest": "Distance to nearest training table"}


def gain_vs_properties(merged: pd.DataFrame, *, track: str, title: str, properties=None):
    """Panels: table properties. Per held-out credit table and model, the five-fold E1 gain against
    the property (rows on a log scale)."""
    if _empty(merged):
        return None
    properties = [p for p in (properties or ("original", "rows", "features", "categorical_share",
                                             "default_rate" if track == "pd" else "zero_share", "nearest"))
                  if p in merged and merged[p].notna().any()]
    if not properties:
        return None
    columns = 3
    nrows = int(np.ceil(len(properties) / columns))
    fig, axes = plt.subplots(nrows, columns, sharey=True, squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(nrows, style.ROW_HEIGHT * 1.25)))
    models = _models(merged)
    for ax, prop in zip(axes.ravel(), properties):
        for model in models:
            s = merged[merged.model == model]
            ax.scatter(s[prop], s.gain, s=style.TABLE_MARKER_SIZE * 1.5, color=style.model_color(model),
                       alpha=style.FILLED_ALPHA, linewidths=0)
        _zero(ax)
        if prop == "rows":
            ax.set_xscale("log")
        ax.set_xlabel(PROPERTY_LABELS.get(prop, prop))
    for ax in axes.ravel()[len(properties):]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r, 0].set_ylabel(f"Δ {SCORE[track]}")
    _model_proxies(axes[0, 0], models)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, per_table=merged[["model", "test_dataset_id", "gain", *properties]])


def transfer_heatmaps(long: pd.DataFrame, *, track: str, title: str):
    """Panels: model. Monitor change of each held-out target table (columns) after CPT on one
    source table (rows), first to last monitor point; one symmetric colour scale for all panels."""
    if _empty(long):
        return None
    sub = long[long.scope == "test"]
    if sub.empty:
        return None
    models = _models(sub)
    sources = sorted(sub.source.unique())
    targets = sorted(sub.target.unique())
    values = sub.change.to_numpy(dtype=float)
    extent = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.0, 1e-6)
    ncols = 2 if len(models) > 1 else 1
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(style.WIDTH_FULL, style.MAX_HEIGHT))
    cmap = plt.get_cmap(style.CMAP_DIVERGING).copy()
    cmap.set_bad(style.color("annotation"), alpha=0.15)
    image = None
    for k, model in enumerate(models):
        ax = axes.ravel()[k]
        matrix = (sub[sub.model == model].pivot_table(index="source", columns="target", values="change")
                  .reindex(index=sources, columns=targets))
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap=cmap, vmin=-extent,
                          vmax=extent, aspect="auto")
        ax.grid(False)
        ax.set_title(model)
        row, col = divmod(k, ncols)
        ax.set_yticks(range(len(sources)), [display_name(s) for s in sources] if col == 0 else [])
        ax.set_xticks(range(len(targets)), [display_name(t) for t in targets] if row == nrows - 1 or
                      k + ncols >= len(models) else [], rotation=90)
        ax.tick_params(length=0)
        if col == 0:
            ax.set_ylabel("Source table")
        if row == nrows - 1:
            ax.set_xlabel("Target table")
    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=axes.ravel().tolist(), label=f"Δ {SCORE[track]} (monitor)", shrink=0.8)
    fig.suptitle(title)
    return _report(fig, transfer=sub[["model", "source", "target", "change"]])
