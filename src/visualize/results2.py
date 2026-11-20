"""Second-campaign publication figures (RQ1-RQ3).

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
from matplotlib.ticker import MaxNLocator

from src.data.dataset_names import display_name
from src.visualize import style
from src.visualize.campaign2 import (
    METRIC_LABELS,
    MONITOR_SCOPES,
    PROFILE_METRICS,
    SEED_LIST,
    SLOPE_QUANTITIES,
)

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


def _seed_text() -> str:
    return ", ".join(str(s) for s in SEED_LIST)


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
                                linewidth=0, label=f"Range over seeds {_seed_text()} (held-out credit)")
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, monitor_curves=curves, seed_range=band)


def arm_styles(settings: pd.DataFrame, *, color_by: str = "label", linestyle_by: str | None = None,
               ordered: bool = True) -> pd.DataFrame:
    """Colour and line style per arm. The reference arm is always black and solid. Other arms are
    coloured by ``color_by``: the trainable-scope colours for ``trainable``, an ordered ramp
    (``ordered``) or the arm palette, which avoids the model hues. ``linestyle_by``:
    ``l2sp_lambda`` (E8) or ``rate_within_scope`` (E9: lower rate dashed, higher solid)."""
    if _empty(settings):
        return pd.DataFrame()
    arms = settings.copy()
    is_ref = arms.slug.eq("reference")
    # An arm without a value for ``color_by`` keeps the reference setting (E12's reference
    # episode size) and is drawn like the reference.
    is_ref = is_ref | arms[color_by].isna()
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
    model per table (seed 42, E1); with E2, a line spans the seeds' gains and a filled marker
    means all seeds agree in sign and exceed their range."""
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
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_xlabel(f"Δ {SCORE[track]}")
        ax.grid(axis="x", alpha=0.25)
    axes[0, 0].set_yticks(range(len(order)), [display_name(t) for t in order])
    axes[0, 0].set_ylim(-0.7, len(order) - 0.3)
    ref = style.COLORS["mean"]
    if has_seeds:
        _proxy_point(axes[0, 0], "Seed 42; all seeds exceed their range", color=ref, s=style.POINT_SIZE)
        _proxy_point(axes[0, 0], "Seed 42; within seed range", facecolors=style.HOLLOW_FACE, edgecolors=ref,
                     s=style.POINT_SIZE)
        _proxy_line(axes[0, 0], f"Range over seeds {_seed_text()}", color=ref, linewidth=style.TABLE_CURVE_LINE * 1.6)
    else:
        _proxy_point(axes[0, 0], "Seed 42 (E1)", color=ref, s=style.POINT_SIZE)
    _proxy_line(axes[0, 0], "Equal-table mean (seed 42)", color=ref, linestyle="--", linewidth=0.9)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, per_table_gain=g[["model", "test_dataset_id", "gain", "original", "folds"]],
                   seed_gains=pd.concat(used) if used else None)


OUTCOME_LABELS = {"roc_auc": "AUC", "brier_score": "Brier score", "r2": "R²", "crps": "CRPS"}


def primary_forest(primary: pd.DataFrame, retention: pd.DataFrame | None = None,
                   contrast: pd.DataFrame | None = None, *, title: str):
    """Panels: the pre-specified outcomes (PD AUC and Brier, LGD R² and CRPS). Per model, three
    estimates with 95% hierarchical-bootstrap intervals, positive = better in every panel: E1 over
    the original model on the held-out credit tables (filled circle; an asterisk marks a
    Holm-adjusted p < α: the primary test), the same on the non-credit retention tables (hollow
    circle: forgetting) and, when E4 exists, credit CPT over CPT on the matched non-credit twins on
    the held-out credit tables (square: the credit-specific part)."""
    if _empty(primary):
        return None
    panels = [(t, m) for t in TRACKS for m in dict.fromkeys(primary[primary.track == t].metric)]
    models = _models(primary)
    fig, axes = plt.subplots(1, len(panels), squeeze=False, sharey=True,
                             figsize=style.figsize(style.WIDTH_FULL, 0.40 if _empty(contrast) else 0.50))
    hollow = squares = False
    for ax, (track, metric) in zip(axes[0], panels):
        rows = [(primary[(primary.track == track) & (primary.metric == metric)], "credit")]
        if not _empty(retention):
            rows.append((retention[(retention.track == track) & (retention.metric == metric)
                                   & ~retention.credit.astype(bool)], "retention"))
        if not _empty(contrast):
            rows.append((contrast[(contrast.track == track) & (contrast.metric == metric)], "contrast"))
        offsets = dict(zip([kind for _, kind in rows], np.linspace(-0.24, 0.24, len(rows)) if len(rows) > 1 else [0.0]))
        for frame, kind in rows:
            for i, model in enumerate(models):
                hit = frame[frame.model == model]
                if hit.empty:
                    continue
                row = hit.iloc[0]
                y = i + offsets[kind]
                color = style.model_color(model)
                ax.plot([row.low, row.high], [y, y], color=color, linewidth=style.INTERVAL_LINE)
                ax.scatter([row["mean"]], [y], s=style.POINT_SIZE, zorder=3, linewidths=0.8, edgecolors=color,
                           marker=style.CONTRAST_MARKER if kind == "contrast" else "o",
                           facecolors=style.HOLLOW_FACE if kind == "retention" else color)
                hollow = hollow or kind == "retention"
                squares = squares or kind == "contrast"
                credit = kind == "credit"
                if credit and str(row.get("decision", "")) in ("better", "worse"):
                    ax.annotate("*", (row.high, y), xytext=(2, -2), textcoords="offset points",
                                fontsize=style.ANNOTATION_SIZE + 2, color=color, va="center")
        _zero(ax, "x")
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_title(f"{TRACK_NAME[track]}: {OUTCOME_LABELS.get(metric, METRIC_LABELS.get(metric, metric))}")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("Improvement")
    ref = style.COLORS["mean"]
    _proxy_point(axes[0, 0], "Held-out credit tables (*: Holm-adjusted p < 0.05)", color=ref, s=style.POINT_SIZE)
    if hollow:
        _proxy_point(axes[0, 0], "Non-credit retention tables", facecolors=style.HOLLOW_FACE, edgecolors=ref,
                     s=style.POINT_SIZE)
    if squares:
        _proxy_point(axes[0, 0], "Credit CPT minus non-credit CPT (E1 − E4), held-out credit",
                     marker=style.CONTRAST_MARKER, color=ref, s=style.POINT_SIZE)
    _proxy_line(axes[0, 0], "95% bootstrap interval", color=ref, linewidth=style.INTERVAL_LINE)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, primary_endpoint=primary, retention=retention, domain_contrast=contrast)


ADAPTATION_ROWS = (("cpt_change", "CPT − original"), ("finetune_change", "FT − original"),
                   ("cpt_after_finetune", "CPT+FT − FT"))


def adaptation_by_size(table: pd.DataFrame, *, title: str):
    """Rows: the three E5 contrasts (CPT alone, fine-tuning alone, CPT on top of fine-tuning);
    columns: task. Per held-out table and model, the five-fold change against the table's labelled
    context rows (log scale), coloured by model."""
    if _empty(table) or "context_rows" not in table:
        return None
    tracks = [t for t in TRACKS if t in set(table.track)]
    models = _models(table)
    fig, axes = plt.subplots(len(ADAPTATION_ROWS), len(tracks), squeeze=False, sharex="col", sharey="col",
                             figsize=(style.WIDTH_FULL, _height(len(ADAPTATION_ROWS), style.ROW_HEIGHT * 0.95)))
    for c, track in enumerate(tracks):
        sub = table[table.track == track]
        for r, (column, label) in enumerate(ADAPTATION_ROWS):
            ax = axes[r, c]
            for k, model in enumerate(models):
                m = sub[sub.model == model]
                ax.scatter(m.context_rows * (1 + 0.04 * (k - 1.5)), m[column], s=style.TABLE_POINT_SIZE,
                           color=style.model_color(model), alpha=style.POINT_ALPHA, linewidths=0)
            _zero(ax)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(TRACK_NAME[track])
            if r == len(ADAPTATION_ROWS) - 1:
                ax.set_xlabel("Labelled context rows of the table")
            ax.set_ylabel(f"{label}\nΔ {SCORE[track]}" if c == 0 else f"Δ {SCORE[track]}")
    _model_proxies(axes[0, 0], models)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, per_table=table)

def leaderboard(board: pd.DataFrame, *, track: str, title: str):
    """Three panels sharing the method rows (sorted by Elo): TabArena-style Elo with its 95%
    table-bootstrap interval (anchor = 1000, dotted), the normalized score (1 = best method on
    every table, 0 = at or below the median method) and improvability (mean relative error gap to
    the best method of each fold, with its 95% interval). Colour: base model or baseline; marker:
    adaptation (hollow without CPT, filled with CPT; circle without and square with fine-tuning)."""
    if _empty(board):
        return None
    from src.visualize.campaign2 import LEADERBOARD_ANCHOR, LEADERBOARD_ERROR
    board = board.sort_values("elo", ascending=False).reset_index(drop=True)
    n = len(board)
    fig, axes = plt.subplots(1, 3, sharey=True, squeeze=False, gridspec_kw=dict(width_ratios=[1.5, 1, 1]),
                             figsize=(style.WIDTH_FULL, _height(n, style.LEADERBOARD_ROW) + 0.35))
    y = np.arange(n)
    for row in board.itertuples():
        variant = row.variant if row.variant in style.VARIANT_MARKERS else "baseline"
        color = style.BASELINE_COLORS.get(row.method, style.model_color(row.base))
        face = color if style.VARIANT_FILLED[variant] else style.HOLLOW_FACE
        marker = dict(marker=style.VARIANT_MARKERS[variant], s=style.POINT_SIZE, facecolors=face,
                      edgecolors=color, linewidths=0.8, zorder=3)
        i = row.Index
        axes[0, 0].plot([row.elo_low, row.elo_high], [i, i], color=color, linewidth=style.INTERVAL_LINE)
        axes[0, 0].scatter([row.elo], [i], **marker)
        axes[0, 1].scatter([row.normalized_score], [i], **marker)
        axes[0, 2].plot([100 * row.improvability_low, 100 * row.improvability_high], [i, i], color=color,
                        linewidth=style.INTERVAL_LINE)
        axes[0, 2].scatter([100 * row.improvability], [i], **marker)
    axes[0, 0].axvline(1000, **style.ZERO_LINE)
    axes[0, 0].set_yticks(y, board.method)
    axes[0, 0].set_ylim(n - 0.5, -0.5)
    axes[0, 0].set_xlabel(f"Elo ({LEADERBOARD_ANCHOR[track].lower()} = 1000)")
    axes[0, 1].set_xlabel("Normalized score")
    axes[0, 1].set_xlim(-0.05, 1.05)
    axes[0, 2].set_xlabel("Improvability (%)")
    axes[0, 2].set_xlim(left=0)
    for ax in axes[0]:
        ax.grid(axis="x", alpha=0.25)
    variants = [v for v in style.VARIANT_LABELS if v in set(board.variant) or (v == "baseline")]
    for v in variants:
        _proxy_point(axes[0, 0], style.VARIANT_LABELS[v], marker=style.VARIANT_MARKERS[v], s=style.POINT_SIZE,
                     facecolors=style.COLORS["mean"] if style.VARIANT_FILLED[v] else style.HOLLOW_FACE,
                     edgecolors=style.COLORS["mean"], linewidths=0.8)
    _proxy_line(axes[0, 0], "95% table-bootstrap interval", color=style.COLORS["mean"], linewidth=style.INTERVAL_LINE)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, leaderboard=board.assign(error=LEADERBOARD_ERROR[track][1]))


def metric_profile_heatmap(profiles: dict, *, title: str):
    """One panel per task: models (rows) × metrics (columns). Colour and text: share and count of
    complete held-out tables on which CPT improved the metric. The mean improvements, whose scales
    differ by orders of magnitude between metrics, are in the attached table."""
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
        metrics = [m for m in PROFILE_METRICS.get(track, METRIC_LABELS) if m in set(p.metric)]
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
                ax.text(j, i, f"{int(row.tables_improved)}/{int(row.tables)}",
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


def precision_agreement(agreement: pd.DataFrame, *, title: str, threshold: float = 0.9):
    """One panel per task. Rows: base model; one point per training table: the cosine between the
    mixed-precision and the FP32 gradient at update zero on the identical episode (1 = same
    direction). Colour and marker: candidate precision; the dashed line marks ``threshold``."""
    if _empty(agreement) or not np.isfinite(agreement.gradient_cosine.to_numpy(dtype=float)).any():
        return None
    tracks = [t for t in TRACKS if t in set(agreement.track)]
    precisions = list(dict.fromkeys(agreement.precision))
    palette = style.ordered(precisions)
    markers = dict(zip(precisions, ("v", "o", "s", "^")))
    fig, axes = plt.subplots(1, len(tracks), squeeze=False, sharex=True, sharey=True,
                             figsize=style.figsize(style.WIDTH_FULL, 0.42))
    models = _models(agreement)
    offsets = np.linspace(-0.18, 0.18, len(precisions)) if len(precisions) > 1 else [0.0]
    for ax, track in zip(axes[0], tracks):
        sub = agreement[agreement.track == track].dropna(subset=["gradient_cosine"])
        for i, model in enumerate(models):
            for offset, precision in zip(offsets, precisions):
                v = sub[(sub.model == model) & (sub.precision == precision)].gradient_cosine.to_numpy()
                if len(v):
                    ax.scatter(v, i + offset + _jitter(len(v), seed=i) * 0.5, s=style.TABLE_MARKER_SIZE * 1.4,
                               color=palette[precision], marker=markers[precision], alpha=style.FILLED_ALPHA,
                               linewidths=0)
        ax.axvline(threshold, color=style.COLORS["mean"], linestyle="--", linewidth=0.8)
        ax.axvline(1.0, **style.ZERO_LINE)
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_xlim(min(-0.05, float(agreement.gradient_cosine.min()) - 0.05), 1.04)
        ax.set_title(TRACK_NAME[track])
        ax.set_xlabel("Cosine with the FP32 gradient")
    for precision in precisions:
        _proxy_point(axes[0, 0], precision, color=palette[precision], marker=markers[precision],
                     s=style.TABLE_MARKER_SIZE * 1.4)
    _proxy_line(axes[0, 0], f"Cosine {threshold:g}", color=style.COLORS["mean"], linestyle="--", linewidth=0.8)
    axes[0, 0].legend()
    fig.suptitle(title)
    summary = (agreement.groupby(["track", "model", "precision", "role"]).gradient_cosine
               .agg(tables="size", compared="count", minimum="min", median="median",
                    below_threshold=lambda v: int((v < threshold).sum())).reset_index())
    return _report(fig, precision_summary=summary, precision_agreement=agreement)


# ------------------------------------------------------------------- RQ2 figures
def tradeoff(overview: pd.DataFrame, *, track: str, title: str):
    """Columns: model. Equal-table mean five-fold gain on the non-credit retention panel against
    the gain on held-out credit tables, per arm of E1, E8 and E9. Colour: trainable scope; marker:
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
    ncols = 2 if len(models) > 1 else 1
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(nrows, style.ROW_HEIGHT * 1.25)))
    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)
    for ax, model in zip(axes.ravel(), models):
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
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(model)
    for k, ax in enumerate(axes.ravel()[:len(models)]):
        if k + ncols >= len(models):
            ax.set_xlabel(f"Held-out credit Δ {SCORE[track]}")
        if k % ncols == 0:
            ax.set_ylabel(f"Retention Δ {SCORE[track]}")
    for scope in [s for s in style.SCOPE_COLORS if s in set(o.trainable)]:
        _proxy_point(axes[0, 0], style.SCOPE_NAMES[scope], color=style.SCOPE_COLORS[scope], marker="s"
                     if scope != "all" else "o", s=20)
    for lam in sorted(o[o.trainable == "all"].l2sp_lambda.unique()):
        _proxy_point(axes[0, 0], f"L2-SP λ = {lam:g}", color=style.COLORS["untuned"],
                     marker=style.LAMBDA_MARKERS.get(lam, "o"), s=20)
    if len(rates) > 1:
        _proxy_point(axes[0, 0], f"Learning rate {min(rates):.0e} (smallest marker)", color=style.COLORS["untuned"],
                     s=min(size.values()))
        _proxy_point(axes[0, 0], f"Learning rate {max(rates):.0e} (largest marker)", color=style.COLORS["untuned"],
                     s=max(size.values()))
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
    (2,048 rows, 8,192 rows, the full pool) against the training-episode size of E12."""
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
        ticks = [v for v in sorted(gains.episode_rows.unique()) if float(np.log2(v)).is_integer()]
        ax.set_xticks(ticks, [f"{int(v) // 1024}k" for v in ticks])
        ax.minorticks_off()
        ax.set_title(model)
        ax.set_xlabel("Episode rows")
    axes[0, 0].set_ylabel("Δ AUC (CPT − original)")
    for context in contexts:
        _proxy_line(axes[0, 0], names[context], color=palette[context], marker="o", markersize=3)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, summary=summary)


def context_curves(table: pd.DataFrame, curve: pd.DataFrame, *, title: str):
    """Rows: task; columns: model. Per-table gain of the CPT model over the original at each number
    of labelled context rows (thin grey, up to each table's largest size) and the equal-table mean
    over the tables that reach every plotted size (the fixed cohort). When E4 was scored, the
    fixed-cohort mean gain of the non-credit CPT model is drawn in the same colour, dotted."""
    if _empty(table) or _empty(curve):
        return None
    t = table[~table.full]
    cv = curve[curve.fixed_cohort]
    tracks = [k for k in TRACKS if k in set(t.track) and k in set(cv.track)]
    if not tracks:
        return None
    models = _models(t)
    fig, axes = plt.subplots(len(tracks), len(models), sharex=True, sharey="row", squeeze=False,
                             figsize=(style.WIDTH_FULL, _height(len(tracks), style.ROW_HEIGHT * 1.25)))
    for r, track in enumerate(tracks):
        for c, model in enumerate(models):
            ax = axes[r, c]
            for _, s in t[(t.track == track) & (t.model == model)].groupby("dataset_id"):
                s = s.sort_values("context_rows")
                ax.plot(s.context_rows, s.gain, color=style.CONNECTOR_COLOR, linewidth=style.TABLE_CURVE_LINE)
            m = cv[(cv.track == track) & (cv.model == model)].sort_values("context_rows")
            ax.plot(m.context_rows, m.gain, color=style.model_color(model), linewidth=style.MEAN_CURVE_LINE)
            if "noncredit_gain" in m and m.noncredit_gain.notna().any():
                ax.plot(m.context_rows, m.noncredit_gain, color=style.model_color(model),
                        linewidth=style.MEAN_CURVE_LINE, linestyle=style.START_LINESTYLES["noncredit_cpt"])
            _zero(ax)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(model)
        axes[r, 0].set_ylabel(f"{TRACK_NAME[track]}: Δ {SCORE[track]}")
    for ax in axes[-1]:
        ax.set_xlabel("Context rows")
    _proxy_line(axes[0, 0], "Single table", color=style.CONNECTOR_COLOR, linewidth=style.TABLE_CURVE_LINE)
    _proxy_line(axes[0, 0], "Equal-table mean, fixed cohort", color=style.COLORS["mean"],
                linewidth=style.MEAN_CURVE_LINE)
    if "noncredit_gain" in cv and cv.noncredit_gain.notna().any():
        _proxy_line(axes[0, 0], "Same, non-credit CPT (E4)", color=style.COLORS["mean"],
                    linewidth=style.MEAN_CURVE_LINE, linestyle=style.START_LINESTYLES["noncredit_cpt"])
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, fixed_cohort_curve=cv, per_table=t)



def context_slopes(summary: pd.DataFrame, *, title: str):
    """One panel per task: per model, the equal-table mean slope of the gain per doubling of the
    labelled context, with its 95% bootstrap interval over tables, for each E6 quantity (marker)."""
    if _empty(summary):
        return None
    tracks = [t for t in TRACKS if t in set(summary.track)]
    quantities = [q for q in SLOPE_QUANTITIES if q in set(summary.quantity)]
    fig, axes = plt.subplots(1, len(tracks), squeeze=False, sharey=True,
                             figsize=style.figsize(style.WIDTH_FULL, 0.42))
    models = _models(summary)
    offsets = dict(zip(quantities, np.linspace(-0.27, 0.27, len(quantities)) if len(quantities) > 1 else [0.0]))
    markers = dict(zip(quantities, style.CONDITION_MARKERS.values()))
    for ax, track in zip(axes[0], tracks):
        frame = summary[summary.track == track]
        for i, model in enumerate(models):
            color = style.model_color(model)
            for _, row in frame[frame.model == model].iterrows():
                y = i + offsets[row.quantity]
                ax.plot([row.low, row.high], [y, y], color=color, linewidth=style.INTERVAL_LINE)
                ax.scatter([row["mean"]], [y], s=style.POINT_SIZE, color=color, marker=markers[row.quantity],
                           zorder=3)
        _zero(ax, "x")
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_title(TRACK_NAME[track])
        ax.set_xlabel(f"Δ {SCORE[track]} per doubling of context")
    for quantity in quantities:
        _proxy_point(axes[0, 0], SLOPE_QUANTITIES[quantity], color=style.COLORS["mean"], s=style.POINT_SIZE,
                     marker=markers[quantity])
    _proxy_line(axes[0, 0], "95% bootstrap interval over tables", color=style.COLORS["mean"],
                linewidth=style.INTERVAL_LINE)
    axes[0, 0].legend()
    fig.suptitle(title)
    return _report(fig, slopes=summary)

PROPERTY_LABELS = {"original": "Original model score", "rows": "Rows (log scale)", "features": "Features",
                   "categorical_share": "Categorical share", "missing_share": "Missing share",
                   "default_rate": "PD class-1 share", "zero_share": "Share of LGD = 0",
                   "one_share": "Share of LGD = 1", "nearest": "Distance to nearest\ntraining table"}


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
    columns = 2 if len(properties) <= 4 else 3
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


def transfer_similarity(frames: dict, *, title: str):
    """Rows: task; columns: model. E7 monitor change of each target against its property distance
    to the source, one point per eligible source-target pair. The
    Spearman correlation and Mantel p-value sit in headroom reserved above the points.
    ``frames[track]`` holds ``pairs`` and ``tests`` (``campaign2.transfer_pairs``/``transfer_similarity``)."""
    tracks = [t for t in TRACKS if t in frames and not _empty(frames[t].get("pairs"))]
    if not tracks:
        return None
    models = _models(pd.concat([frames[t]["pairs"] for t in tracks]))
    fig, axes = plt.subplots(len(tracks), len(models), squeeze=False, sharex="row", sharey="row",
                             figsize=(style.WIDTH_FULL, _height(len(tracks), style.ROW_HEIGHT * 1.15)))
    reports = {}
    for r, track in enumerate(tracks):
        pairs, tests = frames[track]["pairs"], frames[track].get("tests")
        low, high = float(pairs.change.min()), float(pairs.change.max())
        pad = (high - low) or 1.0
        for c, model in enumerate(models):
            ax = axes[r, c]
            s = pairs[pairs.model == model]
            ax.scatter(s.distance, s.change, s=style.TABLE_POINT_SIZE, color=style.model_color(model),
                       alpha=style.POINT_ALPHA, linewidths=0)
            _zero(ax)
            ax.set_ylim(low - 0.05 * pad, high + 0.30 * pad)      # headroom for the test result
            test = tests[tests.model == model] if not _empty(tests) else pd.DataFrame()
            if not test.empty:
                ax.text(0.04, 0.96, f"ρ = {test.iloc[0].spearman:.2f}\np = {test.iloc[0].p_value:.3f}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=style.ANNOTATION_SIZE)
            if r == 0:
                ax.set_title(model)
            if r == len(tracks) - 1:
                ax.set_xlabel("Source–target\ndistance")
        axes[r, 0].set_ylabel(f"{TRACK_NAME[track]}: Δ {SCORE[track]}")
        reports[f"{TRACK_NAME[track]} similarity test"] = tests
        reports[f"{TRACK_NAME[track]} pairs"] = pairs
    fig.suptitle(title)
    return _report(fig, **{k.replace(" ", "_"): v for k, v in reports.items() if not _empty(v)})


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
    # Tables are numbered in id order; the key travels with the figure's report tables.
    codes = {t: str(i + 1) for i, t in enumerate(sorted(set(sources) | set(targets)))}
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
        ax.set_yticks(range(len(sources)), [codes[s] for s in sources] if col == 0 else [])
        ax.set_xticks(range(len(targets)), [codes[t] for t in targets] if row == nrows - 1 or
                      k + ncols >= len(models) else [])
        ax.tick_params(length=0, labelsize=style.ANNOTATION_SIZE - 1)
        if col == 0:
            ax.set_ylabel("Source table (number)")
        if row == nrows - 1:
            ax.set_xlabel("Target table (number)")
    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=axes.ravel().tolist(), label=f"Δ {SCORE[track]} (monitor)", shrink=0.8)
    fig.suptitle(title)
    key = pd.DataFrame([dict(number=int(n), table=display_name(t)) for t, n in codes.items()])
    return _report(fig, table_numbers=key, transfer=sub[["model", "source", "target", "change"]])
