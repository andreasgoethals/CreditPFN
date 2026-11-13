"""Shared CreditPFN figure sizes, typography, colors and markers.

Figures are drawn at their final publication width (Springer LNCS, 122-mm text block). Notebooks call apply()
and use these definitions rather than choosing their own sizes or colors.
"""

from __future__ import annotations

import re
import zlib

import matplotlib as mpl
import numpy as np

# ---------------------------------------------------------------------------
# FIGURE SIZES — Springer LNCS geometry (ECML PKDD): a 122-mm text block; two figures side by
# side are 58 mm each, separated by 6 mm.
#
# DRAW AT FINAL WIDTH, and never rescale a figure in the document. Rescaling carries the text with
# it: 9pt squeezed to 70% arrives as 6.3pt, under the ~7pt floor where small print stops being
# legible on paper. So the point sizes in `_RC` are the point sizes ON THE PRINTED PAGE.
# ---------------------------------------------------------------------------

_MM = 1 / 25.4
WIDTH_FULL = round(122 * _MM, 3)
WIDTH_HALF = round(58 * _MM, 3)
WIDTH_THIRD = round((122 - 12) / 3 * _MM, 3)

#: A full-height figure leaves no room for its caption
#: and pushes every surrounding paragraph onto another page. Half a page is the practical ceiling,
#: and `figsize` clamps to it rather than letting a tall panel grid silently overflow.
MAX_HEIGHT = 3.80   # 96 mm, half of the 193-mm LNCS text height
BAR_LABEL_PAD = 3
LOG_BAR_LIMIT_FACTOR = 4
EXPOSURE_LABEL_PAD = 5
EXPOSURE_LIMIT_FACTOR = 1.30

GOLDEN = 0.618      # height = width * GOLDEN, unless the data wants otherwise
TABLE_CURVE_LINE = 0.55
MEAN_CURVE_LINE = 1.6
GRID_TEXT_LIGHT = "white"
GRID_TEXT_DARK = "black"


def figsize(width: float = WIDTH_FULL, ratio: float = GOLDEN) -> tuple[float, float]:
    """(width, height) in inches, clamped to `MAX_HEIGHT`. `ratio` is height/width.

    Pass `WIDTH_FULL`, `WIDTH_HALF` or `WIDTH_THIRD` — never a number of your own, because the
    whole point is that the figure arrives on the page at exactly the width it was drawn at.
    """
    return (width, min(width * ratio, MAX_HEIGHT))


# ---------------------------------------------------------------------------
# What publication output requires. Everything here is about the figure being correct on paper, so it is
# the same in every project.
# ---------------------------------------------------------------------------

#: Most preferred first. DejaVu Sans is LAST and is the one that matters: it ships with matplotlib,
#: so it is the only face guaranteed present both locally and on a compute node. A missing face
#: makes matplotlib fall back silently, which changes text metrics — moving every label and making
#: a cluster-drawn figure differ from the local one for no visible reason.
_FONT_STACK = ["Source Sans 3", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]

_RC = {
    # Full text width by default, so a figure saved without thinking about it is already the
    # right size for the page.
    "figure.figsize": figsize(),
    "figure.dpi": 110,

    # constrained_layout, and NOT savefig.bbox="tight". Tight-bbox crops to the drawn content, so
    # two figures declared at the same width come out at different widths and the paper's font
    # sizes stop matching between them. constrained_layout fits the content INSIDE the declared
    # size instead. `None` is matplotlib's spelling for "use the declared figure size".
    "figure.constrained_layout.use": True,
    "savefig.bbox": None,
    "savefig.facecolor": "white",
    "savefig.transparent": False,

    # TrueType, not the default Type 3: Type 3 is rejected by several journal submission systems
    # and cannot be searched or copied out of the PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Point sizes ON THE PRINTED PAGE, since the figure is drawn at final width. 9pt sits just
    # under a paper's own 10-11pt, which reads as "part of the document" rather than shrunken;
    # 7pt is the floor below which small print stops being legible on paper.
    "font.family": "sans-serif",
    "font.sans-serif": _FONT_STACK,
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9.5,
}

# ---------------------------------------------------------------------------
# THIS PROJECT'S OWN LOOK
#
# Colours, grid, spines, line widths, marker shapes, colormaps: whatever this project's figures
# need. Put them here rather than in a notebook, so every figure changes together and a reader
# never has to re-learn a legend between two figures of the same paper.
#
# Two things worth deciding up front:
#   * Give a colour a MEANING and keep it. If a name means one thing in one figure and another
#     somewhere else, the legend has to be read twice.
#   * Check the palette is readable when printed, and in greyscale — a paper gets photocopied.
# ---------------------------------------------------------------------------

_PROJECT_RC: dict = {
    # A faint horizontal grid only. Every figure in this project reads a value off a
    # y axis (AUC, RMSE, loss, ECE); vertical rules add ink without adding a reading.
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    # No top/right spines: the plot frame is not data.
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    # Thin lines and small markers survive being printed at the final text width; the defaults are
    # sized for a screen and turn a 6-trial epoch curve into a solid band.
    "lines.linewidth": 1.4,
    "lines.markersize": 3.0,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.4,
    "axes.titlepad": 4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
    # NOTE: no `savefig.bbox` or font-type entries here on purpose — `_RC` above already
    # sets them, and its comment explains why "tight" is the wrong choice for a figure
    # drawn at a declared page width.
}


# ---------------------------------------------------------------------------
# COLOUR — by entity, never by position.
#
# Every figure in this project compares the same handful of things: four model bases,
# their untuned controls, and the GBM/linear baselines. So each one gets ONE colour, fixed
# here, and a reader who has learnt the legend once never re-learns it.
#
# Never `cm.get_cmap("tab10", n)` indexed by position in a list: dropping one arm from a
# plot would shift every colour after it.
#
# Chosen from Okabe–Ito, which is colour-vision-safe and separates in greyscale. Order
# matters: adjacent entries are the ones most often plotted side by side.
# ---------------------------------------------------------------------------

#: Registered series. APPEND, never insert — inserting repaints every figure after it,
#: including ones already in a paper.
COLORS: dict[str, str] = {
    # Four foundation-model bases, fixed across every experiment.
    "v2":            "#CC79A7",
    "v3":            "#0072B2",   # TabPFN v3
    "v2.6":          "#E69F00",   # TabPFN v2.6
    "tabicl":        "#009E73",   # TabICLv2 v2
    "tabicl-v2":     "#009E73",   # compact checkpoint label
    "pd":            "#0072B2",
    "lgd":           "#E69F00",
    # Baselines. Grey-purple-ish, deliberately duller than the families above: they are
    # the reference line, not the result.
    "xgboost":       "#7F7F7F",
    "lightgbm":      "#9E7BB5",
    "catboost":      "#8C6D31",
    "linear":        "#555555",   # logreg (PD) / ridge (LGD)
    # Roles, for figures that contrast trained against its own starting point.
    "untuned":       "#999999",
    "reference":     "#000000",   # the y=0 / y=x rule a panel is read against
    "highlight":     "#D55E00",   # the one arm a figure is about
    "annotation":    "#888888",   # "no data" text, footnotes on an axis
    "improvement":   "#2166AC",
    "deterioration": "#B2182B",
    "mean":          "#000000",
    "median":        "#CC79A7",
    "full":          "#0072B2",
    "frozen":        "#E69F00",
    "Untuned":       "#999999",
    "Adapted reference": "#0072B2",
}

#: Sequential and diverging maps, so a heatmap is not chosen per notebook either.
CMAP_SEQUENTIAL = "viridis"
CMAP_DIVERGING = "RdBu"          # blue: positive effects (improvement); red: deterioration

# Bounded pages keep long dataset and recipe lists legible at the final text width.
PAGE_ROWS = 12
SIMPLE_PAGE_ROWS = 20
PAGE_COLUMNS = 8
CONTINUOUS_TICKS = 4
LEGEND_COLUMNS = 4
JITTER_WIDTH = 0.16
PANEL_RATIO = 0.72
POINT_SIZE = 20
TABLE_POINT_SIZE = 10
POINT_ALPHA = 0.65
BAR_HEIGHT = 0.72
ANNOTATION_SIZE = 7
SAMPLING_LABELS = {"one_sample": "One sample", "full_pass": "Full pass"}
SAMPLING_COLORS = {"one_sample": "#0072B2", "full_pass": "#E69F00"}
SEED_MARKERS = ("o", "s", "^")

#: Height of one panel row in multi-row figures, plus the fixed band for titles/legend.
ROW_HEIGHT = 1.22
FIGURE_BAND = 0.85


def rows_figsize(rows: int, width: float = WIDTH_FULL, *, row_height: float = ROW_HEIGHT,
                 band: float = FIGURE_BAND) -> tuple[float, float]:
    """A figure whose height follows its number of panel rows, clamped to `MAX_HEIGHT`."""
    return (width, min(band + rows * row_height, MAX_HEIGHT))


MODEL_LABELS = {"v2": "TabPFN v2", "v2.6": "TabPFN v2.6", "v3": "TabPFN v3", "tabicl-v2": "TabICLv2"}
_MODEL_TOKEN = re.compile(r"(?<!TabPFN )(?<![\w.-])(?:tabicl-v2|v2\.6|v2|v3)(?![\w.-])")


def model_label(text):
    """Expand standalone display tokens without modifying checkpoint identifiers."""
    if not isinstance(text, str):
        return text
    return _MODEL_TOKEN.sub(lambda match: MODEL_LABELS[match.group()], text)


def page_slices(length, limit=PAGE_ROWS):
    """Balanced pages: 17 rows at a limit of 12 become 9 + 8, not 12 + 5."""
    if limit < 1:
        raise ValueError("Page limit must be positive")
    pages = int(np.ceil(length / limit))
    if not pages:
        return
    size, remainder = divmod(length, pages)
    start = 0
    for page in range(pages):
        stop = start + size + int(page < remainder)
        yield start, stop
        start = stop


def color(name: str) -> str:
    """The colour for a registered series name.

    An unregistered name gets a stable slot from `_EXTRAS` chosen by a checksum of the
    name, NOT by its position in the caller's list — so an unexpected base checkpoint
    still keeps one colour across every figure. Register it in `COLORS` when it becomes
    real.

    `crc32` and not `hash()`: string hashing is salted per interpreter, and the notebook
    runner starts one process per notebook, so `hash()` would give the same series a
    different colour in two figures of the same run.
    """
    if name in COLORS:
        return COLORS[name]
    return _EXTRAS[zlib.crc32(name.encode()) % len(_EXTRAS)]


#: A qualitative ramp for series that are NOT registered entities — dataset ids, corpus
#: members, anything whose membership changes per figure. Okabe-Ito first (colour-blind safe),
#: then Tol's bright set. Distinctness is by POSITION in the caller's list, which is what
#: `color()` cannot offer: it keys on the name so one series keeps one colour everywhere, and
#: with only four fallback slots a crc32 collision put three LGD datasets on the same yellow.
_CATEGORICAL = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77",
    "#661100", "#882255", "#AA4499", "#6699CC",
)


def categorical(names) -> dict:
    """One DISTINCT colour per name, assigned by position.

    Use for dataset ids and other per-figure series sets; use `color()` for the fixed
    entities (a base checkpoint, a baseline model) that must keep one colour project-wide.
    Beyond the palette length colours repeat, so callers drawing more series than this should
    switch to a distribution view — `too_many` is the check for that.
    """
    ordered = list(dict.fromkeys(names))
    return {n: _CATEGORICAL[i % len(_CATEGORICAL)] for i, n in enumerate(ordered)}


#: Remaining Okabe-Ito slots, for series that have not been registered yet.
_EXTRAS = ("#CC79A7", "#56B4E9", "#F0E442", "#D55E00")


def apply() -> None:
    """Install the style. Call once, at the top of every notebook and plotting script."""
    mpl.rcParams.update(_RC)
    mpl.rcParams.update(_PROJECT_RC)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[COLORS["v3"], COLORS["v2.6"], COLORS["tabicl"],
               COLORS["xgboost"], COLORS["lightgbm"], COLORS["linear"]]
    )


# ---------------------------------------------------------------------------
# SCALE — keep per-dataset detail legible as the number of plotted items grows.
#
# A figure with one bar, one row label or one panel per dataset is readable at 25 and
# unreadable at 500: the labels overlap into a grey band, the figure grows past the page,
# and the reader learns nothing. But collapsing everything to a histogram loses the
# per-dataset detail that is genuinely useful at today's size.
#
# So figures ASK how many items they have and adapt. The thresholds below are set by what
# fits a publication text block at the point sizes in `_RC`: ~28 horizontal bars at 9 pt, ~14
# rotated x labels, ~12 small-multiple panels.
# ---------------------------------------------------------------------------

#: Above this many items, a per-item bar chart becomes a distribution.
MAX_BARS = 28
#: Above this many, a small-multiples grid becomes an overlay or a summary.
MAX_PANELS = 12


def too_many(n: int, limit: int = MAX_BARS) -> bool:
    """True when `n` items will not fit legibly on a publication-width figure."""
    return int(n) > int(limit)


def note(ax, text: str) -> None:
    """A single small footnote inside the axes — for "43 of 500 shown", nothing else.

    Figures in this project carry as little text as possible: the caption in
    `output CreditPFN/figures/CAPTIONS.md` is where explanation belongs, because it travels with the
    figure into the manuscript and can be edited without re-running anything.
    """
    ax.text(0.99, 0.01, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color=COLORS["annotation"])


#: Longest axes title that fits one line of a publication-width figure at `axes.titlesize`.
MAX_TITLE = 64

#: Conservative title budget per inch at the configured full publication width.
_TITLE_CHARS_PER_INCH = MAX_TITLE / WIDTH_FULL


def title(ax, text: str) -> None:
    """Set a SHORT axes title, wrapping to the width of THIS figure rather than overrunning.

    Figures in this project end up in a paper, where the caption below the figure carries
    the explanation — so the title is a label for browsing the PDF folder, not a sentence.
    A title longer than the figure is wide silently overlaps the y tick labels.

    The budget is derived from the figure's OWN width. A fixed 52 characters is right only at
    `WIDTH_FULL`: on a `WIDTH_HALF` panel it is twice the space available, which is how
    "Calibration: trained vs its own base (ECE, lower is better)" came to be cut off mid-word
    at the right edge of a single-column figure. Wrapping is greedy over words and unlimited in
    lines — a three-line title is ugly, but a clipped one loses information.
    """
    t = " ".join(str(text).split())
    fig = ax.get_figure()
    budget = max(16, int(fig.get_size_inches()[0] * _TITLE_CHARS_PER_INCH))
    if len(t) <= budget:
        ax.set_title(t)
        return

    lines, cur = [], ""
    for word in t.split(" "):
        cand = f"{cur} {word}".strip()
        if len(cand) <= budget or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    # A trailing separator left dangling at a line end reads as a typo ("… (higher is better) ·").
    lines = [ln.rstrip(" ·,;-") for ln in lines]
    ax.set_title("\n".join(lines))


def _legend_columns(fig, labels):
    """Fit measured legend text inside the declared publication width."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties
    font = FontProperties(size=mpl.rcParams["legend.fontsize"])
    renderer = (fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer")
                else RendererAgg(int(fig.bbox.width), int(fig.bbox.height), fig.dpi))
    widths = [max(renderer.get_text_width_height_descent(line, font, False)[0]
                  for line in label.splitlines()) for label in labels]
    pixels = font.get_size_in_points() * fig.dpi / 72
    handle = (mpl.rcParams["legend.handlelength"] + mpl.rcParams["legend.handletextpad"]) * pixels
    for columns in range(min(LEGEND_COLUMNS, len(labels)), 0, -1):
        required = sum(max(column) + handle for column in np.array_split(widths, columns))
        required += (columns-1) * mpl.rcParams["legend.columnspacing"] * pixels
        required += 2 * mpl.rcParams["legend.borderpad"] * pixels
        if required <= fig.bbox.width or columns == 1:
            return columns


def finish_figure(fig):
    """Reserve a separate legend band so legends never obscure data marks."""
    import textwrap

    from matplotlib.ticker import (
        AutoLocator,
        FixedLocator,
        LogFormatterSciNotation,
        LogLocator,
        MaxNLocator,
        NullFormatter,
    )
    handles, labels = [], []
    if fig._suptitle is not None:
        budget = max(20, int(fig.get_figwidth() * _TITLE_CHARS_PER_INCH))
        fig._suptitle.set_text("\n".join(textwrap.fill(line, budget) for line in
                               model_label(fig._suptitle.get_text()).splitlines()))
    for ax in fig.axes:
        if not ax.get_visible() or ax.get_label() == "<colorbar>":
            continue
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_text(model_label(axis.label.get_text()))
            ticks = axis.get_ticklabels()
            if any(model_label(t.get_text()) != t.get_text() for t in ticks):
                axis.set_ticks(axis.get_ticklocs(), [model_label(t.get_text()) for t in ticks])
        for axis, scale, limits in ((ax.xaxis, ax.get_xscale(), ax.get_xlim()),
                                    (ax.yaxis, ax.get_yscale(), ax.get_ylim())):
            if scale == "linear" and isinstance(axis.get_major_locator(), AutoLocator):
                label = axis.get_label_text().lower()
                count = any(word in label for word in ("number of", "datasets", "outcomes", "updates"))
                fraction = any(word in label for word in ("fraction", "share", "percent", "%"))
                integer = count and not fraction
                axis.set_major_locator(MaxNLocator(nbins=CONTINUOUS_TICKS, integer=integer))
            elif scale == "log" and not isinstance(axis.get_major_locator(), FixedLocator):
                low, high = sorted(limits)
                subs = (1, 2, 5) if low > 0 and np.log10(high / low) < 2 else (1,)
                axis.set_major_locator(LogLocator(base=10, subs=subs))
                axis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False, minor_thresholds=(np.inf,np.inf)))
                axis.set_minor_formatter(NullFormatter())
        legend = ax.get_legend()
        if legend is not None:
            current_handles, current_labels = ax.get_legend_handles_labels()
            for handle, label in zip(current_handles, current_labels):
                label = model_label(label)
                if label and not label.startswith("_") and label not in labels:
                    handles.append(handle)
                    labels.append(label)
            legend.remove()
        if ax.get_title():
            panel_width = fig.get_figwidth() * ax.get_position().width
            width = max(20, int(panel_width * 13))
            # Wrap line by line: an explicit line break in a title is deliberate.
            ax.set_title("\n".join(textwrap.fill(line, width) for line in model_label(ax.get_title()).splitlines()),
                         fontsize=ax.title.get_fontsize(), color=ax.title.get_color(), fontweight=ax.title.get_fontweight())
    if handles:
        fig.legend(handles, labels, loc="outside lower center",
                   ncol=_legend_columns(fig, labels), frameon=False)
    return fig


# Domain comparison (notebooks/general/03_domain_comparison): credit corpus vs non-credit tables.
DOMAIN_COLORS = {"credit": "#0072B2", "retention": "#E69F00", "domain_control": "#009E73"}
DOMAIN_LABELS = {"credit": "Credit corpus", "retention": "Non-credit retention panel",
                 "domain_control": "Non-credit control corpus"}


# ---------------------------------------------------------------------------
# SECOND-CAMPAIGN RESULT FIGURES (src/visualize/results2.py). One meaning per encoding:
#   colour = base model (fixed), line style = which tables a monitor scores, marker = L2-SP
#   strength or training condition, ordered ramps = an ordered setting (learning rate, rows).
# ---------------------------------------------------------------------------

#: Display order of the four base models, everywhere (tables, panels, legends).
MODEL_ORDER = ("TabPFN v2", "TabPFN v2.6", "TabPFN v3", "TabICLv2")
#: Colour of each displayed model label; the same Okabe-Ito hues as the design names in COLORS.
MODEL_COLORS = {"TabPFN v2": COLORS["v2"], "TabPFN v2.6": COLORS["v2.6"],
                "TabPFN v3": COLORS["v3"], "TabICLv2": COLORS["tabicl"]}
#: Small horizontal offsets that separate the four models inside one category row.
MODEL_DODGE = {"TabPFN v2": -0.24, "TabPFN v2.6": -0.08, "TabPFN v3": 0.08, "TabICLv2": 0.24}


def model_color(label: str) -> str:
    """Colour of a model given either its display label or its design name."""
    label = model_label(str(label))
    return MODEL_COLORS.get(label, color(label))


#: Which tables a training monitor scores. Held-out credit carries the result; the other two
#: are context, so they are drawn lighter.
SCOPE_LABELS = {"test": "Held-out credit tables", "train": "Own training tables",
                "ood": "Non-credit retention panel"}
SCOPE_LINESTYLES = {"test": "-", "train": "--", "ood": ":"}
SCOPE_ALPHA = {"test": 1.0, "train": 0.75, "ood": 0.75}
SEED_BAND_ALPHA = 0.22

#: L2-SP strength (E6) as marker shape; the reference strength is the filled circle.
LAMBDA_MARKERS = {0.0: "v", 0.003: "o", 0.03: "^"}
#: Trainable scope (E7, plus the full-weight reference) as colour in the trade-off figure.
SCOPE_COLORS = {"all": "#000000", "frozen_backbone": "#7F7F7F", "diagscale": "#56B4E9",
                "lora_qv_r16": "#D55E00"}
SCOPE_NAMES = {"all": "All weights", "frozen_backbone": "Frozen backbone", "diagscale": "DiagScale",
               "lora_qv_r16": "LoRA r16 (query, value)"}

#: E5 conditions relative to the original model without fine-tuning.
CONDITION_LABELS = {"cpt": "CPT: continued pretraining only", "finetune": "FT: original fine-tuned on the fold",
                    "cpt_finetune": "CPT+FT: CPT model fine-tuned on the fold"}
CONDITION_TICKS = {"cpt": "CPT", "finetune": "FT", "cpt_finetune": "CPT+FT"}
CONDITION_MARKERS = {"cpt": "o", "finetune": "s", "cpt_finetune": "D"}

#: Original model versus its continued-pretrained counterpart in absolute-score panels.
START_LINESTYLES = {"original": "--", "cpt": "-"}
START_LABELS = {"original": "Original", "cpt": "CPT"}

#: Markers for a measured value versus a summary of measured values.
MEAN_MARKER = "D"
MEAN_MARKER_SIZE = 28
TABLE_MARKER_SIZE = 9
FILLED_ALPHA = 0.9
HOLLOW_FACE = "white"
CONNECTOR_COLOR = "#BBBBBB"
CONNECTOR_LINE = 0.5
ZERO_LINE = dict(color="#000000", linewidth=0.6, linestyle=":")
BASELINE_LINE = dict(color=COLORS["linear"], linewidth=0.9, linestyle="--")
#: Consistency colour map for metric-profile heatmaps: share of tables improved, 0 to 1.
CMAP_SHARE = "RdBu"
ORDERED_CMAP = "viridis"
ORDERED_RANGE = (0.0, 0.82)     # the yellow end of viridis is unreadable on white paper


def ordered(names) -> dict:
    """Colours for an ORDERED setting (learning rates, episode sizes, context sizes): a
    perceptually uniform ramp, so the order is visible without the legend."""
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(names))
    cmap = plt.get_cmap(ORDERED_CMAP)
    low, high = ORDERED_RANGE
    steps = np.linspace(low, high, max(len(names), 2))[:len(names)] if len(names) > 1 else [low]
    return {name: mpl.colors.to_hex(cmap(step)) for name, step in zip(names, steps)}
#: L2-SP strength as line style where lines connect learning rates (E6 dose response).
LAMBDA_LINESTYLES = {0.0: "--", 0.003: "-", 0.03: ":"}
#: Experiment family as marker in the quality-versus-cost view.
SLUG_MARKERS = {"reference": "o", "exposure": "s", "episode_size": "^", "table_size": "X"}
#: Vertical space per dataset row in per-table dot plots and per arm row in arm dot plots.
TABLE_ROW_HEIGHT = 0.16
ARM_ROW_HEIGHT = 0.26
#: Marker area range for an ordered setting encoded by size (learning rate in the trade-off view).
SIZE_RANGE = (8, 46)
#: Categorical colours for experiment arms drawn inside one model's panel. They avoid the four
#: model hues, so a colour never means "TabPFN v3" in one figure and "full pass" in another.
ARM_PALETTE = ("#D55E00", "#56B4E9", "#882255", "#7F7F7F", "#44AA99", "#332288", "#999933", "#AA4499")
#: Two learning rates of one trainable scope (E7): the lower dashed, the higher solid.
RATE_LINESTYLES = ("--", "-")
