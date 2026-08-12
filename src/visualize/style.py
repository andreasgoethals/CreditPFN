"""THE shared style. One place, every notebook in THIS project.

MOSTLY EMPTY ON PURPOSE. What the template fixes is the part that follows from the output medium:
**every figure ends up in a scientific paper printed on A4**, so it is drawn at the width it will
occupy on the page and its text is sized to be readable there. That is the same in every project,
so it is filled in below.

WHAT THIS PROJECT FILLS IN is the *look*: the colours, the grid, the spines, the marker shapes.
Two projects plotting different things have no reason to look alike, so the template does not
pretend otherwise — the requirement is only that **every notebook inside one project shares one
style**, defined here and nowhere else. A notebook never picks a colour or a size itself; if it
needs a new one, it gets added here, once, and every figure gains it at the same time. (If a new
project is close to an existing one, copying that project's `style.py` is the fastest start.)

Call `apply()` once at the top of every notebook.
"""

from __future__ import annotations

import zlib

import matplotlib as mpl

# ---------------------------------------------------------------------------
# FIGURE SIZES — A4, and nothing else.
#
# A4 is 210 x 297 mm. With 25 mm margins that leaves a 160 x 247 mm text block, and the numbers
# below are that block in inches.
#
# DRAW AT FINAL WIDTH, and never rescale a figure in the document. Rescaling carries the text with
# it: 9pt squeezed to 70% arrives as 6.3pt, under the ~7pt floor where small print stops being
# legible on paper. So the point sizes in `_RC` are the point sizes ON THE PRINTED PAGE.
# ---------------------------------------------------------------------------

WIDTH_FULL = 6.30   # 160 mm — the full A4 text width
WIDTH_HALF = 3.05   # two side by side, with a ~5 mm gutter
WIDTH_THIRD = 1.95  # three side by side. Label sparingly at this width.

#: The A4 text block is 247 mm tall, but a figure taking all of it leaves no room for its caption
#: and pushes every surrounding paragraph onto another page. Half a page is the practical ceiling,
#: and `figsize` clamps to it rather than letting a tall panel grid silently overflow.
MAX_HEIGHT = 4.80   # 122 mm

GOLDEN = 0.618      # height = width * GOLDEN, unless the data wants otherwise


def figsize(width: float = WIDTH_FULL, ratio: float = GOLDEN) -> tuple[float, float]:
    """(width, height) in inches, clamped to `MAX_HEIGHT`. `ratio` is height/width.

    Pass `WIDTH_FULL`, `WIDTH_HALF` or `WIDTH_THIRD` — never a number of your own, because the
    whole point is that the figure arrives on the page at exactly the width it was drawn at.
    """
    return (width, min(width * ratio, MAX_HEIGHT))


# ---------------------------------------------------------------------------
# What A4 output requires. Everything here is about the figure being correct on paper, so it is
# the same in every project.
# ---------------------------------------------------------------------------

#: Most preferred first. DejaVu Sans is LAST and is the one that matters: it ships with matplotlib,
#: so it is the only face guaranteed present both locally and on a compute node. A missing face
#: makes matplotlib fall back silently, which changes text metrics — moving every label and making
#: a cluster-drawn figure differ from the local one for no visible reason.
_FONT_STACK = ["Source Sans 3", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]

_RC = {
    # A4 full text width by default, so a figure saved without thinking about it is already the
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

    # Point sizes ON THE PRINTED A4 PAGE, since the figure is drawn at final width. 9pt sits just
    # under a paper's own 10-11pt, which reads as "part of the document" rather than shrunken;
    # 7pt is the floor below which small print stops being legible on paper.
    "font.family": "sans-serif",
    "font.sans-serif": _FONT_STACK,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
}

# ---------------------------------------------------------------------------
# THIS PROJECT'S OWN LOOK — fill this in.
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
    # Thin lines and small markers survive being printed at 160 mm; the defaults are
    # sized for a screen and turn a 6-trial epoch curve into a solid band.
    "lines.linewidth": 1.4,
    "lines.markersize": 3.0,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    # NOTE: no `savefig.bbox` or font-type entries here on purpose — `_RC` above already
    # sets them, and its comment explains why "tight" is the wrong choice for a figure
    # drawn at a declared page width.
}


# ---------------------------------------------------------------------------
# COLOUR — by entity, never by position.
#
# Every figure in this project compares the same handful of things: three model families,
# their untuned controls, and the GBM/linear baselines. So each one gets ONE colour, fixed
# here, and a reader who has learnt the legend once never re-learns it.
#
# This replaces what these figures used to do — `cm.get_cmap("tab10", n)` indexed by
# position in a list — where dropping one arm from a plot shifted every colour after it
# and the same model appeared blue in one figure and orange in the next.
#
# Chosen from Okabe–Ito, which is colour-vision-safe and separates in greyscale. Order
# matters: adjacent entries are the ones most often plotted side by side.
# ---------------------------------------------------------------------------

#: Registered series. APPEND, never insert — inserting repaints every figure after it,
#: including ones already in a paper.
COLORS: dict[str, str] = {
    # The three continued-pretraining families. Blue/orange/green is the widest
    # three-way separation Okabe-Ito offers, in print and in greyscale.
    "v3":            "#0072B2",   # TabPFN v3
    "v2.6":          "#E69F00",   # TabPFN v2.6
    "tabicl":        "#009E73",   # TabICLv2 v2
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
}

#: Sequential and diverging maps, so a heatmap is not chosen per notebook either.
CMAP_SEQUENTIAL = "viridis"
CMAP_DIVERGING = "RdBu_r"        # centred on 0 for delta-vs-untuned panels


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
