"""Publication geometry follows the actual ICML template, with print-safe exports."""

from __future__ import annotations

import matplotlib as mpl

from src.visualize import style


def test_figsize_uses_the_icml_widths_and_clamps_the_height() -> None:
    """Every figure is drawn at the width it will occupy on an ICML page, and never taller than half
    of it — a full-height figure leaves no room for its caption."""
    assert style.figsize(style.WIDTH_HALF)[0] == style.WIDTH_HALF
    assert style.WIDTH_THIRD < style.WIDTH_HALF == 3.25 < style.WIDTH_FULL == 6.75
    w, h = style.figsize()
    assert h < w
    assert style.figsize(style.WIDTH_FULL, ratio=3.0)[1] == style.MAX_HEIGHT


def test_apply_defaults_to_the_full_icml_width() -> None:
    """So a figure saved without thinking about its size is already right for the page."""
    style.apply()
    assert tuple(mpl.rcParams["figure.figsize"]) == style.figsize()


def test_apply_sets_what_icml_output_requires() -> None:
    """TrueType because journals reject Type 3; constrained_layout rather than a tight bbox because
    tight-bbox crops to the content, so two figures declared the same width come out different."""
    style.apply()
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["ps.fonttype"] == 42
    assert mpl.rcParams["figure.constrained_layout.use"] is True
    assert mpl.rcParams["savefig.bbox"] is None
    assert mpl.rcParams["font.family"] == ["sans-serif"]


def test_text_stays_legible_on_paper() -> None:
    """Point sizes are the sizes on the printed page, so none may drop under the ~7pt print floor
    or above the paper's own body text."""
    style.apply()
    for key in ("font.size", "axes.titlesize", "axes.labelsize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        assert 7 <= float(mpl.rcParams[key]) <= 11, key


def test_apply_is_idempotent() -> None:
    """A notebook re-runs its setup cell."""
    style.apply()
    style.apply()


def test_fraction_axes_keep_fractional_ticks_even_when_the_label_names_datasets():
    import matplotlib.pyplot as plt
    style.apply()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("Fraction of datasets, largest first")
    ax.set_ylabel("Fraction of clipped updates")
    style.finish_figure(fig)
    assert any(0 < tick < 1 for tick in ax.get_xticks())
    assert any(0 < tick < 1 for tick in ax.get_yticks())
    plt.close(fig)


def test_a_project_can_override_without_editing_the_icml_part() -> None:
    """`_PROJECT_RC` is applied last, so this project's look wins over the defaults but the ICML
    geometry stays where the template put it."""
    original = style._PROJECT_RC.copy()
    style._PROJECT_RC["axes.grid"] = True
    try:
        style.apply()
        assert mpl.rcParams["axes.grid"] is True
        assert mpl.rcParams["pdf.fonttype"] == 42
    finally:
        style._PROJECT_RC.clear()
        style._PROJECT_RC.update(original)
