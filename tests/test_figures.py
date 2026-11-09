"""`src/visualize/figures.py` — the saver.

Every test redirects `output CreditPFN/` into `tmp_path` via `isolated_output`, so the suite never
writes a real figure into the repository.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import pytest

from src.visualize import figures


@pytest.fixture
def fig():
    f, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    yield f
    plt.close(f)


def test_a_pdf_and_only_a_pdf_is_written(isolated_output, fig) -> None:
    """PDF only: the paper uses it, and the notebook displays the figure inline instead of
    keeping a second raster copy on disk to go stale."""
    save = figures.FigureSaver("nb")
    save(fig, "first figure", caption="A line from (0,0) to (1,1).")
    written = save.last_path
    assert written.suffix == ".pdf"
    assert written.is_file() and written.stat().st_size > 0
    assert all(p.suffix == ".pdf" for p in save.folder.iterdir() if p.is_file())
    assert "_metadata" in figures.manifest_path("nb").parts


def test_filenames_are_numbered_in_drawing_order(isolated_output, fig) -> None:
    """Alphabetical order of the files is the order the notebook drew them, which is what
    makes CAPTIONS.md rebuildable from disk without re-executing anything."""
    save = figures.FigureSaver("nb")
    save(fig, "alpha", caption="c")
    save(fig, "beta", caption="c")
    stems = sorted(p.stem for p in save.folder.glob("*.pdf"))
    assert stems == ["general__nb__01_alpha", "general__nb__02_beta"]


def test_the_folder_is_cleared_before_anything_is_drawn(isolated_output, fig) -> None:
    """A stale PDF beside a fresh one is how a paper ends up with a figure that no longer
    matches the code that made it."""
    first = figures.FigureSaver("nb")
    first(fig, "old", caption="c")
    assert (first.folder / "general__nb__01_old.pdf").is_file()

    figures.FigureSaver("nb")  # constructing it clears
    assert not (first.folder / "general__nb__01_old.pdf").exists()


def test_clearing_touches_only_this_notebook(isolated_output, fig) -> None:
    """A notebook deletes ITS OWN figures — never another notebook's."""
    a = figures.FigureSaver("nb_a")
    a(fig, "keep", caption="c")
    figures.FigureSaver("nb_b")
    assert (a.folder / "general__nb_a__01_keep.pdf").is_file()


def test_other_worker_cannot_prune_a_saver_before_its_first_write(isolated_output, fig):
    first = figures.FigureSaver("rq1/first")
    figures.clear("rq1/second")
    assert first.folder.is_dir() and figures.manifest_path(first.notebook).parent.is_dir()
    second = figures.FigureSaver("rq1/second")
    first(fig, "first", caption="First worker's plot.")
    second(fig, "second", caption="Second worker's plot.")
    assert first.last_path.is_file() and second.last_path.is_file()


def test_clearing_leaves_unrecognised_files_alone(isolated_output, fig) -> None:
    """A cleaner that removes what it does not recognise eventually removes something
    irreplaceable."""
    save = figures.FigureSaver("nb")
    stranger = save.folder / "notes.txt"
    stranger.write_text("mine", encoding="utf-8")
    figures.FigureSaver("nb")
    assert stranger.is_file()


def test_the_manifest_records_the_caption(isolated_output, fig) -> None:
    save = figures.FigureSaver("nb")
    save(fig, "hist", caption="Histogram of the target, 40 bins, n = 1,000.")
    entries = json.loads(figures.manifest_path("nb").read_text(encoding="utf-8"))
    assert entries[0]["name"] == "hist"
    assert entries[0]["caption"].startswith("Histogram")
    assert figures.read_manifest("nb") == entries


def test_the_manifest_survives_a_notebook_that_dies_halfway(isolated_output, fig) -> None:
    """Rewritten after every figure, so a crashed run still describes what it did produce."""
    save = figures.FigureSaver("nb")
    save(fig, "one", caption="c")
    assert len(figures.read_manifest("nb")) == 1


def test_a_truncated_manifest_reads_as_missing_not_as_a_crash(isolated_output) -> None:
    """A run killed mid-write leaves invalid JSON; the figures are still on disk."""
    figures.FigureSaver("nb")
    figures.manifest_path("nb").write_text('[{"index": 1,', encoding="utf-8")
    assert figures.read_manifest("nb") == []


def test_a_figure_name_cannot_escape_the_folder(isolated_output, fig) -> None:
    """A `..` or a separator in a figure name must not put a generated file outside
    `output CreditPFN/general/` — the one rule the whole layout rests on.

    Slugification neutralises it before it can become a path, so the file lands inside the
    folder rather than raising. `_guard` behind it is defence in depth: it asserts the
    invariant even if slugification is ever loosened.
    """
    save = figures.FigureSaver("nb")
    save(fig, "../../escaped", caption="c")
    pdf = save.last_path
    assert pdf.parent == save.folder
    assert ".." not in pdf.name


def test_the_write_guard_rejects_a_path_outside_the_folder(isolated_output) -> None:
    """The guard itself, tested directly — it is unreachable through `save()` by design."""
    save = figures.FigureSaver("nb")
    with pytest.raises(ValueError, match="outside"):
        figures._guard(save.folder.parent / "elsewhere.pdf", save.folder)


def test_names_are_slugified_but_stay_readable(isolated_output, fig) -> None:
    save = figures.FigureSaver("nb")
    save(fig, "Target Distribution (LGD)", caption="c")
    pdf = save.last_path
    assert pdf.stem == "general__nb__01_target_distribution__lgd"


def test_summary_flags_a_missing_caption(isolated_output, fig) -> None:
    """A missing caption must be visible in the notebook's own printed summary, not only in
    the document that is supposed to contain it."""
    save = figures.FigureSaver("nb")
    save(fig, "uncaptioned")
    assert "NO CAPTION" in save.summary()
    save2 = figures.FigureSaver("nb2")
    save2(fig, "captioned", caption="A line.")
    assert "NO CAPTION" not in save2.summary()


def test_the_figure_stays_open_so_jupyter_still_shows_it(isolated_output, fig) -> None:
    """The inline render IS the raster copy, so the figure must never be closed on save."""
    save = figures.FigureSaver("nb")
    save(fig, "shown", caption="c")
    assert plt.fignum_exists(fig.number)


def test_clear_returns_how_many_files_went(isolated_output, fig) -> None:
    save = figures.FigureSaver("nb")
    save(fig, "a", caption="c")
    save(fig, "b", caption="c")
    assert figures.clear("nb") == 3  # 2 pdfs + the manifest
    assert figures.clear("never_existed") == 0


def test_fresh_figures_clear_own_metadata_without_touching_debug_logs(isolated_output, fig):
    from src.utils.paths import logs_dir
    for name in ("current", "other"):
        save = figures.FigureSaver(name)
        save(fig, "plot", caption="Caption.")
        log = logs_dir() / f"notebook_{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("previous summary", encoding="utf-8")
    figures.clear("current")
    assert not figures.manifest_path("current").exists()
    assert (logs_dir() / "notebook_current.log").exists()
    assert figures.manifest_path("other").exists() and (logs_dir() / "notebook_other.log").exists()


def test_shared_folder_names_cannot_alias_other_notebooks(isolated_output, fig):
    from src.visualize.paths import figure_prefix
    with pytest.raises(ValueError):
        figure_prefix("rq1/a__b")
    with pytest.raises(ValueError):
        figures.clear("../elsewhere")
    assert figure_prefix("rq1/a/b") != figure_prefix("rq1/a_b")
    parent = figures.FigureSaver("rq1/a")
    nested = figures.FigureSaver("rq1/a/b")
    parent(fig, "same", caption="Parent.")
    nested(fig, "same", caption="Nested.")
    figures.clear("rq1/a")
    assert not parent.last_path.exists() and nested.last_path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows directory attributes")
def test_empty_readonly_download_directories_are_pruned(isolated_output):
    import stat

    from src.visualize.paths import figures_dir
    empty = figures_dir() / "rq1" / "_metadata" / "old_notebook"
    empty.mkdir(parents=True)
    empty.chmod(stat.S_IREAD)
    figures.clear_collection(())
    assert not empty.exists()


def test_summary_repeats_all_displayed_values_in_section_order(isolated_output):
    import numpy as np
    import pandas as pd

    from src.visualize import campaign as cp
    from src.visualize import style
    sink = figures.FigureSaver("rq1/text_contract")
    report = cp.NotebookReport("Report", sink=sink)
    frame = pd.DataFrame({"dataset": [f"Dataset {i:03}" for i in range(80)], "value": np.arange(80)})
    report.table("All datasets", frame)
    report.add("1. Inventory", "No truncation")
    fig, ax = plt.subplots(figsize=style.figsize())
    ax.imshow(np.ma.array([[1, 2], [3, 4]], mask=[[0, 0], [0, 1]]))
    page = cp.Page("matrix", fig, "Observed values; masked cells are unavailable.")
    cp.show(sink, [page])
    report.add("2. Effects", "No imputation")
    summary = report.summary(sink)
    assert summary.index("1. Inventory") < summary.index("Dataset 079") < summary.index("2. Effects")
    assert "unavailable" in summary and "..." not in summary
    assert page.caption in summary and "Figure: matrix" in summary
    report.table("Unassigned", frame.head())
    with pytest.raises(ValueError, match="assigned"):
        report.summary(sink)
