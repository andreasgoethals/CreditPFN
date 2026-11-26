"""Generate the appendix notebooks: every results notebook recomputed on public tables only.

``notebooks/appendix/`` holds one twin per results notebook (``SOURCES``). A twin is the source
notebook with the table scope switched to ``public`` (``campaign2.use_tables``): every loader then
drops the proprietary credit tables, so each experiment's result is recomputed on tables a reader
can obtain. Training is unchanged — the CPT corpora still contained the proprietary tables — so the
twin checks the evaluation, not the training. Training notebooks have no twin: their monitor
curves are diagnostics, not results.

The twins are GENERATED. Edit the source notebook, then run

    python -m src.utils.appendix_notebooks

``tests/test_appendix_notebooks.py`` fails when a twin's cells differ from what this module
generates from the current source. Executed outputs are kept when the cells are unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils.paths import notebooks_dir

#: Appendix notebook -> results notebook it reruns (relative to notebooks/, no suffix).
SOURCES = {
    "01_rq1_results": "rq1/02_results",
    "02_rq2_context_curves": "rq2/01_context_curves",
    "03_rq2_when_and_why": "rq2/02_when_and_why",
    "04_rq3_weight_change_results": "rq3/02_weight_change_results",
    "05_rq3_training_data_results": "rq3/04_training_data_results",
}
#: Held-out credit tables per task without the proprietary ones (checked against the registry by
#: ``tests/test_appendix_notebooks.py`` wherever the registry is available).
PUBLIC_TABLES = {"pd": 17, "lgd": 3}
ALL_TABLES = {"pd": 17, "lgd": 8}
#: Table counts written into source text, rewritten for the public scope.
COUNT_TEXT = {
    f"{ALL_TABLES['pd']} PD, {ALL_TABLES['lgd']} LGD": f"{PUBLIC_TABLES['pd']} PD, {PUBLIC_TABLES['lgd']} LGD",
    f"{ALL_TABLES['pd']} PD and {ALL_TABLES['lgd']} LGD": f"{PUBLIC_TABLES['pd']} PD and {PUBLIC_TABLES['lgd']} LGD",
}
SCOPE_LINE = "c2.use_tables('public')\n"
INTRO = ("**Appendix: public tables only.** Every section below is the one of `notebooks/{source}.ipynb`, "
         "recomputed on {pd} of {pd_all} PD and {lgd} of {lgd_all} LGD held-out tables. "
         "LGD uses the original CPT checkpoints, whose training corpora included proprietary LGD tables. "
         "With {lgd} LGD tables, LGD means rest on very few tables; the per-table rows "
         "are the evidence. In the E7 transfer matrices, proprietary tables still appear as sources (they were "
         "trained on) but no longer as targets.")


def notebooks_root() -> Path:
    return notebooks_dir()


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _rewrite(text: str) -> str:
    for old, new in COUNT_TEXT.items():
        text = text.replace(old, new)
    return text


def build(name: str, source_nb: dict) -> dict:
    """The appendix twin ``appendix/<name>`` of one results notebook (cells only, no outputs)."""
    source = SOURCES[name]
    target = f"appendix/{name}"
    cells, setup_seen = [], False
    for index, original in enumerate(source_nb["cells"]):
        text = _rewrite(_text(original))
        cell = {"cell_type": original["cell_type"], "id": f"ap{index:02d}{original.get('id', '')[:4]}",
                "metadata": {}, "source": None}
        if original["cell_type"] == "markdown" and index == 0:
            title, _, rest = text.partition("\n")
            if not title.startswith("# "):
                raise ValueError(f"{source}: first cell must start with a '# ' title")
            intro = INTRO.format(source=source, pd=PUBLIC_TABLES["pd"], pd_all=ALL_TABLES["pd"],
                                 lgd=PUBLIC_TABLES["lgd"], lgd_all=ALL_TABLES["lgd"])
            text = f"# Appendix, public tables: {title[2:]}\n\n{intro}\n{rest}"
        if original["cell_type"] == "code" and f"FigureSaver('{source}')" in text:
            if "campaign2 as c2" not in text or "style.apply()\n" not in text:
                raise ValueError(f"{source}: setup cell must import campaign2 as c2 and call style.apply()")
            text = text.replace(f"FigureSaver('{source}')", f"FigureSaver('{target}')")
            text = re.sub(r"NotebookReport\('", "NotebookReport('Appendix, public tables: ", text, count=1)
            text = text.replace("style.apply()\n", "style.apply()\n" + SCOPE_LINE, 1)
            setup_seen = True
        cell["source"] = _lines(text)
        if original["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
        if cell["cell_type"] == "code" and setup_seen and SCOPE_LINE in text:
            cells.append({"cell_type": "code", "id": f"ap{index:02d}scop", "metadata": {},
                          "execution_count": None, "outputs": [], "source": _lines(
                              "report.table('Held-out credit tables in scope', "
                              "pd.concat([c2.scope_counts(t) for t in ('pd', 'lgd')], ignore_index=True))")})
    if not setup_seen:
        raise ValueError(f"{source}: no setup cell with FigureSaver('{source}')")
    return {"cells": cells, "metadata": source_nb.get("metadata", {}),
            "nbformat": source_nb.get("nbformat", 4), "nbformat_minor": source_nb.get("nbformat_minor", 5)}


def _same_cells(a: dict, b: dict) -> bool:
    key = lambda nb: [(c["cell_type"], _text(c)) for c in nb["cells"]]  # noqa: E731
    return key(a) == key(b)


def stale() -> list[str]:
    """Appendix notebooks whose cells differ from what the current sources generate."""
    root = notebooks_root()
    out = []
    for name, source in SOURCES.items():
        path = root / "appendix" / f"{name}.ipynb"
        expected = build(name, json.loads((root / f"{source}.ipynb").read_text(encoding="utf-8")))
        if not path.exists() or not _same_cells(expected, json.loads(path.read_text(encoding="utf-8"))):
            out.append(f"appendix/{name}")
    extra = {p.stem for p in (root / "appendix").glob("*.ipynb")} - set(SOURCES)
    return out + sorted(f"appendix/{e} (no source)" for e in extra)


def write() -> list[str]:
    """(Re)generate the stale twins; a twin whose cells are current keeps its executed outputs."""
    root = notebooks_root()
    (root / "appendix").mkdir(exist_ok=True)
    written = []
    for name, source in SOURCES.items():
        path = root / "appendix" / f"{name}.ipynb"
        expected = build(name, json.loads((root / f"{source}.ipynb").read_text(encoding="utf-8")))
        if path.exists() and _same_cells(expected, json.loads(path.read_text(encoding="utf-8"))):
            continue
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        written.append(path.relative_to(root).as_posix())
    return written


if __name__ == "__main__":
    for line in write() or ["appendix notebooks are current"]:
        print(line)
