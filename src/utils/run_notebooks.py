"""Run every notebook in parallel, then rebuild the two summary documents.

    python -m src.utils.run_notebooks                     every notebook, outputs written
                                                          back into the .ipynb
    python -m src.utils.run_notebooks --only 00_general   corpus notebooks
    python -m src.utils.run_notebooks --only experiment1  the main-sweep notebooks
    python -m src.utils.run_notebooks --summaries-only    rebuild the two .md files only

    output CreditPFN/figures/<notebook>/*.pdf   written by the notebooks themselves
    output CreditPFN/figures/CAPTIONS.md        ONE file, all notebooks, notebook order
    output CreditPFN/All_Results.md             every notebook's printed summary, alphabetical

SEPARATE PROCESSES, NOT THREADS: matplotlib's figure registry is global, so two notebooks in
one interpreter would capture each other's figures — silently, giving plausible figures
attributed to the wrong notebook.

Each notebook runs in a fresh kernel and is saved with its outputs. All_Results.md reads the
final code cell's stdout directly from that notebook, including after an interactive Run All.
There are no separate notebook logs, locks or stdout caches. Missing notebook dependencies
are reported rather than silently generating figures beside an unexecuted notebook.

THE RUNNER DOES NOT SAVE FIGURES; each notebook does, through `FigureSaver`, so an interactive
*Run All* produces exactly the same PDFs. The runner adds parallelism and the two documents.

NOTEBOOKS ARE DISCOVERED, NOT LISTED, alphabetically — which is also the order in both summary
documents. A hard-coded list silently stops covering a notebook someone added.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import (
    REPO_ROOT,
    all_results_path,
    captions_path,
    figures_dir,
    notebooks_dir,
)

#: Per-cell kernel timeout, as enforced by nbclient. Model training belongs in scripts.
DEFAULT_TIMEOUT = 1800


@dataclass
class NotebookResult:
    name: str
    ok: bool
    seconds: float
    n_figures: int
    error: str = ""


def discover(names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Relative notebook names, alphabetical, excluding hidden backup folders.

    Selectors match case-insensitive substrings of the complete relative name:
    ``experiment1`` selects a study and ``results_pd`` selects its PD benchmark.
    Including the folder also distinguishes identically named notebooks in different studies.
    """
    root = notebooks_dir()
    found = tuple(sorted(p.relative_to(root).with_suffix("").as_posix()
                         for p in root.rglob("*.ipynb")
                         if not any(part.startswith(".") for part in p.relative_to(root).parts)))
    if not names:
        return found
    return tuple(s for s in found
                 if any(n.lower() in s.lower() for n in names))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _use_selector_event_loop() -> None:
    """Windows only: pick the event loop pyzmq actually needs, before a kernel starts.

    Python defaults to `ProactorEventLoop` on Windows, which has no `add_reader`. pyzmq needs
    it to talk to the kernel, so `jupyter_client` registers an extra tornado selector thread
    and emits a four-line `RuntimeWarning` per kernel — 4 workers, 4 copies, on every run.
    Harmless, and the warning names this exact fix.

    Silencing it matters only because a run that always prints warnings is a run whose
    warnings nobody reads. Set inside the worker process, so it cannot affect anything else;
    guarded by `getattr` because asyncio's policy API is on its way out.
    """
    if sys.platform != "win32":
        return
    import asyncio
    policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy is None:
        return
    try:
        asyncio.set_event_loop_policy(policy())
    except Exception:                          # pragma: no cover — never worth failing a run
        pass


def run_one(name: str, timeout: int = DEFAULT_TIMEOUT) -> NotebookResult:
    """Execute one notebook IN A KERNEL and save it with its outputs.

    Notebook execution belongs to the local analysis stage. nbclient and nbformat come with
    the notebooks extra. A failure is returned and the partially executed notebook is saved.
    """
    started = time.time()
    nb_path = notebooks_dir() / f"{name}.ipynb"
    if not nb_path.is_file():
        return NotebookResult(name, False, 0.0, 0, f"{nb_path} not found")
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError as exc:
        return NotebookResult(name, False, time.time() - started, 0,
                              f"Notebook dependency unavailable: {exc}")

    _use_selector_event_loop()

    out_dir = figures_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(nb_path, as_version=4)
    # Clear ALL old outputs first. If an early cell fails, a later unexecuted summary must
    # not survive from a previous successful run and be published as the current result.
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},   # so `from src...` resolves
        allow_errors=False,
    )
    error = ""
    try:
        client.execute()
    except CellExecutionError as exc:
        error = "\n".join(str(exc).strip().splitlines()[-12:])
    except Exception as exc:                                  # kernel died, timeout, ...
        error = f"{type(exc).__name__}: {exc}"

    # Save whatever ran, even on failure: a notebook that dies at cell 30 should still show
    # the 29 cells that worked, and the traceback is then visible where it happened.
    nbformat.write(nb, nb_path)

    n_figs = len(list(out_dir.glob("*.pdf")))
    return NotebookResult(name, not error, time.time() - started, n_figs, error)

# ---------------------------------------------------------------------------
# The two summary documents
# ---------------------------------------------------------------------------


def _notebook_summary(name: str) -> str:
    """Read the final nonempty code cell's stdout from the saved notebook."""
    path = notebooks_dir() / f"{name}.ipynb"
    if not path.is_file():
        return ""
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in nb.get("cells", [])
             if cell.get("cell_type") == "code" and "".join(cell.get("source", [])).strip()]
    if any(out.get("output_type") == "error" for cell in cells
           for out in cell.get("outputs", [])):
        return "Notebook execution failed; see the saved cell traceback."
    if not cells:
        return ""
    return "".join("".join(out.get("text", "")) for out in cells[-1].get("outputs", [])
                   if out.get("output_type") == "stream" and out.get("name") == "stdout")


def write_captions(notebooks: tuple[str, ...]) -> Path:
    """ONE CAPTIONS.md for the project, grouped per notebook, in notebook order.

    Built from each figure manifest, so it regenerates from disk after an interactive run. A
    figure with no caption gets a loud placeholder rather than being skipped — a gap should be
    visible in the document meant to contain it.
    """
    from src.visualize.figures import read_manifest

    lines = [
        "# Figure captions",
        "",
        "Generated by `python -m src.utils.run_notebooks`. Grouped by notebook, figures in",
        "the order that notebook drew them. Caption text is passed to",
        "`FigureSaver.save(..., caption=...)` in the notebook — edits here are overwritten.",
        "",
        "These are the paper's captions: paste one straight under its figure. Pure description",
        "— what is plotted, on what axes, from how much data. No interpretation.",
        "",
        "Figures are PDFs, drawn at the width they will occupy on an A4 page; never rescale one",
        "in the document, because that rescales its text with it.",
        "",
    ]
    for name in notebooks:
        entries = read_manifest(name)
        lines += [f"## {name}", ""]
        if not entries:
            lines += ["_No figures produced._", ""]
            continue
        for e in entries:
            lines.append(f"**{e['stem']}** — `{e['name']}`")
            lines.append("")
            lines.append(e["caption"] or "> MISSING CAPTION. Add one at the `save()` call.")
            lines.append("")
    path = captions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_all_results(notebooks: tuple[str, ...]) -> Path:
    """Every notebook's printed summary, concatenated. The shape is fixed:

    one block per notebook, **sorted alphabetically by notebook name**; each block is that
    notebook's printed summary (only trailing line padding removed), not a rewrite; it follows the
    notebook's own section order, so the file and the notebook read the same way round.

    Verbatim matters: the moment this file paraphrases, the two disagree and the notebook wins —
    but this file is the one anybody actually reads.
    """
    names = tuple(sorted(notebooks))
    lines = [
        "# All results",
        "",
        "Every notebook's printed summary, verbatim, one block per notebook in alphabetical",
        "order. Each block follows that notebook's own section order.",
        "Generated by `python -m src.utils.run_notebooks`.",
        "",
    ]
    for name in names:
        text = "\n".join(line.rstrip() for line in _notebook_summary(name).strip().splitlines())
        lines += ["---", "", f"## {name}", "", "```", text or "(no output captured)", "```", ""]
    path = all_results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def run_all(
    notebooks: tuple[str, ...] | None = None,
    max_workers: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[NotebookResult]:
    """Run every notebook in parallel, then rebuild both summary documents.

    Rebuilt even when a notebook failed, from whatever the successful ones wrote: a
    half-updated summary beats a stale one, and the failure is reported separately.
    """
    names = discover(notebooks)
    if not names:
        return []
    # Capped at 4: notebooks are numpy-heavy and each already uses several threads, so more
    # workers than this trades parallelism for cache thrashing.
    workers = max_workers or min(len(names), 4)

    results: list[NotebookResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, name, timeout): name for name in names}
        for fut in as_completed(futures):
            results.append(fut.result())

    # ALWAYS over every notebook, never only the ones just run. `CAPTIONS.md` and
    # `All_Results.md` are single project-wide documents assembled from each notebook's
    # figure manifests and saved notebook outputs, so a partial run must not narrow them:
    # `--only 2.0 2.1` used to cut CAPTIONS.md from 435 lines to 191, deleting four
    # notebooks' captions from what is now a tracked file.
    everything = discover()
    write_captions(everything)
    write_all_results(everything)
    return sorted(results, key=lambda r: names.index(r.name))


def summarise(results: list[NotebookResult]) -> str:
    if not results:
        return "No notebooks found in notebooks/."
    lines = ["", "=" * 74, "NOTEBOOK RUN SUMMARY", "=" * 74]
    for r in results:
        lines.append(
            f"  {'OK    ' if r.ok else 'FAILED'} {r.name:<32} "
            f"{r.seconds:6.1f}s  {r.n_figures:2d} figures"
        )
        if not r.ok:
            lines += [f"           {line}" for line in r.error.splitlines()]
    ok = sum(1 for r in results if r.ok)
    lines += [
        "",
        f"{ok}/{len(results)} notebooks OK, {sum(r.n_figures for r in results)} figures",
        f"  figures   -> {figures_dir()}",
        f"  captions  -> {captions_path()}",
        f"  summaries -> {all_results_path()}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point. `--summaries-only` exists because both documents are built from what the notebooks
# left on disk (caption metadata and .ipynb outputs), so after an interactive session they can be regenerated
# without executing anything.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="SELECTOR", help="substrings of notebook-relative paths")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per code cell")
    parser.add_argument("--summaries-only", action="store_true",
                        help="rebuild both documents from disk, run nothing")
    args = parser.parse_args(argv)

    names = discover(tuple(args.only) if args.only else None)
    if not names:
        # Distinguish "the directory is empty" from "your --only matched nothing", which are
        # very different problems and used to print the same sentence.
        if args.only:
            available = ", ".join(discover()) or "(none)"
            print(f"--only {' '.join(args.only)} matched no notebook.\nAvailable: {available}")
            return 1
        print("No notebooks found in notebooks/.")
        return 0

    if args.summaries_only:
        # Same rule as `run_all`: the two documents cover every notebook, whatever --only said.
        names = discover()
        print(f"Rebuilding summaries from disk for: {', '.join(names)}")
        print(f"  captions  -> {write_captions(names)}")
        print(f"  summaries -> {write_all_results(names)}")
        return 0

    print(f"Running {len(names)} notebook(s) in place: {', '.join(names)}")
    results = run_all(names, max_workers=args.workers, timeout=args.timeout)
    print(summarise(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
