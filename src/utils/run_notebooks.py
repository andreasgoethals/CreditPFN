"""Run every notebook in parallel, then rebuild the two summary documents.

    python -m src.utils.run_notebooks                     every notebook, outputs written
                                                          back into the .ipynb
    python -m src.utils.run_notebooks --script-mode        PDFs only, notebooks untouched
    python -m src.utils.run_notebooks --only exploration  substring match on the stem
    python -m src.utils.run_notebooks --only 2.0 2.1      several, e.g. both result notebooks
    python -m src.utils.run_notebooks --summaries-only    rebuild the two .md files only

    output/figures/<notebook>/*.pdf   written by the notebooks themselves
    output/figures/CAPTIONS.md        ONE file, all notebooks, notebook order
    output/All_Results.md             every notebook's printed summary, alphabetical

SEPARATE PROCESSES, NOT THREADS: matplotlib's figure registry is global, so two notebooks in
one interpreter would capture each other's figures — silently, giving plausible figures
attributed to the wrong notebook.

TWO EXECUTION PATHS. By default each notebook runs IN A KERNEL and is saved with its outputs,
so opening it shows the run that just happened. `--script-mode` flattens it to a plain script
instead: nothing extra to install, identical on the cluster, tracebacks point at a line number
rather than a cell index, and the .ipynb is left untouched — but then the notebook's stored
outputs are whatever the last interactive session left, which is a trap when the PDFs beside
them are fresh. Magics are stripped in that path, which is deliberate: a notebook needing one
cannot be executed non-interactively at all.

THE RUNNER DOES NOT SAVE FIGURES; each notebook does, through `FigureSaver`, so an interactive
*Run All* produces exactly the same PDFs. The runner adds parallelism and the two documents.

NOTEBOOKS ARE DISCOVERED, NOT LISTED, alphabetically — which is also the order in both summary
documents. A hard-coded list silently stops covering a notebook someone added.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
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

#: Per-notebook wall-clock limit. A notebook summarises a finished computation; one needing
#: longer is doing work that belongs in a script.
DEFAULT_TIMEOUT = 1800

#: Captured stdout, parked between execution and assembly, then removed. `_figures.json` is
#: KEPT: CAPTIONS.md must be rebuildable from disk without re-executing anything.
STDOUT_FILE = "_stdout.txt"


@dataclass
class NotebookResult:
    name: str
    ok: bool
    seconds: float
    n_figures: int
    error: str = ""


def discover(names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Notebook stems, alphabetical. `names` selects a subset for a partial rerun.

    A selector matches by case-insensitive SUBSTRING against the discovered stems, not by
    equality: the real stems carry a numeric prefix and a space (`0.0. raw_data_exploration`),
    so an exact-match `--only` is unusable from a shell without quoting it precisely — and
    `--only exploration`, the example this module's own docstring gives, silently became a
    "notebook not found" failure instead of running the two exploration notebooks.
    """
    found = tuple(sorted(p.stem for p in notebooks_dir().glob("*.ipynb")))
    if not names:
        return found
    return tuple(s for s in found
                 if any(n.lower() in s.lower() for n in names))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _prelude() -> str:
    """Injected above every flattened notebook. `Agg` because a compute node has no display, and
    stdout is captured so `All_Results.md` can be built without the notebook knowing."""
    return (
        "import matplotlib\n"
        'matplotlib.use("Agg")\n'
        "import io as _io\n"
        "from contextlib import redirect_stdout as _redirect\n"
        "_TEXT = _io.StringIO()\n"
    )


def _build_script(nb_path: Path, text_path: Path) -> str:
    """Flatten a notebook's code cells into one script under the capture prelude."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts = [_prelude(), "\nwith _redirect(_TEXT):\n"]
    for i, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Strip IPython magics and shell escapes: they are syntax errors in a plain
        # interpreter. A notebook that depends on one cannot be run non-interactively,
        # which the compliance rules already forbid.
        source = re.sub(r"^\s*[%!].*$", "", source, flags=re.M)
        body = "\n".join(f"    {line}" for line in source.split("\n"))
        parts.append(f"\n    # ---- cell {i} ----\n{body}\n")
    parts.append(
        "\nimport pathlib as _pl\n"
        f"_pl.Path(r{str(text_path)!r}).write_text(_TEXT.getvalue(), encoding='utf-8')\n"
    )
    return "".join(parts)


def run_one(name: str, timeout: int = DEFAULT_TIMEOUT) -> NotebookResult:
    """Execute one notebook in a fresh process. Never raises — it reports."""
    started = time.time()
    nb_path = notebooks_dir() / f"{name}.ipynb"
    if not nb_path.is_file():
        return NotebookResult(name, False, 0.0, 0, f"{nb_path} not found")

    out_dir = figures_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / STDOUT_FILE

    # The generated script goes to the system temp dir, NOT into the figure folder: the
    # notebook clears that folder as its first act, and on Windows a directory cannot be
    # modified while it holds the script currently being executed from it.
    tmp = Path(tempfile.gettempdir()) / f"nbrun_{name}.py"
    tmp.write_text(_build_script(nb_path, text_path), encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp)],
            cwd=str(REPO_ROOT),   # so `from src...` resolves without an install
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return NotebookResult(name, False, time.time() - started, 0, f"timed out after {timeout}s")
    finally:
        tmp.unlink(missing_ok=True)

    n_figs = len(list(out_dir.glob("*.pdf")))
    if proc.returncode != 0:
        # Only the tail: a full traceback from twelve notebooks buries the one that matters.
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        return NotebookResult(name, False, time.time() - started, n_figs, tail)
    return NotebookResult(name, True, time.time() - started, n_figs)



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


def run_one_in_place(name: str, timeout: int = DEFAULT_TIMEOUT) -> NotebookResult:
    """Execute one notebook IN A KERNEL and save it with its outputs.

    This is what makes opening the notebook show the current run. The flattened-script path
    (`run_one`) produces identical PDFs but cannot write outputs back, so the notebook's own
    inline figures stayed frozen at whatever the last interactive session left there — which
    is how 22 fresh PDFs came to sit beside 20 stale images and four stub panels reading
    "the eval needs both arms".

    `nbclient` ships with the `[notebooks]` extra. If it is missing we fall back to the
    script path rather than failing, because the cluster only needs the PDFs.
    """
    started = time.time()
    nb_path = notebooks_dir() / f"{name}.ipynb"
    if not nb_path.is_file():
        return NotebookResult(name, False, 0.0, 0, f"{nb_path} not found")
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError:
        return run_one(name, timeout=timeout)

    _use_selector_event_loop()

    out_dir = figures_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(nb_path, as_version=4)
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

    # `All_Results.md` reads the captured stdout from disk. In a kernel the prelude that
    # writes that file never runs, so reconstruct it from the executed cells' stream output.
    text = "".join(
        "".join(o.get("text", "")) for cell in nb.cells if cell.get("cell_type") == "code"
        for o in cell.get("outputs", []) if o.get("output_type") == "stream"
    )
    (out_dir / STDOUT_FILE).write_text(text, encoding="utf-8")

    n_figs = len(list(out_dir.glob("*.pdf")))
    return NotebookResult(name, not error, time.time() - started, n_figs, error)

# ---------------------------------------------------------------------------
# The two summary documents
# ---------------------------------------------------------------------------


def _captured_text(name: str) -> str:
    path = figures_dir(name) / STDOUT_FILE
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def write_captions(notebooks: tuple[str, ...]) -> Path:
    """ONE CAPTIONS.md for the project, grouped per notebook, in notebook order.

    Built from each `_figures.json`, so it regenerates from disk after an interactive run. A
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
    notebook's printed summary **verbatim**, not a rewrite; and that summary follows the
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
        text = _captured_text(name).strip()
        lines += ["---", "", f"## {name}", "", "```", text or "(no output captured)", "```", ""]
    path = all_results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _cleanup(notebooks: tuple[str, ...]) -> None:
    """Nothing to clean any more — kept so the call site reads the same.

    `_stdout.txt` used to be deleted here, once folded into `All_Results.md`. That made
    `--summaries-only` destructive: with no stdout on disk it rewrote every block as
    "(no output captured)", turning a 490-line document into 53 lines. Now that
    `All_Results.md` is a tracked file, that would be committed.

    So the capture is KEPT, exactly like `_figures.json` beside it, and for the same stated
    reason: both summary documents must be rebuildable from disk without re-executing
    anything. `figures._OWNED` already lists `_stdout.txt`, so each notebook's own folder is
    still cleared before it draws — the file never accumulates or goes stale.
    """
    return


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def run_all(
    notebooks: tuple[str, ...] | None = None,
    max_workers: int | None = None,
    in_place: bool = True,
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
    worker = run_one_in_place if in_place else run_one
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, name, timeout): name for name in names}
        for fut in as_completed(futures):
            results.append(fut.result())

    # ALWAYS over every notebook, never only the ones just run. `CAPTIONS.md` and
    # `All_Results.md` are single project-wide documents assembled from each notebook's
    # `_figures.json` and `_stdout.txt` on disk, so a partial run must not narrow them:
    # `--only 2.0 2.1` used to cut CAPTIONS.md from 435 lines to 191, deleting four
    # notebooks' captions from what is now a tracked file.
    everything = discover()
    write_captions(everything)
    write_all_results(everything)
    _cleanup(everything)
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
# left on disk (`_figures.json`), so after an interactive Jupyter session they can be regenerated
# without executing anything.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="STEM", help="notebook stems to run")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per notebook")
    parser.add_argument("--script-mode", action="store_true",
                        help="execute as flattened scripts; do NOT update the notebooks "
                             "(what the cluster wants — no kernel, no .ipynb churn)")
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

    how = "as scripts (notebooks NOT updated)" if args.script_mode else "in place"
    print(f"Running {len(names)} notebook(s) {how}: {', '.join(names)}")
    results = run_all(names, max_workers=args.workers, timeout=args.timeout,
                      in_place=not args.script_mode)
    print(summarise(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
