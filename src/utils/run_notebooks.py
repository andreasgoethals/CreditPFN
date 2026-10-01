"""Run every notebook in parallel, then rebuild the two summary documents.

    python -m src.utils.run_notebooks                     every notebook, outputs written
                                                          back into the .ipynb
    python -m src.utils.run_notebooks --only 00_general   corpus notebooks
    python -m src.utils.run_notebooks --only experiment1  the main-sweep notebooks
    python -m src.utils.run_notebooks --summaries-only    rebuild the two .md files only

    output CreditPFN/figures/<experiment>/*.pdf        notebook-prefixed PDFs
    output CreditPFN/figures/CAPTIONS.md                all captions in notebook order
    output CreditPFN/All_Results.md                     every notebook's printed summary, alphabetical

Separate worker processes supervise independent Jupyter kernels. Progress is collected by
the parent process, so simultaneous cell messages do not interleave. Previously expensive
notebooks start first; alphabetical publication order is unaffected. Kernel numerical-library
threads default to one to avoid nesting unrestricted BLAS pools inside parallel notebooks.

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
import multiprocessing
import os
import queue
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils.paths import (
    REPO_ROOT,
    notebooks_dir,
)
from src.visualize.paths import all_results_path, captions_path, figures_dir

#: Per-cell kernel timeout, as enforced by nbclient. Model training belongs in scripts.
DEFAULT_TIMEOUT = 1800
DEFAULT_PROGRESS_INTERVAL = 15.0
_EVENT_QUEUE = None


def _initialize_worker(events) -> None:
    global _EVENT_QUEUE
    _EVENT_QUEUE = events


def _emit(name: str, kind: str, **details) -> None:
    if _EVENT_QUEUE is not None:
        _EVENT_QUEUE.put(dict(name=name, kind=kind, **details))


def _previous_seconds(notebook: dict) -> float:
    """Estimate kernel work from nbclient's existing cell timing metadata."""
    seconds = 0.0
    for cell in notebook.get("cells", []):
        timing = cell.get("metadata", {}).get("execution", {})
        try:
            start = datetime.fromisoformat(timing["iopub.status.busy"])
            end = datetime.fromisoformat(timing["iopub.status.idle"])
            seconds += max(0.0, (end - start).total_seconds())
        except (KeyError, TypeError, ValueError):
            continue
    return seconds


def _cell_sections(notebook: dict) -> dict[int, str]:
    heading = "Setup"
    sections = {}
    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", "")
        source = source if isinstance(source, str) else "".join(source)
        if cell.get("cell_type") == "markdown":
            for line in source.splitlines():
                if line.startswith("#"):
                    heading = line.lstrip("#").strip()
        elif cell.get("cell_type") == "code" and source.strip():
            if "skip-execution" not in cell.get("metadata", {}).get("tags", []):
                sections[index] = " ".join(heading.split())[:90]
    return sections


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


def _clear_code_outputs(notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def run_one(name: str, timeout: int = DEFAULT_TIMEOUT, *,
            notebook_root: Path | None = None, threads_per_worker: int = 1) -> NotebookResult:
    """Execute one notebook IN A KERNEL and save it with its outputs.

    Notebook execution belongs to the local analysis stage. nbclient and nbformat come with
    the notebooks extra. A failure is returned and the partially executed notebook is saved.
    """
    started = time.time()
    _emit(name, "start", pid=os.getpid())
    nb_path = (notebook_root or notebooks_dir()) / f"{name}.ipynb"
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

    from src.visualize.figures import clear, owned_pdfs
    clear(name)  # also clear stale figures if execution fails before the setup cell
    out_dir = figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(nb_path, as_version=4)
    # Clear ALL old outputs first. If an early cell fails, a later unexecuted summary must
    # not survive from a previous successful run and be published as the current result.
    _clear_code_outputs(nb)
    nbformat.write(nb, nb_path)  # cancellation cannot leave a previous successful summary
    sections = _cell_sections(nb)
    positions = {index: position for position, index in enumerate(sections, 1)}

    def cell_execute(cell, cell_index):
        _emit(name, "cell", position=positions[cell_index], total=len(sections),
              section=sections[cell_index])

    def cell_executed(cell, cell_index, execute_reply):
        if positions[cell_index] == len(sections):
            _emit(name, "saving")

    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},   # so `from src...` resolves
        allow_errors=False,
        on_cell_execute=cell_execute, on_cell_executed=cell_executed,
    )
    # Use the runner's interpreter, even if PATH points at another Python or kernelspec.
    client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    kernel_env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        kernel_env[variable] = str(threads_per_worker)
    error = ""
    try:
        client.execute(env=kernel_env)
    except CellExecutionError as exc:
        error = "\n".join(str(exc).strip().splitlines()[-12:])
    except Exception as exc:                                  # kernel died, timeout, ...
        error = f"{type(exc).__name__}: {exc}"

    # Save whatever ran, even on failure: a notebook that dies at cell 30 should still show
    # the 29 cells that worked, and the traceback is then visible where it happened.
    nbformat.write(nb, nb_path)

    n_figs = len(owned_pdfs(name))
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
    from src.visualize.figures import read_manifest, migrate_collection
    from src.visualize.paths import notebook_figures_dir

    migrate_collection(notebooks)

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
        "Figures are PDFs at the ICML paper widths (6.75 inches full, 3.25 inches single column); never rescale one",
        "in the document, because that rescales its text with it.",
        "",
    ]
    for name in sorted(notebooks):
        entries = sorted(read_manifest(name), key=lambda e: e["index"])
        lines += [f"## {name}", ""]
        if not entries:
            lines += ["_No figures produced._", ""]
            continue
        for e in entries:
            relative = (notebook_figures_dir(name) / (e['stem'] + '.pdf')).relative_to(figures_dir()).as_posix()
            lines.append(f"**[{e['stem']}]({relative})** — `{e['name']}`")
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
    from src.utils.paths import all_results_path as legacy_all_results_path
    from src.visualize.figures import _prune_empty
    old = legacy_all_results_path()
    if old != path:
        old.unlink(missing_ok=True)
        _prune_empty(old.parent)
    return path


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def run_all(
    notebooks: tuple[str, ...] | None = None,
    max_workers: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    threads_per_worker: int = 1,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
) -> list[NotebookResult]:
    """Run every notebook in parallel, then rebuild both summary documents.

    Rebuilt even when a notebook failed, from whatever the successful ones wrote: a
    half-updated summary beats a stale one, and the failure is reported separately.
    """
    if max_workers is not None and max_workers < 1:
        raise ValueError("workers must be at least 1")
    if timeout < 1 or threads_per_worker < 1 or progress_interval <= 0:
        raise ValueError("timeout, threads-per-worker and progress-interval must be positive")
    names = discover(notebooks)
    if not names:
        return []
    from src.visualize.figures import clear, clear_collection
    everything = discover()
    workers = min(max_workers or 4, len(names))
    print(f"Using {workers} parallel notebook kernels; {threads_per_worker} numerical "
          "thread(s) per kernel. Progress counts cells, not equal amounts of work.", flush=True)
    estimates = {}
    for index, name in enumerate(names, 1):
        print(f"[prepare {index}/{len(names)}] {name}", flush=True)
        path = notebooks_dir() / f"{name}.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        estimates[name] = _previous_seconds(nb)
        _clear_code_outputs(nb)
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if set(names) == set(everything):
        clear_collection(names)
        all_results_path().unlink(missing_ok=True)
    else:
        for name in names:
            clear(name)
        # Even if cancelled before a worker starts, selected notebooks cannot
        # retain previous summaries/captions beside their cleared outputs.
        write_captions(everything)
        write_all_results(everything)
    # Start previously expensive notebooks first to reduce the idle tail. Publication order
    # stays alphabetical. Independent kernels do not share scientific state.
    scheduled = sorted(names, key=lambda name: (-estimates[name], name))
    print("Starting workers (longest previous execution first).", flush=True)
    results: list[NotebookResult] = []
    active = {}
    started = last_heartbeat = time.monotonic()
    try:
        with multiprocessing.Manager() as manager:
            events = manager.Queue()
            with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker,
                                     initargs=(events,)) as pool:
                futures = {pool.submit(run_one, name, timeout, notebook_root=notebooks_dir(),
                                       threads_per_worker=threads_per_worker): name
                           for name in scheduled}
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
                    while True:
                        try:
                            event = events.get_nowait()
                        except queue.Empty:
                            break
                        name, kind = event["name"], event["kind"]
                        if kind == "start":
                            active[name] = dict(start=time.monotonic(), cell="starting kernel")
                            print(f"[start] {name} (worker PID {event['pid']})", flush=True)
                        elif name in active and kind == "cell":
                            active[name]["cell"] = f"cell {event['position']}/{event['total']}: {event['section']}"
                            print(f"[running] {name} | {active[name]['cell']}", flush=True)
                        elif name in active and kind == "saving":
                            active[name]["cell"] = "saving notebook outputs"
                    for future in done:
                        name = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = NotebookResult(name, False,
                                time.monotonic() - active.get(name, {}).get("start", started),
                                0, f"Worker failed: {type(exc).__name__}: {exc}")
                        results.append(result)
                        active.pop(name, None)
                        print(f"[finished {len(results)}/{len(names)}] {name} | "
                              f"{'OK' if result.ok else 'FAILED'} | {result.seconds:.1f}s | "
                              f"{result.n_figures} figures", flush=True)
                        if not result.ok:
                            print(result.error, flush=True)
                    now = time.monotonic()
                    if now - last_heartbeat >= progress_interval:
                        print(f"[progress] {len(results)}/{len(names)} finished; "
                              f"{len(active)} running; {len(pending) - len(active)} waiting; "
                              f"elapsed {now - started:.0f}s", flush=True)
                        for name, status in active.items():
                            print(f"  {name} | {now - status['start']:.0f}s | {status['cell']}", flush=True)
                        last_heartbeat = now
    finally:
        # Also permits isolated tests using a thread executor without retaining a closed queue.
        global _EVENT_QUEUE
        _EVENT_QUEUE = None

    # ALWAYS over every notebook, never only the ones just run. `CAPTIONS.md` and
    # `All_Results.md` are single project-wide documents assembled from each notebook's
    # figure manifests and saved notebook outputs, so a partial run must not narrow them:
    # `--only 2.0 2.1` used to cut CAPTIONS.md from 435 lines to 191, deleting four
    # notebooks' captions from what is now a tracked file.
    everything = discover()
    print("Rebuilding CAPTIONS.md and All_Results.md for every notebook...", flush=True)
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
    parser.add_argument("--threads-per-worker", type=int, default=1,
                        help="numerical-library threads per notebook kernel (default: 1)")
    parser.add_argument("--progress-interval", type=float, default=DEFAULT_PROGRESS_INTERVAL,
                        help="seconds between running-cell heartbeats (default: 15)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per code cell")
    parser.add_argument("--summaries-only", action="store_true",
                        help="rebuild both documents from disk, run nothing")
    args = parser.parse_args(argv)
    if ((args.workers is not None and args.workers < 1) or args.timeout < 1
            or args.threads_per_worker < 1 or args.progress_interval <= 0):
        parser.error("workers, timeout, threads-per-worker and progress-interval must be positive")

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

    print(f"Running {len(names)} notebook(s) in place: {', '.join(names)}", flush=True)
    results = run_all(names, max_workers=args.workers, timeout=args.timeout,
                      threads_per_worker=args.threads_per_worker,
                      progress_interval=args.progress_interval)
    print(summarise(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
