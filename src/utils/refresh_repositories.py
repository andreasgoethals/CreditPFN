"""Refresh the flat-text snapshots under ``repositories/`` via gitingest.

This module is the canonical way to bring every ``repositories/*.txt``
file back in sync with its upstream GitHub source. Each entry in the
:data:`REPOSITORIES` mapping below pairs a **filename that already
exists in** ``repositories/`` with the GitHub URL it was originally
dumped from. Running this script overwrites every listed file with a
fresh `gitingest <https://github.com/coderamp-labs/gitingest>`_ dump
**of the exact same filename** — so existing grep paths in the rest of
the codebase keep resolving.

Why not auto-derive the URLs from the existing files?
    The pretty filenames (``NanoTabPFN.txt``) don't encode the
    upstream slug, and the flattened directory tree at the top of each
    file (``automl-nanotabpfn/``) is case-lossy. An explicit mapping is
    the only correct option — and ``docs/REPOSITORIES.md`` is where
    the canonical URLs are documented anyway.

Skipped on purpose
    :data:`SKIP_NON_GIT` lists files whose upstream is not a Git repo
    (Hugging Face model card, public docs site without a GitHub mirror,
    …). They print as ``SKIP`` in the summary so it's clear they were
    intentionally not touched.

Safety guards
    * **Atomic swap.** Each refresh writes to a sibling temp file and
      only ``os.replace``-swaps it in on success. If
      ``gitingest.ingest()`` raises (network flake, rate limit, deleted
      repo, …), the existing ``.txt`` is left untouched.
    * **Shrink guard.** If the new file is < 50 % of the size of the
      existing snapshot, the swap is refused and the old file is kept.
      This catches the case where a repo has been gutted, replaced
      with a stub, or moved private. Pass ``--force-shrink`` to
      override (rare; use only when a legitimate slim-down is
      expected).

Running it
    Press *play* in your IDE on this file → refreshes every entry.
    Equivalent CLI calls::

        python src/utils/refresh_repositories.py
        python src/utils/refresh_repositories.py --only "NanoTabPFN.txt"
        python src/utils/refresh_repositories.py --only NanoTabPFN     # .txt added for you
        python src/utils/refresh_repositories.py --only "TabPFN .txt" --only "PFNS.txt"
        python src/utils/refresh_repositories.py --force-shrink

Dependencies
    ``pip install gitingest``  (https://pypi.org/project/gitingest/).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Refuse to swap if the new dump is below this fraction of the existing one.
_SHRINK_GUARD_RATIO = 0.5


# --------------------------------------------------------------------------- #
# Mapping: filename in repositories/  →  upstream GitHub URL
# --------------------------------------------------------------------------- #
#
# Edit this dict to fix a wrong URL or to add a new repo. Filenames are
# matched exactly (case + spaces + trailing dot extension all matter),
# and **the script never renames the output** — so dropping a new entry
# in here is enough to make `git pull && python refresh_repositories.py`
# bring in a new snapshot under its existing name.
#
# Sources cross-checked against docs/REPOSITORIES.md.

REPOSITORIES: dict[str, str] = {
    "NanoTabPFN.txt":
        "https://github.com/automl/nanoTabPFN",
    "On Finetuning Tabular Foundation Models.txt":
        "https://github.com/yandex-research/tabpfn-finetuning",
    "PFNS.txt":
        # automl/PFNs was moved to SamuelGabriel/PFNs; use the new canonical
        # URL directly rather than relying on the GitHub redirect.
        "https://github.com/SamuelGabriel/PFNs",
    "PFNs4BO.txt":
        "https://github.com/automl/PFNs4BO",
    "TabDPT.txt":
        "https://github.com/layer6ai-labs/TabDPT-inference",
    "TabPFN .txt":                              # the trailing space is intentional
        "https://github.com/PriorLabs/tabPFN",
    "TabPFN Client.txt":
        "https://github.com/PriorLabs/tabpfn-client",
    "TabPFN Drift-Resilient.txt":
        "https://github.com/automl/Drift-Resilient_TabPFN",
    "TabPFN Extensions.txt":
        "https://github.com/PriorLabs/tabpfn-extensions",
    "TabPFN V2 Finetuning.txt":
        # Sub-path of the main TabPFN repo (the examples/ folder).
        # gitingest's parse_remote_repo understands /tree/<branch>/<path>.
        "https://github.com/PriorLabs/TabPFN/tree/main/examples",
    "TabPFN Wide.txt":
        # Authoring lab moved from automl/ to pfeiferAI/; the old slug is
        # gone (no GitHub redirect).
        "https://github.com/pfeiferAI/TabPFN-Wide",
    "TabTune.txt":
        "https://github.com/Lexsi-Labs/TabTune",
    "TransformersCanDoBayesianInference.txt":
        # Historical: same upstream as PFNS.txt — kept as a separate file
        # for grep stability against the original 2021 PFN paper.
        "https://github.com/SamuelGabriel/PFNs",
    "VSC Documentation.txt":
        "https://github.com/hpcleuven/VscDocumentation",
}

# Files that exist in repositories/ but cannot be refreshed by gitingest.
# Each entry maps the filename to a short human-readable reason; the
# summary prints both. The existing on-disk file is never touched.
SKIP_NON_GIT: dict[str, str] = {
    "Huggingface TabPFN.txt":
        "Hugging Face model card; not a git repo. "
        "Refresh manually from https://huggingface.co/Prior-Labs/tabpfn_2_5 "
        "and https://huggingface.co/Prior-Labs/tabpfn_2_6.",
    "TabPFN Docs.txt":
        "Prior Labs no longer publishes the docs GitHub source. "
        "Refresh manually from https://docs.priorlabs.ai/overview.",
}


def _default_repositories_dir() -> Path:
    """Resolve ``<repo-root>/repositories`` from this file's location.

    Layout assumed:  ``<repo-root>/src/utils/refresh_repositories.py``.
    """
    return Path(__file__).resolve().parents[2] / "repositories"


def _silence_gitingest_logging() -> None:
    """Mute gitingest's verbose loguru output.

    gitingest streams ~15 INFO lines per repo via loguru. For interactive
    use we only care about the final outcome, so suppress those entirely.
    """
    # Stdlib bridge — covers gitingest's own logger and the noisy HTTP
    # request logs that show up as `logging:callHandlers:1762`.
    logging.getLogger("gitingest").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Native loguru sink — gitingest also logs through loguru directly.
    try:
        from loguru import logger as _loguru_logger
        _loguru_logger.disable("gitingest")
    except ImportError:                                            # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #


class _Result:
    """One row of the final summary table."""

    __slots__ = ("filename", "url", "status", "size_kb", "elapsed", "reason")

    def __init__(
        self,
        filename: str,
        url: str | None,
        status: str,                     # "OK" | "FAIL" | "SKIP"
        *,
        size_kb: float | None = None,
        elapsed: float | None = None,
        reason: str | None = None,
    ) -> None:
        self.filename = filename
        self.url      = url
        self.status   = status
        self.size_kb  = size_kb
        self.elapsed  = elapsed
        self.reason   = reason


# --------------------------------------------------------------------------- #
# One-file refresh (atomic + shrink-guard)
# --------------------------------------------------------------------------- #


def refresh_one(
    filename: str,
    url: str,
    *,
    repositories_dir: Path,
    force_shrink: bool = False,
) -> _Result:
    """Refresh a single ``repositories/<filename>`` from ``url`` atomically.

    Strategy
    --------
    1. ``gitingest.ingest(source=url, output=<filename>.refresh.tmp)`` —
       writes the new content next to the target.
    2. Sanity-check the temp file: must exist with non-zero size.
    3. Shrink guard: if a previous snapshot exists and the new dump is
       smaller than ``_SHRINK_GUARD_RATIO`` of it, refuse the swap.
       Override with ``force_shrink=True``.
    4. ``os.replace`` swaps the temp file in.
    5. On any failure, the temp file is removed and the original file
       (if any) is left untouched.
    """
    target = repositories_dir / filename
    tmp    = repositories_dir / f"{filename}.refresh.tmp"

    # Lazy import: surface a clean error if gitingest isn't installed.
    try:
        from gitingest import ingest
    except ImportError:                                            # pragma: no cover
        return _Result(filename, url, "FAIL",
                       reason="gitingest is not installed. "
                              "Install with: pip install gitingest")

    # Clean up any leftover temp from a previous interrupted run.
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:                                            # pragma: no cover
            pass

    t0 = time.monotonic()
    try:
        ingest(source=url, output=str(tmp))
    except Exception as exc:                                       # noqa: BLE001
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:                                        # pragma: no cover
                pass
        return _Result(filename, url, "FAIL",
                       reason=_summarise_exception(exc))

    # gitingest finished without raising. Sanity-check the temp file.
    if not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:                                        # pragma: no cover
                pass
        return _Result(filename, url, "FAIL",
                       reason="ingest produced an empty/missing file")

    new_size = tmp.stat().st_size
    old_size = target.stat().st_size if target.exists() else 0

    # Shrink guard: refuse to overwrite a substantially larger snapshot.
    # Suggests the upstream got gutted / went private / was replaced
    # with a stub. The existing file stays untouched.
    if (old_size > 0
            and not force_shrink
            and new_size < _SHRINK_GUARD_RATIO * old_size):
        try:
            tmp.unlink()
        except OSError:                                            # pragma: no cover
            pass
        return _Result(
            filename, url, "FAIL",
            reason=(f"shrink guard: new dump is {new_size/1024:.1f} KB, "
                    f"existing is {old_size/1024:.1f} KB "
                    f"(< {int(_SHRINK_GUARD_RATIO*100)}%). "
                    f"Existing file kept. Use --force-shrink to override."),
        )

    os.replace(tmp, target)
    return _Result(filename, url, "OK",
                   size_kb=new_size / 1024,
                   elapsed=time.monotonic() - t0)


def _summarise_exception(exc: BaseException) -> str:
    """One-line, user-readable rendering of an exception.

    Strips git's multi-line CLI output down to the most informative bit,
    keeping the summary table tidy.
    """
    msg = str(exc).strip()
    # Common: "Command failed: git ls-remote ...\nError: remote: Repository not found.\nfatal: ..."
    # Pull out the first "Error:" / "fatal:" line — it's the diagnostic.
    for marker in ("Error:", "fatal:", "Repository not found"):
        if marker in msg:
            for line in msg.splitlines():
                if marker in line:
                    return line.strip()
    # Fallback: first non-empty line.
    for line in msg.splitlines():
        if line.strip():
            return line.strip()
    return f"{type(exc).__name__}: <no message>"


# --------------------------------------------------------------------------- #
# Whole-folder refresh
# --------------------------------------------------------------------------- #


def refresh_all(
    *,
    only: list[str] | None = None,
    repositories_dir: Path | None = None,
    force_shrink: bool = False,
) -> list[_Result]:
    """Refresh every entry in :data:`REPOSITORIES`.

    Parameters
    ----------
    only
        Optional list of filenames to restrict the run to. Each entry
        is normalised: ``"NanoTabPFN"`` and ``"NanoTabPFN.txt"`` both
        work. Unknown names raise :class:`ValueError`.
    repositories_dir
        Where the ``.txt`` files live. Defaults to ``<repo-root>/repositories``.
    force_shrink
        Bypass the shrink guard. Use only when a legitimate slim-down
        of the upstream is expected.

    Returns
    -------
    list of :class:`_Result` — one row per attempted file, in the order
    they were processed (REPOSITORIES order, then SKIP_NON_GIT order).
    """
    repositories_dir = repositories_dir or _default_repositories_dir()
    if not repositories_dir.exists():
        raise FileNotFoundError(
            f"repositories/ directory not found at: {repositories_dir}"
        )

    targets = _resolve_targets(only)

    print(f"Refreshing {len(targets)} repository file(s) from "
          f"{repositories_dir}...", file=sys.stderr, flush=True)

    results: list[_Result] = []
    for i, filename in enumerate(targets, start=1):
        url = REPOSITORIES[filename]
        # Minimal in-flight signal — one line per file, no scrollback noise.
        print(f"  [{i:>2}/{len(targets)}] {filename!s}",
              file=sys.stderr, flush=True)
        results.append(
            refresh_one(filename, url,
                        repositories_dir=repositories_dir,
                        force_shrink=force_shrink)
        )

    # Skipped files — only on a full run; --only callers know what they're doing.
    if only is None:
        for filename in sorted(SKIP_NON_GIT):
            results.append(
                _Result(filename, None, "SKIP", reason=SKIP_NON_GIT[filename])
            )

    return results


def _resolve_targets(only: list[str] | None) -> list[str]:
    """Normalise the ``--only`` list against :data:`REPOSITORIES` keys."""
    if not only:
        return list(REPOSITORIES.keys())

    valid = set(REPOSITORIES.keys())
    resolved: list[str] = []
    bad: list[str] = []
    for name in only:
        candidate = name.strip()
        if candidate not in valid:
            candidate_with_ext = f"{candidate}.txt"
            if candidate_with_ext in valid:
                candidate = candidate_with_ext
            else:
                bad.append(name)
                continue
        resolved.append(candidate)

    if bad:
        valid_sorted = "\n  ".join(sorted(valid))
        raise ValueError(
            f"--only matched no known repositories: {bad!r}\n"
            f"Valid filenames are:\n  {valid_sorted}"
        )
    # Preserve order, drop duplicates while keeping first occurrence.
    seen: set[str] = set()
    return [x for x in resolved if not (x in seen or seen.add(x))]


# --------------------------------------------------------------------------- #
# Summary rendering
# --------------------------------------------------------------------------- #


def _print_summary(results: list[_Result]) -> None:
    """Print the final overview table.

    Layout
    ------
    * One short row per file: status badge + filename + size/time or reason.
    * A counts line at the bottom (OK / FAIL / SKIP).
    * If any FAIL or SKIP rows have reasons, repeat them as a block below
      so the user doesn't have to scan back through the table.
    """
    n_ok   = sum(1 for r in results if r.status == "OK")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")

    bar = "-" * 78
    print()
    print(bar)
    print(f"  Repository refresh - summary    "
          f"(OK: {n_ok}   FAIL: {n_fail}   SKIP: {n_skip})")
    print(bar)

    max_name = max((len(r.filename) for r in results), default=20)
    name_w = min(max_name, 50)

    for r in results:
        name = r.filename.ljust(name_w)
        if r.status == "OK":
            detail = f"{r.size_kb:>7.1f} KB   {r.elapsed:>5.1f}s"
            print(f"  [OK  ]  {name}   {detail}")
        elif r.status == "FAIL":
            print(f"  [FAIL]  {name}   -- see reasons below")
        else:
            print(f"  [SKIP]  {name}   -- see reasons below")

    print(bar)

    # Detailed reasons for non-OK rows, in the same order as the table.
    bad = [r for r in results if r.status in ("FAIL", "SKIP")]
    if bad:
        print("Reasons:")
        for r in bad:
            print(f"  * {r.status} {r.filename}")
            if r.url:
                print(f"      url:    {r.url}")
            print(f"      reason: {r.reason}")
        print(bar)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Refresh repositories/*.txt files via gitingest. "
                     "With no arguments, every entry in REPOSITORIES is "
                     "refreshed (non-Git sources skipped)."),
    )
    parser.add_argument(
        "--only", action="append", default=None, metavar="FILENAME",
        help=("Refresh only this filename (repeatable). Accepts either "
              "'NanoTabPFN' or 'NanoTabPFN.txt'."),
    )
    parser.add_argument(
        "--force-shrink", action="store_true",
        help=("Disable the shrink guard. Allows overwriting an existing "
              "snapshot with a new dump that is smaller than 50%% of it. "
              "Use only when a legitimate slim-down of the upstream is "
              "expected."),
    )
    args = parser.parse_args(argv)

    # Quiet, table-style output. The streaming gitingest INFO logs are
    # muted so the user only sees per-file progress + the final summary.
    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
    _silence_gitingest_logging()

    try:
        results = refresh_all(only=args.only, force_shrink=args.force_shrink)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _print_summary(results)
    n_fail = sum(1 for r in results if r.status == "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
