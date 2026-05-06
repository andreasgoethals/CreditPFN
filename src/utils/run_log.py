"""Per-task logging utility (one file per task, flat ``logs/`` dir).

Naming convention
-----------------
Every task — one slurm job, one local script invocation — produces
**exactly one** log file with the naming convention:

    logs/<task>_<YYYYMMDD>_<HHMMSS>.log

where ``<task>`` is one of: ``data``, ``train_pd``, ``train_lgd``,
``eval_pd``, ``eval_lgd``, etc. Slurm tasks append the array IDs to
the basename to keep array tasks distinct:

    logs/train_pd_<YYYYMMDD>_<HHMMSS>_j<JOBID>_a<TASKID>.log

The log file lives under ``$CREDITPFN_OUTPUT_ROOT/logs/`` (on VSC =
``$VSC_DATA/CreditPFN/logs``, locally = repo's ``logs/``).

Two callers
-----------
* Bash slurm scripts:    compute the log path themselves (so the
                         entire script's stdout+stderr can be
                         redirected with ``exec >`` before any work
                         starts) and then pass it to Python via
                         ``--log-path``.
* Local Python scripts:  let :func:`resolve_run_log` pick a fresh
                         path; it also wires up Python's root logger
                         to write into the file *and* mirror to
                         stdout, so the user sees live output.

Both paths converge on a single :class:`RunLog` handle whose
``.write(message)`` appends one timestamped line.

Slurm vs. local handler policy
-----------------------------
On a slurm node the bash ``exec > $LOG 2>&1`` redirect already routes
stdout into the log file, so adding a Python ``FileHandler`` would
double-write. We detect slurm via ``$SLURM_JOB_ID`` and skip the
``FileHandler`` in that case — the ``StreamHandler`` (stdout) handles
everything.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path
from typing import Iterable

from src.utils.paths import resolve_output_path

DEFAULT_LOG_DIR = "logs"

# Detection: bash's `exec > $LOG 2>&1` in a slurm script means stdout
# already lands in the log file. Adding a Python FileHandler on top
# would duplicate every line. We just rely on the StreamHandler.
def _running_under_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


# --------------------------------------------------------------------------- #
# RunLog handle
# --------------------------------------------------------------------------- #


class RunLog:
    """Append one summary line per call.

    The Python logging system runs through different machinery
    (handlers attached to the root logger by :func:`setup_logging`).
    ``RunLog`` is purely for the run-summary line each top-level
    script writes after its work is done — a one-line "what
    happened" record that appears at the bottom of the log file.
    """

    def __init__(self, log_path: Path) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_top_level(self) -> bool:
        return False

    def write(self, message: str) -> None:
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"{ts}  {message}\n")

    def __repr__(self) -> str:                                # pragma: no cover
        return f"RunLog(path={self.path})"


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #


def make_task_log_path(task_name: str, *, log_dir: str | None = None) -> Path:
    """Return ``logs/<task>_<YYYYMMDD>_<HHMMSS>.log`` (resolved against
    ``$CREDITPFN_OUTPUT_ROOT``).

    On a slurm task, also appends ``_j<JOBID>_a<TASKID>`` so array
    tasks running at the same second don't clash.
    """
    base = resolve_output_path(log_dir or DEFAULT_LOG_DIR)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    job_id  = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job_id is not None:
        suffix += f"_j{job_id}"
    if task_id is not None:
        suffix += f"_a{task_id}"
    return base / f"{task_name}_{stamp}{suffix}.log"


def resolve_run_log(
    log_path: Path | str | None,
    *,
    task_name: str = "run",
    log_dir: str | None = None,
) -> tuple[RunLog, bool]:
    """Top-level vs. child resolution.

    * ``log_path`` is given (e.g. by a slurm script) → append to it
      (child mode); the FileHandler is skipped because slurm has
      already redirected stdout into that file.
    * ``log_path`` is None → build a fresh
      ``<task>_<YYYYMMDD>_<HHMMSS>.log`` and create it (top-level
      local mode); :func:`setup_logging` adds a FileHandler so the
      Python logger writes into it.

    Returns ``(RunLog, is_top_level)``.
    """
    if log_path is not None:
        return RunLog(Path(log_path)), False
    return RunLog(make_task_log_path(task_name, log_dir=log_dir)), True


# --------------------------------------------------------------------------- #
# Root-logger wiring (called by every script's ``run()``)
# --------------------------------------------------------------------------- #


def setup_logging(
    log_path: Path | str,
    *,
    level: int = logging.INFO,
    file_mode: str = "a",
    extra_handlers: Iterable[logging.Handler] | None = None,
) -> None:
    """Configure the root logger to write to stdout AND ``log_path``.

    The FileHandler is **skipped on slurm** (where bash's
    ``exec > $LOG`` already captures stdout to the same file) so the
    log file isn't double-written. Outside slurm, both handlers fire
    so the user sees live output AND the file is created locally.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Wipe any previously attached handlers (a previous run() in the
    # same process — common in tests — could leave file handlers
    # pointing at stale paths).
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if not _running_under_slurm():
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_path), mode=file_mode, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    for handler in extra_handlers or ():
        root.addHandler(handler)
