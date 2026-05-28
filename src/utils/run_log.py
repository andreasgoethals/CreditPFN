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
# Colored / structured formatter
# --------------------------------------------------------------------------- #
#
# We use ANSI escape codes for color. They render correctly in
#   * modern terminals (xterm, iTerm, Windows Terminal, VS Code)
#   * `less -R` / `cat` on Linux (preserve raw bytes)
#   * Most editors that read SLURM log files (Sublime, VS Code, vim)
# Plain text editors WILL show the raw escape codes. The runtime-color
# decision is made per-handler: stream handler gets colors (terminals
# are usually capable); file handler gets plain text (logs are mostly
# read by editors).

# Standard ANSI 8-color codes.
_ANSI_RESET   = "\033[0m"
_ANSI_BOLD    = "\033[1m"
_ANSI_DIM     = "\033[2m"
_ANSI_RED     = "\033[31m"
_ANSI_GREEN   = "\033[32m"
_ANSI_YELLOW  = "\033[33m"
_ANSI_BLUE    = "\033[34m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_CYAN    = "\033[36m"
_ANSI_WHITE   = "\033[37m"

_LEVEL_COLORS = {
    "DEBUG":    _ANSI_DIM,
    "INFO":     _ANSI_CYAN,
    "WARNING":  _ANSI_YELLOW + _ANSI_BOLD,
    "ERROR":    _ANSI_RED + _ANSI_BOLD,
    "CRITICAL": _ANSI_RED + _ANSI_BOLD,
}


class _StructuredFormatter(logging.Formatter):
    """Compact, aligned, optionally-colored log line.

    Output layout:
        HH:MM:SS [LEVEL] module : message
        |       | |    | |    | |
        +-- 8c  | +-7c-+ +18c-+ +-- variable
                +-- bracketed, padded

    Where colors are applied to the level tag and (where helpful) the
    module name. Designed so that visual scanning of a 30 MB SLURM log
    in `less` is easy — every line type has a distinct color.
    """

    def __init__(self, *, use_color: bool):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        lvl = record.levelname
        # Shorten verbose module names so the column stays aligned.
        name = record.name
        if name == "__main__":
            name = "pipeline"
        elif name.startswith("src.train.loop"):
            name = "train.loop"
        elif name.startswith("src.train.tabpfn_preprocessing"):
            name = "tabpfn.preproc"
        elif name.startswith("src.train."):
            name = name.removeprefix("src.train.")
        elif name.startswith("src.eval."):
            name = name.removeprefix("src.eval.")
        elif name.startswith("src.data."):
            name = name.removeprefix("src.data.")
        # Pad / truncate to a fixed width so columns align.
        name = (name[:18]).ljust(18)
        lvl_short = lvl[:5].ljust(5)
        msg = record.getMessage()

        if self.use_color:
            colored_lvl = (
                f"{_LEVEL_COLORS.get(lvl, '')}{lvl_short}{_ANSI_RESET}"
            )
            return f"{ts} [{colored_lvl}] {_ANSI_DIM}{name}{_ANSI_RESET} : {msg}"
        return f"{ts} [{lvl_short}] {name} : {msg}"


# --------------------------------------------------------------------------- #
# Root-logger wiring (called by every script's ``run()``)
# --------------------------------------------------------------------------- #


def _terminal_supports_color() -> bool:
    """Cheap heuristic: stdout is a TTY AND not explicitly disabled.

    Respects the standard ``NO_COLOR`` env var
    (https://no-color.org). ``FORCE_COLOR`` overrides the TTY check
    (useful for `less -R` style consumers).
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        import sys
        return bool(getattr(sys.stdout, "isatty", lambda: False)())
    except Exception:                                                  # pragma: no cover
        return False


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

    **Formatter (2026-05-28):** structured + colored. Stream handler
    gets ANSI colors when the runtime supports them (TTY, FORCE_COLOR,
    no NO_COLOR); file handler stays plain text so the log files don't
    contain escape codes that confuse plain editors. The format is:

        HH:MM:SS [LEVEL] module-padded : message

    which packs ~25 fewer characters per line than the previous
    fully-qualified-module-name format → wider room for the message.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Wipe any previously attached handlers (a previous run() in the
    # same process — common in tests — could leave file handlers
    # pointing at stale paths).
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    # Stream handler: colored when possible.
    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(_StructuredFormatter(use_color=_terminal_supports_color()))
    root.addHandler(stream)

    if not _running_under_slurm():
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_path), mode=file_mode, encoding="utf-8")
        fh.setLevel(level)
        # File handler: plain — log files are usually read in editors
        # which don't render ANSI codes.
        fh.setFormatter(_StructuredFormatter(use_color=False))
        root.addHandler(fh)

    for handler in extra_handlers or ():
        root.addHandler(handler)
