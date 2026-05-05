"""Per-run logging utility.

Each top-level call into the orchestration scripts (e.g.
``scripts/data_pipeline.py``) writes a single new file under
``logs/`` named ``YYYYMMDD_HHMMSS.log`` and appends one summary line
per stage. When the same orchestration is invoked from inside another
script, no new log file is created — the summary line goes to the
caller's existing log file.

The dual-mode behaviour is controlled by passing a ``log_path`` into
``RunLog``:

* ``log_path=None`` → top-level call. A new timestamped file is
  created under ``logs/``.
* ``log_path=<existing>`` → child call. Append to the supplied path.

Public surface
--------------
* :class:`RunLog` — context-manager-friendly handle; call
  ``.write(message)`` to append a single line.
* :func:`new_run_log` — convenience constructor for the top-level
  case (fresh timestamped file).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from src.utils.paths import resolve_output_path

DEFAULT_LOG_DIR = "logs"


class RunLog:
    """Append one summary line per call.

    Parameters
    ----------
    log_path
        File to append to. If the parent directory does not yet exist
        it is created.
    """

    def __init__(self, log_path: Path) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_top_level(self) -> bool:
        return False  # Always behave as an "append" handle once built.

    def write(self, message: str) -> None:
        """Append one timestamped line."""
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"{ts}  {message}\n")

    def __repr__(self) -> str:  # pragma: no cover
        return f"RunLog(path={self.path})"


def new_run_log(log_dir: Path | str | None = None) -> RunLog:
    """Create a brand-new run-log file and return a :class:`RunLog`.

    The filename is the current local timestamp formatted as
    ``YYYYMMDD_HHMMSS.log``. Use this when starting a *top-level*
    pipeline run; pass the resulting handle's ``.path`` into any
    child orchestrator that should append to the same file.
    """
    log_dir = resolve_output_path(log_dir if log_dir is not None else DEFAULT_LOG_DIR)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return RunLog(log_dir / f"{stamp}.log")


def resolve_run_log(
    log_path: Path | str | None,
    log_dir: Path | str | None = None,
) -> tuple[RunLog, bool]:
    """Top-level vs. child resolution.

    If ``log_path`` is given, append to it (child mode). Otherwise
    create a new timestamped file under ``log_dir`` (top-level mode).

    Returns
    -------
    log : RunLog
    is_top_level : bool
        ``True`` if a fresh log file was created, ``False`` if we are
        appending to a caller-supplied file.
    """
    if log_path is not None:
        return RunLog(Path(log_path)), False
    return new_run_log(log_dir), True
