"""Stage frequent Mindwell diagnostics on GPFS, then publish to project storage."""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from src.utils.paths import training_work_dir


class TrainingFiles:
    """One writer's numeric files; final weights keep their existing publication path.

    Epoch/trajectory histories are reconstructed from the recovery checkpoint.
    Resource samples instead append across allocations, so carry those forward.
    Failed publication retains the GPFS files and fails the task, never reports OK.
    """
    SUFFIXES = (".csv", ".trajectory.csv", ".parameters.csv.gz", ".resources.csv")

    def __init__(self, destination: Path, trial: str):
        self.destination, self.trial = Path(destination), trial
        if Path(trial).name != trial or "/" in trial or "\\" in trial:
            raise ValueError("Trial must be a filename, not a path")
        self.directory = self.destination
        working = training_work_dir()
        if working is not None:
            self.directory = working / uuid.uuid4().hex

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory != self.destination:
            previous = self.destination / (self.trial + ".resources.csv")
            if previous.is_file():
                shutil.copyfile(previous, self.directory / previous.name)
        return self

    def __exit__(self, *exc):
        if self.directory == self.destination:
            return False
        self.destination.mkdir(parents=True, exist_ok=True)
        try:
            for suffix in self.SUFFIXES:
                source = self.directory / (self.trial + suffix)
                if not source.is_file():
                    continue
                target = self.destination / source.name
                pending = target.with_name("." + target.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    with source.open("rb") as src, pending.open("xb") as dst:
                        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                        dst.flush()
                        os.fsync(dst.fileno())
                    os.replace(pending, target)
                finally:
                    pending.unlink(missing_ok=True)
        except Exception:
            logging.getLogger(__name__).exception(
                "Could not publish training diagnostics; working files retained at %s", self.directory)
            raise
        # Only this newly generated directory's known files; no recursive cleanup.
        for suffix in self.SUFFIXES:
            (self.directory / (self.trial + suffix)).unlink(missing_ok=True)
        self.directory.rmdir()
        return False
