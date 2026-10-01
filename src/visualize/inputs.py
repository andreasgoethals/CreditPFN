"""Read-only folder/ZIP inputs. DATA and project downloads stay separate from generated output."""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pandas as pd


@lru_cache(maxsize=4)
def _archive(filename, size, modified):
    archive = zipfile.ZipFile(filename)
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive entries make the analysis source ambiguous")
    if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
        raise ValueError("Unsafe archive member name")
    return archive


@dataclass(frozen=True)
class ArchivePath:
    """Read-only path interface; no extraction or filesystem write methods."""
    archive: Path
    member: str = ""

    @property
    def zip(self):
        stat = self.archive.stat()
        return _archive(str(self.archive), stat.st_size, stat.st_mtime_ns)

    def __truediv__(self, part):
        value = PurePosixPath(str(part).replace("\\", "/"))
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("Analysis paths must stay inside their archive")
        return ArchivePath(self.archive, str(PurePosixPath(self.member) / value))

    def __str__(self):
        return f"{self.archive.name}!/{self.member}"

    def __lt__(self, other):
        return str(self) < str(other)

    @property
    def name(self):
        return PurePosixPath(self.member).name

    @property
    def stem(self):
        return PurePosixPath(self.member).stem

    @property
    def parent(self):
        parent = str(PurePosixPath(self.member).parent)
        return ArchivePath(self.archive, "" if parent == "." else parent)

    def resolve(self):
        return self

    def relative_to(self, other):
        if not isinstance(other, ArchivePath) or other.archive != self.archive:
            raise ValueError("Different analysis sources")
        return PurePosixPath(self.member).relative_to(other.member or ".")

    def is_file(self):
        try:
            return not self.zip.getinfo(self.member).is_dir()
        except KeyError:
            return False

    def is_dir(self):
        prefix = self.member.rstrip("/") + "/" if self.member else ""
        return any(n.startswith(prefix) for n in self.zip.namelist())

    def exists(self):
        return self.is_file() or self.is_dir()

    def glob(self, pattern):
        prefix = self.member.rstrip("/") + "/" if self.member else ""
        for name in sorted(self.zip.namelist()):
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            relative = PurePosixPath(name[len(prefix):])
            if len(relative.parts) == len(PurePosixPath(pattern).parts) and relative.match(pattern):
                yield ArchivePath(self.archive, name)

    def rglob(self, pattern):
        prefix = self.member.rstrip("/") + "/" if self.member else ""
        for name in sorted(self.zip.namelist()):
            if name.startswith(prefix) and not name.endswith("/") and PurePosixPath(name).match(pattern):
                yield ArchivePath(self.archive, name)

    def read_bytes(self):
        return self.zip.read(self.member)

    def read_text(self, encoding="utf-8", errors="strict"):
        return self.read_bytes().decode(encoding, errors)

    def stat(self):
        info = self.zip.getinfo(self.member)
        stamp = dt.datetime(*info.date_time, tzinfo=dt.timezone.utc).timestamp()
        return SimpleNamespace(st_size=info.file_size, st_mtime=stamp, st_mtime_ns=int(stamp * 1e9))


def analysis_root(*, project=False):
    """Folder or ZIP containing experiment0, experiment1, etc."""
    key = "CREDITPFN_ANALYSIS_PROJECT_ROOT" if project else "CREDITPFN_ANALYSIS_ROOT"
    value = os.environ.get(key)
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    if root.is_dir():
        return root
    if root.is_file() and root.suffix.lower() == ".zip":
        path = ArchivePath(root)
        wrapped = path / "output CreditPFN"
        return wrapped if wrapped.is_dir() else path
    raise FileNotFoundError(f"{key} must name an existing output folder or ZIP")


def analysis_location(group, area, default):
    primary, project = analysis_root(), analysis_root(project=True)
    # One selected external source never falls back to unrelated local evidence.
    root = (project or primary) if area in ("training", "results", "consolidated") else (primary or project)
    return root / group / area if root is not None else default


def read_csv(path, **kwargs):
    """Identical parsing for ordinary and zipped CSV/CSV.gz files."""
    kwargs.setdefault("float_precision", "round_trip")
    kwargs.setdefault("low_memory", False)
    if isinstance(path, ArchivePath):
        raw = path.read_bytes()
        if path.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        path = io.BytesIO(raw)
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_consolidated(run, table, **kwargs):
    from src.utils.consolidate_output import load_consolidated as read_snapshot, source_files
    from src.utils.paths import group_for_run, training_dir, manifests_dir, results_dir, consolidated_dir
    group = group_for_run(run)
    primary, project = analysis_root(), analysis_root(project=True)
    if primary is None and project is None:
        kwargs.setdefault("training_root", training_dir(experiment=group))
        return read_snapshot(run, table, **kwargs)
    roots = dict(manifest_root=analysis_location(group, "manifests", manifests_dir(group)),
                 result_root=analysis_location(group, "results", results_dir(experiment=group)),
                 training_root=analysis_location(group, "training", training_dir(experiment=group)),
                 snapshot_root=analysis_location(group, "consolidated", consolidated_dir(group)))
    parent = roots["snapshot_root"] / run
    pointer = parent / "LATEST.json"
    if not pointer.is_file():
        return None
    snapshot_name = json.loads(pointer.read_text(encoding="utf-8"))["snapshot"]
    if PurePosixPath(snapshot_name).name != snapshot_name or ".." in snapshot_name:
        raise ValueError("Invalid consolidated snapshot name")
    folder = parent / snapshot_name
    inventory = json.loads((folder / "inventory.json").read_text(encoding="utf-8"))
    # Downloads change mtimes. Check content instead, refusing stale snapshots.
    key = table.replace("trials_", "attempts_")
    actual = source_files(run, roots["manifest_root"], roots["result_root"], roots["training_root"]).get(key, [])
    root = roots["result_root"] if key.startswith("eval_") else roots["training_root"] if key.startswith(("training_", "parameters_", "resources_")) else roots["manifest_root"]
    if actual:
        recorded = {r["path"]: r["sha256"] for r in inventory["sources"].get(key, [])}
        observed = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in actual}
        if observed != recorded:
            return None
    if table not in inventory.get("tables", {}):
        return None
    path = folder / f"{table}.csv.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != inventory["tables"][table]:
        raise ValueError(f"Consolidated checksum failed for {table}")
    return read_csv(path)


def source_description():
    """Reader-facing provenance without exposing the user's home path."""
    descriptions = []
    for project, label in ((False, "DATA"), (True, "Project")):
        source = analysis_root(project=project)
        if source is None:
            descriptions.append(f"{label}: no external source selected.")
            continue
        path = source.archive if isinstance(source, ArchivePath) else source
        stamp = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds")
        descriptions.append(f"{label}: {path.name}; modified {stamp}; read only.")
    descriptions.append("Missing project records are unavailable evidence, not zero effects or failed training.")
    return "\n".join(descriptions)
