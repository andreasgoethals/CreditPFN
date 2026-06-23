"""Environment-aware path resolution: local laptop vs. VSC supercomputer.

The same code base runs in three storage environments:

* **Local laptop / dev** — every artefact lives under the repo root
  (``data/``, ``checkpoints/``, ``output/``, ``logs/``).

* **VSC supercomputer (without staging)** — datasets live on
  ``$VSC_SCRATCH`` (parallel Lustre/GPFS, large quota, no backup;
  purged every 30 days); trained checkpoints and benchmark results
  live on ``$VSC_DATA`` (NFS, backed up, survives purges).

* **VSC supercomputer (with project staging)** — large durable
  artefacts (trained checkpoints, benchmark result CSVs) live on
  the project's *staging* storage (``$CREDITPFN_STAGING_ROOT``,
  resolved from ``/staging/leuven/stg_XXXXX/CreditPFN``). Staging
  is persistent (no purge), large (≥ 1 TB), and has a *low inode
  budget* (~150 k inodes/TB), so it is used only for big-file
  outputs. Logs and manifests stay on ``$VSC_DATA``.

Path kinds and their resolvers
-------------------------------
:func:`resolve_data_path`
    Raw / processed datasets → scratch (fast parallel I/O).
    Falls back to staging if scratch has no data yet.

:func:`resolve_output_path`
    Logs, manifests, epoch snapshots, dedup CSVs — small or
    temporary outputs that benefit from NFS backup.
    On VSC → ``$VSC_DATA/CreditPFN``.

:func:`resolve_staging_path`
    Trained checkpoints and bulk eval results — large, durable,
    few files (respects staging inode budget).
    On VSC with ``$CREDITPFN_STAGING_ROOT`` set → staging.
    Falls back to :func:`resolve_output_path` if staging is unset.

Precedence for each resolver
------------------------------
1. Explicit env-var override (``CREDITPFN_DATA_ROOT``,
   ``CREDITPFN_OUTPUT_ROOT``, ``CREDITPFN_STAGING_ROOT``).
2. VSC auto-detection (if ``$VSC_DATA`` is set).
3. Repo root (local laptop fallback).

Worked example on VSC with staging configured::

    resolve_data_path("data/processed")
        # → /scratch/leuven/.../vsc12345/CreditPFN/data/processed

    resolve_output_path("logs")
        # → /data/leuven/.../vsc12345/CreditPFN/logs

    resolve_staging_path("checkpoints/trained")
        # → /staging/leuven/stg_XXXXX/CreditPFN/checkpoints/trained

On a laptop with no env vars::

    resolve_data_path("data/processed")  # → <repo>/data/processed
    resolve_output_path("logs")          # → <repo>/logs
    resolve_staging_path("checkpoints/trained")  # → <repo>/checkpoints/trained
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

# Resolve once: this module's parent's parent is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT_ENV    = "CREDITPFN_DATA_ROOT"
OUTPUT_ROOT_ENV  = "CREDITPFN_OUTPUT_ROOT"
STAGING_ROOT_ENV = "CREDITPFN_STAGING_ROOT"

# VSC's own environment variables — set automatically on every VSC
# node by the user's login profile. We use them to compute sensible
# defaults when the user hasn't set the explicit CREDITPFN_* overrides.
VSC_DATA_ENV    = "VSC_DATA"
VSC_SCRATCH_ENV = "VSC_SCRATCH"

# Subdir under VSC_DATA / VSC_SCRATCH that this project owns.
PROJECT_NAME = "CreditPFN"


def is_vsc_environment() -> bool:
    """True iff we are running on a VSC node.

    The KU Leuven VSC profile sets ``$VSC_DATA`` and ``$VSC_HOME``
    unconditionally on login, so either is a reliable signal.
    """
    return VSC_DATA_ENV in os.environ or "VSC_HOME" in os.environ


def is_staging_available() -> bool:
    """True iff project staging storage is configured and the path exists."""
    staging = os.environ.get(STAGING_ROOT_ENV)
    if not staging:
        return False
    return Path(staging).exists()


def _vsc_staging_root() -> Path | None:
    """The project staging root path, or None if not configured."""
    staging = os.environ.get(STAGING_ROOT_ENV)
    return Path(staging) if staging else None


# --------------------------------------------------------------------------- #
# Auto-detection of the data root
# --------------------------------------------------------------------------- #
#
# Historically the VSC default was hardcoded as ``$VSC_SCRATCH/CreditPFN``.
# In practice the raw datasets show up in any of three places depending
# on how the user uploaded them:
#
#     1. $VSC_SCRATCH/CreditPFN/data/raw/   ← the documented layout
#     2. $VSC_SCRATCH/data/raw/             ← straight-into-scratch, no project subdir
#     3. $VSC_DATA/CreditPFN/data/raw/      ← they sat in the repo's own data/
#                                            folder when the user cloned
#
# Rather than insist on (1) we probe all three at startup and pick the
# first one that actually contains CSVs under data/raw/{pd,lgd}/. The
# explicit env var ``CREDITPFN_DATA_ROOT`` always wins; this only kicks
# in when the user hasn't set one. ``REPO_ROOT`` is deliberately NOT a
# candidate — see the note on :func:`_candidate_data_roots`.


def _candidate_data_roots() -> list[Path]:
    """Ordered list of VSC-side roots to probe for raw datasets.

    Only consulted when we're on a VSC node (see :func:`_vsc_default_data_root`).
    We deliberately don't include ``REPO_ROOT`` here — on VSC the repo
    typically lives at ``$VSC_DATA/CreditPFN`` (= candidate #3), and on a
    laptop ``_resolve_root`` skips this whole function and uses
    ``REPO_ROOT`` directly. Including ``REPO_ROOT`` would also cause
    autodetect on a developer machine to pick the dev's repo data even
    when VSC env vars are set (e.g. in tests).
    """
    out: list[Path] = []
    scratch = os.environ.get(VSC_SCRATCH_ENV)
    vsc_data = os.environ.get(VSC_DATA_ENV)
    staging = os.environ.get(STAGING_ROOT_ENV)
    if scratch:
        out.append(Path(scratch) / PROJECT_NAME)   # A: canonical scratch
        out.append(Path(scratch))                  # B: no-subdir variant
    if vsc_data:
        out.append(Path(vsc_data) / PROJECT_NAME)  # C: repo's own data/
    if staging:
        out.append(Path(staging))                  # D: project staging (persistent, slower I/O)
    return out


def _root_has_data(root: Path) -> bool:
    """True iff ``root/data/raw/pd/`` or ``root/data/raw/lgd/`` has CSVs."""
    for track in ("pd", "lgd"):
        d = root / "data" / "raw" / track
        try:
            if d.is_dir() and next(d.glob("*.csv"), None) is not None:
                return True
        except (OSError, PermissionError):                                # pragma: no cover
            continue
    return False


@functools.cache
def _autodetect_data_root() -> Path | None:
    """Return the first candidate root that contains raw CSVs, or None.

    Memoised because we'll be called many times during a single pipeline
    run and the filesystem state doesn't change underneath us. Tests
    that monkey-patch env vars between calls should invoke
    ``_autodetect_data_root.cache_clear()`` between cases.
    """
    for candidate in _candidate_data_roots():
        if _root_has_data(candidate):
            return candidate
    return None


def _vsc_default_data_root() -> Path | None:
    """Pick a sensible data root for a VSC run.

    Order of preference:
      1. Whichever candidate path has CSVs under ``data/raw/{pd,lgd}/``
         (see :func:`_autodetect_data_root`).
      2. ``$VSC_SCRATCH/CreditPFN`` — the documented layout, used even
         when no data is on disk yet so downstream "missing raw file"
         warnings point at the canonical location.
    """
    detected = _autodetect_data_root()
    if detected is not None:
        return detected
    scratch = os.environ.get(VSC_SCRATCH_ENV)
    return Path(scratch) / PROJECT_NAME if scratch else None


def _vsc_default_output_root() -> Path | None:
    """``$VSC_DATA/CreditPFN`` if VSC_DATA is set, else None."""
    data = os.environ.get(VSC_DATA_ENV)
    return Path(data) / PROJECT_NAME if data else None


def _resolve_root(*, env_var: str, vsc_default: Path | None) -> Path:
    """Resolve the *root* a relative path should be joined to.

    Precedence:
      1. ``$<env_var>``        (explicit override; what the slurm
                                scripts set)
      2. VSC default           (only if VSC_DATA is set, i.e. we're
                                on a VSC node)
      3. ``REPO_ROOT``         (local laptop fallback)
    """
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit)
    if is_vsc_environment() and vsc_default is not None:
        return vsc_default
    return REPO_ROOT


def _resolve(p: str | os.PathLike, *, env_var: str, vsc_default: Path | None) -> Path:
    """Resolve ``p`` against the root selected by the precedence rules above.

    Absolute paths are returned unchanged (so a yaml can hardcode an
    absolute path when it really wants one).
    """
    path = Path(p)
    if path.is_absolute():
        return path
    return _resolve_root(env_var=env_var, vsc_default=vsc_default) / path


def resolve_data_path(p: str | os.PathLike) -> Path:
    """Resolve a *data* path (raw / processed).

    On VSC: ``$VSC_SCRATCH/CreditPFN`` (auto-detected) or
    ``$CREDITPFN_DATA_ROOT`` (explicit override).
    Locally: repo root.
    """
    return _resolve(p, env_var=DATA_ROOT_ENV, vsc_default=_vsc_default_data_root())


def resolve_output_path(p: str | os.PathLike) -> Path:
    """Resolve a *durable-output* path (logs, manifests, epoch snapshots, dedup CSVs).

    On VSC: ``$VSC_DATA/CreditPFN`` — NFS-backed, survives scratch purges.
    Or ``$CREDITPFN_OUTPUT_ROOT`` (explicit override).
    Locally: repo root.

    For *large* durable artefacts (trained checkpoints, bulk eval results)
    use :func:`resolve_staging_path` instead so they land on project staging.
    """
    return _resolve(p, env_var=OUTPUT_ROOT_ENV, vsc_default=_vsc_default_output_root())


def resolve_staging_path(p: str | os.PathLike) -> Path:
    """Resolve a path for large, durable artefacts: trained checkpoints and bulk results.

    Priority:
      1. ``$CREDITPFN_STAGING_ROOT``  (project staging — persistent, no purge, ≥ 1 TB)
      2. :func:`resolve_output_path`  (``$VSC_DATA`` or repo root — fallback)

    Use for: trained ``.ckpt`` files, benchmark result CSVs.
    Do NOT use for: logs, manifests, figures — those stay on ``$VSC_DATA``
    via :func:`resolve_output_path`.

    Absolute paths are returned unchanged.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    staging = os.environ.get(STAGING_ROOT_ENV)
    if staging:
        return Path(staging) / path
    return resolve_output_path(p)


def get_roots() -> dict[str, Path]:
    """Return the *currently resolved* roots — useful for log lines /
    sanity checks at script startup."""
    staging = _vsc_staging_root()
    return {
        "repo_root":    REPO_ROOT,
        "data_root":    _resolve_root(
            env_var=DATA_ROOT_ENV,   vsc_default=_vsc_default_data_root(),
        ),
        "output_root":  _resolve_root(
            env_var=OUTPUT_ROOT_ENV, vsc_default=_vsc_default_output_root(),
        ),
        "staging_root": staging if staging is not None else REPO_ROOT,
    }


# --------------------------------------------------------------------------- #
# Config-driven data-source selection
# --------------------------------------------------------------------------- #
#
# `config/data.yaml` exposes a `paths.data_source` knob with TWO allowed values:
#
#   "scratch" — raw/processed live on $VSC_SCRATCH/CreditPFN
#               (fast, purged periodically).
#   "data"    — raw/processed live on $VSC_DATA/CreditPFN
#               (durable, backed up).
#
# On a laptop (no $VSC_DATA / $VSC_SCRATCH) the knob is IGNORED and the
# repo's own data/ folder is always used — there is only one place data
# can live locally, so the toggle is meaningless.
#
# Dedup CSVs and manifests always resolve via `resolve_output_path`, which
# uses the independent `OUTPUT_ROOT_ENV` ($VSC_DATA/CreditPFN on VSC, repo
# root locally). They are the "main data directory" and never move.
#
# Implementation: this function sets CREDITPFN_DATA_ROOT before any path
# resolution happens. It MUST run *immediately after* `_load_cfg()` in
# each entry-point script.


_DATA_SOURCE_CHOICES = ("scratch", "data")


def apply_data_source_from_cfg(cfg) -> Path:
    """Apply ``cfg.paths.data_source`` by setting CREDITPFN_DATA_ROOT.

    Two allowed values: ``"scratch"`` or ``"data"``. On a non-VSC machine
    the knob is ignored (the repo root is the only sensible data root).

    Explicit ``$CREDITPFN_DATA_ROOT`` always wins (slurm submitters set it).

    Returns the resolved data root for logging.
    """
    # Slurm submit scripts may have set this explicitly — honour it.
    if os.environ.get(DATA_ROOT_ENV):
        return _resolve_root(
            env_var=DATA_ROOT_ENV, vsc_default=_vsc_default_data_root(),
        )

    # Laptop: knob has no effect — the repo's data/ folder is the only
    # place data can live. Return REPO_ROOT and leave the env var unset.
    if not is_vsc_environment():
        return REPO_ROOT

    # VSC: read the knob and translate to a concrete root.
    paths_section = getattr(cfg, "paths", None)
    choice = str(getattr(paths_section, "data_source", "scratch") or "scratch")
    if choice not in _DATA_SOURCE_CHOICES:
        raise ValueError(
            f"paths.data_source={choice!r}: must be one of "
            f"{_DATA_SOURCE_CHOICES}."
        )

    if choice == "scratch":
        scratch = os.environ.get(VSC_SCRATCH_ENV)
        if scratch is None:
            raise RuntimeError(
                "paths.data_source='scratch' but $VSC_SCRATCH is unset."
            )
        target = Path(scratch) / PROJECT_NAME
    else:  # "data"
        data = os.environ.get(VSC_DATA_ENV)
        if data is None:
            raise RuntimeError(
                "paths.data_source='data' but $VSC_DATA is unset."
            )
        target = Path(data) / PROJECT_NAME

    os.environ[DATA_ROOT_ENV] = str(target)
    # The autodetect cache was filled before we set the env var; reset it
    # so the explicit override wins on subsequent calls.
    _autodetect_data_root.cache_clear()
    return target
