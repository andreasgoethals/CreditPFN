"""Environment-aware path resolution: local laptop vs. VSC supercomputer.

The same code base runs in two storage environments, split across three
VSC tiers:

* **Local laptop / dev** — every artefact lives under the repo root
  (``data/``, ``checkpoints/``, ``output/``, ``logs/``).

* **VSC supercomputer** — three tiers, each for a different purpose:

  - **Project ("staging") storage** ``<staging>/CreditPFN`` — the
    persistent, large (≥ 1 TB), *non-purged* project share. Holds the
    BIG files: **datasets, trained checkpoints, benchmark results**.
    Resolved from ``$CREDITPFN_STAGING_ROOT`` → ``$TABPFN_STAGING_ROOT``
    → the built-in ``DEFAULT_STAGING_ROOT`` (``/lustre1/project/stg_00211``).
    It has a *low inode budget* (~150 k inodes/TB) so it's reserved for
    large files, not thousands of tiny ones.
  - **``$VSC_DATA/CreditPFN``** — NFS, backed up, tight quota. Holds the
    code and the SMALL durable outputs: logs, manifests, per-epoch CSVs,
    notebook figures.
  - **``$VSC_SCRATCH/CreditPFN``** — fast parallel FS (Lustre/GPFS) but
    purged after ~30 days. Optional fast-I/O scratch for datasets; not
    the default.

Path kinds and their resolvers
-------------------------------
:func:`resolve_data_path`
    Raw / processed datasets (the largest files). On VSC → project
    staging by default (auto-detected; falls back to scratch/$VSC_DATA
    only when the CSVs actually live there).

:func:`resolve_output_path`
    Logs, manifests, epoch snapshots, dedup CSVs, figures — small or
    temporary outputs that benefit from NFS backup.
    On VSC → ``$VSC_DATA/CreditPFN``.

:func:`resolve_staging_path`
    Trained checkpoints and bulk eval results — large, durable.
    On VSC → project staging. Falls back to :func:`resolve_output_path`
    (repo root) on a laptop.

Precedence for each resolver
------------------------------
1. Explicit env-var override (``CREDITPFN_DATA_ROOT`` /
   ``CREDITPFN_OUTPUT_ROOT`` / ``CREDITPFN_STAGING_ROOT`` |
   ``TABPFN_STAGING_ROOT``).
2. VSC auto-detection (if ``$VSC_DATA`` / ``$VSC_HOME`` is set).
3. Repo root (local laptop fallback).

Worked example on VSC (staging auto-resolved to /lustre1/project/stg_00211)::

    resolve_data_path("data/processed")
        # → /lustre1/project/stg_00211/CreditPFN/data/processed

    resolve_output_path("logs")
        # → /data/leuven/.../vsc12345/CreditPFN/logs

    resolve_staging_path("checkpoints/trained")
        # → /lustre1/project/stg_00211/CreditPFN/checkpoints/trained

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

# Project ("staging") storage — the persistent, large, non-purged VSC
# project share where the *big* files live: datasets, trained checkpoints,
# and benchmark results. Two env vars are honoured (first non-empty wins):
#   1. CREDITPFN_STAGING_ROOT — project-specific override (matches the
#      CREDITPFN_DATA_ROOT / CREDITPFN_OUTPUT_ROOT naming convention).
#   2. TABPFN_STAGING_ROOT    — the generic var the VSC allocation uses.
# When neither is set but we're on a VSC node, fall back to the BUILT-IN
# default below (the project's actual KU Leuven staging allocation). Off
# VSC (laptop) staging resolves to the repo root like everything else.
STAGING_ROOT_ENV        = "CREDITPFN_STAGING_ROOT"
STAGING_ROOT_ENV_GENERIC = "TABPFN_STAGING_ROOT"
# Built-in default — CreditPFN's KU Leuven project staging allocation.
# This is the BASE dir; the project's files live under <base>/CreditPFN.
DEFAULT_STAGING_ROOT = "/lustre1/project/stg_00211"

# VSC's own environment variables — set automatically on every VSC
# node by the user's login profile. We use them to compute sensible
# defaults when the user hasn't set the explicit CREDITPFN_* overrides.
VSC_DATA_ENV    = "VSC_DATA"
VSC_SCRATCH_ENV = "VSC_SCRATCH"

# Subdir under VSC_DATA / VSC_SCRATCH / staging that this project owns.
PROJECT_NAME = "CreditPFN"


def is_vsc_environment() -> bool:
    """True iff we are running on a VSC node.

    The KU Leuven VSC profile sets ``$VSC_DATA`` and ``$VSC_HOME``
    unconditionally on login, so either is a reliable signal.
    """
    return VSC_DATA_ENV in os.environ or "VSC_HOME" in os.environ


def _append_project(base: Path) -> Path:
    """Append the project subdir to a staging BASE path, unless it's
    already there (so both ``/lustre1/project/stg_00211`` and
    ``/lustre1/project/stg_00211/CreditPFN`` resolve to the same place)."""
    return base if base.name == PROJECT_NAME else base / PROJECT_NAME


def _staging_base() -> Path | None:
    """Resolve the staging BASE directory (before the /CreditPFN subdir), or
    None when no staging is available (laptop with no env vars set).

    Precedence:
      1. ``$CREDITPFN_STAGING_ROOT``  (project-specific override)
      2. ``$TABPFN_STAGING_ROOT``     (the allocation's generic var)
      3. built-in ``DEFAULT_STAGING_ROOT`` — but ONLY on a VSC node
      4. None                          (local laptop → caller uses repo root)
    """
    explicit = os.environ.get(STAGING_ROOT_ENV) or os.environ.get(STAGING_ROOT_ENV_GENERIC)
    if explicit:
        return Path(explicit)
    if is_vsc_environment():
        return Path(DEFAULT_STAGING_ROOT)
    return None


def _vsc_staging_root() -> Path | None:
    """The project staging root (``<base>/CreditPFN``), or None off-VSC."""
    base = _staging_base()
    return _append_project(base) if base is not None else None


def is_staging_available() -> bool:
    """True iff a project staging root resolves AND exists on disk.

    Used for sanity-check log lines; the resolvers themselves degrade
    gracefully (fall back to the repo / output root) when staging is
    absent, so callers never need to branch on this.
    """
    root = _vsc_staging_root()
    if root is None:
        return False
    try:
        return root.exists()
    except (OSError, PermissionError):                                  # pragma: no cover
        return False


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
    staging = _vsc_staging_root()
    scratch = os.environ.get(VSC_SCRATCH_ENV)
    vsc_data = os.environ.get(VSC_DATA_ENV)
    # A: project staging — the CANONICAL home for datasets (they are the
    #    largest files; staging is persistent + large + non-purged). Probed
    #    FIRST so a staging copy always wins.
    if staging is not None:
        out.append(staging)
    if scratch:
        out.append(Path(scratch) / PROJECT_NAME)   # B: scratch project subdir
        out.append(Path(scratch))                  # C: no-subdir scratch variant
    if vsc_data:
        out.append(Path(vsc_data) / PROJECT_NAME)  # D: repo's own data/ folder
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

    Datasets are the largest files in the project, so their canonical home
    is **project staging** (persistent, large, non-purged).

    Order of preference:
      1. Whichever candidate path actually has CSVs under
         ``data/raw/{pd,lgd}/`` (see :func:`_autodetect_data_root`;
         staging is probed first).
      2. The project staging root — used even when no data is on disk yet,
         so downstream "missing raw file" warnings point at the canonical
         upload location.
      3. ``$VSC_SCRATCH/CreditPFN`` if staging somehow isn't resolvable.
    """
    detected = _autodetect_data_root()
    if detected is not None:
        return detected
    staging = _vsc_staging_root()
    if staging is not None:
        return staging
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
    """Resolve a *data* path (raw / processed datasets — the largest files).

    On VSC: project staging ``<staging>/CreditPFN`` (auto-detected, the
    canonical home for datasets) or ``$CREDITPFN_DATA_ROOT`` (explicit
    override). Falls back to scratch/$VSC_DATA only when those actually
    hold the CSVs.
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
      1. Project staging ``<staging>/CreditPFN`` — resolved from
         ``$CREDITPFN_STAGING_ROOT`` / ``$TABPFN_STAGING_ROOT`` / the
         built-in ``DEFAULT_STAGING_ROOT`` on VSC (persistent, no purge).
      2. :func:`resolve_output_path` (``$VSC_DATA`` or repo root) when no
         staging is resolvable — i.e. on a laptop.

    Use for: trained ``.ckpt`` files, benchmark result CSVs.
    Do NOT use for: logs, manifests, figures — those stay on ``$VSC_DATA``
    via :func:`resolve_output_path` (small, NFS-backed, no inode pressure).

    Absolute paths are returned unchanged.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    staging = _vsc_staging_root()
    if staging is not None:
        return staging / path
    return resolve_output_path(p)


def resolve_writable_staging_path(p: str | os.PathLike) -> Path:
    """Like :func:`resolve_staging_path`, but VERIFIED writable — with an
    automatic fallback to the output root (``$VSC_DATA``) when it isn't.

    WHY (run post-mortem, 2026-07-04): all 32 PD training trials of the
    Jul-3 run died at the FIRST checkpoint save with ``[Errno 13] Permission
    denied: /lustre1/project/.../checkpoints/trained`` — project staging was
    readable from the Mindwell compute nodes (data + base checkpoints loaded
    fine) but not writable, and each trial burned ~6 minutes of B200 time
    before hitting the wall. This resolver probes writability (mkdir -p +
    touch + unlink) ONCE per process at path-resolution time — i.e. BEFORE
    any training compute — and falls back to the durable output root with a
    loud warning instead of failing 3 hours into an array.

    The probe result is cached per resolved root, so repeated calls are free.
    """
    import logging
    logger = logging.getLogger(__name__)
    path = Path(p)
    if path.is_absolute():
        return path
    staging = _vsc_staging_root()
    if staging is None:
        return resolve_output_path(p)

    cache = resolve_writable_staging_path.__dict__.setdefault("_probe_cache", {})
    key = str(staging)
    if key not in cache:
        probe_dir = staging / path
        try:
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe = probe_dir / ".write_probe"
            probe.touch()
            probe.unlink()
            cache[key] = True
        except OSError as exc:
            cache[key] = False
            logger.warning(
                "Staging root %s is NOT writable from this node (%s). "
                "Falling back to the output root for %s — artefacts will land "
                "under %s instead. Move them to staging later with "
                "scripts/slurm/stage_to_project.slurm, and check the staging "
                "dir's permissions/mount on this cluster.",
                staging, exc, p, resolve_output_path(p),
            )
    if cache[key]:
        return staging / path
    fallback = resolve_output_path(p)
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


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
# `config/data.yaml` exposes a `paths.data_source` knob with THREE allowed values:
#
#   "staging" — raw/processed live on project staging <staging>/CreditPFN
#               (persistent, large, non-purged). The DEFAULT: datasets are
#               the largest files and belong in project storage.
#   "scratch" — raw/processed live on $VSC_SCRATCH/CreditPFN
#               (fast parallel FS, but purged after ~30 days).
#   "data"    — raw/processed live on $VSC_DATA/CreditPFN
#               (NFS, backed up, tight quota).
#
# On a laptop (no VSC env vars) the knob is IGNORED and the repo's own
# data/ folder is always used — there is only one place data can live
# locally, so the toggle is meaningless.
#
# Dedup CSVs and manifests always resolve via `resolve_output_path`, which
# uses the independent `OUTPUT_ROOT_ENV` ($VSC_DATA/CreditPFN on VSC, repo
# root locally). They are small, NFS-backed, and never move.
#
# Implementation: this function sets CREDITPFN_DATA_ROOT before any path
# resolution happens. It MUST run *immediately after* `_load_cfg()` in
# each entry-point script.


_DATA_SOURCE_CHOICES = ("staging", "scratch", "data")


def apply_data_source_from_cfg(cfg) -> Path:
    """Apply ``cfg.paths.data_source`` by setting CREDITPFN_DATA_ROOT.

    Three allowed values: ``"staging"`` (default), ``"scratch"`` or
    ``"data"``. On a non-VSC machine the knob is ignored (the repo root is
    the only sensible data root).

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
    choice = str(getattr(paths_section, "data_source", "staging") or "staging")
    if choice not in _DATA_SOURCE_CHOICES:
        raise ValueError(
            f"paths.data_source={choice!r}: must be one of "
            f"{_DATA_SOURCE_CHOICES}."
        )

    if choice == "staging":
        target = _vsc_staging_root()
        if target is None:
            raise RuntimeError(
                "paths.data_source='staging' but no staging root could be "
                "resolved (set $CREDITPFN_STAGING_ROOT or $TABPFN_STAGING_ROOT)."
            )
    elif choice == "scratch":
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
