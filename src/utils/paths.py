"""Environment-aware path resolution: local laptop vs. VSC supercomputer.

The same code base runs in two very different storage environments:

* **Local laptop / dev** — every artefact lives under the repo root
  (``data/``, ``checkpoints/``, ``results/``, ``logs/``).

* **VSC supercomputer** — datasets are too large for ``$VSC_DATA``
  (small quota, NFS, slow on big I/O) and must live on
  ``$VSC_SCRATCH`` (parallel BeeGFS, large quota, no backup).
  Conversely, *trained checkpoints* and *benchmark results* must
  live on ``$VSC_DATA`` (backed up) so they survive the periodic
  scratch purges.

How the code knows which environment it's in
--------------------------------------------
The resolver consults three sources, in order, for each kind of path:

1. **Explicit override** — the env var ``CREDITPFN_DATA_ROOT`` (for
   data) or ``CREDITPFN_OUTPUT_ROOT`` (for durable outputs). The
   slurm scripts in ``scripts/slurm/`` honour both, so a slurm-driven
   run can be fully under user control.

2. **VSC auto-detection** — if the explicit override is absent but
   ``$VSC_DATA`` is set in the environment (the VSC environment
   *always* sets this on every node, login or compute), the resolver
   picks defaults. For the **output root** this is always
   ``$VSC_DATA/CreditPFN``. For the **data root** it probes a small
   list of candidate paths and picks the first one that actually
   contains raw CSVs under ``data/raw/{pd,lgd}/``:

       i.   $VSC_SCRATCH/CreditPFN   (the documented layout)
       ii.  $VSC_SCRATCH             (raw datasets uploaded straight to scratch)
       iii. $VSC_DATA/CreditPFN      (the repo's own data/ folder)

   If none of those have data on disk (fresh checkout, first run),
   the resolver falls back to (i) so downstream "missing raw file"
   warnings still point at the canonical place to upload to.

3. **Local fallback** — if neither (1) nor (2) apply, the resolver
   uses the repo root for both. Local laptops never set ``$VSC_DATA``
   so the data folder is just ``<repo>/data/``, exactly as the
   project's untouched dev workflow expects.

A small worked example. After ``ssh login.hpc.kuleuven.be`` (so
``$VSC_DATA=/data/leuven/.../vsc12345`` is set automatically by
KU Leuven's login profile)::

    resolve_data_path("data/cached")
        # → /scratch/leuven/.../vsc12345/CreditPFN/data/cached

    resolve_output_path("checkpoints/trained")
        # → /data/leuven/.../vsc12345/CreditPFN/checkpoints/trained

…and on a laptop with no env vars set::

    resolve_data_path("data/cached")    # → <repo>/data/cached
    resolve_output_path("logs")         # → <repo>/logs

All callers funnel paths through :func:`resolve_data_path` and
:func:`resolve_output_path` rather than hardcoding ``Path(...)`` on
a config string. Absolute paths are always returned unchanged.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

# Resolve once: this module's parent's parent is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT_ENV   = "CREDITPFN_DATA_ROOT"
OUTPUT_ROOT_ENV = "CREDITPFN_OUTPUT_ROOT"

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
# in when the user hasn't set one.


def _candidate_data_roots() -> list[Path]:
    """Ordered list of VSC-side roots to probe for raw datasets."""
    out: list[Path] = []
    scratch = os.environ.get(VSC_SCRATCH_ENV)
    vsc_data = os.environ.get(VSC_DATA_ENV)
    if scratch:
        out.append(Path(scratch) / PROJECT_NAME)   # canonical
        out.append(Path(scratch))                  # no-subdir variant
    if vsc_data:
        out.append(Path(vsc_data) / PROJECT_NAME)  # home fallback
    out.append(REPO_ROOT)                          # local / dev
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
    """Resolve a *data* path (raw / processed / cached).

    On VSC: ``$VSC_SCRATCH/CreditPFN`` (auto-detected) or
    ``$CREDITPFN_DATA_ROOT`` (explicit override).
    Locally: repo root.
    """
    return _resolve(p, env_var=DATA_ROOT_ENV, vsc_default=_vsc_default_data_root())


def resolve_output_path(p: str | os.PathLike) -> Path:
    """Resolve a *durable-output* path (trained checkpoints, results,
    logs, manifests, dedup CSVs).

    On VSC: ``$VSC_DATA/CreditPFN`` (auto-detected) — backed up,
    survives scratch purges. Or ``$CREDITPFN_OUTPUT_ROOT`` (explicit
    override).
    Locally: repo root.
    """
    return _resolve(p, env_var=OUTPUT_ROOT_ENV, vsc_default=_vsc_default_output_root())


def get_roots() -> dict[str, Path]:
    """Return the *currently resolved* roots — useful for log lines /
    sanity checks at script startup."""
    return {
        "repo_root":   REPO_ROOT,
        "data_root":   _resolve_root(
            env_var=DATA_ROOT_ENV,   vsc_default=_vsc_default_data_root(),
        ),
        "output_root": _resolve_root(
            env_var=OUTPUT_ROOT_ENV, vsc_default=_vsc_default_output_root(),
        ),
    }
