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

Two environment variables select between these worlds:

* ``CREDITPFN_DATA_ROOT``  — root for big I/O artefacts (the
  ``data/`` tree). Default = repo root. On VSC, set to
  ``$VSC_SCRATCH/CreditPFN``.

* ``CREDITPFN_OUTPUT_ROOT`` — root for *durable* outputs:
  ``checkpoints/trained``, ``results/``, ``logs/``. Default =
  repo root. On VSC, set to ``$VSC_DATA/CreditPFN`` so they
  survive scratch purges and are backed up.

Both default to the repo root if unset, so the local development
flow keeps working without any environment mutation.

All callers should funnel paths through :func:`resolve_data_path`
and :func:`resolve_output_path` rather than hardcoding ``Path(...)``
on a config string. The resolver leaves *absolute* paths untouched
(so a user can still hardcode an absolute path in a yaml when they
really want one).

Small worked example (on VSC, after ``export
CREDITPFN_DATA_ROOT=$VSC_SCRATCH/CreditPFN
CREDITPFN_OUTPUT_ROOT=$VSC_DATA/CreditPFN``):

    resolve_data_path("data/cached")
        # → /scratch/leuven/.../CreditPFN/data/cached
    resolve_output_path("checkpoints/trained")
        # → /data/leuven/.../CreditPFN/checkpoints/trained
"""

from __future__ import annotations

import os
from pathlib import Path

# Resolve once: this module's parent's parent is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT_ENV   = "CREDITPFN_DATA_ROOT"
OUTPUT_ROOT_ENV = "CREDITPFN_OUTPUT_ROOT"


def _resolve(p: str | os.PathLike, *, env_var: str) -> Path:
    """Resolve ``p`` against the root selected by ``env_var``.

    * Absolute path → returned as-is (no rewrite).
    * Relative path → joined to ``$<env_var>`` if set, else to
      ``REPO_ROOT``.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    root_str = os.environ.get(env_var)
    root = Path(root_str) if root_str else REPO_ROOT
    return root / path


def resolve_data_path(p: str | os.PathLike) -> Path:
    """Resolve a *data* path (raw / processed / cached / dedup / manifests).

    Driven by ``CREDITPFN_DATA_ROOT``. On VSC, point this at scratch.
    """
    return _resolve(p, env_var=DATA_ROOT_ENV)


def resolve_output_path(p: str | os.PathLike) -> Path:
    """Resolve a *durable-output* path (trained checkpoints, results, logs).

    Driven by ``CREDITPFN_OUTPUT_ROOT``. On VSC, point this at
    ``$VSC_DATA/CreditPFN`` so the artefacts survive scratch purges
    and are part of the daily backup.
    """
    return _resolve(p, env_var=OUTPUT_ROOT_ENV)


def get_roots() -> dict[str, Path]:
    """Return the resolved roots — useful for log lines / sanity checks."""
    return {
        "repo_root":   REPO_ROOT,
        "data_root":   Path(os.environ.get(DATA_ROOT_ENV) or REPO_ROOT),
        "output_root": Path(os.environ.get(OUTPUT_ROOT_ENV) or REPO_ROOT),
    }


def is_vsc_environment() -> bool:
    """True iff we are running on a VSC node (used for log decoration)."""
    return "VSC_HOME" in os.environ or "VSC_DATA" in os.environ
