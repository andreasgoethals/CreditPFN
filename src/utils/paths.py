"""Every path in the project. Two VSC tiers, one resolver, relative to the repository root.

THE ONLY MODULE THAT BUILDS A PATH — everything else asks this one. A path assembled at a call
site with `"output/" + name` is correct on a laptop and wrong on the cluster, and the failure
shows up as a full quota or an empty results directory hours into a job.

        project storage  /lustre1/project/stg_00211/<Project>/  big files, allocation-specific quotas
    personal data    $VSC_DATA/<Project>/                   repo + output/, backed up, 75 GiB
    scratch          $VSC_SCRATCH/                          purged after 30 days of no ACCESS

DATA has site-documented snapshots. Project storage quotas and backup policy are
allocation-specific; verify them instead of assuming that persistent means backed up.
Both tiers can be inspected from the cluster shell.

Results and consolidated tables belong on project storage. Live
logs and per-trial metadata stay on DATA until post-run consolidation; both bytes
and inodes require monitoring. See docs/VSC.md.

OFF-CLUSTER EVERY TIER COLLAPSES INTO THE REPO. Pretending `/lustre1` exists on a laptop would
mean two code paths, and the one that only runs on the cluster is the one that breaks.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

#: Filled in by `_template/init_project.py`. The per-project folder name on BOTH shared tiers.
PROJECT_NAME = "CreditPFN"

#: Fallback for project storage when the site variable is not set. The literal path is a
#: last resort, not the primary source — VSC has moved it before.
STAGING_FALLBACK = "/lustre1/project/stg_00211"

#: `parents[2]` because this file is `<root>/src/utils/paths.py`. From __file__, not the working
#: directory, so a script, a test and a notebook agree wherever they were launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Overrides for project storage, in priority order. The supported way to put big files on an
#: external drive locally, and how the tests exercise the staging branch without a cluster.
#: `TABPFN_STAGING_ROOT` is the allocation-wide variable this project inherited from its
#: predecessor and every SLURM script still exports one of the first two.
STAGING_ENV_VARS = ("CREDITPFN_STAGING_ROOT", "TABPFN_STAGING_ROOT", "VSC_STAGING_ROOT")


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


# ---------------------------------------------------------------------------
# Which world are we in?
# ---------------------------------------------------------------------------


def on_vsc() -> bool:
    """True on a VSC node. `$VSC_DATA` is set by the site and by nothing else."""
    return bool(os.environ.get("VSC_DATA"))


def staging_override() -> Path | None:
    """An explicitly requested project-storage root, or None.

    Separate from `staging_root()` because the big-file helpers short-circuit to the repo when
    off-cluster, which would make an override silently do nothing on a laptop.
    """
    for var in STAGING_ENV_VARS:
        p = _env_path(var)
        if p:
            return p
    return None


def _use_staging() -> bool:
    """Should big files go to project storage rather than into the repo?"""
    return on_vsc() or staging_override() is not None


# ---------------------------------------------------------------------------
# The three roots.
# ---------------------------------------------------------------------------


def staging_root() -> Path:
    """Project storage — verify the allocation's quotas and backup policy separately."""
    override = staging_override()
    if override:
        return override
    lustre = _env_path("VSC_PROJECT_LUSTRE1")
    if lustre:
        return lustre / "stg_00211"
    if on_vsc():
        return Path(STAGING_FALLBACK)
    return REPO_ROOT


def data_root() -> Path:
    """Personal data — the small, backed-up tier. The repo lives here on the cluster."""
    return _env_path("VSC_DATA") or REPO_ROOT


def scratch_root() -> Path:
    """Working scratch. Purged after 30 days without access — never a result."""
    return _env_path("VSC_SCRATCH") or REPO_ROOT


def _under(root: Path, *parts: str) -> Path:
    """Join under `root`, inserting the project name only when `root` is SHARED.

    The shared tiers need a `<Project>/` component; the repo root already *is* the project, so
    adding it there would give `<Project>/<Project>/output`.
    """
    if root == REPO_ROOT:
        return root.joinpath(*parts)
    return root.joinpath(PROJECT_NAME, *parts)


# ---------------------------------------------------------------------------
# output/ — the single root for everything the code generates.
# ---------------------------------------------------------------------------


def outputs_dir() -> Path:
    """Root for live logs, metadata and figures; large outputs use project-tier helpers."""
    # PROJECT LAYER: routed through `resolve_output_path` so $CREDITPFN_OUTPUT_ROOT wins.
    # Without this, `logs_dir()` and `resolve_output_path("output/logs")` could disagree.
    return resolve_output_path("output")


def results_dir(*parts: str) -> Path:
    """Fine-grained results: one row per prediction, per-fold scores, anything large.

    Like compact tables, this uses project storage. Per-row predictions would
    fill $VSC_DATA's 75 GiB, and then every job that writes a log also fails.
    """
    # PROJECT LAYER: `resolve_staging_path` adds the same staging precedence plus the two
    # project env vars, and falls back to the output root when staging is unavailable.
    return resolve_staging_path(Path("output", "results", *parts))


def logs_dir() -> Path:
    """Timestamped run logs. Small, many files -> `$VSC_DATA`, not the inode-poor tier."""
    return outputs_dir() / "logs"


def manifests_dir() -> Path:
    """Per-run manifests: the small CSV/JSON record of what a run did."""
    return outputs_dir() / "manifests"


def consolidated_dir() -> Path:
    """Immutable analysis snapshots: a few compressed files on project storage."""
    return resolve_staging_path("output/consolidated")


def figures_dir(notebook: str | None = None) -> Path:
    """`output/figures/`, or one notebook's own folder — a notebook clears its own before drawing
    and must not be able to reach another's."""
    root = outputs_dir() / "figures"
    return root / notebook if notebook else root


def captions_path() -> Path:
    """The ONE shared captions file for every figure in the project."""
    return figures_dir() / "CAPTIONS.md"


def all_results_path() -> Path:
    """Every notebook's printed text summary, concatenated in notebook order."""
    return outputs_dir() / "All_Results.md"


# ---------------------------------------------------------------------------
# Inputs, weights and the repo's own directories.
# ---------------------------------------------------------------------------


def raw_dir(*parts: str) -> Path:
    """`data/raw/` — never modified, never committed, never deleted by the cleaner."""
    # PROJECT LAYER: datasets follow `paths.data_source` in config/data.yaml, which
    # `apply_data_source_from_cfg` turns into $CREDITPFN_DATA_ROOT.
    return resolve_data_path(Path("data", "raw", *parts))


def processed_dir(*parts: str) -> Path:
    """`data/processed/` — a cache. Regenerable, so it is the first thing to delete."""
    # PROJECT LAYER: see `raw_dir`.
    return resolve_data_path(Path("data", "processed", *parts))


def checkpoints_dir(*parts: str) -> Path:
    """Model weights on project storage; a full clean removes only the trained subtree."""
    return resolve_staging_path(Path("checkpoints", *parts))


def notebooks_dir() -> Path:
    return REPO_ROOT / "notebooks"


# ---------------------------------------------------------------------------
# Making a path usable.
# ---------------------------------------------------------------------------


def ensure(path: Path) -> Path:
    """`mkdir -p` the directory and return it. For a file path, its parent."""
    target = path.parent if path.suffix else path
    target.mkdir(parents=True, exist_ok=True)
    return path


def describe() -> dict[str, str]:
    """Every resolved root, for logging at job start — so a run's output can still be found six
    months later on a tier that has since been reorganised."""
    return {
        "project": PROJECT_NAME,
        "on_vsc": str(on_vsc()),
        "repo_root": str(REPO_ROOT),
        "staging_root": str(staging_root()),
        "data_root": str(data_root()),
        "scratch_root": str(scratch_root()),
        "outputs_dir": str(outputs_dir()),
        "results_dir": str(results_dir()),
        "checkpoints_dir": str(checkpoints_dir()),
    }


# =========================================================================== #
#  PROJECT LAYER — CreditPFN
# =========================================================================== #
#
# Everything above is the template's, unchanged apart from the five helpers marked
# "PROJECT LAYER", which delegate here instead of duplicating the rule.
#
# WHY THIS EXISTS. The template resolves a tier from $VSC_DATA alone. This project
# additionally has to honour three things it cannot drop:
#
#   1. $CREDITPFN_DATA_ROOT / $CREDITPFN_OUTPUT_ROOT — exported by all seven SLURM
#      scripts, and the only way the data stage on wICE and the training stage on
#      Mindwell agree on where the corpus is.
#   2. `paths.data_source` in config/data.yaml (staging | scratch | data), applied by
#      `apply_data_source_from_cfg` at every entry point.
#   3. A writability PROBE before any compute (`resolve_writable_staging_path`):
#      staging was readable but not writable from Mindwell on 04-07-2026 and all 32
#      PD trials died at their first checkpoint save.
#
# The four `resolve_*` functions are the form ~100 call sites use, because a config
# file supplies the relative path (`cfg.paths.processed`, `cfg.checkpoint.trained_dir`).
# They resolve to the same locations as the template's helpers — that is asserted in
# tests/test_paths.py, not assumed.
#
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
#: The staging overrides in precedence order — the template's contract name for
#: them. `_staging_base()` iterates THIS tuple, so the two names above cannot
#: drift from what is actually honoured.
STAGING_ENV_VARS = (STAGING_ROOT_ENV, STAGING_ROOT_ENV_GENERIC)
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
    for var in STAGING_ENV_VARS:
        explicit = os.environ.get(var)
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
    """Resolve a *durable-output* path (logs, manifests, epoch snapshots).

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
    via :func:`resolve_output_path`; consolidate and download finished runs before cleanup.

    Absolute paths are returned unchanged.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    staging = _vsc_staging_root()
    if staging is not None:
        return staging / path
    return resolve_output_path(p)


def resolve_base_checkpoint(p: str | os.PathLike) -> Path:
    """Use the explicitly staged, immutable scratch copy for original weights."""
    cache = os.environ.get("CREDITPFN_BASE_CACHE_ROOT")
    if cache:
        candidate = Path(cache) / "checkpoints" / Path(p).name
        if not candidate.is_file():
            raise FileNotFoundError(f"Original checkpoint missing from prepared scratch cache: {candidate}")
        return candidate
    return resolve_staging_path(p)


def resolve_writable_staging_path(p: str | os.PathLike) -> Path:
    """Probe relative checkpoint directories before training, caching each result.

    The named launcher sets CREDITPFN_REQUIRE_STAGING=1 and refuses DATA fallback.
    Other callers may permit the historical output-root fallback explicitly by
    leaving that guard unset. Absolute paths are caller-selected and returned as-is.
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
    key = str(staging / path)
    if key not in cache:
        probe_dir = staging / path
        try:
            probe_dir.mkdir(parents=True, exist_ok=True)
            import uuid
            probe = probe_dir / (".write_probe_" + uuid.uuid4().hex)
            probe.touch()
            probe.unlink()
            cache[key] = True
        except OSError as exc:
            cache[key] = False
            logger.warning(
                "Staging root %s is NOT writable from this node (%s). "
                "Check the project's permissions/mount before running jobs.",
                staging, exc,
            )
    if cache[key]:
        return staging / path
    if os.environ.get("CREDITPFN_REQUIRE_STAGING") == "1":
        raise PermissionError("Project storage is not writable; refusing to put large weights on DATA")
    fallback = resolve_output_path(p)
    logger.warning("Using the explicitly permitted output-root fallback for %s: %s", p, fallback)
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
# Manifests always resolve via `resolve_output_path`, which uses the independent
# `OUTPUT_ROOT_ENV` ($VSC_DATA/CreditPFN on VSC, repo root locally). They are
# small, NFS-backed, and never move.
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


# --------------------------------------------------------------------------- #
# Template module contract (docs/TEMPLATE.md § Module contract)
