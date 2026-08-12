"""Unit tests for ``src.utils.paths``: env-aware path resolution.

The local-vs-VSC routing is driven by two environment variables:

* ``CREDITPFN_DATA_ROOT``   → governs ``resolve_data_path``
* ``CREDITPFN_OUTPUT_ROOT`` → governs ``resolve_output_path``

We don't actually need a VSC node to test this — pytest's
``monkeypatch.setenv`` simulates the env, and assertions check that
the resolver routes paths to the right roots.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.paths import (
    REPO_ROOT, get_roots, is_staging_available, is_vsc_environment,
    resolve_data_path, resolve_output_path, resolve_staging_path,
)
from src.utils import paths as _paths_mod


@pytest.fixture(autouse=True)
def _clear_autodetect_cache():
    """``_autodetect_data_root`` is ``@functools.cache``-d for production
    speed (the filesystem doesn't change underneath one run). For tests
    that monkey-patch env vars between cases that's a state leak, so
    flush the cache around every test."""
    _paths_mod._autodetect_data_root.cache_clear()
    yield
    _paths_mod._autodetect_data_root.cache_clear()


def test_relative_path_resolves_to_repo_root_when_unset(monkeypatch) -> None:
    """No env vars → both resolvers fall back to the repo root."""
    monkeypatch.delenv("CREDITPFN_DATA_ROOT",   raising=False)
    monkeypatch.delenv("CREDITPFN_OUTPUT_ROOT", raising=False)
    assert resolve_data_path("data/cached") == REPO_ROOT / "data" / "cached"
    assert resolve_output_path("logs") == REPO_ROOT / "logs"


def test_data_root_env_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CREDITPFN_DATA_ROOT", str(tmp_path / "scratch"))
    monkeypatch.delenv("CREDITPFN_OUTPUT_ROOT", raising=False)
    assert resolve_data_path("data/cached") == \
        tmp_path / "scratch" / "data" / "cached"
    # Output resolver is unaffected.
    assert resolve_output_path("logs") == REPO_ROOT / "logs"


def test_output_root_env_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CREDITPFN_DATA_ROOT", raising=False)
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path / "data"))
    assert resolve_output_path("checkpoints/trained") == \
        tmp_path / "data" / "checkpoints" / "trained"
    # Data resolver is unaffected.
    assert resolve_data_path("data/cached") == REPO_ROOT / "data" / "cached"


def test_absolute_path_passes_through(monkeypatch, tmp_path) -> None:
    """An already-absolute path is never rewritten — even with env set."""
    monkeypatch.setenv("CREDITPFN_DATA_ROOT", str(tmp_path / "scratch"))
    abs_path = (tmp_path / "explicit" / "place").resolve()
    assert resolve_data_path(abs_path) == abs_path
    assert resolve_output_path(abs_path) == abs_path


def test_get_roots_reflects_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CREDITPFN_DATA_ROOT",   str(tmp_path / "s"))
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path / "d"))
    monkeypatch.delenv("VSC_HOME", raising=False)
    monkeypatch.delenv("VSC_DATA", raising=False)
    roots = get_roots()
    assert roots["data_root"]   == tmp_path / "s"
    assert roots["output_root"] == tmp_path / "d"
    assert roots["repo_root"]   == REPO_ROOT


# =============================================================================
# Auto-detection of VSC vs local (the precedence ladder)
# =============================================================================
#
#   1. explicit $CREDITPFN_DATA_ROOT     ← slurm scripts set this
#   2. VSC default ($VSC_SCRATCH/CreditPFN)
#                                          ← if $VSC_DATA is set
#                                            (= we're on a VSC node)
#   3. repo root                          ← local laptop fallback


def test_explicit_envvar_beats_vsc_autodetect(monkeypatch, tmp_path) -> None:
    """Even if VSC_DATA is set, an explicit CREDITPFN_DATA_ROOT wins
    (this is the contract slurm scripts rely on)."""
    monkeypatch.setenv("VSC_DATA",            "/data/leuven/.../vsc12345")
    monkeypatch.setenv("VSC_SCRATCH",         "/scratch/leuven/.../vsc12345")
    monkeypatch.setenv("CREDITPFN_DATA_ROOT", str(tmp_path / "explicit"))
    monkeypatch.delenv("CREDITPFN_OUTPUT_ROOT", raising=False)
    assert resolve_data_path("data/cached") == tmp_path / "explicit" / "data" / "cached"


def test_vsc_autodetect_uses_staging_for_data(monkeypatch) -> None:
    """On VSC with no data on disk yet, data paths default to project
    STAGING (datasets are the largest files -> project storage). Here we
    pin the staging base via TABPFN_STAGING_ROOT for a deterministic
    assertion that doesn't depend on the built-in default constant."""
    monkeypatch.delenv("CREDITPFN_DATA_ROOT",    raising=False)
    monkeypatch.delenv("CREDITPFN_OUTPUT_ROOT",  raising=False)
    monkeypatch.delenv("CREDITPFN_STAGING_ROOT", raising=False)
    monkeypatch.setenv("TABPFN_STAGING_ROOT", "/lustre1/project/stg_00211")
    monkeypatch.setenv("VSC_DATA",     "/data/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_SCRATCH",  "/scratch/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_HOME",     "/user/leuven/example/vsc12345")  # for is_vsc_environment
    p = resolve_data_path("data/cached")
    assert str(p).replace("\\", "/").endswith(
        "/lustre1/project/stg_00211/CreditPFN/data/cached"
    )


def test_vsc_autodetect_uses_data_for_output(monkeypatch) -> None:
    """On VSC, durable outputs auto-route to ``$VSC_DATA/CreditPFN``."""
    monkeypatch.delenv("CREDITPFN_DATA_ROOT",   raising=False)
    monkeypatch.delenv("CREDITPFN_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("VSC_DATA",     "/data/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_SCRATCH",  "/scratch/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_HOME",     "/user/leuven/example/vsc12345")
    p = resolve_output_path("checkpoints/trained")
    assert str(p).replace("\\", "/").endswith(
        "/data/leuven/example/vsc12345/CreditPFN/checkpoints/trained"
    )


def test_local_fallback_when_no_vsc_envvars(monkeypatch) -> None:
    """A laptop has none of these envvars → repo root for both."""
    for v in ("CREDITPFN_DATA_ROOT", "CREDITPFN_OUTPUT_ROOT",
              "VSC_DATA", "VSC_SCRATCH", "VSC_HOME"):
        monkeypatch.delenv(v, raising=False)
    assert resolve_data_path("data/cached") == REPO_ROOT / "data" / "cached"
    assert resolve_output_path("logs") == REPO_ROOT / "logs"


def test_partial_vsc_envvars_route_data_to_staging(monkeypatch) -> None:
    """If VSC_HOME marks us as on-VSC but VSC_SCRATCH/VSC_DATA are missing,
    data still routes to project staging (its mount is independent of the
    scratch/data vars), while the durable OUTPUT root degrades to repo root
    (no $VSC_DATA to anchor it). Belt-and-braces — shouldn't happen in
    practice, but must not build a broken scratch path."""
    monkeypatch.delenv("CREDITPFN_DATA_ROOT",    raising=False)
    monkeypatch.delenv("CREDITPFN_OUTPUT_ROOT",  raising=False)
    monkeypatch.delenv("CREDITPFN_STAGING_ROOT", raising=False)
    monkeypatch.delenv("TABPFN_STAGING_ROOT",    raising=False)
    monkeypatch.delenv("VSC_DATA",    raising=False)
    monkeypatch.delenv("VSC_SCRATCH", raising=False)
    monkeypatch.setenv("VSC_HOME", "/user/leuven/example/vsc12345")
    assert resolve_data_path("data/cached") == \
        Path("/lustre1/project/stg_00211/CreditPFN/data/cached")
    assert resolve_output_path("logs") == REPO_ROOT / "logs"


def test_get_roots_on_vsc(monkeypatch) -> None:
    """``get_roots()`` reports the VSC defaults when nothing is overridden:
    data -> project staging, output -> $VSC_DATA, staging -> project staging."""
    for v in ("CREDITPFN_DATA_ROOT", "CREDITPFN_OUTPUT_ROOT", "CREDITPFN_STAGING_ROOT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("TABPFN_STAGING_ROOT", "/lustre1/project/stg_00211")
    monkeypatch.setenv("VSC_DATA",    "/data/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_SCRATCH", "/scratch/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_HOME",    "/user/leuven/example/vsc12345")
    roots = get_roots()
    assert str(roots["data_root"]).replace("\\", "/").endswith(
        "/lustre1/project/stg_00211/CreditPFN"
    )
    assert str(roots["output_root"]).replace("\\", "/").endswith(
        "/data/leuven/example/vsc12345/CreditPFN"
    )
    assert str(roots["staging_root"]).replace("\\", "/").endswith(
        "/lustre1/project/stg_00211/CreditPFN"
    )


def test_builtin_staging_default_used_on_vsc_without_env(monkeypatch) -> None:
    """With NO staging env var but a VSC environment, staging resolves to the
    built-in default base (the project's KU Leuven allocation) + /CreditPFN."""
    for v in ("CREDITPFN_STAGING_ROOT", "TABPFN_STAGING_ROOT",
              "CREDITPFN_DATA_ROOT", "CREDITPFN_OUTPUT_ROOT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("VSC_DATA", "/data/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_HOME", "/user/leuven/example/vsc12345")
    got = str(resolve_staging_path("checkpoints/trained")).replace("\\", "/")
    assert got == "/lustre1/project/stg_00211/CreditPFN/checkpoints/trained"


def test_is_vsc_environment_only_true_when_vsc_envvars_present(monkeypatch) -> None:
    monkeypatch.delenv("VSC_HOME", raising=False)
    monkeypatch.delenv("VSC_DATA", raising=False)
    assert is_vsc_environment() is False
    monkeypatch.setenv("VSC_DATA", "/data/leuven/some/path")
    assert is_vsc_environment() is True


# =============================================================================
# resolve_staging_path — project staging tier
# =============================================================================


def test_resolve_staging_path_falls_back_to_output_root_when_unset(monkeypatch) -> None:
    """Off-VSC with no staging env, staging falls back to the repo root."""
    monkeypatch.delenv("CREDITPFN_STAGING_ROOT", raising=False)
    monkeypatch.delenv("TABPFN_STAGING_ROOT",    raising=False)
    monkeypatch.delenv("CREDITPFN_OUTPUT_ROOT",  raising=False)
    monkeypatch.delenv("VSC_DATA", raising=False)
    monkeypatch.delenv("VSC_HOME", raising=False)
    assert resolve_staging_path("checkpoints/trained") == REPO_ROOT / "checkpoints/trained"


def test_resolve_staging_path_uses_staging_root_when_set(monkeypatch) -> None:
    """When CREDITPFN_STAGING_ROOT is set, staging paths use it."""
    staging = "/staging/leuven/stg_00001/CreditPFN"
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", staging)
    assert resolve_staging_path("checkpoints/trained") == Path(staging) / "checkpoints/trained"
    assert resolve_staging_path("output/results") == Path(staging) / "output/results"


def test_resolve_staging_path_passes_absolute_through(monkeypatch) -> None:
    """Absolute paths are returned unchanged regardless of env vars."""
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", "/staging/leuven/stg_00001/CreditPFN")
    abs_path = Path("/abs/path/to/checkpoint.ckpt")
    assert resolve_staging_path(abs_path) == abs_path


def test_is_staging_available_false_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("CREDITPFN_STAGING_ROOT", raising=False)
    monkeypatch.delenv("TABPFN_STAGING_ROOT",    raising=False)
    monkeypatch.delenv("VSC_DATA", raising=False)
    monkeypatch.delenv("VSC_HOME", raising=False)
    assert is_staging_available() is False


def test_is_staging_available_false_when_path_missing(monkeypatch, tmp_path) -> None:
    missing = str(tmp_path / "does_not_exist")
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", missing)
    assert is_staging_available() is False


def test_is_staging_available_true_when_path_exists(monkeypatch, tmp_path) -> None:
    # Staging base + the /CreditPFN project subdir must exist on disk.
    (tmp_path / "CreditPFN").mkdir()
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(tmp_path))
    assert is_staging_available() is True


def test_get_roots_includes_staging(monkeypatch) -> None:
    """get_roots() exposes staging_root."""
    staging = "/staging/leuven/stg_00001/CreditPFN"
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", staging)
    roots = get_roots()
    assert "staging_root" in roots
    assert roots["staging_root"] == Path(staging)


def test_get_roots_staging_falls_back_to_repo_root_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("CREDITPFN_STAGING_ROOT", raising=False)
    monkeypatch.delenv("TABPFN_STAGING_ROOT",    raising=False)
    monkeypatch.delenv("VSC_DATA", raising=False)
    monkeypatch.delenv("VSC_HOME", raising=False)
    roots = get_roots()
    assert roots["staging_root"] == REPO_ROOT


def test_tabpfn_staging_root_env_is_honoured(monkeypatch) -> None:
    """The generic TABPFN_STAGING_ROOT var (no CREDITPFN_ override) resolves
    staging to <base>/CreditPFN."""
    monkeypatch.delenv("CREDITPFN_STAGING_ROOT", raising=False)
    monkeypatch.setenv("TABPFN_STAGING_ROOT", "/lustre1/project/stg_00211")
    assert resolve_staging_path("checkpoints/trained") == \
        Path("/lustre1/project/stg_00211/CreditPFN/checkpoints/trained")


def test_creditpfn_staging_root_takes_precedence_over_tabpfn(monkeypatch) -> None:
    monkeypatch.setenv("TABPFN_STAGING_ROOT",    "/staging/leuven/stg_99999")
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", "/lustre1/project/stg_00211")
    assert resolve_staging_path("x") == Path("/lustre1/project/stg_00211/CreditPFN/x")


def test_builtin_staging_default_on_vsc(monkeypatch) -> None:
    """On a VSC node with no staging env var set, staging resolves to the
    built-in default allocation."""
    monkeypatch.delenv("CREDITPFN_STAGING_ROOT", raising=False)
    monkeypatch.delenv("TABPFN_STAGING_ROOT",    raising=False)
    monkeypatch.setenv("VSC_HOME", "/user/leuven/example/vsc12345")
    assert resolve_staging_path("checkpoints") == \
        Path("/lustre1/project/stg_00211/CreditPFN/checkpoints")


# =============================================================================
# Auto-detection of *where the raw data actually sits* on VSC
# =============================================================================
#
# Real user upload layouts we've seen in the wild:
#
#   A. $VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/   ← documented canonical
#   B. $VSC_SCRATCH/data/raw/{pd,lgd}/             ← straight-into-scratch
#   C. $VSC_DATA/CreditPFN/data/raw/{pd,lgd}/      ← uploaded with the repo
#
# ``_autodetect_data_root`` probes the three in that priority order and
# returns the first one that has CSVs under ``data/raw/pd/`` or
# ``data/raw/lgd/``.


def _seed_raw_csv(root: Path, track: str = "pd") -> None:
    """Drop a stub CSV under ``root/data/raw/<track>/`` so the autodetect
    probe ``_root_has_data`` sees it."""
    d = root / "data" / "raw" / track
    d.mkdir(parents=True, exist_ok=True)
    (d / "stub.csv").write_text("dummy,header\n1,2\n", encoding="utf-8")


def test_autodetect_prefers_scratch_with_project_subdir(monkeypatch, tmp_path) -> None:
    """Layout A wins over B and C when all three have data."""
    scratch  = tmp_path / "scratch"
    vsc_data = tmp_path / "data"
    _seed_raw_csv(scratch / "CreditPFN")   # A
    _seed_raw_csv(scratch)                 # B
    _seed_raw_csv(vsc_data / "CreditPFN")  # C
    monkeypatch.delenv("CREDITPFN_DATA_ROOT", raising=False)
    monkeypatch.setenv("VSC_SCRATCH", str(scratch))
    monkeypatch.setenv("VSC_DATA",    str(vsc_data))
    monkeypatch.setenv("VSC_HOME",    str(tmp_path / "home"))
    assert resolve_data_path("data/raw") == scratch / "CreditPFN" / "data" / "raw"


def test_autodetect_falls_back_to_scratch_root(monkeypatch, tmp_path) -> None:
    """Layout B: data sits straight in $VSC_SCRATCH (no CreditPFN subdir).
    Autodetect must still find it and route there."""
    scratch  = tmp_path / "scratch"
    vsc_data = tmp_path / "data"
    _seed_raw_csv(scratch)                 # B
    _seed_raw_csv(vsc_data / "CreditPFN")  # C — exists too, but loses to B
    monkeypatch.delenv("CREDITPFN_DATA_ROOT", raising=False)
    monkeypatch.setenv("VSC_SCRATCH", str(scratch))
    monkeypatch.setenv("VSC_DATA",    str(vsc_data))
    monkeypatch.setenv("VSC_HOME",    str(tmp_path / "home"))
    assert resolve_data_path("data/raw") == scratch / "data" / "raw"


def test_autodetect_uses_vsc_data_when_scratch_empty(monkeypatch, tmp_path) -> None:
    """Layout C: scratch was purged (or user uploaded straight to $VSC_DATA);
    autodetect routes data reads to ``$VSC_DATA/CreditPFN``."""
    scratch  = tmp_path / "scratch"
    scratch.mkdir()
    vsc_data = tmp_path / "data"
    _seed_raw_csv(vsc_data / "CreditPFN")  # only C
    monkeypatch.delenv("CREDITPFN_DATA_ROOT", raising=False)
    monkeypatch.setenv("VSC_SCRATCH", str(scratch))
    monkeypatch.setenv("VSC_DATA",    str(vsc_data))
    monkeypatch.setenv("VSC_HOME",    str(tmp_path / "home"))
    assert resolve_data_path("data/raw") == \
        vsc_data / "CreditPFN" / "data" / "raw"


def test_autodetect_detects_lgd_csvs_too(monkeypatch, tmp_path) -> None:
    """``_root_has_data`` probes both ``pd/`` *and* ``lgd/`` — either is
    enough to count a root as populated."""
    scratch  = tmp_path / "scratch"
    vsc_data = tmp_path / "data"
    _seed_raw_csv(scratch, track="lgd")  # only LGD CSVs, no PD
    monkeypatch.delenv("CREDITPFN_DATA_ROOT", raising=False)
    monkeypatch.setenv("VSC_SCRATCH", str(scratch))
    monkeypatch.setenv("VSC_DATA",    str(vsc_data))
    monkeypatch.setenv("VSC_HOME",    str(tmp_path / "home"))
    assert resolve_data_path("data/raw") == scratch / "data" / "raw"


def test_autodetect_falls_back_to_staging_when_nothing_found(monkeypatch, tmp_path) -> None:
    """Fresh checkout, no data on disk anywhere → fall back to project
    STAGING (the canonical home for datasets) so downstream "missing raw
    file" warnings point at the right upload location."""
    scratch  = tmp_path / "scratch"
    vsc_data = tmp_path / "data"
    staging  = tmp_path / "staging"
    scratch.mkdir(); vsc_data.mkdir(); staging.mkdir()  # exist but empty
    monkeypatch.delenv("CREDITPFN_DATA_ROOT", raising=False)
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(staging))
    monkeypatch.setenv("VSC_SCRATCH", str(scratch))
    monkeypatch.setenv("VSC_DATA",    str(vsc_data))
    monkeypatch.setenv("VSC_HOME",    str(tmp_path / "home"))
    assert resolve_data_path("data/cached") == \
        staging / "CreditPFN" / "data" / "cached"


def test_explicit_envvar_wins_over_autodetect(monkeypatch, tmp_path) -> None:
    """Even if a candidate VSC root has data on disk, an explicit
    ``CREDITPFN_DATA_ROOT`` always wins — slurm scripts rely on this."""
    scratch  = tmp_path / "scratch"
    explicit = tmp_path / "explicit"
    _seed_raw_csv(scratch / "CreditPFN")    # autodetect *would* find this
    monkeypatch.setenv("CREDITPFN_DATA_ROOT", str(explicit))
    monkeypatch.setenv("VSC_SCRATCH",         str(scratch))
    monkeypatch.setenv("VSC_DATA",            str(tmp_path / "data"))
    monkeypatch.setenv("VSC_HOME",            str(tmp_path / "home"))
    assert resolve_data_path("data/cached") == explicit / "data" / "cached"


# =============================================================================
# logging_setup: per-task log file naming + setup_logging slurm-awareness
# =============================================================================


def test_make_task_log_path_includes_task_and_timestamp(monkeypatch, tmp_path) -> None:
    """``output/logs/<task>_<YYYYMMDD>_<HHMMSS>.log`` schema, flat inside that directory.

    Under `output/` since 11-08-2026, like everything else the code generates."""
    from src.utils.logging_setup import make_task_log_path
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID",       raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)

    p = make_task_log_path("train_pd")
    assert p.parent == tmp_path / "output" / "logs"
    assert p.name.startswith("train_pd_")
    assert p.suffix == ".log"
    # YYYYMMDD_HHMMSS — 15 chars between "train_pd_" and ".log".
    stamp = p.stem.removeprefix("train_pd_")
    assert len(stamp) == 15
    assert stamp[8] == "_"
    assert stamp[:8].isdigit() and stamp[9:].isdigit()


def test_make_task_log_path_appends_slurm_array_ids(monkeypatch, tmp_path) -> None:
    """Slurm array tasks get unique filenames even if they start at the
    same second."""
    from src.utils.logging_setup import make_task_log_path
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID",   "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID",  "7")
    p = make_task_log_path("eval_pd")
    assert "_j12345_a7.log" in p.name


def test_setup_logging_skips_filehandler_under_slurm(monkeypatch, tmp_path) -> None:
    """Under slurm, bash's `exec > $LOG 2>&1` already routes stdout
    to the log file; adding a Python FileHandler would double-write."""
    import logging as _logging
    from src.utils.logging_setup import setup_logging

    monkeypatch.setenv("SLURM_JOB_ID", "999")
    setup_logging(tmp_path / "ignored.log")
    handlers = _logging.getLogger().handlers
    assert any(isinstance(h, _logging.StreamHandler) for h in handlers)
    assert not any(isinstance(h, _logging.FileHandler) for h in handlers)


def test_setup_logging_uses_filehandler_locally(monkeypatch, tmp_path) -> None:
    """Locally (no slurm), both StreamHandler and FileHandler attach
    so the user sees live output AND the log file is created."""
    import logging as _logging
    from src.utils.logging_setup import setup_logging

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    log_file = tmp_path / "out.log"
    setup_logging(log_file)
    handlers = _logging.getLogger().handlers
    assert any(isinstance(h, _logging.StreamHandler) for h in handlers)
    assert any(isinstance(h, _logging.FileHandler) for h in handlers)
    # Triggering a log call should create the file.
    _logging.getLogger("test").info("hello")
    assert log_file.exists()
