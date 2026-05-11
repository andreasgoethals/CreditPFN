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

from src.utils.paths import (
    REPO_ROOT, get_roots, is_vsc_environment,
    resolve_data_path, resolve_output_path,
)


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


def test_vsc_autodetect_uses_scratch_for_data(monkeypatch) -> None:
    """On VSC (VSC_DATA + VSC_SCRATCH set, no CREDITPFN_*), data paths
    auto-route to ``$VSC_SCRATCH/CreditPFN``."""
    monkeypatch.delenv("CREDITPFN_DATA_ROOT",   raising=False)
    monkeypatch.delenv("CREDITPFN_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("VSC_DATA",     "/data/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_SCRATCH",  "/scratch/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_HOME",     "/user/leuven/example/vsc12345")  # for is_vsc_environment
    p = resolve_data_path("data/cached")
    assert str(p).replace("\\", "/").endswith(
        "/scratch/leuven/example/vsc12345/CreditPFN/data/cached"
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


def test_partial_vsc_envvars_dont_trigger_autodetect(monkeypatch) -> None:
    """If VSC_HOME is set but VSC_SCRATCH/VSC_DATA are missing,
    auto-detection silently degrades to repo root rather than building
    a broken path. Belt-and-braces — shouldn't happen in practice."""
    monkeypatch.delenv("CREDITPFN_DATA_ROOT",   raising=False)
    monkeypatch.delenv("CREDITPFN_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("VSC_DATA",    raising=False)
    monkeypatch.delenv("VSC_SCRATCH", raising=False)
    monkeypatch.setenv("VSC_HOME", "/user/leuven/example/vsc12345")
    assert resolve_data_path("data/cached") == REPO_ROOT / "data" / "cached"
    assert resolve_output_path("logs") == REPO_ROOT / "logs"


def test_get_roots_on_vsc(monkeypatch) -> None:
    """``get_roots()`` reports the VSC defaults when nothing is overridden."""
    for v in ("CREDITPFN_DATA_ROOT", "CREDITPFN_OUTPUT_ROOT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("VSC_DATA",    "/data/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_SCRATCH", "/scratch/leuven/example/vsc12345")
    monkeypatch.setenv("VSC_HOME",    "/user/leuven/example/vsc12345")
    roots = get_roots()
    assert str(roots["data_root"]).replace("\\", "/").endswith(
        "/scratch/leuven/example/vsc12345/CreditPFN"
    )
    assert str(roots["output_root"]).replace("\\", "/").endswith(
        "/data/leuven/example/vsc12345/CreditPFN"
    )


def test_is_vsc_environment_only_true_when_vsc_envvars_present(monkeypatch) -> None:
    monkeypatch.delenv("VSC_HOME", raising=False)
    monkeypatch.delenv("VSC_DATA", raising=False)
    assert is_vsc_environment() is False
    monkeypatch.setenv("VSC_DATA", "/data/leuven/some/path")
    assert is_vsc_environment() is True


# =============================================================================
# run_log: per-task log file naming + setup_logging slurm-awareness
# =============================================================================


def test_make_task_log_path_includes_task_and_timestamp(monkeypatch, tmp_path) -> None:
    """``logs/<task>_<YYYYMMDD>_<HHMMSS>.log`` schema, lands under
    ``$CREDITPFN_OUTPUT_ROOT/logs/`` (flat, not in a subdir)."""
    from src.utils.run_log import make_task_log_path
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID",       raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)

    p = make_task_log_path("train_pd")
    assert p.parent == tmp_path / "logs"
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
    from src.utils.run_log import make_task_log_path
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID",   "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID",  "7")
    p = make_task_log_path("eval_pd")
    assert "_j12345_a7.log" in p.name


def test_setup_logging_skips_filehandler_under_slurm(monkeypatch, tmp_path) -> None:
    """Under slurm, bash's `exec > $LOG 2>&1` already routes stdout
    to the log file; adding a Python FileHandler would double-write."""
    import logging as _logging
    from src.utils.run_log import setup_logging

    monkeypatch.setenv("SLURM_JOB_ID", "999")
    setup_logging(tmp_path / "ignored.log")
    handlers = _logging.getLogger().handlers
    assert any(isinstance(h, _logging.StreamHandler) for h in handlers)
    assert not any(isinstance(h, _logging.FileHandler) for h in handlers)


def test_setup_logging_uses_filehandler_locally(monkeypatch, tmp_path) -> None:
    """Locally (no slurm), both StreamHandler and FileHandler attach
    so the user sees live output AND the log file is created."""
    import logging as _logging
    from src.utils.run_log import setup_logging

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    log_file = tmp_path / "out.log"
    setup_logging(log_file)
    handlers = _logging.getLogger().handlers
    assert any(isinstance(h, _logging.StreamHandler) for h in handlers)
    assert any(isinstance(h, _logging.FileHandler) for h in handlers)
    # Triggering a log call should create the file.
    _logging.getLogger("test").info("hello")
    assert log_file.exists()
