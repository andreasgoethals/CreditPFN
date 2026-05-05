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
    roots = get_roots()
    assert roots["data_root"]   == tmp_path / "s"
    assert roots["output_root"] == tmp_path / "d"
    assert roots["repo_root"]   == REPO_ROOT


def test_is_vsc_environment_only_true_when_vsc_envvars_present(monkeypatch) -> None:
    monkeypatch.delenv("VSC_HOME", raising=False)
    monkeypatch.delenv("VSC_DATA", raising=False)
    assert is_vsc_environment() is False
    monkeypatch.setenv("VSC_DATA", "/data/leuven/some/path")
    assert is_vsc_environment() is True
