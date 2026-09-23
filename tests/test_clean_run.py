# Based on the template, extended for CreditPFN's two storage tiers and trained weights.
# A wipe must leave the tracked
# `.gitkeep` markers and their directories behind — without them a fresh clone has nowhere to write.
"""`src/utils/clean_run.py` — the wipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import clean_run


def test_lists_by_default_and_deletes_only_when_asked(isolated_output, capsys) -> None:
    """A listing you meant as a deletion costs one more command; the reverse costs the run."""
    from src.utils.paths import logs_dir

    logs_dir().mkdir(parents=True, exist_ok=True)
    victim = logs_dir() / "run.log"
    victim.write_text("x" * 100, encoding="utf-8")

    clean_run.main([])
    assert victim.exists(), "the default must not delete anything"
    assert "Nothing was deleted" in capsys.readouterr().out

    clean_run.main(["--clean"])
    assert not victim.exists()


def test_a_wipe_keeps_the_directory_skeleton(isolated_output) -> None:
    """`rmtree` would take `output/figures/.gitkeep` with it, and the next clone would have
    nowhere to write."""
    from src.utils.paths import figures_dir, logs_dir

    for folder in (logs_dir(), figures_dir()):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".gitkeep").write_text("", encoding="utf-8")
    per_notebook = figures_dir("nb")
    per_notebook.mkdir(parents=True, exist_ok=True)
    (per_notebook / "01_x.pdf").write_bytes(b"%PDF")
    (logs_dir() / "run.log").write_text("x", encoding="utf-8")

    removed = clean_run.wipe(clean_run.roots()[0])
    assert removed == 2                                  # the pdf and the log, not the markers
    assert (logs_dir() / ".gitkeep").is_file()
    assert (figures_dir() / ".gitkeep").is_file()
    assert not per_notebook.exists()                     # per-run, no marker, so it goes


def test_gitkeep_is_never_counted(isolated_output) -> None:
    """A directory holding only structure markers is already clean."""
    from src.utils.paths import logs_dir

    logs_dir().mkdir(parents=True, exist_ok=True)
    (logs_dir() / ".gitkeep").write_text("", encoding="utf-8")
    assert clean_run.measure(clean_run.roots()[0]) == (0, 0)


def test_both_storage_tiers_are_cleared_on_the_cluster(isolated_output) -> None:
    """`output/results/` lives on project storage there, so clearing only `$VSC_DATA` would leave
    the largest files behind.

    DEVIATION from the template's version, which asserts exactly two roots: CreditPFN adds
    `checkpoints/trained/` on both tiers and `.sentinels/`, so the count here is five. The
    behaviour being pinned is unchanged — both tiers are covered.
    """
    roots = clean_run.roots()
    assert any("staging" in str(r) for r in roots), "project storage not covered"
    assert any("vsc_data" in str(r).lower() for r in roots), "$VSC_DATA not covered"


def test_the_checkpoint_fallback_location_is_cleared(isolated_output) -> None:
    """Trained weights land on `$VSC_DATA` when staging is unwritable from the compute node.
    A clean that misses that copy leaves the resume-skip check pointing at the previous run's
    weights — which silently reused stale checkpoints in 59 of 64 trials on 10-07-2026."""
    trained = [r for r in clean_run.roots() if r.name == "trained"]
    assert len(trained) == 2, f"expected both tiers' trained/ dirs, got {trained}"
    assert clean_run.roots() == [r for r in clean_run.roots() if clean_run.is_safe(r)], (
        "every default root must pass the safety check"
    )


def test_processed_is_opt_in(isolated_output) -> None:
    """Rebuilding the cache can cost far more than re-running the notebooks, so "clean the last
    run" must not silently throw it away."""
    from src.utils.paths import processed_dir

    assert processed_dir() not in clean_run.roots()
    assert processed_dir() in clean_run.roots(processed=True)


def test_fresh_run_removes_all_project_output_and_trained_weights_but_keeps_inputs(isolated_output):
    from src.utils.paths import resolve_staging_path, resolve_output_path, processed_dir
    victims = [resolve_staging_path(f"output/{name}") for name in (
        "results/pd/old.csv", "consolidated/old/LATEST.json",
        "evaluation_cache/old.json.gz", "archives/old.tar.gz")]
    for resolve in (resolve_staging_path, resolve_output_path):
        victims += [resolve(f"checkpoints/trained/pd/old.ckpt{suffix}")
                    for suffix in ("", ".provenance.json", ".resume.pt")]
    inputs = [resolve_staging_path("checkpoints/original.ckpt"),
              resolve_output_path("checkpoints/original.ckpt"),
              resolve_output_path("archive/run-september-2026/tables/history.csv.gz"),
              resolve_staging_path("data/raw/original.csv"),
              processed_dir() / "pd/processed.csv"]
    for path in victims + inputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep or remove")
    clean_run.main([])
    assert all(p.exists() for p in victims + inputs)
    clean_run.main(["--clean"])
    assert all(not p.exists() for p in victims)
    assert all(p.exists() for p in inputs)


def test_cleanup_preflights_all_trees_before_deleting_anything(isolated_output, monkeypatch):
    from src.utils.paths import resolve_staging_path, outputs_dir
    first = outputs_dir() / "logs/keep.log"
    linked = resolve_staging_path("output/results/linked.csv")
    for path in (first, linked):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep")
    # Simulate a symlink portably; Windows symlink creation can require elevation.
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == linked or original(p))
    with pytest.raises(ValueError, match="linked"):
        clean_run.main(["--clean"])
    assert first.exists() and linked.exists()


def test_train_stage_cleanup_includes_recovery_on_both_tiers(isolated_output):
    from src.utils.paths import resolve_staging_path, resolve_output_path
    victims = []
    for resolve in (resolve_staging_path, resolve_output_path):
        for suffix in ("", ".provenance.json", ".resume.pt"):
            p = resolve(f"checkpoints/trained/pd/trial.ckpt{suffix}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"old")
            victims.append(p)
    clean_run.main(["--stages", "train", "--clean"])
    assert all(not p.exists() for p in victims)


def test_full_cleanup_preserves_only_its_active_log(isolated_output, monkeypatch, capsys):
    from src.utils.paths import logs_dir
    logs_dir().mkdir(parents=True)
    active, old = logs_dir() / "maintenance_1.log", logs_dir() / "maintenance_0.log"
    active.write_text("START\n")
    old.write_text("old run\n")
    monkeypatch.setenv("CREDITPFN_ACTIVE_LOG", str(active))
    clean_run.main(["--clean"])
    assert active.read_text() == "START\n" and not old.exists()
    with active.open("a") as stream:
        stream.write("END exit_code=0\n")
    assert "Preserving active" in capsys.readouterr().out


def test_cleanup_rejects_keep_log_outside_log_directory(isolated_output, monkeypatch):
    from src.utils.paths import outputs_dir
    victim = outputs_dir() / "keep.json"
    victim.parent.mkdir(parents=True)
    victim.write_text("keep")
    monkeypatch.setenv("CREDITPFN_ACTIVE_LOG", str(victim))
    with pytest.raises(SystemExit):
        clean_run.main(["--clean"])
    assert victim.exists()


def test_eval_cleanup_invalidates_moved_figure_metadata_and_notebook_logs(isolated_output):
    from src.utils.paths import outputs_dir
    paths = [outputs_dir() / name for name in (
        "figures/example/01_plot.pdf", "manifests/figures/example.json",
        "logs/notebook_example.log", "manifests/train.csv")]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("record", encoding="utf-8")
    clean_run.main(["--clean", "--stages", "eval"])
    assert all(not p.exists() for p in paths[:3])
    assert paths[3].exists()
