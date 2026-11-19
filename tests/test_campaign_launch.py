"""Per-experiment submission stays under Mindwell's queue limit and resumes where it stopped."""
import json

import pytest

from src.utils import campaign_launch as cl


@pytest.fixture
def fake(monkeypatch, tmp_path):
    submitted = []
    monkeypatch.setattr(cl, "_receipt", lambda slug: tmp_path / f"{slug}.json")
    monkeypatch.setattr(cl, "prepare", lambda folder: [])
    monkeypatch.setattr(cl, "_environment", lambda **extra: {"CREDITPFN_RUNTIME_VERSION": "x", **extra})
    monkeypatch.setattr(cl, "config_cost", lambda path: 40)
    monkeypatch.setattr(cl, "require_experiment0", lambda: {"run": "a" * 32, "identity": {"test": True}})

    def launcher(path, env):
        submitted.append((path.name, env["STAGES"], env.get("EVAL_KIND")))
        return f"Submitted train array {100 + len(submitted)} on mindwell (x)\n"

    monkeypatch.setattr(cl, "_run_launcher", launcher)
    return submitted, tmp_path


def test_submission_stops_at_the_budget_and_resumes(fake):
    submitted, root = fake
    queued = iter([0, 40, 80, 120, 160])            # E8 has 16 configs of 40 jobs each
    first = cl.submit_experiment("E8", budget=130, queued=lambda: next(queued))
    assert len(first["submitted"]) == 3 and len(submitted) == 3
    again = cl.submit_experiment("E8", budget=1000, queued=lambda: 0)
    assert len(again["submitted"]) == 18 and len(submitted) == 18      # no config twice
    assert all(stage == "train eval" for _, stage, _ in submitted)
    assert json.loads((root / "update_size_anchoring.json").read_text())["submitted"]


def test_reference_also_queues_classical_baselines_once(fake):
    submitted, _ = fake
    cl.submit_experiment("E1", budget=1000, queued=lambda: 0)
    cl.submit_experiment("E1", budget=1000, queued=lambda: 0)
    classical = [s for s in submitted if s[2] == "classical"]
    assert len(classical) == 2                                           # PD and LGD, once


def test_config_cost_counts_models_partitions_and_evaluation_shards():
    from src.utils.design import config_path
    assert cl.config_cost(config_path("reference", "pd")) == 4 * 4 + cl.EVAL_TASKS * 4
    assert cl.config_cost(config_path("transfer", "lgd", 3)) == 4 * 1          # monitor only, no five-fold
    assert cl.stages(config_path("transfer", "pd", 0)) == "train"
    assert cl.stages(config_path("reference", "pd", 0)) == "train eval"


def test_rejected_pd_target_survives_renaming_labels_and_reordering_rows(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    import pandas as pd
    from src.data.preprocessing import DATASET_METADATA
    from src.utils import paths

    path = tmp_path / "fixture.csv"
    dataset = "0006.synthetic"
    fields = ["loan_amnt", "annual_inc", "int_rate", "dti"]
    frame = pd.DataFrame({f: [1., 2., 3.] for f in fields}).assign(outcome=[0, 1, 0])
    monkeypatch.setitem(DATASET_METADATA, dataset, {"target_column": "outcome"})
    monkeypatch.setattr(paths, "processed_dir", lambda *parts: path)
    hashes = pd.util.hash_pandas_object(frame[fields], index=False).to_numpy(dtype="<u8")
    groups = [hashlib.sha256(np.sort(group).tobytes()).hexdigest()
              for group in (hashes[[0, 2]], hashes[[1]])]
    monkeypatch.setattr(cl, "REJECTED_PD_0006_TARGET", hashlib.sha256(" ".join(sorted(groups)).encode()).hexdigest())
    frame.to_csv(path, index=False)
    assert cl.pd_target_issue(dataset)
    frame["outcome"] = 1 - frame.outcome
    frame.iloc[::-1].to_csv(path, index=False)
    assert cl.pd_target_issue(dataset)
    frame.loc[0, "outcome"] = 0
    frame.to_csv(path, index=False)
    assert cl.pd_target_issue(dataset) is None


def test_evaluation_experiments_require_every_reference_checkpoint(fake, monkeypatch):
    from src.eval import campaign_tasks as ct
    monkeypatch.setattr(ct, "tasks", lambda *a, **k: [dict(track="pd", model="v2", partition=0)])
    monkeypatch.setattr(ct, "cpt_checkpoint", lambda t, references=None: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(RuntimeError, match="finish their training"):
        cl.submit_experiment("E6", budget=1000, queued=lambda: 0)


def test_evaluation_arrays_restart_indices_and_cover_every_task_once(monkeypatch):
    from src.eval import campaign_tasks as ct
    calls = []
    monkeypatch.setattr(ct, "tasks", lambda kind: [dict(track="pd", model="v3", partition=0, arm=0)] * 1190)
    monkeypatch.setattr(ct, "write_tasks_cache", lambda kind, listed: None)
    monkeypatch.setattr(cl, "_exists", lambda resolve: True)
    monkeypatch.setattr(cl, "_sbatch", lambda args, env: calls.append(args) or str(len(calls)))
    state = {"submitted": {}}
    cl._submit_evaluation_tasks("episode_context", state, {}, budget=5000, queued=lambda: 0)
    covered = []
    for args in calls:
        last = int(args[0].rsplit("-", 1)[1])
        assert args[1] == f"--time={cl.TASK_WALLTIME}"
        offset = int(args[2].split("TASK_OFFSET=")[1])
        assert args[0].startswith("--array=0-") and last <= 1000
        covered += [offset + i for i in range(last + 1)]
        assert args[-1] == "episode_context"
    assert covered == list(range(1190))


def test_evaluation_wrapper_accepts_every_task_kind():
    from pathlib import Path
    wrapper = (Path(__file__).resolve().parents[1] / "scripts/slurm/campaign_eval_gpu.slurm").read_text()
    from src.eval.campaign_tasks import KINDS
    assert all(f"{kind})" in wrapper for kind in KINDS) and "TASK_OFFSET" in wrapper


def test_failed_launch_is_recorded_as_uncertain_and_never_resubmitted(fake, monkeypatch):
    submitted, root = fake
    calls = {"n": 0}

    def flaky(path, env):
        calls["n"] += 1
        if calls["n"] == 2:
            raise cl.UncertainSubmission("timed out", "123;mindwell\n")
        return f"Submitted train array {calls['n']} on mindwell (x)\n"
    monkeypatch.setattr(cl, "_run_launcher", flaky)
    with pytest.raises(cl.UncertainSubmission):
        cl.submit_experiment("E8", budget=1000, queued=lambda: 0)
    state = json.loads((root / "update_size_anchoring.json").read_text())
    assert len(state["submitted"]) == 1 and list(state["uncertain"].values())[0]["job_ids"] == ["123"]
    with pytest.raises(RuntimeError, match="inspect squeue"):
        cl.submit_experiment("E8", budget=1000, queued=lambda: 0)
    assert calls["n"] == 2                                     # nothing launched twice


def test_campaign_submission_requires_a_passing_experiment0(monkeypatch, tmp_path):
    from src.utils import pilot_checks, pilot_gate
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(pilot_gate, "identity", lambda **kw: {"inputs": "corrected"})
    monkeypatch.setattr(pilot_checks, "receipts", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="Experiment 0 has not passed"):
        cl.require_experiment0()
    (tmp_path / "summary.json").write_text(json.dumps({"passed": True}))
    with pytest.raises(RuntimeError, match="has not passed"):
        cl.require_experiment0()
    run = pilot_gate.begin(tmp_path, None)
    first = pilot_gate.directory(tmp_path, run)
    from src.utils.atomic import write_json
    write_json(first / "summary.json", dict(run=run, passed=True, identity=pilot_gate.identity()))
    cl.require_experiment0()
    monkeypatch.setattr(pilot_gate, "identity", lambda **kw: {"inputs": "changed"})
    with pytest.raises(RuntimeError, match="changed"):
        cl.require_experiment0()
    monkeypatch.setattr(pilot_gate, "identity", lambda **kw: {"inputs": "corrected"})
    again = pilot_gate.begin(tmp_path, {"evaluation": [0]})
    checks = pilot_gate.current(tmp_path, again)["checks"]
    assert checks["evaluation_00"] == again and checks["evaluation_01"] == run
    with pytest.raises(RuntimeError, match="has not passed"):
        cl.require_experiment0()
    monkeypatch.setenv(pilot_gate.RUN_ENV, run)
    with pytest.raises(RuntimeError, match="superseded"):
        pilot_gate.record_check(tmp_path, "evaluation_00.json", {"passed": True})


def test_experiment0_rerun_submits_only_the_selected_parts(tmp_path, monkeypatch):
    from src.utils import campaign_launch as cl
    from src.utils import pilot_gate
    monkeypatch.setattr(pilot_gate, "identity", lambda **kw: {})
    folder = tmp_path / "E00"
    folder.mkdir()
    for name in ("reference_pd.yaml", "reference_lgd.yaml", "lr1e5_pd.yaml", "lr1e5_lgd.yaml"):
        (folder / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cl, "_environment", lambda **extra: {"CREDITPFN_RUNTIME_VERSION": "x", **extra})
    launched, arrays, ids = [], [], iter(range(100, 200))
    monkeypatch.setattr(cl, "_run_launcher", lambda path, env: launched.append((path.name, env["STAGES"])) or f"Submitted train array {next(ids)} on mindwell\n")
    monkeypatch.setattr(cl, "_sbatch", lambda args, env: arrays.append(args) or str(next(ids)))
    only = {"pilots": ["reference_lgd.yaml", "lr1e5_lgd.yaml"], "classical": ["lgd"],
            "recovery": [4, 5], "evaluation": [4, 5, 6, 7]}
    run = pilot_gate.begin(tmp_path, only)
    monkeypatch.setenv(pilot_gate.RUN_ENV, run)
    preflight = pilot_gate.directory(tmp_path, run) / "preflight.json"
    preflight.write_text('{"passed": false}')
    with pytest.raises(RuntimeError, match="CPU preflight has not passed"):
        cl.submit_experiment0(folder, tmp_path / "rerun.json", only=only)
    assert not launched and not arrays
    preflight.write_text('{"passed": true}')
    state = cl.submit_experiment0(folder, tmp_path / "rerun.json", only=only)
    assert sorted(launched) == [("lr1e5_lgd.yaml", "train"), ("reference_lgd.yaml", "eval"),
                                ("reference_lgd.yaml", "train eval")]
    kinds = [a[-1] for a in arrays]
    assert kinds == ["recovery", "evaluation", "scale", "summary"]
    assert "--array=4,5,6,7" in arrays[2]  # the largest-table check depends on the LGD reference too
    assert "--array=4,5" in arrays[0]                               # just the requested cases
    assert arrays[1][0].startswith("--dependency=afterany:") and "--array=4,5,6,7" in arrays[1]
    assert state["capacity"] is None and state["precision"] is None


# ---------------------------------------------------------------- 18-11-2026 review fixes
def test_an_uncertain_chunk_blocks_resubmission_and_is_saved_before_sbatch(monkeypatch):
    from src.eval import campaign_tasks as ct
    monkeypatch.setattr(ct, "tasks", lambda kind: [dict(track="pd", model="v3", partition=0, arm=0)] * 10)
    monkeypatch.setattr(ct, "write_tasks_cache", lambda kind, listed: None)
    monkeypatch.setattr(cl, "_exists", lambda resolve: True)
    saved = []

    def timeout(args, env):
        raise TimeoutError("sbatch did not answer")
    monkeypatch.setattr(cl, "_sbatch", timeout)
    state = {"submitted": {}}
    with pytest.raises(TimeoutError):
        cl._submit_evaluation_tasks("finetune", state, {}, 5000, lambda: 0, lambda: saved.append(dict(state["submitted"])))
    assert saved == [{"0-9": "uncertain"}]                                # recorded before submitting
    with pytest.raises(RuntimeError, match="may have been submitted"):
        cl._submit_evaluation_tasks("finetune", state, {}, 5000, lambda: 0)


def test_a_second_controller_is_refused_while_the_lock_exists(fake):
    _, root = fake
    (root / "update_size_anchoring.lock").touch()
    with pytest.raises(RuntimeError, match="holds"):
        cl.submit_experiment("E8", budget=1000, queued=lambda: 0)


def test_controller_passes_its_budget_as_the_launcher_ceiling(fake, monkeypatch):
    seen = []
    monkeypatch.setattr(cl, "_run_launcher", lambda path, env: seen.append(env["MAX_QUEUED"])
                        or "Submitted train array 1 on mindwell (x)\n")
    cl.submit_experiment("E8", budget=123, queued=lambda: 0)
    assert set(seen) == {"123"}


def test_rerun_selection_without_pilots_prepares_none_and_empty_is_refused(monkeypatch):
    prepared = []
    monkeypatch.setattr(cl, "prepare", lambda folder, only=None: prepared.append(only) or [])
    cl.main(["prepare", "config/experiments/E00_pilots", "--only", '{"evaluation": [0, 1]}'])
    assert prepared == [[]]
    with pytest.raises(SystemExit, match="non-empty"):
        cl.main(["rerun-experiment0", "config/experiments/E00_pilots", "--only", "{}"])


def test_new_pilot_evidence_is_isolated_on_both_tiers(isolated_output, monkeypatch):
    from src.utils import paths, pilot_checks
    run = "a" * 32
    old_receipts = pilot_checks.receipts()
    old = paths.results_dir(experiment="experiment0")
    checkpoint = "checkpoints/trained/experiment0/pd/model.ckpt"
    old_ckpt = paths.resolve_staging_path(checkpoint)
    monkeypatch.setenv("CREDITPFN_EXPERIMENT", "experiment0")
    monkeypatch.setenv("CREDITPFN_PILOT_SET", run)
    assert paths.results_dir() == old.parent / "evidence" / run / "results"
    assert paths.resolve_staging_path("output CreditPFN/results") == paths.results_dir()
    assert paths.resolve_staging_path("output CreditPFN/experiment0/results") == paths.results_dir()
    assert paths.resolve_staging_path(checkpoint) == old_ckpt.parents[1] / "evidence" / run / "pd/model.ckpt"
    assert run in paths.manifests_dir().parts and run in paths.training_dir().parts
    assert pilot_checks.receipts() == old_receipts
    assert run not in paths.results_dir(experiment="reference").parts
    assert run not in paths.resolve_staging_path("checkpoints/weights.ckpt").parts


def test_slurm_accounting_rejects_failed_or_absent_evidence(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from src.utils import pilot_gate
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=
                        "101_0|COMPLETED|0:0\n101_1|FAILED|1:0\n103_0|COMPLETED|0:0\n"))
    assert pilot_gate.incomplete_jobs({"101", "102", "103_0"}) == [
        "101_1|FAILED|1:0", "102: no accounting record"]
    monkeypatch.setenv("USER", "test")
    monkeypatch.setenv("SLURM_JOB_ID", "88")
    def accounting(action):
        return lambda cmd, **kw: SimpleNamespace(stdout="88\n99\n" if cmd[0] == "squeue" else
            "99|sbatch scripts/slurm/campaign_cpu.slurm " + action)
    for action in ("experiment0", "rerun-experiment0 {}", "summary", ""):
        monkeypatch.setattr(subprocess, "run", accounting(action))
        assert pilot_gate.pending_controllers()
    monkeypatch.setattr(subprocess, "run", accounting("submit E2"))
    assert not pilot_gate.pending_controllers()


def test_pilot_identity_verifies_both_copies_of_the_input(monkeypatch, tmp_path):
    from src.utils import paths, pilot_gate, runtime_validation
    from src.utils.atomic import write_json
    from src.utils.experiment import digest_json, file_digest

    pointer = tmp_path / "ACTIVE_INPUTS.json"
    scratch = tmp_path / "inputs" / "example"
    scratch.mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    for root in (scratch, project):
        (root / "sample").write_bytes(b"data")
    files = {"sample": {"size": 4, "sha256": file_digest(project / "sample")}}
    key = digest_json(files)
    write_json(pointer, {"root": str(scratch), "sha256": key})
    write_json(scratch / "inventory.json", {"files": files, "sha256": key})
    monkeypatch.setenv("CREDITPFN_INPUT_POINTER", str(pointer))
    monkeypatch.setenv("CREDITPFN_RUNTIME_VERSION", "test")
    monkeypatch.delenv("CREDITPFN_USE_SCRATCH", raising=False)
    monkeypatch.setattr(paths, "resolve_staging_path", lambda name: project / name)
    monkeypatch.setattr(runtime_validation, "require", lambda version: {"manifest_sha256": "x", "identity": {"code": "v1"}})
    assert pilot_gate.identity(verify_project=True)["inputs"] == key
    for root, expected in ((project, "PROJECT inputs"), (scratch, "staged scratch input")):
        (root / "sample").write_bytes(b"oops")
        with pytest.raises(RuntimeError, match=expected):
            pilot_gate.identity(verify_project=True)
        (root / "sample").write_bytes(b"data")
