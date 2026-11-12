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
    monkeypatch.setattr(cl, "require_experiment0", lambda: None)

    def launcher(path, env):
        submitted.append((path.name, env["STAGES"], env.get("EVAL_KIND")))
        return f"Submitted train array {100 + len(submitted)} on mindwell (x)\n"

    monkeypatch.setattr(cl, "_run_launcher", launcher)
    return submitted, tmp_path


def test_submission_stops_at_the_budget_and_resumes(fake):
    submitted, root = fake
    queued = iter([0, 40, 80, 120, 160])            # E5 has 16 configs of 40 jobs each
    first = cl.submit_experiment("E5", budget=130, queued=lambda: next(queued))
    assert len(first["submitted"]) == 3 and len(submitted) == 3
    again = cl.submit_experiment("E5", budget=1000, queued=lambda: 0)
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


def test_evaluation_experiments_require_every_reference_checkpoint(fake, monkeypatch):
    from src.eval import campaign_tasks as ct
    monkeypatch.setattr(ct, "tasks", lambda *a, **k: [dict(track="pd", model="v2", partition=0)])
    monkeypatch.setattr(ct, "cpt_checkpoint", lambda t, references=None: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(RuntimeError, match="finish their training"):
        cl.submit_experiment("E11", budget=1000, queued=lambda: 0)


def test_evaluation_arrays_restart_indices_and_cover_every_task_once(monkeypatch):
    from src.eval import campaign_tasks as ct
    calls = []
    monkeypatch.setattr(ct, "tasks", lambda kind: [dict(track="pd", model="v3", partition=0, arm=0)] * 1190)
    monkeypatch.setattr(cl, "_exists", lambda resolve: True)
    monkeypatch.setattr(cl, "_sbatch", lambda args, env: calls.append(args) or str(len(calls)))
    state = {"submitted": {}}
    cl._submit_evaluation_tasks("episode_context", state, {}, budget=5000, queued=lambda: 0)
    covered = []
    for args in calls:
        last = int(args[0].rsplit("-", 1)[1])
        offset = int(args[1].split("TASK_OFFSET=")[1])
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
        cl.submit_experiment("E5", budget=1000, queued=lambda: 0)
    state = json.loads((root / "update_size_anchoring.json").read_text())
    assert len(state["submitted"]) == 1 and list(state["uncertain"].values())[0]["job_ids"] == ["123"]
    with pytest.raises(RuntimeError, match="inspect squeue"):
        cl.submit_experiment("E5", budget=1000, queued=lambda: 0)
    assert calls["n"] == 2                                     # nothing launched twice


def test_campaign_submission_requires_a_passing_experiment0(monkeypatch, tmp_path):
    from src.utils import pilot_checks
    monkeypatch.setattr(pilot_checks, "receipts", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="Experiment 0 has not passed"):
        cl.require_experiment0()
    (tmp_path / "summary.json").write_text(json.dumps({"passed": True}))
    cl.require_experiment0()
