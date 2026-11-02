from copy import deepcopy

import pytest

from src.utils.admission_recovery import dependencies, live_tasks, replacement_plan, translate, unfinished


def evidence(statuses):
    return dict(expected=len(statuses), problems=[], diverged=0,
                trials=[dict(status=s) for s in statuses])


def test_expanded_queue_preserves_array_parent_and_ignores_other_simple_jobs():
    assert live_tasks("123|123_0|RUNNING\n123|123_2|PENDING\n456|456|RUNNING\n") == {("123", 0), ("123", 2)}


def test_replace_only_incomplete_terminal_tasks_not_successful_timeout_artifacts():
    plan = [dict(key="e5_pd_v2_s0", tasks=[0, 1, 2], needs=[], args=["--time=03:00:00", "native.slurm"])]
    accepted = [dict(key=plan[0]["key"], tasks=[0, 1, 2], id="123")]
    report = evidence(["OK", "PENDING", "PENDING"] + ["OK"]*9)
    original = deepcopy(plan)
    repaired = replacement_plan(plan, accepted, {"e5_pd_v2": report}, {("123", 2)},
        {"123_0": "TIMEOUT", "123_1": "FAILED", "123_2": "RUNNING"})
    assert repaired == [dict(plan[0], tasks=[1])]
    assert plan == original


@pytest.mark.parametrize("state", ["", "COMPLETED", "RUNNING", "PENDING", "REQUEUED"])
def test_missing_live_job_requires_confirmed_terminal_failure(state):
    plan = [dict(key="e5_pd_v2_s0", tasks=[0], needs=[], args=[])]
    accepted = [dict(key=plan[0]["key"], tasks=[0], id="123")]
    with pytest.raises(RuntimeError, match="Unconfirmed"):
        replacement_plan(plan, accepted, {"e5_pd_v2": evidence(["PENDING"]+["OK"]*3)}, set(), {"123_0": state})


def test_bad_scientific_evidence_cannot_release_benchmarks():
    report = evidence(["OK"]*4)
    report["problems"] = ["wrong prepared identity"]
    with pytest.raises(RuntimeError, match="audit failed"):
        unfinished(report, 0)
    with pytest.raises(RuntimeError, match="roster"):
        unfinished(evidence(["OK"]*5), 0)


def test_afterok_uses_exact_pending_tasks_and_replacement_arrays():
    item = dict(key="e5_pd_v2_s0")
    report = evidence(["OK", "PENDING", "PENDING"] + ["OK"]*9)
    replacements = [dict(key=item["key"], id="789", tasks=[1])]
    assert dependencies("123", item, report, {("789", 1), ("123", 2)}, replacements) == ["789_1", "123_2"]
    with pytest.raises(RuntimeError, match="not live"):
        dependencies("123", item, report, {("789", 1)}, replacements)
    assert dependencies("purged", item, evidence(["OK"]*12), set(), []) == []


def test_rewritten_command_changes_dependencies_only_and_omits_completed_ones():
    command = ["sbatch", "--parsable", "--array=0,3", "--dependency=afterok:123:456",
               "--time=04:00:00", "--export=ALL,CREDITPFN_CONFIG=immutable.yaml", "scripts/slurm/eval_pd.slurm"]
    original = deepcopy(command)
    actual, journal = translate(command, lambda p: [] if p == "123" else ["789_1"], lambda: set())
    assert actual == [arg if not arg.startswith("--dependency=") else "--dependency=afterok:789_1" for arg in command]
    assert command == original
    assert journal[0]["before"] == ["123", "456"]
    actual, _ = translate(command, lambda p: [], lambda: set())
    assert actual == [arg for arg in command if not arg.startswith("--dependency=")]


@pytest.mark.parametrize("old_dependency", [[], ["--dependency=afterany:123"]])
def test_final_barrier_includes_replacements_even_if_original_ids_are_purged(old_dependency):
    command = ["sbatch", *old_dependency, "scripts/slurm/recovery_cpu.slurm", "dispatch", "repair"]
    actual, journal = translate(command, lambda p: [], lambda: {"789", "456"})
    assert "--dependency=afterany:456:789" in actual
    assert not any(arg == "--dependency=afterany:123" for arg in actual)
    assert journal[-1]["kind"] == "afterany"


def test_unrelated_afterany_dependency_is_preserved():
    command = ["sbatch", "--dependency=afterany:123", "unrelated.slurm"]
    assert translate(command, lambda p: [], lambda: {"789"})[0] == command
