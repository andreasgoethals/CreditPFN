"""E6/E5 task lists cover every held-out table once per model and fold, with no leakage."""
import pytest

from src.eval import campaign_tasks as ct


@pytest.fixture(scope="module")
def all_tasks():
    from src.train.corpus import build_dataset_pool
    if not build_dataset_pool("pd"):
        pytest.skip("processed data unavailable")
    return ct.tasks()


def test_every_table_model_and_fold_once_per_source(all_tasks):
    """E6 scores the E1 (credit) and the E4 (non-credit) checkpoints; the finetune list has E1 only."""
    assert len(all_tasks) == 2 * (17 + 8) * 4 * 5
    keys = {(t["source"], t["track"], t["model"], t["dataset_id"], t["fold"]) for t in all_tasks}
    assert len(keys) == len(all_tasks)
    assert {t["source"] for t in all_tasks} == {"reference", "domain_control"}
    assert {t["source"] for t in ct.tasks(kind="finetune")} == {"reference"}
    reference = [{k: v for k, v in t.items() if k != "source"} for t in all_tasks if t["source"] == "reference"]
    control = [{k: v for k, v in t.items() if k != "source"} for t in all_tasks if t["source"] == "domain_control"]
    assert reference == control     # same tables, partitions, models and folds


def test_each_table_belongs_to_exactly_one_partition(all_tasks):
    owner = {}
    for t in all_tasks:
        owner.setdefault((t["track"], t["dataset_id"]), set()).add(t["partition"])
    assert all(len(p) == 1 for p in owner.values())


def test_cpt_checkpoint_never_saw_the_table(all_tasks):
    """The E1 and E4 runs of the task's partition hold the table out of their training corpus."""
    from src.train.config import load_train_config
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    for source in ("reference", "domain_control"):
        for track in ("pd", "lgd"):
            cfg = load_train_config(config_path=str(ct._references(None, source)[track]))
            for partition in range(4):
                split = split_from_cfg(apply_split_index(cfg.copy(), partition))
                trained = {r.dataset_id for r in split.train}
                held = {t["dataset_id"] for t in all_tasks if t["track"] == track and t["partition"] == partition
                        and t["source"] == source}
                assert held and not (held & trained)


def test_missing_e1_checkpoint_is_refused(all_tasks):
    with pytest.raises(FileNotFoundError):
        ct.cpt_checkpoint(all_tasks[0])


def test_task_receipt_depends_on_target_bytes_and_code(tmp_path, monkeypatch):
    from src.utils import experiment, paths
    table, checkpoint = tmp_path / "data", tmp_path / "weights"
    table.write_bytes(b"first")
    checkpoint.write_bytes(b"same weights")
    monkeypatch.setattr(paths, "processed_dir", lambda *args: table)
    monkeypatch.setattr(experiment, "code_identity", lambda **kw: "code-1")
    task = dict(track="pd", dataset_id="example", fold=0)
    def identity():
        return ct.task_identity("context", task, {}, None, False, {"cpt": checkpoint})
    a = identity()
    table.write_bytes(b"other table")
    b = identity()
    assert a["dataset_sha256"] != b["dataset_sha256"] and a["checkpoints"] == b["checkpoints"]
    monkeypatch.setattr(experiment, "code_identity", lambda **kw: "code-2")
    assert b != identity()


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        ct.run("benchmark", 0)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["context", "finetune"])
def test_quick_task_runs_end_to_end_and_skips_when_complete(all_tasks, kind, tmp_path, monkeypatch):
    """The Experiment 0 path: real runner, smallest PD table, CPT stand-in = original weights."""
    import pandas as pd

    from src.utils import paths
    from src.utils.design import BASES
    monkeypatch.setattr(paths, "results_dir", lambda experiment=None: tmp_path / str(experiment))
    monkeypatch.setattr(ct, "cpt_checkpoint", lambda task, references=None:
                        paths.resolve_base_checkpoint(BASES[task["track"]][task["model"]]))
    monkeypatch.setitem(ct.INFERENCE_VIEWS, "tabpfn", 2)
    index, task = min(((i, t) for i, t in enumerate(all_tasks)
                       if t["track"] == "pd" and t["model"] == "v2" and t["fold"] == 0),
                      key=lambda c: c[1]["n_rows"])
    path = ct.run(kind, index, slug="experiment0", quick=True, device="cpu")
    frame = pd.read_csv(path)
    assert set(frame.start) == {"original", "cpt"} and frame.roc_auc.notna().all()
    if kind == "context":
        assert {32, 512} <= set(frame.context_rows)
    else:   # paired: each start with and without fine-tuning on identical rows
        assert set(zip(frame.start, frame.finetuned)) == {(s, f) for s in ("original", "cpt") for f in (False, True)}
        assert frame.groupby("start").context_rows.nunique().eq(1).all()
    import json

    import numpy as np
    stored = np.load(path.with_name(path.name.replace(".csv.gz", ".predictions.npz")))
    assert {f"{i}__prediction" for i in range(len(frame))} <= set(stored.files)
    assert all(len(stored[f"{i}__prediction"]) == frame.test_rows.iloc[i] for i in range(len(frame)))
    assert ct.run(kind, index, slug="experiment0", quick=True, device="cpu") == path
    receipt = path.parent / "completed" / path.name.replace(".csv.gz", ".json")
    record = json.loads(receipt.read_text())
    record["identity"]["quick"] = False                    # a result from another protocol
    receipt.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="different identity"):
        ct.run(kind, index, slug="experiment0", quick=True, device="cpu")


def test_matched_context_tasks_cover_every_episode_arm_and_its_models(all_tasks):
    tasks = ct.tasks(kind="episode_context")
    per_arm = {}
    for t in tasks:
        per_arm.setdefault(t["arm"], set()).add(t["model"])
    assert per_arm[3] == {"v3", "tabicl-v2"} and all(len(per_arm[a]) == 4 for a in (0, 1, 2, 4))
    assert all(t["track"] == "pd" for t in tasks)
    from src.utils.design import load_design
    rules = ct.policy("v3", load_design(), kind="episode_context")
    assert rules["context_rows"] == [2048, 8192] and rules["include_full_available"]


def test_e5_rate_is_chosen_on_validation_auc_for_pd_and_rmse_for_lgd():
    """PD: the rate with the highest validation AUC wins, even when another has lower log loss."""
    import numpy as np
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    ranks_well = np.array([0.40, 0.41, 0.42, 0.45, 0.46, 0.43, 0.47, 0.44])   # AUC 1, poorly calibrated
    calibrated = np.array([0.10, 0.20, 0.60, 0.70, 0.90, 0.10, 0.40, 0.20])   # lower log loss, AUC < 1
    proba = lambda p: np.column_stack([1 - p, p])                              # noqa: E731
    assert ct.selection_metric("classification", y) == "validation_roc_auc"
    scores = {1e-6: ct.validation_score("classification", y, proba(ranks_well)),
              1e-5: ct.validation_score("classification", y, proba(calibrated))}
    assert scores[1e-6] == -1.0 and min(scores, key=lambda r: (scores[r], r)) == 1e-6
    one_class = np.zeros(4, dtype=int)
    assert ct.selection_metric("classification", one_class) == "validation_log_loss"
    assert ct.validation_score("classification", one_class, proba(np.full(4, 0.1))) > 0
    assert ct.selection_metric("regression", y) == "validation_rmse"
    assert ct.validation_score("regression", np.array([0.0, 1.0]), np.array([0.0, 0.0])) == np.sqrt(0.5)
