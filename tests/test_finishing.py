"""Paper-finishing checks use analytical/synthetic evidence, never research fitting."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from src.visualize import finishing as finish
from src.utils import inference_review as review


def test_bootstrap_is_reproducible_and_omits_tables_not_folds():
    result = finish.bootstrap_tables([1., 2., 3.])
    assert result == finish.bootstrap_tables([1., 2., 3.])
    assert result["mean"] == 2
    assert result["leave_one_out_low"] == 1.5
    assert result["leave_one_out_high"] == 2.5
    assert result["tables"] == 3
    constant = finish.bootstrap_tables([.1]*8)
    assert constant["ci_low"] == pytest.approx(.1)
    assert constant["ci_high"] == pytest.approx(.1)


@pytest.mark.parametrize("values", [[1], [1, np.nan], [[1, 2], [2, 3]]])
def test_bootstrap_rejects_insufficient_or_invalid_tables(values):
    with pytest.raises(ValueError):
        finish.bootstrap_tables(values)


def test_provenance_does_not_treat_hosting_platform_as_shared_population():
    ledger = finish.source_ledger("pd")
    assert len(ledger) == 17
    assert ledger.dataset.is_unique
    assert ledger.source_group.nunique() == len(ledger)  # known twins are in separate tracks
    assert ledger.loc[ledger.provenance.eq("restricted"), "source_url"].eq("").all()
    assert ledger.independence.eq("not established").all()
    combined = pd.concat([ledger, finish.source_ledger("lgd")])
    assert len(combined[combined.source_group.eq("SBA PD/LGD")]) == 2
    assert len(combined[combined.source_group.eq("LendingClub PD/LGD")]) == 2


def fixture_effects():
    return pd.DataFrame(dict(dataset=["A", "B", "C"], base=["v3"]*3, effect=[.1, -.2, .4],
                             learning_rate=[3e-7]*3, l2sp_lambda=[.003]*3,
                             frozen=[False]*3, sampling=["one_sample"]*3))


def test_uncertainty_selects_fixed_recipe_and_provenance_not_outcomes(monkeypatch):
    frame = fixture_effects()
    extras = frame.assign(learning_rate=.01, effect=999)
    monkeypatch.setattr(finish.cp, "benchmark_effects", lambda *a: pd.concat([frame, extras]))
    monkeypatch.setattr(finish, "source_ledger", lambda *a: pd.DataFrame(dict(
        dataset=["A", "B", "C"], provenance=["documented URL", "restricted", "documented URL"],
        documented_public=[True, False, True])))
    observations, estimates, _ = finish.uncertainty_tables(SimpleNamespace(track="pd"))
    assert len(observations) == 6  # three tables, two metrics
    assert np.allclose(estimates.loc[estimates.scenario.eq("All credit tables"), "mean"], .1)
    assert estimates.loc[estimates.scenario.eq("Documented public sources"), "mean"].eq(.25).all()
    assert estimates.tables.tolist() == [3, 2, 3, 2]
    monkeypatch.setattr(finish.cp, "benchmark_effects", lambda *a: pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError, match="one reference"):
        finish.uncertainty_tables(SimpleNamespace(track="pd"))


def test_boundary_errors_validate_rows_and_retain_empty_strata():
    a = pd.DataFrame(dict(fold_idx=np.repeat(range(5), 3), row_idx=range(15),
                          y_true=[0., .5, 1.]*5, y_pred=[.1, .5, .8]*5))
    b = a.assign(y_pred=[.2, .4, .9]*5)
    data = finish.boundary_errors(a, b)
    assert data.rows.eq(5).all()
    assert data.loc[data.region.eq("Target = 1") & data.state.eq("CPT"), "mean_error"].iloc[0] == pytest.approx(-.1)
    with pytest.raises(ValueError, match="Unmatched"):
        finish.boundary_errors(a, b.assign(row_idx=b.row_idx+1))
    constant = a.assign(y_true=.5)
    data = finish.boundary_errors(constant, constant)
    assert data.loc[data.region.eq("Target = 0"), "rows"].eq(0).all()
    assert data.loc[data.region.eq("Target = 0"), "rmse"].isna().all()


def test_density_score_uses_given_original_space_criterion_and_caps_legacy_values():
    from src.model.tabpfn_models import _tabpfn_regression_neg_nll
    # A known non-unit-width histogram: widths [2, 4], masses [1/4, 3/4].
    # Log density is log mass minus log width, not classification log mass.
    from tabpfn.model.bar_distribution import BarDistribution
    criterion = BarDistribution(torch.tensor([0., 2., 6.]))
    logits = torch.log(torch.tensor([[.25, .75], [.25, .75]]))
    targets = np.array([1., 3.])
    result = _tabpfn_regression_neg_nll(None, None, targets,
                                       output=dict(criterion=criterion, logits=logits))
    expected = (np.log(.25/2)+np.log(.75/4))/2
    assert result == pytest.approx(expected)
    loss = torch.tensor([0., float("inf"), float("-inf"), float("nan"), 150., -150.])
    result = _tabpfn_regression_neg_nll(None, None, np.zeros(6),
        output=dict(criterion=lambda *_: loss, logits=torch.zeros(6, 2)))
    assert result == pytest.approx(-100/6)


def test_review_config_targets_real_public_tables_and_reference_recipe():
    from src.data.preprocessing import DATASET_METADATA
    from src.data.dataset_names import is_proprietary
    cfg = OmegaConf.to_container(OmegaConf.load(review.CONFIG))
    assert all(d in DATASET_METADATA and not is_proprietary(d) for d in cfg["datasets"])
    assert cfg["outer_folds"] == 5
    assert cfg["source_recipe"]["learning_rate"] == 3e-7
    assert cfg["source_recipe"]["successful_updates"] == 10000
    assert cfg["boundary_rows"] == [1023, 1024, 10239, 10240]
    cells = review.review_cells(dict(dataset_id=cfg["datasets"][0], fold=0), cfg, 20000)
    assert len(cells) == 30  # six full-test cells plus 24 small fixed-query probes
    assert len(set(cells)) == len(cells)
    assert len(review.review_cells(dict(dataset_id=cfg["datasets"][1], fold=0), cfg, 20000)) == 6
    assert len(review.review_cells(dict(dataset_id=cfg["datasets"][0], fold=1), cfg, 20000)) == 6


def test_review_trace_observes_and_restores_actual_backend_on_error():
    from tabicl._model import attention
    original = attention.F.scaled_dot_product_attention
    q = torch.randn(1, 1, 3, 4)
    with pytest.raises(RuntimeError, match="deliberate"):
        with review.attention_trace() as trace:
            attention.F.scaled_dot_product_attention(q, q, q)
            assert trace == {"torch_SDPA_dispatch/torch.float32/cpu": 1}
            raise RuntimeError("deliberate")
    assert attention.F.scaled_dot_product_attention is original


def test_review_parity_and_plan_refuse_foreign_state(tmp_path, monkeypatch):
    a = np.array([[.2, .8], [.5, .5]])
    assert review.parity_metrics(a, a.copy(), atol=1e-6, rtol=1e-5)["passed"]
    assert not review.parity_metrics(a, a+.01, atol=1e-6, rtol=1e-5)["passed"]
    monkeypatch.setattr(review, "folder", lambda *_: tmp_path)
    monkeypatch.setattr(review, "code_identity", lambda **_: "code")
    monkeypatch.setattr(review, "environment_versions", lambda **_: {"torch": "version"})
    monkeypatch.setattr(review, "launcher_identity", lambda: "launchers")
    payload = dict(code="code", launchers="launchers", versions={"torch": "version"})
    payload["sha256"] = review.digest_json(payload)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload))
    assert review.read_plan("id") == payload
    monkeypatch.setattr(review, "code_identity", lambda **_: "changed")
    with pytest.raises(ValueError, match="environment changed"):
        review.read_plan("id")
    assert review.read_plan("id", current=False) == payload
    path.write_text(json.dumps(dict(payload, code="changed")))
    with pytest.raises(ValueError, match="checksum"):
        review.read_plan("id", current=False)


def test_review_has_separate_storage_and_no_training_submission():
    assert "experiment0" in str(review.folder("test"))
    assert "experiment0" in str(review.folder("test", project=True))
    for bad in ("", "../experiment1", "a/b", "a b"):
        with pytest.raises(ValueError):
            review.folder(bad)
    job = Path("scripts/slurm/inference_review_gpu.slurm").read_text()
    assert "--gpus-per-node=1" in job and "--array=0-14%2" in job
    assert "train_pipeline" not in job


def test_review_requires_campaign_numerical_versions_and_stages_by_hash(tmp_path, monkeypatch):
    versions = {k: "same" for k in ("tabicl", "torch", "numpy", "scikit-learn", "scipy")}
    review.require_campaign_versions(versions, versions)
    with pytest.raises(ValueError, match="Restore the campaign environment"):
        review.require_campaign_versions(versions, dict(versions, tabicl="new"))
    monkeypatch.setenv("VSC_SCRATCH_NODE", str(tmp_path / "scratch"))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    source = tmp_path / "base.ckpt"
    source.write_bytes(b"weight fixture")
    metadata = {"base_sha256": review.file_digest(source)}
    staged = review.stage_weights({"base": source}, metadata, {})
    assert staged["base"].read_bytes() == source.read_bytes()
    assert staged["base"].is_relative_to(tmp_path / "scratch")
    staged["base"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Node-local"):
        review.stage_weights({"base": source}, metadata, {})


def test_review_checks_gpu_cache_before_submission(tmp_path, monkeypatch):
    from src.utils import stage_inputs
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(tmp_path))
    monkeypatch.delenv("CREDITPFN_INPUT_POINTER", raising=False)
    monkeypatch.setattr(stage_inputs, "resolve", lambda p: tmp_path)
    path = tmp_path / "data/processed/pd/table.sanitized.csv"
    path.parent.mkdir(parents=True)
    path.write_text("synthetic input")
    task = dict(dataset_id="table", data_sha256=review.file_digest(path))
    assert review.validate_gpu_inputs([task]) == str(tmp_path)
    path.write_text("different synthetic input")
    with pytest.raises(ValueError, match="differs"):
        review.validate_gpu_inputs([task])


def journal_fixture(path):
    import sqlite3
    from sklearn.metrics import roc_auc_score, brier_score_loss
    cfg = OmegaConf.to_container(OmegaConf.load(review.CONFIG))
    task = dict(dataset_id=cfg["datasets"][1], fold=0)
    plan = dict(config=cfg, tasks=[task], sha256="fixture")
    arrays = dict(context=np.arange(8), validation=np.arange(8, 10), test=np.arange(10, 14),
                  target=np.array([0, 1, 0, 1]), prediction=np.array([[.9, .1], [.3, .7]]*2))
    blob = io.BytesIO()
    np.savez_compressed(blob, **arrays)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE identity (sha TEXT)")
        db.execute("INSERT INTO identity VALUES ('fixture')")
        db.execute("CREATE TABLE cells (key TEXT PRIMARY KEY, metrics TEXT, predictions BLOB)")
        for size, policy, state, scope in review.review_cells(task, cfg, 8):
            row = dict(scope=scope, context_rows=size, policy=policy, state=state, fold=0,
                       test_rows=4, actual_attention={"torch_SDPA_dispatch/torch.float32/cpu": 1},
                       context_sha256=review.digest_json(arrays["context"].tolist()),
                       test_sha256=review.digest_json(arrays["test"].tolist()),
                       roc_auc=roc_auc_score(arrays["target"], arrays["prediction"][:, 1]),
                       brier_score=brier_score_loss(arrays["target"], arrays["prediction"][:, 1]))
            db.execute("INSERT INTO cells VALUES (?,?,?)",
                       (f"{scope}/{size}/{policy}/{state}", json.dumps(row), blob.getvalue()))
    return plan


def test_journal_requires_all_cells_and_recomputes_saved_metrics(tmp_path):
    import sqlite3
    path = tmp_path / "task.sqlite"
    plan = journal_fixture(path)
    rows, checks = review.journal_summary(path, plan, 0)
    assert len(rows) == 6 and all(row["passed"] for row in checks.values())
    with sqlite3.connect(path) as db:
        key, text = db.execute("SELECT key, metrics FROM cells LIMIT 1").fetchone()
        metrics = dict(json.loads(text), roc_auc=.123)
        db.execute("UPDATE cells SET metrics=? WHERE key=?", (json.dumps(metrics), key))
    with pytest.raises(ValueError, match="disagrees"):
        review.journal_summary(path, plan, 0)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM cells WHERE key=?", (key,))
    with pytest.raises(ValueError, match="Missing or unexpected"):
        review.journal_summary(path, plan, 0)


def test_journal_does_not_accept_different_targets_or_row_overlap(tmp_path):
    import sqlite3
    path = tmp_path / "task.sqlite"
    plan = journal_fixture(path)
    with sqlite3.connect(path) as db:
        key, blob = db.execute("SELECT key, predictions FROM cells LIMIT 1").fetchone()
        with np.load(io.BytesIO(blob)) as saved:
            arrays = dict(saved)
        arrays["validation"] = np.array([10, 11])
        changed = io.BytesIO()
        np.savez_compressed(changed, **arrays)
        db.execute("UPDATE cells SET predictions=? WHERE key=?", (changed.getvalue(), key))
    with pytest.raises(ValueError, match="leakage"):
        review.journal_summary(path, plan, 0)


def test_distribution_figure_and_summary_share_every_value(monkeypatch):
    from src.visualize.reporting import page_text
    data = pd.DataFrame([dict(dataset=f"T{i}", metric=metric, untuned=.4+i/10, trained=.5+i/10)
                         for metric in finish.LGD_METRICS for i in range(3)])
    monkeypatch.setattr(finish, "lgd_distribution_scores", lambda *_: data)
    page = finish.plot_lgd_distribution(SimpleNamespace(track="lgd"))[0]
    assert len(page.figure.axes) == 6
    assert page.tables["Point and distribution scores"].equals(data)
    assert "±100" in page.caption
    assert "T2" in page_text(page)


def test_source_audit_reports_known_overlap_without_claiming_deduplication(tmp_path, monkeypatch):
    from src.visualize import inputs
    from src.utils.experiment import digest_json
    monkeypatch.setattr(inputs, "analysis_location", lambda *a: tmp_path)
    monkeypatch.setattr(finish, "source_ledger", lambda *a: pd.DataFrame(dict(
        dataset=["A", "B"], source_group=["shared", "shared"])))
    spec = dict(datasets={"train": [dict(id="A", sha256="same")],
                          "test": [dict(id="B", sha256="same")]})
    payload = dict(identities={digest_json(spec): spec}, partitions=[
        dict(index=0, train=["A"], test=["B"]), dict(index=1, train=["B"], test=["A"])])
    payload["sha256"] = digest_json(payload)
    path = tmp_path / "plans" / "cpt_main_v5_pd.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload))
    run = SimpleNamespace(track="pd", cfg=SimpleNamespace(run_name="cpt_main_v5"))
    ledger = finish.source_partition_audit(run)
    assert ledger.known_shared_source_in_training.eq(1).all()
    assert ledger.identical_file_in_training.eq(1).all()
    payload["partitions"][0]["train"].append("B")
    payload["sha256"] = digest_json({k: v for k, v in payload.items() if k != "sha256"})
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="test table entered"):
        finish.source_partition_audit(run)


def test_review_figure_requires_current_journal_audit(tmp_path, monkeypatch):
    from src.visualize import inference_review as viz
    monkeypatch.setattr(viz, "analysis_location", lambda *a: tmp_path)
    root = tmp_path / "inference_review" / "fixture"
    root.mkdir(parents=True)
    cfg = OmegaConf.to_container(OmegaConf.load(review.CONFIG))
    plan = dict(tasks=[dict(dataset_id=d, fold=f) for d in cfg["datasets"] for f in range(5)])
    plan["sha256"] = review.digest_json(plan)
    (root / "plan.json").write_text(json.dumps(plan))
    for i, task in enumerate(plan["tasks"]):
        records = [dict(scope="benchmark", policy=p, state=s, roc_auc=.7+i*.001)
                   for p in ("auto", "fp32_no_fa3") for s in ("base", "saved", "cpt")]
        (root / f"task_{i:02d}.json").write_text(json.dumps(
            dict(task=i, plan_sha256=plan["sha256"], passed=True, cells=records)))
    coverage, data, _ = viz.evidence()
    assert viz.plot(coverage, data) == []
    audit = dict(passed=True, plan_sha256="foreign", tasks=[dict(task=i, passed=True) for i in range(15)])
    (root / "audit.json").write_text(json.dumps(audit))
    coverage, data, _ = viz.evidence()
    assert viz.plot(coverage, data) == []
    audit["plan_sha256"] = plan["sha256"]
    (root / "audit.json").write_text(json.dumps(audit))
    coverage, data, _ = viz.evidence()
    assert len(viz.plot(coverage, data)) == 1


def test_review_resume_and_independent_audit_use_saved_predictions(tmp_path, monkeypatch):
    import sys
    from src.eval import dataset_loader
    from src.utils import paths
    from sklearn.metrics import roc_auc_score, brier_score_loss
    cfg = OmegaConf.to_container(OmegaConf.load(review.CONFIG))
    cfg["boundary_rows"] = [8, 16]
    cfg["boundary_query_rows"] = 20
    checkpoints = {}
    for name in ("base", "cpt", "saved"):
        path = tmp_path / f"{name}.ckpt"
        path.write_text(name)
        checkpoints[name] = str(path)
    Path(checkpoints["cpt"]+".provenance.json").write_text("{}")
    data_path = tmp_path / "table.csv"
    data_path.write_text("synthetic input identity")
    task = dict(dataset_id=cfg["datasets"][0], fold=0, data_sha256=review.file_digest(data_path),
                base=checkpoints["base"], cpt=checkpoints["cpt"],
                base_sha256=review.file_digest(Path(checkpoints["base"])),
                cpt_sha256=review.file_digest(Path(checkpoints["cpt"])),
                provenance_sha256=review.file_digest(Path(checkpoints["cpt"]+".provenance.json")))
    plan = dict(config=cfg, tasks=[task], saved=checkpoints["saved"],
                saved_sha256=review.file_digest(Path(checkpoints["saved"])), sha256="fixture")
    monkeypatch.setattr(review, "read_plan", lambda *a, **k: plan)
    monkeypatch.setattr(review, "folder", lambda *a, **k: tmp_path / ("project" if k.get("project") else "data"))
    monkeypatch.setattr(review, "stage_weights", lambda p, *a: p)
    monkeypatch.setattr(paths, "processed_dir", lambda *a: data_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    data = SimpleNamespace(y=np.tile([0, 1], 100))
    monkeypatch.setattr(dataset_loader, "load_processed_dataset", lambda *a: data)
    calls = []

    def infer(data, context, validation, test, path, policy, cfg):
        calls.append((path.name, policy, len(context)))
        p = np.where(data.y[test] == 1, .7, .3)
        buf = io.BytesIO()
        np.savez_compressed(buf, context=context, validation=validation, test=test,
                            target=data.y[test], prediction=np.column_stack([1-p, p]))
        row = dict(context_rows=len(context), test_rows=len(test),
                   context_sha256=review.digest_json(context.tolist()),
                   test_sha256=review.digest_json(test.tolist()),
                   roc_auc=roc_auc_score(data.y[test], p), brier_score=brier_score_loss(data.y[test], p),
                   actual_attention={"torch_SDPA_dispatch/torch.float32/cpu": 1})
        return row, buf.getvalue()

    monkeypatch.setattr(review, "run_cell", infer)
    review.run_task("fixture", 0)
    assert len(calls) == 18
    review.run_task("fixture", 0)
    assert len(calls) == 18  # No completed inference cell is repeated.
    assert review.inspect("fixture")["passed"] is True
    receipt = tmp_path / "data" / "task_00.json"
    text = json.loads(receipt.read_text())
    text["cells"][0]["roc_auc"] = .123
    receipt.write_text(json.dumps(text))
    with pytest.raises(RuntimeError, match="incomplete or failed"):
        review.inspect("fixture")
