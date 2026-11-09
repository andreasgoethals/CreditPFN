"""Second-campaign report helpers on synthetic evidence (no cluster output needed)."""
import numpy as np
import pandas as pd

from src.visualize import campaign2 as c2


def _trajectory():
    rows = []
    for run, arm, delta in (("r1", 0, 0.01), ("r2", 1, -0.02)):
        for updates, rows_seen, kind in ((0, 0, "trajectory"), (250, 600_000, "trajectory"),
                                         (420, 1_000_512, "rows_milestone"), (500, 1_200_000, "trajectory")):
            rows.append(dict(run=run, slug="episode_size", arm=arm, model="TabPFN v3", track="pd",
                             record_type=kind, successful_updates=updates, processed_rows=rows_seen,
                             test_metric=0.70 + (delta if updates else 0.0)))
    return pd.DataFrame(rows)


def test_update_curves_start_at_zero_and_use_only_scheduled_points():
    curves = c2.trajectory_curves(_trajectory(), "pd")
    assert set(curves.successful_updates) == {0, 250, 500}
    assert np.allclose(curves[curves.successful_updates == 0].change, 0)
    assert np.isclose(curves[(curves.arm == 1) & (curves.successful_updates == 500)].change.iloc[0], -0.02)


def test_row_milestones_are_labelled_by_threshold_and_measured_from_update_zero():
    curves = c2.trajectory_curves(_trajectory(), "pd", x="processed_rows")
    assert list(curves.processed_rows.unique()) == [1_000_000]
    assert np.isclose(curves[curves.arm == 0].change.iloc[0], 0.01)


def test_missing_evidence_gives_empty_frames():
    assert c2.trajectory_curves(pd.DataFrame(), "pd").empty
    assert c2.gain_summary(pd.DataFrame()).empty
    assert c2.pages(None, "n", "c") == []


def test_seed_noise_uses_three_seeds_and_flags_only_consistent_gains(monkeypatch):
    base = dict(model="TabPFN v2", credit=True, original=0.7)

    def row(arm, table, gain):
        return dict(arm=arm, test_dataset_id=table, gain=gain, **base)
    e1 = pd.DataFrame([row(0, "a", 0.010), row(0, "b", 0.004), row(0, "c", 0.003)])
    e2 = pd.DataFrame([row(0, "a", 0.009), row(1, "a", 0.008), row(0, "b", -0.002), row(1, "b", 0.003),
                       row(0, "c", 0.0005), row(1, "c", 0.003)])
    monkeypatch.setattr(c2, "endpoint_gains", lambda slug, track: e1 if slug == "reference" else e2)
    out = c2.seed_noise("pd").set_index("test_dataset_id")
    assert list(out.filter(like="gain_seed").columns) == ["gain_seed42", "gain_seed43", "gain_seed44"]
    assert out.loc["a", "beyond_noise"] and not out.loc["b", "beyond_noise"] and not out.loc["c", "beyond_noise"]
    assert np.isclose(out.loc["a", "seed_range"], 0.002)


def test_gain_correlations_use_credit_tables_only():
    gains = pd.DataFrame(dict(arm=0, model="m", test_dataset_id=list("abcde") + ["openml_1"],
                              credit=[True] * 5 + [False], gain=[0.1, 0.2, 0.3, 0.4, 0.5, -9.0],
                              original=[0.9, 0.8, 0.7, 0.6, 0.5, 0.1]))
    props = pd.DataFrame(dict(test_dataset_id=list("abcde"), rows=[10, 20, 30, 40, 50]))
    out = c2.gain_correlations(gains, props).set_index("property")
    assert np.isclose(out.loc["rows", "spearman"], 1.0) and out.loc["rows", "tables"] == 5


def test_design_table_counts_match_the_design():
    table = c2.design_table()
    assert table.runs.sum() == 812
    assert set(table.experiment) == {f"E{i}" for i in range(1, 13)}


def test_endpoint_gains_read_metric_csvs_and_require_all_five_folds(monkeypatch, tmp_path):
    from src.utils import paths
    monkeypatch.setattr(paths, "results_dir", lambda experiment=None: tmp_path / experiment)
    def rows(name, source, auc, folds, status="OK"):
        return pd.DataFrame(dict(model_name=name, model_source=source, test_dataset_id="0001.t",
                                 fold_idx=folds, roc_auc=auc, status=status))
    ref = tmp_path / "reference" / "PD"
    (ref / "u").mkdir(parents=True)
    (ref / "t").mkdir(parents=True)
    rows("tabpfn-untuned[tabpfn-v2-classifier]", "tabpfn-untuned", 0.70, range(5)).to_csv(ref / "u/r_1.csv", index=False)
    trained = "tabpfn-trained[cpt2_reference_a0_pd_tabpfn-v2-classifier]"
    rows(trained, "tabpfn-trained", 0.72, range(5)).to_csv(ref / "t/r_1.csv", index=False)
    # Predictions beside the metrics have the same key columns and must be ignored.
    rows(trained, "tabpfn-trained", 0.0, range(5)).to_csv(ref / "t/r_1.csv.gz", index=False)
    gains = c2.endpoint_gains("reference", "pd")
    assert len(gains) == 1 and np.isclose(gains.gain.iloc[0], 0.02) and gains.complete.iloc[0]
    # A failed fold (newer file) makes the comparison incomplete and drops it from summaries.
    rows(trained, "tabpfn-trained", np.nan, [4], status="FAIL").to_csv(ref / "t/r_2.csv", index=False)
    pd.concat([rows(trained, "tabpfn-trained", 0.72, range(4))]).to_csv(ref / "t/r_1.csv", index=False)
    gains = c2.endpoint_gains("reference", "pd")
    assert gains.folds.iloc[0] == 4 and not gains.complete.iloc[0]
    summary = c2.gain_summary(gains)
    assert summary.incomplete_tables.iloc[0] == 1 and "mean_gain" not in summary


def test_task_summaries_pair_rows_and_need_all_five_folds():
    rows = []
    for table, folds in (("a", range(5)), ("b", range(3))):           # b misses two folds
        for fold in folds:
            for start in ("original", "cpt"):
                for tuned, auc in ((False, .70), (True, .73)):
                    rows.append(dict(track="pd", model="TabPFN v2", start=start, dataset_id=table,
                                     fold=fold, finetuned=tuned, roc_auc=auc))
    out = c2.finetune_summary(pd.DataFrame(rows)).set_index("start")
    assert (out.tables == 1).all() and np.allclose(out.mean_finetune_gain, 0.03)
    curve = []
    for table, sizes in (("a", (8, 16, 32)), ("b", (8, 16))):
        for size in sizes:
            for fold in range(5):
                curve += [dict(track="pd", model="m", dataset_id=table, fold=fold, context_rows=size,
                               start="original", roc_auc=.6),
                          dict(track="pd", model="m", dataset_id=table, fold=fold, context_rows=size,
                               start="cpt", roc_auc=.6 + (.02 if table == "a" else .0))]
    full = c2.context_gain_curve(pd.DataFrame(curve)).set_index("context_rows")
    assert full.tables.tolist() == [2, 2, 1]                            # population changes at 32
    common = c2.context_gain_curve(pd.DataFrame(curve), common_up_to=32).set_index("context_rows")
    assert common.tables.tolist() == [1, 1, 1] and np.allclose(common["mean"], .02)
