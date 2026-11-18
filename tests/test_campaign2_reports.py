"""Second-campaign report helpers on synthetic evidence (no cluster output needed)."""
import numpy as np
import pandas as pd

from src.visualize import campaign2 as c2


def test_missing_evidence_gives_empty_frames():
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
    assert table.runs.sum() == 900
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
