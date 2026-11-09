"""Reader-facing names, balanced pages and corpus report figures."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.data import dataset_names
from src.visualize import campaign as cp
from src.visualize import corpus, reporting, style


def test_public_dataset_aliases_work_without_a_private_mapping(monkeypatch):
    monkeypatch.setattr(dataset_names, "_DISPLAY", {})
    assert dataset_names.display_name("0015.credit_risk_dataset") == "Credit Risk"
    assert dataset_names.display_name("0016.bondora_peer2peer") == "Bondora Peer-to-Peer"
    assert dataset_names.display_name("0017.SBA_loans_case") == "SBA Loans Case"
    assert dataset_names.display_name("0008.SBA_loans_case") == "SBA Loans Case (LGD)"
    assert dataset_names.display_name("unknown") == "unknown"


def test_public_model_names_are_idempotent_and_do_not_change_file_ids():
    text = "v2 / v2.6 / v3 / tabicl-v2"
    labels = "TabPFN v2 / TabPFN v2.6 / TabPFN v3 / TabICLv2"
    assert style.model_label(text) == labels
    assert style.model_label(labels) == labels
    filename = "cpt2_reference_a0_s0_pd_tabpfn-v3-classifier-v3_default"
    assert style.model_label(filename) == filename
    assert "TabPFN v2" in reporting.table_text(pd.DataFrame({"base": ["v2"]}))


@pytest.mark.parametrize("length,limit", [(17,12), (16,12), (25,12), (5,4), (17,20), (0,12)])
def test_balanced_pages_cover_every_item_once(length, limit):
    pages = list(style.page_slices(length, limit))
    sizes = [stop-start for start, stop in pages]
    assert [i for start, stop in pages for i in range(start,stop)] == list(range(length))
    if sizes:
        assert max(sizes) <= limit and max(sizes)-min(sizes) <= 1


def test_corpus_overview_shows_every_table_once_with_its_partition():
    style.apply()
    data = pd.DataFrame({"track": ["pd"] * 17 + ["lgd"] * 8,
                         "dataset_id": [f"Dataset {i}" for i in range(25)],
                         "rows": np.geomspace(500, 700000, 25), "features": np.arange(25) + 3,
                         "missing": np.linspace(0, .2, 25),
                         "minority_class_ratio": [.2] * 17 + [np.nan] * 8,
                         "target_mean": [np.nan] * 17 + [.4] * 8})
    partitions = pd.DataFrame({"track": data.track, "dataset": data.dataset_id, "fold": np.arange(25) % 4 + 1})
    pages = corpus.plot_corpus_overview(data, partitions)
    assert len(pages) == 1
    rows, features, missing, target = pages[0].figure.axes[:4]
    labels = [t.get_text() for t in rows.get_yticklabels()]
    assert len(labels) == 25 and labels[0] == "Dataset 16 · P1" and all("· P" in t for t in labels)
    assert rows.get_xscale() == "log" and features.get_xscale() == "log"
    assert missing.get_xlim()[0] == 0 and target.get_xlim() == (0, 1)
    assert len(target.patches) == 25
    table = pages[0].tables["Corpus overview"]
    assert table.target.notna().all() and len(table) == 25


def test_processed_target_statistics_use_current_rows_not_stale_manifest(monkeypatch):
    registered = pd.DataFrame([dict(track="lgd", dataset_id="Table", target_column="y",
                                    target_mean=.7049, target_std=.4182),
                               dict(track="lgd", dataset_id="Missing", target_column="y",
                                    target_mean=.9, target_std=.2)])
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [.1, .7, 1.]})
    monkeypatch.setattr(corpus.exploration, "corpus_summary_table", lambda: registered.copy())
    def load(track, dataset):
        if dataset == "Missing":
            raise FileNotFoundError(dataset)
        return frame
    monkeypatch.setattr(corpus.exploration, "load_sanitized_dataset", load)
    result = corpus.processed_inventory().set_index("dataset_id")
    assert result.loc["Table", "target_mean"] == pytest.approx(frame.y.mean())
    assert result.loc["Table", "target_std"] == pytest.approx(frame.y.std())
    assert pd.isna(result.loc["Missing", "target_mean"])


def test_delivery_checks_recognize_actual_sanitized_filenames(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus.exploration, "_resolve_paths", lambda: {"processed": tmp_path})
    for track, name, target in [("pd", "0008.german", "target"), ("lgd", "0008.SBA_loans_case", "lgd")]:
        (tmp_path/track).mkdir()
        (tmp_path/track/(name+".sanitized.csv")).write_text("feature,"+target+"\n", encoding="utf8")
    data = pd.DataFrame(dict(track=["pd","lgd"], rows=[1000,2000],
        dataset_id=[dataset_names.display_name("0008.german"), dataset_names.display_name("0008.SBA_loans_case")],
        target_in_raw=[False,False]))
    checks = corpus.delivery_checks(data)
    assert checks.target_in_processed.all()
    assert checks.status.eq("Expected preprocessing").all()
    (tmp_path/"pd"/"0008.german.sanitized.csv").unlink()
    checks = corpus.delivery_checks(data)
    assert checks.loc[0,"status"] == "Inspect delivery"


def test_track_scopes_export_name_title_and_caption(monkeypatch):
    import IPython.display
    monkeypatch.setattr(IPython.display,"display",lambda *args: None)
    saved=[]
    sink=SimpleNamespace(save=lambda figure,name,caption,**metadata: saved.append((name,caption)))
    for track in ("pd","lgd"):
        fig,ax=plt.subplots()
        fig.suptitle("Optimizer check")
        ax.plot([1,2],[3,4])
        cp.show(sink,[cp.Page("same_name",fig,"Recorded values.")],track=track)
        assert fig._suptitle.get_text().startswith(track.upper())
    assert [name for name,_ in saved] == ["pd_same_name","lgd_same_name"]
    assert [caption for _,caption in saved] == ["PD. Recorded values.","LGD. Recorded values."]


def test_exposure_merges_equal_markers_and_labels_step_share_ratio():
    data = pd.DataFrame(dict(track=["pd", "pd"], dataset_id=["A", "B"], rows=[100, 90000]))
    table = corpus.planned_exposure(data, row_cap=10000)
    assert table.full_pass_share_ratio.tolist() == pytest.approx([.2, 1.8])
    page = corpus.plot_exposure(table)[0]
    ax = page.figure.axes[0]
    assert len(ax.get_legend_handles_labels()[1]) == 2
    assert {text.get_text() for text in ax.texts} == {"0.20×", "1.80×"}
    assert len(ax.collections) == 2  # No connector collection.
    assert {text.get_color() for text in ax.texts} == {style.color("improvement"), style.color("deterioration")}


def test_lgd_histograms_include_dataset_mean_and_median(monkeypatch):
    import src.data.preprocessing as prep
    monkeypatch.setattr(prep,"DATASET_METADATA", {"synthetic":{"track":"lgd","target_column":"y"}})
    monkeypatch.setattr(corpus.exploration,"load_sanitized_dataset", lambda *args: pd.DataFrame({"y":[0.,0.,.5,1.]}))
    page = corpus.plot_lgd_distributions()[0]
    ax = page.figure.axes[0]
    assert [line.get_xdata()[0] for line in ax.lines] == [.375,.25]
    assert all(line.get_linestyle() == ":" for line in ax.lines)
    assert page.tables["LGD distribution statistics"].iloc[0]["mean"] == .375
