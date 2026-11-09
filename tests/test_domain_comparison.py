"""Table-property comparison of credit and non-credit tables, on synthetic tables."""
import numpy as np
import pandas as pd

from src.data import domain_comparison as dc


def _table(n, rng, *, lgd=False):
    X = pd.DataFrame({"a": rng.lognormal(size=n), "b": rng.integers(0, 5, n).astype(float),
                      "c": rng.choice(["x", "y"], n)})
    if lgd:
        y = np.clip(rng.normal(0.5, 0.6, n), 0, 1)
    else:
        y = (rng.random(n) < 0.1).astype(int)
    return X, y


def test_lgd_target_properties_detect_bounds_and_point_masses():
    rng = np.random.default_rng(0)
    X, y = _table(2000, rng, lgd=True)
    props = dc.table_properties(X, y, ["c"], "lgd")
    assert props["bounded_unit_interval"] and props["share_at_min"] > 0.15 and props["share_at_max"] > 0.15
    assert props["target_bimodality"] > 5 / 9
    assert props["categorical_share"] == 1 / 3


def test_pd_minority_rate_and_feature_properties():
    rng = np.random.default_rng(1)
    X, y = _table(3000, rng)
    props = dc.table_properties(X, y, ["c"], "pd")
    assert abs(props["positive_rate"] - 0.1) < 0.02
    assert 0 <= props["max_abs_spearman"] <= 1 and props["integer_share"] == 0.5


def test_cliffs_delta_and_nearest_neighbours():
    assert dc.cliffs_delta([2, 3, 4], [0, 1]) == 1.0
    assert dc.cliffs_delta([1, 2], [1, 2]) == 0.0
    props = pd.DataFrame({"dataset_id": list("abcd"), "group": ["credit", "credit", "retention", "retention"],
                          **{p: [0.0, 0.1, 5.0, 5.1] for p in dc.TABLE_PROPERTIES}, "positive_rate": [0.1, 0.1, 0.4, 0.4]})
    nearest = dc.nearest_neighbour_agreement(props, "pd")
    assert nearest.same_domain.all()
    emb = dc.embedding(props, "pd")
    assert emb.pc1_explained.iloc[0] > 0.9


def test_normalised_target_and_bimodality_edge_cases():
    assert np.allclose(dc.normalised_target([2, 4, 6]), [0, 0.5, 1])
    assert np.allclose(dc.normalised_target([3, 3]), [0, 0])
    assert np.isnan(dc.bimodality_coefficient([1, 1, 1, 1, 1]))


def _clustered(shift):
    rng = np.random.default_rng(3)
    rows = []
    for group, centre in (("credit", 0.0), ("retention", shift)):
        for i in range(8):
            rows.append(dict(dataset_id=f"{group}{i}", group=group,
                             **{p: rng.normal(0, 1) for p in dc.TABLE_PROPERTIES},
                             positive_rate=centre + rng.normal(0, 0.05)))
    return pd.DataFrame(rows)


def test_distance_test_detects_target_structure_only_where_it_exists():
    frame = _clustered(shift=2.0)
    target = dc.domain_distance_test(frame, "pd", subset="target", permutations=500)
    table = dc.domain_distance_test(frame, "pd", subset="table", permutations=500)
    assert target["permutation_p"] < 0.01 and target["mean_credit_to_noncredit"] > target["mean_within_credit"]
    assert table["permutation_p"] > 0.05


def test_dispersion_ratio_marks_homogeneous_credit_properties():
    frame = _clustered(shift=0.0)
    frame.loc[frame.group == "credit", "features"] = 20.0          # identical in every credit table
    out = dc.dispersion(frame, "pd").set_index("property")
    assert out.loc["features", "credit_to_noncredit_ratio"] == 0.0
