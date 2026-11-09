"""How similar are the credit tables to each other, and how do they differ from non-credit tables?

Every table of the study is described by the same properties, computed on the processed data
the models see: size and shape, feature types and missingness, the shape of the numeric feature
distributions, how strongly single features relate to the target, and above all the target
itself. Credit PD tables are compared with the binary non-credit tables, and credit LGD tables
with the regression non-credit tables, so the comparison never mixes task types. The
non-credit tables are the retention panel (never trained on) and the domain-control corpus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TASKS = {"pd": "classification", "lgd": "regression"}
#: Properties of the table (X) and of the target (y) used in the comparison.
TABLE_PROPERTIES = ("log10_rows", "features", "categorical_share", "missing_share",
                    "median_abs_skew", "heavy_tail_share", "integer_share", "max_abs_spearman",
                    "top5_abs_spearman")
TARGET_PROPERTIES = {
    "pd": ("positive_rate",),
    "lgd": ("bounded_unit_interval", "share_at_min", "share_at_max", "target_skew",
            "target_bimodality", "unique_share"),
}


def _numeric(X: pd.DataFrame) -> pd.DataFrame:
    return X.select_dtypes(include=[np.number]).astype(float)


def table_properties(X: pd.DataFrame, y: np.ndarray, categorical: list, track: str) -> dict:
    """The comparison properties of one table (see TABLE_PROPERTIES / TARGET_PROPERTIES)."""
    y = np.asarray(y, dtype=float)
    numeric = _numeric(X.drop(columns=[c for c in categorical if c in X.columns]))
    skew = numeric.skew(numeric_only=True).abs()
    kurt = numeric.kurt(numeric_only=True)
    integer = numeric.apply(lambda c: bool(np.all(np.isclose(c.dropna(), np.round(c.dropna())))) if c.notna().any() else False)
    rank = numeric.rank().corrwith(pd.Series(y, index=numeric.index).rank()).abs().dropna() if len(numeric.columns) else pd.Series(dtype=float)
    out = dict(rows=len(y), log10_rows=float(np.log10(max(len(y), 1))), features=int(X.shape[1]),
               categorical_share=len(categorical) / max(1, X.shape[1]),
               missing_share=float(X.isna().to_numpy().mean()),
               median_abs_skew=float(skew.median()) if len(skew) else np.nan,
               heavy_tail_share=float((kurt > 10).mean()) if len(kurt) else np.nan,
               integer_share=float(integer.mean()) if len(integer) else np.nan,
               max_abs_spearman=float(rank.max()) if len(rank) else np.nan,
               top5_abs_spearman=float(rank.nlargest(5).mean()) if len(rank) else np.nan)
    if track == "pd":
        values, counts = np.unique(y, return_counts=True)
        out["positive_rate"] = float(counts.min() / counts.sum()) if len(values) == 2 else np.nan
    else:
        low, high = float(np.min(y)), float(np.max(y))
        out.update(bounded_unit_interval=bool(low >= 0 and high <= 1),
                   share_at_min=float(np.mean(np.isclose(y, low))),
                   share_at_max=float(np.mean(np.isclose(y, high))),
                   target_skew=float(pd.Series(y).skew()),
                   target_bimodality=bimodality_coefficient(y),
                   unique_share=float(len(np.unique(y)) / len(y)))
    return out


def bimodality_coefficient(y: np.ndarray) -> float:
    """Sarle's bimodality coefficient; above 5/9 ≈ 0.555 suggests a bimodal or U-shaped target."""
    s = pd.Series(np.asarray(y, dtype=float))
    n = len(s)
    if n < 4 or s.std() == 0:
        return np.nan
    g, k = s.skew(), s.kurt()
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def normalised_target(y: np.ndarray) -> np.ndarray:
    """Target rescaled to [0, 1] (min-max), so shapes can be compared across tables."""
    y = np.asarray(y, dtype=float)
    span = np.max(y) - np.min(y)
    return (y - np.min(y)) / span if span > 0 else np.zeros_like(y)


def load_groups(track: str, *, max_rows: int = 50_000, seed: int = 0) -> list[dict]:
    """Every table of the track: credit corpus, retention panel and (PD) control corpus.

    Large tables are subsampled to ``max_rows`` rows (fixed seed) for the property computation;
    the number of rows reported is the full table size.
    """
    from src.data.retention import load_refs
    from src.eval.dataset_loader import load_processed_dataset
    from src.train.corpus import build_dataset_pool
    tables = []
    sources = [("credit", ref) for ref in build_dataset_pool(track)]
    for panel, group in (("research", "retention"), ("domain_control", "domain_control")):
        try:
            sources += [(group, ref) for ref in load_refs(panel, track)]
        except FileNotFoundError:
            continue        # panel not prepared on this machine: compare what exists
    rng = np.random.default_rng(seed)
    for group, ref in sources:
        if group == "credit":
            data = load_processed_dataset(track, ref.dataset_id)
            X, y, categorical = data.X, np.asarray(data.y), list(data.categorical_columns)
        else:
            frame = pd.read_csv(ref.processed_csv, low_memory=False)
            X, y, categorical = frame.drop(columns="target"), frame["target"].to_numpy(), list(ref.categorical_columns)
        full_rows = len(y)
        if len(y) > max_rows:
            keep = np.sort(rng.choice(len(y), size=max_rows, replace=False))
            X, y = X.iloc[keep].reset_index(drop=True), y[keep]
        tables.append(dict(dataset_id=ref.dataset_id, group=group, X=X, y=y,
                           categorical=categorical, full_rows=full_rows))
    return tables


def property_table(tables: list[dict], track: str) -> pd.DataFrame:
    rows = []
    for t in tables:
        props = table_properties(t["X"], t["y"], t["categorical"], track)
        props.update(rows=t["full_rows"], log10_rows=float(np.log10(t["full_rows"])))
        rows.append(dict(dataset_id=t["dataset_id"], group=t["group"], **props))
    return pd.DataFrame(rows)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) − P(a < b) for two samples: −1..1, 0 = no difference, ±0.47 ≈ large."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())


def group_differences(properties: pd.DataFrame, track: str) -> pd.DataFrame:
    """Per property: credit median, non-credit median, Cliff's delta and the Mann–Whitney p-value."""
    from scipy.stats import mannwhitneyu
    credit = properties[properties.group == "credit"]
    other = properties[properties.group != "credit"]
    rows = []
    for name in (*TABLE_PROPERTIES, *TARGET_PROPERTIES[track]):
        a, b = credit[name].astype(float).to_numpy(), other[name].astype(float).to_numpy()
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        p = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 and np.ptp(np.r_[a, b]) > 0 else np.nan
        rows.append(dict(property=name, credit_median=float(np.median(a)) if len(a) else np.nan,
                         noncredit_median=float(np.median(b)) if len(b) else np.nan,
                         cliffs_delta=cliffs_delta(a, b), mann_whitney_p=float(p)))
    return pd.DataFrame(rows).sort_values("cliffs_delta", key=lambda s: -s.abs()).reset_index(drop=True)


def nearest_neighbour_agreement(properties: pd.DataFrame, track: str) -> pd.DataFrame:
    """Is each table's nearest table (standardised property space) from its own domain?

    A high share for credit tables means credit tables resemble each other more than they
    resemble non-credit tables, which is the premise of domain-specific CPT.
    """
    names = [*TABLE_PROPERTIES, *TARGET_PROPERTIES[track]]
    values = properties[names].astype(float)
    values = values.fillna(values.median())
    z = ((values - values.mean()) / values.std(ddof=0).replace(0, 1)).to_numpy()
    distance = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(distance, np.inf)
    nearest = distance.argmin(1)
    domain = np.where(properties.group.to_numpy() == "credit", "credit", "non-credit")
    frame = pd.DataFrame(dict(dataset_id=properties.dataset_id, domain=domain,
                              nearest=properties.dataset_id.to_numpy()[nearest],
                              nearest_domain=domain[nearest]))
    frame["same_domain"] = frame.domain == frame.nearest_domain
    return frame


def embedding(properties: pd.DataFrame, track: str) -> pd.DataFrame:
    """Two-dimensional PCA of the standardised properties, for plotting."""
    names = [*TABLE_PROPERTIES, *TARGET_PROPERTIES[track]]
    values = properties[names].astype(float)
    values = values.fillna(values.median())
    z = ((values - values.mean()) / values.std(ddof=0).replace(0, 1)).to_numpy()
    u, s, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    coords = u[:, :2] * s[:2]
    explained = (s ** 2 / (s ** 2).sum())[:2]
    return pd.DataFrame(dict(dataset_id=properties.dataset_id, group=properties.group,
                             pc1=coords[:, 0], pc2=coords[:, 1],
                             pc1_explained=explained[0], pc2_explained=explained[1]))


def _names(track: str, subset: str = "all") -> list[str]:
    if subset not in ("all", "table", "target"):
        raise ValueError("subset must be all, table or target")
    return {"all": [*TABLE_PROPERTIES, *TARGET_PROPERTIES[track]], "table": list(TABLE_PROPERTIES),
            "target": list(TARGET_PROPERTIES[track])}[subset]


def _standardised(properties: pd.DataFrame, track: str, subset: str = "all") -> np.ndarray:
    names = _names(track, subset)
    values = properties[names].astype(float)
    values = values.fillna(values.median())
    return ((values - values.mean()) / values.std(ddof=0).replace(0, 1)).to_numpy()


def dispersion(properties: pd.DataFrame, track: str) -> pd.DataFrame:
    """Per property: interquartile range among credit tables and among non-credit tables.

    A ratio well below 1 means credit tables are more alike in that property than non-credit
    tables are: a shared statistical trait of the credit domain, whatever its level.
    """
    credit = properties[properties.group == "credit"]
    other = properties[properties.group != "credit"]
    rows = []
    for name in (*TABLE_PROPERTIES, *TARGET_PROPERTIES[track]):
        a, b = credit[name].astype(float), other[name].astype(float)
        iqr_a = float(a.quantile(0.75) - a.quantile(0.25))
        iqr_b = float(b.quantile(0.75) - b.quantile(0.25))
        rows.append(dict(property=name, credit_iqr=iqr_a, noncredit_iqr=iqr_b,
                         credit_to_noncredit_ratio=iqr_a / iqr_b if iqr_b > 0 else np.nan))
    return pd.DataFrame(rows).sort_values("credit_to_noncredit_ratio").reset_index(drop=True)


def domain_distance_test(properties: pd.DataFrame, track: str, *, subset: str = "all",
                         permutations: int = 5000, seed: int = 0) -> dict:
    """Are credit tables closer to each other than to non-credit tables?

    Statistic: mean credit–non-credit distance minus mean credit–credit distance in standardised
    property space. The p-value permutes the credit/non-credit labels (one-sided, larger = more
    domain structure).
    """
    z = _standardised(properties, track, subset)
    distance = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    labels = (properties.group.to_numpy() == "credit")

    def statistic(mask):
        within = distance[np.ix_(mask, mask)][np.triu_indices(mask.sum(), 1)].mean()
        between = distance[np.ix_(mask, ~mask)].mean()
        return between - within, within, between

    observed, within, between = statistic(labels)
    rng = np.random.default_rng(seed)
    null = np.array([statistic(rng.permutation(labels))[0] for _ in range(permutations)])
    return dict(properties=subset, credit_tables=int(labels.sum()), noncredit_tables=int((~labels).sum()),
                mean_within_credit=float(within), mean_credit_to_noncredit=float(between),
                difference=float(observed), permutation_p=float((1 + (null >= observed).sum()) / (1 + permutations)))
