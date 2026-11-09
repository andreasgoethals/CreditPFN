"""Matched non-credit control corpus for the domain-control experiment (E3).

Every credit table gets one non-credit twin from ``config/control_pool.yaml``, chosen by an
optimal one-to-one assignment (Hungarian algorithm) that minimises the shortfall of the twin
against the credit table in rows, features and feature types (categorical vs numeric), with a
small penalty for excess size. Preparation then subsamples each twin to its credit table's row
count (stratified for PD) and to its number of categorical and numeric features, so the control
matches table by table wherever the twin is large enough. In each partition E3 trains on the
twins of exactly that partition's credit training tables (``corpus.training_source_split``):
same number of tables, matched rows and features. The target is deliberately NOT matched: class
balance and the LGD target shape are part of what makes a table a credit table.

``python -m src.data.control_matching --write`` regenerates ``config/domain_control.yaml``
from the processed credit tables and the pool; without ``--write`` it checks that file.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils.paths import REPO_ROOT

POOL = REPO_ROOT / "config" / "control_pool.yaml"
PANEL = REPO_ROOT / "config" / "domain_control.yaml"
#: Columns kept by the credit sanitize pipeline (config/data.yaml sanitize.max_columns).
FEATURE_CAP = 64
EXCESS_WEIGHTS = (0.05, 0.02)   # rows, features: prefer the closest of the large-enough twins
#: Largest twin (rows x features) worth downloading on a login node: OpenML serves ARFF only, and
#: the 745-million-cell KDDCup09-Upselling file alone is 1.8 GB. Higgs (23 M cells) still fits.
MAX_CELLS = 30_000_000


def credit_profile(track: str) -> list[dict]:
    """Rows, features and categorical features of every processed credit table of a track."""
    from src.train.corpus import build_dataset_pool
    out = []
    for ref in sorted(build_dataset_pool(track), key=lambda r: r.dataset_id):
        features = len(pd.read_csv(ref.processed_csv, nrows=1).columns) - 1
        out.append({"twin": ref.dataset_id.split(".")[0], "rows": int(ref.n_rows),
                    "features": int(features), "categorical": len(ref.categorical_columns)})
    return out


def cost(credit: dict, candidate: dict) -> float:
    rows, features = int(candidate["rows"]), int(candidate["features"])
    usable = min(features, FEATURE_CAP)
    categorical = min(int(candidate["categorical"]), features)
    numeric = features - categorical
    short = lambda need, have: float(np.log(need / have)) if have < need else 0.0
    value = short(credit["rows"], rows) + (short(credit["features"], usable) if usable else 9.0)
    value += (max(0, credit["categorical"] - categorical)
              + max(0, credit["features"] - credit["categorical"] - numeric)) / credit["features"]
    value += EXCESS_WEIGHTS[0] * np.log(max(rows, credit["rows"]) / credit["rows"])
    value += EXCESS_WEIGHTS[1] * np.log(max(features, credit["features"]) / credit["features"])
    return float(value)


def match(profile: list[dict], candidates: list[dict]) -> list[dict]:
    """Optimal one-to-one twins: [{twin, rows, features, categorical, data_id, name, cost}]."""
    from scipy.optimize import linear_sum_assignment
    if len(candidates) < len(profile):
        raise ValueError("Fewer control candidates than credit tables")
    matrix = np.array([[cost(c, t) for t in candidates] for c in profile])
    rows, cols = linear_sum_assignment(matrix)
    return [dict(profile[i], data_id=int(candidates[j]["data_id"]), name=str(candidates[j]["name"]),
                 cost=round(float(matrix[i, j]), 4)) for i, j in zip(rows, cols)]


def panel_entries(twins: list[dict], track: str) -> list[dict]:
    return [{"id": f"openml_{t['data_id']}", "track": track, "source": "openml", "data_id": t["data_id"],
             "name": t["name"], "preprocessing": "credit_sanitize",
             "match": {"twin": t["twin"], "rows": t["rows"], "features": t["features"],
                       "categorical": t["categorical"]}} for t in twins]


def candidates(track: str) -> list[dict]:
    pool = OmegaConf.to_container(OmegaConf.load(POOL), resolve=True)["candidates"]
    return [c for c in pool if c["track"] == track and c["rows"] * c["features"] <= MAX_CELLS]


def build(profiles: dict[str, list[dict]]) -> dict:
    """The generated panel for {track: credit profile}."""
    entries, costs = [], {}
    for track in ("pd", "lgd"):
        twins = match(profiles[track], candidates(track))
        entries += panel_entries(twins, track)
        costs[track] = {t["twin"]: t["cost"] for t in twins}
    return {"domain_control": entries, "match_cost": costs}


def recorded_profiles(panel: dict) -> dict[str, list[dict]]:
    """The credit profiles stored in a generated panel (rows/features are counts, not data)."""
    out = {"pd": [], "lgd": []}
    for item in panel["domain_control"]:
        out[item["track"]].append({k: item["match"][k] for k in ("twin", "rows", "features", "categorical")})
    return {track: sorted(rows, key=lambda r: r["twin"]) for track, rows in out.items()}


def load_panel() -> dict:
    return OmegaConf.to_container(OmegaConf.load(PANEL), resolve=True)


def render(panel: dict) -> str:
    lines = ["# GENERATED by src.data.control_matching from config/control_pool.yaml and the processed",
             "# credit tables. Do not edit; regenerate with `python -m src.data.control_matching --write`.",
             "# One non-credit twin per credit table (`match.twin` is the credit registry number);",
             "# preparation subsamples each twin to `match.rows` rows and `match.features` features of",
             "# which `match.categorical` categorical, where the twin has them. Used only as E3 training data.",
             "domain_control:"]
    for item in panel["domain_control"]:
        m = item["match"]
        lines.append(f"  - {{id: {item['id']}, track: {item['track']}, source: openml, data_id: {item['data_id']}, "
                     f"name: {item['name']}, preprocessing: credit_sanitize, match: {{twin: '{m['twin']}', "
                     f"rows: {m['rows']}, features: {m['features']}, categorical: {m['categorical']}}}}}")
    lines.append("match_cost:   # shortfall of each twin against its credit table (0 = fully matched)")
    for track, costs in panel["match_cost"].items():
        lines.append(f"  {track}: {{{', '.join(f'{repr(k)}: {v}' for k, v in sorted(costs.items()))}}}")
    return "\n".join(lines) + "\n"


def apply_match(frame: pd.DataFrame, categorical: list[str], spec: dict, track: str,
                seed: int) -> tuple[pd.DataFrame, list[str]]:
    """Subsample a prepared twin (``target`` column included) to its credit table's shape.

    Rows: a random subsample of ``spec['rows']`` rows when the twin is larger, stratified by
    class for PD so the twin keeps its own class balance. Features: ``spec['categorical']``
    categorical and the remaining numeric columns, chosen at random; a type the twin lacks is
    filled from the other type. Columns keep their original order.
    """
    rng = np.random.default_rng(seed)
    if len(frame) > spec["rows"]:
        if track == "pd":
            fraction = spec["rows"] / len(frame)
            keep = np.concatenate([rng.choice(idx, size=max(1, int(round(len(idx) * fraction))), replace=False)
                                   for idx in (np.flatnonzero(frame.target.to_numpy() == c)
                                               for c in np.unique(frame.target))])
        else:
            keep = rng.choice(len(frame), size=spec["rows"], replace=False)
        frame = frame.iloc[np.sort(keep)].reset_index(drop=True)
    features = [c for c in frame if c != "target"]
    cats = [c for c in features if c in set(categorical)]
    nums = [c for c in features if c not in set(categorical)]
    want = min(spec["features"], len(features))
    n_cat = min(spec["categorical"], len(cats))
    n_num = min(want - n_cat, len(nums))
    n_cat = min(want - n_num, len(cats))
    chosen = set(rng.choice(cats, size=n_cat, replace=False).tolist() if n_cat else [])
    chosen |= set(rng.choice(nums, size=n_num, replace=False).tolist() if n_num else [])
    columns = [c for c in features if c in chosen]
    return frame[columns + ["target"]], [c for c in cats if c in chosen]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    panel = build({track: credit_profile(track) for track in ("pd", "lgd")})
    text = render(panel)
    if args.write:
        PANEL.write_text(text, encoding="utf-8", newline="\n")
        print(text)
        return 0
    current = PANEL.read_text(encoding="utf-8") if PANEL.is_file() else ""
    if current != text:
        print("config/domain_control.yaml is stale; run with --write", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
