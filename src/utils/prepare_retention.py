"""Download the fixed public retention inputs on a network-enabled login node.

No training or package installation. Existing verified inputs are reused.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

from src.data.retention import panel_config
from src.utils.atomic import write_json
from src.utils.experiment import file_digest
from src.utils.paths import resolve_staging_path


def prepare(panels=("smoke", "research"), root=None):
    root = root or resolve_staging_path("data/retention")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for panel in panels:
        for item in panel_config(panel):
            meta_path = root / (item["id"] + ".json")
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if any(meta.get(k) != value for k, value in item.items()):
                    raise ValueError(f"Panel definition changed: {item['id']}; use a new input identity")
                if file_digest(root / meta["file"]) != meta["sha256"]:
                    raise RuntimeError(f"Retention input changed: {item['id']}")
                records.append(meta)
                continue
            print("Preparing", item["id"], flush=True)
            if item["source"] == "sklearn":
                bunch = getattr(datasets, "load_" + item["name"])(as_frame=True)
                source = "sklearn.datasets.load_" + item["name"]
            else:
                bunch = datasets.fetch_openml(data_id=item["data_id"], as_frame=True,
                    parser="auto", data_home=str(root / "download_cache"), n_retries=2, delay=2)
                if str(bunch.details["version"]) != str(item["version"]):
                    raise ValueError("OpenML version differs from the fixed panel")
                source = f"https://www.openml.org/d/{item['data_id']}"
            X = bunch.data.copy()
            y = pd.Series(bunch.target).reset_index(drop=True)
            X = X.reset_index(drop=True)
            keep = y.notna()
            X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
            cats = [str(c) for c in X if not pd.api.types.is_numeric_dtype(X[c])]
            classes = None
            if item["track"] == "pd":
                encoder = LabelEncoder()
                y = encoder.fit_transform(y.astype(str))
                classes = encoder.classes_.tolist()
                if len(classes) != 2:
                    raise ValueError("The PD retention panel requires binary targets")
            else:
                y = pd.to_numeric(y).to_numpy(dtype=float)
                if not np.isfinite(y).all():
                    raise ValueError("Nonfinite retention target")
            X["target"] = y
            path = root / (item["id"] + ".csv.gz")
            X.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            meta = dict(item, panel=panel, source_url=source, file=path.name, rows=len(X),
                        features=len(X.columns)-1, categorical_columns=cats, classes=classes,
                        sha256=file_digest(path))
            write_json(meta_path, meta, exclusive=True)
            records.append(meta)
    print(json.dumps({"prepared": len(records), "panels": list(panels), "root": str(root)}, indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("smoke", "research", "all"), default="all")
    args = parser.parse_args()
    prepare(("smoke", "research") if args.panel == "all" else (args.panel,))


if __name__ == "__main__":
    main()
