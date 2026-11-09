"""Download the fixed public retention inputs on a network-enabled login node.

No training or package installation. Existing verified inputs are reused. A prepared input
whose recorded definition differs from config/retention.yaml stops preparation; with
``--supersede-changed`` its files are moved (never deleted) to ``superseded/<UTC time>/``
and the table is prepared again.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

from src.data.retention import panel_config
from src.utils.atomic import write_json
from src.utils.experiment import file_digest
from src.utils.paths import REPO_ROOT, resolve_staging_path


def sanitize_frame(X: pd.DataFrame, y, item: dict, cats: list[str]):
    """Apply the credit tables' sanitize pipeline, without the credit LGD [0, 1] target clip."""
    from src.data.sanitize import sanitize_dataset
    cfg = OmegaConf.load(REPO_ROOT / "config" / "data.yaml")
    cfg.sanitize.lgd_target_clip.enabled = False
    if "match" in item:
        # E3 twins keep every clean column here: control_matching.apply_match then selects the
        # credit table's number of categorical and numeric features (never more than the cap).
        cfg.sanitize.max_columns = 10**9
    nums = [c for c in X if c not in cats]
    frame = X.copy()
    frame["target"] = y
    task = "classification" if item["track"] == "pd" else "regression"
    frame, log = sanitize_dataset(frame, item["id"], cfg=cfg, manifest_row={
        "target_column": "target", "task_type": task, "track": item["track"],
        "categorical_columns": ";".join(cats), "numerical_columns": ";".join(nums)})
    kept = [c for c in log["surviving_categorical_columns"] if c in frame]
    return frame.reset_index(drop=True), kept, {
        "max_columns": int(cfg.sanitize.max_columns), "source_features": len(X.columns)}


def _supersede(root, item_id: str, meta: dict, stamp: str):
    target = root / "superseded" / stamp
    target.mkdir(parents=True, exist_ok=True)
    for name in (item_id + ".json", meta.get("file")):
        if name and (root / name).is_file():
            shutil.move(str(root / name), str(target / name))
    print("Superseded", item_id, "->", target, flush=True)


PANELS = ("smoke", "research", "domain_control")
#: Waits (seconds) before each retry of an OpenML download. The server intermittently returns
#: 504 gateway timeouts and truncated large files (failed MD5) on 05-11-2026.
DOWNLOAD_BACKOFF = (30, 60, 120, 240)


def fetch_openml(data_id: int, cache, *, backoff=DOWNLOAD_BACKOFF, sleep=None):
    """``sklearn.datasets.fetch_openml`` with retries for transient server and transfer errors."""
    import time
    import urllib.error

    from sklearn.datasets._openml import OpenMLError
    sleep = sleep or time.sleep
    for attempt, wait in enumerate((*backoff, None)):
        try:
            return datasets.fetch_openml(data_id=data_id, as_frame=True, parser="auto",
                                         data_home=str(cache), n_retries=2, delay=2)
        except (urllib.error.URLError, OSError, OpenMLError, ValueError) as exc:
            if wait is None:
                raise
            print(f"OpenML {data_id}: {type(exc).__name__}: {str(exc)[:120]}; retry {attempt + 1} "
                  f"in {wait} s", flush=True)
            sleep(wait)


def prepare(panels=PANELS, root=None, *, supersede_changed=False):
    root = root or resolve_staging_path("data/retention")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    records = []
    for panel in panels:
        for item in panel_config(panel):
            meta_path = root / (item["id"] + ".json")
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if any(meta.get(k) != value for k, value in item.items()):
                    if not supersede_changed:
                        raise ValueError(f"Panel definition changed: {item['id']}; rerun with "
                                         "--supersede-changed to move the old input aside")
                    _supersede(root, item["id"], meta, stamp)
                else:
                    if file_digest(root / meta["file"]) != meta["sha256"]:
                        raise RuntimeError(f"Retention input changed: {item['id']}")
                    records.append(meta)
                    continue
            print("Preparing", item["id"], flush=True)
            details = {}
            if item["source"] == "sklearn":
                bunch = getattr(datasets, "load_" + item["name"])(as_frame=True)
                source = "sklearn.datasets.load_" + item["name"]
            else:
                bunch = fetch_openml(item["data_id"], root / "download_cache")
                if "version" in item and str(bunch.details["version"]) != str(item["version"]):
                    raise ValueError("OpenML version differs from the fixed panel")
                expected = (item.get("tabarena") or {}).get("n")
                if expected is not None and len(bunch.data) != int(expected):
                    raise ValueError(f"{item['id']}: {len(bunch.data)} rows, TabArena lists {expected}")
                source = f"https://www.openml.org/d/{item['data_id']}"
                details = {k: bunch.details.get(o) for k, o in (("version", "version"), ("license", "licence"))
                           if k not in item and bunch.details.get(o) is not None}
                if "version" in details:
                    details["version"] = int(details["version"])
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
            processing = {}
            if item.get("preprocessing") == "credit_sanitize":
                X, cats, processing = sanitize_frame(X, y, item, cats)
            elif item.get("preprocessing") is not None:
                raise ValueError(f"Unknown retention preprocessing: {item['preprocessing']}")
            else:
                X["target"] = y
            if "match" in item:   # E3 twin: subsample to its credit table's rows and feature types
                from src.data.control_matching import apply_match
                processing["unmatched_rows"], processing["unmatched_features"] = len(X), len(X.columns) - 1
                X, cats = apply_match(X, cats, item["match"], item["track"], seed=int(item["data_id"]))
            path = root / (item["id"] + ".csv.gz")
            # LF and a fixed gzip time: identical bytes on Windows and Linux for the same table.
            X.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0}, lineterminator="\n")
            meta = dict(item, **details, **processing, panel=panel, source_url=source,
                        file=path.name, rows=len(X), features=len(X.columns)-1,
                        categorical_columns=cats, classes=classes, sha256=file_digest(path))
            write_json(meta_path, meta, exclusive=True)
            records.append(meta)
    print(json.dumps({"prepared": len(records), "panels": list(panels), "root": str(root)}, indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=(*PANELS, "all"), default="all")
    parser.add_argument("--supersede-changed", action="store_true",
                        help="move prepared inputs whose definition changed to superseded/ and re-prepare")
    args = parser.parse_args()
    prepare(PANELS if args.panel == "all" else (args.panel,),
            supersede_changed=args.supersede_changed)


if __name__ == "__main__":
    main()
