"""Fixed public non-credit panels, kept completely outside the training corpus."""
from __future__ import annotations

import json
import pandas as pd
from omegaconf import OmegaConf

from src.utils.paths import REPO_ROOT, resolve_data_path


def panel_config(panel: str) -> list[dict]:
    cfg = OmegaConf.load(REPO_ROOT / "config" / "retention.yaml")
    if panel not in ("smoke", "research"):
        raise ValueError("retention_panel must be none, smoke or research")
    return OmegaConf.to_container(cfg[panel], resolve=True)


def load_refs(panel: str, track: str):
    from src.train.corpus import DatasetRef
    if panel == "none":
        return []
    root = resolve_data_path("data/retention")
    refs = []
    for item in panel_config(panel):
        if item["track"] != track:
            continue
        meta_path = root / (item["id"] + ".json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Prepare the retention panel on a login node first: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(meta.get(k) != value for k, value in item.items()):
            raise ValueError(f"Retention catalog and prepared metadata differ: {item['id']}")
        path = root / meta["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        from src.utils.experiment import file_digest
        if file_digest(path) != meta["sha256"]:
            raise ValueError(f"Retention data checksum mismatch: {item['id']}")
        refs.append(DatasetRef(item["id"], track,
            "classification" if track == "pd" else "regression", "target",
            tuple(meta["categorical_columns"]), path, int(meta["rows"])))
    return refs


def processed_dataset(track: str, dataset_id: str):
    from src.eval.dataset_loader import ProcessedDataset
    panels = [p for p in ("smoke", "research")
              if any(d["id"] == dataset_id and d["track"] == track for d in panel_config(p))]
    refs = [r for panel in panels for r in load_refs(panel, track) if r.dataset_id == dataset_id]
    if len(refs) != 1:
        raise KeyError(dataset_id)
    ref = refs[0]
    frame = pd.read_csv(ref.processed_csv)
    return ProcessedDataset(frame.drop(columns="target"), frame.target.to_numpy(),
        list(ref.categorical_columns), ref.task_type, ref.dataset_id, track)


def is_retention(dataset_id: str) -> bool:
    return any(d["id"] == dataset_id for p in ("smoke", "research") for d in panel_config(p))
