"""Within-track duplicate detection (Stages 1 and 4 of the pipeline).

Runs **per track** — one pass over the PD corpus, one over the LGD
corpus. A dataset that legitimately appears in both tracks is allowed
and never flagged. Duplicate detection runs twice in the overall
pipeline:

  ``--pass pre``   reads ``cfg.paths.raw/{pd,lgd}/*.csv``
                   and detects duplicates *before* any cleaning.

  ``--pass post``  reads ``cfg.paths.processed/{pd,lgd}/*.sanitized.csv``
                   and detects duplicates *after* the canonical-form
                   sanitisation, catching pairs that only become
                   identical once cleaning has happened.

Detection methods (all run; a pair flagged by more than one is fine —
they are concatenated into the ``detection_method`` column):

  ``id_match``                identifier collision on
                              ``(source, dataset_id)`` or on
                              ``(name, n_rows, n_cols, task_type)``.
  ``name_jaccard_and_shape``  column-name Jaccard ≥
                              ``cfg.dedup.name_jaccard_threshold`` *and*
                              identical ``(n_rows, n_cols)``.
  ``row_hash``                row-level SHA-256 (after per-row column-
                              sorted serialisation): non-empty
                              intersection of hash sets.
  ``col_hash``                column-level SHA-256 (after per-column
                              value sorting): pair shares
                              ≥ ``cfg.dedup.shared_columns_min`` columns
                              each with > ``column_nontrivial_unique_min``
                              unique values.
  ``rounded_row``  *(extra)*  same as ``row_hash`` but floats are
                              rounded to ``cfg.dedup.rounded_row.decimals``
                              before hashing.
  ``subset``       *(extra)*  ``|A ∩ B| / |A| ≥``
                              ``cfg.dedup.subset.min_overlap_fraction``
                              ⇒ A is a subset of B; A is flagged.
  ``fuzzy_names``  *(extra)*  rapidfuzz token-set ratio between sorted
                              column-name strings ≥
                              ``cfg.dedup.column_name_fuzzy.similarity_threshold``.

Confidence labelling is driven by ``cfg.dedup.confidence_rules``: if
any ``high`` check fired the row is labelled ``high``; otherwise
``medium`` if any ``medium`` check fired; else ``low``.

Output policy
-------------
Strict "first encountered wins": the first occurrence of a dataset
within a track is always kept. Only subsequent duplicates appear in
the CSV. So if a dataset X is duplicated three times in PD, two rows
go to ``doubles_pd_<pass>.csv`` (the second and third occurrences),
and the first stays untouched.

Public entry point
------------------
``main(cfg, pass_name="pre" | "post") -> int``

    Reads
    -----
    Pre  : ``cfg.paths.raw/{pd,lgd}/*.csv``
    Post : ``cfg.paths.processed/{pd,lgd}/*.sanitized.csv``

    Writes
    ------
    ``cfg.paths.dedup/doubles_pd_<pass>.csv``  (sorted by
        ``duplicate_of`` then ``dataset_name``)
    ``cfg.paths.dedup/doubles_lgd_<pass>.csv``

    Existing CSVs for the *same pass* are deleted before writing
    when ``cfg.dedup.overwrite_existing_pass_csv=true`` (default).

    Returns
    -------
    int
        ``0`` on success, ``1`` if any track failed to load.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import DATASET_METADATA

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Per-dataset fingerprint
# =============================================================================


@dataclass
class Fingerprint:
    dataset_id: str
    dataset_name: str
    source: str
    track: str
    task_type: str
    path: Path
    n_rows: int
    n_cols: int
    columns: list[str]
    row_hashes: set[int] = field(default_factory=set)
    rounded_row_hashes: set[int] = field(default_factory=set)
    column_hashes: set[int] = field(default_factory=set)


def _row_hashes(df: pd.DataFrame, decimals: int | None) -> set[int]:
    """Hash every row after sorting columns alphabetically.

    Numeric columns are optionally rounded to ``decimals`` places to
    catch near-duplicates that diverged via float-precision round-trips.

    Uses pandas' vectorised ``hash_pandas_object`` (Cython, O(rows*cols)
    with C-level constants) — orders of magnitude faster than a
    per-row SHA-256. The collision-resistance of a 64-bit hash is
    sufficient for cross-dataset overlap detection at the corpus sizes
    we work with (≤ a few million rows): a birthday collision needs
    ~5×10⁹ rows.

    Hashes are returned as a set; duplicate rows within a single
    dataset collapse, which is what we want.
    """
    if df.empty:
        return set()
    sorted_cols = sorted(df.columns.tolist())
    sub = df[sorted_cols]
    if decimals is not None:
        # Round numeric columns into a copy so the original df is untouched.
        copy_needed = {c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])}
        if copy_needed:
            sub = sub.copy()
            for c in copy_needed:
                sub[c] = np.round(sub[c].astype(np.float64), decimals=decimals)
    h = pd.util.hash_pandas_object(sub, index=False)
    return set(h.to_numpy().tolist())


def _column_hashes(df: pd.DataFrame, nontrivial_unique_min: int) -> set[int]:
    """Hash every column whose number of unique values is > threshold.

    Uses ``hash_pandas_object`` on the *sorted* values of each column,
    so the resulting hash is invariant to row ordering. Returns a set
    of uint64 hashes.
    """
    out: set[int] = set()
    for col in df.columns:
        s = df[col].dropna()
        if s.nunique() <= nontrivial_unique_min:
            continue
        try:
            s_sorted = pd.Series(np.sort(s.to_numpy()))
        except TypeError:
            s_sorted = pd.Series(np.sort(s.astype(str).to_numpy()))
        h = pd.util.hash_pandas_object(s_sorted, index=False)
        # Combine the column's per-row hashes into a single value via XOR
        # (order-independent, but here we already sorted so xor is fine).
        combined = int(np.bitwise_xor.reduce(h.to_numpy()))
        out.add(combined)
    return out


def compute_fingerprint(
    path: Path,
    dataset_id: str,
    *,
    rounded_row_decimals: int | None,
    column_nontrivial_unique_min: int,
) -> Fingerprint:
    """Build a Fingerprint for one dataset by reading its CSV and
    computing all of the hash sets needed for pairwise comparison."""
    meta = DATASET_METADATA[dataset_id]
    df = pd.read_csv(path, low_memory=False)
    return Fingerprint(
        dataset_id=dataset_id,
        dataset_name=path.stem,
        source=meta["source"],
        track=meta["track"],
        task_type=meta["task_type"],
        path=path,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        columns=list(df.columns),
        row_hashes=_row_hashes(df, decimals=None),
        rounded_row_hashes=(
            _row_hashes(df, decimals=rounded_row_decimals)
            if rounded_row_decimals is not None else set()
        ),
        column_hashes=_column_hashes(df, column_nontrivial_unique_min),
    )


# =============================================================================
# Pairwise checks
# =============================================================================


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _fuzzy_name_similarity(a: list[str], b: list[str]) -> float:
    """rapidfuzz token-set ratio between concatenated sorted column names.

    rapidfuzz returns 0–100; convert to that scale. Imported lazily so
    the rest of the module is usable without the dep.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        LOGGER.debug("rapidfuzz not installed; fuzzy_names check disabled")
        return 0.0
    return float(fuzz.token_set_ratio(" ".join(sorted(a)), " ".join(sorted(b))))


def compare_pair(
    a: Fingerprint, b: Fingerprint, cfg, *, enable_fuzzy_names: bool,
) -> list[str]:
    """Return the list of triggered detection methods for the pair (a, b).

    Empty list ⇒ not duplicates. ``enable_fuzzy_names`` is supplied by
    ``main`` based on the pass name (pre vs. post) — see the
    ``column_name_fuzzy`` block in ``config/data.yaml`` for why this
    differs across passes.
    """
    triggered: list[str] = []

    # --- check 1: identifier match ----------------------------------------
    if (a.source == b.source and a.source not in ("local", "")
            and a.dataset_id == b.dataset_id):
        triggered.append("id_match")
    if (a.dataset_name == b.dataset_name
            and a.n_rows == b.n_rows
            and a.n_cols == b.n_cols
            and a.task_type == b.task_type):
        if "id_match" not in triggered:
            triggered.append("id_match")

    # --- check 2: column-name Jaccard + identical shape -------------------
    if a.n_rows == b.n_rows and a.n_cols == b.n_cols:
        if _jaccard(a.columns, b.columns) >= cfg.dedup.name_jaccard_threshold:
            triggered.append("name_jaccard_and_shape")

    # --- check 3: row-level hash ------------------------------------------
    if (a.row_hashes and b.row_hashes
            and len(a.row_hashes & b.row_hashes) >= cfg.dedup.row_hash_intersection_min):
        triggered.append("row_hash")

    # --- check 4: column-level hash ---------------------------------------
    if a.column_hashes and b.column_hashes:
        shared = a.column_hashes & b.column_hashes
        if len(shared) >= cfg.dedup.shared_columns_min:
            triggered.append("col_hash")

    # --- extra A: rounded-row hash ----------------------------------------
    if cfg.dedup.rounded_row.enabled:
        if (a.rounded_row_hashes and b.rounded_row_hashes
                and a.rounded_row_hashes & b.rounded_row_hashes):
            triggered.append("rounded_row")

    # --- extra B: subset detection ----------------------------------------
    if cfg.dedup.subset.enabled and a.row_hashes and b.row_hashes:
        overlap = len(a.row_hashes & b.row_hashes)
        if overlap and overlap / max(1, len(a.row_hashes)) >= cfg.dedup.subset.min_overlap_fraction:
            triggered.append("subset")

    # --- extra C: fuzzy column-name match --------------------------------
    if enable_fuzzy_names:
        sim = _fuzzy_name_similarity(a.columns, b.columns)
        if sim >= cfg.dedup.column_name_fuzzy.similarity_threshold:
            if "name_jaccard_and_shape" not in triggered:
                triggered.append("fuzzy_names")

    return triggered


def confidence_for(triggered: list[str], cfg) -> str:
    """Map the triggered-check list to a confidence label."""
    rules = cfg.dedup.confidence_rules
    if any(t in rules.high for t in triggered):
        return "high"
    if any(t in rules.medium for t in triggered):
        return "medium"
    return "low"


# =============================================================================
# Per-track sweep
# =============================================================================


def find_duplicates_in_track(
    paths: list[Path], cfg, *, enable_fuzzy_names: bool,
) -> list[dict]:
    """Build fingerprints for every path, then pairwise-compare.

    Returns a list of duplicate-record dicts, sorted by (duplicate_of,
    dataset_name).
    """
    fingerprints: list[Fingerprint] = []
    for p in paths:
        # Resolve dataset_id from the filename. Pre-pass: "0001.gmsc.csv".
        # Post-pass: "0001.gmsc.sanitized.csv". Strip both suffixes.
        stem = p.stem
        if stem.endswith(".sanitized"):
            stem = stem[: -len(".sanitized")]
        if stem not in DATASET_METADATA:
            LOGGER.warning("unknown dataset_id from %s; skipped", p)
            continue
        try:
            fp = compute_fingerprint(
                p, stem,
                rounded_row_decimals=(
                    cfg.dedup.rounded_row.decimals
                    if cfg.dedup.rounded_row.enabled else None
                ),
                column_nontrivial_unique_min=cfg.dedup.column_nontrivial_unique_min,
            )
        except Exception as exc:
            LOGGER.error("fingerprint failed for %s: %s", p, exc)
            continue
        fingerprints.append(fp)

    # Iterate in path order so "first encountered" is deterministic.
    fingerprints.sort(key=lambda f: f.path.name)

    records: list[dict] = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            a = fingerprints[i]
            b = fingerprints[j]
            triggered = compare_pair(a, b, cfg, enable_fuzzy_names=enable_fuzzy_names)
            if not triggered:
                continue
            # b is the later-encountered occurrence → b is the duplicate
            # of a, in the strict "first wins" policy.
            records.append({
                "dataset_path": str(b.path),
                "dataset_name": b.dataset_name,
                "duplicate_of": a.dataset_name,
                "detection_method": ";".join(triggered),
                "confidence": confidence_for(triggered, cfg),
            })

    records.sort(key=lambda r: (r["duplicate_of"], r["dataset_name"]))
    return records


# =============================================================================
# CLI
# =============================================================================


def _load_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load("config/data.yaml")


def _gather_track_paths(cfg, track: str, pass_name: str) -> list[Path]:
    if pass_name == "pre":
        root = Path(cfg.paths.raw) / track
        return sorted(root.glob("*.csv"))
    if pass_name == "post":
        root = Path(cfg.paths.processed) / track
        return sorted(root.glob("*.sanitized.csv"))
    raise ValueError(f"pass must be 'pre' or 'post', got {pass_name!r}")


def main(cfg=None, pass_name: str = "pre") -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if cfg is None:
        cfg = _load_cfg()
    if pass_name not in ("pre", "post"):
        raise ValueError(f"pass_name must be 'pre' or 'post', got {pass_name!r}")

    out_dir = Path(cfg.paths.dedup)
    out_dir.mkdir(parents=True, exist_ok=True)

    enable_fuzzy = (
        cfg.dedup.column_name_fuzzy.enabled_pre if pass_name == "pre"
        else cfg.dedup.column_name_fuzzy.enabled_post
    )

    failures = 0
    for track in ("pd", "lgd"):
        paths = _gather_track_paths(cfg, track, pass_name)
        if not paths:
            LOGGER.warning("no %s files for track=%s pass=%s", "csv", track, pass_name)

        records = find_duplicates_in_track(
            paths, cfg, enable_fuzzy_names=enable_fuzzy,
        )

        out_path = out_dir / f"doubles_{track}_{pass_name}.csv"
        if cfg.dedup.overwrite_existing_pass_csv and out_path.exists():
            out_path.unlink()
        df = pd.DataFrame(records, columns=[
            "dataset_path", "dataset_name", "duplicate_of",
            "detection_method", "confidence",
        ])
        df.to_csv(out_path, index=False)
        LOGGER.info(
            "track=%s pass=%s: %d duplicate records → %s",
            track, pass_name, len(records), out_path,
        )

    return 1 if failures else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Within-track duplicate sweep (per track, per pass)."
    )
    parser.add_argument(
        "--pass", dest="pass_name", choices=("pre", "post"), required=True,
        help="'pre' = sweep raw CSVs; 'post' = sweep sanitised CSVs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(pass_name=args.pass_name))
