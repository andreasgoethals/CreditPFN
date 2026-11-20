"""Rebuild the PD 0006 outcome key from the original HackerEarth and LendingClub releases.

Run `python -m src.data.lendingclub_outcomes --archives DIRECTORY` with the four pinned
LoanStats ZIPs (or add `--download` to acquire them). Only preprocessing needs the
resulting small, checksum-verified key in data/raw/sources/lendingclub/.
"""
import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.atomic import write_json
from src.utils.experiment import file_digest
from src.utils.paths import raw_dir

DATASET = "0006.hackerearth"
TARGET = "loan_status"
KEY_NAME = "terminal_outcomes_2012_2015.csv"
CHALLENGE_SHA256 = "63a4b2360d1c60eb67042a3c35f7f1b0a466ddffed2084cad3feee2138985b1b"
EARLY_TABLE_SHA256 = "91969386d42aa27a910a464b7987a9b225451282fa2934baf2c1afb9ef6480c7"
ARCHIVES = {
    "LoanStats3a.csv.zip": "6f93fc97f0ad26718ed66c6199931ac6204ce80d7c14da0215b529b0891f12be",
    "LoanStats3b.csv.zip": "d777843163ecef43d311a66eacda175cd2bfb22e608da629d3b334a8f317a239",
    "LoanStats3c.csv.zip": "608efe0ef44ea82b44c0bff0f924355ed533141ca1fcb65a4eb2c9885fa58079",
    "LoanStats3d.csv.zip": "069657c7f9645bc6cf566f6594d11c02b7b2177dce9a2e458873e8c04b12497f",
}
MATCH_FIELDS = ("loan_amnt funded_amnt term int_rate grade sub_grade annual_inc home_ownership "
                "purpose addr_state dti revol_bal open_acc total_acc delinq_2yrs inq_last_6mths").split()
CHECK_FIELDS = ["pub_rec", "revol_util"]
KEY_SHA256 = "af8e31982b3b5c0518c65d5cbfe1afebd11cd2129aa1d364121a5419dcd025c6"
PROCESSED_SHA256 = "d9c056ab818a91a8eafd53dd4302b1c954ada8e8a36e1374d8a9d2b39d660ae7"


def key_path(raw_csv: Path) -> Path:
    return raw_csv.parent.parent / "sources" / "lendingclub" / KEY_NAME


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name in [*MATCH_FIELDS, *CHECK_FIELDS]:
        if name in {"grade", "sub_grade", "home_ownership", "purpose", "addr_state"}:
            frame[name] = frame[name].astype("string").str.strip().str.lower()
        elif name == "term":
            frame[name] = pd.to_numeric(frame[name].astype("string").str.extract(r"(\d+)", expand=False))
        else:
            values = frame[name].astype("string").str.strip().str.removesuffix("%")
            frame[name] = pd.to_numeric(values, errors="raise").round(5)
    complete = frame.dropna(subset=MATCH_FIELDS)
    return complete.loc[~complete.duplicated(MATCH_FIELDS, keep=False)]


def match_outcomes(challenge: pd.DataFrame, lender: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Require one-to-one complete profiles across *all* releases before restricting issue years."""
    if challenge["member_id"].isna().any() or not challenge["member_id"].is_unique:
        raise ValueError("Challenge member IDs must identify distinct loan rows")
    matched = _normalise(challenge).merge(_normalise(lender), on=MATCH_FIELDS,
                                          suffixes=("_challenge", "_lender"), validate="one_to_one")
    if not matched.pub_rec_challenge.eq(matched.pub_rec_lender).fillna(False).all():
        raise ValueError("Lender linkage disagrees on independent public-record counts")
    both = matched[["revol_util_challenge", "revol_util_lender"]].notna().all(axis=1)
    if not np.isclose(matched.loc[both, "revol_util_challenge"].to_numpy(dtype=float),
                      matched.loc[both, "revol_util_lender"].to_numpy(dtype=float), atol=.011, rtol=0).all():
        raise ValueError("Lender linkage disagrees on independently checked revolving utilization")
    matches = matched["issue_year"].between(2012, 2015)
    later = matched.loc[matches, ["member_id", "issue_year", "loan_status_lender"]].copy()
    terminal = later.pop("loan_status_lender").map({"Fully Paid": 0, "Charged Off": 1})
    if terminal.isna().any():
        raise ValueError("A later-vintage matched loan lacks a terminal lender outcome")
    later[TARGET] = terminal.astype("int64")
    counts = later.groupby(["issue_year", TARGET]).size()
    report = dict(challenge_rows=len(challenge), lender_rows=len(lender), unique_matches=len(matched),
                  unmatched_or_ambiguous=len(challenge) - len(matched), early_matches=int((~matches).sum()),
                  verified_revolving_utilizations=int(both.sum()), kept=len(later),
                  paid=int((terminal == 0).sum()), charged_off=int((terminal == 1).sum()),
                  by_issue_year={str(year): {"paid": int(counts.get((year, 0), 0)),
                                            "charged_off": int(counts.get((year, 1), 0))}
                                 for year in sorted(later.issue_year.unique())})
    return later[["member_id", TARGET]].sort_values("member_id", ignore_index=True), report


def audit_early_table(early: pd.DataFrame, lender: pd.DataFrame) -> dict:
    """Check 0004 independently using the application fields its distinct extract retained."""
    early = early.rename(columns=lambda name: name.replace(".", "_")).copy()
    early["int_rate"] *= 100
    keys = ["int_rate", "installment", "dti", "revol_bal", "inq_last_6mths", "delinq_2yrs", "pub_rec"]
    reference = lender[[*keys, "annual_inc", "revol_util", "issue_year", TARGET]].copy()
    for frame in (early, reference):
        for name in [*keys, "revol_util"]:
            values = frame[name].astype("string").str.strip().str.removesuffix("%")
            frame[name] = pd.to_numeric(values).round(2)
    unique_early = early.dropna(subset=keys).drop_duplicates(keys, keep=False)
    unique_ref = reference.dropna(subset=keys).drop_duplicates(keys, keep=False)
    unique = unique_early.merge(unique_ref, on=keys, suffixes=("_early", "_lender"), validate="one_to_one")

    def corroborated(rows):
        return rows.loc[np.isclose(np.exp(rows.log_annual_inc), rows.annual_inc, rtol=0, atol=1)
                        & np.isclose(rows.revol_util_early, rows.revol_util_lender, rtol=0, atol=.011)]

    verified = corroborated(unique)
    terminal = verified[TARGET].str.split(":").str[-1].str.strip().map({"Fully Paid": 0, "Charged Off": 1})
    if terminal.isna().any() or not terminal.eq(verified.not_fully_paid).all():
        raise ValueError("PD 0004 outcome disagrees with a corroborated lender record")
    # Also look for possible later duplicates without the uniqueness filter, so an ambiguous
    # early match cannot hide a loan from the same population as the replacement 0006.
    candidates = early.dropna(subset=keys).merge(reference.loc[reference.issue_year >= 2012].dropna(subset=keys),
                                                  on=keys, suffixes=("_early", "_lender"))
    later = len(corroborated(candidates))
    if later or len(verified) != 9_512:
        raise ValueError(f"PD 0004 verification changed: {len(verified)} confirmed outcomes, {later} later matches")
    return dict(rows=len(early), confirmed=len(verified), paid=int((terminal == 0).sum()),
                charged_off=int((terminal == 1).sum()), corroborated_later_pairs=later, sha256=EARLY_TABLE_SHA256)


def _read_archive(path: Path, download: bool) -> pd.DataFrame:
    if not path.is_file() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            pending = Path(temp) / path.name
            with urllib.request.urlopen("https://resources.lendingclub.com/" + path.name, timeout=90) as response:
                with pending.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if file_digest(pending) != ARCHIVES[path.name]:
                raise ValueError(f"Published LendingClub archive checksum differs: {path.name}")
            pending.replace(path)
    if file_digest(path) != ARCHIVES[path.name]:
        raise ValueError(f"Wrong LendingClub archive checksum: {path.name}")
    with zipfile.ZipFile(path) as archive:
        member, = [name for name in archive.namelist() if name.endswith(".csv") and not name.startswith("__")]
        with archive.open(member) as stream:
            skip = int(b"loan_amnt" not in stream.readline())
        with archive.open(member) as stream:
            columns = [*MATCH_FIELDS, *CHECK_FIELDS, "installment", "issue_d", TARGET]
            frame = pd.read_csv(stream, skiprows=skip, usecols=columns, low_memory=False)
    frame = frame.loc[frame.issue_d.notna()].copy()
    frame["issue_year"] = pd.to_datetime(frame.pop("issue_d"), format="%b-%Y").dt.year
    return frame


def build_key(raw_csv: Path, archives: Path, *, download: bool = False) -> dict:
    from src.data.preprocessing import write_csv_atomic
    if file_digest(raw_csv) != CHALLENGE_SHA256:
        raise ValueError("The HackerEarth CSV differs from the verified challenge release")
    early_csv = raw_csv.with_name("0004.lendingclub.csv")
    if file_digest(early_csv) != EARLY_TABLE_SHA256:
        raise ValueError("The earlier LendingClub table differs from the registered 0004 extract")
    challenge = pd.read_csv(raw_csv, usecols=["member_id", *MATCH_FIELDS, *CHECK_FIELDS, TARGET], low_memory=False)
    lender = pd.concat([_read_archive(archives / name, download) for name in ARCHIVES], ignore_index=True)
    key, report = match_outcomes(challenge, lender)
    report["earlier_table"] = audit_early_table(pd.read_csv(early_csv), lender)
    if report["unique_matches"] != 532_152 or len(key) != 506_724:
        raise ValueError(f"Unexpected LendingClub cohort: {report}")
    path = key_path(raw_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        pending = Path(temp) / KEY_NAME
        write_csv_atomic(key, pending, index=False)
        checksum = file_digest(pending)
        if checksum != KEY_SHA256:
            raise ValueError(f"Unexpected LendingClub outcome-key checksum: {checksum}")
        pending.replace(path)
    report.update(archives=ARCHIVES, challenge_sha256=CHALLENGE_SHA256, key_sha256=checksum,
                  matched_fields=MATCH_FIELDS, independent_checks=CHECK_FIELDS)
    write_json(path.with_suffix(".json"), report)
    return report


def verified_labels(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.is_file() or file_digest(path) != KEY_SHA256:
        raise ValueError(f"Missing or invalid verified LendingClub key at {path}. "
                         "Rebuild it with python -m src.data.lendingclub_outcomes --archives DIRECTORY")
    key = pd.read_csv(path, dtype={"member_id": "int64", TARGET: "int64"}).set_index("member_id")
    if raw.member_id.isna().any() or not raw.member_id.is_unique:
        raise ValueError("Challenge member IDs must identify distinct loan rows")
    labels = raw.member_id.map(key[TARGET])
    if labels.notna().sum() != len(key):
        raise ValueError("The challenge input lacks loans from the verified LendingClub cohort")
    selected = raw.loc[labels.notna()].copy()
    selected[TARGET] = labels[labels.notna()].astype("int64")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="download missing pinned public archives")
    parser.add_argument("--raw-csv", type=Path, help="original HackerEarth CSV beside the 0004 extract")
    args = parser.parse_args()
    raw_csv = args.raw_csv or raw_dir("pd", DATASET + ".csv")
    print(json.dumps(build_key(raw_csv, args.archives, download=args.download), indent=2))


if __name__ == "__main__":
    main()
