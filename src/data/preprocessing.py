"""Dataset-specific surgical fixes (data-pipeline helper).

This module is the single source of truth for *per-dataset* metadata
and *per-dataset* surgical preprocessing — the kinds of fixes that
cannot be automated (drop a column we know to be an internal ID,
decode a hand-crafted string-to-numeric mapping, parse a "5yrs 3mon"
column into months, drop columns derived from the target, …).

Two public surfaces:

1. ``DATASET_METADATA`` — frozen dict literal mapping ``dataset_id`` to
   target column / task type / known categorical columns / source. Used
   by :mod:`src.data.register` to bootstrap the manifest, and by every
   downstream stage that needs to know "what is the target column" or
   "which columns are categorical".
2. ``apply_dataset_specific_fixes(df, dataset_id)`` — surgical fixes
   applied per dataset. Idempotent. Most datasets pass through almost
   unchanged; a handful with known data-quality issues get explicit
   surgery.

What this module deliberately does NOT do:

* No statistical operations (no PCA, no scaling, no winsorisation).
  Those are sanitize.py's job, *only* for things TabPFN's internal
  preprocessing pipeline does not already handle.
* No ``np.log1p`` skew transforms. TabPFN's per-estimator ensemble
  cycles through ``safepower`` / ``quantile_uni`` / ``kdi`` / ``robust``
  on every fit (see ``repositories/TabPFN Docs.txt`` line 6346–6352);
  pre-applying a log on disk would fight that ensemble.
* No outlier clipping. ``OUTLIER_REMOVAL_STD = 12.0`` (classifier) /
  ``None`` (regressor) inside TabPFN already handles this with the
  right semantics (context-only statistics, soft log-squash). See
  ``repositories/REPOSITORIES.md::Outlier handling`` for the full
  argument.
* No target normalisation. TabPFN's ``RegressorBatch.znorm_space_bardist_``
  z-normalises regression targets internally and inverts at predict
  time.

Public entry point
------------------
``main(cfg: OmegaConf) -> int``
    Smoke-test only — the module is meant to be imported, not run.
    When invoked from the command line, iterates over every raw CSV,
    applies the registered surgical fix, and prints (dataset_id, raw
    shape, post-fix shape, target column). Writes nothing to disk.

    Reads
    -----
    ``cfg.paths.raw/{pd,lgd}/<id>.csv`` — every raw file present.

    Writes
    ------
    Nothing.

    Returns
    -------
    int
        ``0`` if every dataset's fix ran cleanly, ``1`` if any
        ``dataset_id`` raised. The script does not abort on a
        per-dataset failure — failures are logged and counted.

Per-dataset metadata (target columns, categorical column hints, ID
columns to drop, leakage columns to drop, bespoke string decodings)
was determined by inspecting each raw CSV directly. Statistical
transforms (log / power / quantile) are deliberately left to TabPFN's
internal preprocessing pipeline at training time and are NOT applied
here — see :mod:`src.data.sanitize` for the rationale.

Adding a new dataset
--------------------
Designed to scale to the 3000-dataset corpus we will buy. Every new
dataset takes **exactly two changes** to this file (often just one):

1. **Required.** Call :func:`_register` with the dataset's metadata:

   .. code-block:: python

       _register(
           "0042.new_dataset",
           track="pd",
           task_type="classification",
           target_column="default_flag",
           categorical_columns=["region", "industry"],
           source="vendor-foo",
           source_url="https://...",
       )

   That's it — no other code change needed if the dataset is "clean
   enough" (no ID columns to drop, no bespoke string decodings, etc.).
   The default surgical fix is the identity function.

2. **Optional.** If the dataset needs surgery (drop ID columns, decode
   bespoke strings, …), define a fix function decorated with
   :func:`_register_fix`:

   .. code-block:: python

       @_register_fix("0042.new_dataset")
       def _fix_new_dataset(df: pd.DataFrame) -> pd.DataFrame:
           df = df.copy()
           df = df.drop(columns=[c for c in ["row_id"] if c in df.columns])
           return df

   The decorator self-registers; you do **not** need to update any
   central dispatch table.

A bootstrap helper :func:`register_from_records` is provided for
bulk-importing metadata from a vendor-supplied CSV/JSON when scaling
to 3000+ datasets — see its docstring for the contract.
"""

from __future__ import annotations

import logging
import re
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Per-dataset metadata
# =============================================================================
# Frozen via MappingProxyType at module bottom. Source-of-truth for:
#   - target column
#   - task type ("classification" or "regression")
#   - track ("pd" or "lgd")
#   - hand-curated categorical column hints (empty list ≡ "all numeric
#     features apart from the target are numerical")
#   - source provenance
#
# When a new dataset is added: append a row here AND register a fix
# function below if any surgery is needed.

_RAW_METADATA: dict[str, dict] = {
    # ---- PD (classification) -----------------------------------------------
    "0001.gmsc": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "SeriousDlqin2yrs",
        "categorical_columns": [],
        "source": "kaggle",
        "source_url": "https://www.kaggle.com/c/GiveMeSomeCredit",
    },
    "0002.taiwan_creditcard": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "default.payment.next.month",
        "categorical_columns": [],
        "source": "uci",
        "source_url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
    },
    "0003.vehicle_loan": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "loan_default",
        "categorical_columns": ["manufacturer_id", "Employment.Type", "State_ID"],
        "source": "kaggle",
        "source_url": "https://www.kaggle.com/datasets/avikpaul4u/vehicle-loan-default-prediction",
    },
    "0004.lendingclub": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "not.fully.paid",
        "categorical_columns": [],   # auto-inferred at register time
        "source": "lendingclub",
        "source_url": "https://www.lendingclub.com/info/download-data.action",
    },
    "0005.myhom": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "loan_default",
        "categorical_columns": [],
        "source": "local",
        "source_url": None,
    },
    "0006.hackerearth": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "loan_status",
        "categorical_columns": [
            "addr_state", "home_ownership", "verification_status",
            "purpose", "application_type", "grade", "sub_grade",
            "initial_list_status",
        ],
        "source": "hackerearth",
        "source_url": None,
    },
    "0007.cobranded": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "default_ind",
        "categorical_columns": [],
        "source": "local",
        "source_url": None,
    },
    "0008.german": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "target",
        "categorical_columns": [
            "feature_1", "feature_3", "feature_4", "feature_6", "feature_7",
            "feature_9", "feature_10", "feature_12", "feature_14", "feature_15",
            "feature_17", "feature_19", "feature_20",
        ],
        "source": "uci",
        "source_url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
    },
    "0009.bank_status": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "Loan Status",
        "categorical_columns": [],
        "source": "kaggle",
        "source_url": "https://www.kaggle.com/datasets/zhijinzhai/loandata",
    },
    "0010.thomas": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "BAD",
        "categorical_columns": [],
        "source": "thomas-credit-textbook",
        "source_url": None,
    },
    "0011.loan_default": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "loss",     # binarised in the surgical fix
        "categorical_columns": [],
        "source": "kaggle",
        "source_url": "https://www.kaggle.com/c/loan-default-prediction",
    },
    "0012.home_credit": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "TARGET",
        "categorical_columns": [
            "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
            "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
            "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE",
            "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
        ],
        "source": "kaggle",
        "source_url": "https://www.kaggle.com/c/home-credit-default-risk",
    },
    "0013.hmeq": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "BAD",
        "categorical_columns": ["REASON", "JOB"],
        "source": "uci-style-textbook",
        "source_url": None,
    },
    "0014.algorithmwatch": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "arrears",
        "categorical_columns": [],   # 2 987 features — auto-inferred
        "source": "algorithmwatch",
        "source_url": None,
    },
    "0015.credit_risk_dataset": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "loan_status",
        # `loan_grade` (A..G) is ORDINAL — the surgical fix maps it to
        # an integer rank, so it's NOT in the categoricals list any more.
        # The other two object columns are nominal and stay categorical.
        "categorical_columns": ["person_home_ownership", "loan_intent"],
        "source": "kaggle",
        "source_url": "https://www.kaggle.com/datasets/laotse/credit-risk-dataset",
    },
    "0016.bondora_peer2peer": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "is_default",
        # Of the 31 raw columns only ~7 are safe at origination — the
        # rest are post-loan payment progression / direct default
        # indicators. The surgical fix drops all leakage; see comments
        # there for the column-by-column rationale.
        "categorical_columns": ["country", "customer_risk_rating"],
        "source": "bondora",
        "source_url": "https://www.bondora.com/marketing/media/LoanData.zip",
    },
    "0017.SBA_loans_case": {
        "track": "pd",
        "task_type": "classification",
        "target_column": "Default",
        # `RevLineCr`, `LowDoc`: nominal Y/N (after dirty-value cleanup);
        # `State`, `BankState`: 2-letter US state codes;
        # `NAICS`: 6-digit industry code (high cardinality).
        # Other small-int columns (NewExist, UrbanRural, FranchiseCode,
        # New, RealEstate, Recession) are binary/3-level flags — left as
        # numeric for simplicity; sanitize.py won't drop them.
        "categorical_columns": [
            "State", "BankState", "NAICS", "RevLineCr", "LowDoc",
        ],
        "source": "kaggle",
        "source_url": "https://www.kaggle.com/datasets/larsen0966/sba-loans-case-data-set",
    },
    # ---- LGD (regression) --------------------------------------------------
    "0001.heloc": {
        "track": "lgd",
        "task_type": "regression",
        "target_column": "LGD_ACTG",
        "categorical_columns": [],
        "source": "local",
        "source_url": None,
    },
    "0002.loss2": {
        "track": "lgd",
        "task_type": "regression",
        "target_column": "_ELGD",
        "categorical_columns": [],
        "source": "local",
        "source_url": None,
    },
    "0003.axa": {
        "track": "lgd",
        "task_type": "regression",
        "target_column": "lgd_time",
        "categorical_columns": [],
        "source": "local",
        "source_url": None,
    },
    "0004.base_model": {
        "track": "lgd",
        "task_type": "regression",
        "target_column": "LGD_brute",
        "categorical_columns": [],
        "source": "local",
        "source_url": None,
    },
    "0005.base_modelisation": {
        "track": "lgd",
        "task_type": "regression",
        "target_column": "lgd_defaut",
        "categorical_columns": [],
        "source": "local",
        "source_url": None,
    },
    "0006.lgd_freddie": {
        "track": "lgd",
        "task_type": "regression",
        "target_column": "lgd",
        "categorical_columns": [],
        "source": "freddie-mac",
        "source_url": "https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset",
    },
    "0007.lgd_lendingclub": {
        "track": "lgd",
        "task_type": "regression",
        "target_column": "lgd",
        "categorical_columns": [],
        "source": "lendingclub",
        "source_url": "https://www.lendingclub.com/info/download-data.action",
    },
    # ---- LGD twin of 0017 (same physical file in raw/lgd/) ----------------
    # The SBA dataset has both default labels AND charge-off amounts, so we
    # use it twice: once for PD (whole population) and once for LGD (defaults
    # only, target derived from ChgOffPrinGr / DisbursementGross). The
    # surgical fix for the LGD copy filters to defaults and derives `lgd`.
    "0008.SBA_loans_case": {
        "track": "lgd",
        "task_type": "regression",
        "target_column": "lgd",      # derived in the surgical fix
        "categorical_columns": [
            "State", "BankState", "NAICS", "RevLineCr", "LowDoc",
        ],
        "source": "kaggle",
        "source_url": "https://www.kaggle.com/datasets/larsen0966/sba-loans-case-data-set",
    },
}

#: Frozen view of the metadata. Importable by other modules; cannot be
#: mutated at runtime.
DATASET_METADATA: Mapping[str, Mapping] = MappingProxyType(
    {k: MappingProxyType(v) for k, v in _RAW_METADATA.items()}
)


def list_dataset_ids(track: str | None = None) -> list[str]:
    """Return all known dataset IDs, optionally filtered by track."""
    if track is None:
        return list(DATASET_METADATA.keys())
    return [k for k, v in DATASET_METADATA.items() if v["track"] == track]


def get_metadata(dataset_id: str) -> Mapping:
    """Return the frozen metadata mapping for one dataset."""
    if dataset_id not in DATASET_METADATA:
        raise KeyError(
            f"Unknown dataset_id={dataset_id!r}. Register it in "
            f"src.data.preprocessing._RAW_METADATA."
        )
    return DATASET_METADATA[dataset_id]


# =============================================================================
# Per-dataset surgical fixes
# =============================================================================
# Each fix function takes the raw DataFrame and returns a cleaned
# DataFrame. Fixes are surgical — only column drops, value decodings,
# parsings of bespoke string formats, and dataset-specific row-dedup
# where the dataset is known to ship with artefactual duplicate rows.
#
# Fix functions self-register via the @_register_fix(<dataset_id>)
# decorator. A dataset that does not need any surgery does not need a
# fix function at all — it falls through to the no-op
# :func:`_passthrough`. The ``unknown_dataset_policy`` in
# ``cfg.preprocessing`` decides whether dataset IDs that are not in
# ``DATASET_METADATA`` raise or pass through.

#: Populated by the @_register_fix decorator below. Keys are
#: dataset_ids; values are functions ``(df: DataFrame) -> DataFrame``.
_FIX_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}


def _register_fix(
    dataset_id: str,
) -> Callable[[Callable[[pd.DataFrame], pd.DataFrame]],
              Callable[[pd.DataFrame], pd.DataFrame]]:
    """Decorator: register a fix function for ``dataset_id``.

    Usage::

        @_register_fix("0042.new_dataset")
        def _fix_new_dataset(df: pd.DataFrame) -> pd.DataFrame:
            return df.drop(columns=[c for c in ["id"] if c in df.columns])

    The decorator stores the function in :data:`_FIX_FUNCTIONS` and
    returns it unchanged, so the function can also be called directly
    in tests.
    """
    def _decorator(func):
        if dataset_id in _FIX_FUNCTIONS:
            raise ValueError(
                f"_register_fix: duplicate registration for {dataset_id!r}"
            )
        _FIX_FUNCTIONS[dataset_id] = func
        return func
    return _decorator


def _passthrough(df: pd.DataFrame) -> pd.DataFrame:
    """Identity fix — used for datasets that need no surgery."""
    return df


@_register_fix("0001.gmsc")
def _fix_gmsc(df: pd.DataFrame) -> pd.DataFrame:
    # gmsc has no internal ID columns to drop and the target is already
    # encoded {0, 1}. No surgery needed.
    return df


@_register_fix("0002.taiwan_creditcard")
def _fix_taiwan_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if "SEX" in df.columns:
        # SEX is encoded as {1: male, 2: female} in the UCI source. Re-map
        # to a 0/1 binary so the column has a sensible numeric ordering
        # for any sklearn-style baseline (TabPFN itself doesn't care about
        # the encoding direction).
        df["SEX"] = df["SEX"].replace({2: 1, 1: 0, "2": 1, "1": 0})
    return df


def _two_digit_year_to_full(value: object) -> int | float:
    """Helper for vehicle_loan dates of the form '17-01-83' or '01/05/83'."""
    if pd.isna(value):
        return np.nan
    s = "".join(ch for ch in str(value) if ch.isdigit())
    if not s:
        return np.nan
    yy = int(s[-2:]) if len(s) >= 2 else int(s)
    return 2000 + yy if 0 <= yy < 20 else 1900 + yy


def _yrs_mons_to_months(value: object) -> int | float:
    """Helper for vehicle_loan 'Xyrs Ymon' strings → integer months.

    Handles both the dataset's actual format ``'1yrs 11mon'`` (no space
    between number and unit) and the more verbose ``'1 year 11 months'``
    style. Earlier ``str.split()``-based versions of this function silently
    returned 0 for every row of vehicle_loan because ``'1yrs'`` is a
    single token that doesn't satisfy ``isdigit()``.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).lower()
    # Accept both 'yr'/'yrs' and 'year'/'years'; 'mon'/'mons' and 'month'/'months'.
    y = re.search(r"(\d+)\s*(?:yr|year)",  s)
    m = re.search(r"(\d+)\s*(?:mon|month)", s)
    years  = int(y.group(1)) if y else 0
    months = int(m.group(1)) if m else 0
    return years * 12 + months


@_register_fix("0003.vehicle_loan")
def _fix_vehicle_loan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [
        "UniqueID", "branch_id", "supplier_id",
        "Current_pincode_ID", "Employee_code_ID", "MobileNo_Avl_Flag",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Date-of-birth + disbursal-date → integer year, then derive Age.
    if "Date.of.Birth" in df.columns:
        df["Date.of.Birth"] = df["Date.of.Birth"].apply(_two_digit_year_to_full)
    if "DisbursalDate" in df.columns:
        df["DisbursalDate"] = df["DisbursalDate"].apply(_two_digit_year_to_full)
    if {"Date.of.Birth", "DisbursalDate"}.issubset(df.columns):
        df["Age"] = df["DisbursalDate"] - df["Date.of.Birth"]
        df = df.drop(columns=["DisbursalDate", "Date.of.Birth"])

    # The CNS-score description column ships with a free-text "<letter>-<bucket>"
    # format. Collapse to canonical buckets, then to ordinal integers.
    if "PERFORM_CNS.SCORE.DESCRIPTION" in df.columns:
        bucket_norm = {
            "C-Very Low Risk": "Very Low Risk", "A-Very Low Risk": "Very Low Risk",
            "D-Very Low Risk": "Very Low Risk", "B-Very Low Risk": "Very Low Risk",
            "M-Very High Risk": "Very High Risk", "L-Very High Risk": "Very High Risk",
            "F-Low Risk": "Low Risk", "E-Low Risk": "Low Risk", "G-Low Risk": "Low Risk",
            "H-Medium Risk": "Medium Risk", "I-Medium Risk": "Medium Risk",
            "J-High Risk": "High Risk", "K-High Risk": "High Risk",
        }
        df["PERFORM_CNS.SCORE.DESCRIPTION"] = df["PERFORM_CNS.SCORE.DESCRIPTION"].replace(
            bucket_norm
        )
        risk_to_int = {
            "No Bureau History Available": -1,
            "Not Scored: No Activity seen on the customer (Inactive)": -1,
            "Not Scored: Sufficient History Not Available": -1,
            "Not Scored: No Updates available in last 36 months": -1,
            "Not Scored: Only a Guarantor": -1,
            "Not Scored: More than 50 active Accounts found": -1,
            "Not Scored: Not Enough Info available on the customer": -1,
            "Very Low Risk": 4, "Low Risk": 3, "Medium Risk": 2,
            "High Risk": 1, "Very High Risk": 0,
        }
        df["PERFORM_CNS.SCORE.DESCRIPTION"] = (
            df["PERFORM_CNS.SCORE.DESCRIPTION"].map(risk_to_int)
        )

    if "AVERAGE.ACCT.AGE" in df.columns:
        df["AVERAGE.ACCT.AGE"] = df["AVERAGE.ACCT.AGE"].apply(_yrs_mons_to_months)
    if "CREDIT.HISTORY.LENGTH" in df.columns:
        df["CREDIT.HISTORY.LENGTH"] = df["CREDIT.HISTORY.LENGTH"].apply(_yrs_mons_to_months)

    return df


@_register_fix("0004.lendingclub")
def _fix_lendingclub(df: pd.DataFrame) -> pd.DataFrame:
    # No ID column, no leakage columns. Categoricals auto-inferred downstream.
    return df


@_register_fix("0005.myhom")
def _fix_myhom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])
    return df


@_register_fix("0006.hackerearth")
def _fix_hackerearth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = ["member_id", "batch_enrolled", "emp_title", "pymnt_plan",
                 "desc", "title", "zip_code"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if "emp_length" in df.columns:
        s = df["emp_length"].astype(str)
        s = s.replace("< 1 year", "0").replace("10+ years", "11").replace("10+", "11")
        s = s.str.replace(" years", "", regex=False).str.replace(" year", "", regex=False)
        df["emp_length"] = pd.to_numeric(s, errors="coerce")

    if "last_week_pay" in df.columns:
        s = df["last_week_pay"].astype(str)
        s = s.str.replace("th week", "", regex=False).replace("NA", np.nan)
        df["last_week_pay"] = pd.to_numeric(s, errors="coerce")

    target = DATASET_METADATA["0006.hackerearth"]["target_column"]
    if target in df.columns:
        df = df.dropna(subset=[target])

    return df


@_register_fix("0007.cobranded")
def _fix_cobranded(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(["na", "missing"], np.nan)
    if "application_key" in df.columns:
        df = df.drop(columns=["application_key"])
    if "mvar47" in df.columns:
        df["mvar47"] = df["mvar47"].replace({"C": 1, "L": 0})

    # Some columns are object-typed but mostly numeric; coerce best-effort
    # but only commit the coercion when the bulk of the column parses
    # successfully. Sanitize will catch genuine non-numeric columns and
    # treat them as categorical via the manifest.
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null_in = df[col].notna().sum()
            non_null_out = converted.notna().sum()
            if non_null_in > 0 and non_null_out / non_null_in >= 0.95:
                df[col] = converted
    return df


@_register_fix("0008.german")
def _fix_german(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # The OpenML German Credit CSV ships with no header — the first data
    # row gets misinterpreted as a header. Detect and rename.
    needs_rename = (
        list(df.columns)[0] in (0, "0")
        or "Unnamed: 0" in df.columns
        or "target" not in df.columns
    )
    if needs_rename:
        n_cols = df.shape[1]
        df.columns = [f"feature_{i}" for i in range(1, n_cols)] + ["target"]

    target = DATASET_METADATA["0008.german"]["target_column"]
    df = df.dropna(subset=[target])
    df[target] = df[target].replace({1: 0, 2: 1})
    return df


@_register_fix("0009.bank_status")
def _fix_bank_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all").reset_index(drop=True)
    df = df.drop(columns=[c for c in ["Loan ID", "Customer ID"] if c in df.columns])

    if "Loan Status" in df.columns:
        df["Loan Status"] = df["Loan Status"].replace({"Fully Paid": 0, "Charged Off": 1})
        df["Loan Status"] = pd.to_numeric(df["Loan Status"], errors="coerce")

    if "Term" in df.columns:
        df["Term"] = df["Term"].replace({"Short Term": 0, "Long Term": 1})
        df["Term"] = pd.to_numeric(df["Term"], errors="coerce")

    if "Home Ownership" in df.columns:
        df["Home Ownership"] = df["Home Ownership"].replace({
            "Own Home": 0, "Home Mortgage": 1, "HaveMortgage": 1, "Rent": 2,
        })
        df["Home Ownership"] = pd.to_numeric(df["Home Ownership"], errors="coerce")

    if "Purpose" in df.columns:
        df["Purpose"] = df["Purpose"].replace({
            "Debt Consolidation": 0, "Debt Consolidation Loan": 0,
            "Home Improvements": 1, "Home Improvement": 1,
            "Buy House": 2, "Buy a Car": 3, "major_purchase": 4,
            "Business Loan": 5, "small_business": 5,
            "Take a Trip": 6, "Vacation": 6, "Other": 7, "other": 7,
        })
        df["Purpose"] = pd.to_numeric(df["Purpose"], errors="coerce")

    if "Years in current job" in df.columns:
        s = df["Years in current job"].astype(str).replace("< 1 year", "0").replace("10+ years", "11").replace("10+", "11")
        s = s.str.replace(" years", "", regex=False).str.replace(" year", "", regex=False)
        df["Years in current job"] = pd.to_numeric(s, errors="coerce")

    return df.dropna(subset=["Loan Status"])


@_register_fix("0010.thomas")
def _fix_thomas(df: pd.DataFrame) -> pd.DataFrame:
    return df  # No surgery needed.


@_register_fix("0011.loan_default")
def _fix_loan_default(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 770+ feature columns in this dataset are mostly numeric but a small
    # fraction ship as object dtype with stray non-numeric tokens. Force
    # everything except the target to numeric, marking failures as NaN.
    target = DATASET_METADATA["0011.loan_default"]["target_column"]
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df = df.apply(pd.to_numeric, errors="coerce")

    # Binarise the target: any non-zero loss → default.
    if target in df.columns:
        df[target] = np.where(df[target].fillna(0) == 0, 0, 1).astype(np.int64)
    return df


@_register_fix("0012.home_credit")
def _fix_home_credit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])
    return df


@_register_fix("0013.hmeq")
def _fix_hmeq(df: pd.DataFrame) -> pd.DataFrame:
    return df  # No surgery needed.


@_register_fix("0014.algorithmwatch")
def _fix_algorithmwatch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target = DATASET_METADATA["0014.algorithmwatch"]["target_column"]
    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce").astype("Int64")
    return df


# ----- 0015 — Credit Risk Dataset (Kaggle) ----------------------------- #
# 32,581 rows × 12 columns. Target is `loan_status` ∈ {0, 1}. Two object
# columns are nominal categorical (`person_home_ownership`, `loan_intent`)
# and stay as strings — they're listed in DATASET_METADATA. The other
# two object columns need surgery:
#
#   * `loan_grade` ∈ {A, B, C, D, E, F, G} is an ORDINAL credit grade
#     (like S&P bond ratings: A is best, G is worst). We map it to
#     integers 0..6 here so that downstream models receive the
#     ordering. If we left it as a string and let `register.py`
#     auto-detect it as categorical, the downstream OrdinalEncoder
#     would use an arbitrary alphabetical mapping that happens to
#     coincide with the correct order — but only by coincidence. An
#     explicit mapping is more honest and protects us if the data
#     ever ships with non-letter grades.
#
#   * `cb_person_default_on_file` ∈ {Y, N} is a binary indicator;
#     we map to {1, 0}.
#
# `person_emp_length` has occasional sentinel-like values up to 123
# (years of employment); the data card doesn't document a meaning for
# values > 60, so we treat anything > 60 as NaN — the model's
# NanHandlingEncoder will pick it up downstream.

_LOAN_GRADE_ORDER = {g: i for i, g in enumerate(["A", "B", "C", "D", "E", "F", "G"])}


@_register_fix("0015.credit_risk_dataset")
def _fix_credit_risk_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "loan_grade" in df.columns:
        df["loan_grade"] = (
            df["loan_grade"].astype("object").map(_LOAN_GRADE_ORDER)
            .astype("Int64")
        )
    if "cb_person_default_on_file" in df.columns:
        df["cb_person_default_on_file"] = (
            df["cb_person_default_on_file"]
            .map({"Y": 1, "N": 0}).astype("Int64")
        )
    if "person_emp_length" in df.columns:
        df.loc[df["person_emp_length"] > 60, "person_emp_length"] = np.nan
    return df


# ----- 0016 — Bondora P2P loans (full European P2P platform dump) ------ #
# 737,889 rows × 31 columns. Target is `is_default` ∈ {True, False, NaN};
# we coerce to {1, 0} and drop NaN-target rows.
#
# CRITICAL: ~22 of the 31 columns are POST-LOAN data (payment progression,
# default-timeline indicators, current-state flags) that would not be
# available at loan origination. Using them would massively inflate
# the model's apparent performance. We strip them all. Specifically:
#
#   Pure leakage (drop): the loan-status duplicates and any column that
#                       is non-zero only AFTER default has occurred —
#                       e.g. `loan_status`, `loan_status_risk`,
#                       `principal_debt`, `late_fee_paid_total`,
#                       `months_in_default`, `has_default_within_12_months`.
#
#   Payment progression (drop): rolling balances and totals that are
#                       updated as the borrower pays — `principal_balance`,
#                       `principal_paid_total`, `interest_paid_total`,
#                       `extra_interest_paid_total`, `maintenance_fee_paid_total`,
#                       `repaid_amount_total`, `projected_npv_return`.
#
#   Timeline columns (drop): `loan_issued_at`, `early_repaid_at`,
#                       `is_early_repaid_within_14_days`,
#                       `loan_last_recorded_action_date_local`,
#                       `next_payment_nr`, `next_payment_date_local`,
#                       `debt_occured_date_local`, `days_past_due_principal`,
#                       `months_on_book`. (Dates are also free-form strings
#                       we'd otherwise have to parse.)
#
#   ID (drop): `loan_id` (UUID).
#
# What's left as "safe-at-origination" features:
#   country, issued_amount, initial_interest_rate, nr_of_payments,
#   initial_loan_duration, combined_income, customer_risk_rating, is_default
#
# This honest treatment yields a small (~7-feature) but uncontaminated
# dataset. Reviewers will trust this; "Bondora got AUC=0.99" by including
# `months_in_default` they will not.

_BONDORA_LEAKAGE_COLS = [
    "loan_id",
    "loan_issued_at",
    "early_repaid_at",
    "is_early_repaid_within_14_days",
    "loan_status",                          # categorical default indicator
    "loan_last_recorded_action_date_local",
    "principal_balance",
    "principal_debt",
    "principal_paid_total",
    "interest_paid_total",
    "extra_interest_paid_total",
    "late_fee_paid_total",
    "maintenance_fee_paid_total",
    "next_payment_nr",
    "next_payment_date_local",
    "debt_occured_date_local",
    "days_past_due_principal",
    "months_in_default",
    "months_on_book",
    "loan_status_risk",
    "repaid_amount_total",
    "has_default_within_12_months",
    "projected_npv_return",
]


@_register_fix("0016.bondora_peer2peer")
def _fix_bondora_peer2peer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target = DATASET_METADATA["0016.bondora_peer2peer"]["target_column"]
    # Coerce {True, False, "True", "False", "TRUE", "FALSE"} → {1, 0}.
    if target in df.columns:
        df[target] = (
            df[target].astype(str).str.strip().str.lower()
            .map({"true": 1, "false": 0}).astype("Int64")
        )
        df = df[df[target].notna()].copy()
    df = df.drop(
        columns=[c for c in _BONDORA_LEAKAGE_COLS if c in df.columns],
    )
    return df


# ----- 0017 — SBA Loans Case (Kaggle; doubles as 0008.SBA for LGD) ----- #
# 2,102 rows × 35 columns. The CSV in this folder ships with currency
# columns already coerced to plain integers (no '$' or ',') — unlike
# some other public mirrors of the same dataset. The surgery here is
# pure column drops:
#
#   IDs / free text (drop):  LoanNr_ChkDgt, Name, Bank, City, Zip
#                            (Zip is a 5-digit US ZIP — too high
#                             cardinality for TabPFN; State/BankState
#                             stay as the 2-letter geo features.)
#
#   Sampling artefact (drop): `Selected` — a 0/1 flag indicating an
#                             evenly-sampled 50/50 subset. Not a
#                             credit-risk feature; would leak the
#                             sampling design.
#
#   Mystery / redundant (drop): `xx` is exactly `DisbursementDate +
#                             daysterm` (the loan maturity date) — a
#                             pure derived feature.
#
#   Leakage (drop): `MIS_Status` ∈ {"P I F", "CHGOFF"} is a 1-to-1
#                   reflection of `Default`; `ChgOffDate` is only
#                   non-null when the loan defaulted; `ChgOffPrinGr`
#                   is the charge-off principal (component of the
#                   LGD target — but a strong leakage proxy for PD too).
#
#   Dirty categorical levels (re-encode): `RevLineCr` has stray
#                   values {Y, N, 0, T} — 0 maps to N (consistent
#                   with the SBA codebook), T is undocumented → NaN.
#                   `LowDoc` has stray {Y, N, S, A, 0} — S/A/0 → NaN.

_SBA_DROP_COLS_COMMON = [
    "LoanNr_ChkDgt", "Name", "Bank", "City", "Zip",
    "Selected", "xx",
    "MIS_Status", "ChgOffDate",
]


def _clean_sba_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    if "RevLineCr" in df.columns:
        df["RevLineCr"] = (
            df["RevLineCr"].astype(str).str.strip()
            .map({"Y": "Y", "N": "N", "0": "N"})    # T / others → NaN
        )
    if "LowDoc" in df.columns:
        df["LowDoc"] = (
            df["LowDoc"].astype(str).str.strip()
            .map({"Y": "Y", "N": "N"})              # S / A / 0 → NaN
        )
    return df


@_register_fix("0017.SBA_loans_case")
def _fix_sba_pd(df: pd.DataFrame) -> pd.DataFrame:
    """PD copy: target = `Default` (0/1, already encoded). Drop the
    leakage / ID / mystery columns plus `ChgOffPrinGr` (the LGD-target
    component would otherwise leak into the PD model)."""
    df = df.copy()
    drop_cols = _SBA_DROP_COLS_COMMON + ["ChgOffPrinGr"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = _clean_sba_categoricals(df)
    return df


@_register_fix("0001.heloc")
def _fix_heloc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = ["REC", "DLGD_Econ", "PrinBal", "PayOff", "DefPayOff",
                 "ObsDT", "DefDT", "DefPrinBal"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if "LienPos" in df.columns:
        df["LienPos"] = df["LienPos"].replace({"Unknow": 0, "First": 1, "Second": 2})
        df = df.infer_objects()

    target = DATASET_METADATA["0001.heloc"]["target_column"]
    if target in df.columns:
        # HELOC is shipped with artefactual exact-duplicate rows that are
        # NOT independent observations. Drop in-place by identical feature
        # vectors. (Dataset-specific quirk; not the same as cross-dataset
        # dedup.)
        feat = [c for c in df.columns if c != target]
        before = len(df)
        df = df.drop_duplicates(subset=feat, keep="first").reset_index(drop=True)
        n = before - len(df)
        if n:
            LOGGER.info("heloc: dropped %d artefactual duplicate rows", n)
    return df


@_register_fix("0002.loss2")
def _fix_loss2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [
        "_ELGDnum1", "_ELGDnum2", "id1", "Alltel_Client",
        "REO_Appraisal_Date", "Origination_Date", "date_vintage_year",
        "date_vintage_year_month", "Servicing_Loss", "lr1", "lss_rt",
        "_Loss_Amount", "lss_amt", "Investor_Category", "_Proceeds",
        "_Net_sales_Proceeds", "_reo_sales_price", "_SellingCosts",
        "REO_Sales_Price",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    target = DATASET_METADATA["0002.loss2"]["target_column"]
    if target in df.columns:
        df = df.dropna(subset=[target])

        feat = [c for c in df.columns if c != target]
        before = len(df)
        df = df.drop_duplicates(subset=feat, keep="first").reset_index(drop=True)
        n = before - len(df)
        if n:
            LOGGER.info("loss2: dropped %d artefactual duplicate rows", n)
    return df


@_register_fix("0003.axa")
def _fix_axa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    leakage = ["Recovery_rate", "y_logistic", "lnrr", "Y_probit", "event"]
    df = df.drop(columns=[c for c in leakage if c in df.columns])
    return df


@_register_fix("0004.base_model")
def _fix_base_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Wide column dump from a relational LGD pipeline; drop everything that
    # is either an internal ID, a temporal field, or an LGD-derived label.
    drop_cols = [
        # Identifiers
        "DEAL_DocUNID", "DEAL_MainID", "DEAL_FacilityIdentifier",
        "DEAL_StarWebIdentifier", "DFLT_MainID", "DFLT_SPM", "DFLT_DAI",
        "DFLT_BDR", "DFLT_LegalEntityName", "DFLT_StarWeb_PCRU",
        "DFLT_ClientNAE", "DFLT_ParentSPM", "DFLT_ParentSIREN",
        "DFLT_ParentDAI", "DFLT_ParentLegalEntityName", "DFLT_ParentPCRU",
        "DFLT_ParentNAE", "DFLT_subject", "FCLT_DealUNID",
        "FCLT_BCEIdentifier", "FCLT_Identifier", "FCLT_BookingUnit",
        "fclt_docunid",
        # Dates / temporal leakage
        "DEAL_TransactionStartDate", "DEAL_TransactionEndDate",
        "DEAL_DateComposed", "DEAL_LastUpDate", "DFLT_SGDefaultDate",
        "DFLT_PublicDefaultDate", "DFLT_EndDefaultDate", "DFLT_SGRatingDate",
        "DFLT_RatingDate1YPD", "DFLT_ParentDefaultDateIf",
        "DFLT_ParentSGRatingDate", "DFLT_ParentRatingDate1YPD",
        "DFLT_DateComposed", "DFLT_LastUpdate", "DATE_DECLAR_CT",
        "FCLT_StartDate", "FCLT_EndDate", "FCLT_DefaultDate",
        "FCLT_DateComposed", "FCLT_LastUpdate", "date",
        "DEAL_ConstructionEndDate", "DEAL_ConstructionStartDate",
        "DEAL_StatusUpDate", "DEAL_DeleteDate", "DFLT_DeletedDate",
        "FCLT_DeleteDate",
        # Free-text columns
        "FCLT_CommentsOnLimit", "FCLT_subject", "DEAL_GoverningLawRecovery",
        "DEAL_PFRU", "DEAL_subject",
        # Target-derived / leakage labels
        "lgd_cat_15", "lgd_cat_10", "lgd_cat_5", "LGD_log", "LGD_deF",
        "LGD_norm", "sortie", "RecAssoFlag",
        # Misc deletion / housekeeping flags
        "DEAL_AverageRents", "DEAL_ExpectedVacancyRate",
        "DEAL_StrikeLESSEEOption", "DFLT_DeleteStatus", "DFLT_JRIRating",
        "DFLT_ParentJRIRating", "FCLT_DeleteStatus",
        "FCLT_IrrevocableLocOffshore", "flag_eps", "flag_fcltcurrency",
        "fac_ss_commcov", "flag_pme", "Flag_specifique", "flag_specperi",
        "Categorie_AV",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    target = DATASET_METADATA["0004.base_model"]["target_column"]
    if target in df.columns:
        df = df.dropna(subset=[target])
    return df


@_register_fix("0005.base_modelisation")
def _fix_base_modelisation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in ["Ident_cliej_spm", "ID_CONC_ORIGIN_CDL",
                                       "id_crc", "id_unique"]
                          if c in df.columns])
    leakage = [
        "lgd_5_sscout_ligne", "lgd_corr", "lgd_defaut_nt", "lgd_3class",
        "lgd_2class", "lgd_log", "lgd_t", "lgd_1log", "logit_lgd",
        "Dt_entree_defaut", "Dt_sortie_defaut", "flag_defaut_moins_1an",
        "auto_av_defaut", "util_av_defaut", "defaut_clos",
        "defaut_clos_4nonclos", "duree_1A_av_defaut", "util_av_defaut_tot",
        "auto_av_defaut_tot", "defaut_M1Y", "defaut_P1Y",
    ]
    df = df.drop(columns=[c for c in leakage if c in df.columns])

    target = DATASET_METADATA["0005.base_modelisation"]["target_column"]
    if target in df.columns:
        df = df.dropna(subset=[target])
    return df


@_register_fix("0006.lgd_freddie")
def _fix_lgd_freddie(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    target = DATASET_METADATA["0006.lgd_freddie"]["target_column"]
    if target in df.columns:
        df = df.dropna(subset=[target])
    return df


@_register_fix("0007.lgd_lendingclub")
def _fix_lgd_lendingclub(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in ["addr_state", "purpose", "id"]
                          if c in df.columns])

    target = DATASET_METADATA["0007.lgd_lendingclub"]["target_column"]
    if target in df.columns:
        df = df.dropna(subset=[target])
    return df


# ----- 0008 — SBA Loans Case (LGD twin of 0017.SBA_loans_case) --------- #
# Same CSV as `0017.SBA_loans_case` in the PD track; the raw file lives
# in BOTH raw/pd/ AND raw/lgd/ so each track can apply its own surgical
# fix to a private copy. The LGD-side fix:
#
#   1. Filters to defaulted loans only (Default == 1) — LGD is only
#      defined for loans that actually defaulted.
#   2. Derives the LGD target = clip(ChgOffPrinGr / DisbursementGross,
#      0, 1). The clip handles the ~handful of cases where the charge-
#      off is reported larger than disbursement (data entry artefacts).
#   3. Drops the same ID / leakage / mystery columns the PD fix drops,
#      plus the two target components (ChgOffPrinGr is the LGD
#      numerator; DisbursementGross stays as a feature — its size IS
#      predictive of LGD even though it appears in the target ratio).
#   4. Drops the original `Default` column (also leakage for LGD: only
#      the defaulted rows are kept, so it's a constant 1).

@_register_fix("0008.SBA_loans_case")
def _fix_sba_lgd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Filter to defaulted loans.
    if "Default" in df.columns:
        df = df[df["Default"] == 1].copy()

    # 2. Derive the LGD target ratio. Clipping to [0, 1] is the job of
    # sanitize.py's global `lgd_target_clip` — every LGD dataset goes
    # through the same clip, so no per-dataset clipping logic here.
    disbursement = pd.to_numeric(df.get("DisbursementGross"), errors="coerce")
    chargeoff    = pd.to_numeric(df.get("ChgOffPrinGr"),      errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lgd"] = chargeoff / disbursement

    # 3. Drop ID / leakage / mystery / target-component / now-constant.
    drop_cols = _SBA_DROP_COLS_COMMON + [
        "Default",        # always 1 in the filtered dataset
        "ChgOffPrinGr",   # numerator of the target — leakage
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 4. Re-encode the same dirty categorical levels as the PD twin.
    df = _clean_sba_categoricals(df)

    return df


# =============================================================================
# Bulk metadata import (helper for the 3000-dataset case)
# =============================================================================


def register_from_records(records: list[dict]) -> None:
    """Bulk-add metadata entries from a vendor-provided list of dicts.

    Each record must have at minimum ``dataset_id``, ``track``,
    ``task_type``, and ``target_column``; ``categorical_columns``,
    ``source``, and ``source_url`` are optional and default to empty.

    The intended use is one-shot at module load time when scaling
    beyond the hand-coded corpus, e.g.

    .. code-block:: python

        import json
        with open("vendor_metadata.json") as f:
            register_from_records(json.load(f))

    Re-freezes ``DATASET_METADATA`` after insertion. **Never** mutate
    metadata after it has been read by other modules.
    """
    global DATASET_METADATA
    for r in records:
        did = r["dataset_id"]
        if did in _RAW_METADATA:
            raise ValueError(f"register_from_records: {did!r} already known")
        _RAW_METADATA[did] = {
            "track":               r["track"],
            "task_type":           r["task_type"],
            "target_column":       r["target_column"],
            "categorical_columns": list(r.get("categorical_columns", [])),
            "source":              r.get("source", "local"),
            "source_url":          r.get("source_url"),
        }
    DATASET_METADATA = MappingProxyType(
        {k: MappingProxyType(v) for k, v in _RAW_METADATA.items()}
    )


def _validate_consistency() -> None:
    """Sanity-check at module load: every fix function targets a known
    dataset_id."""
    extras = set(_FIX_FUNCTIONS) - set(_RAW_METADATA)
    if extras:
        raise RuntimeError(
            f"src.data.preprocessing: fix function(s) registered for "
            f"unknown dataset_id(s): {sorted(extras)}"
        )


_validate_consistency()


def apply_dataset_specific_fixes(
    df: pd.DataFrame,
    dataset_id: str,
    *,
    unknown_dataset_policy: str = "error",
) -> pd.DataFrame:
    """Apply the registered surgical fix for ``dataset_id``.

    Parameters
    ----------
    df
        Raw DataFrame as loaded from the source CSV.
    dataset_id
        e.g. ``"0001.gmsc"`` or ``"0004.base_model"``.
    unknown_dataset_policy
        ``"error"`` (default) raises ``KeyError`` for unregistered IDs;
        ``"passthrough"`` returns the input unchanged.

    Returns
    -------
    pandas.DataFrame
        New DataFrame (the fix functions copy on entry; the input is
        never mutated).
    """
    fix = _FIX_FUNCTIONS.get(dataset_id)
    if fix is None:
        if unknown_dataset_policy == "passthrough":
            return df
        raise KeyError(
            f"No surgical-fix function registered for dataset_id="
            f"{dataset_id!r}. Add one in src.data.preprocessing."
        )
    return fix(df)


# =============================================================================
# CLI smoke-test entry point
# =============================================================================


def _load_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load("config/data.yaml")


def main(cfg=None) -> int:
    """Smoke-test only — see module docstring."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if cfg is None:
        cfg = _load_cfg()
    raw_root = cfg.paths.raw

    failures = 0
    for dataset_id, meta in DATASET_METADATA.items():
        path = f"{raw_root}/{meta['track']}/{dataset_id}.csv"
        try:
            df = pd.read_csv(path, low_memory=False)
            fixed = apply_dataset_specific_fixes(df, dataset_id)
            target = meta["target_column"]
            target_present = target in fixed.columns
            LOGGER.info(
                "%-26s raw=%s post-fix=%s target=%s present=%s",
                dataset_id, df.shape, fixed.shape, target, target_present,
            )
            if not target_present:
                LOGGER.warning("  → target column missing after fix")
                failures += 1
        except FileNotFoundError:
            LOGGER.warning("missing raw file: %s — skipped", path)
        except Exception as exc:
            LOGGER.error("%s failed: %s", dataset_id, exc, exc_info=True)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
