# Data pipeline — what happens to a credit-risk dataset at every stage

This document walks one (raw CSV) → (TabPFN-ready tensor) trip through
the CreditPFN pipeline. It exists because the journey is long: there
are five distinct transformations, three on-disk artefacts per dataset,
and two different downstream consumers (TabPFN training and the
classical-baseline eval pipeline) that need DIFFERENT preprocessing.

Read this end-to-end the first time. The "Quick reference" at the
bottom is for revisits.

---

## Stage map

```
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 0 — Raw                                                    │
│  data/raw/{pd,lgd}/<id>/raw.csv                                   │
│  User-supplied. Free-form. Strings, numbers, NaNs, junk columns.  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │  scripts/data_pipeline.py
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Dedup PRE pass (diagnostic only)                       │
│  src/data/dedup.py --pass pre                                     │
│  Detects within-track dataset duplicates BEFORE any cleaning.     │
│  Writes a report; does not remove anything.                       │
│  → data/dedup/doubles_{pd,lgd}_pre.csv                            │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Register                                               │
│  src/data/register.py                                             │
│  Reads each raw CSV. Applies surgical fixes (drop ID columns,     │
│  parse bespoke string formats, decode "5yrs 3mon" → months,       │
│  remove leakage columns). Computes per-dataset metadata.          │
│  → data/manifest_{pd,lgd}.csv                                     │
│    (one row per dataset: n_rows, n_cols, missing rate, class      │
│     balance, target mean/std, content-aware shape hash)           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Sanitize  ← This is the heart of the data pipeline    │
│  src/data/sanitize.py                                             │
│  Per-dataset, dataset-agnostic cleaning. See "Stage 3 in detail". │
│  → data/processed/{pd,lgd}/<id>.sanitized.csv                     │
│    (sometimes a small feature_groups.json sidecar for the         │
│     FeatureAgglomeration step)                                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Dedup POST pass (diagnostic only)                      │
│  src/data/dedup.py --pass post                                    │
│  Catches duplicates that only become identical after sanitize.    │
│  → data/dedup/doubles_{pd,lgd}_post.csv                           │
└──────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            │                                    │
            ▼                                    ▼
┌──────────────────────┐               ┌─────────────────────────────┐
│ STAGE 5a — TabPFN    │               │  STAGE 5b — Baselines       │
│ continued pretraining│               │  XGBoost / CatBoost /       │
│ (src/train/*)        │               │  LogReg / LinReg            │
│                      │               │  (src/eval/* baseline path) │
│ Reads sanitized CSV. │               │  Reads sanitized CSV.       │
│ Then per epoch:      │               │                             │
│  • clean_data        │               │  • Numeric columns as-is    │
│    (cached/dataset)  │               │  • Categoricals: label-     │
│  • TabPFNEnsemble-   │               │    encoded (XGB) or native- │
│    Preprocessor      │               │    cat (CatBoost)           │
│    (per step)        │               │                             │
│  • outlier soft-clip │               │  NO TabPFN-style transforms │
│  • model.forward     │               │  (no squashing scaler,      │
│                      │               │   no quantile, no SVD).     │
└──────────────────────┘               └─────────────────────────────┘
```

The split at the bottom is the most important design choice in the
project. Both TabPFN and the boosting baselines consume the SAME
`data/processed/.../*.sanitized.csv` — so they see the same input data
(fair comparison). They then each apply the preprocessing that's RIGHT
for them: TabPFN's pretraining requires its specific
ensemble-preprocessor stack; XGBoost/CatBoost want the raw values.

---

## Stage 3 in detail — what `sanitize.py` actually does

This is where most of the work happens. Steps in order, per dataset:

### 3.1 Surgical per-dataset fixes (from `DATASET_METADATA`)
Hand-written in `src/data/preprocessing.py`. Examples:
- `gmsc`: drop the leading row-number column.
- `lendingclub`: parse `"5yrs 3mon"` to integer months in
  `emp_length`.
- `home_credit`: drop columns flagged as label-leakage from the
  Kaggle competition writeups.
- `heloc`: drop ~10 k artefactual duplicate rows.

Surgical fixes are dataset-specific and pre-registered — they do NOT
change between runs.

### 3.2 Drop columns
- Exact-duplicate columns (data, not name): keep first.
- Columns that are > 90 % NaN.
- Constant columns (single unique value across the whole dataset).

### 3.3 Coerce numeric strings
Any column that pandas inferred as `object` but where every value
parses as a number → cast to `float`. `NaN` for unparseable rows.

### 3.4 Cast to float32
After string coercion, numerical columns are cast to `float32`.
Out-of-range values become `NaN` BEFORE the cast (no overflow warnings;
we explicitly mask them first — see `_safe_cast_to_float32`).

### 3.5 Replace `±inf` with `NaN`
TabPFN's downstream encoder treats `NaN` natively; `inf` it cannot.

### 3.6 FeatureAgglomeration (only when n_features > 128)
When a dataset has more than 128 features (e.g. `loan_default` with
769 cols, `algorithmwatch` with 2986 cols), we apply
`sklearn.cluster.FeatureAgglomeration` to the numerical columns,
clustering them into ≤ 128 clusters via Ward linkage. **Categoricals
are kept unchanged** — agglomeration only operates on numerics. The
cluster assignments are written to a sidecar
`<id>.sanitized.feature_groups.json` for reproducibility.

This is the mechanism that lets a 2986-column dataset fit through
TabPFN's hard limit of 2000 features per estimator (TabPFN-2.5 paper,
appendix B).

### 3.7 Label-encode classification targets
Map raw target labels to `{0, ..., K-1}`. Sorted lexicographically so
the encoding is stable across runs.

### 3.8 Clip LGD targets to [0, 1]
LGD = loss given default = fraction of exposure lost. By definition
in [0, 1]. We clip values outside this range (typical: a few rows
with 1.02 or 1.05 from accounting roundoff, a few with −0.001 from
recovery > exposure edge cases) and log the count.

### 3.9 Save
- `data/processed/<track>/<id>.sanitized.csv` — the final on-disk
  artefact.
- The target column is in there; downstream code reads it via
  `DATASET_METADATA[id]["target_column"]`.

---

## Stage 5a — TabPFN-side per-step preprocessing

The sanitized CSV is the input to TabPFN training but is **not** what
the model forward pass sees. Per training step the
`ProcessedDatasetLoader.__getitem__` runs an additional preprocessing
pipeline that mirrors the official TabPFN finetune
(`repositories/TabPFN .txt:26147-26319`):

### 5a.1 `clean_data` (once per dataset, cached)
TabPFN's own `clean_data(X, feature_schema)` is invoked ONCE per
parent dataset in the training process (cached in
`src/train/tabpfn_preprocessing.py::_CLEAN_CACHE`). It:
- Calls `fix_dtypes` to ensure all columns are numeric (ordinal-
  encoding string categoricals to integer codes).
- Calls `process_text_na_dataframe` to handle NA values.
- Returns a numeric numpy array + `FeatureSchema`.

This step matches TabPFN's `_initialize_dataset_preprocessing`
(line 7686-7733 of the dump) and is the reason categorical-as-string
columns work without us having to pre-encode them in `sanitize.py`.

### 5a.2 EnsembleConfig generation (once per dataset, cached)
`generate_classification_ensemble_configs` /
`generate_regression_ensemble_configs` builds the per-estimator
configuration: one of N preprocessor configs (squashing scaler /
quantile / none), one of N feature shifts, one of N class permutations.
For `n_estimators_finetune=2` we get 2 distinct configs per dataset.
These are **stable per dataset across all epochs** — matches the
published behaviour at `TabPFN .txt:26604-26635`.

### 5a.3 Per-step subsample + ctx/query split
Per training step (= per dataset visited within an epoch):
- Draw a fresh stratified subsample of up to
  `max_rows_per_epoch` rows from the cleaned numeric array.
- Split into context (1 − qf fraction) and query (qf fraction).
- For LGD, z-normalize the target on context-only statistics
  (clamping std to 1e-8 if degenerate — mirrors the official path).

### 5a.4 `TabPFNEnsemblePreprocessor.fit_transform_ensemble_members`
Per step we instantiate a fresh `TabPFNEnsemblePreprocessor` with the
cached `ensemble_configs`, fit it on the context split, and obtain
N preprocessed views — each potentially with different feature counts
(SVD/polynomial add columns), different rows (subsampling), and
different label encodings (class-permutation augmentation).
`member.transform_X_test(X_query)` applies the same per-member
pipeline to the query split.

### 5a.5 Outlier soft-clip (just before model forward)
TabPFN's GPU step `TorchSoftClipOutliersStep` (`TabPFN .txt:35959-35967`)
soft-clips numerical columns to ±12σ for the classifier (None for
regressor). We invoke it from `_forward_one_member` immediately before
the model call. Categorical columns pass through unmodified.

### 5a.6 Model forward + loss
`PerFeatureTransformer.forward(combined_x, train_y, categorical_inds, …)`
returns logits over query positions only. CE loss (PD) or
bar-distribution NLL (LGD) is computed against the canonical-order
target, with class-permutation unscramble for classifier members that
were trained on permuted labels.

---

## Stage 5b — baselines path

The classical baselines (XGBoost, CatBoost, LogReg, LinReg) operate
on the sanitized CSV with minimal further preprocessing:

- **XGBoost / LogReg / LinReg**: categoricals are label-encoded
  (deterministically per CV fold via sklearn `OrdinalEncoder` fit on
  the train fold). Numerics passed through.
- **CatBoost**: native categorical handling (`cat_features` parameter
  carries the positional indices). Numerics passed through.

These models do **not** see TabPFN's squashing scaler / SVD / fingerprint
pipeline. By design — that pipeline is specific to TabPFN's pretrained
weight expectations, not a general-purpose preprocessor.

Reads: `data/processed/<track>/<id>.sanitized.csv` (same file as
TabPFN). The eval pipeline (`src/eval/dataset_loader.py`)
deterministically splits this into outer K-folds for CV.

---

## Why two preprocessors

TabPFN was pretrained on synthetic tasks pre-processed with squashing
scaler + quantile transforms + SVD. To get the published performance
out of it, inference time must apply the **same** preprocessing. So
TabPFN's sklearn API does it automatically inside `predict_proba`.

XGBoost / CatBoost / LogReg / LinReg expect raw values (XGB and LR)
or label-encoded values (LR for categoricals, XGB for categoricals).
Forcing TabPFN's squashing scaler on them would actually HURT their
performance — they're designed to handle outliers via tree splits or
regularization, not soft-clipping.

The clean separation in our code:
- TabPFN's preprocessing lives in `src/train/tabpfn_preprocessing.py`
  and the inference-time equivalent inside `TabPFNClassifier`.
- Baselines preprocess in their `fit()` methods (`src/model/boosting.py`,
  `src/model/linear.py`).

The training data is identical (same sanitized CSV); only the
downstream transformations differ.

---

## Quick reference

| Stage | Module | Reads | Writes |
|---|---|---|---|
| 1 | `src/data/dedup.py --pass pre`  | `data/raw/{pd,lgd}/<id>/raw.csv` | `data/dedup/doubles_<track>_pre.csv` |
| 2 | `src/data/register.py`          | `data/raw/{pd,lgd}/`             | `data/manifest_{pd,lgd}.csv` |
| 3 | `src/data/sanitize.py`          | `data/raw/` + manifest           | `data/processed/{pd,lgd}/<id>.sanitized.csv` |
| 4 | `src/data/dedup.py --pass post` | `data/processed/`                | `data/dedup/doubles_<track>_post.csv` |
| 5a | `src/train/dataloader.py` + `src/train/tabpfn_preprocessing.py` | `data/processed/.../*.sanitized.csv` | Live tensors (no disk artefact) |
| 5b | `src/eval/benchmark.py` + `src/model/{boosting,linear,tabpfn_models}.py` | `data/processed/.../*.sanitized.csv` | Eval CSVs at `output/results/...` |

### Resume semantics
See [README.md § 5](../README.md#5-re-submitting-the-pipeline-resume-semantics--cleanup). The data stage is idempotent — it
skips datasets whose sanitized CSV is already on disk.

### Cleanup
`python -m src.utils.pipeline_clean --stages data` wipes everything
the data pipeline produces (processed CSVs, dedup reports, manifests,
data logs) but **never** touches `data/raw/`.

---

## Common gotchas

- **"I changed a surgical fix in `DATASET_METADATA`; why didn't it
  apply?"** — the sanitized CSV is already on disk. The data pipeline
  saw it and skipped. Run `pipeline_clean --stages data` first.

- **"My categorical column became numeric in the sanitized CSV"** —
  `sanitize.py:3.3` (coerce numeric strings) will cast a categorical
  if every value happens to parse as a number. Mark the column as
  categorical in `DATASET_METADATA[id]["categorical_columns"]` so
  TabPFN's `clean_data` re-encodes it correctly.

- **"FeatureAgglomeration ran on a dataset with < 128 features"** —
  it shouldn't. Check the log line; if it did, the dataset has
  hidden duplicate columns inflating the count.

- **"LGD training has negative losses"** — by design. See
  [README.md § 7 design notes](../README.md#design-notes-the-why) and the
  bar-distribution NLL discussion. Negative NLL means the model has
  placed sharp probability mass on the true target — a good sign.
