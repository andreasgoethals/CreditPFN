# Method — corpus, models, capacity, and the decisions behind them

Everything you need to trust a number from this project, in the order you need it:
what happens to a dataset before a model sees it, which base checkpoints are swept and
why, how many rows each model gets per step and per eval fold, and the implementation
choices that look like bugs and are deliberate.

What each run *measured* is [`RESULTS.md`](RESULTS.md). What was tried and failed is
[`AGENTS_MEMORY.md`](AGENTS_MEMORY.md). How to run it on the cluster is
[`VSC.md`](VSC.md).

Merged on 12-08-2026 from four documents — `METHOD.md`, `METHOD.md`,
`METHOD.md` and `METHOD.md` — which split along file boundaries rather than
along topics: the row caps were half data-pipeline and half checkpoint, and the code
notes belonged to all four.

---

## Contents

1. [The data pipeline](#1-the-data-pipeline) — raw CSV to training input
2. [Base checkpoints](#2-base-checkpoints) — what is swept, and their licences
3. [Context-size caps](#3-context-size-caps) — measured rows per step and per fold
4. [Deliberate oddities](#4-deliberate-oddities) — code that looks wrong and is not

---

## 1. The data pipeline

This document walks one (raw CSV) → (TabPFN-ready tensor) trip through
the CreditPFN pipeline. It exists because the journey is long: there
are five distinct transformations, three on-disk artefacts per dataset,
and two different downstream consumers (TabPFN training and the
classical-baseline eval pipeline) that need DIFFERENT preprocessing.

The whole data stage runs on a single **wICE CPU node** (`batch`
partition, no GPU — see `scripts/slurm/data.slurm` and
[docs/VSC.md](VSC.md)). It is the only stage that does not
need a GPU.

Read this end-to-end the first time. The "Quick reference" at the
bottom is for revisits.

> **Where the artefacts live (VSC).** On the cluster the relative paths
> below are NOT all on the same storage tier. By default
> (`config/data.yaml → paths.data_source: "staging"`) the bulky corpus
> — `data/raw/` and the sanitized `data/processed/` CSVs — lives in
> **project staging** `<staging>/CreditPFN` (large, persistent, the one
> tier both wICE and Mindwell can see). The smaller, durable
> diagnostics — the dedup reports (`output/manifests/dedup/`) and the manifests
> (`output/manifests/manifest_{pd,lgd}.csv`) — ALWAYS resolve to the **output root**
> `$VSC_DATA/CreditPFN` (NFS, backed up). `data_source` can be flipped to
> `"scratch"` (`$VSC_SCRATCH/CreditPFN`, purged ~30 d) or `"data"`
> (`$VSC_DATA/CreditPFN`); on a laptop the knob is ignored and everything
> sits under the repo root. See [docs/VSC.md](VSC.md) for the
> storage-tier rationale.

---

### Stage map

```
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 0 — Raw                                                    │
│  data/raw/{pd,lgd}/<id>.csv        [staging on VSC]               │
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
│  → output/manifests/dedup/doubles_{pd,lgd}_pre.csv   [output root, durable]   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Register                                               │
│  src/data/register.py                                             │
│  Reads each raw CSV. Applies surgical fixes (drop ID columns,     │
│  parse bespoke string formats, decode "5yrs 3mon" → months,       │
│  remove leakage columns). Computes per-dataset metadata.          │
│  → output/manifests/manifest_{pd,lgd}.csv [output root, durable]  │
│    (one row per dataset: n_rows, n_cols, missing rate, class      │
│     balance, target mean/std, content-aware shape hash)           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Sanitize  ← This is the heart of the data pipeline    │
│  src/data/sanitize.py                                             │
│  Per-dataset, dataset-agnostic cleaning. See "Stage 3 in detail". │
│  → data/processed/{pd,lgd}/<id>.sanitized.csv  [staging on VSC]   │
│    (the sanitized CSV is the only artefact; feature selection      │
│     keeps real columns, so no synthetic-feature sidecar)          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Dedup POST pass (diagnostic only)                      │
│  src/data/dedup.py --pass post                                    │
│  Catches duplicates that only become identical after sanitize.    │
│  → output/manifests/dedup/doubles_{pd,lgd}_post.csv  [output root, durable]   │
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

### Stage 3 in detail — what `sanitize.py` actually does

This is where most of the work happens. Steps in order, per dataset:

#### 3.1 Surgical per-dataset fixes (from `DATASET_METADATA`)
Hand-written in `src/data/preprocessing.py`. Examples:
- `gmsc`: drop the leading row-number column.
- `lendingclub`: parse `"5yrs 3mon"` to integer months in
  `emp_length`.
- `home_credit`: drop columns flagged as label-leakage from the
  Kaggle competition writeups.
- `heloc`: drop ~10 k artefactual duplicate rows.

Surgical fixes are dataset-specific and pre-registered — they do NOT
change between runs.

#### 3.2 Drop columns
- Exact-duplicate columns (data, not name): keep first.
- Columns that are > 90 % NaN.
- Constant columns (single unique value across the whole dataset).

#### 3.3 Coerce numeric strings
Any column that pandas inferred as `object` but where every value
parses as a number → cast to `float`. `NaN` for unparseable rows.

#### 3.4 Cast to float32
After string coercion, numerical columns are cast to `float32`.
Out-of-range values become `NaN` BEFORE the cast (no overflow warnings;
we explicitly mask them first — see `_safe_cast_to_float32`).

#### 3.5 Replace `±inf` with `NaN`
TabPFN's downstream encoder treats `NaN` natively; `inf` it cannot.

#### 3.6 Feature selection (only when n_features > `sanitize.max_columns`)
When a dataset has more numerical features than the cap
(`sanitize.max_columns`, currently **64**; e.g. `loan_default` 768,
`home_credit` 119, `algorithmwatch` 2985, the LGD `base_model*` ~255),
we **select a subset of the real columns** rather than averaging them:
keep the top-`max_columns` by scale-free (min-max-normalised) variance,
after greedily dropping columns whose `|Pearson r|` with an
already-kept column exceeds `feature_selection.corr_threshold` (0.95).
**Categoricals are kept unchanged.** The kept columns are *original*
features with their *real* distributions — so continued pretraining
specialises the prior toward genuine credit features.

Why selection, not the previous `FeatureAgglomeration`: cluster means
are synthetic averaged columns with smoothed distributions and no
real-world meaning, which works against the goal of adjusting the prior
to real credit data (and is inconsistent with what the model sees at
inference). It is **unsupervised** (never touches `y`) → no label leak.

**On the cap value.** Real-TabPFN (Garg 2025) doesn't reduce features
at all — it curates datasets to ≤ 500 features and caps each at 400 000
*cells* (trimming rows, not columns). TabPFN-v2's documented sweet spot
is ≤ 100 features; v2.6 / v3 handle up to ~2 000. We use a tighter cap
(64) because all but ~3 PD / ~2 LGD datasets are already < 64 features,
and a tight cap maximises the per-step row budget (cells = rows ×
features). Raise `sanitize.max_columns` to 100 / 128 to preserve more
features at the cost of fewer rows; re-run the data stage to apply.

#### 3.7 Label-encode classification targets
Map raw target labels to `{0, ..., K-1}`. Sorted lexicographically so
the encoding is stable across runs.

#### 3.8 Clip LGD targets to [0, 1]
LGD = loss given default = fraction of exposure lost. By definition
in [0, 1]. We clip values outside this range (typical: a few rows
with 1.02 or 1.05 from accounting roundoff, a few with −0.001 from
recovery > exposure edge cases) and log the count.

#### 3.9 Save
- `data/processed/<track>/<id>.sanitized.csv` — the final on-disk
  artefact.
- The target column is in there; downstream code reads it via
  `DATASET_METADATA[id]["target_column"]`.

---

### Stage 5a — TabPFN-side per-step preprocessing

The sanitized CSV is the input to TabPFN training but is **not** what
the model forward pass sees. Per training step the
`ProcessedDatasetLoader.__getitem__` runs an additional preprocessing
pipeline that mirrors the official TabPFN finetune
(`tfm-library/repositories/TabPFN .txt:26147-26319`):

#### 5a.1 `clean_data` (once per dataset, cached)
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

#### 5a.2 EnsembleConfig generation (once per dataset, cached)
`generate_classification_ensemble_configs` /
`generate_regression_ensemble_configs` builds the per-estimator
configuration: one of N preprocessor configs (squashing scaler /
quantile / none), one of N feature shifts, one of N class permutations.
For `n_estimators_finetune=2` we get 2 distinct configs per dataset.
These are **stable per dataset across all epochs** — matches the
published behaviour at `TabPFN .txt:26604-26635`.

#### 5a.3 Per-step subsample + ctx/query split
Per training step (= per dataset visited within an epoch):
- Draw a fresh stratified subsample of up to
  `max_rows_per_epoch` rows from the cleaned numeric array.
- Split into context (1 − qf fraction) and query (qf fraction).
- For LGD, z-normalize the target on context-only statistics
  (clamping std to 1e-8 if degenerate — mirrors the official path).

#### 5a.4 `TabPFNEnsemblePreprocessor.fit_transform_ensemble_members`
Per step we instantiate a fresh `TabPFNEnsemblePreprocessor` with the
cached `ensemble_configs`, fit it on the context split, and obtain
N preprocessed views — each potentially with different feature counts
(SVD/polynomial add columns), different rows (subsampling), and
different label encodings (class-permutation augmentation).
`member.transform_X_test(X_query)` applies the same per-member
pipeline to the query split.

#### 5a.5 Outlier soft-clip (just before model forward)
TabPFN's GPU step `TorchSoftClipOutliersStep` (`TabPFN .txt:35959-35967`)
soft-clips numerical columns to ±12σ for the classifier (None for
regressor). We invoke it from `_forward_one_member` immediately before
the model call. Categorical columns pass through unmodified.

#### 5a.6 Model forward + loss
`PerFeatureTransformer.forward(combined_x, train_y, categorical_inds, …)`
returns logits over query positions only. CE loss (PD) or
bar-distribution NLL (LGD) is computed against the canonical-order
target, with class-permutation unscramble for classifier members that
were trained on permuted labels.

---

### Stage 5b — baselines path

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

### Why two preprocessors

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

### Quick reference

Storage-tier shorthand below: **[stg]** = project staging
`<staging>/CreditPFN` (default corpus home on VSC), **[out]** = output
root `$VSC_DATA/CreditPFN` (durable). On a laptop both collapse to the
repo root.

| Stage | Module | Reads | Writes |
|---|---|---|---|
| 1 | `src/data/dedup.py --pass pre`  | `data/raw/{pd,lgd}/<id>.csv` [stg] | `output/manifests/dedup/doubles_<track>_pre.csv` [out] |
| 2 | `src/data/register.py`          | `data/raw/{pd,lgd}/` [stg]            | `output/manifests/manifest_{pd,lgd}.csv` [out] |
| 3 | `src/data/sanitize.py`          | `data/raw/` [stg] + manifest          | `data/processed/{pd,lgd}/<id>.sanitized.csv` [stg] |
| 4 | `src/data/dedup.py --pass post` | `data/processed/` [stg]               | `output/manifests/dedup/doubles_<track>_post.csv` [out] |
| 5a | `src/train/dataloader.py` + `src/train/tabpfn_preprocessing.py` | `data/processed/.../*.sanitized.csv` [stg] | Live tensors (no disk artefact) |
| 5b | `src/eval/benchmark.py` + `src/model/{boosting,linear,tabpfn_models}.py` | `data/processed/.../*.sanitized.csv` [stg] | Eval CSVs at `output/results/...` [stg] |

#### Resume semantics
See [README.md § 5](../README.md#5-re-submitting-the-pipeline-resume-semantics--cleanup). The data stage is idempotent — it
skips datasets whose sanitized CSV is already on disk.

#### Cleanup
`python -m src.utils.clean_run --clean --stages data` wipes everything
the data pipeline produces — the processed CSVs (staging), the dedup
reports + manifests + data logs (output root) — but **never** touches
`data/raw/`.

---

### Common gotchas

- **"I changed a surgical fix in `DATASET_METADATA`; why didn't it
  apply?"** — the sanitized CSV is already on disk. The data pipeline
  saw it and skipped. Run `clean_run --clean --stages data` first.

- **"My categorical column became numeric in the sanitized CSV"** —
  `sanitize.py:3.3` (coerce numeric strings) will cast a categorical
  if every value happens to parse as a number. Mark the column as
  categorical in `DATASET_METADATA[id]["categorical_columns"]` so
  TabPFN's `clean_data` re-encodes it correctly.

- **"Feature selection dropped columns on a dataset with ≤ 64
  features"** — it shouldn't. Selection only fires when a dataset
  exceeds `sanitize.max_columns` (64). If it fired below that, the
  dataset has hidden duplicate columns inflating the count.

- **"LGD training has negative losses"** — by design. See
  [README.md § 7 design notes](../README.md#design-notes-the-why) and the
  bar-distribution NLL discussion. Negative NLL means the model has
  placed sharp probability mass on the true target — a good sign.

---

## 2. Base checkpoints

Tabular-foundation-model weights used as starting points for continued
pretraining on credit-risk data, plus the trained checkpoints our
sweeps emit. Two families are swept: **TabPFN** (v2.6, v3) and
**TabICLv2** (added 2026-08-04). **Do not edit or commit new
checkpoints without updating this file.**

> **Where the weights actually live.** On VSC, both the **base**
> checkpoints and the **trained** outputs live in project *staging*,
> not in the git tree. Paths in this repo (e.g.
> `checkpoints/tabpfn-v3-classifier-v3_default.ckpt`) are resolved
> through `resolve_staging_path()` (`src/utils/paths.py`) to
> `/lustre1/project/stg_00211/CreditPFN/checkpoints/...` (override base
> with `$CREDITPFN_STAGING_ROOT` / `$TABPFN_STAGING_ROOT`). Both
> clusters (wICE for eval, Mindwell for training) can see this tier, so
> it is the hand-off point for trained weights. Off-VSC (laptop),
> `resolve_staging_path` falls back to the repo root, so the same
> relative paths resolve under `<repo>/checkpoints/`. The base weights
> are read once at job start and cached in RAM. See
> [VSC.md](VSC.md) for the storage-tier topology.

All facts below are sourced from:

- The upstream `tabpfn` package README (mirrored at
  `tfm-library/repositories/TabPFN .txt`, lines 649–751 and 2606–2623).
- Prior Labs' HuggingFace model cards (verbatim mirror at
  `tfm-library/repositories/Huggingface TabPFN.txt`).
- Hollmann et al. 2025 (*Nature*) and Grinsztajn et al. 2026
  (arXiv:2511.08667), Appendix C.

The inventory lists every base `.ckpt`, what training data Prior Labs
used to produce it, and a brief note on what role each plays in
**our** continued pretraining experiments. Trained outputs and their
on-disk format are documented further below.

From the TabPFN family only the **v2.6** and **v3** synthetic-only
bases are used (the older **v2.5** base has been dropped from the sweep
entirely). Alongside them the sweep includes the **TabICLv2** base of
the matching head — so both tracks run three bases.

TabICLv2 sources:

- Upstream code + finetuning internals (mirrored at
  `tfm-library/repositories/TabICLv2.txt`); the pip package is pinned
  `tabicl[finetune]>=2.1.1,<3` because our shim imports its private
  `tabicl._finetune.data` helpers.
- Weights: HuggingFace `jingang/TabICL`.
- Qu et al., *TabICLv2: A Tabular Foundation Model for In-Context
  Learning on Large Data* (see `tfm-library/SUMMARIES.md`).

### All our bases are synthetic-only

Every checkpoint we sweep was trained from scratch on millions of
*synthetic* tabular datasets sampled from a structural-causal-model
prior — no real-world data has touched the weights. **TabPFN-v3 and
all v2.6 variants** ship synthetic-only, and neither has a released
"real-finetuned" variant. That is exactly the clean starting point the
Real-TabPFN recipe (Garg et al. 2025) assumes: begin from the synthetic
prior, then continue-pretrain on real data — in our case the curated
credit-risk corpus — so any downstream gain is attributable purely to
that corpus.

### Inventory (verified against upstream)

| File | Size | Origin | Training data | Role in this project |
|---|---|---|---|---|
| `tabpfn-v3-classifier-v3_default.ckpt`         | 213 MB | HF `Prior-Labs/tabpfn_3` | **Synthetic-only.** The v3 HF card states *"TabPFN-3 is trained purely on synthetic tabular tasks."* New multi-stage transformer architecture (24 main layers); ≤1 M samples × ≤2 000 features (vs. 50 k for v2.6). | **Default sweep base.** Latest released checkpoint with the strongest published benchmarks (SOTA on TabArena, TALENT). |
| `tabpfn-v3-regressor-v3_default.ckpt`          | 233 MB | HF `Prior-Labs/tabpfn_3` | **Synthetic-only.** Same v3 card statement applies; no real-finetuned v3 regressor yet. | **Default sweep base** for LGD. |
| `tabpfn-v2.6-classifier-v2.6_default.ckpt`     | 43 MB  | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** — the v2.6 card states *"TabPFN-2.6 is trained purely on synthetic tabular tasks"*; no real-finetuned v2.6 variant has been released. | Sweep base: the cleanest v2.6 base available. |
| `tabpfn-v2.6-regressor-v2.6_default.ckpt`      | 51 MB  | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** (same card statement). No real-finetuned v2.6 regressor yet. | Sweep base: cleanest v2.6 regressor base. |
| `tabicl-classifier-v2-20260212.ckpt`           | 110 MB | HF `jingang/TabICL` | **Synthetic-only.** TabICLv2 is pretrained on synthetic tabular tasks; 3-stage architecture (column embedder → row interactor → ICL predictor), ~27 M params. Classifier head emits 10 logit columns. | **Second-family sweep base (PD).** Tests whether the CPT result generalises beyond one architecture/prior. |
| `tabicl-regressor-v2-20260212.ckpt`            | 114 MB | HF `jingang/TabICL` | **Synthetic-only**, same architecture; regression head emits 999 quantiles on context-z-normalised targets (no bar distribution). | **Second-family sweep base (LGD).** |

#### Getting the TabICLv2 weights onto VSC (one-time, from a LOGIN node)

Compute nodes have **no outbound network**, and our loaders pass
`allow_auto_download=False` so a missing file fails loudly instead of
stalling a GPU job. Stage both checkpoints once:

```bash
python -c "
from huggingface_hub import hf_hub_download
import shutil, os
dest = os.environ.get('CREDITPFN_STAGING_ROOT', '/lustre1/project/stg_00211/CreditPFN') + '/checkpoints'
os.makedirs(dest, exist_ok=True)
for f in ('tabicl-classifier-v2-20260212.ckpt', 'tabicl-regressor-v2-20260212.ckpt'):
    p = hf_hub_download('jingang/TabICL', f)
    shutil.copy2(p, dest + '/' + f)
    print('staged', f)
"
```

### How to read the naming conventions

For **v3** the naming is: only `_default` (synthetic-only). No
specialist or real-finetuned variants have been released yet.

For **v2.6** the naming is: `_default` = synthetic-only (no
real-finetuned variant published yet).

Both conventions are confirmed verbatim by the HuggingFace cards
mirrored at `tfm-library/repositories/Huggingface TabPFN.txt`.

### What we sweep over

The training config (`config/train.yaml::tunable`) treats the base
checkpoint as a tuneable knob and sweeps over the released
synthetic-only bases for v2.6 and v3:

| Track           | Sweep includes (default)                        | What each tells us                                                        |
|-----------------|-------------------------------------------------|---------------------------------------------------------------------------|
| PD (classifier) | `v3_default` · `v2.6_default` · `tabicl-v2`     | v3 synthetic-only · v2.6 synthetic-only · second family (different prior + architecture) |
| LGD (regressor) | `v3_default` · `v2.6_default` · `tabicl-v2`     | same three, regression heads                                              |

The total grid per track is then `3 bases × 4 LRs × 2 adapt-modes ×
1 qf × 1 acc × 2 epoch-pass-modes = 48 trials`. The current LR grid is
`[3e-7, 1e-6, 1e-5, 3e-5]` — the bottom rung `3e-7` matches
Real-TabPFN, `1e-5` is TabICLv2's own finetuning default, and `3e-5`
approaches Rubachev's separate single-dataset finetuning median
(~3.9e-5); `1e-4` is excluded because it diverged on the no-LoRA +
`qf=0.20` setting (revisit now that `weight_decay=0.0`). See the
hyperparameter-rationale table in the README for the full literature
comparison.

**The adapt-mode axis is family-specific.** `use_lora=true` means LoRA
for TabPFN, and **freeze-backbone** (train the ICL module only) for
TabICLv2 — that family's own pretraining stage-3 regime. The reason is
empirical: full SFT collapsed TabICLv2 in two independent reports
(TabZilla accuracy 0.873 → 0.567 in Tanna 2026; "failed to train
TabICLv2" in Kolberg 2026), so the freeze-backbone arm is TabICLv2's
safe-adaptation arm rather than a parameter-count experiment. Its
checkpoints are tagged `_iclhead` instead of `_lora`.

Every base in this sweep follows the methodologically clean
ablation recipe of Real-TabPFN (Garg et al. 2025): start from
the synthetic-only checkpoint, continue-pretrain on real data —
in our case our curated credit-risk corpus. Any downstream gain on
credit-risk benchmarks is attributable purely to that corpus.

The eval pipeline (`scripts/eval_pipeline.py`) scores all of these
side-by-side against XGBoost / CatBoost plus a linear baseline
(LogReg for PD, LinReg for LGD) and the *untuned* versions of each
base, so the question of "which base wins" gets answered empirically
on the held-out test split. Eval reads the trained checkpoints from
staging and only scores those present on disk at submit time — see
[the trained-checkpoints section](#trained-checkpoints-our-outputs)
below.

### Architecture differences across versions

|                                       | v2.6                          | v3                                       |
|---------------------------------------|-------------------------------|------------------------------------------|
| Layers                                | 24 (fixed)                    | 24 main layers (multi-stage transformer) |
| Attention pattern                     | TabPFNv2-style alternating    | Multi-stage transformer-based            |
| Sample limit (intended)               | ≤ 50 000                      | ≤ 1 000 000                              |
| Feature limit (intended)              | ≤ 2 000                       | ≤ 2 000                                  |
| Real-finetuned variant published?     | No (only synthetic `_default`)| No (only synthetic `_default`)           |
| Model technical report                | Grinsztajn et al. 2026 (arXiv:2511.08667, same architecture family) | Grinsztajn et al. 2026, *TabPFN-3 Technical Report* |
| Approximate checkpoint size           | ~43–51 MB                     | ~213–233 MB                              |
| License                               | `tabpfn-2.6-license-v1.0`     | `tabpfn-3-license-v1.0`                  |

### Trained checkpoints (our outputs)

Each sweep trial writes a finetuned checkpoint to:

```
resolve_staging_path(checkpoint.trained_dir) / <track> / <descriptive_name>.ckpt
```

With the default `checkpoint.trained_dir: "checkpoints/trained"` in
[`../config/train.yaml`](../config/train.yaml), that resolves on VSC to
`/lustre1/project/stg_00211/CreditPFN/checkpoints/trained/<pd|lgd>/…`.
(Training actually resolves via `resolve_writable_staging_path`, which
probes staging writability first: if the compute node can't write staging
— the 2026-07-03 Mindwell failure mode — checkpoints are saved under
`$VSC_DATA/CreditPFN/checkpoints/trained/` instead, and the eval gate
archives them into staging afterwards. Either way the durable copy ends
up in project storage; see docs/VSC.md §0.2.)
Alongside each `.ckpt` we write a `<name>.ckpt.provenance.json`
sidecar (full training-time hyperparameters, the train/test dataset
lists, walltime, GPU, library versions) so a checkpoint can be
inspected without `torch.load`. The training manifest
`output/training/manifests/<run_name>_<track>.csv` records one row per
trial with a `status ∈ {OK, FAIL, SKIP, DIVERGED}`; the eval pipeline
rosters only `OK`/`SKIP` rows whose `.ckpt` exists on disk (it
excludes `FAIL` and `DIVERGED`).

#### Descriptive filename schema

The basename encodes the tunable hyperparameters
(`descriptive_name()` in `src/train/loop.py`):

```
<run_name>_<track>_<base-stem>_lr<lr>_seed<seed>[_qf<qf>][_acc<K>][_fullpass][_lora].ckpt
```

- `<base-stem>` — the base checkpoint's filename stem (e.g.
  `tabpfn-v3-classifier-v3_default`).
- `lr<lr>` — learning rate in `%.0e` form with the `+` stripped (e.g.
  `lr1e-05`).
- `qf<qf>` — query fraction × 100, zero-padded (e.g. `qf20`); omitted
  when `query_fraction` is `None`.
- `acc<K>` — `accumulate_grad_batches`; omitted when `None`.
- `_fullpass` — present only for `epoch_pass_mode == "full_pass"`; the
  default `one_sample` adds no tag.
- `_lora` / `_iclhead` — present only when `use_lora` is true;
  `_iclhead` for TabICLv2 bases (freeze-backbone), `_lora` for TabPFN.

Optional segments are dropped when their value is the default/`None`,
so the default one-step-per-dataset, full-FT sweep produces the same
short names as the original (pre-2026-06-01) runs.

#### On-disk save format (Prior Labs 4-key dict)

`save_finetuned()` (`src/train/model.py`) persists a single
`torch.save` dict with the **four mandatory keys** Prior Labs'
`load_model` expects:

| Key | Contents |
|---|---|
| `state_dict` | the model weights (LoRA adapters merged back into the base via `merge_and_unload()`, so a LoRA trial is indistinguishable from a full-FT save and needs no PEFT at load time) |
| `config` | `asdict()` of the pydantic `ArchitectureConfig` |
| `architecture_name` | one of `tabpfn_v2`, `tabpfn_v2_5`, `tabpfn_v2_6`, `tabpfn_v3` — tells the loader which architecture class to instantiate |
| `inference_config` | `asdict()` of the `InferenceConfig` (pydantic `extra="forbid"`) |

The file round-trips through `TabPFNClassifier(model_path=…)` /
`TabPFNRegressor(model_path=…)`. We also add a `provenance` key plus
the JSON sidecar described above.

> **Previously-fixed bug (do not regress).** An earlier
> `save_finetuned` wrote only `{state_dict, config}`. With
> `architecture_name`/`inference_config` missing, `load_model` falls
> back to the v2 architecture and v3/v2.6 weights fail to load with
> *"Missing key(s) in state_dict"* — which broke **every** eval. All
> four keys are now mandatory. `architecture_name` is resolved via the
> upstream `_resolve_architecture_name`; if that private symbol is
> unavailable it falls back to a config-class-name match and **raises**
> on an unknown class rather than guessing `"base"` (which would
> silently mislabel a v3 checkpoint as v2.5 and make it unloadable).

#### v2.6 vs v3 criterion handling (regressor / LGD)

The two architectures differ in how the bar-distribution criterion is
restored at load time, and our save format is compatible with both:

- **v3** — `model.forward` takes `test_targets_MB`, so the loader
  **strips** the `criterion.*` keys from the state-dict and rebuilds
  the `FullSupportBarDistribution` from the model's own
  `regression_borders` buffer.
- **v2.6** — has no `test_targets_MB`, so the loader **requires** the
  `criterion.*` keys to be present in the state-dict.

`save_finetuned` always merges the regressor's `criterion.*`
parameters into the saved state-dict. Those keys are *required* by
v2.6 and *harmlessly stripped* by v3, so the same write path
round-trips correctly for both. (Classifiers carry no bar-distribution
criterion, so this only concerns the LGD regressor checkpoints.)

#### TabICLv2 save format (upstream 2-key dict)

TabICLv2 checkpoints use a different, simpler schema, written by
`save_finetuned_tabicl()` (`src/train/tabicl_model.py`):

| Key | Contents |
|---|---|
| `config` | the `TabICLv2(**kwargs)` init dict, with `recompute` reset to `False` so inference doesn't pay for gradient checkpointing |
| `state_dict` | CPU model weights (no adapter merging exists for this family — freeze-backbone trials simply have fewer changed tensors) |

Plus our `provenance` key and the same `.provenance.json` sidecar. The
file round-trips through `TabICLv2Classifier(model_path=…)` /
`TabICLv2Regressor(model_path=…)`; upstream's loader reads only the two
keys above with `weights_only=True` and ignores extras, so provenance
must stay JSON-safe primitives.

Two family differences worth remembering when reading results:

- **No exact predictive density.** The regressor emits 999 quantiles,
  not a bar distribution, so `neg_nll` is `NaN` for TabICLv2 rows. Never
  compare density metrics across families; the planned cross-family
  density metric is CRPS (computable from the quantiles).
- **`recompute=True` during training.** Gradient checkpointing is the
  main VRAM lever. It is also what upstream does: TabICLv2's own stage-3
  pretraining enables it above 20K samples.

#### TabICLv2 context sizes — use the paper, not the library defaults

A correction worth recording, because the first values were wrong:

| | Value | Where it comes from |
|---|---|---|
| Eval fold cap | **1 000 000** (same as v3) | Qu et al. 2026: 1M samples × 500 features in ~450 s under 50 GB GPU + 24 GB CPU (hierarchical CPU/disk offloading); QASSMax scalable softmax keeps attention sharp at long context, so million-scale tables are handled "natively, without retrieval and distillation" |
| Training rows/step | **26 000** (same as v3) | Inside TabICLv2's own stage-3 pretraining range (400–60 000 samples, log-uniform); equal to v3 so a cross-family difference is not confounded with context size |

The earlier values (10 000 train / 50 000 eval) came from
`max_data_size=10_000`, a **default argument of their convenience
finetuning wrapper**, and from mirroring v2.6's cap. Neither is a
capability limit — million-scale in-context inference is TabICLv2's
headline result, and a 50 000-row eval cap would have handicapped it 20×
against v3 while discarding the model's main advantage.

Both numbers are still **unmeasured on our hardware**. Run
`scripts/probe_row_cap.py` (it has a TabICLv2 branch, grid 10k–60k) before
committing a full sweep to them, and watch eval walltime on the first
run — a large-context model overruns the clock rather than OOM-ing.

### Licence

All weights are released under Prior Labs' research-only licences
(`tabpfn-2.6-license-v1.0`, `tabpfn-3-license-v1.0`). Testing,
evaluation, and internal benchmarking are explicitly allowed;
commercial use, client deliverables, or commercial decision-making
based on the model's outputs are not. Full text in the licence
files inside each HF repo.

---

## 3. Context-size caps

Single authority on **how many rows each model sees at a time**, for both
families and both stages. The config files carry the numbers; this file carries
the reasoning, the measurements, and the traps.

Settings that point here:

| Setting | File | Stage |
|---|---|---|
| `finetuning.max_rows_per_epoch` | `config/data.yaml` | training |
| `finetuning.max_cells_per_epoch` | `config/data.yaml` | training (off) |
| `train.n_estimators_finetune{,_tabicl}` | `config/train.yaml` | training |
| `max_rows_per_model` | `config/eval.yaml` | evaluation |

Re-run `scripts/probe_row_cap.py` (via `scripts/slurm/probe_row_cap.slurm` —
**never bare on a login node**) whenever the GPU, `sanitize.max_columns`, or an
ensemble size changes. Nothing here is valid for a different configuration.

---

### 1. Training: rows per step

A training step forwards **all `n_estimators_finetune` ensemble members** and
holds every member's graph for one backward pass. So:

```
peak memory  ≈  n_estimators  ×  rows  ×  (per-member cost per row)
```

The values in `config/data.yaml` are the **PD (2-member) caps**. `loop.py`
scales them down by `2/n_estimators` for LGD's 8 members, so both tracks hold
roughly the same GPU memory ("member-aware row-cap scaling" in
`train_one_config`). That scaling is **TabPFN-only** — it was calibrated on
TabPFN's memory slope, and TabICLv2 is pinned to 2 members on both tracks anyway.

#### Measured — B200 (183 GiB), 2026-08-05, job 11509346

64 features, `query_fraction=0.20`, real forward+backward. TabPFN probed at
**1 member**; TabICLv2 at **2 members with `recompute=True`** (its actual training
configuration), so compare the per-member column, not the raw peaks.

| Base | Measurements | Slope /1k rows | Per member | First failure |
|---|---|---|---|---|
| TabPFN v3 | 20k → 50.3 GB · 50k → 124.9 GB | 2.49 GB | 2.49 GB | OOM at 100k |
| TabPFN v2.6 | 9k → 49.2 GB · 20k → 108.8 GB · 30k → 169.3 GB | 5.72 GB | 5.72 GB | OOM at 50k |
| TabICLv2 | 10k → 10.5 GB · 26k → 26.8 GB | 1.02 GB | **0.51 GB** | cuDNN at 40k |

Memory is ~linear in rows; the intercept is negligible (weights are tiny next
to activations). Step times are 0.5–2.5 s throughout — **except** the first
timed step of a job, which includes CUDA warm-up (v3 read 7.58 s at 20k and
2.53 s at 50k; don't misread that as superlinear).

The 2026-07-08 probe measured v3 and v2.6 within ~2 % of the above. An even
earlier "0.93 GB per 1k rows" figure was a **bad measurement** — it timed the
lightweight monitor eval, not a training step. Never trust the monitor's
`gpu_peak_alloc` as the training peak.

#### Derived caps

Target ~130 GB peak, leaving ~53 GiB for optimizer state, fragmentation and the
rolling eval snapshot.

| Base | PD (2 members) | LGD (8 members, auto-scaled) | Predicted peak |
|---|---|---|---|
| v3 | **26 000** | 6 500 | ~129 GB |
| v2.6 | **11 000** | 2 750 | ~126 GB |
| TabICLv2 | **26 000** | 26 000 (2 members, not scaled) | ~27 GB |

PD's large datasets (hackerearth 532k, home_credit 307k, vehicle_loan 233k) bind
on the cap, so 26k buys real context — the lever Real-TabPFN attributes gains
to. LGD datasets are all ≤16k rows, so the LGD caps truncate only lgd_freddie.

Lookup key is the leading `v<MAJOR>[.<MINOR>]` of the base filename, or
`tabicl` for that family; `default` is the fallback.

#### Why TabICLv2 is set to 26 000

1. **Parity.** If TabICLv2 trained on 10k rows/step while v3 trained on 26k,
   every TabICLv2-vs-v3 difference would confound **architecture** with **context
   size**, making the cross-family comparison — the whole reason the family was
   added — uninterpretable.
2. **It matches TabICLv2's own pretraining.** Qu et al. 2026 §B.1: stage 1 = 1 024
   samples, stage 2 = 400–10 240, **stage 3 = 400–60 000** (log-uniform), with
   gradient checkpointing above 20k "to avoid the out-of-memory error" — exactly
   the `recompute=True` we force. The paper credits large-sample exposure for
   large-data generalisation (>10k-row rank 5.50 → 4.71 from stage 2 to 3).

An earlier value of 10 000 came from `max_data_size=10_000`, a **default
argument of tabicl's convenience finetuning wrapper** — a library default, not a
capability limit, and far below their own pretraining regime.

#### TabICLv2's ceiling is a cuDNN kernel limit, not memory

At 40 000 and 60 000 rows the step dies with:

```
RuntimeError: Expected mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good()
              to be true, but got false
```

That is cuDNN's **fused multi-head-attention graph** failing on a long attention
sequence — **not** an OOM. There were ~140 GB still free. At 26k, TabICLv2 uses
26.8 of 183 GB (15 %), and linear extrapolation says memory alone would allow
~180k rows.

So 26 000 happens to be both the parity value and safely under a hard ceiling
somewhere in 26k–40k. **Do not raise it** without first moving the attention
backend off cuDNN (e.g. `torch.backends.cuda.enable_cudnn_sdp(False)`) *and*
re-probing. Raising it blind produces that opaque error mid-sweep.

#### Known confound: LGD context is not symmetric

TabPFN uses 8 members on LGD, so its caps auto-scale to v3 6 500 / v2.6 2 750
rows per step, while TabICLv2 keeps 2 members and therefore 26 000. Because every
LGD dataset is ≤16k rows, **TabICLv2 sees complete tables while v3 sees 6 500-row
samples.**

This is deliberate: each family runs at its own official member count (the fair
"as its authors intended" protocol), and changing TabPFN's LGD member count
would break comparability with the run-4 results. But it means the **LGD
cross-family comparison is entangled with context size and must be reported as
such**, not presented as a pure architecture comparison.

#### The cell budget (`max_cells_per_epoch`, currently off)

When non-null for a base, per-step rows become
`min(max_rows_per_epoch, max_cells // n_features)` — narrow datasets get more
rows, wide ones fewer, at roughly constant cell count.

This fits **v3 only**: TabPFN-3's capacity is a cell-budget frontier (its report
§2.4 treats 1M rows × 200 features as equivalent to 100k × 2000), and its
3-stage design decouples the ICL stage from feature count while row-chunking
activation memory. It is **wrong for v2.6**, whose dual attention costs
`O(r²·c + r·c²)` — quadratic in rows — so v2.6 stays on a pure row cap.

To enable for v3: set a cell budget *and* raise `max_rows_per_epoch.v3` to the
row ceiling narrow datasets may reach (e.g. 8 000 000 cells, 100 000 rows), then
validate against OOM before a full sweep.

---

### 2. Evaluation: rows per fold

`max_rows_per_model` caps the **training partition of each CV fold** only. The
held-out test partition is never capped — the model predicts on every test row
in one call.

| Base | Cap | Why |
|---|---|---|
| v3 | 1 000 000 | TabPFN-3's published envelope (report §2.4) |
| v2.6 | 50 000 | Its published design envelope ("up to 50,000 data points"). Reduced from 100k on 2026-07-13: dual attention is O(rows²), and one v2.6 × algorithmwatch fold took ~40 min on an A100, blowing 8 cells past the old 2 h walltime. 50k stays in-envelope and cuts the quadratic term 4×. |
| TabICLv2 | 1 000 000 | Million-scale in-context inference is TabICLv2's headline capability: Qu et al. report 1M samples × 500 features in ~450 s under 50 GB GPU + 24 GB CPU via hierarchical CPU/disk offloading, and QASSMax (scalable softmax, logits × `s·log n`) exists to keep attention sharp at long context. |

Our corpus tops out at 532k rows and ≤64 features, so 1M means "no cap in
practice" for both TabPFN v3 and TabICLv2 — which is also what keeps their
trained-vs-untuned comparisons paired.

The cap resolves from the **base checkpoint** for both the trained and untuned
handle, so a family's two arms always see the same fold size. (A 2026-08-04 bug
had the untuned branch miss its key and silently fall through to `default`,
scoring untuned-v3 on 50k-row folds while trained-v3 got 1M.)

**Untested at scale by us:** TabICLv2's *training* path fails above ~26k rows via
the cuDNN issue above. Inference is a different code path — upstream demonstrates
1M and their wrapper chunks and offloads — so the cap is left at 1M. If TabICLv2
eval cells fail with that same `mha_graph` message on the large PD datasets, the
fix is to disable the cuDNN SDPA backend, **not** to lower the cap. Such
failures surface as FAIL rows carrying the error text, not silently.

Classical baselines (XGBoost / CatBoost / LogReg / LinReg) are never capped —
they see the full training fold. Their HPO uses a separate
`hpo.<model>.max_rows` subsample.

---

## 4. Deliberate oddities

Every entry here is something a reader — human or agent — would reasonably flag as a bug, redundancy
or leftover, and which breaks something real if "cleaned up". Each one also carries a `why` comment
at its site; this file is the index, so a reviewer can check before editing rather than after.

Dead ends live in `AGENTS_MEMORY.md`; what changed and when in `CHANGELOG.md`.

### Adaptation modes and regularisation

- **`use_lora=True` means *freeze-backbone* for TabICLv2, LoRA for TabPFN.** One grid axis, two family
  meanings; the run tags are `_iclhead` vs `_lora`. Do not unify them — the literature says full SFT
  breaks TabICLv2, so this is that family's safe-adaptation arm. `descriptive_name()` derives the tag
  from the base filename, so every call site stays consistent without call-site changes.
- **L2-SP applies to TabICLv2 even in freeze-backbone mode**
  (`l2sp_applicable = (family == "tabicl") or (not use_lora)`) because the trainable ICL head still
  drifts from its pretrained values. For TabPFN + LoRA the base weights are frozen and the adapters
  have no pretrained anchor, so L2-SP is genuinely inert there — hence the asymmetry.
- **`model.col_embedder.eval()` is re-applied after every `model.train()`** in the epoch loop.
  TabICLv2 routes its forward pass on module training flags, so `train()` would silently restore the
  backbone's train-time behaviour. Looks redundant; is not. (Distinct from using `.eval()` *to
  freeze*, which is the 06-08-2026 bug in `AGENTS_MEMORY.md` — freezing is `requires_grad=False`.)
- **TabICLv2 ships one non-trainable `Parameter`** (`row_interactor.tf_row.rope.freqs`, RoPE
  constants), so "every parameter requires grad" is a false assertion even in full-FT mode.

### Memory, capacity and context size

- **Member-aware row scaling** in `train_one_config`: the `max_rows_per_epoch` values in
  `config/data.yaml` are the **PD 2-member** caps, and the code divides by `n_estimators / 2` for
  LGD. Removing this apparently redundant scaling re-introduces the LGD OOM.
- **The LGD context asymmetry is a known confound, not a bug.** On LGD, TabPFN uses 8 ensemble
  members so its caps auto-scale to v3 6 500 / v2.6 2 750, while TabICLv2 keeps 2 members and 26 000.
  Since every LGD dataset is ≤16k rows, TabICLv2 sees full tables and v3 sees a 6.5k sample. Kept
  deliberately — each family runs at its own official member count, and changing TabPFN's would
  break run-4 comparability — but the LGD cross-family comparison is confounded with context size
  and must be stated as such in the paper.

### Paths, saving, eval

- **`resolve_writable_staging_path` and `resolve_staging_path` are distinct on purpose** (probe +
  fallback vs plain resolution). Trained checkpoints use the former, because staging has been
  unwritable from compute nodes before.
- **`neg_nll` is clamped to ±100 nats** (`tabpfn_models.py`) to guard the v2.6 regressor density
  underflow that produced `-inf` and poisoned every aggregate that touched it.
- **`epoch_eval_every=5`:** the monitor eval runs on every 5th epoch, and the divergence detector's
  metric window uses only *monitored* epochs (`monitored_metrics`) — otherwise the NaNs from skipped
  epochs would look like a collapse.

### Environment and docs

- **"Genius login node" in the docs and scripts is correct.** Neither wICE nor Mindwell has its own
  login node; you always SSH to Genius and submit cross-cluster. A past audit flagged this as a bug;
  it is not.
- **Citations into `tfm-library/repositories/*.txt` are file-level, with no line numbers,** on
  purpose: those dumps are periodically refreshed and line numbers drift by thousands. Cite symbol
  names.
