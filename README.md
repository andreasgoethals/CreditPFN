# CreditPFN

Continued pretraining of TabPFN (v2.5 / v2.6) on a curated corpus of real-world
credit-risk datasets. The goal is to specialise the tabular foundation
model's in-context-learning prior toward the structures, feature
distributions, and label noise characteristic of credit-risk data,
and to evaluate whether a credit-specialised foundation model
outperforms the generalist TabPFN on downstream credit-risk tasks.

## Background

**TabPFN** is a transformer-based tabular foundation model that
performs in-context learning over entire tabular datasets in a single
forward pass. Each version ships two separate checkpoints:

- a **classifier** used here for **Probability of Default (PD)**
  prediction, and
- a **regressor** used here for **Loss Given Default (LGD)**
  estimation.

These two checkpoints have different weights and must be adapted
independently during continued pretraining.

**Which base checkpoint?** A choice we will treat as a *training-stage
hyperparameter* and benchmark, not a decision baked in at the
data-pipeline stage. The two main candidates are:

- **TabPFN-2.6** — the most recent architecture (24 layers).
  Both default checkpoints (`tabpfn-v2.6-classifier-v2.6_default.ckpt`
  and `tabpfn-v2.6-regressor-v2.6_default.ckpt`) are *synthetic-only*,
  verified by the HuggingFace model card ("TabPFN-2.6 is trained
  purely on synthetic tabular tasks") and the package source. v2.6
  has no real-finetuned variant published — there is no
  `Real-TabPFN-2.6` yet.
- **TabPFN-2.5** — the previous family (18–24 layers). Ships with
  several checkpoints; the methodologically clean base for our
  continued pretraining is `tabpfn-v2.5-classifier-v2.5_default-2.ckpt`
  (synthetic-only) and `tabpfn-v2.5-regressor-v2.5_default.ckpt` (also
  synthetic-only, despite the unsuffixed name — the regressor default
  *is* the synthetic-only one for v2.5). The corresponding
  `_default.ckpt` for the v2.5 *classifier* is real-finetuned (Prior
  Labs' generic 43-dataset Real-TabPFN-2.5 corpus) and is the right
  comparison baseline.

The full inventory plus the full citation chain that grounds these
claims lives in
[`checkpoints/CHECKPOINTS.md`](checkpoints/CHECKPOINTS.md). When
`src/train/` is implemented we will benchmark continued pretraining
from both v2.5 and v2.6 synthetic-only bases and compare against
Real-TabPFN-2.5 as a published baseline.

**Continued pretraining** — as introduced for tabular foundation
models in *Real-TabPFN* (Garg et al., 2025,
[arXiv:2507.03971](https://arxiv.org/abs/2507.03971)) — extends the
synthetic-prior pretraining of TabPFN with additional training on a
curated corpus of real tabular datasets from a target domain. This
project applies the same methodology, but to a different domain:
credit risk.

**Credit risk modelling** has two primary quantitative use cases that
map directly onto the TabPFN checkpoints above:

1. **Probability of Default (PD)** — binary classification of whether
   an obligor will default within a given horizon.
2. **Loss Given Default (LGD)** — regression of the fraction of
   exposure lost conditional on default.

## Compute

Training is run on the VSC supercomputer using A100 GPUs with SLURM
job scheduling. Job scripts and SLURM templates live under `scripts/`.

## Repository layout

```
CreditPFN/
├── README.md
├── .gitignore
├── requirements.txt
├── checkpoints/                  TabPFN base weights + CHECKPOINTS.md
├── config/
│   ├── data.yaml                 every knob for src/data/* (this README's focus)
│   ├── training.yaml             (placeholder; populated when src/train/ lands)
│   └── base.yaml                 (placeholder)
├── data/                         (gitignored)
│   ├── raw/{pd,lgd}/<id>.csv     hand-curated input corpus
│   ├── processed/{pd,lgd}/       <id>.sanitized.csv (sanitize.py output)
│   ├── cached/{pd,lgd}/<id>/     chunk_NNN.npz + meta.json (dataset.py output)
│   ├── dedup/                    doubles_{track}_{pre,post}.csv (dedup.py output)
│   ├── manifest_pd.csv           register.py output (PD)
│   └── manifest_lgd.csv          register.py output (LGD)
├── logs/                         one timestamped file per orchestrator run
├── papers/                       PDF library + Literature.md (chronological summary)
├── repositories/                 read-only reference corpus + REPOSITORIES.md
├── scripts/
│   └── data_pipeline.py          end-to-end orchestrator (5 stages + logging)
├── src/
│   ├── data/                     ← Stage 1–5 modules
│   ├── utils/                    run-log helper, etc.
│   ├── train/                    (placeholder; multi-table fine-tuning loop)
│   ├── eval/                     (placeholder)
│   └── model/                    (placeholder)
├── notebooks/                    three exploration notebooks
│   ├── 0.0. raw_data_exploration.ipynb        — what did the vendor deliver?
│   ├── 0.1. processed_data_exploration.ipynb  — did sanitize produce sensible inputs?
│   └── 0.2. cached_data_exploration.ipynb     — is the .npz cache training-ready?
└── tests/                        smoke + unit tests, flat
    └── test_data.py              all data-pipeline tests in one file
```

## Quick start

> **Python 3.12 strongly recommended.** Several dependencies
> (scikit-learn, parts of torch) do not yet ship prebuilt wheels for
> Python 3.14, so `pip install` will try to compile from source and
> fail. Use `py -3.12` (Windows) or `python3.12` (Linux/macOS).

```bash
# 1. Create the project venv (once). Use Python 3.12 explicitly.
py -3.12 -m venv .venv --prompt CreditPFN     # Windows / PowerShell
# python3.12 -m venv .venv --prompt CreditPFN # Linux / macOS

.venv/Scripts/activate              # Windows / PowerShell
# source .venv/bin/activate         # Linux / macOS
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Run the full data pipeline end-to-end.
python scripts/data_pipeline.py --fresh       # rebuild from scratch
# python scripts/data_pipeline.py             # incremental (skip cached)
# python scripts/data_pipeline.py --datasets 0001.gmsc 0001.heloc

# 3. Run the tests.
pytest -q tests/test_data.py

# 4. (Optional) open the exploration notebooks in VS Code / Jupyter:
#    notebooks/0.0. raw_data_exploration.ipynb
#    notebooks/0.1. processed_data_exploration.ipynb
#    notebooks/0.2. cached_data_exploration.ipynb
```

## Data pipeline

Five stages run in this order. Every script reads
[`config/data.yaml`](config/data.yaml) — a single yaml with every
non-obvious knob commented and grouped by file-of-first-use. The
end-to-end orchestrator is
[`scripts/data_pipeline.py`](scripts/data_pipeline.py); each stage
can also be invoked independently:

```bash
python -m src.data.dedup --pass pre   # 1. duplicate sweep on raw/
python -m src.data.register           # 2. build manifests
python -m src.data.sanitize           # 3. surgical fixes + agnostic clean
python -m src.data.dedup --pass post  # 4. duplicate sweep on processed/
python -m src.data.dataset            # 5. chunk + cache to .npz
```

| Stage | Module | Reads | Writes |
|---|---|---|---|
| 1 | [`src/data/dedup.py`](src/data/dedup.py)        `--pass pre` | `data/raw/{pd,lgd}/*.csv` | `data/dedup/doubles_{track}_pre.csv` |
| 2 | [`src/data/register.py`](src/data/register.py) | `data/raw/{pd,lgd}/*.csv` + hardcoded `DATASET_METADATA` | `data/manifest_{pd,lgd}.csv` |
| 3 | [`src/data/sanitize.py`](src/data/sanitize.py) | `data/raw/{pd,lgd}/*.csv` + manifests | `data/processed/{pd,lgd}/<id>.sanitized.csv` |
| 4 | [`src/data/dedup.py`](src/data/dedup.py)        `--pass post` | `data/processed/{pd,lgd}/*.sanitized.csv` | `data/dedup/doubles_{track}_post.csv` |
| 5 | [`src/data/dataset.py`](src/data/dataset.py)   | `data/processed/{pd,lgd}/*.sanitized.csv` + manifests | `data/cached/{track}/<id>/chunk_NNN.npz` + `meta.json` |

Plus one importable helper used by stages 2 and 3:

* [`src/data/preprocessing.py`](src/data/preprocessing.py) —
  per-dataset `DATASET_METADATA` (target column, categorical hints,
  source) and per-dataset *surgical* fixes (drop ID columns, decode
  bespoke string formats, parse "5yrs 3mon" → integer months, drop
  target-derived leakage columns). Importable; not a CLI stage.

### What each stage does, in one sentence

* **`preprocessing.apply_dataset_specific_fixes(df, id)`** — drops ID
  columns, decodes hand-crafted strings, parses dates, and removes
  target-leakage columns for the 21 known datasets. *No* statistical
  operations: no log-transforms, no scaling, no clipping.
* **`dedup.py`** — eight detection methods (identifier match,
  column-name Jaccard + identical shape, row-level pandas hash,
  column-level hash, rounded-row hash, subset detection, fuzzy
  column-name match) per pass, per track. First-encountered wins.
* **`register.py`** — applies the surgical fixes, then computes
  per-dataset metadata (n_rows / n_cols, missing rate, class balance,
  target mean/std, content-aware shape hash). Idempotent: re-running
  updates rows in place.
* **`sanitize.py`** — applies the surgical fixes, then a
  dataset-agnostic clean: drop exact-duplicate columns, drop columns
  with NaN rate > 90%, drop constant columns, coerce numeric strings,
  cast numericals to float32, replace ±inf with NaN, optional
  FeatureAgglomeration to ≤ 128 columns (Ward linkage on
  StandardScaler-distances, output features are unscaled per-cluster
  means), label-encode classification targets, clip LGD targets to
  [0, 1].
* **`dataset.py`** — chunks each sanitised dataset into ≤ 20 000-row
  chunks (stratified for PD, random for LGD), splits each chunk
  60% context / 40% query, ordinal-encodes categoricals **with the
  encoder fit on context only** (so query categories unseen in
  context get the unknown-value sentinel `-1`, mirroring TabPFN's
  inference scenario), writes numpy `.npz` per chunk plus a
  `meta.json` sidecar.

## Data exploration

Three notebooks under `notebooks/`, designed to scale to the
3 000-dataset corpus:

* `0.0. raw_data_exploration.ipynb` — what did the vendor
  deliver? Per-track shape and missingness on raw CSVs,
  per-dataset target distribution for LGD, anomaly scan.
* `0.1. processed_data_exploration.ipynb` — did sanitisation
  produce sensible inputs? Same plot family as raw but on the
  post-sanitise corpus.
* `0.2. cached_data_exploration.ipynb` — is the cache healthy
  for training? Chunk count / size, encoder-leakage sanity check
  (unknown-sentinel rate per dataset), within-dataset target
  consistency across chunks.

All three load `cfg` from `config/data.yaml` by default. Corpus
summaries are **memoised** so the first cell pays the disk-read
cost once (~90 s on the wide datasets) and every subsequent
plot reads from RAM.

### What sanitize.py deliberately does NOT do

TabPFN's package handles these steps internally — see
[`repositories/REPOSITORIES.md`](repositories/REPOSITORIES.md) §
"Outlier handling" for the verified analysis:

| Step | Why we don't pre-apply it |
|---|---|
| Outlier winsorisation | TabPFN's `OUTLIER_REMOVAL_STD = 12.0` (classifier) / `None` (regressor) handles outliers with the right semantics (context-only statistics, soft log-squash) |
| `PowerTransformer` / `QuantileTransformer` / `RobustScaler` | TabPFN's per-estimator inference ensemble cycles through these on every fit; pre-applying any of them on disk fights that ensemble |
| NaN imputation | `NanHandlingEncoderStep` handles NaNs natively (replaces with a learned default + emits a binary indicator) |
| Regression target z-normalisation | `RegressorBatch.znorm_space_bardist_` standardises the target internally and inverts at predict time |

## Tests

```bash
pytest -q tests/data/test_pipeline.py
```

Tests cover the public contract of every module: surgical fixes
preserve the target column on all 21 raw CSVs, manifest rows are
typed correctly per task, dedup pairwise comparisons trigger on the
right checks, dataset-chunking helpers are deterministic and
disjoint.

## References

The full paper library lives under [`papers/`](papers/) with a
chronological, detailed summary in
[`papers/Literature.md`](papers/Literature.md). The most directly
relevant works for this project:

- **Garg et al., 2025.** *Real-TabPFN — Improving Tabular Foundation
  Models via Continued Pre-training With Real-World Data.*
  [arXiv:2507.03971](https://arxiv.org/abs/2507.03971) — the recipe
  we follow.
- **Hollmann et al., 2025.** *Accurate predictions on small data
  with a tabular foundation model.* (Nature) — the TabPFNv2
  architecture we instantiate.
- **Grinsztajn et al., 2025.** *TabPFN-2.5: Advancing the State of
  the Art in Tabular Foundation Models.*
  [arXiv:2511.08667](https://arxiv.org/abs/2511.08667) — the
  successor architecture used by our v2.6 checkpoints.
- **Rubachev et al., 2025.** *On Finetuning Tabular Foundation
  Models.* — fine-tuning hyperparameter ranges relevant to our
  training stage.
- **Kolberg et al., 2026.** *TabPFN-Wide: Continued Pre-Training
  for Extreme Feature Counts.* — the source of the
  `FeatureAgglomeration` design we use in `sanitize.py`.

Local code dumps under
[`repositories/`](repositories/REPOSITORIES.md) cover the public
TabPFN package, the docs site, the v2.5 / v2.6 HuggingFace model
cards, NanoTabPFN, the V2-Finetuning recipe, and the underlying PFN
framework. Read-only — do not edit.
