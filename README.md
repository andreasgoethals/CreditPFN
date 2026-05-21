# CreditPFN

Continued pretraining of TabPFN (v2.6 / v3) on a curated corpus
of real-world credit-risk datasets. The aim is to specialise the
tabular foundation model's in-context-learning prior toward the
structures, feature distributions, and label noise of credit-risk
data, and to test whether a credit-specialised foundation model
outperforms generalist TabPFN on downstream PD / LGD tasks.

The whole project is organised as a three-stage pipeline — **data →
train → eval** — where each stage has its own orchestrator, config
yaml, and result layout.

## Table of contents

1. [Overview](#1-overview)
2. [Background](#2-background)
3. [Quick start](#3-quick-start)
   - [3.1 Install](#31-install)
   - [3.2 Verify the install (local laptop is fine)](#32-verify-the-install-local-laptop-is-fine)
   - [3.3 Real training and eval require a CUDA cluster](#33-real-training-and-eval-require-a-cuda-cluster)
4. [Repository layout](#4-repository-layout)
   - [4.1 `src/` — pipeline source code](#41-src--pipeline-source-code)
   - [4.2 `config/` — three YAML configs, one per stage](#42-config--three-yaml-configs-one-per-stage)
   - [4.3 `scripts/` — CLI entrypoints and SLURM templates](#43-scripts--cli-entrypoints-and-slurm-templates)
   - [4.4 `notebooks/` — exploration and result visualisations](#44-notebooks--exploration-and-result-visualisations)
   - [4.5 `tests/` — unit and smoke tests](#45-tests--unit-and-smoke-tests)
   - [4.6 `docs/` — project documentation](#46-docs--project-documentation)
   - [4.7 `papers/` and `repositories/` — reference material](#47-papers-and-repositories--reference-material)
   - [4.8 `checkpoints/` — base and trained TabPFN weights (gitignored)](#48-checkpoints--base-and-trained-tabpfn-weights-gitignored)
   - [4.9 Runtime trees: `data/`, `output/`, `logs/` (gitignored)](#49-runtime-trees-data-output-logs-gitignored)
5. [Data pipeline](#5-data-pipeline)
6. [Training pipeline](#6-training-pipeline)
7. [Eval pipeline](#7-eval-pipeline)
8. [References](#8-references)

---

## 1. Overview

Three pipeline stages, each with one config yaml and one CLI script:

| Stage | Config | Orchestrator | What it does |
|---|---|---|---|
| **Data**  | [`config/data.yaml`](config/data.yaml)   | [`scripts/data_pipeline.py`](scripts/data_pipeline.py)   | Dedup → register → sanitize → dedup. Writes one sanitized CSV per dataset under `data/processed/`. CPU-only; ~10 minutes for the full 17 PD + 8 LGD corpus. |
| **Train** | [`config/train.yaml`](config/train.yaml) | [`scripts/train_pipeline.py`](scripts/train_pipeline.py) | Continued pretraining of every `(base × LR × LoRA)` tuple in the tunable grid. Reads sanitized CSVs directly, draws a fresh per-epoch subsample for each dataset. Writes finetuned `.ckpt` files + provenance + per-epoch CSVs. **Requires a CUDA GPU.** |
| **Eval**  | [`config/eval.yaml`](config/eval.yaml)   | [`scripts/eval_pipeline.py`](scripts/eval_pipeline.py)   | K-fold cross-validation of every model on every held-out test dataset (XGBoost, CatBoost, LogReg / LinReg, untuned and trained TabPFN). Writes one CSV per `(model × dataset × fold)`. **Requires a CUDA GPU.** |

The notebooks under `notebooks/` consume the outputs of all three
stages and drop publication-quality PDFs under `output/figures/`. See
chapter 4 for the per-folder breakdown.

---

## 2. Background

**TabPFN** is a transformer-based tabular foundation model that
performs in-context learning over entire tabular datasets in a single
forward pass. Each version ships two separate checkpoints:

- a **classifier** used here for **Probability of Default (PD)**, and
- a **regressor** used here for **Loss Given Default (LGD)**.

The two have different weights and must be adapted independently.

**Which base checkpoint?** Treated as a *training-stage
hyperparameter*, not a decision baked in at the data-pipeline stage.
The default sweep covers v3 (newest, synthetic-only) and v2.6
(synthetic-only). The v2.5 family was dropped on 2026-05-21 — its
loaded checkpoint exposes module names PEFT cannot suffix-match for
LoRA and its internal scaler produces NaN on constant columns. The
full inventory plus the citation chain that grounds each provenance
claim lives in [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

**Continued pretraining** — as introduced for tabular foundation
models in *Real-TabPFN* (Garg et al., 2025,
[arXiv:2507.03971](https://arxiv.org/abs/2507.03971)) — extends the
synthetic-prior pretraining of TabPFN with additional training on a
curated corpus of real tabular datasets from a target domain. This
project applies the same methodology to credit risk, with the
specific objectives:

1. **PD** — binary classification of whether an obligor will default
   within a given horizon.
2. **LGD** — regression of the fraction of exposure lost given default.

---

## 3. Quick start

> **You almost certainly cannot run the interesting parts of this
> repository on a laptop.** Continued pretraining and the cross-model
> benchmark both require a CUDA GPU with **≥ 16 GB VRAM** plus
> substantial system RAM. A laptop with a CPU-only Python install can
> run the **data pipeline**, the **test suite**, and open the
> **notebooks** against pre-existing outputs — useful for debugging
> and dataset curation, but nothing else.
>
> The real workflow lives on an HPC cluster. This project was
> developed against KU Leuven's VSC (Flemish Supercomputer Centre);
> step-by-step VSC-specific instructions live in
> [`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md). The notes below are
> general and will adapt to any SLURM-managed cluster with a CUDA GPU
> partition.

### 3.1 Install

Python **3.11 or 3.12** is required. `torch` and `scikit-learn` don't
ship Python-3.14 wheels yet, so a fall-back compile from source will
fail on most platforms.

```bash
python3.12 -m venv .venv --prompt CreditPFN          # Linux / macOS
# py -3.12 -m venv .venv --prompt CreditPFN          # Windows / PowerShell
source .venv/bin/activate                            # Linux / macOS
# .venv\Scripts\activate                             # Windows / PowerShell
pip install --upgrade pip
pip install -r requirements.txt

# TabPFN install gotcha (read this).
# This project's src/train/model.py uses the current Prior Labs API
# (4-tuple load_model_criterion_config return; version="v3"/"v2.6";
# download_if_not_exists kwarg). PyPI's `tabpfn` is pinned at 2.2.x with
# an older API and will TypeError on the first model load. Override:
pip install --upgrade "tabpfn @ git+https://github.com/PriorLabs/tabPFN.git@main"
```

### 3.2 Verify the install (local laptop is fine)

These three operate on CPU; a laptop is sufficient. Use them to make
sure the project builds and reads its data correctly before
submitting any cluster jobs.

```bash
pytest -q tests/                                   # ~5 min, no GPU needed
python scripts/data_pipeline.py                    # ~10 min, builds data/processed/
jupyter notebook notebooks/                        # open the data-exploration notebooks
```

### 3.3 Real training and eval require a CUDA cluster

The recommended entry point is the **chained submitter**. It loads
[`config/data.yaml`](config/data.yaml), resolves the storage tier
(`scratch` for fast-purge / `data` for durable), submits the data
job, then submits training and eval as dependents:

```bash
# On a cluster login node, after cloning + installing + uploading raw CSVs:
bash scripts/slurm/submit_full_pipeline.sh
```

To run a single stage by hand, the per-stage SLURM templates live in
[`scripts/slurm/`](scripts/slurm/). All of them read the
`paths.data_source` knob in [`config/data.yaml`](config/data.yaml)
to decide whether raw and processed data live on fast scratch or
durable storage — see chapter 5 for the data layout and
[`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md) for VSC-specific paths,
partitions, and a failure-mode cheat sheet.

Hydra-style CLI overrides work from any entrypoint (no yaml edits):

```bash
python scripts/train_pipeline.py track=lgd train.epochs=30
python scripts/eval_pipeline.py  --method xgboost  --test-dataset 0001.gmsc
```

---

## 4. Repository layout

```text
CreditPFN/
├── README.md
├── requirements.txt
├── src/                      pipeline source code         (see 4.1)
├── config/                   three YAML configs           (see 4.2)
├── scripts/                  CLI + SLURM templates        (see 4.3)
├── notebooks/                exploration + viz notebooks  (see 4.4)
├── tests/                    pytest suite                 (see 4.5)
├── docs/                     long-form documentation      (see 4.6)
├── papers/                   PDF library                  (see 4.7)
├── repositories/             upstream code dumps          (see 4.7)
├── checkpoints/              base + trained TabPFN .ckpt  (see 4.8, gitignored)
├── data/                     raw + processed corpus       (see 4.9, gitignored)
├── output/                   everything the code writes   (see 4.9, gitignored)
└── logs/                     one log file per task        (see 4.9, gitignored)
```

### 4.1 `src/` — pipeline source code

| Subpackage | Role | Public CLI |
|---|---|---|
| [`src/data/`](src/data)   | Four data-pipeline stages (dedup pre · register · sanitize · dedup post) plus `preprocessing.py` for per-dataset surgical fixes. Output is one sanitized CSV per dataset under `data/processed/`. | `python -m src.data.<stage>` for any one stage, or `scripts/data_pipeline.py` for the chain. |
| [`src/train/`](src/train) | The continued-pretraining loop: corpus split (`corpus.py`), the on-the-fly dataloader (`dataloader.py`) that reads sanitized CSVs and draws a fresh random subsample every epoch, TabPFN load/save + LoRA wrapping (`model.py`), training loop with per-epoch monitor (`loop.py`), metrics (`metrics.py`). | `scripts/train_pipeline.py` |
| [`src/model/`](src/model) | sklearn-style wrappers for every model the eval scores: XGBoost + CatBoost (with Optuna HPO), LogReg / LinReg (default-hyperparam baselines), TabPFN-untuned, TabPFN-trained. Single `base.py::BaselineModel` protocol so the eval loop stays model-agnostic. | importable only |
| [`src/eval/`](src/eval)   | The cross-model benchmark: processed-CSV loader, K-fold splitter with inner train/val, comprehensive metrics computation, results-dir routing, skip-existing rerun guard. | `scripts/eval_pipeline.py` |
| [`src/utils/`](src/utils) | Cross-cutting helpers: env-aware path resolver (`paths.py`), one-file-per-task run logging (`run_log.py`), notebook figure sink (`figures.py`), training / eval visualisation helpers (`training_viz.py`, `eval_viz.py`), upstream code refresh (`refresh_repositories.py`). | `python src/utils/refresh_repositories.py` |

### 4.2 `config/` — three YAML configs, one per stage

Every knob lives in one of three YAMLs. Each yaml is the single
source of truth for its stage; the corresponding script imports it
via OmegaConf and accepts Hydra-style overrides on the CLI
(`key.nested=value`).

| File | Drives | Main sections |
|---|---|---|
| [`config/data.yaml`](config/data.yaml)   | `src/data/*` + `scripts/data_pipeline.py` | paths (incl. `data_source: "scratch" \| "data"`), `finetuning.max_rows_per_epoch` + `query_fraction`, dedup detection thresholds, sanitize knobs (max missing rate, FeatureAgglomeration, LGD target clip) |
| [`config/train.yaml`](config/train.yaml) | `src/train/*` + `scripts/train_pipeline.py` | `tunable.*` (sweep axes: base checkpoint × LR × LoRA), corpus split (Mode A fractions / Mode B explicit IDs), optimizer + scheduler, LoRA cfg, train loop |
| [`config/eval.yaml`](config/eval.yaml)   | `src/eval/*` + `scripts/eval_pipeline.py` | enabled baselines, K-fold + inner-val fractions, per-fold Optuna budget, `max_rows_per_model` (per-architecture training-context cap), results dir |

What is **deliberately not in YAML**: anything that never changes
across runs (optimizer family AdamW, cosine schedule, metric column
order). Those are hardcoded in code — searchable constants near the
top of the relevant module.

### 4.3 `scripts/` — CLI entrypoints and SLURM templates

One orchestrator per pipeline stage, plus SLURM templates for the
cluster:

| File | What it does |
|---|---|
| [`scripts/data_pipeline.py`](scripts/data_pipeline.py)   | Run all four data stages end-to-end, or just the ones you ask for (`--datasets ...`). Idempotent; `--fresh` rebuilds from scratch. |
| [`scripts/train_pipeline.py`](scripts/train_pipeline.py) | Iterate the `cfg.tunable` cartesian grid; one trial per call when `--single` or `--trial-index` (SLURM array). Auto-fills missing sanitized CSVs by invoking the data pipeline for just those IDs. |
| [`scripts/eval_pipeline.py`](scripts/eval_pipeline.py)   | Score every model on every test dataset, K-fold CV. Skip-existing by default; `--rerun` to force. Filterable with `--method` / `--test-dataset` / `--task-index`. |
| [`scripts/slurm/*.slurm`](scripts/slurm/)               | SLURM templates: one per data / train / eval stage, plus `submit_full_pipeline.sh` for the chained submission. |

### 4.4 `notebooks/` — exploration and result visualisations

Four notebooks, two for data exploration (run after the data
pipeline) and two for training / eval visualisation (run after the
respective pipeline). Every notebook drops its figures as PDFs into
`output/figures/<notebook-slug>/` via the figure sink helper in
[`src/utils/figures.py`](src/utils/figures.py); the per-notebook
directory is **wiped on each re-run**, so stale figures never linger.
All plotting code lives in the corresponding helper module under
`src/utils/`; the notebook cells contain only function calls so the
narrative stays scannable and the logic stays testable.

| Notebook | What it shows |
|---|---|
| `0.0. raw_data_exploration.ipynb`          | What did the vendor deliver? Shapes, missing-rates, target distributions on raw CSVs. |
| `0.1. processed_data_exploration.ipynb`    | Did sanitisation produce sensible inputs? Same plots as 0.0 but on the post-sanitize CSVs. |
| `1.0. training_visualization.ipynb`        | All trained CreditPFN variants in one dashboard — per-trial loss / lr / metric curves, cross-trial overlays, LR sweep, LoRA effect, time/accuracy Pareto, convergence diagnostics, leaderboard. Consumes `output/training/`. |
| `2.0. final_results.ipynb`                 | The headline eval leaderboard — per-method box plots, per-dataset heatmaps, pairwise win-rate matrix (à la TabPFN-3 Fig 3), trained-vs-untuned scatter (à la Real-TabPFN), fold stability, threshold calibration. Consumes `output/results/`. |

Corpus summaries in the data notebooks are memoised so the first
cell pays the disk-read cost once and every subsequent plot reads
from RAM.

### 4.5 `tests/` — unit and smoke tests

```bash
pytest -q tests/    # ~5 min on a laptop; torch-dependent tests skip when torch is missing
```

One file per `src/` subpackage. Tests lean toward *failure-mode
coverage* over behavioural completeness: a few sharp tests that
catch real regressions if a future refactor breaks the contract.
Tests requiring a real TabPFN checkpoint on disk are guarded by
`pytest.importorskip("tabpfn")` so the suite stays runnable in a
stripped-down CI image.

| File | Coverage |
|---|---|
| [`tests/test_data.py`](tests/test_data.py)   | data pipeline (preprocessing → register → sanitize → dedup) + surgical-fix correctness per dataset |
| [`tests/test_paths.py`](tests/test_paths.py) | env-aware path resolution (local-vs-cluster routing) + `data_source` cfg knob |
| [`tests/test_train.py`](tests/test_train.py) | corpus split (`DatasetRef`), dataloader (`ProcessedDatasetLoader` including per-epoch reshuffle), LR schedule, descriptive name, end-to-end mocked training loop |
| [`tests/test_model.py`](tests/test_model.py) | baseline wrappers on synthetic data, model registry |
| [`tests/test_eval.py`](tests/test_eval.py)   | per-cell scoring, K-fold benchmark on synthetic processed CSVs, per-method CSV dirs, rerun-skip with full-fold semantics |

### 4.6 `docs/` — project documentation

| File | What it is |
|---|---|
| [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md)     | Inventory of every base `.ckpt` we ship (v2.6 / v3): training data (synthetic-only), sample/feature caps, layer counts, licence terms. Cross-referenced to HF model cards and the TabPFN-2.5 paper (which documents the architecture family). |
| [`docs/LITERATURE.md`](docs/LITERATURE.md)       | Chronological tour of every paper under `papers/`, with a "For CreditPFN" pointer per paper. The most directly relevant works (Real-TabPFN, TabPFNv2, TabPFN-2.5, TabPFN-3, Rubachev finetuning, TabPFN-Wide) are flagged at the top. |
| [`docs/REPOSITORIES.md`](docs/REPOSITORIES.md)   | What each `repositories/*.txt` dump is, why we keep it, and which lines to grep when designing each pipeline stage. Refresh script: `python src/utils/refresh_repositories.py`. |
| [`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md)         | **VSC-specific deployment guide** (KU Leuven's Vlaamse Supercomputer Centre): OnDemand portal, conda env, dataset upload, partition / GPU choice, the SLURM submit chain, failure-mode cheat sheet. Read this only when you're about to deploy on VSC; everything in this README applies to any SLURM cluster. |

### 4.7 `papers/` and `repositories/` — reference material

* [`papers/`](papers/) — PDF library of every paper we cite. The
  same set is summarised chronologically in
  [`docs/LITERATURE.md`](docs/LITERATURE.md) with extracted-text
  versions under `papers/text/` for grep-friendly search.
* [`repositories/`](repositories/) — flat-text dumps of the upstream
  Python packages we depend on (TabPFN, TabPFN extensions, the PFN
  reference implementation, NanoTabPFN, VSC documentation, …).
  Catalogued in [`docs/REPOSITORIES.md`](docs/REPOSITORIES.md);
  refreshed with `python src/utils/refresh_repositories.py`. These
  are read-only references — the project does not import any of them.

### 4.8 `checkpoints/` — base and trained TabPFN weights (gitignored)

* `checkpoints/*.ckpt` — base weights downloaded from Prior Labs
  (v2.6, v3 in both classifier and regressor flavours). The
  inventory and provenance live in
  [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md). The actual `.ckpt`
  files are gitignored because they're large; collaborators download
  them once during environment setup.
* `checkpoints/trained/{pd,lgd}/*.ckpt` — finetuned weights produced
  by `train_pipeline.py`. Each is paired with a
  `<file>.ckpt.provenance.json` sidecar that records every
  hyperparameter, the training/test dataset IDs, the GPU used, and
  the wall-clock time — readable via
  `src.train.model.load_provenance(path)`.

### 4.9 Runtime trees: `data/`, `output/`, `logs/` (gitignored)

These directories are populated by the pipeline scripts. They are
gitignored because the contents are large and machine-specific.

```text
data/                           # data pipeline input + sanitized output
├── raw/{pd,lgd}/<id>.csv       # hand-curated input corpus (you supply this)
├── processed/{pd,lgd}/         # <id>.sanitized.csv — the on-disk training input
├── dedup/                      # doubles_{track}_{pre,post}.csv (always durable)
└── manifest_{pd,lgd}.csv       # per-track dataset manifest (one row per dataset)

output/                         # everything the code writes (except trained .ckpt)
├── training/
│   ├── manifests/<run>_<track>.csv         one row per trial
│   └── epochs/<track>/<descriptive>.csv    per-epoch (loss, lr, train/test metric)
├── results/<TRACK>/<method>/<run>_<ts>.csv eval-pipeline CSVs (one per task)
└── figures/<notebook-slug>/*.pdf           per-notebook PDF figure dumps

logs/<task>_<ts>[_j<jid>_a<tid>].log        one log file per task (flat dir)
```

On a laptop, `data/` and `output/` live under the repo root. On a
cluster, they are split between fast and durable storage tiers
according to `paths.data_source` in `config/data.yaml`:

* `data_source: "scratch"` — `data/raw/`, `data/processed/` on fast
  scratch storage (subject to monthly purge); dedup files and
  manifests still on durable storage.
* `data_source: "data"` — everything on durable storage.

The eval results (`output/results/`), training manifests
(`output/training/`), figures (`output/figures/`), checkpoints, and
logs always live on durable storage regardless of `data_source`.

---

## 5. Data pipeline

Four stages, in order. The end-to-end driver is
[`scripts/data_pipeline.py`](scripts/data_pipeline.py); each stage can
also run independently via `python -m src.data.<stage>`. There is
**no `.npz` chunking step** — the sanitized CSV is the canonical
on-disk training input, and the training loop builds batches on the
fly.

| # | Module | Reads | Writes |
|---|---|---|---|
| 1 | [`src/data/dedup.py`](src/data/dedup.py) `--pass pre`        | `data/raw/{pd,lgd}/*.csv` | `data/dedup/doubles_{track}_pre.csv` |
| 2 | [`src/data/register.py`](src/data/register.py)               | raw CSVs + `DATASET_METADATA` | `data/manifest_{pd,lgd}.csv` |
| 3 | [`src/data/sanitize.py`](src/data/sanitize.py)               | raw CSVs + manifests | `data/processed/{pd,lgd}/<id>.sanitized.csv` |
| 4 | [`src/data/dedup.py`](src/data/dedup.py) `--pass post`       | processed CSVs | `data/dedup/doubles_{track}_post.csv` |

Plus one importable helper used by stages 2 and 3:

* [`src/data/preprocessing.py`](src/data/preprocessing.py) —
  `DATASET_METADATA` (target column, categorical hints, source) and
  per-dataset **surgical** fixes (drop ID columns, decode bespoke
  string formats, parse `"5yrs 3mon"` → integer months, remove
  target-leakage columns). Currently registers **17 PD + 8 LGD**
  datasets. No statistical operations here — no log-transforms, no
  scaling, no clipping.

### Stage descriptions

* **`dedup.py`** — eight detection methods per pass per track
  (identifier match, column-name Jaccard + identical shape,
  row-level pandas hash, column-level hash, rounded-row hash, subset
  detection, fuzzy column-name match). First-encountered wins.
* **`register.py`** — applies surgical fixes, then computes
  per-dataset metadata (n_rows / n_cols, missing rate, class balance,
  target mean/std, content-aware shape hash). Idempotent: re-running
  updates rows in place and preserves existing IDs not in the current
  filter.
* **`sanitize.py`** — surgical fixes, then a dataset-agnostic clean:
  drop exact-duplicate columns, drop > 90 %-NaN columns, drop constant
  columns, coerce numeric strings, cast numericals to float32
  (out-of-range values become NaN before the cast — no overflow
  warnings), ±inf → NaN, optional FeatureAgglomeration to ≤ 128
  columns (Ward linkage, unscaled per-cluster means), label-encode
  classification targets, clip LGD targets to [0, 1].

The eval pipeline reads the same sanitized CSVs but applies its own
K-fold CV split + per-model row cap (see chapter 7). The training
loop reads them via
`src/train/dataloader.py::ProcessedDatasetLoader`, which draws a
fresh random subsample of `finetuning.max_rows_per_epoch` rows from
each parent dataset every epoch and applies a context-only ordinal
encoder (so query categories unseen in context get `-1`, mirroring
TabPFN's inference scenario).

### What sanitize.py deliberately does NOT do

TabPFN's package handles these internally — see
[`docs/REPOSITORIES.md`](docs/REPOSITORIES.md) § "Outlier handling":

| Step | Why we don't pre-apply it |
|---|---|
| Outlier winsorisation | TabPFN's `OUTLIER_REMOVAL_STD = 12.0` (classifier) / `None` (regressor) handles outliers with the right semantics. |
| `PowerTransformer` / `QuantileTransformer` / `RobustScaler` | TabPFN's per-estimator inference ensemble cycles through these on every fit. |
| NaN imputation | `NanHandlingEncoderStep` handles NaNs natively (learned default + binary indicator). |
| Regression target z-normalisation | `RegressorBatch.znorm_space_bardist_` standardises the target internally and inverts at predict time. |

---

## 6. Training pipeline

A thin orchestrator over `src/train/`. The single source of truth for
hyperparameters is [`config/train.yaml`](config/train.yaml), in three
layers:

* **Tunable HPs** (`tunable.*` lists at the top) — base checkpoint,
  learning rate, LoRA on/off. Anything genuinely unknown in advance.
* **Fixed HPs** (single values below) — epochs, AMP, gradient
  clipping, warmup fraction, per-epoch monitor subsample. Follow
  TabPFN's `FinetunedTabPFNClassifier` defaults where those are
  well-tuned. The per-step subsample size lives in
  [`config/data.yaml`](config/data.yaml) (`finetuning.max_rows_per_epoch`,
  default 10 000 = Prior Labs' documented default).
* **Hardcoded in code** — optimizer family (AdamW), betas
  ((0.9, 0.999)), scheduler family (linear-warmup → cosine-decay).
  Never change between runs.

### Three invocation modes

```bash
# Cartesian product of all tunable lists (default; local sequential).
python scripts/train_pipeline.py

# One trial only — head of every tunable list. Good for smoke tests.
python scripts/train_pipeline.py --single

# One trial picked by index N. Designed for SLURM arrays.
python scripts/train_pipeline.py --trial-index $SLURM_ARRAY_TASK_ID

# How many trials does the current cfg expand to?
python scripts/train_pipeline.py --list-trials
```

Out-of-range trial indices exit zero cleanly (soft no-op), so an
over-sized SLURM array is safe.

### Auto-process hook

Before training starts, the pipeline checks that every dataset it
needs has a sanitized CSV on disk under `data/processed/<track>/`.
Missing CSVs trigger `scripts/data_pipeline.py` transparently for
just those IDs. Net effect: `train_pipeline.py` runs from a fresh
checkout and fills the processed corpus as needed.

### Configurable training datasets

Two paths into the train/test split, both in `cfg.corpus`:

* **Mode A — fraction-based** (default).
  `train_fraction` / `test_fraction` slice the registered corpus
  count-wise, deterministic in `cfg.seed`.

* **Mode B — explicit lists**. Set `train_dataset_ids` and/or
  `test_dataset_ids` to fix specific datasets in one or both buckets:

  ```yaml
  corpus:
    train_dataset_ids: ["0001.gmsc"]
    test_dataset_ids: []
  ```

  IDs unknown to `DATASET_METADATA` raise a clear error with the full
  list of valid IDs for the active track — no silent skips.

### Worked recipes

| Goal | Command |
|---|---|
| Debug, 1 dataset, 1 HP set                   | `python scripts/train_pipeline.py --single corpus.train_dataset_ids=[0001.gmsc] train.epochs=3` |
| Debug, 1 dataset, HP grid                    | `python scripts/train_pipeline.py corpus.train_dataset_ids=[0001.gmsc] train.epochs=3` |
| 5 specific PD datasets, 1 HP set             | `python scripts/train_pipeline.py --single track=pd corpus.train_dataset_ids='[0001.gmsc,0002.taiwan_creditcard,0003.vehicle_loan,0004.lendingclub,0009.bank_status]'` |
| Full corpus, 1 HP set                        | `python scripts/train_pipeline.py --single` |
| Full corpus, full HP grid                    | `python scripts/train_pipeline.py` |
| Full corpus, full HP grid, on the cluster    | `bash scripts/slurm/submit_full_pipeline.sh` — see [`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md) |

Hydra-style CLI overrides (`key=value`) write through the in-memory
cfg; they are NOT persisted to `config/train.yaml`. A debug run does
not break a teammate's next full run.

### Outputs

Each trial writes:

| Artefact                                                              | Path                                                                |
|-----------------------------------------------------------------------|---------------------------------------------------------------------|
| Final-epoch weights                                                   | `checkpoints/trained/<track>/<descriptive_name>.ckpt`               |
| Provenance sidecar (HPs, train/test IDs, GPU, walltime, …)            | `<descriptive_name>.ckpt.provenance.json`                           |
| Manifest row consumed by the eval pipeline                            | `output/training/manifests/<run_name>_<track>.csv`                  |
| Per-epoch CSV (epoch, train_loss, lr, train/test metric, epoch_time)  | `output/training/epochs/<track>/<descriptive_name>.csv`             |
| Full run log (slurm stdout + python logger)                           | `logs/train_<track>_<ts>[_j<jid>_a<tid>].log`                       |

Filename schema:
`<run_name>_<track>_<base-stem>_lr<lr>_seed<seed>[_lora].ckpt`.
Identical re-runs overwrite in place; trials with different HPs land
in distinct files.

### Trained-checkpoint provenance

Every saved checkpoint at `checkpoints/trained/<track>/*.ckpt` is
paired with a `<file>.ckpt.provenance.json` sidecar (and an identical
copy embedded under the `"provenance"` key inside the `.ckpt` itself)
recording:

- All hyperparameters used (base, lr, weight_decay, betas, scheduler,
  warmup fraction, epochs, accumulate, grad clip, amp, ctx/query
  sample sizes, seed, `use_lora` + LoRA config).
- Sorted training-dataset and test-dataset ID lists.
- Counts of train/test datasets.
- `training_time_seconds`, GPU name, `torch_version`, `tabpfn_version`,
  ISO-8601 `saved_at`.

Use `src.train.model.load_provenance(path)` to read either path
without loading the model weights.

### Design notes (the why)

* **Linear-warmup → cosine-decay LR** — matches HuggingFace's
  `get_cosine_schedule_with_warmup`, which is what TabPFN's
  `FinetunedTabPFNClassifier` uses internally. Verified in
  `tests/test_train.py::test_warmup_cosine_schedule_landmarks`.
* **No validation set** — with ~17 PD + ~8 LGD datasets, holding out
  a separate val bucket leaves so few datasets to fit on that the
  early-stopping signal becomes pure noise. We use fixed-epoch
  training and pick between hyperparameter settings *post hoc* on
  the test set in the eval stage.
* **Per-epoch monitor eval** — at the end of every epoch the loop
  scores the model on a small subsample of each train- and
  test-dataset (ROC-AUC for PD, RMSE for LGD). Cheap (~500 rows per
  chunk) but enough to see whether the model is still improving.

---

## 7. Eval pipeline

[`scripts/eval_pipeline.py`](scripts/eval_pipeline.py) scores every
model on every held-out test **dataset** using K-fold cross-validation
with an inner train/val split.

### Row-cap policy

Both the eval and training pipelines read the same sanitized CSVs
(`data/processed/{pd,lgd}/<id>.sanitized.csv`). The eval applies its
caps **inside each CV fold**, only to the training partition — the
held-out test partition is never capped, so the model predicts on
every row in one `predict_proba` call (TabPFN-v3's internal
`inference_row_chunk_size = 2048` handles arbitrarily large test
sets gracefully).

| Model family                          | Train-fold cap                                                              | Test fold      | HPO subsample                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------|----------------|--------------------------------------------------------------------------------|
| `tabpfn-untuned` / `tabpfn-trained`   | `cfg.max_rows_per_model[<v>]` (v3: 1 000 000; v2.x: 100 000)                | **full**       | n/a                                                                            |
| `xgboost` / `catboost`                | none                                                                        | full           | `cfg.hpo.<m>.max_rows = 50 000` (stratified subsample of inner-train; HPO only) |
| `logreg` / `linreg`                   | none                                                                        | full           | n/a                                                                            |

### CV semantics — 80 / 16 / 20 per fold

```text
Outer K-fold (cfg.cv.n_folds = 5):
    train     → 80% of dataset
    test      → 20%                  ← final metrics computed here

Inner split (cfg.cv.inner_val_fraction = 0.20):
    sub-train → 64% of dataset       ← model fits on this
    val       → 16%                  ← Optuna HPO objective + F1-threshold tuner
```

Optuna runs once per CV fold (5 studies per `(model × dataset)` at
`n_folds=5`), each with `hpo.<m>.n_trials` trials. LogReg / LinReg are
intentionally untuned — they're the "what does plain linear modelling
do" baseline.

### Test-dataset resolution

For `tabpfn-trained` models the test datasets come from each
checkpoint's `.provenance.json` (each checkpoint scored on its OWN
held-out set, recorded at training time). For `tabpfn-untuned` and
classical baselines the test datasets come from the cfg corpus split.
Both routes give the same set when seed and fractions match (the
default), so the comparison is apples-to-apples by construction.

### Comprehensive metrics (one row per model × dataset × fold)

Wide format; NaN where not applicable.

| Group                              | Columns                                                                                          |
|------------------------------------|--------------------------------------------------------------------------------------------------|
| Classification — threshold-free    | `roc_auc`, `log_loss`, `pr_auc`                                                                  |
| Classification — threshold-tuned   | `optimal_threshold` (max-F1 on inner-val), then `f1`, `accuracy`, `precision`, `recall` on test  |
| Regression                         | `rmse`, `mae`, `r2`, `neg_nll` (TabPFN-only)                                                     |
| Bookkeeping                        | `n_train_rows`, `n_val_rows`, `n_test_rows`, `elapsed_sec`, `status`, `error`, `timestamp`       |

### Re-runs are idempotent

Before scoring, each (model × dataset) pair is checked against
existing CSVs under `output/results/<TRACK>/<method-dirname>/`.
Pairs whose **all folds** are already `OK` are skipped. So:

- **First run** — scores every baseline + untuned + trained variant.
- **Re-run after adding a new trained checkpoint** — scores only the
  new checkpoint's pairs; baselines reuse rows from disk.

Force fresh scoring with `--rerun`. To rescore a single method, delete
its directory under `output/results/<TRACK>/` and re-submit.

### Filters and run modes

```bash
# Default — every model × every test dataset.
python scripts/eval_pipeline.py track=pd

# Only one method or one dataset.
python scripts/eval_pipeline.py track=pd --method xgboost --test-dataset 0001.gmsc

# SLURM array: ONE (model × dataset) per task.
N=$(python scripts/eval_pipeline.py --list-tasks track=pd)
sbatch --array=0-$((N - 1))%32 scripts/slurm/eval_pd.slurm
```

Out-of-range `--task-index` exits zero cleanly, so an over-sized
array doesn't fail.

### Results layout

Everything the code writes lives under `output/`. Trained
checkpoints stay under `checkpoints/trained/` so they can be wiped
independently. See section 4.9 for the full tree.

Method-directory names compress the published checkpoint filenames
(`tabpfn-v3-classifier-v3_default.ckpt` → `v3-default`); the
track-specific "classifier"/"regressor" infix is dropped because the
parent `PD/` or `LGD/` already encodes it. Trained variants append
`__lr<rate>[__lora]` so different HPs / LoRA modes land in different
folders.

Every benchmark invocation gets a fresh `<timestamp>` — earlier runs
are never overwritten. Aggregate with pandas:

```python
import pandas as pd, glob
files = glob.glob("output/results/PD/*/creditpfn_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.groupby(["model_name", "model_source"])[
    ["roc_auc", "f1", "log_loss", "rmse"]
].agg(["mean", "std", "count"])
```

---

## 8. References

The full paper library lives under [`papers/`](papers/) with a
chronological, detailed summary in
[`docs/LITERATURE.md`](docs/LITERATURE.md). The most directly relevant
works for this project:

- **Garg et al., 2025.** *Real-TabPFN — Improving Tabular Foundation
  Models via Continued Pre-training With Real-World Data.*
  [arXiv:2507.03971](https://arxiv.org/abs/2507.03971) — the recipe we
  follow.
- **Hollmann et al., 2025.** *Accurate predictions on small data with
  a tabular foundation model.* (Nature) — TabPFNv2 architecture.
- **Grinsztajn et al., 2025.** *TabPFN-2.5: Advancing the State of
  the Art in Tabular Foundation Models.*
  [arXiv:2511.08667](https://arxiv.org/abs/2511.08667) — the
  successor architecture used by our v2.6 / v3 checkpoints.
- **Grinsztajn et al., 2026.** *TabPFN-3: Technical Report.*
  [arXiv:2605.13986](https://arxiv.org/abs/2605.13986) — current
  generation, used by our `v3-default` base checkpoint.
- **Rubachev et al., 2025.** *On Finetuning Tabular Foundation
  Models.* — finetuning hyperparameter ranges that anchor our
  training stage.
- **Kolberg et al., 2026.** *TabPFN-Wide: Continued Pre-Training for
  Extreme Feature Counts.* — source of the `FeatureAgglomeration`
  design used in `sanitize.py`.

Local code dumps under
[`repositories/`](repositories/) (catalogued in
[`docs/REPOSITORIES.md`](docs/REPOSITORIES.md)) cover the public TabPFN
package, the docs site, the v2.5 / v2.6 / v3 HuggingFace model cards (v2.5 kept for scholarly reference; not used in our sweep),
NanoTabPFN, the V2-Finetuning recipe, and the underlying PFN
framework. Read-only — refresh with
`python src/utils/refresh_repositories.py`.
