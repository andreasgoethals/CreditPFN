# CreditPFN

Continued pretraining of TabPFN (v2.5 / v2.6 / v3) on a curated corpus
of real-world credit-risk datasets. The aim is to specialise the
tabular foundation model's in-context-learning prior toward the
structures, feature distributions, and label noise of credit-risk
data, and to test whether a credit-specialised foundation model
outperforms generalist TabPFN on downstream PD / LGD tasks.

The whole project is organised as a three-stage pipeline — **data →
train → eval** — where each stage has its own orchestrator, config
yaml, and result layout. Read the rest of this README in any order;
the section names mirror the directory names.

## Background

**TabPFN** is a transformer-based tabular foundation model that
performs in-context learning over entire tabular datasets in a single
forward pass. Each version ships two separate checkpoints:

- a **classifier** used here for **Probability of Default (PD)**, and
- a **regressor** used here for **Loss Given Default (LGD)**.

The two have different weights and must be adapted independently.

**Which base checkpoint?** Treated as a *training-stage
hyperparameter*, not a decision baked in at the data-pipeline stage.
The default sweep covers v3 (newest, synthetic-only), v2.6
(synthetic-only), and two v2.5 variants (one synthetic-only, one
real-finetuned). The full inventory plus the citation chain that
grounds each provenance claim lives in
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

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

## Repository layout

```text
CreditPFN/
├── README.md
├── requirements.txt
├── config/                          three YAMLs — one per stage
│   ├── data.yaml                    every knob for src/data/*
│   ├── train.yaml                   every knob for src/train/* + train_pipeline.py
│   └── eval.yaml                    every knob for src/eval/* + eval_pipeline.py
├── src/                             all the pipeline code (see below)
├── scripts/                         CLI entrypoints + SLURM templates
├── tests/                           ~230 unit + smoke tests, one file per src/ subpackage
├── docs/                            documentation (see below)
├── papers/                          PDF library
├── repositories/                    read-only reference corpus (upstream code dumps)
├── notebooks/                       three data-exploration notebooks
├── checkpoints/                     TabPFN base weights (gitignored)
│   └── trained/{pd,lgd}/            finetuned weights produced by train_pipeline (gitignored)
├── data/                            big I/O artefacts (gitignored)
│   ├── raw/{pd,lgd}/<id>.csv        hand-curated input corpus
│   ├── processed/{pd,lgd}/          <id>.sanitized.csv
│   ├── cached/{pd,lgd}/<id>/        chunk_NNN.npz + meta.json
│   └── dedup/                       doubles_{track}_{pre,post}.csv
├── manifests/<run>_<track>.csv      one row per trained checkpoint (gitignored)
├── results/                         (gitignored)
│   ├── benchmark/<TRACK>/<method>/  eval CSVs (one per run × task)
│   └── training/<track>/            per-epoch CSVs (loss, lr, train/test metric)
└── logs/<task>_<ts>[_j<jid>_a<tid>].log    one log file per task (flat dir)
```

### What's in `src/`

| Subpackage | Role | Public CLI |
|---|---|---|
| [`src/data/`](src/data)   | 5 data-pipeline stages (dedup · register · sanitize · dedup · dataset) plus `preprocessing.py` for per-dataset surgical fixes and `cache.py` for cache-state inspection. | `python -m src.data.<stage>` for any one stage, or `scripts/data_pipeline.py` for the chain. |
| [`src/train/`](src/train) | The continued-pretraining loop: corpus split (`corpus.py`), dataloader with per-chunk resample (`dataloader.py`), TabPFN load/save + LoRA wrapping (`model.py`), training loop with per-epoch eval (`loop.py`), metrics (`metrics.py`). | `scripts/train_pipeline.py` |
| [`src/model/`](src/model) | sklearn-style wrappers for every model the eval scores: XGBoost + CatBoost (with Optuna HPO), LogReg / LinReg (default-hyperparam baselines), TabPFN-untuned and TabPFN-trained. Single `base.py::BaselineModel` protocol so the eval loop stays model-agnostic. | importable only |
| [`src/eval/`](src/eval)   | The cross-model benchmark: processed-CSV loader, K-fold splitter with inner train/val, comprehensive metrics computation, results-dir routing, skip-existing rerun guard. | `scripts/eval_pipeline.py` |
| [`src/utils/`](src/utils) | Cross-cutting helpers: env-aware path resolver (`paths.py`), one-file-per-task run logging (`run_log.py`), repository corpus refresh (`refresh_repositories.py`). | `python src/utils/refresh_repositories.py` |

### What's in `config/`

Every knob lives in one of three YAMLs. Each yaml is the single source
of truth for its stage; the corresponding script imports it via
OmegaConf and accepts Hydra-style overrides on the CLI
(`key.nested=value`).

| File | Drives | Main sections |
|---|---|---|
| [`config/data.yaml`](config/data.yaml)   | `src/data/*` + `scripts/data_pipeline.py` | paths, dedup detection thresholds, sanitize knobs (max missing rate, FeatureAgglomeration, LGD target clip), chunking, cache fingerprint |
| [`config/train.yaml`](config/train.yaml) | `src/train/*` + `scripts/train_pipeline.py` | `tunable.*` (sweep axes: base checkpoint × LR × LoRA), corpus split (Mode A fractions / Mode B explicit IDs), optimizer + scheduler, LoRA cfg, train loop |
| [`config/eval.yaml`](config/eval.yaml)   | `src/eval/*` + `scripts/eval_pipeline.py` | enabled baselines, K-fold + inner-val fractions, per-fold Optuna budget, TabPFN row caps, results dir |

What is **deliberately not in YAML**: anything that never changes
across runs (optimizer family AdamW, cosine schedule, multi-chunk
policy, metric column order). Those are hardcoded in code — searchable
constants near the top of the relevant module.

### What's in `docs/`

| File | What it is |
|---|---|
| [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md)     | Inventory of every base `.ckpt` we ship: training data (synthetic vs real-finetuned), sample/feature caps, layer counts, licence terms. Cross-referenced to HF model cards and the v2.5 paper. |
| [`docs/LITERATURE.md`](docs/LITERATURE.md)       | Chronological tour of the 26 papers in `papers/`, with a "For CreditPFN" pointer per paper. The five most directly relevant works (Real-TabPFN, TabPFNv2, TabPFN-2.5, Rubachev finetuning, TabPFN-Wide) are flagged at the top. |
| [`docs/REPOSITORIES.md`](docs/REPOSITORIES.md)   | What each `repositories/*.txt` dump is, why we keep it, and which lines to grep when designing each pipeline stage. Refresh script: `python src/utils/refresh_repositories.py`. |
| [`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md)         | Step-by-step VSC deployment recipe: OnDemand portal, conda env, dataset upload to scratch, SLURM submit chain, failure-mode cheat sheet. Read only when you're about to run on the cluster. |

### What's in `scripts/`

One orchestrator per pipeline stage, plus SLURM templates for the
cluster:

| File | What it does |
|---|---|
| [`scripts/data_pipeline.py`](scripts/data_pipeline.py)   | Run all 5 data stages end-to-end, or just the ones you ask for (`--datasets ...`). Idempotent; `--fresh` to rebuild. |
| [`scripts/train_pipeline.py`](scripts/train_pipeline.py) | Iterate the `cfg.tunable` cartesian grid; one trial per call when `--single` or `--trial-index` (SLURM array). Auto-fills the cache for missing datasets. |
| [`scripts/eval_pipeline.py`](scripts/eval_pipeline.py)   | Score every model on every test dataset, K-fold CV. Skip-existing by default; `--rerun` to force. Filterable with `--method` / `--test-dataset` / `--task-index`. |
| [`scripts/slurm/*.slurm`](scripts/slurm/)               | SLURM templates: one per data / train / eval stage, plus `submit_full_pipeline.sh` for the chained submission. |

## Quick start

> **Python 3.12 strongly recommended.** Several deps (scikit-learn,
> parts of torch) don't ship Python-3.14 wheels yet, so `pip install`
> will try to compile from source and fail. Use `py -3.12` (Windows)
> or `python3.12` (Linux/macOS).

```bash
# 1. Create the venv (once).
py -3.12 -m venv .venv --prompt CreditPFN     # Windows / PowerShell
# python3.12 -m venv .venv --prompt CreditPFN # Linux / macOS

.venv/Scripts/activate              # Windows / PowerShell
# source .venv/bin/activate         # Linux / macOS
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠ **TabPFN version caveat.** PyPI's `tabpfn` caps at `2.2.1` with an
> older API than this project's training code expects (4-tuple
> `load_model_criterion_config` return, `version="v3"`,
> `download_if_not_exists` kwarg). After `pip install -r
> requirements.txt`, install the matching Prior Labs wheel on top:
>
> ```bash
> pip install --upgrade "tabpfn @ git+https://github.com/PriorLabs/tabPFN.git@main"
> ```
>
> Skip this and `train_pipeline.py` will `TypeError` on the first
> model load. Eval against pre-existing checkpoints is unaffected.

```bash
# 2. Run the data pipeline end-to-end (one-time; idempotent).
python scripts/data_pipeline.py
# python scripts/data_pipeline.py --fresh
# python scripts/data_pipeline.py --datasets 0001.gmsc

# 3. Continued pretraining (auto-fills missing cache).
python scripts/train_pipeline.py              # full cartesian grid
# python scripts/train_pipeline.py --single   # one trial (head of every tunable list)
# python scripts/train_pipeline.py track=lgd  # LGD regressor

# 4. Cross-model benchmark on the held-out test split.
python scripts/eval_pipeline.py track=pd
python scripts/eval_pipeline.py track=lgd

# 5. Tests.
pytest -q tests/
```

For cluster deployment see [`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md) —
SLURM chain, dataset upload to scratch, failure-mode cheat sheet.

## Data pipeline

Five stages, in order. The end-to-end driver is
[`scripts/data_pipeline.py`](scripts/data_pipeline.py); each stage can
also run independently via `python -m src.data.<stage>`.

| # | Module | Reads | Writes |
|---|---|---|---|
| 1 | [`src/data/dedup.py`](src/data/dedup.py) `--pass pre`        | `data/raw/{pd,lgd}/*.csv` | `dedup/doubles_{track}_pre.csv` |
| 2 | [`src/data/register.py`](src/data/register.py)               | raw CSVs + `DATASET_METADATA` | `manifest_{pd,lgd}.csv` |
| 3 | [`src/data/sanitize.py`](src/data/sanitize.py)               | raw CSVs + manifests | `data/processed/{pd,lgd}/<id>.sanitized.csv` |
| 4 | [`src/data/dedup.py`](src/data/dedup.py) `--pass post`       | processed CSVs | `dedup/doubles_{track}_post.csv` |
| 5 | [`src/data/dataset.py`](src/data/dataset.py)                 | processed CSVs + manifests | `data/cached/{track}/<id>/chunk_NNN.npz` + `meta.json` |

Plus one importable helper used by stages 2 and 3:

* [`src/data/preprocessing.py`](src/data/preprocessing.py) —
  `DATASET_METADATA` (target column, categorical hints, source) and
  per-dataset **surgical** fixes (drop ID columns, decode bespoke
  string formats, parse `"5yrs 3mon"` → integer months, remove
  target-leakage columns). Currently registers **17 PD + 8 LGD**
  datasets. No statistical operations here — no log-transforms, no
  scaling, no clipping.

### One-sentence stage descriptions

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
  columns, coerce numeric strings, cast numericals to float32,
  ±inf → NaN, optional FeatureAgglomeration to ≤ 128 columns (Ward
  linkage, unscaled per-cluster means), label-encode classification
  targets, clip LGD targets to [0, 1].
* **`dataset.py`** — chunks each sanitised dataset into ≤
  `cfg.dataset.max_rows_per_chunk` rows (stratified for PD, random
  for LGD), splits each chunk 60 % context / 40 % query, ordinal-
  encodes categoricals **with the encoder fit on context only**
  (so query categories unseen in context get `-1`, mirroring TabPFN's
  inference scenario), writes one `.npz` per chunk plus a
  `meta.json` sidecar.

### What sanitize.py deliberately does NOT do

TabPFN's package handles these internally — see
[`docs/REPOSITORIES.md`](docs/REPOSITORIES.md) § "Outlier handling":

| Step | Why we don't pre-apply it |
|---|---|
| Outlier winsorisation | TabPFN's `OUTLIER_REMOVAL_STD = 12.0` (classifier) / `None` (regressor) handles outliers with the right semantics. |
| `PowerTransformer` / `QuantileTransformer` / `RobustScaler` | TabPFN's per-estimator inference ensemble cycles through these on every fit. |
| NaN imputation | `NanHandlingEncoderStep` handles NaNs natively (learned default + binary indicator). |
| Regression target z-normalisation | `RegressorBatch.znorm_space_bardist_` standardises the target internally and inverts at predict time. |

### Data-exploration notebooks

Under [`notebooks/`](notebooks/), designed to scale to the
3 000-dataset corpus:

* `0.0. raw_data_exploration.ipynb` — what did the vendor deliver?
* `0.1. processed_data_exploration.ipynb` — did sanitisation produce
  sensible inputs?
* `0.2. cached_data_exploration.ipynb` — is the `.npz` cache healthy
  for training?

Corpus summaries are memoised so the first cell pays the disk-read
cost once and every subsequent plot reads from RAM.

## Training pipeline

A thin orchestrator over `src/train/`. The single source of truth for
hyperparameters is [`config/train.yaml`](config/train.yaml), in three
layers:

* **Tunable HPs** (`tunable.*` lists at the top) — base checkpoint,
  learning rate, LoRA on/off. Anything genuinely unknown in advance.
* **Fixed HPs** (single values below) — epochs, AMP, gradient
  clipping, warmup fraction, sample sizes, per-epoch eval subsample.
  Follow TabPFN's `FinetunedTabPFNClassifier` defaults where those
  are well-tuned.
* **Hardcoded in code** — optimizer family (AdamW), betas
  ((0.9, 0.999)), scheduler family (linear-warmup → cosine-decay),
  multi-chunk policy (`first_chunk_only`). Never change between runs.

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

### Auto-cache hook

Before training starts, the pipeline checks that every dataset it
needs is materialised under `data/cached/<track>/<id>/`. Missing
datasets trigger `scripts/data_pipeline.py` transparently for just
those IDs. Net effect: `train_pipeline.py` runs from a fresh checkout
and fills the cache as needed.

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

| Artefact                                                              | Path                                                             |
|-----------------------------------------------------------------------|------------------------------------------------------------------|
| Final-epoch weights                                                   | `checkpoints/trained/<track>/<descriptive_name>.ckpt`            |
| Provenance sidecar (HPs, train/test IDs, GPU, walltime, …)           | `<descriptive_name>.ckpt.provenance.json`                        |
| Manifest row consumed by the eval pipeline                            | `manifests/<run_name>_<track>.csv`                               |
| Per-epoch CSV (epoch, train_loss, lr, train/test metric, epoch_time)  | `output/training/epochs/<track>/<descriptive_name>.csv`                |
| Full run log (slurm stdout + python logger)                           | `logs/train_<track>_<ts>[_j<jid>_a<tid>].log`                    |

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
- Number of train/test chunks.
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

## Eval pipeline

[`scripts/eval_pipeline.py`](scripts/eval_pipeline.py) scores every
model on every held-out test **dataset** using K-fold cross-validation
with an inner train/val split.

### Why processed CSVs (not cached chunks)

The `.npz` chunks are sized for TabPFN's in-context inference at
*training* time. For *evaluation*, XGBoost / CatBoost have no
row-count limit and would be underestimated if capped at the chunk
size. So the eval reads
`data/processed/{pd,lgd}/<id>.sanitized.csv` directly, with the cap
policy:

| Model family                          | Pre-CV cap                         | Final fit + test eval | HPO subsample                           |
|---------------------------------------|------------------------------------|-----------------------|-----------------------------------------|
| `tabpfn-untuned` / `tabpfn-trained`   | `cfg.max_rows_tabpfn = 100 000`    | uses the capped data  | n/a                                     |
| `xgboost` / `catboost`                | none                               | full dataset          | `cfg.hpo.<m>.max_rows = 50 000` (stratified subsample of inner-train; HPO only) |
| `logreg` / `linreg`                   | none                               | full dataset          | n/a                                     |

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

```text
results/
├── benchmark/                                          eval pipeline output
│   ├── PD/
│   │   ├── xgboost/                                    creditpfn_<ts>__task<i>_ds-<id>.csv
│   │   ├── catboost/                                   …
│   │   ├── logreg/                                     …
│   │   ├── tabpfn-untuned__v3-default/                 …
│   │   └── tabpfn-trained__v3-default__lr1e-04/        …
│   └── LGD/  …
└── training/                                           train pipeline output (per-epoch)
    ├── pd/   creditpfn_pd_<base-stem>_lr1e-04_seed42.csv
    └── lgd/  …
```

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

## Tests

```bash
pytest -q tests/    # ~230 tests, ~3.5 min
```

| File | Coverage |
|---|---|
| `test_data.py`  | data pipeline (preprocessing → register → sanitize → dedup → dataset) + surgical-fix correctness per dataset |
| `test_paths.py` | env-aware path resolution (local-vs-VSC routing) |
| `test_train.py` | corpus split, dataloader, LR schedule, descriptive name, end-to-end mocked training loop |
| `test_model.py` | cache helper, baseline wrappers on synthetic data, model registry |
| `test_eval.py`  | per-cell scoring, K-fold benchmark on synthetic chunks, per-method CSV dirs, rerun-skip with full-fold semantics |

The suite leans toward *failure-mode coverage* over behavioural
completeness. Tests requiring a real TabPFN checkpoint on disk are
guarded by `pytest.importorskip` so the suite stays runnable in a
stripped-down CI image.

## References

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
- **Rubachev et al., 2025.** *On Finetuning Tabular Foundation
  Models.* — finetuning hyperparameter ranges that anchor our
  training stage.
- **Kolberg et al., 2026.** *TabPFN-Wide: Continued Pre-Training for
  Extreme Feature Counts.* — source of the `FeatureAgglomeration`
  design used in `sanitize.py`.

Local code dumps under
[`repositories/`](repositories/) (catalogued in
[`docs/REPOSITORIES.md`](docs/REPOSITORIES.md)) cover the public TabPFN
package, the docs site, the v2.5 / v2.6 HuggingFace model cards,
NanoTabPFN, the V2-Finetuning recipe, and the underlying PFN
framework. Read-only — refresh with
`python src/utils/refresh_repositories.py`.
