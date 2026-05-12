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
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md). When
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

## Project layout (key directories)

```
$CREDITPFN_DATA_ROOT/          (= scratch on VSC, repo locally)
├── data/
│   ├── raw/{pd,lgd}/<id>.csv          input corpus
│   ├── processed/{pd,lgd}/             sanitize.py output
│   └── cached/{pd,lgd}/<id>/           dataset.py output (.npz + meta.json)

$CREDITPFN_OUTPUT_ROOT/        (= $VSC_DATA on VSC, repo locally)
├── dedup/                              within-track duplicate sweeps
├── manifest_pd.csv | manifest_lgd.csv  per-track dataset metadata
├── manifests/<run_name>_<track>.csv    one row per trained checkpoint
├── checkpoints/trained/<track>/        finetuned weights + .provenance.json sidecars
├── results/benchmark/{PD,LGD}/<method>/ benchmark CSVs, one per (run × task)
├── results/training/<track>/            per-epoch (loss, lr, elapsed) CSVs, one per trial
└── logs/<task>_<ts>[_j<JID>_a<TID>].log  one log file per task (flat dir)
```

## Compute

**Quick reference**: see [`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md) for
the full step-by-step VSC deployment recipe (one-time setup, the
`data → train → eval` chain in one command, three common workflows,
and a failure-mode cheat sheet).

Training runs on the VSC (KU Leuven) supercomputer:
- **Data preprocessing** → Genius `batch` partition (CPU, 8 cores, 40 GB).
- **Continued pretraining** → wICE `gpu_h100` partition (NVIDIA H100 NVL, 96 GB).
  One SLURM array task per training trial; trials run in parallel.
- **Eval/benchmark** → wICE `gpu_h100` (one job per track).

Slurm templates live under `scripts/slurm/`. The full chain
(data → train arrays → eval) is one command:
`bash scripts/slurm/submit_full_pipeline.sh`.

### Local vs. VSC paths

The same code base runs in two storage worlds — laptop and VSC —
and **auto-detects which one it's in.** Two layers, in order:

1. **Explicit override.** If `CREDITPFN_DATA_ROOT` /
   `CREDITPFN_OUTPUT_ROOT` are set in the environment, the resolver
   uses them. The slurm scripts in `scripts/slurm/` set both
   explicitly so a slurm-driven run is fully under user control.

2. **VSC auto-detection.** Otherwise, if `$VSC_DATA` is set
   (= we're on a VSC node — the KU Leuven login profile sets it
   unconditionally), the resolver picks VSC defaults:

       CREDITPFN_DATA_ROOT   → $VSC_SCRATCH/CreditPFN     (big I/O)
       CREDITPFN_OUTPUT_ROOT → $VSC_DATA/CreditPFN        (durable)

   So even if you SSH into a login node and just run
   `python scripts/data_pipeline.py` interactively (no slurm), the
   right thing happens — no env-var setup needed.

3. **Local fallback.** Neither (1) nor (2) → the repo root.
   Laptops never set `$VSC_DATA`, so every artefact lands under
   `<repo>/data/`, `<repo>/logs/`, `<repo>/results/`, etc.,
   exactly as the dev workflow expects.

The split between the two roots:

| Resolver               | Used for                                                                                            | Where it routes on VSC               | Where it routes locally |
|------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------|-------------------------|
| `resolve_data_path`    | big I/O artefacts: `data/raw`, `data/processed`, `data/cached`                                      | `$VSC_SCRATCH/CreditPFN`             | repo root               |
| `resolve_output_path`  | durable artefacts: `dedup/`, `manifest_*.csv`, `checkpoints/trained/`, `results/`, `logs/`, `manifests/` | `$VSC_DATA/CreditPFN`                | repo root               |

Datasets are too big for the `$VSC_DATA` quota so they live on
`$VSC_SCRATCH` (large, parallel BeeGFS, no backup). Everything that
must survive a scratch purge — dedup CSVs, the per-track manifest
of trained checkpoints, the checkpoints themselves, the benchmark
results, every log file — lives on `$VSC_DATA` (backed up nightly).
Verified by `tests/test_paths.py`.

### Logs: one flat directory, one file per task

Every slurm job (and every local script invocation) produces
**exactly one** log file:

```
$OUTPUT_ROOT/logs/<task>_<YYYYMMDD>_<HHMMSS>[_j<JOBID>_a<TASKID>].log
```

— flat, no subfolders. The slurm scripts use `exec > "$LOG" 2>&1` so
that bash echos, `nvidia-smi`, the python orchestrator's stdout, and
the per-step training loop's logger calls all land in the same file.
Slurm's own `--output=/dev/null` so we don't get a competing stub
file. Locally, the python `setup_logging()` helper attaches both a
`StreamHandler` (live stdout) and a `FileHandler` to the timestamped
file; under slurm the FileHandler is suppressed (bash already routed
stdout to the log file, no double-write).

### Trained-checkpoint provenance

Every saved checkpoint at
`checkpoints/trained/<track>/<descriptive_name>.ckpt` is paired with a
sidecar `<descriptive_name>.ckpt.provenance.json` that records:

- All hyperparameters used (base, lr, weight_decay, betas, scheduler
  type + warmup fraction, epochs, accumulate_grad_batches, grad clip,
  amp, ctx/query sample sizes, seed). The multi-chunk policy is fixed
  to `first_chunk_only` and recorded in the sidecar for completeness.
- The list of training datasets (sorted dataset_ids)
- The list of test datasets (sorted dataset_ids)
- Number of train/test chunks
- `training_time_seconds` (wall-clock)
- The specific GPU (`torch.cuda.get_device_name(0)`, e.g. `"NVIDIA H100 NVL"`)
- `torch_version`, `tabpfn_version`
- `saved_at` ISO-8601 timestamp

The same dict is also embedded inside the `.ckpt` itself under the
`"provenance"` key (alongside `state_dict` and `config`), so the
checkpoint is fully self-describing — even moved years from now,
`torch.load(...)["provenance"]` recovers everything. Use
`src.train.model.load_provenance(path)` to read either path
without loading the model weights.

## Repository layout

```
CreditPFN/
├── README.md
├── .gitignore
├── requirements.txt
├── checkpoints/                  TabPFN base weights + CHECKPOINTS.md
│   └── trained/{pd,lgd}/         continued-pretrained weights from train_pipeline
├── config/
│   ├── data.yaml                 every knob for src/data/*
│   ├── train.yaml                every knob for src/train/* + scripts/train_pipeline.py
│   └── eval.yaml                 every knob for src/eval/* + scripts/eval_pipeline.py
├── data/                         (gitignored)
│   ├── raw/{pd,lgd}/<id>.csv     hand-curated input corpus
│   ├── processed/{pd,lgd}/       <id>.sanitized.csv (sanitize.py output)
│   ├── cached/{pd,lgd}/<id>/     chunk_NNN.npz + meta.json (dataset.py output)
│   ├── dedup/                    doubles_{track}_{pre,post}.csv (dedup.py output)
│   ├── manifest_pd.csv           register.py output (PD)
│   └── manifest_lgd.csv          register.py output (LGD)
├── logs/                         per-run logs (one file per task)
│   └── <task>_<YYYYMMDD>_<HHMMSS>[_j<JID>_a<TID>].log
├── manifests/                    one CSV per (run × track), produced by train_pipeline
│   └── <run_name>_<track>.csv    training manifest (one row per trained ckpt)
├── results/
│   ├── benchmark/<TRACK>/<method>/<run>_<ts>__task<N>_ds-<id>.csv   eval CSVs
│   └── training/<track>/<descriptive_name>.csv                     per-epoch CSVs
├── papers/                       PDF library + Literature.md (chronological summary)
├── repositories/                 read-only reference corpus + REPOSITORIES.md
├── scripts/
│   ├── data_pipeline.py          end-to-end data orchestrator (5 stages + logging)
│   ├── train_pipeline.py         continued-pretraining orchestrator (single / grid / slurm-array)
│   ├── eval_pipeline.py          cross-model benchmark on the held-out test split
│   └── slurm/                    SLURM array files (train_pd, train_lgd, eval)
├── src/
│   ├── data/                     Stage 1–5 modules + cache helper
│   ├── train/                    continued-pretraining loop, corpus split, dataloader, model loader
│   ├── model/                    baseline + TabPFN wrappers (XGB, CatBoost, LogReg, LinReg, TabPFN-untuned/trained)
│   ├── eval/                     benchmark.py — score every model on every test chunk
│   └── utils/                    run-log helper, etc.
├── notebooks/                    three exploration notebooks
│   ├── 0.0. raw_data_exploration.ipynb        — what did the vendor deliver?
│   ├── 0.1. processed_data_exploration.ipynb  — did sanitize produce sensible inputs?
│   └── 0.2. cached_data_exploration.ipynb     — is the .npz cache training-ready?
└── tests/                        smoke + unit tests, flat (one file per src/ subpackage)
    ├── test_data.py
    ├── test_train.py
    ├── test_model.py
    └── test_eval.py
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
```

> ⚠ **TabPFN version caveat.** PyPI's `tabpfn` package currently caps
> at `2.2.1` (versions ≥ 6 on PyPI are an unrelated package that took
> the name). The training code in `src/train/` is written against the
> newer Prior Labs API documented in `repositories/TabPFN .txt`
> (4-tuple `load_model_criterion_config` return, `version="v3"`,
> `download_if_not_exists` kwarg). After running the line above, install
> the matching wheel manually:
>
> ```bash
> # Example — adjust to whichever wheel Prior Labs has shipped you.
> pip install --upgrade "tabpfn @ git+https://github.com/PriorLabs/tabPFN.git@main"
> ```
>
> Skip this and `python scripts/train_pipeline.py` will fail with a
> TypeError on the first model load. Eval against pre-existing
> checkpoints is unaffected.

```bash
# 2. Run the full data pipeline end-to-end (one-time; idempotent).
python scripts/data_pipeline.py               # incremental (skip valid cache)
# python scripts/data_pipeline.py --fresh     # rebuild from scratch
# python scripts/data_pipeline.py --datasets 0001.gmsc 0001.heloc

# 3. Continued pretraining (auto-runs the data pipeline for any
#    missing dataset; idempotent if everything is already cached).
python scripts/train_pipeline.py              # full cartesian grid
# python scripts/train_pipeline.py --single   # one trial
# python scripts/train_pipeline.py track=lgd  # train the LGD regressor

# 4. Cross-model benchmark on the held-out test split.
python scripts/eval_pipeline.py track=pd
python scripts/eval_pipeline.py track=lgd

# 5. Run the tests.
pytest -q tests/

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
  columns, decodes hand-crafted strings, parses dates, ordinal-maps
  credit grades (A..G → 0..6), and removes target-leakage columns for
  every registered dataset. Currently covers **17 PD** + **8 LGD**
  datasets (the SBA dataset is registered once per track because it
  carries both a binary default label and a charge-off principal that
  derives the LGD target). *No* statistical operations: no
  log-transforms, no scaling, no clipping.
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

## Training pipeline

The training pipeline is a thin orchestrator over `src/train/`. The
single source of truth for hyperparameters is
[`config/train.yaml`](config/train.yaml), structured in two layers:

* **Tunable HPs** (lists at the top of the file) — base checkpoint,
  learning rate. Anything that is genuinely unknown in advance and
  must be picked empirically.
* **Fixed HPs** (single values, below) — epochs, AMP, gradient
  clipping, warmup fraction, sample sizes. These follow TabPFN's own
  `FinetunedTabPFNClassifier` defaults wherever those are well-tuned.
* **Hardcoded in code** — optimizer family (AdamW), betas
  ((0.9, 0.999)), scheduler family (warmup → cosine), multi-chunk
  policy (`first_chunk_only`). These never change between runs, so
  they live in `src/train/loop.py`, not in YAML.

The script — *not* `src/train/` — decides what to do with the
tunable lists. Three modes:

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

### Auto-cache hook

Before training starts, the pipeline checks that every dataset it
needs is materialised under `data/cached/<track>/<id>/`. If anything
is missing, `scripts/data_pipeline.py` is invoked transparently for
just those IDs. Net effect: you can run `train_pipeline.py` straight
on a fresh checkout and it will fill the cache as needed. Running
the data pipeline up-front is still recommended for large corpora
(cleaner logs, deterministic timing).

### Configurable training datasets

Two paths into the train/test split, both in `cfg.corpus`:

* **Mode A — fraction-based** (default).
  `train_fraction` / `test_fraction` slice the registered corpus
  count-wise, deterministic in `cfg.seed`.

* **Mode B — explicit lists**. Set `train_dataset_ids` and/or
  `test_dataset_ids` to fix specific datasets in one or both
  buckets:

  ```yaml
  corpus:
    train_dataset_ids: ["0001.gmsc"]            # debug: train on one dataset
    test_dataset_ids: []                         # remaining → test
  ```

  An ID may not appear in both lists (raises). Useful for debugging
  the procedure on a single dataset before launching a real run, or
  for an outer driver that wants full control over the split.

### Worked recipes

The same script + config drives every workflow you'll need:

| Goal | Command |
|---|---|
| **Debug, 1 dataset, 1 HP set** | `python scripts/train_pipeline.py --single corpus.train_dataset_ids=[0001.gmsc] train.epochs=3` |
| **Debug, 1 dataset, HP grid** | `python scripts/train_pipeline.py corpus.train_dataset_ids=[0001.gmsc] train.epochs=3` |
| **Continued pretraining on 5 specific PD datasets, 1 HP set** | `python scripts/train_pipeline.py --single track=pd corpus.train_dataset_ids='[0001.gmsc,0002.taiwan_creditcard,0003.vehicle_loan,0004.lendingclub,0009.bank_status]'` |
| **Continued pretraining on 5 specific PD datasets, HP grid** | `python scripts/train_pipeline.py track=pd corpus.train_dataset_ids='[0001.gmsc,0002.taiwan_creditcard,0003.vehicle_loan,0004.lendingclub,0009.bank_status]'` |
| **Full corpus, 1 HP set** | `python scripts/train_pipeline.py --single` |
| **Full corpus, full HP grid** | `python scripts/train_pipeline.py` |
| **Full corpus, full HP grid, parallelised on VSC** | `bash scripts/slurm/submit_full_pipeline.sh` |

Notes:

* Hydra-style overrides on the right-hand-side of the command write
  through the in-memory `cfg`; they are NOT persisted to
  `config/train.yaml`. So a debug run does not break a teammate's
  next full run.
* The auto-cache hook only materialises the datasets the run actually
  needs. So the 5-dataset debug runs above don't trigger preprocessing
  for the other ~10 PD datasets.
* `--single` picks the head of every tunable list — a quick way to
  confirm the loop runs end-to-end before launching the full grid.

### SLURM (parallelised training on VSC)

One trial per SLURM task, dispatched via array index. The repo
ships:

| File                                    | What it submits                                                                |
|-----------------------------------------|--------------------------------------------------------------------------------|
| `scripts/slurm/data.slurm`              | One CPU job: full data pipeline (Genius `batch`).                                |
| `scripts/slurm/train_pd.slurm`          | Array job: one **trial** per task, PD track (wICE `gpu_h100`).                   |
| `scripts/slurm/train_lgd.slurm`         | Array job: one **trial** per task, LGD track (wICE `gpu_h100`).                  |
| `scripts/slurm/eval_pd.slurm`           | Array job: one **(model × test_dataset)** per task, PD track.                    |
| `scripts/slurm/eval_lgd.slurm`          | Array job: one **(model × test_dataset)** per task, LGD track.                   |
| `scripts/slurm/submit_full_pipeline.sh` | Submits all of the above with `--dependency=afterok:` chaining.                  |

Inside each slurm file the convention is identical:

```bash
#SBATCH --output=/dev/null              # let bash's `exec >` own the log file
#SBATCH --error=/dev/null

set -euo pipefail
export PYTHONUNBUFFERED=1
export CREDITPFN_DATA_ROOT="${VSC_SCRATCH}/CreditPFN"
export CREDITPFN_OUTPUT_ROOT="${VSC_DATA}/CreditPFN"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/<task>_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1
# … env activation, echos, python -u <script> --log-path "$LOG" …
```

End-to-end submit:

```bash
ssh login.hpc.kuleuven.be
cd $VSC_DATA/CreditPFN
bash scripts/slurm/submit_full_pipeline.sh
```

Internally it does:

```bash
DATA_JID=$(sbatch --parsable scripts/slurm/data.slurm)
N_PD=$(python scripts/train_pipeline.py --list-trials track=pd)
TRAIN_PD_JID=$(sbatch --parsable --dependency=afterok:$DATA_JID \
                  --array=0-$((N_PD - 1))%4 scripts/slurm/train_pd.slurm)
N_PD_EVAL=$(python scripts/eval_pipeline.py --list-tasks track=pd)
sbatch --dependency=afterok:$TRAIN_PD_JID \
       --array=0-$((N_PD_EVAL - 1))%32 scripts/slurm/eval_pd.slurm
# … same for LGD (eval_lgd.slurm)
```

Each training-array task runs:

```bash
python scripts/train_pipeline.py --trial-index $SLURM_ARRAY_TASK_ID track=pd
```

and appends one row to `manifests/<run_name>_<track>.csv` — the
manifest the eval pipeline reads — plus a per-epoch CSV under
`results/training/<track>/<descriptive_name>.csv`. **Failures don't
bring down the chain**: if trial 7 of 9 fails, trials 0..8 still ran,
the manifest still has 8 OK rows, and the eval will benchmark every
checkpoint that landed.

### Outputs

| File / dir | What it is |
|---|---|
| `checkpoints/trained/<track>/<descriptive_name>.ckpt` | Final-epoch weights. Filename encodes track, base, lr, policy, seed. Round-trips through `TabPFNClassifier(model_path=...)`. **Permanent — kept for life of the project.** |
| `checkpoints/trained/<track>/<descriptive_name>.ckpt.provenance.json` | Sidecar with HPs, training datasets, training time, GPU, etc. (See "Trained-checkpoint provenance" above.) |
| `manifests/<run_name>_<track>.csv` | One row per trained config. Read by the eval pipeline. |
| `logs/<task>_<YYYYMMDD>_<HHMMSS>[_j<JID>_a<TID>].log` | One log file per task — flat directory, captures slurm boilerplate + python output + training loop in one place. |

On VSC these all sit under `$CREDITPFN_OUTPUT_ROOT` (= `$VSC_DATA/CreditPFN`),
which is backed up. The training data on `$CREDITPFN_DATA_ROOT`
(= `$VSC_SCRATCH/CreditPFN`) is **not** backed up — but it can
always be re-derived from the raw CSVs.

Internals (the why):

* **Linear-warmup → cosine-decay LR** — matches HuggingFace's
  `get_cosine_schedule_with_warmup`, which is what TabPFN's
  `FinetunedTabPFNClassifier` uses internally. Verified
  numerically in `tests/test_train.py::test_warmup_cosine_schedule_landmarks`.
* **No validation set** — with ~17 PD + ~8 LGD datasets, holding
  out a separate val bucket leaves so few datasets to fit on that
  early-stopping signal becomes pure noise. We use fixed-epoch
  training and pick between hyperparameter settings *post-hoc* on
  the test set in the eval stage.

## Eval pipeline

`scripts/eval_pipeline.py` scores every model on every held-out test
**dataset** using K-fold cross-validation with an inner train/val
split.

### Why processed CSVs (not cached chunks)

The `.npz` chunks under `data/cached/` are sized for TabPFN's
in-context inference at *training* time. For *evaluation*, XGBoost /
CatBoost have no row-count limit and would be underestimated if
they only ever saw 20–100k-row chunks. So the eval reads the
sanitised dataset directly:

```
data/processed/{pd,lgd}/<id>.sanitized.csv
```

…with the cap policy:

| Model family | Pre-CV cap | Final fit + test eval | HPO subsample |
|---|---|---|---|
| `tabpfn-untuned` / `tabpfn-trained` | `cfg.max_rows_tabpfn = 100,000` (architectural — applies once, globally, before splitting) | uses the capped dataset | n/a |
| `xgboost` / `catboost` | no cap | uses the FULL dataset | `cfg.hpo.<m>.max_rows = 50,000` (stratified subsample of the inner-train set; HPO objective only — final fit ignores it) |
| `logreg` / `linreg` | no cap | uses the FULL dataset | n/a |

This matches the user's design: the only architectural cap is for
TabPFN; everything else trains on everything. HPO can be sped up
without ever capping the final fit.

### CV semantics — 80 / 16 / 20 per fold

For each test dataset:

```
Outer K-fold (cfg.cv.n_folds = 5):
    train      → 80% of dataset
    test       → 20%                  ← final metrics computed here

Inner split of the train fold (cfg.cv.inner_val_fraction = 0.20):
    sub-train  → 64% of dataset       ← model fits on this
    validation → 16% of dataset       ← Optuna HPO objective for
                                        XGBoost / CatBoost AND
                                        F1-threshold tuning for
                                        binary classification
```

Validation is essential for the tabular foundation models too —
without it, F1 / accuracy / precision / recall for PD would just use
the implicit 0.5 threshold, which is rarely optimal. Optuna runs
once per CV fold (5 studies per (model × dataset) at `n_folds=5`),
each with `hpo.<m>.n_trials` trials. LogReg / LinReg are intentionally
not tuned — they're the "what does plain linear modelling do"
baseline.

**Important:** the Optuna HPO objective uses **the eval pipeline's
inner-val split** (same 16% across every model in a fold), NOT a
fresh internal split inside the wrapper. This keeps the HPO objective
comparable across models and uses the same val data that the F1
threshold tuner needs anyway. The F1 threshold itself is picked in
O(n log n) via `sklearn.metrics.precision_recall_curve` — no quadratic
scan over the full probability array.

### Comprehensive metrics (one row per model × dataset × fold)

The eval CSV is wide format. NaN where not applicable.

| Group | Columns |
|---|---|
| Classification — threshold-free  | `roc_auc`, `log_loss`, `pr_auc` |
| Classification — threshold-tuned | `optimal_threshold` (max-F1 on inner-val), then `f1`, `accuracy`, `precision`, `recall` computed on the test fold AT that threshold |
| Regression                       | `rmse`, `mae`, `r2`, `neg_nll` (TabPFN-only) |
| Bookkeeping                      | `n_train_rows`, `n_val_rows`, `n_test_rows`, `elapsed_sec`, `status`, `error`, `timestamp` |

### Test-dataset resolution

For `tabpfn-trained` models the test datasets come from each
checkpoint's `.provenance.json` (so every checkpoint is scored on
its OWN held-out set — recorded at training time). For
`tabpfn-untuned` and classical baselines the test datasets come
from the cfg corpus split. Both routes give the same set when the
seed and fractions match (which they do by default), so the
comparison is apples-to-apples by construction.

### Local + slurm-array (parallelised) modes

```bash
# Local — single process, all (model × chunk × fold) cells in one run.
python scripts/eval_pipeline.py track=pd
python scripts/eval_pipeline.py track=lgd

# Restrict to one method or one dataset for debugging:
python scripts/eval_pipeline.py track=pd --method xgboost --test-dataset 0001.gmsc

# Slurm array — ONE (model × test_dataset) per task. With ~3 000
# datasets × ~25 models, this fans the heavy Optuna tasks out
# across many slurm jobs concurrently.
N=$(python scripts/eval_pipeline.py --list-tasks track=pd)
sbatch --array=0-$((N - 1))%32 scripts/slurm/eval_pd.slurm
```

Each slurm task writes its own
`results/benchmark/<TRACK>/<method>/<run_name>_<timestamp>_<task_tag>.csv`
(the `<task_tag>` includes the dataset_id), so concurrent tasks
NEVER write to the same file — no locking, no races. Aggregation is
a single `pd.read_csv` over a glob.

Per [`config/eval.yaml`](config/eval.yaml):

| Knob | Default | Effect |
|---|---|---|
| `cv.n_folds`                   | 5  | Stratified-K-fold per test dataset; results report mean ± std over folds. |
| `hpo.xgboost.n_trials`         | 25 | Per-fold Optuna HPO budget for XGBoost (TPE sampler). 0 = use defaults. |
| `hpo.catboost.n_trials`        | 25 | Same for CatBoost. |
| `hpo.<m>.timeout_seconds`      | 600 | Wall-clock cap per study (whichever hits first). |
| `tabpfn_n_estimators`          | 16 | TabPFN inference-time ensemble size (untuned + trained). |

Models compared:

| Source | Models |
|---|---|
| `baseline`        | XGBoost (Optuna-tuned), CatBoost (Optuna-tuned), LogReg (defaults, PD only), LinReg (defaults, LGD only) |
| `tabpfn-untuned`  | One per checkpoint in `cfg.tunable.<track>_base_paths` |
| `tabpfn-trained`  | Every OK row in `manifests/<run_name>_<track>.csv` |

### Permanent results layout

The `results/` tree has two siblings — one for benchmarking, one for
training diagnostics — so neither can clobber the other:

```
results/
├── benchmark/                          eval pipeline output
│   ├── PD/
│   │   ├── xgboost/                                  creditpfn_<ts>[__task<i>_ds-<id>].csv
│   │   ├── catboost/                                 creditpfn_<ts>[__task<i>_ds-<id>].csv
│   │   ├── logreg/                                   creditpfn_<ts>[__task<i>_ds-<id>].csv
│   │   ├── tabpfn-untuned__v2.6-default/             creditpfn_<ts>[__task<i>_ds-<id>].csv
│   │   ├── tabpfn-untuned__v2.5-default-2/           creditpfn_<ts>[__task<i>_ds-<id>].csv
│   │   ├── tabpfn-trained__v2.6-default__lr1e-05/    creditpfn_<ts>[__…].csv
│   │   └── tabpfn-trained__v2.5-default-2__lr5e-05/  creditpfn_<ts>[__…].csv
│   └── LGD/
│       └── …
└── training/                           train pipeline output
    ├── pd/                             per-epoch CSV (epoch, train_loss, lr, elapsed_sec)
    │   ├── creditpfn_pd_<base-stem>_lr1e-05_seed42.csv
    │   └── …
    └── lgd/
        └── …
```

The TabPFN-variant directory names under `benchmark/` compress the
published filenames (`tabpfn-v2.6-classifier-v2.6_default.ckpt` →
`v2.6-default`, `tabpfn-v2.5-regressor-v2.5_real.ckpt` → `v2.5-real`,
`tabpfn-v2.5-classifier-v2.5_default-2.ckpt` → `v2.5-default-2`); the
track-specific "classifier"/"regressor" infix is dropped because the
parent `PD/` or `LGD/` already encodes it. Trained variants append
`__lr<rate>` so two trials with different HPs land in different
folders. (The multi-chunk policy is no longer a sweep axis — fixed to
`first_chunk_only`.)

Each timestamp is unique to one benchmark run, so re-running the eval
**never overwrites** earlier results — every comparison this project
ever ran is permanently archived. Aggregate with pandas:

```python
import pandas as pd, glob
files = glob.glob("results/benchmark/PD/*/creditpfn_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.groupby(["model_name", "model_source"])[
    ["roc_auc", "f1", "log_loss", "rmse"]
].agg(["mean", "std", "count"])
```

The per-epoch training CSVs let you see how each trial's training loss
evolves — useful for debugging convergence and choosing a sane
`train.epochs` cap. Aggregate the same way:

```python
files = glob.glob("results/training/pd/*.csv")
hist = pd.concat([pd.read_csv(f).assign(run=Path(f).stem) for f in files])
```

## Tests

```bash
pytest -q tests/
```

| File | Coverage |
|---|---|
| `test_data.py`  | data pipeline (preprocessing → register → sanitize → dedup → dataset) |
| `test_paths.py` | env-aware path resolution (local-vs-VSC routing) |
| `test_train.py` | corpus split, dataloader, LR schedule, descriptive name, end-to-end mocked training loop |
| `test_model.py` | cache helper, baseline wrappers (XGB, CB, LogReg, LinReg) on synthetic data, model registry |
| `test_eval.py`  | per-cell scoring, K-fold benchmark on synthetic chunks, per-method CSV dirs, manifest loading |

Tests intentionally lean toward *failure-mode coverage* over
behavioural completeness. Tests requiring a real TabPFN checkpoint
on disk are guarded by `pytest.importorskip` so the suite stays
runnable in a stripped-down CI image.

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
