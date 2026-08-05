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
   - [4.7 `tfm-library/papers/` and `tfm-library/repositories/` — reference material](#47-papers-and-repositories--reference-material)
   - [4.8 `checkpoints/` — base and trained TabPFN weights (gitignored)](#48-checkpoints--base-and-trained-tabpfn-weights-gitignored)
   - [4.9 Runtime trees: `data/`, `output/`, `logs/` (gitignored)](#49-runtime-trees-data-output-logs-gitignored)
5. [Re-submitting the pipeline (resume semantics + cleanup)](#5-re-submitting-the-pipeline-resume-semantics--cleanup)
6. [Data pipeline](#6-data-pipeline)
7. [Training pipeline](#7-training-pipeline)
8. [Eval pipeline](#8-eval-pipeline)
9. [References](#9-references)

---

<a id="1-overview"></a>

## 1. Overview

Three pipeline stages, each with one config yaml and one CLI script:

| Stage | Config | Orchestrator | What it does |
|---|---|---|---|
| **Data**  | [`config/data.yaml`](config/data.yaml)   | [`scripts/data_pipeline.py`](scripts/data_pipeline.py)   | Dedup → register → sanitize → dedup. Writes one sanitized CSV per dataset under `data/processed/`. CPU-only; ~10 minutes for the full 17 PD + 8 LGD corpus. |
| **Train** | [`config/train.yaml`](config/train.yaml) | [`scripts/train_pipeline.py`](scripts/train_pipeline.py) | Continued pretraining of every `(base × LR × LoRA × query_fraction × accumulate_grad_batches)` tuple in the tunable grid. Reads sanitized CSVs directly, draws a fresh per-epoch subsample for each dataset, then applies TabPFN's official preprocessor (squashing scaler / quantile / SVD) to every step's data before the model sees it. Writes finetuned `.ckpt` files + provenance + per-epoch CSVs. **Requires a CUDA GPU.** |
| **Eval**  | [`config/eval.yaml`](config/eval.yaml)   | [`scripts/eval_pipeline.py`](scripts/eval_pipeline.py)   | K-fold cross-validation of every model on every held-out test dataset (XGBoost, CatBoost, LogReg / LinReg, untuned and trained TabPFN). Writes one CSV per `(model × dataset × fold)`. **Requires a CUDA GPU.** |

The notebooks under `notebooks/` consume the outputs of all three
stages and drop publication-quality PDFs under `output/figures/`. See
chapter 4 for the per-folder breakdown.

---

<a id="2-background"></a>

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

<a id="3-quick-start"></a>

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

<a id="31-install"></a>

### 3.1 Install

Python **3.11, 3.12, or 3.13** is supported. Python **3.14** is
excluded — some compiled deps (`torch`, `scikit-learn`) still lack
cp314 wheels, so a fall-back compile from source will fail on most
platforms.

```bash
python3.12 -m venv .venv --prompt CreditPFN          # Linux / macOS
# py -3.12 -m venv .venv --prompt CreditPFN          # Windows / PowerShell
source .venv/bin/activate                            # Linux / macOS
# .venv\Scripts\activate                             # Windows / PowerShell
pip install --upgrade pip
pip install -e ".[dev,notebooks]"      # runtime + pytest + Jupyter (see pyproject.toml)

# TabPFN install gotcha (read this).
# This project's src/train/model.py uses the current Prior Labs API
# (4-tuple load_model_criterion_config return; version="v3"/"v2.6";
# download_if_not_exists kwarg). PyPI's `tabpfn` is pinned at 2.2.x with
# an older API and will TypeError on the first model load. Override:
pip install --upgrade "tabpfn @ git+https://github.com/PriorLabs/tabPFN.git@main"
```

<a id="32-verify-the-install-local-laptop-is-fine"></a>

### 3.2 Verify the install (local laptop is fine)

These three operate on CPU; a laptop is sufficient. Use them to make
sure the project builds and reads its data correctly before
submitting any cluster jobs.

```bash
pytest -q tests/                                   # ~5 min, no GPU needed
python scripts/data_pipeline.py                    # ~10 min, builds data/processed/
jupyter notebook notebooks/                        # open the data-exploration notebooks
```

<a id="33-real-training-and-eval-require-a-cuda-cluster"></a>

### 3.3 Real training and eval require a CUDA cluster

The recommended entry point is the **multi-cluster submitter**. It
loads [`config/data.yaml`](config/data.yaml), resolves the storage
tiers, and submits each stage to the right cluster: **data prep →
wICE (CPU)**, **continued pretraining → Mindwell B200 (GPU)**,
**evaluation → wICE H100 (GPU)**:

```bash
# On a Genius login node, after cloning + installing + uploading raw CSVs:
bash scripts/slurm/run_full_pipeline.sh
```

Continued pretraining runs **only on Mindwell** — its B200 GPUs have
192 GiB VRAM (2.4× an H100), which we use to train at a larger
in-context size. The big files (datasets, trained checkpoints, results)
live in **project staging** (`/lustre1/project/stg_00211/CreditPFN`,
overridable via `$CREDITPFN_STAGING_ROOT` / `$TABPFN_STAGING_ROOT`);
small bookkeeping (logs, manifests, figures) lives on `$VSC_DATA`.
Because VSC clusters have separate Slurm controllers, the stages are
**not** `afterok`-chained across clusters — see chapter 5 and
[`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md) for the storage tiers,
cluster table, partitions, and a failure-mode cheat sheet.

To run a single stage by hand, the per-stage SLURM templates live in
[`scripts/slurm/`](scripts/slurm/). The `paths.data_source` knob in
[`config/data.yaml`](config/data.yaml) picks the dataset tier
(`staging` default / `scratch` / `data`).

Hydra-style CLI overrides work from any entrypoint (no yaml edits):

```bash
python scripts/train_pipeline.py track=lgd train.epochs=30
python scripts/eval_pipeline.py  --method xgboost  --test-dataset 0001.gmsc
```

---

<a id="4-repository-layout"></a>

## 4. Repository layout

```text
CreditPFN/
├── README.md
├── pyproject.toml            package metadata + dependencies
├── src/                      pipeline source code         (see 4.1)
├── config/                   three YAML configs           (see 4.2)
├── scripts/                  CLI + SLURM templates        (see 4.3)
├── notebooks/                exploration + viz notebooks  (see 4.4)
├── tests/                    pytest suite                 (see 4.5)
├── docs/                     long-form documentation      (see 4.6)
├── tfm-library/              SHARED knowledge base (git submodule -> TFM_Library repo): papers + code dumps + literature summaries  (see 4.7)
├── checkpoints/              base + trained TabPFN .ckpt  (see 4.8, gitignored)
├── data/                     raw + processed corpus       (see 4.9, gitignored)
├── output/                   everything the code writes   (see 4.9, gitignored)
└── logs/                     one log file per task        (see 4.9, gitignored)
```

<a id="41-src--pipeline-source-code"></a>

### 4.1 `src/` — pipeline source code

| Subpackage | Role | Public CLI |
|---|---|---|
| [`src/data/`](src/data)   | Four data-pipeline stages (dedup pre · register · sanitize · dedup post) plus `preprocessing.py` for per-dataset surgical fixes. Output is one sanitized CSV per dataset under `data/processed/`. | `python -m src.data.<stage>` for any one stage, or `scripts/data_pipeline.py` for the chain. |
| [`src/train/`](src/train) | The continued-pretraining loop: corpus split (`corpus.py`), the on-the-fly dataloader (`dataloader.py`) that reads sanitized CSVs and draws a fresh random subsample every epoch, TabPFN load/save + LoRA wrapping (`model.py`), training loop with per-epoch monitor (`loop.py`), metrics (`metrics.py`). | `scripts/train_pipeline.py` |
| [`src/model/`](src/model) | sklearn-style wrappers for every model the eval scores: XGBoost + CatBoost (with Optuna HPO), LogReg / LinReg (default-hyperparam baselines), TabPFN-untuned, TabPFN-trained. Single `base.py::BaselineModel` protocol so the eval loop stays model-agnostic. | importable only |
| [`src/eval/`](src/eval)   | The cross-model benchmark: processed-CSV loader, K-fold splitter with inner train/val, comprehensive metrics computation, results-dir routing, skip-existing rerun guard. | `scripts/eval_pipeline.py` |
| [`src/utils/`](src/utils) | Cross-cutting helpers: env-aware path resolver (`paths.py`), one-file-per-task run logging (`run_log.py`), notebook figure sink (`figures.py`), training / eval visualisation helpers (`training_viz.py`, `eval_viz.py`), upstream code refresh (`refresh_repositories.py`). | `python tfm-library/scripts/refresh_repositories.py` |

<a id="42-config--three-yaml-configs-one-per-stage"></a>

### 4.2 `config/` — three YAML configs, one per stage

Every knob lives in one of three YAMLs. Each yaml is the single
source of truth for its stage; the corresponding script imports it
via OmegaConf and accepts Hydra-style overrides on the CLI
(`key.nested=value`).

| File | Drives | Main sections |
|---|---|---|
| [`config/data.yaml`](config/data.yaml)   | `src/data/*` + `scripts/data_pipeline.py` | paths (incl. `data_source: "staging" \| "scratch" \| "data"`, default `"staging"`), `finetuning.max_rows_per_epoch` + `query_fraction`, dedup detection thresholds, sanitize knobs (max missing rate, ≤ 64-feature selection, LGD target clip) |
| [`config/train.yaml`](config/train.yaml) | `src/train/*` + `scripts/train_pipeline.py` | `tunable.*` (sweep axes: base checkpoint × LR × LoRA × query_fraction × accumulate_grad_batches), corpus split (Mode A fractions / Mode B explicit IDs), optimizer + scheduler, LoRA cfg, train loop |
| [`config/eval.yaml`](config/eval.yaml)   | `src/eval/*` + `scripts/eval_pipeline.py` | enabled baselines, K-fold + inner-val fractions, per-fold Optuna budget, `max_rows_per_model` (per-architecture training-context cap), results dir |

What is **deliberately not in YAML**: anything that never changes
across runs (optimizer family AdamW, cosine schedule, metric column
order). Those are hardcoded in code — searchable constants near the
top of the relevant module.

<a id="43-scripts--cli-entrypoints-and-slurm-templates"></a>

### 4.3 `scripts/` — CLI entrypoints and SLURM templates

One orchestrator per pipeline stage, plus SLURM templates for the
cluster:

| File | What it does |
|---|---|
| [`scripts/data_pipeline.py`](scripts/data_pipeline.py)   | Run all four data stages end-to-end, or just the ones you ask for (`--datasets ...`). Idempotent; `--fresh` rebuilds from scratch. |
| [`scripts/train_pipeline.py`](scripts/train_pipeline.py) | Iterate the `cfg.tunable` cartesian grid; one trial per call when `--single` or `--trial-index` (SLURM array). Auto-fills missing sanitized CSVs by invoking the data pipeline for just those IDs. **Skips trials whose finetuned checkpoint already exists** — re-submission is safe. |
| [`scripts/eval_pipeline.py`](scripts/eval_pipeline.py)   | Score every model on every test dataset, K-fold CV. Skip-existing by default; `--rerun` to force. Filterable with `--method` / `--test-dataset` / `--task-index`. |
| [`scripts/slurm/*.slurm`](scripts/slurm/)               | SLURM templates: one per data / train / eval stage, plus `run_full_pipeline.sh` for the chained submission. |
| [`src/utils/pipeline_clean.py`](src/utils/pipeline_clean.py) | Stage-level cleanup utility. `python -m src.utils.pipeline_clean --stages train,eval` wipes the outputs of the named stage(s) so the next re-submit starts those stages from scratch. See chapter 5 for details. |

<a id="44-notebooks--exploration-and-result-visualisations"></a>

### 4.4 `notebooks/` — exploration and result visualisations

Six notebooks: two for data exploration (run after the data
pipeline), two for training visualisation (PD / LGD), and two for
final-results visualisation (PD / LGD) — the visualisation notebooks
run after the respective pipeline. Every notebook drops its figures as PDFs into
`output/figures/<notebook-slug>/` via the figure sink helper in
[`src/utils/figures.py`](src/utils/figures.py); the per-notebook
directory is **wiped on each re-run**, so stale figures never linger.
All plotting code lives in the corresponding helper module under
`src/utils/`; the notebook cells contain only function calls so the
narrative stays scannable and the logic stays testable.

| Notebook | What it shows |
|---|---|
| `0.0. raw_data_exploration.ipynb`          | What did the vendor deliver? Corpus shape-space scatter (features vs rows, per-track → combined, log + linear), per-dataset missing-cell bars, target / class-balance landscape on the raw CSVs. |
| `0.1. processed_data_exploration.ipynb`    | Did sanitisation produce sensible inputs? Same shape + missingness views on the post-sanitize CSVs, plus the 64-feature selection effect and feature-type composition. |
| `1.0. training_visualization.ipynb`        | **PD** trained variants in one dashboard — per-trial loss / lr / metric curves, cross-trial overlays, LR sweep, LoRA effect, time/accuracy Pareto, convergence diagnostics, leaderboard. `TRACK='pd'`; consumes `output/training/`. |
| `1.1. training_visualization.ipynb`        | **LGD** counterpart of 1.0 — identical dashboard with `TRACK='lgd'`. |
| `2.0. final_results.ipynb`                 | **PD** eval leaderboard — per-method box plots, per-dataset heatmaps, pairwise win-rate matrix (à la TabPFN-3 Fig 3), trained-vs-untuned scatter (à la Real-TabPFN), fold stability, threshold calibration. `TRACK='pd'`; consumes `output/results/`. |
| `2.1. final_results.ipynb`                 | **LGD** counterpart of 2.0 — identical leaderboard with `TRACK='lgd'`. |

Corpus summaries in the data notebooks are memoised so the first
cell pays the disk-read cost once and every subsequent plot reads
from RAM.

<a id="45-tests--unit-and-smoke-tests"></a>

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

<a id="46-docs--project-documentation"></a>

### 4.6 `docs/` — project documentation

| File | What it is |
|---|---|
| [`docs/PAPER_ROADMAP.md`](docs/PAPER_ROADMAP.md) | **Start here for the scientific status.** Plain-language account of what the experiments have shown, whether the contribution is novel (with the nearest-neighbour papers), and the ordered list of what is still missing before writing a paper. |
| [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) | Deep-dive on the data stage: one raw CSV's full journey through dedup → register → sanitize, plus the two divergent downstream preprocessing paths (TabPFN vs classical baselines). Companion to `config/data.yaml`. |
| [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md)     | Inventory of every base `.ckpt` we ship (TabPFN v2.6 / v3 and TabICL v2): training data (synthetic-only), sample/feature caps, save formats per family, licence terms, and the one-time TabICL staging command. Cross-referenced to the HF model cards and Grinsztajn et al. 2026 (arXiv:2511.08667). |
| [`tfm-library/SUMMARIES.md`](tfm-library/SUMMARIES.md)       | Chronological tour of every paper under `tfm-library/papers/`, with a "For CreditPFN" pointer per paper. The most directly relevant works (Real-TabPFN, TabPFNv2, TabPFN-2.5, TabPFN-3, Rubachev finetuning, TabPFN-Wide) are flagged at the top. |
| [`tfm-library/SYNTHESIS.md`](tfm-library/SYNTHESIS.md)             | Cross-paper synthesis of the whole tabular-foundation-model paradigm (PFNs → TabPFN line → scaling → adaptation → extensions → critique), with a lineage timeline, design-axis comparison, and a one-card-per-paper appendix. Narrative companion to the per-paper `SUMMARIES.md`. |
| [`tfm-library/REPOSITORIES.md`](tfm-library/REPOSITORIES.md)   | What each `tfm-library/repositories/*.txt` dump is, why we keep it, and which lines to grep when designing each pipeline stage. Refresh script: `python tfm-library/scripts/refresh_repositories.py`. |
| [`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md)         | **VSC-specific deployment guide** (KU Leuven's Vlaamse Supercomputer Centre): OnDemand portal, conda env, dataset upload, partition / GPU choice, the SLURM submit chain, failure-mode cheat sheet. Read this only when you're about to deploy on VSC; everything in this README applies to any SLURM cluster. |

<a id="47-papers-and-repositories--reference-material"></a>

### 4.7 `tfm-library/` — the shared TFM knowledge base (git submodule)

All reference material lives in **[TFM_Library](https://github.com/andreasgoethals/TFM_Library)**,
a separate private repository mounted here as a git submodule at
`tfm-library/` and **shared across multiple projects** (one canonical
collection of papers, summaries, and upstream code dumps). Inside:

* `tfm-library/papers/` — every collected paper (PDF + text extraction).
* `tfm-library/SUMMARIES.md` / `tfm-library/SYNTHESIS.md` — per-paper
  summaries and the cross-paper synthesis of the TFM field.
* `tfm-library/repositories/` — flat-text dumps of the upstream
  implementations (TabPFN, extensions, finetuning repos, …), catalogued in
  `tfm-library/REPOSITORIES.md`. Read-only references — nothing is imported.

Submodule mechanics (the only three commands you need):

```bash
git clone --recursive <this-repo>        # new clone WITH the library
git submodule update --init              # ...or populate it after a plain clone/pull
git submodule update --remote tfm-library  # pull the library's latest state, then commit the bump
```

The submodule pins a specific library commit (reproducible by
construction). Agents: read `tfm-library/AGENTS.md` before editing
anything inside the mountpoint.

<a id="48-checkpoints--base-and-trained-tabpfn-weights-gitignored"></a>

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

<a id="49-runtime-trees-data-output-logs-gitignored"></a>

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

On a laptop, `data/`, `output/`, and `checkpoints/` all live under the
repo root. On VSC they are split across three storage tiers:

* **Project staging** (`/lustre1/project/stg_00211/CreditPFN`, the big
  files) — **datasets** (`data/raw`, `data/processed`), **trained
  checkpoints** (`checkpoints/trained/`), and **eval results**
  (`output/results/`). Persistent, large, non-purged.
* **`$VSC_DATA/CreditPFN`** (small, NFS-backed) — logs, training
  manifests (`output/training/`), per-epoch CSVs, notebook figures
  (`output/figures/`), dedup files.
* **`$VSC_SCRATCH`** — optional fast-I/O working copy of datasets.

The dataset tier is picked by `paths.data_source` in
`config/data.yaml` (`staging` default / `scratch` / `data`); the
staging root resolves from `$CREDITPFN_STAGING_ROOT` →
`$TABPFN_STAGING_ROOT` → the built-in `/lustre1/project/stg_00211`
default. Trained-checkpoint saves *probe* staging writability first
(`resolve_writable_staging_path`) and fall back to `$VSC_DATA` with a
loud warning when the compute node can't write staging (the Mindwell
failure mode of 2026-07-03); the eval gate then archives any fallback
checkpoints into staging automatically, so the durable copy of every
big artefact always ends up in project storage. See
[`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md) §0.2.

---

<a id="5-re-submitting-the-pipeline-resume-semantics--cleanup"></a>

## 5. Re-submitting the pipeline (resume semantics + cleanup)

Every stage is designed to be **idempotent** — re-submitting picks up
where the previous run left off without redoing work it has already
done. The skip logic lives in three places, one per stage:

| Stage | What is treated as "already done" | What re-submission does |
|---|---|---|
| **Data**  | A dataset is skipped if its `data/processed/<track>/<id>.sanitized.csv` exists on disk. Checked per-dataset by `_ensure_processed` in `scripts/train_pipeline.py` and by `scripts/data_pipeline.py` itself. | Datasets already on disk are skipped silently. Missing datasets are sanitized fresh. |
| **Train** | A trial is skipped if BOTH `checkpoints/trained/<track>/<descriptive_name>.ckpt` AND `<...>.ckpt.provenance.json` exist. (Added 2026-05-27.) | A "SKIP" row is appended to the manifest for that trial; the SLURM array task exits cleanly without retraining. |
| **Eval**  | A `(model, dataset)` pair is skipped if every fold has an existing `status="OK"` row in its per-method CSV under `output/results/`. Partial-failure pairs (some folds FAIL) ARE retried. | Pairs already complete are skipped. New trials added to the training manifest after a previous eval run get scored. `--rerun` flag forces fresh scoring. |

**Logs** are NEVER auto-deleted between runs. Each SLURM job creates a
new `logs/<task>_<ts>_j<jid>_a<tid>.log` file (suffixed with the
trial's descriptive name on rename — see the training pipeline). They
accumulate until you explicitly delete them.

**To force a fresh start of one or more stages**, use the cleanup
utility:

```bash
# wipe just the training stage (keeps data + eval intact)
python -m src.utils.pipeline_clean --stages train

# wipe training and eval (keeps the sanitized CSVs)
python -m src.utils.pipeline_clean --stages train,eval

# nuclear option — wipe everything
python -m src.utils.pipeline_clean --stages all

# dry-run first to see what would be deleted
python -m src.utils.pipeline_clean --stages all --dry-run
```

`pipeline_clean` deletes every output of the listed stage(s) — sanitized
CSVs, dedup files, finetuned checkpoints, manifests, per-epoch CSVs,
benchmark CSVs, notebook figures, AND that stage's log files. It NEVER
touches `data/raw/` (the input corpus) or `checkpoints/*.ckpt` (the
base TabPFN weights — only `checkpoints/trained/` is in scope). Full
file catalogue lives in the module's docstring at
[`src/utils/pipeline_clean.py`](src/utils/pipeline_clean.py).

The util has no heavy dependencies (no `omegaconf`, no `torch`) — it
runs on a bare login node. It reads `config/{data,eval,train}.yaml`
when available (via `omegaconf` or `PyYAML`) and falls back to
hardcoded defaults otherwise.

> **Typical re-submit workflow.** You change something in
> `config/train.yaml` (different LR sweep, different LoRA targets) and
> want to retrain. The previous run's checkpoints would now be stale,
> so:
>
> ```bash
> python -m src.utils.pipeline_clean --stages train,eval
> sbatch scripts/slurm/train_pd.slurm
> ```
>
> Data stays — you didn't change anything that would invalidate the
> sanitized CSVs. Eval is cleared too because the trained checkpoints
> changed, so old eval results would mix freshly-trained scores with
> stale-base scores.

---

<a id="6-data-pipeline"></a>

## 6. Data pipeline

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
  detection, fuzzy column-name match). **Diagnostic only**: the stage
  writes a `doubles_{track}_{pre,post}.csv` report listing every
  flagged pair with the detection method and confidence label
  (`high` / `medium` / `low`) but does NOT remove any dataset from
  the corpus. The first occurrence of a dataset within a track is
  always considered the canonical one; only subsequent duplicates
  appear in the report. To act on findings, manually delete a CSV
  from `data/raw/<track>/` and re-run `pipeline_clean --stages data`
  followed by `sbatch scripts/slurm/data.slurm`.
* **`register.py`** — applies surgical fixes, then computes
  per-dataset metadata (n_rows / n_cols, missing rate, class balance,
  target mean/std, content-aware shape hash). Idempotent: re-running
  updates rows in place and preserves existing IDs not in the current
  filter.
* **`sanitize.py`** — surgical fixes, then a dataset-agnostic clean:
  drop exact-duplicate columns, drop > 90 %-NaN columns, drop constant
  columns, coerce numeric strings, cast numericals to float32
  (out-of-range values become NaN before the cast — no overflow
  warnings), ±inf → NaN, optional unsupervised feature **selection** to
  ≤ 64 real columns (top by scale-free variance, de-correlated; not
  averaged), label-encode
  classification targets, clip LGD targets to [0, 1].

The eval pipeline reads the same sanitized CSVs but applies its own
K-fold CV split + per-model row cap (see chapter 8). The training
loop reads them via
`src/train/dataloader.py::ProcessedDatasetLoader`, which draws a
fresh random subsample of `finetuning.max_rows_per_epoch` rows from
each parent dataset every epoch and then **runs TabPFN's official
preprocessor** (`TabPFNEnsemblePreprocessor`, see
`src/train/tabpfn_preprocessing.py`) on the context split — same
squashing-scaler + quantile + SVD + class-permutation pipeline
that runs at inference time inside `TabPFNClassifier.predict_proba`.
This ensures training-time inputs match the distribution the model
was pretrained on (added 2026-05-27; before this the model was
trained on raw heavy-tailed features and inference applied
preprocessing, which produced calibration-collapse — see chat).

### What sanitize.py deliberately does NOT do

TabPFN's package handles these internally — see
[`tfm-library/REPOSITORIES.md`](tfm-library/REPOSITORIES.md) § "Outlier handling":

| Step | Why we don't pre-apply it |
|---|---|
| Outlier winsorisation | TabPFN's `OUTLIER_REMOVAL_STD = 12.0` (classifier) / `None` (regressor) handles outliers with the right semantics. We invoke it from `src/train/tabpfn_preprocessing.py::apply_outlier_clip` just before each model forward. |
| `PowerTransformer` / `QuantileTransformer` / `RobustScaler` / `SquashingScaler` / SVD | TabPFN's per-estimator inference ensemble cycles through these on every fit. We invoke them per-step in training (`TabPFNEnsemblePreprocessor.fit_transform_ensemble_members`) and TabPFN re-runs them at inference. |
| NaN imputation | TabPFN's preprocessor handles NaNs natively (learned default + binary indicator) — no pre-imputation needed. |
| Regression target z-normalisation | Applied per-step inside `tabpfn_preprocessing.build_ensemble_members` on context-only stats, mirroring TabPFN's `znorm_space_bardist_`. |
| Class label permutation (data augmentation) | Generated per-step inside `TabPFNEnsemblePreprocessor` via `class_shift_method="shuffle"`. The loop unpermutes the logit columns before the CE loss (matches the inference path at `TabPFN .txt:8511-8523`). |

---

<a id="7-training-pipeline"></a>

## 7. Training pipeline

A thin orchestrator over `src/train/`. **Continued pretraining runs only
on the Mindwell B200 cluster** (192 GiB VRAM) — see
[`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md). The single source of truth for
hyperparameters is [`config/train.yaml`](config/train.yaml), in three
layers:

* **Tunable HPs** (`tunable.*` lists at the top) — base checkpoint
  (TabPFN v3 / TabPFN v2.6 / **TabICL v2**), learning rate
  (`{3e-7, 1e-6, 1e-5, 3e-5}` — spans from a
  direct reproduction of Real-TabPFN's `3e-7` through TabICL's own
  finetuning default `1e-5` up to `3e-5`, near the
  separate Rubachev single-dataset FT median (~3.9e-5); `1e-4` is excluded because it
  diverged on no-LoRA + qf 0.20 — revisit now that `weight_decay=0.0`),
  the parameter-efficient-adaptation flag, query_fraction,
  accumulate_grad_batches, and epoch
  pass-mode (`one_sample` / `full_pass`). Anything genuinely unknown in
  advance. The full cartesian product is the default sweep (currently
  **3 bases × 4 LRs × 2 adapt-modes × 1 qf × 1 acc × 2 epoch-pass-modes
  = 48 trials per track**, run as a 48-task SLURM array on the 24
  Mindwell B200 GPUs). See "Hyperparameter rationale vs. the literature"
  below.
* **Two model families.** The base list mixes **TabPFN** and **TabICL
  v2** (added 2026-08-04); the family is detected from the checkpoint
  filename (`src/train/tabicl_compat.py::model_family`) and selects the
  loader, the loss, the row cap, and the save schema. The `use_lora`
  axis is family-specific: LoRA for TabPFN, **freeze-backbone** (train
  the ICL module only) for TabICL — that family's own pretraining
  stage-3 regime, chosen because full SFT collapsed TabICL in two
  independent reports (TabZilla accuracy 0.873 → 0.567 in Tanna 2026;
  "failed to train TabICL" in Kolberg 2026). TabICL trials are tagged
  `_iclhead` instead of `_lora`. TabICL losses are upstream's own:
  cross-entropy over the first `n_classes` of its 10 logit columns, and
  mean pinball loss over its 999-quantile head. Because that head is
  not a bar distribution, `neg_nll` is undefined for TabICL — density
  numbers are never comparable across families (CRPS is the planned
  cross-family density metric).
* **Fixed HPs** (single values under `train.*`) — epochs, AMP, gradient
  clipping, warmup fraction, per-epoch monitor subsample,
  `n_estimators_finetune` (ensemble members per training step —
  **per-track: `pd: 2`, `lgd: 8`**, matching the official
  `FinetunedTabPFNClassifier` / `FinetunedTabPFNRegressor` defaults;
  TabICL overrides both to `2` via `n_estimators_finetune_tabicl`,
  matching *its* wrappers).
  Follow each package's defaults where those are well-tuned. The per-step subsample size lives in
  [`config/data.yaml`](config/data.yaml) (`finetuning.max_rows_per_epoch`,
  PD/two-member caps: **26 000 for v3, 11 000 for v2.6, 10 000 for
  TabICL**; LGD's eight
  members scale the TabPFN caps to 6 500 / 2 750 — sized from a B200
  fwd+bwd probe
  while chasing Real-TabPFN's "more context → bigger gains"; plus an
  optional v3-only `max_cells_per_epoch` cell budget. The TabICL cap is
  upstream's own chunk size, **not** a B200 measurement — probe it
  before raising it). `weight_decay` is
  **0.0** (matching Rubachev's study and the official wrappers; Garg et al.
  do not report this value); anti-forgetting uses Real-TabPFN's L2-SP anchor
  (`l2sp_lambda=0.003`), not decay-to-origin.
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
| Full corpus, full HP grid, on the cluster    | `bash scripts/slurm/run_full_pipeline.sh` — see [`docs/VSC_GUIDE.md`](docs/VSC_GUIDE.md) |

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
| Per-epoch CSV (epoch incl. `-1` = pre-FT baseline; train_loss, lr, train/test primary + secondary metric, epoch_time)  | `output/training/epochs/<track>/<descriptive_name>.csv`             |
| Full run log (slurm stdout + python logger)                           | `logs/train_<track>_<ts>[_j<jid>_a<tid>].log`                       |

Filename schema:
`<run_name>_<track>_<base-stem>_lr<lr>_seed<seed>[_qf<qf>][_acc<K>][_fullpass][_lora].ckpt`.
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
* **Fixed-sample monitor eval** — at baseline, the first/final epoch, and
  every `train.epoch_eval_every` epochs, the loop
  scores the model on a small subsample of each train- and
  test-dataset (ROC-AUC for PD, RMSE for LGD, at
  `train.epoch_eval_subsample_samples = 2000` rows). The exact sampled rows
  and context/query split stay fixed across the learning curve; changing them
  would confound model improvement with sampling noise. Cheap but enough to
  see whether the model is still improving. Set it to `0` to disable the
  monitor entirely (divergence detection then ignores the metric signal).
* **Up-front debug banner** — before the very first forward pass the loop
  emits a single multi-line log block (`[run] [hyperparams] [corpus]
  [env] [hardware] [versions] [paths]`) capturing the resolved HPs, the
  SLURM array/cluster/node, GPU + VRAM, library versions, and the base /
  save paths. It is logged early on purpose so the run is fully
  reconstructable even if the job is later OOM-killed or diverges.
* **Divergence detection + early abort** — when the loss stays
  constant, or AUC pegs at 0.5 (random), or the AMP scaler skips >50 %
  of recent steps for `train.divergence_patience = 5` epochs, the
  training loop aborts and records `status=DIVERGED` in the manifest.
  This prevents wasting 3+ hours of GPU on a dead model (observed in
  `train_pd_*qf20_acc1*` runs of 2026-05-28 before the safeguard was
  added).
* **L2-SP anti-forgetting penalty (optional)** — `optimizer.l2sp_lambda`
  adds `0.5·λ·‖w − w₀‖²` to the loss, penalising drift of the weights
  away from the **synthetic-prior start `w₀`** (not toward zero, which is
  what `weight_decay` does). This is the regularizer used by Garg et al.'s
  Real-TabPFN corpus-level continued pretraining, with the same `λ=0.003`;
  it is not part of Rubachev et al.'s separate single-dataset finetuning
  study. It is **on by default for full-FT trials** at
  `λ = 0.003` (set `optimizer.l2sp_lambda: 0.0` to disable) and is
  **orthogonal to the LoRA axis but full-FT-only** — with LoRA the base
  weights are frozen and cannot drift, so L2-SP is inert and silently
  skipped (effectively: full-FT gets L2-SP, LoRA doesn't). The penalty
  enters only the back-prop loss; the logged CE / NLL curves stay the pure
  data loss. The effective λ is recorded in each checkpoint's provenance.
  (`λ = 0.003` is a starting point chosen for this setup, not a tuned
  constant — adjust if anti-forgetting is too strong or too weak.)

### Hyperparameter rationale vs. the literature

Every training knob is compared against three distinct references:
**Real-TabPFN** (Garg et al., corpus-level continued pretraining),
**On Finetuning Tabular Foundation Models** (Rubachev et al., 342
single-dataset FT runs), and the official `FinetunedTabPFN*` wrappers.
They are not the same study or code release.

| Knob | CreditPFN | Reference | Why |
|------|-----------|-----------|-----|
| Optimizer | AdamW, β=(0.9, 0.999) | Garg: AdamW; Rubachev/wrappers: AdamW | match |
| `weight_decay` | **0.0** | Garg: not reported; Rubachev + wrappers: 0.0 | avoids stacking decay-to-origin on L2-SP; do not claim this reproduces an unreported Garg setting |
| LR grid | **{3e-7, 1e-6, 1e-5, 3e-5}** | Garg: fixed **3e-7**; Rubachev: tuned 5e-6–5e-4 (median ≈3.9e-5); wrappers: 2e-5 clf / 1e-5 reg | lowest rung reproduces Garg; upper rungs span transfer to newer bases and Rubachev's FT range |
| Schedule | linear-warmup → cosine, warmup 0.10 | Garg: linear warmup → cosine; Rubachev: constant LR | matches Garg; differs from Rubachev |
| Training budget | fixed 50 corpus epochs + divergence-abort | Garg: 20 000 steps; Rubachev: early-stop patience 16, eval every 10 steps | **OURS** — dataset-balanced epochs/full-pass ablation replace Garg's fixed step budget; no validation bucket is held out |
| `n_estimators_finetune` | **pd 2 / lgd 8** | wrapper defaults: clf 2, reg 8 | match (per-track) |
| Loss | CE (PD) / bar-distribution NLL (LGD) | same | match — what TabPFN was pretrained with |
| query_fraction | 0.20 (80 % context / 20 % query) | Garg: 0.40 (60/40); package default: 0.20 | deliberate deviation to package default |
| Context size (`max_rows_per_epoch`) | PD: **v3 26 000 / v2.6 11 000**; LGD scaled to 6 500 / 2 750 for eight members | Garg: up to 20 000 rows and 400 000 cells; larger context helped | B200-measured member-aware caps, not a paper default |
| L2-SP anchor | λ=0.003, full-FT only | Garg: **λ=0.003** | matches Garg for full FT; inert when LoRA freezes the base |
| LoRA | swept on/off (r=8, α=16) | Rubachev studies LoRA; Garg reports full-model CPT only | parameter-efficient comparison point |
| Preprocessing | package ensemble preprocessing | Garg: ordinal categoricals + noisy-quantile numerics | analogous, not byte-for-byte identical |

Like Real-TabPFN, CreditPFN performs corpus-level continued pretraining:
each batch contains one table, but tables are repeatedly sampled from a
multi-dataset corpus. CreditPFN changes the domain, adds an LGD track and
LoRA/pass-mode ablations, uses newer v2.6/v3 bases, and replaces Garg's fixed
20 000-step budget with 50 dataset-balanced epochs.

### Optimization objective — why CE / NLL, not AUC

* **PD (classifier): `torch.nn.CrossEntropyLoss`.** Three reasons:
  (i) CE is differentiable everywhere, AUC isn't (it's a step function
  in predictions — gradient is 0 almost everywhere); (ii) CE optimizes
  BOTH rank-ordering AND probability calibration, while AUC optimizes
  only rank — but credit-risk regulation (Basel III) requires calibrated
  PD probabilities for expected-loss and capital calculations;
  (iii) TabPFN was pretrained with CE — switching to a different
  objective for finetuning would create a train/finetune mismatch that
  damages the synthetic prior. Differentiable surrogates for AUC
  (Wilcoxon-style, ROC-Star) exist but are unstable and typically used
  as a regularizer added to CE, not a replacement. We track AUC as the
  primary *evaluation* metric (more interpretable across imbalance
  levels) but optimize CE.
* **LGD (regressor): bar-distribution NLL.** TabPFN's regressor head
  emits *logits over a discrete histogram* (the "bar distribution",
  Müller et al. 2023). Its NLL `−log(density)` is what TabPFN was
  pretrained on, and crucially it trains the **uncertainty estimate**
  simultaneously with the point estimate — a Gaussian-NLL alternative
  would force unimodal homoscedastic predictions; an MSE alternative
  would throw away the uncertainty quantification that's the whole
  point of TabPFN's regression head. Regulators care about the full
  loss distribution, not just the mean. Negative loss values are
  expected: a bar-distribution NLL `= −log(density)` can be negative
  whenever the density exceeds 1.0 (narrow histogram buckets, sharp
  predictions). RMSE / R² are tracked as *evaluation* metrics.
* **TabICL uses its own two objectives.** Classification is
  cross-entropy over the first `n_classes` of its 10 logit columns;
  regression is the mean pinball loss over 999 quantile levels
  (`linspace(0,1,Q+2)[1:-1]`) on context-z-normalised targets. Both are
  copied verbatim from `tabicl._finetune`, i.e. each family is trained
  with the objective it was pretrained on — the whole point of continued
  pretraining. The practical consequence for reporting: a quantile head
  has no exact predictive density, so **`neg_nll` is never comparable
  between TabPFN and TabICL** (and is recorded as NaN for TabICL).

### Methodology & limitations (statistical validity)

The core methodology is sound — preprocessing mirrors TabPFN's
inference path exactly (no train/inference skew), the CE / NLL
objectives match what the model was pretrained with, and the eval
stage's inner/outer split keeps HPO and threshold-tuning out of the
test fold. The honest caveats below are limitations of the
*experimental design*, not bugs:

* **Small test corpus → wide confidence intervals.** A 70/30
  dataset-level split of ~17 PD + ~8 LGD datasets leaves only ~5 PD and
  ~2–3 LGD *test* datasets. Per-dataset K-fold CV tightens the
  within-dataset estimate, but the across-dataset mean rests on a
  handful of datasets — report per-dataset results, not just the pooled
  mean.
* **Best-of-48 selection on the test set (winner's curse).** With no
  validation set, the best of 48 trials is picked by test performance;
  the maximum over 48 noisy estimates is upward-biased. Prefer the
  per-architecture trained-vs-untuned delta over the absolute best, and
  report the trial distribution.
* **Within-domain split ≠ Real-TabPFN's external benchmark.** We split
  one credit corpus into train/test datasets, so they may share vendor
  quirks. This is the right test of "does credit-specialisation help on
  credit data," but it measures less external generalisation than a
  cross-benchmark protocol would.
* **Feature selection is fit on the whole dataset** (in `sanitize.py`,
  before the eval CV split), so the kept-column set is chosen using rows
  that later land in test folds. The selection is *unsupervised* (a
  scale-free-variance ranking + correlation de-duplication that never sees
  `y`), so there is no label leakage — only a small optimistic bias in
  feature construction.
* **No multiple-comparison correction across ~20 metrics.** Pre-register
  the primary metric (roc_auc for PD, rmse for LGD) and treat the rest
  as diagnostic.

---

<a id="8-eval-pipeline"></a>

## 8. Eval pipeline

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
| `tabpfn-untuned` / `tabpfn-trained`   | `cfg.max_rows_per_model[<v>]` (v3: 1 000 000; v2.6: 50 000)                 | **full**       | n/a                                                                            |
| `tabicl-untuned` / `tabicl-trained`   | `cfg.max_rows_per_model["tabicl"]` (50 000 — conservative, unmeasured)      | **full**       | n/a                                                                            |
| `xgboost` / `catboost`                | none                                                                        | full           | `cfg.hpo.<m>.max_rows = 50 000` (stratified subsample of inner-train; HPO only) |
| `logreg` / `linreg`                   | none                                                                        | full           | `cfg.hpo.<m>.n_trials = 50` (tunes `C` / `alpha`)                              |

The cap resolves from the **base checkpoint** for both the trained and
the untuned handle, so a family's two arms always see the same fold size
(a 2026-08-04 fix: the untuned branch used to miss its key and silently
fall through to the `default` cap).

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
`n_folds=5`), each with `hpo.<m>.n_trials` trials. All four baselines
get **50 trials** — including LogReg's `C` and LinReg's `alpha`, which
dominate a linear model on collinear credit features — so a
foundation-model win cannot be dismissed as under-tuned baselines
(changed 2026-06-24).

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
| Classification — threshold-free    | `roc_auc`, `log_loss`, `pr_auc`, `brier_score`                                                   |
| Classification — threshold-tuned   | `optimal_threshold` (max-F1 on inner-val), then `f1`, `accuracy`, `precision`, `recall`, `specificity`, `balanced_accuracy`, `mcc`, `cohen_kappa` on test |
| Regression                         | `rmse`, `mae`, `median_ae`, `mape`, `r2`, `explained_variance`, `pearson_r`, `spearman_r`, `neg_nll` (TabPFN-only) |
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

<a id="9-references"></a>

## 9. References

The full paper library lives under [`tfm-library/papers/`](tfm-library/papers/) with a
chronological, detailed summary in
[`tfm-library/SUMMARIES.md`](tfm-library/SUMMARIES.md). The most directly relevant
works for this project:

- **Garg et al., 2025.** *Real-TabPFN — Improving Tabular Foundation
  Models via Continued Pre-training With Real-World Data.*
  [arXiv:2507.03971](https://arxiv.org/abs/2507.03971) — the recipe we
  follow.
- **Hollmann et al., 2025.** *Accurate predictions on small data with
  a tabular foundation model.* (Nature) — TabPFNv2 architecture.
- **Grinsztajn et al., 2026.** *TabPFN-2.5: Advancing the State of
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
  Extreme Feature Counts.* — continued-pretraining for >500-feature
  data via a *feature-widening synthetic prior*. Note: TabPFN-Wide
  argues *against* dimensionality reduction (it uses
  `FeatureAgglomeration` only as a baseline to beat). Our `sanitize.py`
  step is unsupervised feature **selection** (keep real columns, don't
  average) — an independent, pragmatic feature cap, **not** derived from
  this paper.

Local code dumps under
[`tfm-library/repositories/`](tfm-library/repositories/) (catalogued in
[`tfm-library/REPOSITORIES.md`](tfm-library/REPOSITORIES.md)) cover the public TabPFN
package, the docs site, the v2.5 / v2.6 / v3 HuggingFace model cards (v2.5 kept for scholarly reference; not used in our sweep),
NanoTabPFN, the V2-Finetuning recipe, and the underlying PFN
framework. Read-only — refresh with
`python tfm-library/scripts/refresh_repositories.py`.
