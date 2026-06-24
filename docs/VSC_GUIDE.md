# CreditPFN on VSC — deployment guide

End-to-end recipe for running the pipeline (**data → train → eval**) on
a VSC site via the [Open OnDemand](https://openondemand.org/) web
portal. The whole flow lives behind one command:

```bash
bash scripts/slurm/submit_full_pipeline.sh
```

The rest of this guide is the story of how to get there: where the code
lives, where the data lives, how the SLURM stages chain together, and
how to vary what gets trained.

---

## 0 · One-time setup

### 0.1 Open a shell on the cluster

In the OnDemand portal, click **Clusters → Login (Server) Shell Access**.
A terminal opens in a new browser tab on a Genius login node — prompt
looks like `[May/11 15:21] vscXXXXX@tier2-p-login-N $` (where `N` is
1–4). **Neither wICE nor Mindwell has its own login node** — you always
SSH into Genius and submit to the target cluster via `#SBATCH
--clusters=<name>` in the job script (every `.slurm` here already does).
Every command below runs in that shell.

### 0.1b The clusters (KU Leuven Tier-2)

CreditPFN spans two GPU clusters. **Continued pretraining runs ONLY on
Mindwell** (the B200's 192 GiB VRAM lets us train at a much larger
in-context size); everything else runs on wICE.

| Cluster | GPU partition | GPU | VRAM | GPUs | Cores/GPU | Account | Used for |
|---------|---------------|-----|------|------|-----------|---------|----------|
| **Mindwell** | `gpu_b200` | B200 (Blackwell) | **192 GiB** | 3×8 = **24** | 24 | `lp_mindwell_pilot` (free pilot) | **Training** |
| **wICE** | `gpu_h100` | H100 SXM5 | 80 GiB | 5×4 = 20 | 16 | `lp_verbekelab` | Eval |
| **wICE** | `gpu_a100` | A100 SXM4 | 80 GiB | 4×4 = 16 | 18 | `lp_verbekelab` | Eval (extra capacity) |
| **wICE** | `batch` | — (CPU) | — | — | — | `lp_verbekelab` | Data prep |

Notes:
- **GPU walltime caps at 72 h** on both clusters (there is no `gpu_*_long`
  partition — only CPU partitions have 7-day `_long` variants).
- **Backfill:** shorter `--time` ⇒ better queue position. Tighten your
  walltime estimate after the first trial (read `epoch_dt` in the log).
- **Mindwell pilot:** `lp_mindwell_pilot` is free during the pilot. Check
  it's still active: `sam-balance -A lp_mindwell_pilot`. B200 credit
  weight is 437.5/GPU-min (cheaper than H100's 569.4) once it's paid.
- **Maximum parallelism:** PD and LGD training run as independent 24-wide
  arrays on Mindwell; eval runs as 32-wide arrays on wICE — so PD+LGD
  train simultaneously while wICE eval churns through whatever's done.

### 0.2 Three storage tiers, three purposes

VSC gives every user three storage tiers. The crucial distinction —
and the source of most confusion — is **personal data storage
(`$VSC_DATA`) vs. project ("staging") storage**:

| Tier | What it is | Holds (CreditPFN) | Backed up? | Purged? | Quota |
|------|------------|-------------------|-----------|---------|-------|
| **Project staging** (`/lustre1/project/stg_00211`) | Group/project share on the **Lustre** parallel filesystem (`$VSC_PROJECT_LUSTRE1`). The **big files** live here. | **Datasets** (raw + processed), **trained checkpoints**, **benchmark result CSVs** | No (single copy) | No | Large (≥ 1 TB); **low inode budget** — few big files, not many tiny ones |
| **Personal data** (`$VSC_DATA`) | Your own NFS home-for-data. | Code + **small durable outputs**: logs, manifests, per-epoch CSVs, notebook figures | **Yes** (snapshots) | No | **75 GiB** (tight) |
| **Scratch** (`$VSC_SCRATCH`) | Fast parallel FS (Lustre on wICE/Genius, GPFS on Mindwell). | Optional fast-I/O working copies | No | **~30 days** | 500 GiB |

**Why datasets live in project staging, not `$VSC_DATA`:** the datasets
are the single largest artefact in the project. `$VSC_DATA`'s 75 GiB
quota is too small for the full corpus, and scratch gets purged every
30 days. Project staging is large, persistent, and shared — so it is
the canonical home for datasets, trained checkpoints, and results.
`$VSC_DATA` is reserved for the small, frequently-rewritten, *backed-up*
bookkeeping outputs (logs/manifests/figures).

**Filesystem caveat (Lustre vs GPFS).** Staging is on **Lustre**, which
wICE and Genius see natively. **Mindwell** (where training runs) is on
**GPFS** and VSC asks that *sustained* I/O on Mindwell use GPFS. Our
training only does a one-time read of the processed CSVs at startup
(then caches them in RAM) plus a single checkpoint write at the end —
not sustained I/O — so reading datasets directly from Lustre staging on
Mindwell is fine. If a Mindwell job is ever killed for I/O, copy the
processed data to `$VSC_SCRATCH` (GPFS) first and
`export CREDITPFN_DATA_ROOT=$VSC_SCRATCH/CreditPFN` for that run.

**How the split is configured.** `paths.data_source` in
[`config/data.yaml`](../config/data.yaml) picks the dataset tier:
`"staging"` (default), `"scratch"`, or `"data"`. Logs/manifests/figures
always go to `$VSC_DATA`; trained checkpoints + results always go to
staging. The resolver finds the staging root from, in order:
`$CREDITPFN_STAGING_ROOT` → `$TABPFN_STAGING_ROOT` → the built-in
default `/lustre1/project/stg_00211`. So on VSC it **just works** with no
env var set; override only if your allocation differs.

The env vars `$CREDITPFN_DATA_ROOT`, `$CREDITPFN_OUTPUT_ROOT` and
`$CREDITPFN_STAGING_ROOT` remain available as escape-hatch overrides
(highest precedence — honoured before the yaml).

### 0.3 Clone the repo

The code is public at
[github.com/andreasgoethals/CreditPFN](https://github.com/andreasgoethals/CreditPFN).
Clone it into `$VSC_DATA` (so it's backed up), then `git pull` before
every run:

```bash
cd $VSC_DATA
git clone https://github.com/andreasgoethals/CreditPFN.git
cd CreditPFN
```

After the clone the layout is:

```text
$VSC_DATA/CreditPFN/
├── src/                 all the pipeline code (data, train, eval, model, utils)
├── scripts/             CLI entrypoints + SLURM templates
├── config/              data.yaml, train.yaml, eval.yaml — the only knobs
├── repositories/        flat-text dumps of upstream code (read-only reference)
├── docs/                this file + CHECKPOINTS.md + LITERATURE.md
├── tests/               pytest suite (236 tests)
└── pyproject.toml       package metadata + dependencies
```

### 0.4 Create the conda env (one time)

```bash
mamba create -y -n CreditPFN python=3.12      # or `conda` if mamba isn't installed
source activate CreditPFN
pip install -e ".[dev,notebooks]"             # deps from pyproject.toml
```

**TabPFN caveat.** PyPI's `tabpfn` caps at `2.2.1`, which has an older
API than the code expects. Install the matching Prior Labs release on
top:

```bash
pip install --upgrade "tabpfn @ git+https://github.com/PriorLabs/tabPFN.git@main"
```

Without this, `train_pipeline.py` will `TypeError` on the first model
load. Eval against pre-existing checkpoints is unaffected.

### 0.5 Upload datasets and base checkpoints

The big files go to **project staging** (everything else is in git):

| What                                                                    | Destination                                          | How                                                |
|-------------------------------------------------------------------------|------------------------------------------------------|----------------------------------------------------|
| Raw credit-risk datasets (`*.csv`)                                      | `/lustre1/project/stg_00211/CreditPFN/data/raw/{pd,lgd}/` | WinSCP / FileZilla / `scp` (Globus for >1 GB)      |
| Base TabPFN checkpoints (`tabpfn-v3-*.ckpt`, `tabpfn-v2.6-*.ckpt`, …)    | `/lustre1/project/stg_00211/CreditPFN/checkpoints/`   | WinSCP, or `wget` from Hugging Face on a login node |

Staging lives on Lustre (`$VSC_PROJECT_LUSTRE1/stg_00211`) and the big
files **stay there forever** (no purge). Logs and other small outputs go
to `$VSC_DATA` automatically — you never copy those by hand.

**Concrete transfer commands (run on your laptop).** The Genius login
node can write to the Lustre staging path, so you can `scp`/`rsync`
straight there even though your file browser opens in `$VSC_DATA` — just
give the absolute `/staging/...` path as the destination:

```bash
# From your laptop, in the folder that holds your local data/ + checkpoints/.
# Replace vsc38338 with your VSC id (and stg_00211 if your allocation differs).
STAGE=/lustre1/project/stg_00211/CreditPFN

# 1) Raw datasets (the largest files) → staging
scp -r data/raw/*       vsc38338@login.hpc.kuleuven.be:"$STAGE/data/raw/"
# 2) Base TabPFN checkpoints → staging
scp -r checkpoints/*.ckpt vsc38338@login.hpc.kuleuven.be:"$STAGE/checkpoints/"

# rsync is better for big/resumable transfers if you have it (Git-Bash/WSL):
rsync -ah --info=progress2 data/raw/ vsc38338@login.hpc.kuleuven.be:"$STAGE/data/raw/"
```

For transfers larger than ~1 GB, prefer **Globus** (both Lustre and GPFS
have endpoints) or a wICE `interactive`-partition transfer job; VSC
recommends transfer jobs over Globus for >1 TB. Pack many small files
into an archive first — staging's Lustre is bad at metadata storms and
its inode budget is limited.

**If you truly cannot write to `/staging` from your transfer tool** (you
can only reach `$VSC_DATA`): upload into `$VSC_DATA/CreditPFN/data/` and
`$VSC_DATA/CreditPFN/checkpoints/`, then relocate to staging with the
helper job (it runs on a wICE compute node, which mounts the Lustre that
holds staging):

```bash
scp -r data/raw/*        vsc38338@login.hpc.kuleuven.be:'$VSC_DATA/CreditPFN/data/raw/'
scp -r checkpoints/*.ckpt vsc38338@login.hpc.kuleuven.be:'$VSC_DATA/CreditPFN/checkpoints/'
# then, on the VSC login node:
sbatch scripts/slurm/stage_to_project.slurm     # copies $VSC_DATA → staging
```

If your project's staging allocation isn't `stg_00211`, set
`export CREDITPFN_STAGING_ROOT=/lustre1/project/stg_XXXXX` (or
`TABPFN_STAGING_ROOT`) and the whole pipeline — and the helper job —
follows.

Base checkpoints can also be fetched from Hugging Face directly on a
VSC login node — see `docs/CHECKPOINTS.md` for the exact `.ckpt`
filenames the loader expects.

### 0.6 The exact layout the pipeline expects (auto-detected)

The data pipeline reads from `$CREDITPFN_DATA_ROOT/data/raw/{pd,lgd}/<id>.csv`.
The submitter (`submit_full_pipeline.sh`) auto-detects where you put
the data and sets `CREDITPFN_DATA_ROOT` for the slurm jobs. The probe
order (staging first — datasets' canonical home) is:

  1. `/lustre1/project/stg_00211/CreditPFN/data/raw/{pd,lgd}/` — **canonical** (project staging)
  2. `$VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/`              — scratch project subdir
  3. `$VSC_SCRATCH/data/raw/{pd,lgd}/`                        — straight-into-scratch
  4. `$VSC_DATA/CreditPFN/data/raw/{pd,lgd}/`                 — repo-local

Whichever directory actually contains CSVs wins — so if you uploaded to
staging as in §0.5 (the canonical place), it is found automatically.
After uploading, run `bash scripts/slurm/submit_full_pipeline.sh` — it
prints `CREDITPFN_DATA_ROOT: <resolved path>` so you can verify it picked
the right one.

To force a specific location (e.g. you staged data into scratch for a
throughput experiment), set the env var explicitly — it always wins over
autodetect:

```bash
export CREDITPFN_DATA_ROOT="$VSC_SCRATCH/CreditPFN"
bash scripts/slurm/submit_full_pipeline.sh
```

### 0.7 Why staging, not scratch, is the default

Datasets default to **project staging** precisely because it is
*not* purged. `$VSC_SCRATCH` auto-cleans files untouched for ~30 days
(and `mv` / `rsync -a` don't refresh the access time, so a freshly
*moved* file can still be purged — stage with `cp`). Staging avoids
this entirely: upload your datasets once and they persist.

If you deliberately want the fast-but-ephemeral scratch tier (e.g. for a
throughput experiment), set `paths.data_source: "scratch"` in
`config/data.yaml`; for everything on `$VSC_DATA`, set `"data"`.
`submit_full_pipeline.sh` resolves the cfg and propagates
`CREDITPFN_DATA_ROOT` / `CREDITPFN_OUTPUT_ROOT` / `CREDITPFN_STAGING_ROOT`
through `sbatch --export` to every job.

---

## 1 · The full chain in one command

From the cloned repo:

```bash
cd $VSC_DATA/CreditPFN
source activate CreditPFN      # one-time per shell session
git pull
bash scripts/slurm/submit_full_pipeline.sh
```

(The submitter auto-activates the env if it can find one, but doing it
explicitly first avoids surprises on shells where conda isn't on
`$PATH`.)

That submits the pipeline across **two clusters** (training is
Mindwell-only; data prep and eval run on wICE):

```text
data.slurm                          ──    CPU  (wICE batch)        → writes datasets to staging
                                          │
   (datasets persist in project staging, visible from both clusters)
                                          │
train_pd.slurm   (array, N jobs)    ──    GPU  (Mindwell gpu_b200) → continued pretraining ONLY here
train_lgd.slurm  (array, N jobs)    ──    GPU  (Mindwell gpu_b200)
                                          │
   (trained checkpoints persist in project staging)
                                          │
eval_pd.slurm    (2 arrays)         ──    GPU  (wICE gpu_h100 + gpu_a100)
eval_lgd.slurm   (2 arrays)         ──    GPU  (wICE gpu_h100 + gpu_a100)
```

**No cross-cluster `afterok` chain.** VSC runs wICE and Mindwell as
separate Slurm clusters, and `--dependency` does **not** work across
them. So the three stages are submitted independently and sequenced via
the shared staging tier:
- Run `data` first and let it finish (datasets land in staging).
- `train` (Mindwell) reads those datasets; submit after data completes.
- `eval` (wICE) is robust — it scores whatever checkpoints exist in
  staging and skips the rest, so re-running `STAGES=eval` after training
  finishes picks up late trials.

Optional knobs (set before invoking):

```bash
STAGES="data train eval"   # submit a subset, e.g. STAGES=train
TRACKS="pd lgd"            # one track only if you want
TRAIN_CONCURRENCY=24       # max in-flight Mindwell B200 tasks (24 GPUs)
EVAL_CONCURRENCY=32        # max in-flight wICE eval tasks
bash scripts/slurm/submit_full_pipeline.sh
```

Watch progress across both clusters with **Active Jobs** in OnDemand, or
`squeue --me --clusters=wice,mindwell`. Every task writes a single log
file at
`$VSC_DATA/CreditPFN/logs/<task>_<YYYYMMDD>_<HHMMSS>_j<jid>_a<tid>.log`
(including a full debug banner: host, cluster, GPU, VRAM, library
versions, resolved storage roots, and every hyperparameter).

---

## 2 · Stage 1 — data preprocessing

CPU-only, on **wICE `batch`**. Reads from `<data_root>/data/raw/`
(project staging by default), writes processed artefacts to the same
root (the data pipeline has **no `.npz` chunking step** — sanitized CSVs
are the canonical training input):

```text
data/raw/{pd,lgd}/<id>.csv          (you uploaded to staging)
        ↓ dedup --pass pre          → data/dedup/doubles_{track}_pre.csv     ($VSC_DATA, durable)
        ↓ register                  → data/manifest_{pd,lgd}.csv             ($VSC_DATA, durable)
        ↓ sanitize                  → data/processed/{pd,lgd}/<id>.sanitized.csv  (staging, data_root)
        ↓ dedup --pass post         → data/dedup/doubles_{track}_post.csv    ($VSC_DATA, durable)
```

The processed CSVs land in staging so the Mindwell training stage (a
different cluster) can read them. Submit just this stage:
`sbatch --clusters=wice scripts/slurm/data.slurm`.

**Idempotent.** Re-running detects existing processed CSVs and skips
them; pass `--fresh` (uncomment the line in `data.slurm`) to rebuild
from scratch.

---

## 3 · Stage 2 — continued pretraining

### 3.1 What gets swept

`config/train.yaml` Section 0 declares the cartesian sweep:

```yaml
tunable:
  classifier_base_paths:
    - "checkpoints/tabpfn-v3-classifier-v3_default.ckpt"
    - "checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"
  regressor_base_paths:
    - "checkpoints/tabpfn-v3-regressor-v3_default.ckpt"
    - "checkpoints/tabpfn-v2.6-regressor-v2.6_default.ckpt"
  learning_rates:    [3.0e-7, 1.0e-6, 1.0e-5, 3.0e-5]
  use_lora:          [false, true]
  query_fractions:   [0.20]
  accumulate_grad_batches: [1]
  epoch_pass_modes:  ["one_sample", "full_pass"]
```

Default = **2 bases × 4 LRs × 2 LoRA × 1 qf × 1 acc × 2 epoch-pass-modes
= 32 trials per track**. (The grid now reaches `3e-5` ≈ Real-TabPFN's
per-dataset median LR; `1e-4` is still excluded because it diverged on
no-LoRA + qf=0.20 — revisit now that `weight_decay=0.0`; see the comment
in [`config/train.yaml`](../config/train.yaml) and the hyperparameter
rationale in the README. v2.5 was dropped on 2026-05-21 — see
`docs/CHECKPOINTS.md`. Regression uses `n_estimators_finetune: 8`,
classification `2`, matching the official wrappers.) One
SLURM array task per trial. Each parent dataset contributes exactly
one training step per epoch (no chunking — see the 2026-05-20
refactor and the `ProcessedDatasetLoader` in
`src/train/dataloader.py`). Recompute the current trial count any
time with:

```bash
python scripts/train_pipeline.py --list-trials track=pd
```

### 3.2 Which datasets to train on

Section 2 of `config/train.yaml` — Mode A (fractions) or Mode B
(explicit lists):

```yaml
corpus:
  train_dataset_ids: ["0001.gmsc", "0002.taiwan_creditcard"]
  test_dataset_ids:  ["0017.SBA_loans_case"]
```

Each saved checkpoint's `.provenance.json` records the test list so
the eval pipeline knows which datasets to score it on later. The
training log reports both lists up front:

```text
Training datasets (n=20): 0001.gmsc, 0002.taiwan_creditcard, …
Held-out test datasets (n=5): 0017.SBA_loans_case, …
```

Unknown dataset IDs (typos) raise a clear error listing the valid IDs
for the active track — no silent skips.

### 3.3 Submit

Training runs on **Mindwell B200** (the `#SBATCH --clusters=mindwell`
header is already in the scripts):

```bash
N_PD=$(python scripts/train_pipeline.py --list-trials track=pd)
sbatch --clusters=mindwell --array=0-$((N_PD - 1))%24 scripts/slurm/train_pd.slurm

N_LGD=$(python scripts/train_pipeline.py --list-trials track=lgd)
sbatch --clusters=mindwell --array=0-$((N_LGD - 1))%24 scripts/slurm/train_lgd.slurm
```

The `.slurm` files have generous default array bounds; over-sizing is
safe (surplus array tasks exit zero cleanly). Each task writes (big
files → staging; small bookkeeping → `$VSC_DATA`):

| Artefact                                           | Tier        | Path                                                                |
|----------------------------------------------------|-------------|---------------------------------------------------------------------|
| Final-epoch weights                                | **staging** | `checkpoints/trained/<track>/<descriptive_name>.ckpt`               |
| Provenance sidecar (HPs, train/test IDs, GPU, …)   | **staging** | `<descriptive_name>.ckpt.provenance.json`                           |
| Manifest row (consumed by the eval pipeline)       | `$VSC_DATA` | `output/training/manifests/<run_name>_<track>.csv`                  |
| Per-epoch CSV (loss, lr, train/test metric, time)  | `$VSC_DATA` | `output/training/epochs/<track>/<descriptive_name>.csv`             |
| Full run log (incl. debug banner)                  | `$VSC_DATA` | `logs/train_<track>_<ts>_j<jid>_a<tid>.log`                         |

Filename schema:
`<run_name>_<track>_<base-stem>_lr<lr>_seed<seed>[_lora].ckpt`.

---

## 4 · Stage 3 — cross-model benchmark

### 4.1 What gets compared

| Source            | Models                                                                                |
|-------------------|---------------------------------------------------------------------------------------|
| `baseline`        | XGBoost + CatBoost (Optuna-tuned), LogReg (PD), LinReg (LGD)                          |
| `tabpfn-untuned`  | One per checkpoint in `cfg.tunable.<track>_base_paths` — the "before" weights         |
| `tabpfn-trained`  | Every OK row in `output/training/manifests/<run_name>_<track>.csv` — the "after" weights |

Every continued-pretrained checkpoint is picked up automatically.

### 4.2 Splits per (model × dataset × fold)

5-fold stratified CV per test dataset. Each train fold is 80/20-split
again into sub-train + inner-val; that inner-val is shared by the
Optuna HPO objective (XGBoost / CatBoost) and the F1-threshold tuner
(PD). For TabPFN-family models, only the **training partition** of
each fold is capped at `cfg.max_rows_per_model[<v>]` (v3: 1 000 000,
v2.x: 100 000) — the held-out test partition is **always full** and
gets predicted in a single `predict_proba` call (TabPFN-v3's
internal `inference_row_chunk_size = 2048` handles arbitrarily large
test sets). Non-TabPFN baselines see the full dataset.

### 4.3 Re-runs are idempotent

Before scoring, each `(model × dataset)` pair is checked against the
existing CSVs under
`output/results/<TRACK>/<method-dirname>/`. Pairs whose **all
folds** are already `OK` are skipped:

- **First run** — scores every baseline + untuned + trained variant.
- **Re-run after adding a new trained checkpoint** — scores only the
  new checkpoint's pairs. XGBoost / CatBoost / LogReg / LinReg /
  untuned-TabPFN are reused from disk.

Force a fresh scoring with `--rerun`. To rescore a single method,
delete its directory under `output/results/<TRACK>/` and re-submit.

### 4.4 Submit (parallelised across BOTH wICE GPU pools)

The easy path — `submit_full_pipeline.sh` (or `STAGES=eval bash …`) — splits
each track's task range into two arrays and launches one on **`gpu_h100`**
(20 GPUs) and one on **`gpu_a100`** (16 GPUs), so eval drains across all 36
wICE GPUs at once. The split is contiguous (no overlap), and even if it
weren't, the skip-existing guard makes concurrent arrays safe. Control it
with the `EVAL_PARTITIONS` knob (default `"gpu_h100 gpu_a100"`; set to
`"gpu_h100"` for a single pool).

To submit by hand instead — e.g. the full range on each pool, letting the
idempotent skip-existing logic cooperatively drain the work:

```bash
N_PD=$(python scripts/eval_pipeline.py --list-tasks track=pd)
sbatch --array=0-$((N_PD - 1))%32 --partition=gpu_h100 scripts/slurm/eval_pd.slurm
sbatch --array=0-$((N_PD - 1))%32 --partition=gpu_a100 scripts/slurm/eval_pd.slurm   # 2nd array

N_LGD=$(python scripts/eval_pipeline.py --list-tasks track=lgd)
sbatch --array=0-$((N_LGD - 1))%32 --partition=gpu_h100 scripts/slurm/eval_lgd.slurm
sbatch --array=0-$((N_LGD - 1))%32 --partition=gpu_a100 scripts/slurm/eval_lgd.slurm  # 2nd array
```

Already-scored pairs exit zero in seconds; surplus array tasks (if the
array is sized larger than the grid) do the same. Each task logs a
one-line per-(model × dataset) score summary (`roc_auc`/`rmse` mean over
OK folds), so the eval log alone shows the results without opening the CSVs.

### 4.5 Output layout

Benchmark result CSVs are **big, canonical artefacts** → they live in
**project staging** (alongside datasets and trained checkpoints).
Manifests, per-epoch CSVs, figures and logs are small, frequently
rewritten bookkeeping → `$VSC_DATA`.

```text
/lustre1/project/stg_00211/CreditPFN/output/
└── results/                                 eval-pipeline output (staging)
    ├── PD/
    │   ├── xgboost/                                  creditpfn_<ts>__task<N>_ds-<id>.csv
    │   ├── catboost/                                 creditpfn_<ts>__task<N>_ds-<id>.csv
    │   ├── logreg/                                   creditpfn_<ts>__task<N>_ds-<id>.csv
    │   ├── tabpfn-untuned__v3-default/               creditpfn_<ts>__task<N>_ds-<id>.csv
    │   └── tabpfn-trained__v3-default__lr1e-05/      creditpfn_<ts>__task<N>_ds-<id>.csv
    └── LGD/  …

$VSC_DATA/CreditPFN/output/
├── training/
│   ├── manifests/                           one row per trained checkpoint
│   │   ├── creditpfn_pd.csv
│   │   └── creditpfn_lgd.csv
│   └── epochs/                              per-trial per-epoch CSVs
│       ├── pd/   creditpfn_pd_<base-stem>_lr1e-05_seed42[_lora].csv
│       └── lgd/  …
└── figures/                                 notebook PDF dumps (wiped on each run)
    ├── 0.0_raw_data_exploration/
    ├── 0.1_processed_data_exploration/
    ├── 1.0_training_visualization/
    └── 2.0_final_results/
```

(Trained `*.ckpt` + `.provenance.json` files live in staging too, under
`checkpoints/trained/<track>/` — see §3.3.) If
`$CREDITPFN_STAGING_ROOT` does not resolve (e.g. on a laptop), the eval
pipeline falls back to writing results under `$VSC_DATA` /
`$CREDITPFN_OUTPUT_ROOT`.

Every benchmark invocation gets a fresh `<timestamp>` — earlier runs
are never overwritten. Aggregate across runs:

```python
import pandas as pd, glob
files = glob.glob("/lustre1/project/stg_00211/CreditPFN/output/results/PD/*/creditpfn_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.groupby(["model_name", "model_source"])[
    ["roc_auc", "f1", "log_loss", "pr_auc"]
].agg(["mean", "std", "count"])
```

Row schema (classification): `roc_auc, log_loss, pr_auc, brier_score,
ece, optimal_threshold, f1, accuracy, precision, recall, specificity,
balanced_accuracy, mcc, cohen_kappa`. Row schema (regression): `rmse,
mae, median_ae, mape, r2, explained_variance, pearson_r, spearman_r,
neg_nll` (`neg_nll` is currently always NaN — the TabPFN wrapper does
not expose bar-distribution NLL). The threshold-tuned classification
metrics (everything from `f1` onward) use the max-F1 threshold chosen on
the inner-validation split and are NaN for multiclass.

---

## 5 · Common workflows

### 5.1 Add a new dataset

1. Upload the raw CSV to
   `/lustre1/project/stg_00211/CreditPFN/data/raw/{pd,lgd}/<id>.csv`
   (the canonical, persistent home — see §0.5).
2. On your laptop, register a metadata entry in
   `src/data/preprocessing.py::_RAW_METADATA`. Add a surgical-fix
   function only if the dataset needs one — clean datasets fall through
   automatically.
3. Commit + push, then on VSC:
   `git pull && bash scripts/slurm/submit_full_pipeline.sh`.

The data stage processes only the new ID; training re-runs all
variants; the eval reuses every baseline row that's already on disk
and only scores the new (model × dataset) cells.

### 5.2 Test a single LR across all bases

Edit `config/train.yaml`:

```yaml
tunable:
  learning_rates: [1.0e-5]
```

Grid drops to `2 bases × 1 LR × 2 LoRA × 1 qf × 1 acc × 2 pass-modes = 8`
per track. Re-submit; `--list-trials` reflects the new count.

### 5.3 Benchmark a subset

```bash
# Only XGBoost + one trained checkpoint:
python scripts/eval_pipeline.py track=pd \
    --method xgboost \
    --method "tabpfn-trained[creditpfn_pd_tabpfn-v3-classifier-v3_default_lr1e-05_seed42]"

# Only one test dataset:
python scripts/eval_pipeline.py track=pd --test-dataset 0001.gmsc

# Force re-scoring even if results exist:
python scripts/eval_pipeline.py track=pd --method xgboost --rerun
```

---

## 6 · Failure-mode cheat sheet

| Symptom                                                 | What to do                                                                                                       |
|---------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: No module named 'omegaconf'` on submit | The conda env isn't active. Run `source activate CreditPFN` first; the submitter also tries to activate it itself. |
| `sbatch: error: Batch job submission failed: Job dependency problem` | A stage targets a different cluster than its dependency — VSC runs wICE and Mindwell as separate Slurm controllers, so cross-cluster `afterok:` chains don't work. This is exactly why the data/train/eval stages carry **no** `--dependency` flag and are sequenced via shared staging instead (wICE scripts use `#SBATCH --clusters=wice`; Mindwell train scripts use `#SBATCH --clusters=mindwell`). (The `<jobid>;wice` or `<jobid>;mindwell` suffix in `sbatch --parsable` output on a Genius login is NOT this error — it just means the jobid lives in the target cluster's controller, which is normal; the submitter strips it.) |
| Data preprocessing log shows `missing raw file: …/<id>.csv — skipped` for everything | The pipeline auto-probes for the CSVs (staging first — see §0.6); if none are found you'll get this. Re-upload to `/lustre1/project/stg_00211/CreditPFN/data/raw/{pd,lgd}/` (see §0.5). The submitter prints the resolved `CREDITPFN_DATA_ROOT` on launch — verify it matches where you actually uploaded. |
| `TypeError` on first model load in training             | PyPI tabpfn 2.2.1 has the old API — install the Prior Labs wheel (see §0.4).                                     |
| Array task produces no log file                         | The SLURM `--output=/dev/null` is set; check the `exec >` redirection in the `.slurm` script.                    |
| One trial fails, the rest succeed                       | Manifest row gets `status=FAIL`; the eval auto-skips that checkpoint.                                            |
| Eval task says `KeyError: <id> not in cache`            | The auto-cache hook re-runs the data pipeline for missing IDs — let it finish.                                   |
| Out-of-memory on `gpu_b200` (training)                  | Lower `finetuning.max_rows_per_epoch` in `config/data.yaml` (B200 default v3 20 000 / v2.6 9 000). The per-epoch debug line reports `gpu_peak_alloc` — check it against the 192 GiB budget and adjust. |
| Mindwell job killed for I/O                             | Direct Lustre-staging reads from Mindwell (GPFS) were throttled. Copy processed data to scratch first: `cp -r /lustre1/project/stg_00211/CreditPFN/data $VSC_SCRATCH/CreditPFN/` then `export CREDITPFN_DATA_ROOT=$VSC_SCRATCH/CreditPFN`. |
| `sbatch: Job dependency problem` across clusters        | You tried to `afterok`-chain a wICE job to a Mindwell job. Cross-cluster deps don't work — submit the stages independently (the submitter already does). |
| Wrong test set scored for a checkpoint                  | Check `<checkpoint>.ckpt.provenance.json` — the eval reads that, not the live cfg.                               |
| Eval re-run produces 0 new CSVs                         | The skip-existing guard fired — every pair was already scored. Pass `--rerun` to force.                          |

---

## TL;DR

```bash
# One-time, in an OnDemand shell:
cd $VSC_DATA
git clone https://github.com/andreasgoethals/CreditPFN.git && cd CreditPFN
mamba create -y -n CreditPFN python=3.12 && source activate CreditPFN
pip install -e ".[dev,notebooks]"
pip install --upgrade "tabpfn @ git+https://github.com/PriorLabs/tabPFN.git@main"

# Upload raw datasets AND base .ckpt files to project staging:
#   /lustre1/project/stg_00211/CreditPFN/data/raw/{pd,lgd}/
#   /lustre1/project/stg_00211/CreditPFN/checkpoints/
# (If your allocation differs: export CREDITPFN_STAGING_ROOT=/lustre1/project/stg_XXXXX)

# Per experiment (training on Mindwell, data+eval on wICE):
source activate CreditPFN     # if not already in this shell
git pull
bash scripts/slurm/submit_full_pipeline.sh    # data → wICE; train → Mindwell; eval → wICE
squeue --me --clusters=wice,mindwell
```

| Where to look | Tier        | Path                                                              |
|---------------|-------------|-------------------------------------------------------------------|
| Code          | `$VSC_DATA` | `$VSC_DATA/CreditPFN/src/`                                        |
| Datasets      | **staging** | `/lustre1/project/stg_00211/CreditPFN/data/`                       |
| Models        | **staging** | `/lustre1/project/stg_00211/CreditPFN/checkpoints/trained/<track>/*.ckpt` |
| Results       | **staging** | `/lustre1/project/stg_00211/CreditPFN/output/results/<TRACK>/<method>/*.csv` |
| Logs          | `$VSC_DATA` | `$VSC_DATA/CreditPFN/logs/<task>_<ts>_j<jid>_a<tid>.log`          |
| Per-epoch     | `$VSC_DATA` | `$VSC_DATA/CreditPFN/output/training/epochs/<track>/*.csv`        |
