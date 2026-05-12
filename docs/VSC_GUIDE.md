# CreditPFN on VSC — deployment guide

Step-by-step recipe for running the full pipeline
(**data → train → eval**) through your VSC site's **Open OnDemand**
web portal (see the
[Open OnDemand project](https://openondemand.org/) for the underlying
software; each VSC institution provides its own portal URL).

The whole flow is one command:
`bash scripts/slurm/submit_full_pipeline.sh`.

---

## 0 · One-time setup

### 0.1 Open a shell on the cluster

In the OnDemand portal: **Clusters → Login (Server) Shell Access**
opens a terminal in a new browser tab and drops you on a login node.
The prompt looks like:

```text
[May/11 15:21] vsc38338@tier2-p-login-1 $
```

Everything below runs in that shell unless explicitly noted.

### 0.2 Clone the GitHub repo

The repo is **not** auto-pulled. Clone it once into `$VSC_DATA` (which
is backed up); update it later with `git pull` before each run.

```bash
cd $VSC_DATA
git clone <your-repo-url> CreditPFN
cd CreditPFN
```

For private repos you'll need credentials. The recommended path is
SSH keys — generate one on the VSC login node and register it with
GitHub:

* GitHub docs:
  [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) ·
  [Generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) ·
  [Adding the key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

Quick form, run once on the login node:

```bash
ssh-keygen -t ed25519 -C "vsc38338@vscentrum.be" -f ~/.ssh/id_ed25519_github
cat ~/.ssh/id_ed25519_github.pub      # paste into github.com → Settings → SSH keys
printf 'Host github.com\n  IdentityFile ~/.ssh/id_ed25519_github\n  AddKeysToAgent yes\n' >> ~/.ssh/config
ssh -T git@github.com                 # should greet you by username
```

### 0.3 Create the conda env (one time)

```bash
mamba create -y -n CreditPFN python=3.12      # or `conda` if mamba isn't installed
source activate CreditPFN
pip install -r requirements.txt
```

### 0.4 Upload datasets and base checkpoints (manual, not via OnDemand)

OnDemand's **Files** app can browse `$VSC_HOME` and `$VSC_DATA`, but
**not `$VSC_SCRATCH`** — and the raw datasets must live on scratch
because they're large and `$VSC_DATA` has a tight quota. Use a
separate tool with `sftp` / `scp` access:

| Tool                                                       | Platform        |
|------------------------------------------------------------|-----------------|
| [WinSCP](https://winscp.net/eng/index.php)                 | Windows         |
| [FileZilla](https://filezilla-project.org/) (SFTP profile) | cross-platform  |
| `scp` / `rsync` on the command line                        | macOS / Linux   |

Two upload categories — everything else is in git:

| What                                                        | Destination                                  |
|-------------------------------------------------------------|----------------------------------------------|
| Raw credit-risk datasets (`*.csv`)                          | `$VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/`  |
| Base TabPFN checkpoints (`tabpfn-v2.5-*.ckpt`, `tabpfn-v2.6-*.ckpt`) | `$VSC_DATA/CreditPFN/checkpoints/`           |

Base checkpoints can also be fetched directly on the login node — they
live on Hugging Face under `Prior-Labs/TabPFN-v2-clf` and
`Prior-Labs/TabPFN-v2-reg`; see `docs/CHECKPOINTS.md` for the exact
filenames.

### 0.5 Path roots

Auto-detected from `$VSC_DATA` / `$VSC_SCRATCH` (see
`src/utils/paths.py`); the SLURM scripts also set them explicitly.

| Variable                    | What it covers                                                                                | VSC default              |
|-----------------------------|-----------------------------------------------------------------------------------------------|--------------------------|
| `$CREDITPFN_DATA_ROOT`      | `data/raw/`, `data/processed/`, `data/cached/` — big I/O artefacts                            | `$VSC_SCRATCH/CreditPFN` |
| `$CREDITPFN_OUTPUT_ROOT`    | `logs/`, `manifests/`, `dedup/`, `checkpoints/trained/`, `results/` — small, must survive     | `$VSC_DATA/CreditPFN`    |

---

## 1 · The full chain in one command

```bash
cd $VSC_DATA/CreditPFN
git pull                                       # always pull first
bash scripts/slurm/submit_full_pipeline.sh
```

Submits six SLURM jobs with `--dependency=afterok` chaining:

```text
data.slurm                          ──┐  CPU
        ↓ afterok                     │
train_pd.slurm   (array, N jobs)    ──┤  GPU
train_lgd.slurm  (array, N jobs)    ──┤
        ↓ afterok                     │
eval_pd.slurm    (array, M jobs)    ──┤  GPU
eval_lgd.slurm   (array, M jobs)    ──┘
```

Optional knobs (set before running):

```bash
TRAIN_CONCURRENCY=4    EVAL_CONCURRENCY=32    \
TRACKS="pd lgd"        bash scripts/slurm/submit_full_pipeline.sh
```

Watch progress with **Active Jobs** in OnDemand, or
`squeue --me --clusters=genius,wice`. Per-task logs are all in one
flat directory:
`$VSC_DATA/CreditPFN/logs/<task>_<YYYYMMDD>_<HHMMSS>_j<jid>_a<tid>.log`.

---

## 2 · Stage 1 — data preprocessing

Pipeline (CPU):

```text
data/raw/{pd,lgd}/<id>.csv          (you uploaded)
        ↓ dedup --pass pre          → dedup/doubles_{track}_pre.csv
        ↓ register                  → manifest_{pd,lgd}.csv
        ↓ sanitize                  → data/processed/{pd,lgd}/<id>.sanitized.csv
        ↓ dedup --pass post         → dedup/doubles_{track}_post.csv
        ↓ dataset (chunk + cache)   → data/cached/{pd,lgd}/<id>/chunk_*.npz
```

**Submit just this stage**:
`sbatch scripts/slurm/data.slurm`.

The pipeline is **idempotent** — re-running it skips datasets whose
cache fingerprint matches the current manifest row, processed CSV
content, and dataset-config hash. To force a fresh rebuild, pass
`--fresh` (uncomment the line in `data.slurm`).

---

## 3 · Stage 2 — continued pretraining

### 3.1 Tunable knobs

Open **`config/train.yaml`**. Section 0:

```yaml
tunable:
  classifier_base_paths:
    - "checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"
    - "checkpoints/tabpfn-v2.5-classifier-v2.5_default-2.ckpt"
    - "checkpoints/tabpfn-v2.5-classifier-v2.5_default.ckpt"
  regressor_base_paths:
    - "checkpoints/tabpfn-v2.6-regressor-v2.6_default.ckpt"
    - "checkpoints/tabpfn-v2.5-regressor-v2.5_default.ckpt"
    - "checkpoints/tabpfn-v2.5-regressor-v2.5_real.ckpt"
  learning_rates: [1.0e-5, 5.0e-5, 1.0e-4]
```

Cartesian product = **3 × 3 = 9 trials per track**, one SLURM array
task per trial. The multi-chunk policy is fixed (`first_chunk_only`)
and hardcoded in `src/train/loop.py`. Confirm the count with
`python scripts/train_pipeline.py --list-trials track=pd`.

### 3.2 Which datasets to train on

Section 2 of `config/train.yaml` — Mode A (fraction-based, default
80/20) or Mode B (explicit lists):

```yaml
corpus:
  train_dataset_ids: ["0001.gmsc", "0002.taiwan_creditcard"]
  test_dataset_ids:  ["0017.SBA_loans_case"]
```

The test list is recorded in each saved checkpoint's
`.provenance.json`, and the training log reports both lists
explicitly:

```text
Training datasets (n=20): 0001.gmsc, 0002.taiwan_creditcard, …
Held-out test datasets (n=5): 0017.SBA_loans_case, …
```

### 3.3 Submit

```bash
N_PD=$(python scripts/train_pipeline.py --list-trials track=pd)
sbatch --array=0-$((N_PD - 1))%4 scripts/slurm/train_pd.slurm

N_LGD=$(python scripts/train_pipeline.py --list-trials track=lgd)
sbatch --array=0-$((N_LGD - 1))%4 scripts/slurm/train_lgd.slurm
```

Each task picks its tuple via `--trial-index $SLURM_ARRAY_TASK_ID`,
runs `cfg.train.epochs` of continued pretraining, then writes:

| Artefact                                                      | Path                                                        |
|---------------------------------------------------------------|-------------------------------------------------------------|
| Final-epoch weights                                           | `checkpoints/trained/<track>/<descriptive_name>.ckpt`       |
| Provenance sidecar (HPs, train/test IDs, GPU, walltime, …)    | `<descriptive_name>.ckpt.provenance.json`                   |
| Manifest row (consumed by the eval)                           | `manifests/<run_name>_<track>.csv`                          |
| Per-epoch CSV (epoch, train_loss, lr, elapsed_sec)            | `results/training/<track>/<descriptive_name>.csv`           |
| Full log                                                      | `logs/train_<track>_<ts>_j<jid>_a<tid>.log`                 |

Filename schema:
`<run_name>_<track>_<base-stem>_lr<lr>_seed<seed>.ckpt`.
Identical re-runs overwrite in place; trials with different HPs land
in distinct files.

---

## 4 · Stage 3 — cross-model benchmark

### 4.1 What gets compared

| Source            | Models                                                                       |
|-------------------|------------------------------------------------------------------------------|
| `baseline`        | XGBoost (Optuna-tuned), CatBoost (Optuna-tuned), LogReg (PD), LinReg (LGD)   |
| `tabpfn-untuned`  | One per checkpoint in `cfg.tunable.<track>_base_paths` (the "before" weights) |
| `tabpfn-trained`  | Every OK row in `manifests/<run_name>_<track>.csv` (the "after" weights)     |

Every continued-pretrained checkpoint is picked up automatically.

### 4.2 Splits

5-fold stratified CV per test dataset (`cfg.cv.n_folds = 5`). Each
train fold is 80/20-split into sub-train + inner-val. The inner-val
split is shared between the Optuna HPO objective (XGBoost / CatBoost)
and the F1-threshold tuner (PD). TabPFN inference is capped at
`cfg.max_rows_tabpfn = 100 000`; non-TabPFN baselines see the full
dataset.

### 4.3 Reruns: skip-existing is the default

Before scoring, each (model × dataset) pair is checked against
existing CSVs under
`results/benchmark/<TRACK>/<method-dirname>/`. Pairs that already
have an `OK` row are **skipped**. So:

- **First run** — scores every baseline + every tabpfn-untuned +
  every tabpfn-trained variant.
- **Re-run with a new trained checkpoint** — scores only the new
  checkpoint on its test datasets. XGBoost / CatBoost / LogReg /
  LinReg / untuned-TabPFN are reused from disk.

Force a fresh scoring with `--rerun`. Rescore just one method by
deleting its directory under `results/benchmark/<TRACK>/` and
re-running.

### 4.4 Submit

```bash
N_PD=$(python scripts/eval_pipeline.py --list-tasks track=pd)
sbatch --array=0-$((N_PD - 1))%32 scripts/slurm/eval_pd.slurm

N_LGD=$(python scripts/eval_pipeline.py --list-tasks track=lgd)
sbatch --array=0-$((N_LGD - 1))%32 scripts/slurm/eval_lgd.slurm
```

Tasks whose pair is already scored exit zero (the skip guard fires
before any heavy work).

### 4.5 Results layout

```text
$VSC_DATA/CreditPFN/results/
├── benchmark/                              ← eval pipeline
│   ├── PD/
│   │   ├── xgboost/                                  creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   ├── catboost/                                 creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   ├── logreg/                                   creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   ├── tabpfn-untuned__v2.6-default/             creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   └── tabpfn-trained__v2.6-default__lr1e-04/    creditpfn_<ts>__task<N>_ds-<id>.csv
│   └── LGD/  …
└── training/                               ← train pipeline (per-epoch CSVs)
    ├── pd/
    │   └── creditpfn_pd_<base-stem>_lr1e-04_seed42.csv
    └── lgd/  …
```

Every benchmark invocation gets a fresh `<timestamp>` — earlier runs
are never overwritten.

Aggregate across runs:

```python
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in glob.glob(
    "$VSC_DATA/CreditPFN/results/benchmark/PD/*/creditpfn_*.csv")],
    ignore_index=True)
df.groupby(["model_name", "model_source"])[
    ["roc_auc", "f1", "log_loss", "pr_auc"]
].agg(["mean", "std", "count"])
```

Row schema: `roc_auc, log_loss, pr_auc, optimal_threshold, f1,
accuracy, precision, recall` for classification; `rmse, mae, r2,
neg_nll` for regression.

---

## 5 · Common workflows

### 5.1 Add a new dataset

1. Upload the raw CSV to `$VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/<id>.csv`
   (WinSCP / FileZilla / `scp`).
2. On your laptop, register a metadata entry + surgical-fix function
   in `src/data/preprocessing.py`. Commit + push.
3. On VSC: `git pull && bash scripts/slurm/submit_full_pipeline.sh`.
   The data stage processes only the new ID; training re-runs all
   variants (the corpus split changed); the eval scores only the new
   (model × dataset) cells thanks to skip-existing.

### 5.2 Test a single LR across all bases

Edit `config/train.yaml`:

```yaml
tunable:
  learning_rates: [1.0e-4]
```

Grid drops to `3 × 1 = 3` per track. Re-submit; `--list-trials`
reflects the new count.

### 5.3 Benchmark one method against one trained checkpoint

```bash
python scripts/eval_pipeline.py track=pd \
    --method xgboost \
    --method "tabpfn-trained[creditpfn_pd_tabpfn-v2.6-classifier-v2.6_default_lr1e-04_seed42]"

# Score one model on a single test dataset (debugging):
python scripts/eval_pipeline.py track=pd --test-dataset 0001.gmsc

# Force re-scoring (ignore existing CSVs):
python scripts/eval_pipeline.py track=pd --method xgboost --rerun
```

---

## 6 · Failure-mode cheat sheet

| Symptom                                                 | What to do                                                                                                       |
|---------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Array task produces no log file                         | Slurm `--output=/dev/null` is set; check the script's `exec >` redirection produced a valid log path.            |
| One trial fails, the rest succeed                       | Manifest row gets `status=FAIL`; the eval skips that checkpoint (filters on OK rows).                            |
| Eval task says `KeyError: <id> not in cache`            | The auto-cache hook re-runs the data pipeline for missing IDs — let it finish.                                   |
| Out-of-memory on `gpu_h100`                             | Lower `train.n_finetune_ctx_plus_query_samples` from 100 000 to 50 000.                                          |
| Wrong test set scored for a checkpoint                  | Check the checkpoint's `.provenance.json` `test_datasets` field. The eval reads that file, not the live cfg.     |
| Eval re-run produces 0 new CSVs                         | The skip-existing guard fired — every pair was already scored. Pass `--rerun` to force.                          |
| `git pull` fails on first push from VSC                 | SSH key not yet registered with GitHub — re-do §0.2.                                                              |

---

## TL;DR

```bash
# One-time, in an OnDemand shell:
cd $VSC_DATA && git clone <repo-url> CreditPFN && cd CreditPFN
mamba create -y -n CreditPFN python=3.12 && source activate CreditPFN
pip install -r requirements.txt
# Upload raw datasets to $VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/
# and base .ckpt files to $VSC_DATA/CreditPFN/checkpoints/ via WinSCP.

# Per experiment:
git pull
bash scripts/slurm/submit_full_pipeline.sh
squeue --me --clusters=genius,wice
```

| Where to look | Path                                                              |
|---------------|-------------------------------------------------------------------|
| Logs          | `$VSC_DATA/CreditPFN/logs/<task>_<ts>_j<jid>_a<tid>.log`          |
| Models        | `$VSC_DATA/CreditPFN/checkpoints/trained/<track>/*.ckpt`          |
| Results       | `$VSC_DATA/CreditPFN/results/benchmark/<TRACK>/<method>/*.csv`    |
| Per-epoch     | `$VSC_DATA/CreditPFN/results/training/<track>/*.csv`              |
