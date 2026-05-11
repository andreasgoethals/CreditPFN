# CreditPFN on VSC — end-to-end deployment guide

Step-by-step recipe for running the full pipeline (**data → train →
eval**) on the KU Leuven VSC cluster via the **Open OnDemand web
portal** at <https://ondemand.hpc.kuleuven.be>. The whole flow is one
command (`bash scripts/slurm/submit_full_pipeline.sh`); the sections
below show how to vary hyperparameters and interpret the outputs.

---

## 0. One-time setup (per VSC account)

Everything happens **inside the browser**. No SSH, no local terminal,
no MFA prompts beyond logging in to OnDemand once.

1. Open <https://ondemand.hpc.kuleuven.be>, click the KU Leuven logo,
   sign in with your KU Leuven credentials. You land on the OnDemand
   dashboard.
2. The repo is already at `$VSC_DATA/CreditPFN/` — it's auto-pulled
   from GitHub by the institutional sync, so source files always
   reflect the latest pushed commit. **You never run `git pull`
   manually.**
3. Open the **Interactive Shell** app
   (`Clusters → Login (Server) Shell Access` ⇒ a shell on a login node
   in a new browser tab — fine for setup; for anything heavier launch
   the dedicated *Interactive Shell* app under *Interactive Apps* so
   you land on a compute node). The login shell is enough for the
   commands below.
4. Create the conda env **once**:

   ```bash
   cd $VSC_DATA/CreditPFN
   mamba create -y -n CreditPFN python=3.12      # or `conda` if no mamba
   source activate CreditPFN
   pip install -r requirements.txt

   # Sanity check — should print 9 (3 bases × 3 LRs, default cfg).
   python scripts/train_pipeline.py --list-trials track=pd
   ```

5. **Upload the unsynced files** — anything `.gitignore` excludes
   from git is not auto-synced. There are two categories:

   | What                                              | Destination                                     | How                            |
   |---------------------------------------------------|-------------------------------------------------|--------------------------------|
   | Raw credit-risk datasets (`*.csv`)                | `$VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/`     | OnDemand **Files** app, WinSCP, or `scp`/`rsync` |
   | Base TabPFN checkpoints (`tabpfn-v2.5-*.ckpt`, `tabpfn-v2.6-*.ckpt`) | `$VSC_DATA/CreditPFN/checkpoints/`              | Same — drop the 6 `.ckpt` files in the folder |

   The OnDemand **Files** app handles small batches well (drag-and-drop
   in the browser). For the ~25-dataset corpus a single tarball
   uploaded via WinSCP to scratch and untarred from a login shell is
   fastest. As the corpus grows past a few GB, use **Globus** (the
   button is in the Files app) instead.

   Everything else — code, configs, tests, docs, scripts, slurm
   templates — is in git and arrives automatically with the auto-pull.

Path roots used throughout (auto-detected — see `src/utils/paths.py`):

| Variable                    | What it covers                                                                                | Default on VSC                  |
|-----------------------------|-----------------------------------------------------------------------------------------------|---------------------------------|
| `$CREDITPFN_DATA_ROOT`      | `data/raw/`, `data/processed/`, `data/cached/` (big I/O artefacts)                            | `$VSC_SCRATCH/CreditPFN`        |
| `$CREDITPFN_OUTPUT_ROOT`    | `logs/`, `manifests/`, `dedup/`, `checkpoints/trained/`, `results/` (small + must survive)    | `$VSC_DATA/CreditPFN`           |

The SLURM scripts set both env vars explicitly; an interactive shell
session uses the same defaults via auto-detection.

---

## 1. The full chain in one command

From an Interactive Shell on a login node:

```bash
cd $VSC_DATA/CreditPFN
bash scripts/slurm/submit_full_pipeline.sh
```

Submits **six** SLURM jobs with `--dependency=afterok` chaining:

```
data.slurm                         (1 job;  genius batch CPU,    ~15 min)
        ↓ afterok
train_pd.slurm  (array)            (N jobs; wice gpu_h100,       ~2 h each)
train_lgd.slurm (array)            (N jobs; wice gpu_h100,       ~2 h each)
        ↓ afterok
eval_pd.slurm   (array)            (M jobs; wice gpu_h100,       ~1 h each)
eval_lgd.slurm  (array)            (M jobs; wice gpu_h100,       ~1 h each)
```

Optional knobs:

```bash
TRAIN_CONCURRENCY=4    EVAL_CONCURRENCY=32    \
TRACKS="pd lgd"        bash scripts/slurm/submit_full_pipeline.sh
```

Watch progress in the **Active Jobs** OnDemand app, or from a shell:
`squeue --me --clusters=genius,wice`. Per-task logs land in **one
flat directory**:
`$VSC_DATA/CreditPFN/logs/<task>_<YYYYMMDD>_<HHMMSS>_j<jid>_a<tid>.log`.

---

## 2. Stage 1 — data preprocessing

```
data/raw/{pd,lgd}/<id>.csv          (you uploaded)
        ↓ dedup --pass pre          dedup/doubles_{track}_pre.csv
        ↓ register                  manifest_{pd,lgd}.csv
        ↓ sanitize                  data/processed/{pd,lgd}/<id>.sanitized.csv
        ↓ dedup --pass post         dedup/doubles_{track}_post.csv
        ↓ dataset (chunk + cache)   data/cached/{pd,lgd}/<id>/chunk_*.npz
```

~15 min on a Genius `batch` node for the current 25-dataset corpus.
Scales linearly in #datasets; the `FeatureAgglomeration` step is the
bottleneck for the wide ones (> 2 000 features).

Submit just this stage: `sbatch scripts/slurm/data.slurm`.

The pipeline is **idempotent** — re-running it skips datasets whose
cache fingerprint matches the current manifest row, processed CSV
content, and dataset-config hash. To force a fresh rebuild, pass
`--fresh` (uncomment the line in `data.slurm`).

---

## 3. Stage 2 — continued pretraining

### 3.1 Choosing what to train

Open **`config/train.yaml`** in the OnDemand Files editor. Section 0
lists two tunable knobs — the only things that vary across runs:

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

Cartesian product = **3 × 3 = 9 trials per track** = one SLURM array
task per trial = one trained checkpoint. (The multi-chunk policy is
fixed to `first_chunk_only` — one chunk per parent dataset — hardcoded
in `src/train/loop.py`.) Trim the lists to shrink the sweep; re-check
the count with `python scripts/train_pipeline.py --list-trials track=pd`.

> **Editor tip.** OnDemand's `ctrl+s` saves to your *local* browser
> instead of the file on VSC. Use the *Save* button in the editor's
> top-left corner.

### 3.2 Choosing which datasets to train on

Section 2 of `config/train.yaml` — Mode A (fraction-based, default
80/20) or Mode B (explicit lists):

```yaml
corpus:
  train_dataset_ids: ["0001.gmsc", "0002.taiwan_creditcard"]
  test_dataset_ids:  ["0017.SBA_loans_case"]
```

The test list is recorded in each saved checkpoint's
`.provenance.json` so the eval knows what to score it on later.

### 3.3 Smoke test before bulk submit

From the Interactive Shell:

```bash
python scripts/train_pipeline.py --single \
    track=pd \
    train.epochs=2 \
    corpus.train_dataset_ids=[0001.gmsc]

# Writes (under $VSC_DATA/CreditPFN/):
#   checkpoints/trained/pd/<descriptive_name>.ckpt
#   checkpoints/trained/pd/<descriptive_name>.provenance.json
#   manifests/<run_name>_pd.csv                       (manifest entry)
#   results/training/pd/<descriptive_name>.csv        (per-epoch loss/lr/elapsed)
#   logs/train_pd_<ts>.log                            (full per-trial log)
```

The log line lists the training datasets, the held-out test datasets,
and their IDs explicitly — easy to grep for.

### 3.4 Full submit

```bash
N_PD=$(python scripts/train_pipeline.py --list-trials track=pd)
sbatch --array=0-$((N_PD - 1))%4 scripts/slurm/train_pd.slurm

N_LGD=$(python scripts/train_pipeline.py --list-trials track=lgd)
sbatch --array=0-$((N_LGD - 1))%4 scripts/slurm/train_lgd.slurm
```

Each task picks its tuple via `--trial-index $SLURM_ARRAY_TASK_ID`,
loads the base TabPFN checkpoint, runs `cfg.train.epochs` (default 50)
of continued pretraining, then saves weights +
`.provenance.json` and appends one row to
`manifests/<run_name>_<track>.csv`.

Filename schema:
`<run_name>_<track>_<base-stem>_lr<lr>_seed<seed>.ckpt`.
Identical re-runs overwrite in place; trials with different HPs land
in distinct files, so the manifest grows monotonically.

---

## 4. Stage 3 — cross-model benchmark

### 4.1 What gets compared

| Source            | Models scored on the held-out test datasets                                  |
|-------------------|------------------------------------------------------------------------------|
| `baseline`        | XGBoost (Optuna-tuned), CatBoost (Optuna-tuned), LogReg (PD), LinReg (LGD)   |
| `tabpfn-untuned`  | One per checkpoint in `cfg.tunable.<track>_base_paths` (the "before" weights) |
| `tabpfn-trained`  | Every OK row in `manifests/<run_name>_<track>.csv` (the "after" weights)     |

Every continued-pretrained checkpoint is automatically picked up — no
extra configuration.

### 4.2 Splits per (model × dataset × fold)

5-fold stratified CV per test dataset (`cfg.cv.n_folds = 5`). Each
train fold is further 80/20-split into sub-train + inner-val. The
inner-val split is the **Optuna HPO objective** for XGBoost / CatBoost
AND the **F1-threshold tuning target** for PD. TabPFN inference is
capped at `cfg.max_rows_tabpfn = 100 000` rows; non-TabPFN baselines
see the full dataset.

### 4.3 Reruns: skip-existing is the default

The eval is **idempotent across reruns**. Before scoring, each
(model × dataset) pair is checked against existing CSVs under
`results/benchmark/<TRACK>/<method-dirname>/`. Pairs that already have
an `OK` row are skipped. Concretely:

* Run the eval after the first training sweep → it scores every
  baseline + every tabpfn-untuned + every tabpfn-trained variant.
* Add a new trained checkpoint, re-submit the eval → it scores only
  the new checkpoint on its test datasets. XGBoost, CatBoost, LogReg,
  LinReg, and untuned-TabPFN reuse the rows from the first run.

To force a fresh scoring of everything, pass `--rerun`. To rescore
just one method, delete its directory under `results/benchmark/<TRACK>/`
and re-run normally.

### 4.4 Submit

```bash
N_PD=$(python scripts/eval_pipeline.py --list-tasks track=pd)
sbatch --array=0-$((N_PD - 1))%32 scripts/slurm/eval_pd.slurm

N_LGD=$(python scripts/eval_pipeline.py --list-tasks track=lgd)
sbatch --array=0-$((N_LGD - 1))%32 scripts/slurm/eval_lgd.slurm
```

`--list-tasks` returns `n_models × n_test_datasets`. Slurm tasks
whose pair is already scored exit zero in a few seconds (the skip
guard fires before any heavy work).

### 4.5 Results layout

```
$VSC_DATA/CreditPFN/results/
├── benchmark/                               eval-pipeline output
│   ├── PD/
│   │   ├── xgboost/                         creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   ├── catboost/                        creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   ├── logreg/                          creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   ├── tabpfn-untuned__v2.6-default/    creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   └── tabpfn-trained__v2.6-default__lr1e-04/  creditpfn_<ts>__task<N>_ds-<id>.csv
│   └── LGD/  …
└── training/                                train-pipeline output (per-epoch CSVs)
    ├── pd/
    │   └── creditpfn_pd_<base-stem>_lr1e-04_seed42.csv
    └── lgd/  …
```

Every benchmark invocation gets a new `<timestamp>` — earlier runs
are never overwritten. Aggregate across runs:

```python
import pandas as pd, glob
df = pd.concat([pd.read_csv(f)
                for f in glob.glob("$VSC_DATA/CreditPFN/results/benchmark/PD/*/creditpfn_*.csv")],
               ignore_index=True)
df.groupby(["model_name", "model_source"])[
    ["roc_auc", "f1", "log_loss", "pr_auc"]
].agg(["mean", "std", "count"])
```

Each row has the full metric block: `roc_auc, log_loss, pr_auc,
optimal_threshold, f1, accuracy, precision, recall` for classification;
`rmse, mae, r2, neg_nll` for regression. (Column order is hardcoded
in the eval pipeline — no `metrics:` knob in the config.)

---

## 5. Three common workflows

### 5.1 Add a new dataset and re-train all variants

1. Upload the raw CSV to `$VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/<id>.csv`
   via the OnDemand Files app or WinSCP.
2. On your laptop, register a metadata entry + surgical-fix function
   in `src/data/preprocessing.py`. Commit + push to GitHub → the
   institutional auto-pull picks it up on VSC.
3. From the Interactive Shell on VSC:
   `bash scripts/slurm/submit_full_pipeline.sh`. The data stage
   re-runs only for the new dataset; training re-runs all variants
   (because the corpus split changed); the eval reuses every baseline
   row that's already on disk and only scores the new checkpoints +
   the new test dataset.

### 5.2 Test a single learning rate across all bases

Edit `config/train.yaml` in the OnDemand Files editor:

```yaml
tunable:
  learning_rates: [1.0e-4]
```

Grid drops to `3 × 1 = 3` per track. Re-submit the chain;
`--list-trials` reflects the new count.

### 5.3 Benchmark only XGBoost vs one TabPFN-trained checkpoint

```bash
python scripts/eval_pipeline.py track=pd \
    --method xgboost \
    --method "tabpfn-trained[creditpfn_pd_tabpfn-v2.6-classifier-v2.6_default_lr1e-04_seed42]"

# Or to score one model on a single test dataset (debugging):
python scripts/eval_pipeline.py track=pd --test-dataset 0001.gmsc

# Force re-scoring (ignore existing CSVs):
python scripts/eval_pipeline.py track=pd --method xgboost --rerun
```

---

## 6. Failure-mode cheat sheet

| Symptom                                              | What to do                                                                                                       |
|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Array task hangs without writing a log file          | Slurm `--output=/dev/null` is set; check the script's `exec >` redirection actually computed a valid log path.   |
| One trial fails, the rest succeed                    | The manifest row still gets written with `status=FAIL`; the eval will skip that checkpoint (it filters on OK rows). |
| Eval task gets `KeyError: 0009.bank_status not in cache` | The auto-cache hook in `scripts/eval_pipeline.py` re-runs the data pipeline transparently for missing IDs — wait. |
| Out-of-memory on `gpu_h100`                          | Lower `train.n_finetune_ctx_plus_query_samples` from 100 000 to 50 000.                                          |
| Wrong test set scored for a checkpoint               | Confirm the checkpoint's `.provenance.json` lists the expected `test_datasets`. The eval reads that file, not the live cfg. |
| Eval re-run produces 0 new CSVs                      | The skip-existing guard fired — every pair was already scored. Pass `--rerun` to force.                          |
| "Pull from main" not visible on VSC                  | The auto-pull runs on a fixed schedule. Use the OnDemand Files app to verify the commit hash in `.git/HEAD` if needed. |

---

## TL;DR

```bash
# One-time, inside the OnDemand portal (https://ondemand.hpc.kuleuven.be):
#   1. Open Interactive Shell.
cd $VSC_DATA/CreditPFN
mamba create -y -n CreditPFN python=3.12 && source activate CreditPFN
pip install -r requirements.txt
#   2. Upload raw datasets to $VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/
#      and base .ckpt files to $VSC_DATA/CreditPFN/checkpoints/.

# Per experiment:
#   1. Push the relevant code/config change to GitHub (auto-pulled to VSC).
#   2. From the Interactive Shell:
bash scripts/slurm/submit_full_pipeline.sh
squeue --me --clusters=genius,wice
```

Logs:    `$VSC_DATA/CreditPFN/logs/<task>_<ts>_j<jid>_a<tid>.log`
Models:  `$VSC_DATA/CreditPFN/checkpoints/trained/<track>/*.ckpt`
Results: `$VSC_DATA/CreditPFN/results/benchmark/<TRACK>/<method>/*.csv`
