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
1–4). wICE has no dedicated login node; you always SSH into Genius and
submit to wICE via `#SBATCH --cluster=wice` in the job script, which
every `.slurm` here already does. Every command below runs in that
shell.

### 0.2 Two storage tiers, two purposes

VSC gives every user two storage roots. CreditPFN uses them
deliberately:

| Root             | Holds                                                                                       | Default for CreditPFN     |
|------------------|---------------------------------------------------------------------------------------------|---------------------------|
| `$VSC_DATA`      | Code + small artefacts that must survive — logs, manifests, results, trained checkpoints.   | `$VSC_DATA/CreditPFN`     |
| `$VSC_SCRATCH`   | Large I/O — raw datasets, processed CSVs, cached `.npz` chunks. Big, fast, **not backed up**. | `$VSC_SCRATCH/CreditPFN`  |

The two env vars `$CREDITPFN_DATA_ROOT` (→ scratch) and
`$CREDITPFN_OUTPUT_ROOT` (→ data) are auto-detected from `$VSC_DATA` /
`$VSC_SCRATCH`; you don't have to set them manually unless you want a
non-default layout.

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
├── tests/               pytest suite (228+ tests)
└── requirements.txt
```

### 0.4 Create the conda env (one time)

```bash
mamba create -y -n CreditPFN python=3.12      # or `conda` if mamba isn't installed
source activate CreditPFN
pip install -r requirements.txt
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

Two things have to be transferred manually — everything else is in git:

| What                                                                    | Destination                                  | How                                      |
|-------------------------------------------------------------------------|----------------------------------------------|------------------------------------------|
| Raw credit-risk datasets (`*.csv`)                                      | `$VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/`  | WinSCP / FileZilla / `scp` / `rsync`     |
| Base TabPFN checkpoints (`tabpfn-v3-*.ckpt`, `tabpfn-v2.6-*.ckpt`, …)    | `$VSC_DATA/CreditPFN/checkpoints/`           | Same — or `wget` from Hugging Face       |

OnDemand's built-in **Files** app can browse `$VSC_HOME` and `$VSC_DATA`
but **not** `$VSC_SCRATCH`, which is exactly where the datasets need to
live. WinSCP / FileZilla / `scp` reach scratch over SFTP. For files
larger than a few GB, prefer **Globus** (button in the OnDemand Files
app).

Base checkpoints can also be fetched from Hugging Face directly on the
login node — see `docs/CHECKPOINTS.md` for the exact `.ckpt` filenames
the loader expects.

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

That submits six SLURM jobs with `afterok` chaining:

```text
data.slurm                          ──┐  CPU (wICE batch)
        ↓ afterok                     │
train_pd.slurm   (array, N jobs)    ──┤  GPU (wICE gpu_h100)
train_lgd.slurm  (array, N jobs)    ──┤
        ↓ afterok                     │
eval_pd.slurm    (array, M jobs)    ──┤  GPU
eval_lgd.slurm   (array, M jobs)    ──┘
```

Optional knobs (set before invoking):

```bash
TRAIN_CONCURRENCY=4    EVAL_CONCURRENCY=32    \
TRACKS="pd lgd"        bash scripts/slurm/submit_full_pipeline.sh
```

Watch progress with **Active Jobs** in OnDemand, or
`squeue --me --clusters=genius,wice`. Every task writes a single log
file at
`$VSC_DATA/CreditPFN/logs/<task>_<YYYYMMDD>_<HHMMSS>_j<jid>_a<tid>.log`.

---

## 2 · Stage 1 — data preprocessing

CPU-only. Reads from `$VSC_SCRATCH/CreditPFN/data/raw/`, writes
processed and cached artefacts back to scratch:

```text
data/raw/{pd,lgd}/<id>.csv          (you uploaded)
        ↓ dedup --pass pre          → dedup/doubles_{track}_pre.csv
        ↓ register                  → manifest_{pd,lgd}.csv
        ↓ sanitize                  → data/processed/{pd,lgd}/<id>.sanitized.csv
        ↓ dedup --pass post         → dedup/doubles_{track}_post.csv
        ↓ dataset (chunk + cache)   → data/cached/{pd,lgd}/<id>/chunk_*.npz
```

Submit just this stage: `sbatch scripts/slurm/data.slurm`.

**Idempotent.** Re-running skips datasets whose cache fingerprint
matches the current manifest, processed CSV, and dataset config.
Pass `--fresh` (uncomment the line in `data.slurm`) to rebuild from
scratch.

---

## 3 · Stage 2 — continued pretraining

### 3.1 What gets swept

`config/train.yaml` Section 0 declares the cartesian sweep:

```yaml
tunable:
  classifier_base_paths:
    - "checkpoints/tabpfn-v3-classifier-v3_default.ckpt"
    - "checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"
    - "checkpoints/tabpfn-v2.5-classifier-v2.5_default-2.ckpt"
    - "checkpoints/tabpfn-v2.5-classifier-v2.5_default.ckpt"
  regressor_base_paths:
    - "checkpoints/tabpfn-v3-regressor-v3_default.ckpt"
    - "checkpoints/tabpfn-v2.6-regressor-v2.6_default.ckpt"
    - "checkpoints/tabpfn-v2.5-regressor-v2.5_default.ckpt"
    - "checkpoints/tabpfn-v2.5-regressor-v2.5_real.ckpt"
  learning_rates: [1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4]
  use_lora:       [false, true]
```

Default = **4 bases × 4 LRs × 2 LoRA = 32 trials per track**. One
SLURM array task per trial. The multi-chunk policy is fixed to
`first_chunk_only` (one chunk per parent dataset) and hardcoded in
`src/train/loop.py`. Recompute the current count any time with:

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

```bash
N_PD=$(python scripts/train_pipeline.py --list-trials track=pd)
sbatch --array=0-$((N_PD - 1))%4 scripts/slurm/train_pd.slurm

N_LGD=$(python scripts/train_pipeline.py --list-trials track=lgd)
sbatch --array=0-$((N_LGD - 1))%4 scripts/slurm/train_lgd.slurm
```

The `.slurm` files have generous default array bounds; over-sizing is
safe (surplus array tasks exit zero cleanly). Each task writes:

| Artefact                                           | Path                                                             |
|----------------------------------------------------|------------------------------------------------------------------|
| Final-epoch weights                                | `checkpoints/trained/<track>/<descriptive_name>.ckpt`            |
| Provenance sidecar (HPs, train/test IDs, GPU, …)   | `<descriptive_name>.ckpt.provenance.json`                        |
| Manifest row (consumed by the eval pipeline)       | `manifests/<run_name>_<track>.csv`                               |
| Per-epoch CSV (loss, lr, train/test metric, time)  | `results/training/<track>/<descriptive_name>.csv`                |
| Full run log                                       | `logs/train_<track>_<ts>_j<jid>_a<tid>.log`                      |

Filename schema:
`<run_name>_<track>_<base-stem>_lr<lr>_seed<seed>[_lora].ckpt`.

---

## 4 · Stage 3 — cross-model benchmark

### 4.1 What gets compared

| Source            | Models                                                                                |
|-------------------|---------------------------------------------------------------------------------------|
| `baseline`        | XGBoost + CatBoost (Optuna-tuned), LogReg (PD), LinReg (LGD)                          |
| `tabpfn-untuned`  | One per checkpoint in `cfg.tunable.<track>_base_paths` — the "before" weights         |
| `tabpfn-trained`  | Every OK row in `manifests/<run_name>_<track>.csv` — the "after" weights              |

Every continued-pretrained checkpoint is picked up automatically.

### 4.2 Splits per (model × dataset × fold)

5-fold stratified CV per test dataset. Each train fold is 80/20-split
again into sub-train + inner-val; that inner-val is shared by the
Optuna HPO objective (XGBoost / CatBoost) and the F1-threshold tuner
(PD). TabPFN inference is row-capped at `cfg.max_rows_tabpfn = 100 000`;
non-TabPFN baselines see the full dataset.

### 4.3 Re-runs are idempotent

Before scoring, each `(model × dataset)` pair is checked against the
existing CSVs under
`results/benchmark/<TRACK>/<method-dirname>/`. Pairs whose **all
folds** are already `OK` are skipped:

- **First run** — scores every baseline + untuned + trained variant.
- **Re-run after adding a new trained checkpoint** — scores only the
  new checkpoint's pairs. XGBoost / CatBoost / LogReg / LinReg /
  untuned-TabPFN are reused from disk.

Force a fresh scoring with `--rerun`. To rescore a single method,
delete its directory under `results/benchmark/<TRACK>/` and re-submit.

### 4.4 Submit

```bash
N_PD=$(python scripts/eval_pipeline.py --list-tasks track=pd)
sbatch --array=0-$((N_PD - 1))%32 scripts/slurm/eval_pd.slurm

N_LGD=$(python scripts/eval_pipeline.py --list-tasks track=lgd)
sbatch --array=0-$((N_LGD - 1))%32 scripts/slurm/eval_lgd.slurm
```

Already-scored pairs exit zero in seconds; surplus array tasks (if the
array is sized larger than the grid) do the same.

### 4.5 Output layout

```text
$VSC_DATA/CreditPFN/results/
├── benchmark/                               eval-pipeline output
│   ├── PD/
│   │   ├── xgboost/                                  creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   ├── catboost/                                 creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   ├── logreg/                                   creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   ├── tabpfn-untuned__v3-default/               creditpfn_<ts>__task<N>_ds-<id>.csv
│   │   └── tabpfn-trained__v3-default__lr1e-04/      creditpfn_<ts>__task<N>_ds-<id>.csv
│   └── LGD/  …
└── training/                                train-pipeline output (per-epoch CSVs)
    ├── pd/   creditpfn_pd_<base-stem>_lr1e-04_seed42.csv
    └── lgd/  …
```

Every benchmark invocation gets a fresh `<timestamp>` — earlier runs
are never overwritten. Aggregate across runs:

```python
import pandas as pd, glob
files = glob.glob("$VSC_DATA/CreditPFN/results/benchmark/PD/*/creditpfn_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.groupby(["model_name", "model_source"])[
    ["roc_auc", "f1", "log_loss", "pr_auc"]
].agg(["mean", "std", "count"])
```

Row schema: `roc_auc, log_loss, pr_auc, optimal_threshold, f1, accuracy,
precision, recall` for classification; `rmse, mae, r2, neg_nll` for
regression.

---

## 5 · Common workflows

### 5.1 Add a new dataset

1. Upload the raw CSV to `$VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/<id>.csv`.
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
  learning_rates: [1.0e-4]
```

Grid drops to `4 × 1 × 2 = 8` per track. Re-submit; `--list-trials`
reflects the new count.

### 5.3 Benchmark a subset

```bash
# Only XGBoost + one trained checkpoint:
python scripts/eval_pipeline.py track=pd \
    --method xgboost \
    --method "tabpfn-trained[creditpfn_pd_tabpfn-v3-classifier-v3_default_lr1e-04_seed42]"

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
| `sbatch: error: Batch job submission failed: Job dependency problem` | A stage targets a different cluster than its dependency — VSC has separate Slurm controllers for Genius and wICE, so cross-cluster `afterok:` chains don't work. Every `.slurm` header in this repo uses `#SBATCH --cluster=wice` for exactly this reason. (The `<jobid>;wice` suffix in `sbatch --parsable` output on a Genius login is NOT this error — it just means the jobid lives in wICE's controller, which is normal.) |
| `TypeError` on first model load in training             | PyPI tabpfn 2.2.1 has the old API — install the Prior Labs wheel (see §0.4).                                     |
| Array task produces no log file                         | The SLURM `--output=/dev/null` is set; check the `exec >` redirection in the `.slurm` script.                    |
| One trial fails, the rest succeed                       | Manifest row gets `status=FAIL`; the eval auto-skips that checkpoint.                                            |
| Eval task says `KeyError: <id> not in cache`            | The auto-cache hook re-runs the data pipeline for missing IDs — let it finish.                                   |
| Out-of-memory on `gpu_h100`                             | Lower `train.n_finetune_ctx_plus_query_samples` from 100 000 to 50 000.                                          |
| Wrong test set scored for a checkpoint                  | Check `<checkpoint>.ckpt.provenance.json` — the eval reads that, not the live cfg.                               |
| Eval re-run produces 0 new CSVs                         | The skip-existing guard fired — every pair was already scored. Pass `--rerun` to force.                          |

---

## TL;DR

```bash
# One-time, in an OnDemand shell:
cd $VSC_DATA
git clone https://github.com/andreasgoethals/CreditPFN.git && cd CreditPFN
mamba create -y -n CreditPFN python=3.12 && source activate CreditPFN
pip install -r requirements.txt
pip install --upgrade "tabpfn @ git+https://github.com/PriorLabs/tabPFN.git@main"

# Upload raw datasets to $VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/
# and base .ckpt files to $VSC_DATA/CreditPFN/checkpoints/ via WinSCP.

# Per experiment:
source activate CreditPFN     # if not already in this shell
git pull
bash scripts/slurm/submit_full_pipeline.sh
squeue --me --clusters=genius,wice
```

| Where to look | Path                                                              |
|---------------|-------------------------------------------------------------------|
| Code          | `$VSC_DATA/CreditPFN/src/`                                        |
| Data          | `$VSC_SCRATCH/CreditPFN/data/`                                    |
| Logs          | `$VSC_DATA/CreditPFN/logs/<task>_<ts>_j<jid>_a<tid>.log`          |
| Models        | `$VSC_DATA/CreditPFN/checkpoints/trained/<track>/*.ckpt`          |
| Results       | `$VSC_DATA/CreditPFN/results/benchmark/<TRACK>/<method>/*.csv`    |
| Per-epoch     | `$VSC_DATA/CreditPFN/results/training/<track>/*.csv`              |
