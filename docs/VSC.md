# Running CreditPFN on the VSC

Everything you type to get a run onto the KU Leuven cluster and the results back onto your
laptop, in the order you do it. Nothing else — what the sweep contains and why is
[`METHOD.md`](METHOD.md), what each run measured is [`RESULTS.md`](RESULTS.md).

Two facts that explain most of this document:

- **You always log in to Genius.** wICE and Mindwell have no login node of their own; you
  submit to them from Genius with `--clusters=<name>`, which every `.slurm` here already
  does.
- **Big files live on project storage, readable files on `$VSC_DATA`.** You browse the
  second directly; you have to pull the first down before you can look at it. That split
  is why §3 exists.

---

## 1 · First time only

### 1.1 Get a shell

OnDemand portal → **Clusters → Login (Server) Shell Access**. The prompt looks like
`[Aug/12 15:21] vsc38338@tier2-p-login-3 $`. Every command in §1 and §2 runs there.

### 1.2 Clone and build the environment

```bash
cd $VSC_DATA
git clone --recursive https://github.com/andreasgoethals/CreditPFN.git
cd CreditPFN
conda create -y -n CreditPFN --clone base    # reuses the base torch/CUDA stack
conda activate CreditPFN
pip install -e ".[dev,notebooks]"
```

The env **must** be called `CreditPFN` — that is the name the job scripts activate.

Two installs the `pip install` above cannot get right on its own:

```bash
# PyPI's tabpfn is pinned at 2.2.x with an older API; training TypeErrors on first load.
pip install --upgrade "tabpfn @ git+https://github.com/PriorLabs/tabPFN.git@main"

# The [finetune] extra is REQUIRED, not optional: it carries `transformers`. Without it
# you get an install that works for inference and dies at training with
# ModuleNotFoundError: transformers.
pip install --upgrade "tabicl[finetune]>=2.1.1,<3"
```

Verify before spending GPU time:

```bash
python -c "from src.train.tabpfn_compat import smoke_test; smoke_test('pd'); smoke_test('lgd')"
python -c "from src.train.tabicl_compat import smoke_test; smoke_test('pd')"
```

### 1.3 Upload the datasets

Raw CSVs go to **project storage**, not `$VSC_DATA`. From your laptop:

```bash
rsync -ah --info=progress2 data/raw/ \
  vsc38338@login.hpc.kuleuven.be:/lustre1/project/stg_00211/CreditPFN/data/raw/
```

Layout the pipeline expects: `data/raw/pd/<id>.csv` and `data/raw/lgd/<id>.csv`.

### 1.4 Stage the base checkpoints

**From a login node** — compute nodes have no outbound internet, and the loaders pass
`allow_auto_download=False` so a missing file fails loudly instead of silently downloading
mid-job.

```bash
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
dest = "/lustre1/project/stg_00211/CreditPFN/checkpoints"
os.makedirs(dest, exist_ok=True)
for repo, files in {
    "jingang/TabICL": ("tabicl-classifier-v2-20260212.ckpt",
                       "tabicl-regressor-v2-20260212.ckpt"),
}.items():
    for f in files:
        shutil.copy2(hf_hub_download(repo, f), f"{dest}/{f}")
        print("staged", f)
PY
```

TabPFN's v2.6 / v3 base weights go in the same directory — upload them with the same
`rsync` as §1.3 if you already have them locally.

---

## 2 · Every run

```bash
# 1. Get this run's code onto the cluster. The cluster pulls origin/main, so anything
#    committed-but-unpushed does not exist here.
cd $VSC_DATA/CreditPFN && git pull origin main && git submodule update --init

# 2. See what the previous run left behind. Deletes nothing.
python -m src.utils.clean_run

# 3. Wipe it. Keeps data/raw/, the base checkpoints, and data/processed/.
python -m src.utils.clean_run --clean

# 4. Launch data → train → eval. Fire once and walk away.
bash scripts/slurm/run_full_pipeline.sh

# 5. Watch. Two controllers, so two queries.
squeue -M mindwell -u $USER      # training  (gpu_b200)
squeue -M wice     -u $USER      # data, gate, eval
```

Add `--processed` to step 3 only when the sanitising logic changed — rebuilding the cache
costs far more than re-running the models.

Useful variants:

```bash
STAGES="eval" bash scripts/slurm/run_full_pipeline.sh    # re-score, skip training
STAGES="data train" bash scripts/slurm/run_full_pipeline.sh
EVAL_TASKS=8 bash scripts/slurm/run_full_pipeline.sh     # fewer, longer eval tasks
```

Cancel everything:

```bash
scancel -M mindwell -u $USER ; scancel -M wice -u $USER
```

The three stages are sequenced by **sentinel files**, not by `--dependency`: VSC runs wICE
and Mindwell as separate Slurm controllers and cross-cluster dependencies do not work.
That is why the launcher looks more complicated than a dependency chain, and why you must
not "simplify" it into one.

---

## 3 · Getting the results back

A finished run is spread across both storage tiers, and the half you most want to read is
the half you cannot browse. Pull it down **from your laptop**, into the repository, where
the notebooks already look for it.

### 3.1 The three things worth downloading

| From | To (in your local checkout) | What it is | Size |
|---|---|---|---|
| `$VSC_DATA/CreditPFN/output/manifests/` | `output/manifests/` | one row per trial, per-epoch CSVs, the resolved configs | a few MB |
| `$VSC_DATA/CreditPFN/output/logs/` | `output/logs/` | one `.log` per task — the only record of what happened | tens of MB |
| `/lustre1/.../CreditPFN/output/results/` | `output/results/` | per-fold scores, one CSV per model × dataset | tens of MB |

```bash
VSC=vsc38338@login.hpc.kuleuven.be
DATA=/data/leuven/383/vsc38338/CreditPFN
STAGE=/lustre1/project/stg_00211/CreditPFN

rsync -ah --info=progress2 "$VSC:$DATA/output/manifests/" output/manifests/
rsync -ah --info=progress2 "$VSC:$DATA/output/logs/"      output/logs/
rsync -ah --info=progress2 "$VSC:$STAGE/output/results/"  output/results/
```

**Do not download `checkpoints/trained/`.** It is 5–8 GB per sweep and nothing local reads
it: the notebooks work from the manifests and results above. Pull one `.ckpt` by hand only
if you intend to run inference with it.

### 3.2 Check what arrived

```bash
python -m src.utils.clean_run          # lists every local tree with file counts
```

Then confirm the run is complete rather than half-transferred:

```bash
python - <<'PY'
import pandas as pd
from src.utils.paths import manifests_dir, results_dir
for track in ("pd", "lgd"):
    m = manifests_dir() / f"creditpfn_{track}.csv"
    if m.is_file():
        df = pd.read_csv(m)
        print(f"{track}: {len(df)} trials", df["status"].value_counts().to_dict())
        print(f"      steps {df['total_optimizer_steps'].min()}–{df['total_optimizer_steps'].max()}"
              f", corpus {df['n_train_datasets'].iloc[0]} train / {df['n_test_datasets'].iloc[0]} test")
print("result CSVs:", sum(1 for _ in results_dir().rglob("*.csv")))
PY
```

A trial count below the grid size, or a `total_optimizer_steps` far under
`train.target_total_steps`, means the run is partial — check the logs before believing any
number from it.

### 3.3 Turn it into figures

```bash
python -m src.utils.run_notebooks
```

Writes `output/figures/<notebook>/*.pdf`, one shared `output/figures/CAPTIONS.md`, and
`output/All_Results.md`. Those captions are the manuscript's captions.

### 3.4 Write it down

Add one row to the Runs table in [`AGENTS_MEMORY.md`](AGENTS_MEMORY.md) and one entry with
its Configuration table to [`RESULTS.md`](RESULTS.md). Both are read before the next run is
designed; a run nobody recorded gets repeated.

---

## 4 · When it breaks

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: transformers` at training start | The `[finetune]` extra is missing. §1.2. Inference works without it, which is why this only shows up on the cluster. |
| `TypeError` on the first model load | PyPI's stale `tabpfn`. Install the Prior Labs wheel, §1.2. |
| `pip install` succeeded but the job disagrees | An active virtualenv silently beats `conda activate`. `deactivate`, then check `which python pip`. Every job log's `Active conda env:` line is the authority. |
| "missing raw file … skipped" for every dataset | The CSVs are not where the resolver looked. The launcher prints the resolved `CREDITPFN_DATA_ROOT`; compare it with §1.3. |
| Jobs sit `PENDING` for hours | Normal on a busy partition. A **shorter** `--time` backfills better than a longer one — the 48 h request in run-5 got 1–2 GPUs, the 10 h request in run-7 got 15–21. |
| "user env retrieval failed requeued held" | A cross-cluster `sbatch` without `--export`. Every script here sets `#SBATCH --export=ALL`; do not remove it. |
| Eval re-run produces no new CSVs | The skip-existing guard fired — everything was already scored. `--rerun` forces it. |
| A trial has `status=FAIL` | The manifest keeps the row and the eval skips that checkpoint. Read its log; the grid is still usable, and partial grids are flagged as such. |
| Out of memory during training | Lower `finetuning.max_rows_per_epoch` in `config/data.yaml`, then **re-run the probe** — the caps are measurements, not guesses ([`METHOD.md`](METHOD.md#3-context-size-caps)). |
| Wrong test set scored for a checkpoint | The eval reads `<ckpt>.provenance.json`, not the live config. A mismatch warning names both. |

---

## 5 · Reference

### Storage

| Tier | Path | Holds | Notes |
|---|---|---|---|
| **Personal data** | `$VSC_DATA/CreditPFN/` | the repo, `output/` except results | 75 GiB, backed up, browsable |
| **Project storage** | `/lustre1/project/stg_00211/CreditPFN/` | datasets, checkpoints, `output/results/` | large, backed up, **low inode budget** — few big files, not thousands of small ones |

`$VSC_SCRATCH` is not used: it is purged after 30 days *without access*, and neither `mv`
nor `rsync -a` counts as an access.

### Clusters

| Cluster | Partition | Used for | Why |
|---|---|---|---|
| Genius | — | login and submission only | the only login node |
| Mindwell | `gpu_b200` | continued pretraining | 192 GiB VRAM per GPU is what allows the large in-context sizes; 24 GPUs cluster-wide |
| wICE | `gpu_h100`, `gpu_a100` | evaluation | the eval is inference-bound and fits comfortably |
| wICE | `batch` | data stage, eval gate | CPU only |

Account: `lp_verbekelab` (verify with `sacctmgr -s show user $USER cluster=mindwell` —
note `sacctmgr` takes no `-M`, unlike `squeue`/`sinfo`/`scancel`).

### Cost

B200 time is charged at **437.5 credits per GPU-minute**. A full run at the current grid is
roughly 120 GPU-hours ≈ 3.1 M credits. Check the balance with `sam-balance` before
launching, and prefer a short `--time` — it costs the same and starts sooner.
