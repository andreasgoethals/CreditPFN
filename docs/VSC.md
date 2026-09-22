# Running CreditPFN on the VSC

Everything you type to get a run onto the KU Leuven cluster and the results back onto your
laptop, in the order you do it. Nothing else — what the sweep contains and why is
[`METHOD.md`](METHOD.md), what each run measured is [`RESULTS.md`](RESULTS.md) .

Two facts that explain most of this document:

- **You always log in to Genius.** wICE and Mindwell have no login node of their own; you
  submit to them from Genius with `--clusters=<name>`, which every `.slurm` here already
  does.
- **Big files live on project storage, readable files on `$VSC_DATA`.** You browse the
  second directly; you have to pull the first down before you can look at it. That split
  is why §3 exists.

## Contents

1. [First time only](#1--first-time-only) — shell, environment, datasets, base checkpoints
2. [Every run](#2--every-run) — pull, clean, launch, watch
3. [Getting the results back](#3--getting-the-results-back) — download, check, figures, record
4. [When it breaks](#4--when-it-breaks) — the failure cheat sheet
5. [Reference](#5--reference) — storage, clusters, cost

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

Raw CSVs go to **project storage**, not `$VSC_DATA`. From your laptop, in PowerShell, at the
repository root:

```powershell
scp -r data\raw\pd data\raw\lgd vsc38338@login.hpc.kuleuven.be:/lustre1/project/stg_00211/CreditPFN/data/raw/
```

Layout the pipeline expects: `data/raw/pd/<id>.csv` and `data/raw/lgd/<id>.csv`.

### 1.4 Stage the base checkpoints

**From a login node** — compute nodes have no outbound internet, and every loader passes
`allow_auto_download=False` so a missing file fails loudly at trial start instead of downloading
mid-job on a GPU you are paying for.

```bash
python -m src.utils.stage_checkpoints --download
```

It reads the base ladder from `config/train.yaml`, reports what is present and what is missing,
and fetches whatever has a public source. TabPFN v2.6 and v3 have none — copy those in by hand
with the same `scp` as §1.3. Adding a base to the config is enough for this to know about it.

---

## 2 · Every run

Two launchers. **`run_experiment.sh`** submits one experiment's grid over every dataset
split — what you usually want. **`run_full_pipeline.sh`** fires a single config's
data → train → eval once and self-sequences it (see the end of this section).

```bash
# 1. Get this run's code onto the cluster — it pulls origin/main, so a commit you have not
#    PUSHED does not exist here.
cd $VSC_DATA/CreditPFN && git pull origin main && git submodule update --init

# 2. See what the previous run left behind (deletes nothing), then wipe it. Keeps data/raw,
#    the base checkpoints, and data/processed. Add --processed only if sanitising changed.
python -m src.utils.clean_run
python -m src.utils.clean_run --clean

# 3. Launch. Export the workers knob FIRST (it defaults to serial) and run DETACHED: the
#    submit loop sleeps while it waits for queue room under the 500-task cap, so a foreground
#    shell that closes would kill it. STAGES="train eval" chains eval after each split's
#    training (same-cluster afterany); the default is train only.
export CREDITPFN_DATALOADER_WORKERS=-1
nohup bash -c 'STAGES="train eval" bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml \
            && STAGES="train eval" bash scripts/slurm/run_experiment.sh config/experiment1_lgd.yaml' \
      > ~/pfn_submit.log 2>&1 &

# 4. Watch. Training AND eval run on Mindwell now; only the data stage is on wICE.
squeue -M mindwell -u $USER      # training + eval (gpu_b200)
tail -f ~/pfn_submit.log         # the submit loop's own progress
```

`run_experiment.sh` assumes the processed corpus exists; a missing `<id>.sanitized.csv` is
sanitised on the fly by the training job (`train_pipeline.py`'s auto-process hook), or run
the data stage yourself with `run_full_pipeline.sh` or `sbatch scripts/slurm/data.slurm`.

Useful variants:

```bash
STAGES=eval bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml   # score an already-trained experiment
SPLITS=4    bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml   # only the first 4 splits
DRY=1       bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml   # print the sbatch lines, submit nothing
```

Cancel this project's jobs on both controllers (leaves any other project's jobs alone):

```bash
for M in mindwell wice; do for J in creditpfn-pd-train creditpfn-lgd-train creditpfn-pd-eval creditpfn-lgd-eval
  do scancel -M $M -u $USER --name=$J; done; done
```

**Why the launcher is not one `--dependency` chain.** VSC runs wICE and Mindwell as
separate Slurm controllers, and a cross-cluster dependency does not work.
`run_full_pipeline.sh` therefore sequences its data → train → eval across clusters by
**sentinel files** on `$VSC_DATA` (visible everywhere) rather than `afterok`;
`run_experiment.sh`, whose train and eval both land on Mindwell, chains eval after training
with an ordinary same-cluster `afterany`. Do not "simplify" either into a cross-cluster
dependency.

---

## 3 · Getting the results back

A finished run is spread across both storage tiers, and the half you most want to read is
the half you cannot browse. Pull it down **from your laptop**, into the repository, where
the notebooks already look for it.

### 3.1 Download everything — one command

Three trees matter, spread across both tiers:

| From | To (in your local checkout) | What it is | Size |
|---|---|---|---|
| `$VSC_DATA/CreditPFN/output/logs/` | `output/logs/` | one `.log` per task — the only record of what happened | tens of MB |
| `$VSC_DATA/CreditPFN/output/manifests/` | `output/manifests/` | one row per trial, per-epoch CSVs, the resolved configs | a few MB |
| `/lustre1/.../CreditPFN/output/results/` | `output/results/` | per-fold scores, one CSV per model × dataset | tens of MB |

**Run this from the repository root in PowerShell.** It pulls all three, from both tiers, in
one SSH connection — so one authentication prompt, not three:

```powershell
cmd /c "ssh vsc38338@login.hpc.kuleuven.be ""tar czf - -C /data/leuven/383/vsc38338/CreditPFN output/logs output/manifests -C /lustre1/project/stg_00211/CreditPFN output/results"" | tar xzf - -C ."
```

Why it is wrapped in `cmd /c` rather than piped in PowerShell: a PowerShell 5.1 pipeline
decodes bytes as text, which corrupts the gzip stream. Inside `cmd` the pipe is byte-clean.
`tar` ships with Windows 11, and the remote `-C` switches make the archive paths land exactly
as `output/logs`, `output/manifests` and `output/results` — so re-running it just refreshes
in place.

If the nested quoting fights you, three plain `scp` calls do the same job:

```powershell
scp -r vsc38338@login.hpc.kuleuven.be:/data/leuven/383/vsc38338/CreditPFN/output/logs output\
```

```powershell
scp -r vsc38338@login.hpc.kuleuven.be:/data/leuven/383/vsc38338/CreditPFN/output/manifests output\
```

```powershell
scp -r vsc38338@login.hpc.kuleuven.be:/lustre1/project/stg_00211/CreditPFN/output/results output\
```

**Do not download `checkpoints/trained/`.** It is 5–8 GB per sweep and nothing local reads
it: the notebooks work from the manifests and results above. Pull one `.ckpt` by hand only
if you intend to run inference with it.

### 3.2 Check what arrived

```powershell
python -m src.utils.clean_run
```

That lists every local tree with file counts. Then confirm the run is complete rather than
half-transferred — a PowerShell here-string into `python -`, since PowerShell has no heredoc:

```powershell
@'
import pandas as pd
from src.utils.paths import manifests_dir, results_dir
for track in ("pd", "lgd"):
    # training manifests are <run>_<track>.csv (exp1 writes one per split, exp1_sNN_<track>.csv);
    # manifest_<track>.csv is the DATASET manifest, so exclude it.
    files = sorted(p for p in manifests_dir().glob(f"*_{track}.csv") if not p.name.startswith("manifest_"))
    if not files:
        print(f"{track}: no training manifest yet"); continue
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{track}: {len(files)} manifest(s), {len(df)} trial rows", df["status"].value_counts().to_dict())
    print(f"      steps {df['total_optimizer_steps'].min()}-{df['total_optimizer_steps'].max()}")
print("result CSVs:", sum(1 for _ in results_dir().rglob("*.csv")))
'@ | python -
```

A trial count below the grid size, or a `total_optimizer_steps` far under
`train.target_total_steps`, means the run is partial — check the logs before believing any
number from it.

### 3.3 Turn it into figures

```powershell
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
| GPU under-utilised; the log says `num_workers=0` | `CREDITPFN_DATALOADER_WORKERS` was not exported before submit, so the per-step preprocessing runs serial and starves the GPU. It cannot be changed on a running job — cancel, `export CREDITPFN_DATALOADER_WORKERS=-1`, resubmit, and confirm a new log says `num_workers>0`. |
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
nor `scp -p` counts as an access.

### Clusters

| Cluster | Partition | Used for | Why |
|---|---|---|---|
| Genius | — | login + submission only | the only login node |
| Mindwell | `gpu_b200` | **training and eval** (default) | 192 GiB VRAM/GPU allows the large in-context sizes; and 24 B200s gave 15–21 concurrent jobs against wICE's ~0.2, so eval moved here too |
| wICE | `batch` | data stage (CPU) | the only stage that needs no GPU |
| wICE | `gpu_h100`, `gpu_a100` | optional eval / TabICL spill | set `EVAL_CLUSTER=wice` or `TABICL_DEST="wice gpu_a100"` when Mindwell is backed up; 36 GPUs university-wide and heavily contended |

Account: `lp_verbekelab` (verify with `sacctmgr -s show user $USER cluster=mindwell` —
note `sacctmgr` takes no `-M`, unlike `squeue`/`sinfo`/`scancel`).

### Cost

B200 time is charged at **437.5 credits per GPU-minute**. A full run at the current grid is
roughly 120 GPU-hours ≈ 3.1 M credits. Check the balance with `sam-balance` before
launching, and prefer a short `--time` — it costs the same and starts sooner.
