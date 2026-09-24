# CreditPFN on VSC

The scientific design is in [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md). This runbook covers storage, download, fresh starts, launch and recovery. Use the **CreditPFN** conda environment and **$VSC_DATA/CreditPFN** repository. Commands below are Bash on VSC unless labeled PowerShell. The agent has not submitted any cluster jobs.

## Storage

| Tier | Location | Role |
|---|---|---|
| DATA | `$VSC_DATA/CreditPFN` | Repository, live logs/CSV shards, immutable plans and small summaries |
| Project | `/lustre1/project/stg_00211/CreditPFN` | Canonical data, original/current weights, recovery states, evaluation results and compact tables |
| Mindwell GPFS | `$VSC_SCRATCH_GPFS1/CreditPFN/inputs/<hash>` | Verified working copies of processed tables and original weights |
| Node scratch | `$VSC_SCRATCH_NODE` | Transient monitoring weights and job-local caches |

Both bytes and inodes matter. The [official storage documentation](https://docs.vscentrum.be/leuven/tier2_hardware/kuleuven_storage.html) describes DATA's 75 GiB default, scratch policies and cluster-local I/O. It does **not** establish the current quota or backup policy of allocation `stg_00211`. Check `myquota` and the allocation's actual limits. Persistent storage is not automatically a backup; scratch is purged.

Intensive Mindwell I/O belongs on GPFS; wICE uses Lustre. `$VSC_SCRATCH` changes with the compute cluster. Use the explicit GPFS variable when staging from a wICE CPU node for Mindwell. The pinned [VSC documentation](<../tfm-library/repositories/VSC Documentation.txt>) covers `KU Leuven storage` and `Transferring data between Lustre and GPFS`.

Confirmed cluster jobs and the next outstanding gate are recorded in [AGENTS_MEMORY.md](AGENTS_MEMORY.md). Local historical measurements in `archive/` are not inputs to the fresh run.

Logs caused most of the byte pressure. Three sampled large local logs each contained 53,130 copies of the same scikit-learn deprecation warning, plus thousands of nonfinite-loss warnings. The known deprecation is filtered in the parent and spawned workers. Numerical warnings retain their first diagnostic, logarithmically spaced count summaries and final segment totals; exact skip totals remain in epoch records. Other warnings, fatal errors and tracebacks remain visible. Thread pools are capped and per-step logging is less frequent. Concurrent jobs retain independent shards; consolidation happens after writers stop. The modern launcher refuses to fall back to DATA for large weights when project storage is unwritable.

### Rename an existing output tree

`output CreditPFN/` is an explicit project override of the generic template's `output/` name. Keep the `CREDITPFN_OUTPUT_ROOT` and `CREDITPFN_STAGING_ROOT` variables pointing at the **CreditPFN project roots**, not at this subdirectory. Model weights remain in `checkpoints/`, on project storage on VSC.

With CreditPFN writers stopped, commit/push locally and pull on VSC. Git moves the tracked summary files; ignored logs/manifests/results need migration on both tiers. Preview the merge:

```bash
sbatch --time=00:10:00 scripts/slurm/maintenance.slurm migrate-output
```

After the preview succeeds, apply it:

```bash
sbatch --time=00:10:00 scripts/slurm/maintenance.slurm migrate-output --apply
```

Read each maintenance log under `output CreditPFN/logs/` and wait for exit 0 before the next step. The migration moves within each tier, checks identical duplicates before removing them, and refuses different contents sharing a destination. It is safe to rerun after interruption. Data, weights and Downloads are outside its scope. Legacy relative configuration paths beginning with `output/` resolve to the new name; explicit absolute paths retain their meaning and must be updated by their owner.

### Identity and null-audit semantics

Training identities cover training/data sources, shared numerical metrics and runtime dependencies. Evaluation code, consolidation, plotting and other maintenance changes do not invalidate training. Changes to actual training logic or scientific settings still require a new named plan. Evaluation checks the original plan against each trained checkpoint's provenance and uses a separate code/environment fingerprint, including classical learners and HPO, to invalidate stale cached scores. A dry submission reports a failed plan check even while showing the proposed submission shape.

The zero-LR audit loads both TabPFN checkpoints through the installed upstream loader before comparing tensors. Legacy v2 serialized names convert to a newer architecture; v3 regression borders may originate in the model instead of a separate criterion. All loaded model tensors and inference criterion buffers, including borders, must match exactly. Only `criterion.losses_per_bucket`, an accumulated training-loss diagnostic, is excluded. The per-dataset monitoring comparison remains required. Format conversion alone is not evidence of changed weights.

### Output layout

```text
DATA/CreditPFN/output CreditPFN/
  logs/*.log                          cluster and retained debugging logs
  manifests/<run>_sNN_<track>.csv       attempt records, retained for eval/resume
  manifests/plans/<run>_<track>.json   immutable identities and partitions
  manifests/resolved/                 per-entry-point configurations
  manifests/scheduler/                concurrency-pool records and locks
  manifests/figures/<notebook>.json    caption metadata, separate from PDFs
  figures/<notebook>/*.pdf            publication figures only
  manifests/epochs/<track>/
    <trial>.csv                        epoch diagnostics, updates and timing
    <trial>.trajectory.csv             fixed-update per-dataset monitors

PROJECT/CreditPFN/
  data/{raw,processed}/
  checkpoints/<original>.ckpt
  checkpoints/trained/<track>/
    <trial>.ckpt
    <trial>.ckpt.provenance.json       completion marker and effective identity
    <trial>.ckpt.resume.pt             latest optimizer/RNG/cursor state
  output CreditPFN/results/<PD|LGD>/<method>/    final fold metrics and identity sidecars
  output CreditPFN/evaluation_cache/             reusable controls
  output CreditPFN/consolidated/<run>/
    LATEST.json
    <timestamp-id>/
      inventory.json
      attempts_{pd,lgd}.csv.gz         all attempts, including failures
      trials_{pd,lgd}.csv.gz           latest recorded outcome and attempt count
      training_{pd,lgd}.csv.gz         epochs/trajectories via record_type
      eval_{pd,lgd}.csv.gz             row-fold metrics
```

Local paths default to the repository. Consolidation writes eight compressed CSVs plus inventory into a new immutable snapshot and atomically publishes LATEST. It verifies source/readback checksums. Do not accumulate unbounded snapshots. After removing raw histories, restore them from a saved copy before reconsolidating that run; incomplete replacement is deliberately refused. A clean new run needs none of the old snapshots.

During debugging, keep cluster output on VSC and share the relevant log text. Files downloaded for inspection stay where the user put them; do not import them into local `output CreditPFN/` or create extra inspection manifests. The final campaign download combines DATA's logs/manifests and project's results/consolidated tables under local `output CreditPFN/`. Downloading only DATA's output folder does not include project results. Notebook execution is a local analysis step: stdout is already in `.ipynb`, and `All_Results.md` reads the final summary cell. No notebook logs or locks are generated. The small `manifests/figures/*.json` files record PDF captions and order, as required by FigureSaver; scheduler locks exist only for cluster submission coordination.

All Slurm jobs use `output CreditPFN/logs/<task>_<job-id>_r<restart>.log` (for example `maintenance_62111400_r0.log`); environment activation errors and final exit status go into that same file. Direct `sbatch` works even when `output CreditPFN/logs/` did not exist at submission: the batch shell creates the directory before opening its log. A maintenance cleanup preserves its own active log. The retired default-grid launcher and root-level completion markers are no longer used; use `run_experiment.sh` with an explicit phase config. `checkpoints/` and `data/processed/` are explicit template extensions, not misplaced logs/results.

Only final weights and the latest recovery state persist. Recovery is removed after final publication, including a terminal divergent outcome. Intermediate trajectory weights are transient. Corpus schema/count inspection is cached per unchanged file within each process, and plan generation reuses resolved partitions across recipes. New configs disable raw prediction arrays while keeping computed metrics/calibration diagnostics. Enable predictions only for a separately named diagnostic evaluation with its own storage budget.

## Update and inspect

Commit and push locally yourself, then pull on VSC. Preserve the working environment until its null controls pass; do not upgrade a floating upstream branch during a prepared campaign. Dependencies are in `pyproject.toml`; training also relies on the compatibility shims and TabICL finetuning extras.

```bash
cd "$VSC_DATA/CreditPFN"
git pull --ff-only
git submodule update --init --recursive
export CREDITPFN_USE_SCRATCH=0
source scripts/slurm/_activate_env.sh
export CREDITPFN_OUTPUT_ROOT="$VSC_DATA/CreditPFN"
export CREDITPFN_STAGING_ROOT="/lustre1/project/stg_00211/CreditPFN"
unset CREDITPFN_DATA_ROOT CREDITPFN_BASE_CACHE_ROOT CREDITPFN_USE_SCRATCH
myquota
sam-balance
squeue -M mindwell,wice -u "$USER"
python -m src.utils.stage_checkpoints
python -m src.utils.prepare_experiment --config config/experiment1/pd.yaml
python -m src.utils.prepare_experiment --config config/experiment1/lgd.yaml
```

Preview should show **256 trials per track**, four folds, with all 25 registered tables across tracks. The base-file-size estimate of final weights excludes serialization overhead and simultaneous recovery states. All main, seed and sampling-study weights must fit the actual project quota. An active Python virtualenv can override conda: deactivate it first and inspect each job's printed environment.

Heavy copying, hashing, compression, data preparation and CPU baseline HPO belong on compute nodes. The previews above perform no training.

## Historical output and full resets

**The new run needs no old output or trained checkpoints.** A small local historical copy is useful only for explaining earlier results and failures. There is no requirement to keep that copy on VSC or preserve invalid trained models. The local September archive contains merged measurements, compressed original small records/configuration, and bounded log excerpts/counts, about 66 MB altogether. It contains the available local DATA snapshot and the downloaded project output, not a guaranteed last-minute copy of every cluster manifest. Old trained weights need not be downloaded. Deleting them removes the ability to generate new predictions from those models.

The September archive is already organized; its README and inventory describe that one-off consolidation. Leave it separate from active output. No new downloads, archive tools or copying through DATA are needed during debugging. For a future reset, verify any historical copy the user wants before deleting its originals. The ignored `archive/` folder survives the cleaner. Final analysis downloads are described below.

On **VSC**, with all training/evaluation/submission writers stopped, inspect the existing cleaner's preview:

```bash
cd "$VSC_DATA/CreditPFN"
export CREDITPFN_USE_SCRATCH=0
source scripts/slurm/_activate_env.sh
export CREDITPFN_OUTPUT_ROOT="$VSC_DATA/CreditPFN"
export CREDITPFN_STAGING_ROOT="/lustre1/project/stg_00211/CreditPFN"
squeue -M mindwell,wice -u "$USER"
python -m src.utils.clean_run
```

After verifying the wanted local copy and those target paths, this **deletes all previous output and trained weights on both tiers**:

```bash
sbatch --time=00:10:00 scripts/slurm/maintenance.slurm clean --clean
```

This clears old logs, manifests/plans, results, compact snapshots, evaluation caches, trained weights/recovery states and old submission state, while preserving the cleanup job's active log. It preserves raw data, processed tables and original base weights. Do not add `--processed` for a model-only restart. Do not use full cleanup once new work has started: it is deliberately a complete reset, not a per-run selector. The utility refuses symlinks/junctions and validates every target tree before deletion.

The new output structure is created by the jobs. Stage inputs and prepare new plans only after cleanup finishes. Keep the local legacy folder outside active `output CreditPFN/` so notebooks show the new experiment alone.

## Stage inputs, null controls and pilots

Reuse processed tables if cleaning/schema is unchanged. If preparation changes, rebuild deliberately on a CPU node before making plans. A checksum establishes content identity, not scientific appropriateness.

```bash
sbatch scripts/slurm/maintenance.slurm stage --write
```

Wait for success. Staging copies 25 processed tables and eight bases to an immutable GPFS cache and publishes `ACTIVE_INPUTS.json`. Jobs fingerprint the working copies. If scratch is purged, restage from canonical inputs. An intentional wICE GPU spill should instead stage to `$VSC_SCRATCH_LUSTRE1/CreditPFN` and export its `CREDITPFN_INPUT_POINTER`.

Run repository preflight on a CPU node before preparing plans. It uses the actual training grid, verifies all registered datasets, compares train/eval partitions and checks epoch rails and measured caps:

```bash
sbatch --time=00:15:00 scripts/slurm/maintenance.slurm preflight
```

Run commands one at a time and retain the job ID printed by `sbatch`. Submission returns before the job executes; a queued job is not a hung terminal. If a pasted block stalls, interrupt the foreground command with Ctrl+C, then inspect `squeue` and `sacct` before submitting again. Accepted jobs remain submitted. Avoid hiding the submission response in shell command substitution during debugging.

Inspect the log and `sacct` state. Passing preflight does not establish GPU correctness. For an environment/hardware report use `python -m src.utils.cluster_report`; add GPU checks only through `scripts/slurm/cluster_report.slurm` when needed. Capacity probes measure synthetic BF16 forward/backward memory in both full and frozen modes, with the configured member counts and query fraction. They exclude optimizer state, L2-SP anchors and evaluation overhead; retain the measured caps until real pilots establish sufficient margin. Unexpected probe errors fail the job instead of silently changing adaptation mode. The named conda environment must work; jobs no longer fall back to another environment.

Prepare null/pilot plans on VSC, where package/data identities are known:

```bash
for track in pd lgd; do
  for phase in null pilot budget; do
    sbatch scripts/slurm/maintenance.slurm prepare --config "config/experiment0/${phase}_${track}.yaml" --write
  done
done
```

Wait for the plans. They are immutable: changes to training settings, training sources, input bytes or the training environment require a fresh named plan. Evaluation and maintenance fixes have separate identities and do not invalidate training. No input restaging is required for a change confined to output routing or submission. The launcher checks the relevant plan before submitting any jobs, and dry previews report that check too. A full read-only verification, including input hashes, is available on a CPU node:

```bash
sbatch --time=00:15:00 scripts/slurm/maintenance.slurm prepare --config config/experiment0/null_pd.yaml --check
```

Start with the **16 zero-LR controls**:

```bash
DRY=1 WALLTIME=00:30:00 bash scripts/slurm/run_experiment.sh config/experiment0/null_pd.yaml
DRY=1 WALLTIME=00:30:00 bash scripts/slurm/run_experiment.sh config/experiment0/null_lgd.yaml
WALLTIME=00:30:00 bash scripts/slurm/run_experiment.sh config/experiment0/null_pd.yaml
WALLTIME=00:30:00 bash scripts/slurm/run_experiment.sh config/experiment0/null_lgd.yaml
```

The 30-minute null-control allocation is a provisional short request, not a measured runtime guarantee; inspect the first logs before adjusting it. After they finish, these CPU audits must report `passed: true`, equal tensors and equal per-dataset monitors:

```bash
sbatch scripts/slurm/maintenance.slurm audit --config config/experiment0/null_pd.yaml --null
sbatch scripts/slurm/maintenance.slurm audit --config config/experiment0/null_lgd.yaml --null
```

Then run the **32 short pilots**, 250 successful updates, conservative/high LR endpoints, both adaptations:

```bash
bash scripts/slurm/run_experiment.sh config/experiment0/pilot_pd.yaml
bash scripts/slurm/run_experiment.sh config/experiment0/pilot_lgd.yaml
```

Audit the pilot configs after completion. Reports separate training/monitoring time and extrapolate 5k/10k/20k walltimes with margin; also inspect actual GPU peaks and CPU MaxRSS. If preparation is limiting throughput, use maintenance `prepare --profile-workers 0 4 8 --write` with the pilot config, then submit the printed generated YAML paths. All three worker settings across both tracks total 96 short trials. Choose measured throughput that fits memory, not the maximum worker count.

Before bulk automatic requeue, exercise one separately named positive-LR pilot with a short segment or Slurm warning. Verify resumption to the exact budget, unique trajectory points and recovery-file removal. Completed trials are skipped, so a deliberate canary needs its own config/plan name. CPU tests verify uninterrupted/resumed equality with dropout in all three modes; actual CUDA recovery is still a cluster gate.

The **eight long reference pilots**, `config/experiment0/budget_{pd,lgd}.yaml`, cover one full-update reference per base/task through 20k updates. Their 0/250/1k/2.5k/5k/10k/20k measurements show whether 5k truncates substantial behavior. Their schedule has a 20k horizon; early points do not substitute for 5k-schedule results. After the pilot decision, keep final budget/milestones consistent in main, seed and sampling configs before writing their plans. If increasing the budget, also raise `max_epochs_for_step_budget` enough to cover six LGD table visits per epoch (the existing 2,000-epoch rail cannot reach 20k one_sample updates).

## Main grid and recovery

Prepare `config/experiment{1,2,3}/{pd,lgd}.yaml` for the main, seed and sampling studies using maintenance `prepare --write`, after the scientific choices are settled. Main = 512 trials; separate sampling = 96; reference seed check = 32; total = 640, plus 56 null/pilot trials. Seed 43 repeats only the predefined full-update reference from the main grid. Protocol 4 uses disjoint row partitions for full-pass/accumulation; create fresh `cpt_*_v4` plans and do not reuse protocol-3 plans or weights.

After the gates, a typical short-segment submission is:

```bash
export GLOBAL_CONCURRENCY=16
export THROTTLE=4
export TRIALS_PER_TASK=1
export SEGMENT_MINUTES=90
export CREDITPFN_AUTO_REQUEUE=1
DRY=1 bash scripts/slurm/run_experiment.sh config/experiment1/pd.yaml
DRY=1 bash scripts/slurm/run_experiment.sh config/experiment1/lgd.yaml
bash scripts/slurm/run_experiment.sh config/experiment1/pd.yaml
bash scripts/slurm/run_experiment.sh config/experiment1/lgd.yaml
```

Use `tmux`/screen for long submission loops: they can wait for quota room. The 450-task headroom is conservative for the historical 500-task limit; verify current QOS limits. `THROTTLE` applies per array. **GLOBAL_CONCURRENCY bounds arrays submitted through this launcher per controller**, across tracks/phases. It does not cover unrelated projects, direct sbatch or the combined count across controllers.

The pool reserves lanes using `afterany` dependencies on earlier arrays. A straggler can leave reserved lanes idle: measure this tradeoff. Shorter realistic requests improve **backfill opportunities**, not automatic priority; fairshare, GPU availability and other jobs still matter. See [Slurm scheduling](https://slurm.schedmd.com/sched_config.html) and [VSC queue explanations](https://docs.vscentrum.be/compute/jobs/why_doesn_t_my_job_start.html). Ninety minutes is an example to profile, not a proven optimum.

A 90-minute work segment requests 100 minutes, reserving time for monitoring/saving. Slurm warns the batch shell; it forwards the signal to Python. At a completed optimizer boundary, Python saves weights, optimizer, scheduler, scaler, RNG and cursor, then exits 75. With automatic requeue enabled the element requeues, up to `CREDITPFN_MAX_REQUEUES` (default 20). Otherwise resubmit the same config after inspection. Periodic checkpoints limit hard-kill loss; abrupt termination cannot guarantee a new save. See [array/requeue semantics](https://slurm.schedmd.com/job_array.html) and [signals](https://slurm.schedmd.com/sbatch.html).

`SEGMENT_MINUTES=0` disables planned segmentation. `WALLTIME` and `ACC_WALLTIME` then override whole-trial requests; defaults are provisional historical estimates. One trial/task avoids repeating packed siblings on requeue. Measured two-member training caps remain **v2 10k, v2.6 11k, v3 26k, TabICLv2 26k**. Do not silently lower caps for failing recipes or raise them because a backbone is frozen.

Launch experiment 2 (`config/experiment2/{pd,lgd}.yaml`) after matching main references are available. Launch experiment 3 (`config/experiment3/{pd,lgd}.yaml`) as a separate sampling comparison after the main protocol is fixed. It includes accumulation, which is more expensive per update; use its own timing evidence.

## Evaluation and compact results

Audit training coverage before evaluation. A completed descriptive grid can include recorded divergence, but pending/infrastructure failures are not successes. Foundation scoring uses GPUs and classical HPO uses CPUs. Untuned/classical controls are cached by data, weights, settings, code and environment; hits are re-emitted under the current run for correct pairing, not counted as new independent measurements.

```bash
STAGES=eval bash scripts/slurm/run_experiment.sh config/experiment1/pd.yaml
STAGES=eval bash scripts/slurm/run_experiment.sh config/experiment1/lgd.yaml
STAGES=eval EVAL_KIND=classical bash scripts/slurm/run_experiment.sh config/experiment1/pd.yaml
STAGES=eval EVAL_KIND=classical bash scripts/slurm/run_experiment.sh config/experiment1/lgd.yaml
```

Repeat for seed and sampling configs after their training finishes. Classical controls can be computed earlier to populate the cache; they do not require trained weights. Keep HPO budgets and evaluation seed fixed. `EVAL_TASKS` controls cost packing; `EVAL_CONCURRENCY` defaults to four and shares the controller pool. `EVAL_WALLTIME` defaults to two GPU/four CPU hours; profile large-table tasks. Successful cells survive resubmission.

Same-controller `STAGES="train eval"` can chain via `afterany`, so failed training siblings do not suppress scoring of good checkpoints. Mixed-controller combined submissions are refused. Do not start eval-only foundation scoring against a still-changing checkpoint roster: task packing assumes a stable roster.

With phase writers stopped:

```bash
sbatch scripts/slurm/maintenance.slurm consolidate --run cpt_main_v4 --apply
sbatch scripts/slurm/maintenance.slurm consolidate --run cpt_sampling_v4 --apply
sbatch scripts/slurm/maintenance.slurm consolidate --run cpt_seeds_v4 --apply
```

Once the campaign is complete, the user downloads the contents of both cluster output folders into the same local `output CreditPFN/`: DATA supplies logs/manifests; project storage supplies results, evaluation caches and consolidated tables. For analysis alone, the compact `output CreditPFN/consolidated/<run>` directories, including LATEST and the referenced snapshot, are sufficient. Do not import a debug download during an active campaign. Project storage is accessible in the same WinSCP/SFTP session at `/lustre1/project/stg_00211/CreditPFN/output CreditPFN`; no intermediate DATA copy is necessary. Keep final weights on project storage unless needed locally. Each notebook selects its own experiment from its config; do not set a global run filter. In PowerShell, run `.\.venv\Scripts\python.exe -m src.utils.run_notebooks`. Exploration requires local data; training/results plots use compact tables. Keep the private-name mapping with the private-data checkout.

Download completed new results and retain the final new weights needed for the ongoing study. Historical records from previous experiments do not need to occupy VSC storage. Full cleanup deletes every phase, so use it only when deliberately retiring the whole campaign.

## Failure checks

- Identity mismatch: check environment, code, inputs and plan. Do not bypass the guard.
- Exit 75 / INTERRUPTED: resume the same trial; inspect restart count if automatic requeue stopped.
- DIVERGED: report the numerical outcome; do not repeatedly rerun until success.
- Scheduler `pending_submission`: the response was uncertain. Reconcile squeue with `output CreditPFN/manifests/scheduler/pool-<cluster>.json` before another submission. Never delete active pool state.
- Unwritable project storage: fix the mount/permissions; do not fill DATA with weights.
- OOM/cuDNN failure: preserve the log and use a separately named capacity/kernel probe before revising the protocol.
- Import failure: inspect the printed environment and compatibility smoke tests; do not install packages in a GPU job.
- Quota pressure: stop writers, consolidate and download completed results, then remove only records you have chosen to retire. Full cleanup is a complete campaign reset, not a mid-run quota remedy.

After a pilot, use `sacct -M mindwell -j JOBID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode` and the trial logs. Allocation time, CPU memory and internal training time are distinct measurements. Record actual cluster runs in the agent-memory table; local validation cannot substitute for them.


## Notebook organization and read-only inspection

`notebooks/00_general/` describes raw and processed inputs. `experiment0/` covers null controls, short pilots and budget pilots; `experiment1/` separates training and final benchmark for PD/LGD; `experiment2/` compares the extra seed with the main reference; `experiment3/` compares all three sampling modes. The runner discovers subfolders and mirrors notebook-relative paths under `figures/` and `manifests/figures/`.

To inspect a supplied download without moving it, set the analysis-only variable to the downloaded **output directory**, not its parent:

```powershell
$env:CREDITPFN_ANALYSIS_ROOT = Join-Path $env:USERPROFILE 'Downloads/output CreditPFN'
```

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
```

The source folder remains read-only. Generated PDFs, caption metadata and summaries stay in the repository's `output CreditPFN/`; executed notebook outputs stay in the notebooks. Raw/processed corpus exploration still reads the canonical local data. A DATA-only download has no project-tier benchmark results; those sections report unavailable evidence. To return to local campaign input:

```powershell
Remove-Item Env:CREDITPFN_ANALYSIS_ROOT
```

Config relocation alone preserves scientific settings and run names. Existing null controls can still be audited with `config/experiment0/null_{pd,lgd}.yaml`. Do not reprepare their immutable plans merely because the file moved. Subsequent training phases need plans prepared against the source and environment they will actually run.
