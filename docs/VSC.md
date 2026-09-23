# CreditPFN on VSC

The scientific design is in [PAPER_ROADMAP.md](PAPER_ROADMAP.md). This runbook covers storage, archive, launch and recovery. Use the **CreditPFN** conda environment and **$VSC_DATA/CreditPFN** repository. Commands below are Bash on VSC unless labeled PowerShell. The agent has not submitted any cluster jobs.

## Storage and current inventory

| Tier | Location | Role |
|---|---|---|
| DATA | `$VSC_DATA/CreditPFN` | Repository, live logs/CSV shards, immutable plans and small summaries |
| Project | `/lustre1/project/stg_00211/CreditPFN` | Canonical data, original/final weights, recovery states, evaluation results, compact tables and archives |
| Mindwell GPFS | `$VSC_SCRATCH_GPFS1/CreditPFN/inputs/<hash>` | Verified working copies of processed tables and original weights |
| Node scratch | `$VSC_SCRATCH_NODE` | Transient monitoring weights and job-local caches |

Both bytes and inodes matter. The [official storage documentation](https://docs.vscentrum.be/leuven/tier2_hardware/kuleuven_storage.html) describes DATA's 75 GiB default, scratch policies and cluster-local I/O. It does **not** establish the current quota or backup policy of allocation `stg_00211`. Check `myquota` and the allocation's actual limits. Persistent storage is not automatically a backup; scratch is purged.

Intensive Mindwell I/O belongs on GPFS; wICE uses Lustre. `$VSC_SCRATCH` changes with the compute cluster. Use the explicit GPFS variable when staging from a wICE CPU node for Mindwell. The pinned [VSC documentation](<../tfm-library/repositories/VSC Documentation.txt>) covers `KU Leuven storage` and `Transferring data between Lustre and GPFS`.

**User inventory, 22-09-2026:** DATA output 7.1 GB / 4,913 files: logs 7.1 GB, manifests 82 MB. DATA fallback checkpoints 2.6 GB. Project output 7.4 MB; project checkpoints 41 GB / 412 files. File counts are not verified completed trials. Retagging of 338 L2-SP survivors is already done. No jobs were running when reported.

Logs caused most of the byte pressure. Warning filters now run inside spawned workers, thread pools are capped, and per-step logging is less frequent. Concurrent jobs retain independent shards; consolidation happens after writers stop. The modern launcher refuses to fall back to DATA for large weights when project storage is unwritable.

### Output layout

```text
DATA/CreditPFN/output/
  logs/
  manifests/<run>_sNN_<track>.csv       attempt records, retained for eval/resume
  manifests/plans/<run>_<track>.json   immutable identities and partitions
  manifests/resolved/                 per-entry-point configurations
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
  output/results/<PD|LGD>/<method>/    final fold metrics and identity sidecars
  output/evaluation_cache/             reusable controls
  output/consolidated/<run>/
    LATEST.json
    <timestamp-id>/
      inventory.json
      attempts_{pd,lgd}.csv.gz         all attempts, including failures
      trials_{pd,lgd}.csv.gz           latest recorded outcome and attempt count
      training_{pd,lgd}.csv.gz         epochs/trajectories via record_type
      eval_{pd,lgd}.csv.gz             row-fold metrics
  output/archives/
    <run>-evidence-<timestamp-id>.tar.gz
    <run>-evidence-<timestamp-id>.tar.gz.json
```

Local paths default to the repository. Consolidation writes eight compressed CSVs plus inventory into a new immutable snapshot and atomically publishes LATEST. It verifies source/readback checksums. Do not accumulate unbounded snapshots. After pruning raw histories, restore them from the archive before reconsolidating that run; incomplete replacement is deliberately refused.

Only final weights and the latest recovery state persist. Recovery is removed after successful final publication. Intermediate trajectory weights are transient. New configs disable raw prediction arrays while keeping computed metrics/calibration diagnostics. Enable predictions only for a separately named diagnostic evaluation with its own storage budget.

## Update and inspect

Commit and push locally yourself, then pull on VSC. Preserve the working environment until its null controls pass; do not upgrade a floating upstream branch during a prepared campaign. Dependencies are in `pyproject.toml`; training also relies on the compatibility shims and TabICL finetuning extras.

```bash
cd "$VSC_DATA/CreditPFN"
git pull --ff-only
git submodule update --init --recursive
conda activate CreditPFN
export CREDITPFN_OUTPUT_ROOT="$VSC_DATA/CreditPFN"
export CREDITPFN_STAGING_ROOT="/lustre1/project/stg_00211/CreditPFN"
unset CREDITPFN_DATA_ROOT CREDITPFN_BASE_CACHE_ROOT CREDITPFN_USE_SCRATCH
myquota
sam-balance
squeue -M mindwell,wice -u "$USER"
python -m src.utils.stage_checkpoints
python -m src.utils.prepare_experiment --config config/experiment1_pd.yaml
python -m src.utils.prepare_experiment --config config/experiment1_lgd.yaml
```

Preview should show **256 trials per track**, four folds, with all 25 registered tables across tracks. The base-file-size estimate of final weights excludes serialization overhead and simultaneous recovery states. All main, seed and separate-study weights must fit the actual project quota. An active Python virtualenv can override conda: deactivate it first and inspect each job's printed environment.

Heavy copying, hashing, compression, data preparation and CPU baseline HPO belong on compute nodes. The previews above perform no training.

## Archive and retire legacy output

Keep a compact **evidence archive**: logs, raw measurements, resolved configs, provenance, compressed tables and a checksum/size index of old weights. This preserves debugging evidence and explains what old results mean. It cannot repair an invalid run or generate new predictions after its weights are deleted.

The archive contains all logs/resolved snapshots because historical names did not consistently identify their run. Measurements and weight retirement are scoped to the named run. It excludes checkpoint bytes, datasets, original bases and raw prediction arrays. The included docs/configs describe the **archiving checkout**, not necessarily historical training code. Keep both archive and adjacent JSON inventory, preferably with a second copy elsewhere before retirement.

Stop all experiment writers, then preview and create:

```bash
python -m src.utils.archive_experiment --run exp1
sbatch scripts/slurm/maintenance.slurm archive --run exp1 --write --quiescent
```

The CPU job prints the exact verified archive path in `maintenance_<jobid>.log`, outside the tree it archives. Review the counts and index. Set `ARCHIVE` to that exact printed `.tar.gz` path. These commands preview only:

```bash
python -m src.utils.archive_experiment --prune "$ARCHIVE"
python -m src.utils.archive_experiment --retire "$ARCHIVE"
```

After verification and deciding that the indexed old weights are no longer needed:

```bash
sbatch scripts/slurm/maintenance.slurm archive --prune "$ARCHIVE" --apply --quiescent
sbatch scripts/slurm/maintenance.slurm archive --retire "$ARCHIVE" --apply --quiescent
```

Pruning removes unchanged archived logs, resolved snapshots and epoch/trajectory shards. Root attempt manifests, plans and compact results remain. Retirement separately removes only indexed, unchanged checkpoint files within configured trained-weight roots. Both preflight the deletion set; changed sources abort. Other runs and original bases are untouched. **Weights cannot be recovered from this compact archive.**

Repeat for other historical run names if needed. The index distinguishes trained weights from original bases in the DATA checkpoint tree. A corrected rerun needs fresh training/evaluation names, not deletion of validated raw data, processed tables or original weights.

## Stage inputs, null controls and pilots

Reuse processed tables if cleaning/schema is unchanged. If preparation changes, rebuild deliberately on a CPU node before making plans. A checksum establishes content identity, not scientific appropriateness.

```bash
sbatch scripts/slurm/maintenance.slurm stage --write
```

Wait for success. Staging copies 25 processed tables and eight bases to an immutable GPFS cache and publishes `ACTIVE_INPUTS.json`. Jobs fingerprint the working copies. If scratch is purged, restage from canonical inputs. An intentional wICE GPU spill should instead stage to `$VSC_SCRATCH_LUSTRE1/CreditPFN` and export its `CREDITPFN_INPUT_POINTER`.

Prepare null/pilot plans on VSC, where package/data identities are known:

```bash
for track in pd lgd; do
  for phase in experiment0 pilot budget_pilot; do
    sbatch scripts/slurm/maintenance.slurm prepare --config "config/${phase}_${track}.yaml" --write
  done
done
```

Wait for the plans. They are immutable: changed scientific settings/code/data/environment require a fresh named plan, not overwriting an active one. Start with the **16 zero-LR controls**:

```bash
DRY=1 bash scripts/slurm/run_experiment.sh config/experiment0_pd.yaml
DRY=1 bash scripts/slurm/run_experiment.sh config/experiment0_lgd.yaml
bash scripts/slurm/run_experiment.sh config/experiment0_pd.yaml
bash scripts/slurm/run_experiment.sh config/experiment0_lgd.yaml
```

After they finish, these CPU audits must report `passed: true`, equal tensors and equal per-dataset monitors:

```bash
sbatch scripts/slurm/maintenance.slurm audit --config config/experiment0_pd.yaml --null
sbatch scripts/slurm/maintenance.slurm audit --config config/experiment0_lgd.yaml --null
```

Then run the **32 short pilots**, 250 successful updates, conservative/high LR endpoints, both adaptations:

```bash
bash scripts/slurm/run_experiment.sh config/pilot_pd.yaml
bash scripts/slurm/run_experiment.sh config/pilot_lgd.yaml
```

Audit the pilot configs after completion. Reports separate training/monitoring time and extrapolate 5k walltime with margin; also inspect actual GPU peaks and CPU MaxRSS. If preparation is limiting throughput, use maintenance `prepare --profile-workers 0 4 8 --write` with the pilot config, then submit the printed generated YAML paths. All three worker settings across both tracks total 96 short trials. Choose measured throughput that fits memory, not the maximum worker count.

Before bulk automatic requeue, exercise one separately named positive-LR pilot with a short segment or Slurm warning. Verify resumption to the exact budget, unique trajectory points and recovery-file removal. Completed trials are skipped, so a deliberate canary needs its own config/plan name. CPU tests verify uninterrupted/resumed equality with dropout in all three modes; actual CUDA recovery is still a cluster gate.

The **eight long reference pilots**, `budget_pilot_{pd,lgd}.yaml`, cover one full-update reference per base/task through 20k updates. Their 0/250/1k/2.5k/5k/10k/20k measurements show whether 5k truncates substantial behavior. Their schedule has a 20k horizon; early points do not substitute for 5k-schedule results. After the pilot decision, keep final budget/milestones consistent in main, seed and sampling configs before writing their plans.

## Main grid and recovery

Prepare `experiment1`, `seed_check` and `sampling` for both tracks using maintenance `prepare --write`, after the scientific choices are settled. Main = 512 trials; added reference seeds = 128; separate sampling = 96. Seed split indices 0–3 use seed 43; 4–7 use seed 44; dataset partitions match main.

After the gates, a typical short-segment submission is:

```bash
export GLOBAL_CONCURRENCY=16
export THROTTLE=4
export TRIALS_PER_TASK=1
export SEGMENT_MINUTES=90
export CREDITPFN_AUTO_REQUEUE=1
DRY=1 bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml
DRY=1 bash scripts/slurm/run_experiment.sh config/experiment1_lgd.yaml
bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml
bash scripts/slurm/run_experiment.sh config/experiment1_lgd.yaml
```

Use `tmux`/screen for long submission loops: they can wait for quota room. The 450-task headroom is conservative for the historical 500-task limit; verify current QOS limits. `THROTTLE` applies per array. **GLOBAL_CONCURRENCY bounds arrays submitted through this launcher per controller**, across tracks/phases. It does not cover unrelated projects, direct sbatch or the combined count across controllers.

The pool reserves lanes using `afterany` dependencies on earlier arrays. A straggler can leave reserved lanes idle: measure this tradeoff. Shorter realistic requests improve **backfill opportunities**, not automatic priority; fairshare, GPU availability and other jobs still matter. See [Slurm scheduling](https://slurm.schedmd.com/sched_config.html) and [VSC queue explanations](https://docs.vscentrum.be/compute/jobs/why_doesn_t_my_job_start.html). Ninety minutes is an example to profile, not a proven optimum.

A 90-minute work segment requests 100 minutes, reserving time for monitoring/saving. Slurm warns the batch shell; it forwards the signal to Python. At a completed optimizer boundary, Python saves weights, optimizer, scheduler, scaler, RNG and cursor, then exits 75. With automatic requeue enabled the element requeues, up to `CREDITPFN_MAX_REQUEUES` (default 20). Otherwise resubmit the same config after inspection. Periodic checkpoints limit hard-kill loss; abrupt termination cannot guarantee a new save. See [array/requeue semantics](https://slurm.schedmd.com/job_array.html) and [signals](https://slurm.schedmd.com/sbatch.html).

`SEGMENT_MINUTES=0` disables planned segmentation. `WALLTIME` and `ACC_WALLTIME` then override whole-trial requests; defaults are provisional historical estimates. One trial/task avoids repeating packed siblings on requeue. Measured two-member training caps remain **v2 14k, v2.6 11k, v3 26k, TabICLv2 26k**. Do not silently lower caps for failing recipes or raise them because a backbone is frozen.

Launch seed configs after main for simple scheduling/cache reuse, or alongside once the protocol is fixed. Their two references are predetermined, not selected winners. The separate sampling study is more expensive in accumulate mode; use its own timing evidence.

## Evaluation and compact results

Audit training coverage before evaluation. A completed descriptive grid can include recorded divergence, but pending/infrastructure failures are not successes. Foundation scoring uses GPUs and classical HPO uses CPUs. Untuned/classical controls are cached by data, weights, settings, code and environment; hits are re-emitted under the current run for correct pairing, not counted as new independent measurements.

```bash
STAGES=eval bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml
STAGES=eval bash scripts/slurm/run_experiment.sh config/experiment1_lgd.yaml
STAGES=eval EVAL_KIND=classical bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml
STAGES=eval EVAL_KIND=classical bash scripts/slurm/run_experiment.sh config/experiment1_lgd.yaml
```

Repeat for seed/sampling configs after training finishes. Classical controls can be computed earlier to populate the cache; they do not require trained weights. Keep HPO budgets and evaluation seed fixed. `EVAL_TASKS` controls cost packing; `EVAL_CONCURRENCY` defaults to four and shares the controller pool. `EVAL_WALLTIME` defaults to two GPU/four CPU hours; profile large-table tasks. Successful cells survive resubmission.

Same-controller `STAGES="train eval"` can chain via `afterany`, so failed training siblings do not suppress scoring of good checkpoints. Mixed-controller combined submissions are refused. Do not start eval-only foundation scoring against a still-changing checkpoint roster: task packing assumes a stable roster.

With phase writers stopped:

```bash
sbatch scripts/slurm/maintenance.slurm consolidate --run cpt_main_v3 --apply
sbatch scripts/slurm/maintenance.slurm consolidate --run cpt_seeds_v3 --apply
sbatch scripts/slurm/maintenance.slurm consolidate --run cpt_sampling_v3 --apply
```

Download the needed `output/consolidated/<run>` directories, including LATEST and the referenced snapshot, into local `output/consolidated/`. Keep final weights on project storage unless needed locally. In PowerShell, set `$env:CREDITPFN_VIZ_RUN = 'cpt_main_v3'`, then run `.\.venv\Scripts\python.exe -m src.utils.run_notebooks`. Exploration requires local data; training/results plots use compact tables. Keep the private-name mapping with the private-data checkout.

Archive completed new phases similarly, but keep final new weights needed for reproducibility; retirement is optional and separate.

## Failure checks

- Identity mismatch: check environment, code, inputs and plan. Do not bypass the guard.
- Exit 75 / INTERRUPTED: resume the same trial; inspect restart count if automatic requeue stopped.
- DIVERGED: report the numerical outcome; do not repeatedly rerun until success.
- Scheduler `pending_submission`: the response was uncertain. Reconcile squeue with `output/manifests/scheduler/pool-<cluster>.json` before another submission. Never delete active pool state.
- Unwritable project storage: fix the mount/permissions; do not fill DATA with weights.
- OOM/cuDNN failure: preserve the log and use a separately named capacity/kernel probe before revising the protocol.
- Import failure: inspect the printed environment and compatibility smoke tests; do not install packages in a GPU job.
- Quota pressure: stop writers, archive and verify, then prune indexed unchanged shards.

After a pilot, use `sacct -M mindwell -j JOBID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode` and the trial logs. Allocation time, CPU memory and internal training time are distinct measurements. Record actual cluster runs in the agent-memory table; local validation cannot substitute for them.
