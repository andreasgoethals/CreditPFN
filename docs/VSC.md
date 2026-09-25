# CreditPFN on VSC

Scientific choices live in [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md); completed cluster evidence lives in [AGENTS_MEMORY.md](AGENTS_MEMORY.md). Commands here are Bash on VSC unless labeled PowerShell. Use the `CreditPFN` conda environment and `$VSC_DATA/CreditPFN` checkout. Local tests cannot certify the installed B200 runtime.

## Storage and output

| Tier | Location | Contents |
|---|---|---|
| DATA | `$VSC_DATA/CreditPFN` | Repository, logs, small manifests/plans and workflow state |
| Project | `/lustre1/project/stg_00211/CreditPFN` | Canonical data, weights, detailed measurements, predictions and consolidated tables |
| Mindwell GPFS | `$VSC_SCRATCH_GPFS1/CreditPFN/inputs/<hash>` | Verified immutable working copies of processed/public tables and base weights |
| Node scratch | `$VSC_SCRATCH_NODE` | Transient monitor weights and runtime caches |

Both bytes and inodes matter. The [official KU Leuven storage documentation](https://docs.vscentrum.be/leuven/tier2_hardware/kuleuven_storage.html) describes DATA's 75 GiB default and scratch policy. It does not establish the current quota or backup policy of `stg_00211`; inspect `myquota` and the allocation's limits. Intensive Mindwell reads belong on GPFS; wICE uses Lustre. The explicit GPFS path is used even when staging from a wICE CPU job. See the pinned [VSC snapshot](<../tfm-library/repositories/VSC Documentation.txt>), symbols `KU Leuven storage` and `Transferring data between Lustre and GPFS`.

The user requires `output CreditPFN/`, with the experiment layer below it. The name intentionally differs from the template's generic `output/`; quote paths containing the space:

```text
DATA/CreditPFN/output CreditPFN/
  general/
    logs/                       data preparation and general maintenance
    manifests/scheduler/        shared per-controller capacity pool, cluster locks
    figures/CAPTIONS.md         shared caption index after local analysis
    All_Results.md              shared notebook summaries after local analysis
  experiment0/                  likewise experiment1, experiment2, experiment3
    logs/                       one file per job/attempt, including setup and exit status
    manifests/
      <run>_sNN_<track>.csv      trial outcomes; used by evaluation and resume
      plans/                    immutable code/data/config/environment identities
      resolved/                 effective entry-point configurations
      workflow/                 experiment-0 stage state and audit receipts
      figures/                  local notebook caption metadata
    figures/<notebook>/         local publication PDFs

PROJECT/CreditPFN/
  data/{raw,processed,retention}/
  checkpoints/<base>.ckpt
  checkpoints/trained/<experiment>/<track>/
    <trial>.ckpt                final weights
    <trial>.ckpt.provenance.json
    <trial>.ckpt.resume.pt      latest optimizer/RNG/cursor state while unfinished
  output CreditPFN/<experiment>/
    training/<track>/
      <trial>.csv               epoch objectives, gradients, skips and timings
      <trial>.trajectory.csv    fixed-update credit/non-credit scores and drift
      <trial>.parameters.csv.gz milestone tensor summaries, compressed
      <trial>.resources.csv     periodic device/process counters
    results/<PD|LGD>/<method>/  outer-fold scores, identities and compressed predictions
    evaluation_cache/          reusable untuned/classical controls and predictions
    consolidated/<run>/        LATEST + immutable compressed analysis snapshot
```

Locally both tiers collapse into one `output CreditPFN/` tree. Keep `CREDITPFN_OUTPUT_ROOT` and `CREDITPFN_STAGING_ROOT` pointing to the enclosing **CreditPFN project roots**, not `output CreditPFN/`. Weights retain the template's explicit `checkpoints/` exception. This experiment layer and project-tier training diagnostics are intentional template extensions.

Fresh `cpt_*_v5` plans are independent of v4 records and weights. Both tiers create `output CreditPFN/` automatically. Relative legacy `output/...` settings resolve to the canonical name rather than creating another tree. Keep any historical copy only if wanted; a fresh run does not require it. Downloads stay where the user put them. No notebook logs or notebook locks are created.

Experiment 0 currently uses `cpt_*_v5_check2` identities after the GPU-counter repair. The preceding v5 null weights passed exact model/monitor checks, but their resource measurements failed. Preserve that evidence separately; changed source must not overwrite its immutable plans or reuse those measurements as successful samples. The research grids and budgets are unchanged.

Only final weights and the most recent recovery state persist. Intermediate milestone weights are transient; numeric measurements persist. After final publication (including numerical divergence), recovery state is removed. Known repetitive deprecation messages are filtered and repeated numerical errors are counted with bounded log summaries; fatal errors remain visible. Do not hide errors to make a run look successful.

A clean experiment 1 means an empty `experiment1/`, not erasing experiment 0. Keep the accepted workflow receipts, plans, audits and numerical trajectories/resources/parameter summaries: they establish the controls and justify the selected budget. After both experiment-0 parts are accepted, obsolete failed-attempt logs and null/recovery-only weights can be considered for removal using exact inventories, with writers stopped. Preserve the long-pilot scientific evidence. **Do not use `clean --clean` merely to start experiment 1:** it clears every experiment and trained-weight tree. Tiny general scheduler state belongs to the shared launcher and should not be manually reset while jobs are active.

For read-only project-output inspection, run each separately:

```bash
du -h --max-depth=2 /lustre1/project/stg_00211/CreditPFN/output "/lustre1/project/stg_00211/CreditPFN/output CreditPFN"
```

```bash
find /lustre1/project/stg_00211/CreditPFN -maxdepth 2 -type d
```

A missing legacy `output/` is expected after the rename. These totals exclude weights; inspect `checkpoints/trained/` separately when assessing space. ZIP downloads of logs/manifests can be inspected directly, without extracting or copying their contents into the repository.

## One-command experiment 0

Commit and push the reviewed changes locally yourself, then pull on VSC. The agent never pushes. With CreditPFN writers stopped, run each command separately:

```bash
cd "$VSC_DATA/CreditPFN"
```

```bash
git pull --ff-only
```

```bash
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
```

```bash
conda activate CreditPFN
```

```bash
command -v python
```

Confirm this points to `miniconda3/envs/CreditPFN/bin/python`. The job activator also removes an inherited virtualenv that would shadow conda, verifies imports, and fails rather than using a different environment. Use explicit interactive activation: sourcing the complete job activator interactively previously stalled.

For a new GPU environment or a resource-sampling failure, verify counters with one allocation capped at three minutes. Do not repeat this after successful training resource audits. It loads no model or dataset and creates only the normal experiment-0 maintenance log:

```bash
CREDITPFN_CONFIG= CREDITPFN_EXPERIMENT=experiment0 sbatch --clusters=mindwell --partition=gpu_b200 --gpus-per-node=1 --time=00:03:00 scripts/slurm/maintenance.slurm preflight --gpu-resources
```

Mindwell requires `--gpus-per-node`; its submission plugin rejects `--gpus` before creating a job. Read `output CreditPFN/experiment0/logs/maintenance_<JOBID>_r0.log`. Require `gpu_status: sampled` in the JSON and `END exit_code=0`; idle utilization may correctly be zero. On failure, inspect `gpu_error`/`gpu_exit_code` before allocating training jobs. The sampler uses the allocated GPU's UUID with NVIDIA's prefix, logs a bounded failure message once per trial, and leaves unsupported counters empty. After the standalone check passes:

```bash
bash scripts/slurm/run_experiment0.sh part1
```

This is the only experiment-0 part-1 launch command. It downloads/reuses the two packaged and eight pinned public non-credit datasets on the network-enabled login node. No installation occurs. Compute nodes never download. A short wICE CPU job checks configuration, prepares plans and stages inputs. It then releases, in order:

1. **16 null controls**, 2 successful zero-LR updates each; initially 30-minute requests.
2. CPU audit: exact upstream-loaded weights/inference buffers, fixed-monitor parity, parameter records and successful GPU resource measurements.
3. **32 short pilots**, 250 updates; initially one-hour requests.
4. CPU audit: complete identities, budgets, trajectories and no divergence.
5. **Eight recovery pairs**, four bases × two tasks: 12 uninterrupted updates versus stop at 5 and resume to 12; eight one-hour GPU allocations. Each also runs five-fold scoring on its small packaged non-credit table, checking metrics, requested quantiles and complete row-prediction output through the final evaluation code.
6. CPU audit and `output CreditPFN/experiment0/manifests/workflow/part1_passed.json`.

GPU completion callbacks update a locked DATA ledger. The last successful task submits a short CPU audit; that audit alone releases the next stage. Stages advance without a polling allocation and without assuming cross-controller dependency support. Submission can still pause if the user submit quota is full; no GPU allocation waits for another stage. Incomplete/failed tasks stop progression. A unique submission claim prevents accidentally launching the same part twice. Do not delete claims or pool state while jobs run; inspect failed/uncertain submission logs before arranging recovery.

Each ledger folder contains `state.json` and structured audit reports. Job IDs appear in the preparation/audit job logs. Read `output CreditPFN/experiment0/logs/`; no files need downloading during debugging. Check the queue separately:

```bash
squeue -M mindwell,wice -u "$USER" -o "%.18i %.28j %.10T %.12M %R"
```

A queued job is not a hung shell. If submission fails to return within about 45 seconds, do not blindly repeat it: acceptance may be uncertain. Inspect queue/accounting first. Do not cancel unrelated CreditICL/TabPFNCredit jobs.

Part 1 never launches the large sweep. After reviewing its timing/diagnostics, launch the **eight longer budget pilots** separately:

```bash
bash scripts/slurm/run_experiment0.sh part2
```

Part 2 checks the current part-1 receipt and unchanged prepared input/environment identities. Its 20k-update runs use two-hour work segments plus ten minutes for monitoring/checkpoint publication. They automatically requeue up to the configured limit (default 20); reaching that limit leaves a resumable checkpoint for inspection, not a claimed completion. `BUDGET_SEGMENT_MINUTES` may change segment length after reviewing pilot costs. `NULL_WALLTIME` and `PILOT_WALLTIME` override the initial part-1 allocations.

The long pilots measure 0/250/1k/2.5k/5k/10k/20k updates. Their cosine schedule spans 20k: a point at 5k is not a 5k-schedule experiment. Inspect curves and cost, then choose and freeze the research horizon. 5k remains provisional.

## Identity, recovery and resource choices

Training plans fingerprint scientific settings, code, package versions, base bytes, credit inputs and the public monitoring panel. Incompatible completed checkpoints are rejected. Evaluation has its own fingerprints, so plot/benchmark orchestration changes need not alter training identity. Never bypass a mismatch; use a new run identity for changed training semantics.

Null audits compare canonical upstream-loaded tensors because TabPFN v2 may convert serialized key names and v3 may construct inference criterion borders. All inference tensors/borders must match. The accumulated diagnostic `criterion.losses_per_bucket` affects neither returned loss nor predictions and is excluded from the inference-state gate. Recovery audits report its delta separately, together with complete saved tensor-state equality, and gate on the remaining tensors plus fixed monitor trajectories with explicit tolerances. Tolerant equality does not prove bitwise CUDA determinism.

If recovery fails, first inspect the existing pairs on a CPU node, substituting the identifier from the workflow folder. This creates only the normal maintenance log and never trains, changes a ledger or grants a passing receipt:

```bash
CREDITPFN_CONFIG= CREDITPFN_EXPERIMENT=experiment0 sbatch --time=00:10:00 scripts/slurm/maintenance.slurm inspect-recovery --id WORKFLOW_ID
```

The log separates the largest non-diagnostic state difference from `criterion.losses_per_bucket`, checks selected checkpoint identities, and reports monitor differences at updates 0, 5 and 12. Exit 0 means inspection completed; check the comparison fields for failures. A difference already at update 5 predates the pause and cannot be attributed solely to resuming. Preserve successful null/pilot trials and investigate independent-run repeatability before allocating another full recovery stage. Do not weaken tolerances or delete immutable plans to get past this gate.

To isolate GPU arithmetic after a pre-pause mismatch, use one small model/batch diagnostic:

```bash
CREDITPFN_CONFIG= CREDITPFN_EXPERIMENT=experiment0 sbatch --clusters=mindwell --partition=gpu_b200 --gpus-per-node=1 --time=00:05:00 scripts/slurm/maintenance.slurm probe-repeatability --track pd --base v2
```

It uses 512 synthetic rows, 16 features and the configured member count. Each of the three profiles repeats the forward/backward calculation three times, restoring weights, criterion buffers and Python/NumPy/Torch/CUDA RNG each time. No optimizer step or checkpoint write occurs. The profiles are default kernels, deterministic kernels, and deterministic math attention without TF32; BF16 autocast remains enabled. `CUBLAS_WORKSPACE_CONFIG` is fixed before CUDA initialization and printed in the report, so this is a controlled diagnostic, not an exact recreation of an unspecified production environment.

Read the experiment-0 maintenance log's `profiles`: `repeatable: false` demonstrates variation in that calculation; `status: error` can identify a kernel lacking a deterministic implementation. Exit 0 means at least one profile was measured, not that recovery passed. Even three repeatable profiles on this small batch do not certify all production shapes, preprocessing, monitoring or resume behavior. No workflow receipt is changed. `--base` also accepts `v2.6`, `v3` and `tabicl`, and `--track lgd` selects regression; widen the diagnostic only when the first result warrants it.

Each optimizer-boundary recovery saves weights, optimizer, scheduler, scaler, Python/NumPy/Torch/CUDA RNG, sampler cursor and accumulated history. Restoring preserves the full-budget learning-rate schedule. Slurm's warning reaches Python; it saves and exits 75. Abrupt node failure can lose work since the last periodic checkpoint. One trial/task prevents rerunning packed siblings.

Measured two-member training caps remain **v2 10k, v2.6 11k, v3 26k, TabICLv2 26k**. Keep them until a new probe and actual training show enough margin; frozen weights do not imply cheap activations. Four data workers are the initial compromise. Compare 0/4/8 only if recorded data waiting or CPU pressure warrants it. GPU samples and internal training time are not billed allocation time: inspect `sacct` as well.

`GLOBAL_CONCURRENCY` defaults to 16 per controller for jobs submitted through the bounded launcher. `THROTTLE` defaults to four per array. This does not cap unrelated projects or combine both controllers. Shared lane reservations can leave capacity idle behind stragglers; inspect throughput before expanding them. The 450 queued-task headroom is provisional against historical limits; verify live QOS. Shorter realistic requests improve [backfill opportunities](https://slurm.schedmd.com/sched_config.html), not guaranteed priority.

## Experiments 1–3 and evaluation

After part 2 and the horizon decision, prepare immutable plans on CPU nodes for `config/experiment{1,2,3}/{pd,lgd}.yaml`. Keep the chosen budget/milestones consistent. Counts are **512 main, 32 additional seed, 96 sampling**. Experiment 0 adds 72 training arms across both parts, including the 16 small recovery arms. Never count a requeued segment as another scientific trial.

Example main-PD preparation:

```bash
sbatch scripts/slurm/maintenance.slurm prepare --config config/experiment1/pd.yaml --write
```

After successful preparation, preview a short resumable launch:

```bash
DRY=1 SEGMENT_MINUTES=90 CREDITPFN_AUTO_REQUEUE=1 bash scripts/slurm/run_experiment.sh config/experiment1/pd.yaml
```

Submit without `DRY=1` only after reviewing the plans and pilot costs. Repeat for LGD. The main/seed/sampling launches remain deliberate user actions; experiment 0 never releases them automatically. Large submission loops may wait for queue room; use a persistent terminal session.

Final scoring uses five outer folds, an inner validation split for baseline HPO/F1 thresholds/calibration, native context caps and full outer test folds. Credit and non-credit domains are tagged and analyzed separately. Reusable untuned/classical metrics and predictions must match every evaluation fingerprint.

```bash
STAGES=eval bash scripts/slurm/run_experiment.sh config/experiment1/pd.yaml
```

```bash
STAGES=eval EVAL_KIND=classical bash scripts/slurm/run_experiment.sh config/experiment1/pd.yaml
```

Repeat for LGD and completed experiments 2/3. GPU foundation scoring and CPU classical HPO are separate allocations. Start foundation evaluation only against a stable trained roster. Defaults are two GPU/four CPU hours; profile actual packed tasks. Successful cells survive resubmission. Predictions use parquet when available and gzip CSV otherwise; no installation is required for the fallback.

With a run's writers stopped, consolidate it:

```bash
sbatch scripts/slurm/maintenance.slurm consolidate --run cpt_main_v5 --apply
```

Consolidation verifies input/readback checksums and atomically publishes one snapshot: attempts, latest trials, training curves, parameter summaries, resources and evaluation tables for each task. Keep tensor/resource tables separate to avoid a large sparse join. Raw predictions and weights remain in place. Do not accumulate unlimited snapshots; a previous snapshot with missing raw sources cannot be overwritten by a partial reconstruction.

At final analysis, download the contents of **both** cluster `output CreditPFN/` folders into the same local `output CreditPFN/`. Project storage can be browsed/downloaded directly through the same SFTP session; no intermediate DATA copy is needed. Analysis-only compact snapshots must include `LATEST.json` and its referenced directory. Keep datasets/private display mappings locally for corpus notebooks; keep final weights on project storage unless needed elsewhere. A supplied download can be read without moving it using `CREDITPFN_ANALYSIS_ROOT`, pointing to the downloaded output directory containing the experiment folders.

## Cleanup and failure inspection

Historical output is not required for the new run. The old `output CreditPFN/` tree is no longer active; removing it is a separate deliberate action. `python -m src.utils.clean_run` previews the **new** output trees and trained weights on both tiers. `maintenance.slurm clean --clean` deletes that whole campaign, preserving the current maintenance log and all original data/base weights. Stop writers and inspect the preview first. Never use full cleanup midway through a campaign; it also deletes plans, recovery state and submission state. No notebook locks exist.

- Identity mismatch: reconcile code, config, inputs and environment; do not disable fingerprinting.
- Workflow failure: inspect its state/audit and the named logs; later stages were not released.
- Exit 75: saved interruption, not success; inspect requeue count and resume the same identity.
- Numerical divergence: retain the outcome; do not repeat until a favorable result appears.
- `pending_submission`: acceptance is uncertain; reconcile the queue with `output CreditPFN/general/manifests/scheduler/` before retrying.
- Unwritable project storage: fix the mount/permissions; modern jobs refuse large-file fallback to DATA.
- Missing scratch: restage verified canonical inputs on a CPU node.
- OOM or kernel error: retain the first diagnostic and run a separate named probe; never silently change row caps.
- Quota pressure: inspect bytes and inodes, finish/stop writers, consolidate and retire only chosen artifacts.

Record actual cluster outcomes in agent memory. Source review, synthetic CPU training and notebook execution are validation evidence, not proof that the next GPU allocation will pass.
