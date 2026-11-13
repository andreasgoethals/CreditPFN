# CreditPFN on VSC

Scientific methods belong in [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md), primary sources in
[LITERATURE.md](LITERATURE.md), and dated run evidence in [AGENTS_MEMORY.md](AGENTS_MEMORY.md).
Commands below are Bash on VSC unless labeled PowerShell. The pinned [VSC Library](../vsc-library/README.md)
is read-only; this runbook owns project-specific paths and decisions. The first study's operating
history is in Git history (commit `2c8ef18`).

## Access

Follow [TEMPLATE.md](TEMPLATE.md#vsc-access-windows-and-agents). Authentication is manual:
Andreas runs the following in Windows PowerShell and approves KU Leuven Authenticator when
his existing certificate is missing or expired. Agents never initiate login or schedule reminders.

```powershell
step ssh login --context VSC
```

Reuse the existing Windows SSH agent. Use native `C:\Windows\System32\OpenSSH\ssh.exe`
with `-F none -T`, batch mode, strict known-host checking, bounded timeouts and an LF Bash
script on stdin, with **no trailing remote command**. Begin the remote script with `exec 2>&1`
and `set -eu`, emit start/completion markers and check the real exit status. Empty exit-zero
output does not prove execution. Do not export credentials. Login nodes are for bounded metadata,
scheduler queries and short downloads (`prepare_retention`); hashing, copies and computation run
through Slurm.

## Storage (moved 04-11-2026)

All CreditPFN compute runs on **Mindwell** (GPU `gpu_b200`, CPU `batch_graniterapids`), whose
native filesystem is GPFS (`vsc-library/guides/STORAGE.md`: intensive I/O on the compute
cluster's native filesystem). wICE is no longer used.

| Tier | Location and role |
|---|---|
| DATA | `/data/leuven/383/vsc38338/CreditPFN`: the Git checkout, compact logs, manifests, receipts, the preserved Conda comparator environment (`$VSC_DATA/miniconda3/envs/CreditPFN`) |
| PROJECT | `/gpfs1/project/prj_00013/CreditPFN`: canonical datasets (`data/`), base checkpoints (`checkpoints/`), trained weights, detailed training/results output and the sealed runtimes (`runtimes/`) |
| Mindwell scratch | `$VSC_SCRATCH_GPFS1/CreditPFN`: staged working inputs and runtime copy; temporary |
| Node-local scratch | `$VSC_SCRATCH_NODE/...`: per-job library caches, removed after success |

The move copied `checkpoints`, `data` and `runtimes` from Lustre `stg_00211`, verified every file
and symlink by SHA-256 (receipt `STORAGE_MOVE_20261104.sha256` in the PROJECT root), and then
deleted `/lustre1/project/stg_00211/CreditPFN`. `paths.staging_root()` resolves
`$VSC_PROJECT_GPFS1/prj_00013`; the shell defaults use `/gpfs1/project/prj_00013`. `prj_00013`
also holds Andreas's JEVPFN project: never touch anything outside `CreditPFN/`.
The output folder name is `output CreditPFN/` on both persistent tiers (deliberate deviation).
`data/raw` directories are write-protected on purpose.

## Runtime

`env20261029f` (Python 3.13.11, torch 2.12.0, tabpfn 8.0.8 at Git `27923f4`, tabicl 2.1.1) is the
immutable sealed container in `runtimes/`. Its environment specification is recorded from the
Conda comparator by `runtime_cpu.slurm record` (byte-identical on 04-11-2026, SHA-256 `c9ce4607…`);
the recorded copy was deleted with the DATA output on 05-11-2026, so the chain below records it
again first.
The local `.venv` uses Python 3.13.11 and the same pins (`config/runtime/`, `tests/test_environment_parity.py`).

Container jobs need a runtime acceptance for the current code identity (all of `src/`, see
`src.utils.experiment.code_identity`): `runtime_cpu.slurm prepare` → ten-case `runtime_gpu.slurm`
array (container vs Conda on the same B200, original checkpoints, plus FP32 capacity) →
`runtime_cpu.slurm accept`. Any change under `src/` needs a fresh acceptance.

## Deployment

1. Edit and test locally (`python -m pytest -q`; notebooks when plotting changes).
2. Commit and push. On VSC: `git pull --ff-only` in the clean DATA checkout, verify the commit
   and the submodule pins. Never patch VSC source; never change files used by queued jobs.
3. On the login node, prepare the retention panel and the E4 control twins once (downloads with
   recorded checksums; the twins include 1-2 million-row tables, so allow about 30 minutes):
   `python -m src.utils.prepare_retention --supersede-changed` (in the Conda comparator env).
4. Submit acceptance and the campaign controller as one dependency chain (below).

### Experiment 0

```bash
cd "$VSC_DATA/CreditPFN"
RECORD="$VSC_DATA/CreditPFN/output CreditPFN/experiment0/manifests/runtime_builds/record$(date +%Y%m%d)"
J0=$(sbatch --parsable scripts/slurm/runtime_cpu.slurm record env20261029f "$RECORD")
J1=$(sbatch --parsable --dependency=afterok:${J0%%;*} scripts/slurm/runtime_cpu.slurm prepare env20261029f "$RECORD/environment.json")
J2=$(sbatch --parsable --dependency=afterok:${J1%%;*} --array=0-9 scripts/slurm/runtime_gpu.slurm env20261029f)
J3=$(sbatch --parsable --dependency=afterok:${J2%%;*} scripts/slurm/runtime_cpu.slurm accept env20261029f)
sbatch --parsable --dependency=afterok:${J3%%;*} --export=ALL,CREDITPFN_RUNTIME_VERSION=env20261029f scripts/slurm/campaign_cpu.slurm experiment0
```

`campaign_cpu.slurm experiment0` runs the gate tests (upstream parity with inputs required), the
whole suite for the record, the LGD density audit, writes every pilot plan, then submits all pilots
(`run_experiment.sh`, partition 0 only, no running cap), the reference pilots' five-fold
benchmark and classical baselines, the GPU check arrays (`pilot_gpu.slurm recovery|capacity|precision|evaluation`)
and a CPU summary after everything. Results: DATA/PROJECT `output CreditPFN/experiment0/`,
receipts in `experiment0/manifests/pilots/` (`submission.json`, `summary.json`). The summary also
audits the reference pilots' five-fold benchmark and baselines; `submit E<n>` refuses to start
unless that summary passed.

Inspect the queue:

```bash
squeue -M mindwell -u "$USER" -o "%.20i %.28j %.12T %.12M %.40R"
```

## Campaign submission (one experiment at a time)

Mindwell's default QoS allows **500 queued jobs per user**; every array element counts, other
projects of the same user included. Arrays hold at most 1,001 elements. One pass of all training
experiments would queue about 1,800 jobs (E6 alone 640), so experiments are submitted separately:

```bash
sbatch --export=ALL,CREDITPFN_RUNTIME_VERSION=env20261029f scripts/slurm/campaign_cpu.slurm submit E6
```

The controller prepares the experiment's plans and submits each arm config through
`run_experiment.sh` (training, then that partition's five-fold evaluation with `afterok`, four
evaluation shards per partition). Before every config it counts the user's queued jobs and stops
cleanly when the next config would exceed 480; rerunning the same command continues from the
receipt `output CreditPFN/<slug>/manifests/submission.json`. E1 also queues the classical
baselines once. E5 (fine-tuning) and E11 (context curves) are evaluation-only arrays over
`src.eval.campaign_tasks` and refuse to start until every E1 checkpoint exists; E10's
matched-context evaluation is submitted by rerunning `submit E10` after its training finished.
Campaign jobs request 24 hours for training and evaluation until Experiment 0's audit gives
measured timings; set `CREDITPFN_TRAIN_WALLTIME` / `CREDITPFN_EVAL_WALLTIME` (HH:MM:SS) then.
A launcher that fails or times out is recorded as `uncertain` in the receipt and is never resubmitted
automatically: inspect `squeue`, then delete that entry before rerunning.
Suggested order: E1 → E2, E4, E8–E10, E12, E3 → E6, E7 (largest) → E5, E11.

## Measured caps

| Base | Training episode ceiling (context + query, 2 views) | Used in the campaign |
|---|---:|---:|
| TabPFN v2 | 10,000 | 10,000 |
| TabPFN v2.6 | 11,000 | 10,000 |
| TabPFN v3 / TabICLv2 | 26,000 | 26,000 |

Inference contexts: v2 10,000 (model limit), v2.6 50,000, v3/TabICLv2 up to the full fold
(FP32 capacity is measured in Experiment 0). Do not raise a cap without a new GPU probe.

## Slurm practice (vsc-library)

No `--mem` together with `--cpus-per-task`; evaluation waits for training with `afterok`;
requeues name `--clusters` and are capped (`CREDITPFN_MAX_REQUEUES`, default 20); scratch copies
do not preserve timestamps (runtime cache verifies sizes and SHA-256 instead of mtimes); GPU jobs
contain only GPU work, CPU preparation runs in separate CPU jobs. GPU jobs request 24 cores per
B200 until Experiment 0 measures actual use. Never blanket-delete scratch; per-job caches are
removed by their own job after success.
