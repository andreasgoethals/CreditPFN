# CreditPFN on VSC

Scientific methods belong in [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md), primary sources in
[LITERATURE.md](LITERATURE.md), and dated run evidence in [AGENTS_MEMORY.md](AGENTS_MEMORY.md).
Commands below are Bash on VSC unless labeled PowerShell. The pinned [VSC Library](../vsc-library/README.md)
is read-only; this runbook owns project-specific paths and decisions.

## Access, storage policy and migration status

Follow [TEMPLATE.md](TEMPLATE.md#vsc-access-windows-and-agents). Authentication is manual:
Andreas runs the following in Windows PowerShell and approves KU Leuven Authenticator when
his existing certificate is missing or expired. Agents never initiate login or schedule reminders.

```powershell
step ssh login --context VSC
```

Reuse the existing Windows SSH agent. Use native `C:\Windows\System32\OpenSSH\ssh.exe`
with `-F none -T`, batch mode, strict known-host checking, bounded timeouts and an LF Bash
script on stdin, with **no trailing remote command**. Begin the remote script with `exec 2>&1`
and `set -eu`, emit start/completion markers and check the real exit status. The tested gateway
otherwise omitted stderr. Empty exit-zero output does not prove execution. Do not export credentials.
Login nodes are for bounded metadata and scheduler queries; hashing, inventories and computational
checks run through Slurm.

### Storage ownership

| Tier | CreditPFN location and role |
|---|---|
| DATA | `/data/leuven/383/vsc38338/CreditPFN`: actual Git checkout and `.git`, compact logs, manifests, receipts |
| PROJECT | `/lustre1/project/stg_00211/CreditPFN`: canonical datasets, weights, predictions, detailed training/results and versioned sealed runtimes |
| Mindwell scratch | `$VSC_SCRATCH_GPFS1/CreditPFN`: staged working inputs/runtime; temporary only |
| wICE scratch | `$VSC_SCRATCH_LUSTRE1/CreditPFN`: staged working inputs/runtime; temporary only |
| Node-local scratch | `$VSC_SCRATCH_NODE/CreditPFN/job_cache/<user>/<job>-r<restart>`: owned disposable library caches |

GPFS PROJECT is not assumed to be allocated. Check current quotas before staging more assets.
The output name is `output CreditPFN/` on both persistent tiers, a deliberate template deviation.
The shared publication tree `figures/{rq1,rq2,rq3,rq4,experiment0,general}` and actual ICML figure
widths are the other documented deviations.

`env20261029f` is the immutable PROJECT container rebuilt from the recorded environment:
238 distribution versions and installed Git-source bytes matched at construction. Runtime acceptance
is separately bound to the actual image digest, interpreter, package versions and source identity.
The old DATA Conda environment is retained solely for explicit compatibility comparisons/rollback;
new computational jobs require `CREDITPFN_RUNTIME_VERSION` and cannot silently fall back to it.
Never relocate an installed environment.

### Current campaign state

At the verified 30-10-2026 full audit, all **445 FP32 refresh tasks** completed: 320 benchmark
shards and 125 context tasks. Independent audit `62474641` passed 459 benchmark comparisons,
2,295 folds and 30,967,719 saved predictions, plus 5,550 context cells / 147,493,764 predictions.
The audit verified unchanged benchmark cohorts/targets and complete fixed context-test rows.
These results remain staged until the final publication gate; notebook results are still the prior download.

The original extension gate failed: seven smoke cases passed; four DiagScale cases hit an obsolete
v3-only guard; one LoRA LGD v2.6 case exited 139 without a Python traceback. Its capacity jobs and
dependent controllers were cancelled, and no extension research was released. The guard correction
retains strict projection validation; native fault traces are enabled for the fresh gate. A new source
requires fresh runtime acceptance and all 80 extension checks, not reuse of passing old-source receipts.

The frozen DATA checkout `CreditPFN-study-v2` at `4bf45351075b3722d85a366da22b8df5f2322f75`
remains necessary to audit/publish the original `study20261029` refresh under its own source identity.
Do not update it or rebind its receipts. The normal DATA checkout can be updated only after its
own consumers are absent. Its previously mismatched TFM checkout and uninitialized VSC Library
must be verified and synchronized to the parent Git pins before reuse. Inspect local submodule
changes before updating; both libraries are read-only from this project.

## Public numbering and recorded campaign identities

`config/study.yaml` is read by the implementation and is the layout authority. Public config
folders, launcher names, notebook prose and scientific output use the following numbering.
Only immutable run IDs, embedded checkpoint campaign identities and dated evidence retain their
recorded identities. They are provenance, not alternative public experiment numbers.

### Recorded campaign identities

| Public analysis | Config folder | Persistent result payload | Recorded checkpoint campaign |
|---|---|---|---|
| E1 learning rate/progress | `config/experiment1` | `shared/main_grid` | `experiment1` |
| E2 seed sensitivity | `config/experiment2` | `experiment2` | `experiment2` |
| E3 L2-SP / E4 freezing | Reuse E1 config, no independent grid | `shared/main_grid` via `STUDY.json` pointers | `experiment1` |
| E5 DiagScale | `config/experiment5` | `experiment5` | `experiment5` |
| E6 LoRA | `config/experiment6` | `experiment6` | `experiment8` |
| E7 sampling | `config/experiment7` | `experiment7` | `experiment3` |
| E8 corpus composition | `config/experiment8` | `experiment8` | `experiment6` |
| E9 episode size | `config/experiment9` | `experiment9` | `experiment7` |
| E10 target context | `config/experiment10` | `experiment10` | `experiment4` |

Weights remain `checkpoints/trained/<recorded campaign>/<track>/`; run names and checkpoint
basenames are unchanged for resume and provenance. Embedded `experiment.output_group` and internal
`CREDITPFN_EXPERIMENT` denote that recorded identity; path helpers map it to the public payload.
**Do not set that internal variable from a public number by guessing.** Public launchers/cleanup
arguments perform the conversion. Job logging reads the config's explicit owner, never its folder number.
The shell bootstrap mapping is regression-tested against the YAML authority.

Both DATA and PROJECT already have schema-2 `STUDY_LAYOUT.json` markers and verified migration
journals. The 512-run factorial is stored once; E1/E3/E4 are analyses of it, never three independent
campaigns. E9 reuses E8's large-table 8,192-row control. Do not duplicate those controls or delete E8
while a saved E9 plan consumes it. Never apply the numbering permutation twice to an already marked tree.

## Deployment and one gated workflow

1. Edit and test locally. Run full pytest and all notebooks after plotting/notebook changes.
2. Commit and push validated source. Inspect remote jobs, checkout changes and submodules.
3. In a safe DATA checkout, `git pull --ff-only`, update clean submodules to their recorded pins,
   and verify the exact commit. Preserve any checkout used by active, queued or recoverable jobs.
4. Run current-source CPU/GPU gates through Slurm. Inspect real artifacts and logs on failure,
   repair locally, push and deploy again. Never patch remote source directly.
5. Only passing independent gates release the already authorized research and final publication.

The recovery workflow below queues the full sequence once from the sealed container, including
CPU preparation, ten runtime GPU checks, independent runtime acceptance, extension preparation,
12 correctness cases, 68 capacity cases, independent inspection, research and final audit/publication.
Use a fresh ID after a source change. This recorded specification is preserved in canonical DATA:

```bash
CREDITPFN_RUNTIME_VERSION=env20261029f sbatch scripts/slurm/study_cpu.slurm submit study20261030a --runtime-version env20261029f --specification "$PWD/output CreditPFN/experiment0/manifests/runtime_builds/study20261029/checkout/runtime/env20261029f/environment.json" --refresh-id study20261029 --refresh-checkout /data/leuven/383/vsc38338/CreditPFN-study-v2
```

**Do not rerun this example if its durable submission receipt already exists.** Inspect the receipt
and queue first. `--refresh-id` plus `--refresh-checkout` binds the original clean commit, preparation
hash and runtime; it skips completed migration/FP32 computation. The old evaluator independently
checks and publishes its own results from its frozen checkout. A partial publication resumes its
journal and validates every retained file, rather than treating absent moved source files as missing work.

Runtime checks cover packages, metrics, tensors/autograd and serialization; eight same-B200
original-versus-container comparisons (four bases × PD/LGD, original and CPT weights); and two
explicit-FP32 TabICL production-context shapes. All use fixed inputs and declared tolerances.
Successful runtime construction or Slurm exit zero alone is not acceptance.

### TabPFN v2/v2.6 extension

After all 80 checks pass, the controller queues **128 new training trials, 250 context tasks and
896 foundation-benchmark shards** across public E5/E6/E8/E9/E10. No base-weight download or recipe
change is part of this recovery. Arrays have no artificial running-task throttle. Admission retries
respect actual Slurm limits; benchmark dependencies follow only the training they consume.
Accepted job IDs are persisted before further submissions; ambiguous responses require reconciliation.

Correctness checks cover packed/separate projections, untouched base weights, nonzero adapter
gradients, initialization parity, live/export/reload/cache parity, automatic/FP32 inference and
interrupted-versus-uninterrupted training at every adapter rate. Capacity checks exercise real
partition tables, production ensemble/features and 32 updates. Full-context probes cover each model's
inference cap. These checks do not certify other future datasets, packages or caps.

| Base | Measured training episode ceiling | E10 inference context ceiling |
|---|---:|---:|
| TabPFN v2 | 10,000 | 10,000 |
| TabPFN v2.6 | 11,000 | 50,000 |
| TabPFN v3 / TabICLv2 | 26,000 | 50,000 in this study |

Training ceilings describe context-plus-query rows under the recorded feature/ensemble setup;
inference context is a different quantity. E9 uses 2,048, the shared 8,192 control, and the relevant
measured maximum. Do not increase caps without a new actual GPU probe. Five-fold test rows remain
complete while E10 reduces nested labeled context; v2/v2.6 never exceed their respective context caps.

Small extension records live on DATA at `output CreditPFN/<public payload>/extensions/<ID>/`;
large results/trajectories/predictions mirror that path on PROJECT. Weights use the recorded checkpoint
campaign with the same extension ID. Validation lives in `experiment0/extensions/<ID>/` on both tiers.
A reader accepts one prepared extension or an explicit `CREDITPFN_ANALYSIS_EXTENSION_ID`; it never
chooses by performance. Failed scopes must not masquerade as completed extension evidence.

### Experiment 6: LoRA workflow

LoRA uses `train.adaptation_mode: lora` and native query/value projections, not the old frozen/PEFT
flag. Its original dedicated gate is `CREDITPFN_LORA_GATE`; isolated validation lives under
`experiment0/lora/<ID>/`. The v2/v2.6 extension has its separate 80-case gate. Do not substitute
one gate's receipt for another. The accepted scientific recipe and known differences from DiagScale
are recorded in the research brief. Prior completed E6 trials are preserved; the new extension adds coverage.

### Individual commands

The whole-study controller above is the approved path for the remaining campaign. Standalone
`run_experiment5.sh`, `run_experiment6.sh`, `run_experiment8.sh`, `run_experiment9.sh` and
`run_experiment10.sh` now follow **public** numbers. `run_benchmark.sh experiment7`, for example,
means sampling. E3/E4 reject separate benchmark submission because E1 supplies that evidence.
The older standalone launchers preserve their explicit compatibility/concurrency options; they
are not the uncapped controller used for the current extension. Do not use them to duplicate
work already accepted in the workflow receipts.

Inspect the live queue independently of artifact completion:

```bash
squeue -M mindwell,wice -u "$USER" -o "%.20i %.28j %.12T %.12M %.65R %.80E"
```

`Dependency`, `Priority`, `Resources` and an explicitly configured array throttle are scheduler
states, not failed scientific output. An environment-retrieval hold needs a targeted `scontrol`
inspection; never cancel/resubmit an entire campaign to clear one held task. SIGUSR1/requeue
checkpoints, `.resume.pt` files and trial manifests must be preserved until terminal success.

## Cleanup and controlled cache lifetime

The user approved scoped disposable-cache cleanup and cache-policy tests on 30-10-2026.
`job_cache.py` creates an exclusively owned job/restart directory, preferring node-local scratch
and falling back to the native cluster scratch. NVIDIA, Triton, matplotlib and other library caches
are confined there. Successful jobs remove only their verified owned directory; failed jobs retain
it for diagnosis. Cache-cleanup failure is logged separately and cannot rewrite the scientific exit status.
Datasets, shared staged inputs, runtime images and recovery files never belong in these directories.

`storage_cache.slurm` inventories older numeric per-job cache directories in a Slurm allocation.
It requires exact project/scratch ownership, terminal successful CreditPFN accounting state,
no other active CreditPFN consumers, known cache contents and unchanged inventory before deletion.
Unknown/failed jobs, symlinks, foreign-owned files and unfamiliar contents are retained. Its preview
and apply receipts record exact paths/counts/bytes; never use a blanket scratch deletion.

```bash
sbatch scripts/slurm/storage_cache.slurm
```

The already approved scoped application runs the same checks before removal:

```bash
sbatch scripts/slurm/storage_cache.slurm apply
```

Do not run a second cleanup while the first or another CreditPFN consumer is active. Use the
appropriate cluster/partition override for wICE; shared training/input/runtime trees stay protected.

Scientific cleanup is separate and defaults to preview. `clean_run --experiment experiment6`
means public LoRA; internal recovery can explicitly use `--campaign experiment8`. Clearing shared
E1/E3/E4 scientific evidence requires `--shared-factorial`. Figure cleanup uses metadata dependencies
across RQ and overview folders, not guessed experiment folders. It removes the dependent PDFs and
caption entries; rerun notebooks to regenerate inline output and final summaries.

Extension cleanup affects only its named additions on both tiers, including tagged publication pages:

```bash
sbatch scripts/slurm/maintenance.slurm clean-extensions --id FAILED_ID
```

After reviewing ownership/consumers and preserving necessary failure evidence, add `--apply`.
Optional `--experiment` accepts public 0/5/6/8/9/10. Preserve original experiments and all base weights.
Do not delete submitted-phase receipts and assume reissuing them will recreate deleted completed work;
use a reconciled recovery or fresh ID.

## Final publication, collection and retirement

Final publication requires complete trial rosters, finite predictions, independently reproduced metrics,
fixed fold membership/targets and full context-test rows. Incomplete evidence remains explicit.
The old FP32 publication and the new extension audit remain separately source-bound.

Collect both complementary persistent output roots only after successful publication. The destination
must have the same schema-2 layout; never overlay a new-layout download onto an unmarked historical
tree. Local data/checkpoints stay outside Git. Rerun all 15 RQ/E0/general notebooks:

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 2
```

Check every changed figure visually, PDF/inline identity, caption ownership and complete final printed
summaries in `All_Results.md`. Scientific payload folders follow public experiments; figures follow RQs.

Keep compact failed-test evidence, source/runtime pins, migration checksums, submission receipts and
publication journals. Historical pre-Git build helpers and failed-build records are already archived
and hash-verified in canonical DATA `experiment0/manifests/runtime_builds/study20261029/`. Six obsolete
runtime-build workspaces and three empty failed runtime directories were retired by `62471310`.
After its final publication consumers finish, archive remaining provenance and remove the temporary
`CreditPFN-study-v2` checkout and only verified obsolete task-owned staging. Remove one-off migration
helpers after their recovery role ends; retain the layout reader and immutable migration journal.
The comparator environment stays until compatibility/rollback no longer needs it. Other projects,
scientific output and recoverable checkpoints are outside this cleanup.
