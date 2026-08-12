# Agents' memory — runs and dead ends

What is worth carrying between sessions: **the cluster runs that have been done**, and **the things
that turned out not to work**. Read it before starting; add to it as you go.

Not the changelog — that records edits to the repository. This records experience: what was run,
what came out, and what is already known to fail.

**Keep it short.** One line per run, four per dead end. Newest first, dates `DD-MM-YYYY`. Never
delete an entry: a run you would otherwise repeat and a dead end you already paid for are both
evidence.

Detail the table cannot hold lives in `RESULTS.md` (what each run measured), `ROW_CAPS.md` (the
measured context caps) and `CODE_NOTES.md` (code that looks wrong but is deliberate).

## Runs

One row per cluster run worth remembering — which is most of them, because *"have we already tried
that configuration?"* is the question this table exists to answer.

| Date | Run | Outcome | Notes |
|---|---|---|---|
| 07-08-2026 | run-6 · 36 trials/track, 3 bases, 100 epochs, `target_total_steps` 9100 | **done** | First fully green run: 36/36 train + 84/84 eval cells, drained in 7.1 h. Best PD mAUC 0.7620 (v3 1e-6 LoRA), best LGD RMSE 0.1335 (v3 1e-6 full). Half the eval pool never logged, so trained-vs-untuned is not computable for v3/TabICL. 54.9 GPU-h. `RESULTS.md` |
| 05-08-2026 | run-5 · 48 trials/track, first two-family run (TabICL added) | **partial** | 80/96 trials OK; the 16 `_iclhead` trials crashed (freeze-via-`.eval()`, see dead ends 06-08-2026). Eval never ran — the 21 h gate expired — so every number is a 2 000-row monitor eval. Drift 0.02 % of ‖w₀‖ at 3e-7: the PD null was undertraining. 43.7 GPU-h. |
| 05-08-2026 | probe · `probe_row_cap.slurm` j11509346, all three bases on B200 | **done** | The measurement the row caps come from: v3 2.49 GB/1k rows, v2.6 5.72, TabICL 0.51 per member. TabICL's ceiling is a cuDNN fused-attention failure between 26k and 40k, not memory. `ROW_CAPS.md` |
| 11-07-2026 | run-4 · 64 trials/track, TabPFN v3 + v2.6, 50 epochs | **done** | The clean homogeneous sweep and still the reference for cross-version science. 64/64 trained, 63 checkpoints straight to staging. PD: continued pretraining ≈ zero effect on discrimination (best Δ +0.0004). LGD: NLL improves while RMSE worsens. 8 PD eval cells walltime-killed. `RESULTS.md` |
| 10-07-2026 | rerun after `clean_run` · 64 trials/track | **contaminated** | 59/64 trials SKIPped on stale 09-07 FP16 checkpoints in the `$VSC_DATA` fallback dir. Only PD v3 a0–a4 actually retrained. Do not cite any number from this run. |
| 09-07-2026 | run-3 · 64 trials/track, first BF16 run | **partial** | LGD 32/32; PD 27/32 (tasks 0–4 ended together without a traceback, consistent with external termination). Monitor deltas invalid — the monitor re-seeded every epoch. |
| 08-07-2026 | probe · `probe_row_cap.py` on B200, v3 + v2.6 | **done** | First real memory measurement; replaced the fictional 100k/30k caps. v3 ≈ 2.5 GB/1k rows, v2.6 ≈ 5.7. Per-step cost is × `n_estimators`. |
| 04-07-2026 | run-2 · 32 PD + 32 LGD trials | **crashed** | Every trial died: PD at its first checkpoint save (staging readable but not writable from Mindwell), LGD on `bar_distribution` moving in tabpfn 8.x. Both fixed; see dead ends. |
| 03-07-2026 | run-1 · first full sweep attempt | **crashed** | 0 usable trials. The run that produced the writability probe, the import compat layer, and the preflight smoke tests. |

## Dead ends

Anything that cost more than a couple of minutes and did not work — including what was eventually
fixed, because the fix is one changelog line and the dead end was the hour.

### 08-08-2026

**Splitting the eval pools by model parity.**
- **Tried:** two eval pools (H100 + A100), tasks assigned by `model_index % 2`, so each pool held a
  disjoint half of the models over all datasets.
- **Result:** the `gpu_a100` pool (jobs …490/…492) never logged anything in run-6, and that lost
  **12 of 24 models outright — including `tabpfn-untuned[v3]` and `tabicl-untuned`.**
  Trained-vs-untuned is therefore *not computable* for v3 or TabICL from run-6.
- **Why:** model-parity makes the two pools structurally non-overlapping in the model dimension, so
  losing one pool deletes whole models rather than degrading resolution. The headline comparison in
  this project is trained-vs-untuned; a split that can drop the untuned control is the one split
  shape that can invalidate the run.
- **Instead:** split by **task stride** (every pool sees every model on a subset of datasets/folds),
  so a lost pool costs coverage, never a whole model. Changed 08-08-2026.

**Holding `epochs` fixed across bases and calling it equal exposure.**
- **Tried:** one `epochs: 50` (then 100) for every base, assuming that equalised training exposure.
- **Result:** v2.6 received **2.2× more optimiser steps than v3** at the same `epochs` — a silent
  confound sitting underneath every cross-version comparison in runs 4–6.
- **Why:** steps/epoch is `sum(ceil(rows_i / row_cap))`, and the row cap is per-base (v3 26 000,
  v2.6 11 000). A smaller cap means more steps per epoch, not smaller steps.
- **Instead:** `train.target_total_steps` trims epochs per base so every base gets the same step
  budget. Never compare bases on `epochs`; compare on steps.

### 06-08-2026

**Freezing TabICL's backbone with `.eval()`.**
- **Tried:** `model.col_embedder.eval()` (plus the row interactor) to freeze the backbone for the
  `_iclhead` adaptation arm.
- **Result:** **all 16 `_iclhead` trials failed**, both tracks, a few steps in: *"A view was created
  in no_grad mode and is being modified inplace with grad mode enabled."*
- **Why:** TabICL branches `if self.training: _train_forward() else: _inference_forward()` in
  **both** `ColEmbedder` and `RowInteractor`. Eval mode routes to the inference path, which runs
  under `torch.no_grad()` and writes CLS tokens into its own input in place
  (`interaction.py::_inference_forward`). Upstream's `_set_training_mode` has the same latent bug,
  so upstream code is not evidence that this is safe.
- **Instead:** freeze with `requires_grad=False` **only**, never `.eval()`. Nothing is lost —
  dropout defaults to 0.0 and there is no BatchNorm. See `CODE_NOTES.md` for the one place
  `.eval()` *is* still correct (`col_embedder` after `model.train()`).

### 05-08-2026

**`pip install tabicl` for a training environment.**
- **Tried:** pinning `tabicl>=2.1.1,<3` and verifying it with an import smoke test.
- **Result:** inference worked; **training died at step 1 on the cluster** with
  `ModuleNotFoundError: transformers`. The smoke test that existed specifically to catch this
  passed locally.
- **Why:** `tabicl._finetune/__init__` → `base` → `tabicl.train._optim` → `from transformers import
  get_*_schedule*`, and tabicl declares `transformers` **only** under its
  `finetune`/`pretrain`/`all` extras. The sklearn wrappers are lazily imported via `__getattr__`, so
  a plain install looks healthy. It passed locally because the dev venv already had transformers
  4.57.6 from another project. **An import smoke test only proves the current env; it cannot
  validate a dependency declaration.**
- **Instead:** pin `tabicl[finetune]>=2.1.1,<3` — declare the extra rather than hand-picking
  `transformers`, so upstream changing its finetune deps keeps us right.

**Blaming the version in that error message.**
- **Tried:** the first preflight failure message said the tabicl *version* was wrong.
- **Result:** sent a real debugging session down the wrong path (checking pins, reinstalling) while
  the actual cause was a missing extra.
- **Why:** the message named the most visible knob instead of the measured cause.
- **Instead:** the message now names the extra verbatim and says *"this is an EXTRA, not a version
  problem"*. Regression-tested in `tests/test_tabicl.py`.

**Running `scripts/probe_row_cap.py` bare on a login node.**
- **Tried:** running the memory probe interactively on `tier2-p-login-4` to save a queue wait.
- **Result:** it found the **Quadro P6000 display GPU** (sm_61) and died inside `model.to(device)`
  with a misleading `CUDA error: out of memory`.
- **Why:** `torch.cuda.is_available()` returns True for a GPU whose architecture the installed
  PyTorch has no kernels for. A capacity probe on the wrong GPU is worse than no probe, because its
  numbers get copied into `config/data.yaml` — cf. the 04-07-2026 "0.93 GB" entry.
- **Instead:** the probe checks compute capability against `torch.cuda.get_arch_list()` and refuses
  on a login hostname without `SLURM_JOB_ID`, printing `sbatch scripts/slurm/probe_row_cap.slurm`.

**Trusting `conda activate` to decide where `pip` installs.**
- **Tried:** `conda activate CreditPFN && pip install X`.
- **Result:** reported success while installing into `TabPFNCredit/tabpfncreditvenv` — a *different
  project's* venv.
- **Why:** an active virtualenv puts `$VIRTUAL_ENV/bin` ahead of the conda env on `PATH`, and conda
  does not remove it. Later the same failure recurred with an Lmod Python module shadowing conda.
- **Instead:** `_activate_env.sh` strips a shadowing venv from `PATH`, unsets `VIRTUAL_ENV`,
  prepends `$CONDA_PREFIX/bin`, and `hash -r`s — loudly. Interactively there is no protection:
  `deactivate` first and check `which python pip`. Every job log's `Active conda env:` line is the
  authority on where deps must live.

**Sizing the eval gate's walltime from the compute, not the queue.**
- **Tried:** 21 h walltime for the wICE gate job that watches Mindwell and releases eval.
- **Result:** **eval never ran in run-5.** The gate expired with PD at 19/47 trials done; every
  number from run-5 is a 2 000-row monitor eval, not the real 5-fold CV.
- **Why:** the run needed 47 h of wall-clock for 43.7 GPU-h — 7.5 h queued, then Mindwell granted
  only 1–2 concurrent GPUs instead of the 24 requested. Gate walltime must cover *queueing*, which
  is not predictable from the compute.
- **Instead:** 72 h / 8400 polls, and the timeout message now prints the exact recovery command.
  Separately: request a **short** trial walltime so tasks backfill (run-6 drained in 7.1 h).

### 04-08-2026

**Reading TabICL's context limits off the library's default arguments.**
- **Tried:** set `max_rows_per_epoch.tabicl: 10000` and `max_rows_per_model.tabicl: 50000` from
  `max_data_size=10_000` in their convenience finetune wrapper.
- **Result:** wrong by 2.6× (training) and 20× (eval); the user challenged the values and was right.
  It would have handicapped TabICL against v3 and thrown away its headline capability.
- **Why:** **a library default is not a capability limit.** Their own stage-3 pretraining runs
  400–60 000 samples (grad-checkpointed above 20K), and Qu et al. 2026 report 1M samples × 500
  features in ~450 s under 50 GB GPU + 24 GB CPU via offloading; QASSMax exists precisely to keep
  attention sharp at long context.
- **Instead:** caps come from the paper, then get **measured** (`scripts/probe_row_cap.py`) — 26 000
  train (= v3 parity, so architecture is not confounded with context size) and 1 000 000 eval. Full
  numbers in `ROW_CAPS.md`.

**Resolving the untuned eval row cap by stripping a dirname-style prefix.**
- **Tried:** `resolve_max_rows_for_handle` stripped `"tabpfn-untuned__"` off `handle.name` to find
  the base tag.
- **Result:** the strip **never matched** (untuned names use brackets: `tabpfn-untuned[v3]`), so
  every untuned model silently fell through to the `default` cap — untuned-v3 scored on 50k-row
  folds while trained-v3 got 1M.
- **Why:** two naming conventions for the same concept, and a string operation that fails *silently*
  when it doesn't match. It biases the project's headline comparison on any dataset above the
  default cap.
- **Instead:** both branches resolve through `_short_base_tag` on the **base checkpoint**, never the
  display name. Regression test in `tests/test_tabicl.py`.

**Growing the grid without grepping for hardcoded bounds.**
- **Tried:** adding the two TabICL bases (32 → 48 trials/track).
- **Result:** two places still carried numbers derived from the old count — the
  `#SBATCH --array=0-199` fallback in `eval_{pd,lgd}.slurm` (PD now needs ~275 tasks, so the tail
  would have been **silently dropped**) and the trial-count comments.
- **Why:** `run_full_pipeline.sh` sizes arrays dynamically, so a bare `sbatch` is the only path that
  can truncate — which makes it the path nobody tests.
- **Instead:** after changing any `sweep.*` list, grep for hardcoded array bounds and trial counts.
  Bumped to `0-399`.

### 30-07-2026

**Trusting an agent changelog entry that a document was added.**
- **Tried:** looking for `docs/EDW_DATASET_FEASIBILITY.md`, recorded in the history as written.
- **Result:** the file exists nowhere — never committed, and the analysis behind it is gone.
- **Why:** a changelog entry records intent at write time, not the state of the working tree; an
  uncommitted file is indistinguishable from one that was never created.
- **Instead:** surviving substance, recorded here so it is not lost twice: EDW ABS panel data is
  technically viable after a leakage-safe ETL, **but** EDW's Jan-2026 standard terms prohibit AI
  training unless the university agreement overrides. Legal question unresolved. Re-create the doc
  if the question becomes live.

### 11-07-2026

**Re-running the sweep without checking the fallback checkpoint directory.**
- **Tried:** `clean_run.py`, then a fresh 64-trial sweep on the new BF16 code path.
- **Result:** **59 of 64 trials SKIPped** on stale 09-07 FP16 checkpoints and the run was
  contaminated; only PD v3 a0–a4 actually retrained.
- **Why:** the resume-skip check looks in *both* staging and the `$VSC_DATA` fallback (by design —
  see the 04-07-2026 staging entry), but `clean_run.py` only cleaned staging.
- **Instead:** `clean_run.py` cleans both roots. Anything that resolves two possible locations must
  be cleaned in both.

**Splitting the eval pools by raw task index parity.**
- **Tried:** `task_index % 2` to assign work to the H100 and A100 pools.
- **Result:** with LGD's 2 datasets × 2 pools, **all** `lgd_lendingclub` work landed in the slow A100
  pool and went unscored for hours.
- **Why:** index parity correlates with dataset when the dataset count is small and even — the
  stride and the structure line up.
- **Instead:** `--list-tasks --pools K --pool i`. (Superseded again on 08-08-2026 — see that entry;
  model-parity fixed this failure and introduced a worse one.)

### 10-07-2026

**Re-seeding the monitor evaluation every epoch.**
- **Tried:** the per-epoch monitor drew its own sample each time it ran.
- **Result:** every historical epoch baseline→final delta is **invalid** — rows and splits changed
  under the metric, so the "improvement" included resampling noise.
- **Why:** a progress metric must hold everything but the model fixed; a fresh sample makes it a
  measurement of the sampler.
- **Instead:** one fixed sample, reused across epochs. Numbers before 10-07-2026 are not comparable
  to numbers after.

**Treating the two continued-pretraining papers as one method.**
- **Tried:** reading LR / schedule / regularisation settings from whichever of the two local sources
  mentioned them.
- **Result:** nearly mixed **Garg et al. Real-TabPFN** (71-table corpus CPT; LR 3e-7;
  warmup→cosine; L2-SP λ=0.003; 20k steps; 60/40 context/query) with **Rubachev et al. On
  Finetuning** (single-dataset FT/PEFT study; tuned 5e-6…5e-4; patience 16; constant LR; no L2-SP).
- **Why:** the local `On Finetuning….txt` dump is Rubachev's repo, not a Real-TabPFN code release —
  the filename suggests otherwise. Garg does not report AdamW weight decay at all.
- **Instead:** cite per-paper, and treat a repo dump's filename as a hint, never as provenance.

**Running heavy multi-agent workflows to completion in one pass** *(date approximate)*.
- **Tried:** large fan-out workflows (cleanup sweep, literature synthesis) on Fable-5.
- **Result:** hit per-model usage limits mid-run; partial completion.
- **Why:** the cap is per-model and shared across the fan-out, so agent count multiplies the risk.
- **Instead:** expect partial completion and resume from the runId (cached agents return instantly),
  or run fewer/cheaper agents.

### 08-07-2026

**Committing a fix and telling the user to rerun.**
- **Tried:** editing, committing locally, then asking for a rerun on the cluster.
- **Result:** **the single most expensive recurring error in this project.** The VSC pulls
  `origin/main`, local `main` was repeatedly `[ahead N]`, so every rerun pulled **stale** code and
  hit bugs already fixed on disk.
- **Why:** "fixed" felt equivalent to "fixed everywhere". The tell is a traceback whose line numbers
  don't match the local file; diagnose with `git status -sb` / `git log origin/main..HEAD`.
- **Instead:** commit, then **the user pushes** (never me — hard rule), then confirm
  `git show origin/main:<file>` carries the change *before* anyone reruns. Stay on `main`; a branch
  never reaches the cluster.

**Hardening a best-effort pre-check into a hard failure.**
- **Tried:** a cleanup turned `_ensure_processed`'s vacuous "0 candidate datasets" warning into
  `raise RuntimeError`.
- **Result:** killed **all** training trials on both tracks.
- **Why:** two compounding mistakes. `corpus.{train,test}_dataset_ids` is a **per-track mapping**
  (`{pd: [...], lgd: [...]}`) and `list(dict)` yields the **keys**, so the check saw 0 candidates on
  every track; and a pre-check that cannot see the real corpus must not be authoritative — the real
  validation is downstream in `split_from_cfg`.
- **Instead:** parse per-track config via `resolve_ids_for_track` (never `list()`), and keep
  best-effort checks as warnings.

**Submitting cross-cluster with a bare `sbatch`.**
- **Tried:** `sbatch` to Mindwell from a Genius login node without `--export`.
- **Result:** *"user env retrieval failed requeued held"* — jobs held, not run.
- **Why:** without `--export`, Slurm spawns the login shell on the target node to rebuild the
  environment, which fails at this site.
- **Instead:** `#SBATCH --export=ALL` on every directly submittable script (all 7).
  `run_full_pipeline.sh` passes `--export` on the CLI, so it was never affected. **Do not remove it.**

**Assuming a successful `conda activate` means a usable interpreter.**
- **Tried:** activating the named env and running python.
- **Result:** `python` resolved to `/bin/python`; every project import missing.
- **Why:** a broken/empty named env activates **without error** while contributing nothing to
  `PATH`.
- **Instead:** `_activate_env.sh` verifies python lives under `$CONDA_PREFIX` and imports the deps,
  else falls back to `base`. Repair with
  `conda create -n CreditPFN --clone base && pip install -e ".[dev]"`.

**Scaling row caps from a paper plus one measurement.**
- **Tried:** caps of 100k (v3) / 30k (v2.6), derived from the 04-07 "0.93 GB @ 20k" figure and a
  paper-scaling argument.
- **Result:** OOM.
- **Why:** the figure was a bad measurement (see 04-07-2026), and the real driver was missed
  entirely: a step forwards **all** `n_estimators_finetune` members and holds every member's graph
  for one backward, so per-step memory ≈ members × per-member. PD uses 2, LGD 8.
- **Instead:** measured caps only (`ROW_CAPS.md`), member-aware scaling in `train_one_config`. **Do
  not raise a cap without re-running the probe.**

### 04-07-2026

**Reading the monitor's `gpu_peak_alloc` as the training peak.**
- **Tried:** taking 0.93 GB @ 20k rows from a job log as the per-step training cost.
- **Result:** fiction — off by ~50× — and it propagated straight into `config/data.yaml` and then
  into OOMs.
- **Why:** the number came from the lightweight 32-estimator **monitor eval**, not from a training
  step with a backward pass.
- **Instead:** capacity numbers come from `scripts/probe_row_cap.py` (explicit fwd+bwd), never from
  a log line that happens to mention memory.

**Assuming project staging is writable because it is readable.**
- **Tried:** saving trained checkpoints straight to `/lustre1/project/stg_00211/CreditPFN`.
- **Result:** all 32 PD trials died at the first checkpoint save with `Errno 13` — after full
  training compute was spent.
- **Why:** staging was readable but not writable from Mindwell compute nodes (a dir-level
  perms/ACL problem the code cannot fix; the user chmod'd it on 11-07).
- **Instead:** `resolve_writable_staging_path` probes writability **before** any compute and falls
  back to `$VSC_DATA` with a loud warning; the eval gate archives fallback checkpoints back to
  staging. A trained checkpoint can legitimately be in **either** place — do not assume staging.

**Importing tabpfn internals directly.**
- **Tried:** `from tabpfn.architectures.base import bar_distribution`.
- **Result:** killed all 32 LGD trials on an import error (classifiers unaffected, so it looked
  track-specific).
- **Why:** tabpfn 8.x moved it to `.shared`, and moved the ensemble preprocessor to
  `preprocessing.ensemble`. Private layout is not API.
- **Instead:** import tabpfn internals **only** via `src/train/tabpfn_compat.py`, which tries all
  known paths and aliases `sys.modules` for old pickles.

**Releasing the eval gate on queue completion.**
- **Tried:** letting eval start once the training array left the queue (later: once one shared
  success sentinel appeared).
- **Result:** eval ran against missing or partial checkpoints.
- **Why:** "no longer queued" includes failed, cancelled and walltime-killed; one shared sentinel
  cannot distinguish 1 success from 47.
- **Instead:** one `train_ok_<track>_<index>` sentinel **per task**; the gate requires the full
  planned count and computes the post-training roster from what actually exists.

### 02-07-2026

**Submitting under `lp_mindwell_pilot`.**
- **Tried:** the free pilot account for Mindwell B200 jobs.
- **Result:** *"Invalid account or account/partition combination"* — every submission rejected.
- **Why:** the pilot ended when Mindwell went to production; the account is dead, not merely empty.
- **Instead:** `lp_verbekelab` (verify with `sacctmgr -s show user $USER cluster=mindwell`; note
  `sacctmgr` has no `-M` flag, unlike `squeue`/`sinfo`/`scancel`).

### 23-06-2026

**Chaining the two clusters with a Slurm dependency.**
- **Tried:** `--dependency=afterok` from a wICE job on a Mindwell job.
- **Result:** unsupported on VSC; the chain cannot be expressed.
- **Why:** Slurm dependencies are per-cluster; there is no cross-cluster job state to depend on.
- **Instead:** bridge with files and a watcher — a `data_done` sentinel on `$VSC_DATA` (NFS, visible
  everywhere) that train waits for, plus a wICE 1-CPU "gate" job that polls Mindwell via `squeue`
  and which eval `afterok`-depends on. This is why `run_full_pipeline.sh` is more complex than a
  dependency chain. **It is deliberate — do not "simplify" it into afterok.**
