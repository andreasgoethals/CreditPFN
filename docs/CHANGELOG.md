# Changelog

What changed in this repository, newest first. One chapter per date, `DD-MM-YYYY`; one bullet per
change, as short as it can be — the *what*, plus the *why* only when it is not obvious. The detail
belongs in the commit.

Two things do NOT go here: what a cluster run measured and what turned out not to work — both live
in [`AGENTS_MEMORY.md`](AGENTS_MEMORY.md). Durable facts go in `RESULTS.md`, `CODE_NOTES.md` and
`ROW_CAPS.md`.

Entries above 11-08-2026 follow this rule. Below it they use an older, longer house style
(`### <change> — <agent>` with What/Why/Verified bullets) and are left as written, because a past
day is never rewritten.

## 11-08-2026

- **Brought the repository in line with the renewed `docs/TEMPLATE.md`** — a starting point rather
  than a contract, so the previous pass's compliance test, `scripts/check.py`, CI workflow and
  `CITATION.cff` are gone, and the `tunable:` → `sweep:` config rename is reverted.
- `src/utils/paths.py` is now the template's file with a marked **project layer** underneath: the
  four `resolve_*` functions, the `paths.data_source` knob and the writability probe. The template's
  `outputs_dir`/`results_dir`/`raw_dir`/`processed_dir`/`checkpoints_dir` delegate to it, so both
  APIs resolve to the same paths — asserted in `tests/test_paths.py`, not assumed.
- **Everything generated now lives under `output/`.** `output/runs/epochs/` → `output/manifests/`,
  `data/manifest_{track}.csv` → `output/manifests/`, `data/dedup/` → `output/manifests/dedup/`, and
  run logs from `<root>/logs/` → `output/logs/` (all 8 SLURM scripts updated). Deleted a stale
  `output/training/` tree from a June layout nothing references.
- Utilities moved out of `scripts/`, which now holds only the four experiments and `slurm/`:
  `clean_run.py`, `run_notebooks.py`, `update_tfm_library.py`, `config.py`, `logging_setup.py` are
  the template's files under `src/utils/`, run with `python -m`. `clean_run` keeps this project's
  extra targets (trained weights on **both** tiers, `.sentinels/`) and its refusal list.
- `src/visualize/style.py` filled in: an 11-entry Okabe–Ito palette keyed by **entity**, and all 29
  `figsize=` calls in `eval_viz`/`training_viz` now go through `style.figsize(WIDTH_FULL, ratio=…)`.
  Colours used to come from `cm.get_cmap("tab10", n)` indexed by list position, so dropping one arm
  from a plot repainted every arm after it.
- Notebooks: three real bugs fixed — the repo-root probe still looked for `src/utils/training_viz.py`
  and would have walked past the root; `FigureSaver` was keyed on an underscored name the runner
  never reads, so `CAPTIONS.md` and `All_Results.md` saw zero figures; and `display()` is an IPython
  builtin the runner does not provide. Also added `style.apply()` and stripped the hand-written
  `NN_` prefixes the saver now supplies. `python -m src.utils.run_notebooks`: **6/6, 79 figures.**
- `docs/AGENTS_MEMORY.md` is tracked and rewritten in the new two-part shape: a **Runs table**
  seeded with all nine cluster runs and probes since 03-07-2026, then the 28 dead ends unchanged.
- `src/utils/run_log.py` → `logging_setup.py`, the template's name for that slot, keeping this
  project's implementation (SLURM-array-aware filenames, structured formatter). Two modules
  configuring logging is the drift the template exists to prevent.
- `src/utils/config.py` fills in the template's one ask: `dump_resolved()` writes the fully
  resolved config, storage roots and SLURM ids to `output/manifests/resolved/` at every entry
  point. The YAML in `config/` may have been edited since a result was produced.
- `CLAUDE.md` is now one line, `@AGENTS.md`; `AGENTS.md` is the template's with a CreditPFN section.
- Processed-data CSVs are written atomically (temp file + `os.replace`) in
  `src/data/{dedup,register,sanitize}.py` — a job killed mid-write left a truncated CSV that pandas
  parses happily and training silently used.

## 08-08-2026

### Add per-dataset loss and per-stage drift to the per-epoch CSV — Claude

- **What:** Each per-epoch CSV row now carries `loss__<dataset_id>` for every
  training dataset and `drift__<stage>` for every top-level model stage, and
  monitored epochs log a `drift by stage:` line. Stages are grouped on the
  first dotted component of the parameter name, so it is architecture-agnostic
  (TabICL → col_embedder / row_interactor / icl_predictor). The CSV writer
  widens its header and pads earlier rows when a new column first appears,
  since the baseline row has no per-dataset losses and drift only exists on
  monitored epochs.
- **Why:** The scalar epoch loss hides whether the corpus is learned uniformly
  or some datasets improve while others degrade — the per-dataset forgetting
  question. Per-stage drift answers WHERE credit-specialisation lands, a claim
  nobody has made for tabular foundation models and one the single aggregate
  drift number cannot support.
- **Verified:** Real training run produces both column families, the CSV stays
  a rectangle across the baseline / non-monitored / monitored mix, and stage
  drift ordered icl_predictor > row_interactor > col_embedder as expected for
  a freeze-nothing run.

### Fix the context-balance metric to use canonical labels — Claude

- **What:** `ctx_pos` is now measured in the batch builders from the ORIGINAL
  sampled labels and carried on every batch type (`TabPFNBatch`,
  `TabPFNEnsembleBatch`, `TabICLTrainBatch`) through `.to(device)`.
- **Why:** The first version counted `y > 0` on the batch tensors, but both
  families class-permute per ensemble member — so on 25 %-positive fixture data
  it reported **74.7 %**. The metric added to verify balanced sampling was
  itself unreliable, and after the device move a non-propagated field silently
  reverted to NaN and the number vanished entirely.
- **Verified:** Now reads 25.22 % on 25 %-positive data; new test asserts all
  three batch types preserve it through `.to()`.

### Log the realised context class balance — Claude

- **What:** Each epoch line now reports `ctx_pos=NN.NN%` — the mean positive
  rate of the context the model actually trained on (classification only).
- **Why:** `context_sampling: balanced` was switched on for run-6 and nothing
  in the logs could confirm it took effect. Proportional sampling of a
  1 %-default dataset leaves ~1 % positives in context; balanced leaves far
  more, so this one number verifies the intervention and quantifies how much
  minority signal the model is actually exposed to.

### Record post-hoc recalibrated metrics for every model — Claude

- **What:** Every binary eval row now also carries `ece_platt`,
  `brier_score_platt`, `log_loss_platt` and the `_isotonic` trio, fitted on
  the inner-val split and applied to the test fold WITHOUT retraining. Raw
  columns are untouched, so `ece - ece_platt` is the recoverable share.
- **Why:** CPT worsens calibration consistently across run-4 and run-6 while
  leaving discrimination flat. Whether that is recoverable for free decides
  whether the finding reads "use CPT and recalibrate" or "CPT damages
  calibration irreparably" — and Purucker et al. found TabPFN one of only two
  models recalibration makes WORSE, so it is genuinely open. For credit risk
  the probability is the regulated quantity, so this may matter more than AUC.
- **Verified:** On deliberately over-confident probabilities ECE 0.1077 →
  0.0023 (Platt) / 0.0027 (isotonic) with AUC unchanged; multiclass yields NaN
  without raising; 3 new tests.

### Sweep downward, A/B the query fraction, log drift as a percentage — Claude

- **What:** LRs `[1e-6, 1e-5, 3e-5]` → `[3e-7, 1e-6, 3e-6]`; `query_fractions`
  now sweeps `[0.20, 0.40]`; grid 18 → 36/track (arrays `0-35`). Training now
  measures ‖w0‖ once at anchor time and prints `drift=‖w−w0‖/‖w0‖` as a
  percentage on every epoch line.
- **Why:** With a real step budget, run-6 showed LOWER LR wins on both tracks,
  and 3e-7 (Real-TabPFN's exact rate) was only inert because run-5 had 600
  steps. qf 0.40 doubles the loss signal per step at identical memory. The
  drift figure answers "did the model actually change?" without hand
  arithmetic on the L2-SP term — the question that dominated the last two
  post-mortems.

### Make the grid test structural instead of a magic number — Claude

- **What:** `test_training_grid_contains_both_families` asserted a hardcoded
  trial count; it now asserts that the pipeline's real grid expansion equals
  the cartesian product of the configured axes, and that every base appears.
- **Why:** The constant broke on all three deliberate reshapes (48 → 18 → 36),
  three false alarms that caught no bug. The structural invariant still
  catches a silently dropped or duplicated axis.

### Equalise the training budget across architectures — Claude

- **What:** New `train.target_total_steps` (9 100) trims epochs per base so
  every base runs the same number of optimizer steps; `train.epochs` becomes
  an upper bound. v2.6 now takes 45 epochs, v3/tabicl 100.
- **Why:** Under `full_pass`, steps/epoch = Σ ceil(rows / row_cap), so a base
  with a smaller memory-driven row cap silently trains LONGER. In the 07-08
  run v2.6 got 20 300 steps and v3/tabicl 9 100 at identical `epochs: 100` —
  a 2.2× budget gap that confounds every cross-architecture comparison. It
  showed in the drift: v2.6 @3e-5 reached l2sp 0.61 vs v3's 0.0045.

### Spread eval pools by task instead of by model — Claude

- **What:** `--pools/--pool` now strides over the task list (`i % pools`)
  instead of splitting on model index (`m % pools`).
- **Why:** The 07-08 run lost the gpu_a100 pool, and model-parity meant that
  cost 12 of 24 models OUTRIGHT — including `tabpfn-untuned[v3]` and
  `tabicl-untuned`, the two controls the whole trained-vs-untuned comparison
  rests on. Under task-stride a lost pool costs each model roughly half its
  datasets instead, so every model survives and comparisons stay computable.

### Record the context-sampling strategy — Claude

- **What:** `context_sampling` now appears in the training debug banner and in
  each checkpoint's provenance.
- **Why:** It was added on 06-08 but never surfaced, so the 07-08 logs give no
  way to confirm balanced sampling was actually active — unacceptable for the
  axis the literature says matters most.

### Analyse the 07-08 two-family run — Claude

- **What:** 36/36 training trials and 84/84 eval cells OK; the freeze-backbone
  fix held (all 6 `_iclhead` trials trained), the step budget rose 600 → 9 100
  / 20 300, weight drift became non-trivial, and the LR floor worked (final LR
  = 5 % of peak). Training drained in 7.1 h vs 47 h. **No repository changes
  beyond the fixes above.**
- **Why:** First run where the pipeline, the budget and the eval all worked.
- **Verified:** Half the eval roster is still missing (the a100 pool never
  logged), so trained-vs-untuned is not yet computable for v3 or TabICL.

---

## 07-08-2026

### Restructure this log newest-first with DD-MM-YYYY dates — Claude

- **What:** Reversed the day blocks so the most recent work is at the top of
  the file, rewrote every day heading (and the dates inside the entries) as
  `DD-MM-YYYY`, and updated house-style rule 1 to match — new entries now go
  at the TOP, and a same-day change is added at the top of that day.
- **Why:** The log is read at the start of every task to learn recent state,
  so the newest entry should be the first thing on screen.
- **Verified:** All 23 change entries and 7 day blocks preserved; no ISO-format
  date remains in the file.

---

## 06-08-2026

### Fix the TabICL freeze-backbone crash that killed 16 trials — Claude

- **What:** Freezing used `.eval()`, but TabICL branches `if self.training:
  _train_forward else: _inference_forward` in both ColEmbedder and
  RowInteractor, so eval mode routed the frozen stages onto the no_grad,
  KV-cached inference path — which writes CLS tokens into its input in place
  and raises "a view created in no_grad mode modified inplace". Freezing is
  now `requires_grad=False` only; the model stays on the train forward path.
- **Why:** All 16 `_iclhead` trials failed on both tracks while all 16 TabICL
  full-FT trials completed — the tell that the freeze, not TabICL, was wrong.
- **Verified:** New regression test runs 5 optimiser steps through the real
  freeze path with `recompute=True` and asserts the stages stay in train mode.

### Give the eval gate a walltime that matches reality — Claude

- **What:** Gate walltime 21 h → 72 h and poll budget 2400 → 8400; on timeout
  it now prints the exact recovery command and how to check readiness.
- **Why:** The run needed 47 h of wall-clock (7.5 h queued, then 1–2 concurrent
  GPUs instead of 24), so the gate expired with PD at task 19/47 and **eval
  never ran at all** — 43.7 GPU-h of training went unscored.

### Keep Python caches out of $VSC_HOME — Claude

- **What:** `_activate_env.sh` now exports `XDG_CACHE_HOME`, `HF_HOME`,
  `TORCH_HOME`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `MPLCONFIGDIR`,
  `PIP_CACHE_DIR` and `WANDB_DIR` under `$VSC_DATA/.cache/creditpfn`, and
  disables wandb.
- **Why:** A 99 %-of-3 GB quota warning on `/user/leuven/383/…`. We set no
  cache variables, so every default landed in HOME — the ~225 MB HF copy of
  the TabICL checkpoints, the pip wheel cache, and Triton/CUDA JIT caches from
  96 GPU jobs.

### Retarget the sweep at the real problem: the model never moved — Claude

- **What:** Measured weight drift from the logged L2-SP term: at lr 3e-7 the
  weights moved ~0.02 % of ‖w₀‖. Real-TabPFN pairs that LR with 20 000 steps;
  we ran 600–10 150. So: dropped lr 3e-7 and the `one_sample` pass mode (both
  provably inert), raised epochs 50 → 100, and added `scheduler.min_lr_fraction`
  so a short cosine no longer decays to exactly zero. Grid 48 → 18 trials per
  track, each trained ~7× longer for a similar GPU budget.
- **Why:** The PD null result was an artefact of undertraining, not evidence
  about continued pretraining.

### Right-size the training walltime requests — Claude

- **What:** PD 48 h → 10 h, LGD 12 h → 4 h, both sized from run-5's measured
  per-trial times (v2.6 full_pass ~2.4 h at 50 epochs → ~5 h at 100).
- **Why:** Slurm ran only 1–2 of the 24 requested array tasks in run-5, so
  43.7 GPU-h stretched over 47 h of wall-clock. A 48 h request is nearly
  impossible to backfill into gaps between other jobs; a 10 h one is easy, so
  this should buy real queue position at no cost to the science.

### Add balanced context construction — Claude

- **What:** New `finetuning.context_sampling` (`stratified` | `balanced`),
  shared by both families so it can never confound a cross-family comparison.
  Default switched to `balanced`.
- **Why:** Tanna et al. 2026 measure balanced context as worth 3–4 AUC points
  over uniform on Home Credit and Lending Club — both in our training corpus —
  a larger effect than the choice of model. Proportional sampling put ~260
  positives in a 26 000-row context; balanced puts every positive available.
- **Verified:** On a synthetic 1 %-positive 500k-row set, 260 → 4 999
  positives at the same 26 000 rows; regression falls back to uniform.

### Make the TabICL monitor match its eval settings — Claude

- **What:** The per-epoch monitor scored TabICL with 32 ensemble members while
  `config/eval.yaml` scores it with 8. Added
  `train.epoch_eval_n_estimators_tabicl: 8`.
- **Why:** The monitor exists so its curves are comparable to the final eval
  numbers; running it at a different ensemble size broke that and cost 4× the
  monitor time.

---

## 05-08-2026

### Correct the TabICL row caps from the paper — Claude

- **What:** Eval `max_rows_per_model.tabicl` 50 000 → **1 000 000** (= v3):
  Qu et al. 2026 report 1M samples × 500 features in ~450 s under 50 GB GPU,
  so 50k discarded the model's headline capability and handicapped it 20×
  against v3. Training `max_rows_per_epoch.tabicl` 10 000 → **26 000** (= v3):
  unequal caps confound architecture with context size, and 10 000 came from
  `max_data_size=10_000`, a default argument of their convenience finetune
  wrapper — their own stage-3 pretraining uses 400–60 000 samples. Probe grid
  widened to 10k–60k.
- **Why:** A wrong cap silently biases the cross-family comparison the second
  family exists to make.
- **Verified:** Read from the TabICLv2 PDF (§4.2, App. H.2, §B.1), not from
  library defaults. Still unmeasured on VSC hardware.

### Require the `tabicl[finetune]` extra — Claude

- **What:** `transformers` is declared only under tabicl's `finetune` extra,
  so a plain install works for inference and dies at training. Pin is now
  `tabicl[finetune]>=2.1.1,<3`; the error message names the extra instead of
  blaming the version; `smoke_test` reports core / inference / finetune
  separately, in ASCII.
- **Why:** The original message misdirected a real debugging session, and a
  `UnicodeEncodeError` in a prolog check would abort the job it protects.
- **Verified:** Regression test blocks `transformers` and asserts the message.
  **Supersedes** the `tabicl>=2.1.1,<3` command in the 04-08-2026 entries.

### Harden the environment and probe entry points — Claude

- **What:** `_activate_env.sh` strips a shadowing `$VIRTUAL_ENV/bin` from PATH
  before activating conda (an active venv silently won, so `pip` installed
  into another project's venv). `probe_row_cap.py` now refuses an unusable
  GPU — on a login node it had grabbed the Quadro P6000 display card (sm_61,
  no kernels in this PyTorch) and died with a misleading CUDA OOM — and
  prints the sbatch command instead. New `train_pipeline.py --trial-family N`
  lets the SLURM prolog run the TabICL preflight only for TabICL trials, so a
  missing optional dep costs 16 trials rather than all 48.
- **Why:** Each of these cost real cluster time, and a capacity probe on the
  wrong GPU is worse than none — its numbers get copied into `config/`.
- **Verified:** 276 tests; both SLURM scripts pass `bash -n`.

### Restructure this log and formalise its style — Claude

- **What:** Regrouped every entry under one `##` heading per day with one
  `###` heading per change, rewrote all of them into the same
  What/Why/Verified shape, and added the "House style" ruleset at the top.
  Older days were edited (normally forbidden by rule 6) at the user's
  explicit request for this one pass.
- **Why:** Entries had drifted into three different formats and several were
  long enough that the actual change was hard to find.

### Move the agent log and memory into docs/ — Claude

- **What:** `CHANGELOG.md` → `docs/CHANGELOG.md` (via `git mv`, so
  the rename is tracked) and `AGENTS_MEMORY.md` → `docs/AGENTS_MEMORY.md`.
  Updated every pointer: `CLAUDE.md` and `AGENTS.md` (4 lines each),
  `CLAUDE.local.md`, the two `.gitignore` comments, this file's house-style
  rule, and the README doc table. The `.gitignore` pattern is a bare filename,
  so the memory file stays ignored at its new path.
- **Why:** The user asked to de-clutter the repo root. These two were the only
  loose root files with no tooling dependency — everything else there
  (`.gitignore`, `.gitmodules`, `pyproject.toml`, `CLAUDE.md`, `AGENTS.md`,
  `README.md`) is resolved from the root by git, pip/pytest, or agent
  auto-loading and would break if moved.
- **Verified:** No Python, config or SLURM file references either name; both
  moved files contain zero relative markdown links, so nothing re-resolves.
  The 10-07-2026 entry's mention of introducing `AGENTS_MEMORY.md` is left
  bare on purpose — it narrates where the file was then (rule 6).

### Rescope the temporal-split plan after auditing the corpus — Claude

- **What:** Measured that only 5 of 25 raw datasets carry a parseable date
  column, that `sanitize.py` drops them, and that **none of the 5 PD test
  sets** has one (only LGD `loss2` does). Rewrote that roadmap item into three
  scoped options: a `loss2`-only case study, re-pinning the corpus split, or
  re-sourcing fuller raw files.
- **Why:** It had been listed as a scheduling task when it is really a corpus
  decision, and would have been discovered only mid-implementation.

---

## 04-08-2026

### Move CreditPFN-specific literature notes out of the shared library — Claude

- **What:** Created `tfm-library/PROJECT_SPECIFIC.md` (gitignored by the
  library, so submodule status stays clean) with the library pin recorded in
  its header, and repointed the 4 stale paper links in notebooks 2.0/2.1 at
  the new `papers/<year>/<MM>_...` layout. Touched no shared library file.
- **Why:** The library became strictly project-neutral, and the old paper
  paths would 404 after its layout move.

### Add TabICL v2 as a second continued-pretraining family — Claude

- **What:** New `src/train/tabicl_{compat,model}.py`,
  `src/model/tabicl_models.py`, `tests/test_tabicl.py`. Family is detected
  from the checkpoint filename and drives the loader, losses (upstream's own
  CE and 999-quantile pinball), row cap, and save schema. The `use_lora` grid
  axis means freeze-backbone for TabICL — its own stage-3 regime, since full
  SFT collapsed it in two independent reports — tagged `_iclhead`. Grid is now
  48 trials/track.
- **Why:** Every result so far came from one architecture and one synthetic
  prior; a second family is the strongest answer to "is the PD null result
  architecture-specific?".
- **Verified:** 275 tests, including a no-mock `train_one_config` run on a
  tiny checkpoint for PD-freeze-backbone and LGD-full-FT.

### Fix six bugs found while wiring the second family — Claude

- **What:**
  - `resolve_max_rows_for_handle` never matched its untuned key → untuned
    models used the `default` eval row cap while trained ones used the
    architecture cap (an un-paired headline comparison).
  - `eval_*.slurm` fallback `--array` bound of 200 would silently drop ~75 PD
    tasks at the new grid size → 400.
  - `run_full_pipeline.sh` did not check `--list-trials`, so a failure
    expanded to `--array=0--1`.
  - `_decode_method_dirname` folded `__fullpass` into the base tag, averaging
    one_sample and full_pass results together in the notebooks.
  - ±inf reached the TabICL transformer, whose sklearn wrappers also reject an
    all-NaN column — both absorbed now.
  - Provenance did not record `n_estimators_finetune`.
- **Why:** The first two silently bias results rather than failing loudly.

---

## 30-07-2026

### Assess European DataWarehouse ABS data — Codex

- **What:** Researched EU/UK availability, PD/LGD/prepayment fields,
  reporting-history harmonisation and access terms, and mapped the panel data
  to our pipeline. No code changes.
- **Why:** EDW data are technically viable after leakage-safe panel-to-table
  ETL, but its January 2026 terms prohibit AI training unless the university
  agreement overrides that. *(The `docs/EDW_DATASET_FEASIBILITY.md` this
  entry claims was never committed — see `docs/AGENTS_MEMORY.md` §4.)*

---

## 13-07-2026

### Consume the TFM literature library as a git submodule — Claude

- **What:** Converted the papers/repositories/literature docs into the shared
  `TFM_Library` repo, mounted at `tfm-library/` (branch-tracking main), and
  updated every reference across README, docs/VSC.md, CLAUDE.md/AGENTS.md,
  `.gitignore`, src, config, docs and tests. Upstream renames:
  `LITERATURE`→`SUMMARIES`, `summary`→`SYNTHESIS`.
- **Why:** One canonical knowledge base shared across projects instead of
  per-project drift, pinned per-commit for reproducibility.

### Analyse run-4 and close the two eval coverage gaps — Claude

- **What:** All ~325 logs of the first clean run: pipeline fully green (64/64
  fresh BF16 trials). Science — PD discrimination unchanged by CPT (best
  +0.0004), LoRA a no-op at every LR, LGD NLL-vs-RMSE trade-off replicates
  (v3 full-FT up to −0.32 nats at +0.009 RMSE; v2.6 degrades on both). Fixes:
  eval walltime 2 h→5 h and v2.6 eval row cap 100k→50k (8 walltime-killed
  cells); noted one missing LGD cell, recoverable by an eval re-run.
- **Why:** This homogeneous sweep is the citable baseline experiment.

---

## 11-07-2026

### Analyse the PD eval logs (read-only) — Claude

- **What:** Parsed all 98 `eval_pd_*.log`; 97/98 complete with 5/5 folds and
  no failed cells. No trained config beat its untuned base. **No repository
  changes.**
- **Why:** Needed to know whether the run was usable before acting on it.

### Fix the stale-checkpoint contamination and the eval pool split — Claude

- **What:** From ~200 logs: (a) `clean_run.py` missed the `$VSC_DATA`
  fallback checkpoint dir, so 59/64 trials silently SKIPped onto stale FP16
  checkpoints — it now wipes both locations plus `.sentinels`; (b) the eval
  pool split used raw index parity, sending all of `lgd_lendingclub` to the
  slow A100 pool — replaced with a model-parity split
  (`--list-tasks --pools K --pool i`); (c) collapsed the three per-epoch log
  lines into one greppable line.
- **Why:** The "clean" rerun was contaminated by surviving checkpoints and a
  whole LGD dataset went unscored.
- **Verified:** 243+ tests pass.

---

## 10-07-2026

### Audit the repository and 183 VSC logs, then fix the confounds — Codex

- **What:** Documented the provisional PD/LGD findings and fixed monitor
  sampling, AMP overflow accounting (→ BF16), warning noise, manifest
  races/duplicates, partial-grid gating, exact eval sizing, and Slurm
  end-state logging. Introduced `AGENTS_MEMORY.md` plus the Claude/Codex
  instruction files.
- **Why:** The run was incomplete and its monitor deltas were confounded,
  while repeated warnings and sentinel/manifest races let incomplete
  experiments look successful.

### Review Codex's changes and repair the eval gate — Claude

- **What:** 4-way review (loop.py, run_log/manifest lock, pipelines,
  orchestration). Confirmed Codex's L2-SP claim against the Garg paper —
  λ=0.003 **is** Real-TabPFN's value. Then fixed: the eval gate was
  all-or-nothing (one diverged trial skipped a whole track) → now submits if
  ≥1 trial trained and loudly flags partial grids; guarded `fcntl.flock`
  `OSError` so a no-flock filesystem degrades to the thread lock; added
  `.gitattributes` (LF for `*.sh`/`*.slurm`); recorded the never-push rule.
- **Why:** A strict gate would never fire in a 4-LR sweep where divergence is
  an expected outcome, and the CRLF/flock issues were latent landmines.
