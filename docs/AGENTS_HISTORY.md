# Agent change history

Concise hand-off log between Claude and Codex: what changed, what was
reviewed, what was deliberately left alone, and why — so neither agent has to
reconstruct a prior chat session. Read it at the start of every repository
task; add an entry before ending every request or session, read-only ones
included.

## House style (follow exactly)

1. **One `##` heading per DAY** (`## YYYY-MM-DD`), newest at the bottom.
   Append to today's heading if it already exists — never open a second one.
2. **One `###` heading per CHANGE**, ending in `— Claude` or `— Codex`.
   Title it as a short imperative claim ("Correct the TabICL row caps"), not
   a topic label ("row caps").
3. **Each change gets exactly these bullets**, in this order. Omit a bullet
   only when it genuinely does not apply:
   - `**What:**` what changed, in 1–3 sentences.
   - `**Why:**` the reason, in 1 sentence.
   - `**Verified:**` how it was checked (test count, log inspected, etc.).
4. **Cap a change at ~8 lines.** Long bug lists become a nested bullet list
   of one line each. If it will not fit, it belongs in `docs/` and this entry
   should link to it.
5. **Facts, not narrative.** No command output, raw logs, dataset contents,
   secrets, or anything already in the permanent docs.
6. **Never rewrite an older day.** If a past entry became wrong, say so in
   today's entry and mark the old one superseded.
7. **Division of labour.** Transient findings and pitfalls →
   `docs/AGENTS_MEMORY.md` (gitignored). Durable facts → the committed docs
   alongside this file. This file records only *changes*.

---

## 2026-07-10

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

---

## 2026-07-11

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

## 2026-07-13

### Consume the TFM literature library as a git submodule — Claude

- **What:** Converted the papers/repositories/literature docs into the shared
  `TFM_Library` repo, mounted at `tfm-library/` (branch-tracking main), and
  updated every reference across README, VSC_GUIDE, CLAUDE.md/AGENTS.md,
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

## 2026-07-30

### Assess European DataWarehouse ABS data — Codex

- **What:** Researched EU/UK availability, PD/LGD/prepayment fields,
  reporting-history harmonisation and access terms, and mapped the panel data
  to our pipeline. No code changes.
- **Why:** EDW data are technically viable after leakage-safe panel-to-table
  ETL, but its January 2026 terms prohibit AI training unless the university
  agreement overrides that. *(The `docs/EDW_DATASET_FEASIBILITY.md` this
  entry claims was never committed — see `docs/AGENTS_MEMORY.md` §4.)*

---

## 2026-08-04

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

## 2026-08-05

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
  **Supersedes** the `tabicl>=2.1.1,<3` command in the 2026-08-04 entries.

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

- **What:** `AGENTS_HISTORY.md` → `docs/AGENTS_HISTORY.md` (via `git mv`, so
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
  The 2026-07-10 entry's mention of introducing `AGENTS_MEMORY.md` is left
  bare on purpose — it narrates where the file was then (rule 6).

### Rescope the temporal-split plan after auditing the corpus — Claude

- **What:** Measured that only 5 of 25 raw datasets carry a parseable date
  column, that `sanitize.py` drops them, and that **none of the 5 PD test
  sets** has one (only LGD `loss2` does). Rewrote that roadmap item into three
  scoped options: a `loss2`-only case study, re-pinning the corpus split, or
  re-sourcing fuller raw files.
- **Why:** It had been listed as a scheduling task when it is really a corpus
  decision, and would have been discovered only mid-implementation.

## 2026-08-06

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
