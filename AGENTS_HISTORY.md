# Agent change history

## Purpose

This tracked file is the concise hand-off log between Claude and Codex. It lets
either agent see what the other changed, reviewed, or deliberately left alone,
and why, without reconstructing prior chat sessions. Both agents must read it
at the start of each repository task and append exactly one entry before ending
each user request or agent session, including read-only sessions.

Keep entries short but decision-complete. Do not copy command output, raw logs,
dataset contents, secrets, or information already explained in permanent docs.
Use `AGENTS_MEMORY.md` for transient experiment findings and known pitfalls;
promote stable facts to the normal documentation.

## Entry format

```text
### YYYY-MM-DD — Claude|Codex
- Changed/reviewed: ...
- Why: ...
```

Append new entries below in chronological order.

### 2026-07-10 — Codex
- Changed/reviewed: Audited the full repository/literature and 183 VSC logs;
  documented the provisional PD/LGD findings; fixed monitor sampling, AMP
  overflow accounting/BF16, warning noise, manifest races/duplicates, partial
  grid gating, exact eval sizing, Slurm end-state logging, and stale methodology
  docs; introduced shared `AGENTS_MEMORY.md` plus Claude/Codex instructions.
- Why: The run was incomplete and its monitor deltas were confounded, while
  repeated warnings and shared sentinel/manifest races obscured failures and
  allowed incomplete experiments to look successful.

### 2026-07-10 — Claude
- Changed/reviewed: Reviewed all of Codex's uncommitted changes (4-way: loop.py,
  run_log/manifest lock, train/eval pipeline, orchestration). Verified the
  monitor-seed fix, BF16+disabled-GradScaler, AMP counting, per-task sentinels,
  exact eval sizing, and the L2-SP/3e-7 Real-TabPFN claims (checked against the
  Garg paper — Codex was right, L2-SP λ=0.003 IS in Real-TabPFN). Then fixed:
  (1) the eval gate was all-or-nothing (`SUCCESS -ne EXPECTED` skipped a whole
  track if ONE trial diverged/failed/over-sized) → now submits if >=1 trained
  and loudly flags partial grids; (2) guarded `fcntl.flock` OSError so a
  no-flock filesystem degrades to the thread lock instead of aborting the
  manifest write; (3) added `.gitattributes` (LF for *.sh/*.slurm — Codex's
  edits had introduced CRLF); (4) deleted the now-superseded, un-ignored
  `CLAUDE.local.md` (its content lives in AGENTS_MEMORY.md); (5) recorded the
  never-push hard rule in AGENTS_MEMORY.md.
- Why: Codex's work was high-quality and mostly correct, but the strict eval
  gate would realistically never fire in a 4-LR sweep where divergence is an
  expected outcome, and the CRLF/flock/gitignore issues were latent landmines.

---

## 2026-07-11 - Claude (subagent): PD eval log analysis (read-only, no repository changes)

- Parsed all 98 eval_pd_*.log in Downloads/logs (j61424066 even array tasks 0-184,
  93 files; j61424067 odd tasks, 5 files a1-a9). 97/98 complete with 5/5 folds OK,
  0 failed cells, no tracebacks; a9 (catboost x credit_risk) still running at snapshot.
  Odd tasks 11-183 (87 pairs) not yet evaluated. No trained config beats its untuned
  base; v3 lr3e-05 fullpass full-wt collapsed (AUC 0.50); v3 fullpass full-wt at
  lr>=1e-06 degraded (likely stale FP16 checkpoints from Jul-9 run). Findings returned
  to orchestrator, not written to repo.

---

## 2026-07-11 - Claude

- Changed/reviewed: Analyzed all ~200 Jul-10/11 run logs (4-agent sweep). Found
  + fixed: (1) clean_run.py missed the $VSC_DATA fallback checkpoint dir ->
  59/64 trials silently SKIPped on stale Jul-9 FP16 checkpoints (now wipes both
  locations + .sentinels, dedupes resolved paths); (2) eval pool split by raw
  index parity sent ALL of lgd_lendingclub to the slow A100 pool -> new
  model-parity split (`--list-tasks --pools K --pool i` in eval_pipeline.py,
  used by the generated eval-submit script); (3) collapsed the 3 per-epoch log
  lines into ONE comprehensive greppable line (user request); (4) l2sp display
  %.4f -> %.3e (was rendering 0.0000). Confirmed staging checkpoints/trained
  is unwritable EVEN from wICE (mkdir denied; infra fix needed, not code).
  Tests: 243+ pass, exit 0.
- Why: The "clean" rerun was contaminated by surviving old checkpoints; the
  pool-split artifact left a whole LGD dataset unscored; per-epoch diagnostics
  were spread over 3 lines.

---

## 2026-07-13 - Claude

- Changed/reviewed: Converted CreditPFN to consume the shared TFM_Library
  repo (github.com/andreasgoethals/TFM_Library) as a git submodule at
  `tfm-library/` (branch-tracking main). Removed the now-duplicated papers/,
  repositories/, docs/{LITERATURE,summary,REPOSITORIES}.md and
  src/utils/refresh_repositories.py; updated every reference (README tree +
  4.6/4.7, VSC_GUIDE clone instructions now `--recursive` + `git submodule
  update --init`, CLAUDE.md/AGENTS.md pointer to tfm-library/AGENTS.md,
  .gitignore Wide.txt entry dropped, path citations across src/ + config +
  docs + tests). Doc renames upstream: LITERATURE->SUMMARIES,
  summary->SYNTHESIS.
- Why: One canonical knowledge base shared across projects instead of
  per-project drift; the library is pinned per-commit for reproducibility
  and bumped with `git submodule update --remote tfm-library`.

---

## 2026-07-13 - Claude (run-4 analysis)

- Changed/reviewed: Analyzed all ~325 logs of the first CLEAN run (4-agent
  sweep): pipeline fully green (64/64 fresh BF16 trials, gate/sentinels/pools
  all correct). Science: PD discrimination unchanged by CPT (best +0.0004);
  LoRA a no-op at every LR; LGD NLL-vs-RMSE trade-off REPLICATES (v3 full-FT
  up to -0.32 nats at +0.009 RMSE; v2.6 degrades on both). Fixes: eval
  walltime 2h->5h + v2.6 eval row cap 100k->50k (8 walltime-killed
  v2.6xalgorithmwatch cells); noted missing a48 LGD cell (one-off, recovered
  by an eval re-run).
- Why: The homogeneous sweep is the citable baseline experiment; the two
  eval fixes close the only coverage gaps.

---

### 2026-07-30 — Codex
- Changed/reviewed: Researched European DataWarehouse public ABS data and
  added `docs/EDW_DATASET_FEASIBILITY.md`; verified EU/UK availability,
  PD/LGD/prepayment fields, reporting-history harmonisation, access and
  current use terms, then mapped the panel data to CreditPFN's pipeline. No
  code changes.
- Why: EDW data are technically viable after leakage-safe panel-to-table ETL,
  but its January 2026 standard terms prohibit AI training unless the
  university agreement explicitly overrides that restriction.

## 2026-08-04 - Claude

- Changed/reviewed: Created `tfm-library/PROJECT_SPECIFIC.md` from the
  parent-folder handoff file (all CreditPFN-specific literature notes moved
  out of the now project-neutral library docs), filled in the library pin
  (221bac0, "Update Skeleton") in its header, and verified the library's
  gitignore keeps the file invisible (submodule status stays clean). Rule-6
  sweep: updated the 4 stale `../papers/YYYY_...` links in notebooks
  2.0/2.1 to the new `../tfm-library/papers/<year>/<MM>_...` layout
  (Garg 2025 -> 2025/07_, TabPFN-3 report -> 2026/05_); no other old-form
  references existed in tracked or agent files. Did NOT touch any shared
  library file (read-only contract respected).
- Why: The library became strictly project-neutral on 2026-08-04; the
  single sanctioned home for CreditPFN literature notes is the gitignored
  PROJECT_SPECIFIC.md, and old paper paths would 404 after the layout move.

---
