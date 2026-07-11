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
