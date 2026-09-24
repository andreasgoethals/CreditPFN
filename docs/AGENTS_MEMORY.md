# Agents' memory — runs and dead ends

What is worth carrying between sessions: **the cluster runs that have been done**, and **the things
that turned out not to work**. Read it before starting; add to it as you go.

Not the changelog — that records edits to the repository. This records experience: what was run,
what came out, and what is already known to fail.

**Keep it short.** One line per run, four per dead end. Newest first, dates `DD-MM-YYYY`. Never
delete an entry: a run you would otherwise repeat and a dead end you already paid for are both
evidence.

Method and research context live in `RESEARCH_BRIEF.md`; operational/storage details and measured caps live in `VSC.md`. The runs table below retains historical headline measurements.

## Current handover — 24-09-2026 experiment folders and analysis notebooks

- Read Downloads in place: **93 files / 1,306,137 bytes**. Migration **62124946** moved 87 files on both tiers and exited 0; preflight **62124955** reports zero failures/warnings and exit 0. Corrected null audits **62124956 (PD)** and **62124957 (LGD)** each report eight complete, zero pending/diverged/problems, eight exact canonical saved-state matches, eight monitor matches and `passed: true`, exit 0. The 16 check3 controls are now audited; do not spend GPUs repeating them merely because configs moved. Two-update cost extrapolations are not production walltime evidence.
- User requested experiment 0 = debugging/pilots; 1 = main sweep; 2 = seed sensitivity; 3 = sampling/accumulation. Moved all 12 phase configs into those four directories and verified their parsed payloads are identical to the preceding HEAD. Shared data/train/eval defaults remain at the config root. Existing run names remain intact. Main/seed/sampling counts remain **512 / 32 additional / 96**; experiment 0 retains 16 null + 32 short + 8 budget pilots.
- Replaced six flat notebooks with **11 ordered notebooks** under `00_general/` and experiments 0–3. New reports cover corpus geometry/quality/partitions/exposure, operational gates, update-indexed trajectories, matched factor/seed/sampling effects, complete-fold benchmark comparisons, complementary metrics and cost. Figures use bounded pages, shared A4 style, PDF-only FigureSaver and final section-ordered text summaries. No result PDFs are fabricated for unavailable measurements. Runner discovery and figure/caption paths support nested notebook names.
- Added optional **CREDITPFN_ANALYSIS_ROOT**, naming the downloaded output directory itself, for read-only analysis without import. Only visualization readers use it; generated figures/summaries still go to the repository output. Normal cluster storage variables and checkpoint locations are unchanged.
- Validation: full suite **479 passed, 1 skipped** (optional on-disk manifests), nine known constant-input toy-regression warnings; **102 focused checks passed** after the final analysis corrections. All **11 notebooks passed**, producing **39 PDFs** from local corpus data and downloaded null-control evidence; unstarted phases retain planned coverage without fabricated result figures. The publication scan covered 61 artifacts with zero private-name matches. Actual notebook figures and synthetic full-grid heatmaps/trajectories were visually inspected at A4 width. All nine experiment notebooks passed a final serial rerun. Downloads' 93-file content digest is unchanged. No install, real model training, cluster submission, push or Downloads mutation by the agent.
- **Next operational phase:** positive-LR/recovery verification and short pilots under experiment 0, with fresh source-matched VSC plans. Main horizon/walltimes still require the pilots. Config moves do not rename completed controls.

## Previous handover — 24-09-2026 review fixes and output rename; CPU audits still required

- User-authorized template deviation: generated artifacts now use `output CreditPFN/` on DATA and project storage; original/trained weights remain in `checkpoints/`. The local tree was renamed. Downloads was already user-renamed and was read in place; no downloaded files were moved, copied or removed.
- Read the expanded Downloads copy: **89 files / 1,055,974 bytes**, including 27 logs and 16 check3 trajectory CSVs. Recomputed all 16 `[0, 2]` milestone/per-dataset parity checks successfully. All six downloaded plan checksums pass, including check3. The 16 GPU controls completed at user HEAD **eb6e016**, so the review's statement that no GPU work had run is stale. New CPU audits **62112998 (PD)** and **62113000 (LGD)** report 8 completed / 0 pending / 0 divergent each, with **all 16 per-dataset monitoring comparisons equal**, but exit 1 on raw serialized-state comparisons.
- Raw audit differences: v2 conversion changes serialized attention/MLP key names; LGD additionally changes the loss diagnostic `criterion.losses_per_bucket`, and v3 publishes criterion buffers derived from its model. The corrected audit compares canonical upstream-loaded states, excluding only the accumulated loss diagnostic while retaining exact weight/border checks. Upstream evidence: `load_model_criterion_config`, `_resolve_regression_borders`, `FullSupportBarDistribution.forward`; library pin **e5ce01614eebe520af303f2b5bfd212298eab2be**. Local TabPFN differs from the cluster version; corrected checks against actual trained weights remain a VSC gate.
- Independently checked the 20 review findings. Fixed identity scope, cache dependencies, empty divergence resume rows, stable packing, latest-fold retry semantics, preprocessing-before-hashing, terminal recovery cleanup, stderr/job-ID separation, dry readiness, analysis labels/grouping/scale/privacy/columns/timing/saves/missing sizes. Cosmetic logs now print configured prefetch and distinguish successful-update targets from the epoch safety rail. Shared metrics moved to `src/eval/metrics.py` so evaluation orchestration changes do not invalidate training.
- Two obsolete untracked local files were removed: `config/experiment2_pd.yaml` and `scripts/slurm/stage_to_project.slurm`. Compared with their last tracked versions, they only lacked the later sampling setting/referenced the retired launcher. No Downloads, archive, datasets or checkpoints were deleted.
- Validation so far: full suite **466 passed, 1 skipped** (optional on-disk manifests), followed by **163 passing focused checks** after final corrections. Nine pre-existing constant-input toy regression warnings in the full suite. All 15 current Bash/Slurm scripts parse; local migration preview reports no remaining legacy output directory. **6/6 notebooks passed, 75 PDFs**; both result notebooks reran after caption/interpretation cleanup. The publication scan checked **89 artifacts, zero private-name matches**; every notebook has a final stdout summary and no function/class definitions. Final review/privacy/visualization tests: **52 passed**. All five extracted metric functions are AST-identical to the previous implementations. No legacy output directory, root-level logs or notebook locks were created.
- **Next:** user commits/pushes, pulls on VSC, previews/applies `maintenance.slurm migrate-output` with CreditPFN writers stopped; verify exit 0. Rerun the two CPU `audit --config config/experiment0_{pd,lgd}.yaml --null` jobs against existing check3 checkpoints. Do not reprepare check3 under this source or launch the main grid. Retain its controls until corrected audits pass; then plan the next short positive-LR/recovery and pilot checks against settled source. No cluster job, install, push or real model training was performed by the agent.

## Previous handover — 23-09-2026 check3 GPU controls completed; checkpoint audits next

- Read three Downloads logs in place: **62112231** passes CPU preflight with zero failures/warnings; **62112290/62112294** write the PD/LGD **cpt_null_v4_check3** plans, eight trials each. All three report END exit 0 and the correct CreditPFN conda environment. No plan JSONs were supplied, so their fingerprints were not independently checked here; submission validates the prepared identity before allocating GPUs.
- Local source is user commit **eb6e016**. No code/config changes are needed, so do not rename the phase, prepare it again, restage inputs or repeat cleanup. This update only records the supplied evidence.
- User supplied completed Slurm accounting and Downloads/logs: **all 16 controls COMPLETED / 0:0**, each matched to one log, update 2, drift 0, matching displayed baseline/final metrics and a project-storage save message. PD arrays **11599282/11599283/11599284/11599285** took 67–79 s/task; LGD **11599304/11599305/11599306/11599307** took 45–49 s/task. All used source eb6e016, CreditPFN conda, B200 CUDA, GPFS inputs, workers 4 and the intended dataset split. No logged numerical/OOM/traceback failures; the sole warning is the handled unrelated virtualenv, once per job. The 25 supplied logs total 200,155 bytes (16 training logs: 173,081 bytes); read in place, never copied.
- **Next:** submit two short CPU maintenance audits with `--config config/experiment0_{pd,lgd}.yaml --null`. Require eight completed/zero pending, no problems, exact saved tensor equality and per-dataset monitor parity for both tracks. Logs only show rounded aggregate metrics and reported drift; checkpoint/trajectory files remain on VSC and were not independently inspected locally. Do not repeat controls or launch pilots before these audits. Both bounded login diagnostics and explicit main-terminal activation passed; the earlier shared-activator stall remains unexplained. Positive-LR recovery, 32 short pilots and eight budget pilots precede the 512/32/96 main/seed/sampling trials; their 5k horizon remains provisional.
- Diagnostic cleanup to address after checkpoint audits, before preparing pilots: the DataLoader log hardcodes `prefetch=4` while the actual loader receives configured 2; `Starting 2000 epochs` prints the safety ceiling despite the exact two-update stop. These labels do not change training, and source/plans stay fixed for the current audit.

## Previous handover — 23-09-2026 source cleanup and completed check2 preparation (superseded above)

- Inspected Downloads logs in place: **62111400 (PD)** and **62111404 (LGD)** both report `END exit_code=0`, eight null trials each, correct CreditPFN conda environment; 39/32 seconds respectively. These are preparation logs, not GPU training evidence. No plan files were supplied this time, so their fingerprints were not independently checked.
- Source cleanup is based on user HEAD `ca03eec`; library pin remains `e5ce01614eebe520af303f2b5bfd212298eab2be`. Shared training/grid and evaluation loaders now live in `src/train/config.py` and `src/eval/config.py`; capacity math lives in `src/train/capacity.py`. No reusable module imports scripts. Shared Slurm bodies remove PD/LGD duplication, all jobs use the same output/logs convention, and grid/family lookup errors stop jobs visibly.
- Fixed evaluation CLI precedence, scalar/empty/filtered-single grid handling and the capacity report's silent full-update fallback for frozen probes. Probe measurements remain synthetic forward/backward screens, excluding optimizer/L2-SP/eval overhead; measured caps and the scientific grid are unchanged. Removed completed L2-SP migration and old storage-move commands; retained historical readers and necessary upstream compatibility. Replaced stale ignored CLAUDE.local.md claims (exp1 running, serial workers, col_embedder.eval) with pointers to shared current documents.
- **Next:** user commits/pushes the final changes and pulls on VSC; run CPU preflight and prepare `experiment0_{pd,lgd}.yaml` as **cpt_null_v4_check3**. Source changes invalidate check2; preserve those tiny records but do not launch against them. No data restaging, downloads or full cleanup. After these CPU checks pass, run the 16 null controls, then positive-LR recovery and pilots before the main grid.
- A read-only local inspection now finds the previously blocked 12 duplicate imports/transcripts absent; this turn did not delete or move them. Downloads/archive/data/weights remain untouched. Notebook execution creates only its expected figures, caption metadata and summaries.
- Validation: **419 passed, 1 skipped**, nine known constant-input toy LGD warnings; skipped check needs optional on-disk data manifests. **6/6 notebooks, 75 PDFs**, zero private-name hits in extracted PDF text. Python syntax/import direction, 15 Bash files, 35 active documentation links, CLI help, both eight-control dry previews and git diff --check passed. Ruff is not installed; no lint pass is claimed. Local preflight passes its structural checks but reports the two absent local v2 weights across 12 phase checks; user VSC preflight already found all eight bases. Recheck on VSC after pulling. No pushes, installs, submitted cluster jobs or real model training by the agent.

## Previous handover — 23-09-2026 output workflow and confirmed CPU preflight (superseded above)

- User preference: cluster output stays on VSC during debugging. Read files in Downloads in place; never import them into local output without an explicit request. The user downloads the two complementary output trees for final local analysis. Notebook logs are unnecessary; the only locks are scheduler coordination on VSC. Figure-caption JSONs remain required for rebuilding CAPTIONS.md.
- Removed the notebook script-only path and duplicate transcript writer. All_Results.md now reads the final saved code cell's stdout. Clear every old code-cell output before execution so an early failure cannot preserve a previous successful summary; errors remain in the notebook. Figure regeneration never deletes debugging logs.
- User's 13:36 queue/accounting output showed no active wICE jobs and only the four earlier maintenance jobs. The pasted multi-command block had not submitted preflight/plans. Cause of the apparent stall is unconfirmed; a later single-command submission returned normally. Give commands in ordinary messages, one command per code block: the question UI flattens their formatting.
- User submitted **62110877**, CPU preflight, and supplied its complete log: **0 failures, 0 warnings, exit 0**, 13:38:31–13:39:17. All 25 processed tables and eight original checkpoints resolve on project storage. Correct CreditPFN conda environment selected after stripping an unrelated active virtualenv. No evaluation results directory yet is expected. No GPU controls were run.
- Next: finish/push the local edits as the user, pull on VSC, then prepare the two cpt_null_v4_check2 plans against the settled source. Do not repeat staging or full cleanup. No check2 preparation appears in the supplied accounting. GPU null parity, positive-LR interrupted recovery and pilot timing remain outstanding.
- Cleanup of 12 local duplicate files was rejected by automatic approval review as "blocked by policy"; they remain: three copied maintenance logs, six notebook transcripts and three JSONs under output/manifests/imported/2026-09-23. Downloads, archive, datasets and weights were untouched. The local preflight log is useful retained debugging evidence.
- Validation: **400 passed, 1 skipped**, nine known constant-input toy LGD warnings; skip is optional on-disk data manifests. **6/6 notebooks, 75 PDFs**, zero private-name hits in PDF text; both tracked-file privacy tests passed after regeneration. Both summaries rebuild byte-identically without transcripts, and figure folders contain PDFs only. PD/LGD null-launch previews show eight controls each and submit nothing. No new dependencies, pushes, agent-submitted cluster jobs or real training.

## Previous handover — 23-09-2026 downloaded preparation and launch audit (superseded above)

- Inspected the new Downloads output: stage job **62109541** reports 33 files / 1,246,187,891 bytes copied; prepare jobs **62109753/62109754** report eight null trials each. Both plan and all trial hashes validate; their source hash matches the LF-normalized `65ea866` tree. PD partition has 13 train / 4 held-out tables; LGD 6 / 2. Logs establish preparation, not successful GPU training; no fresh training/eval records were downloaded.
- Copied the three logs into local `output/logs/` and the two superseded plans into `output/manifests/imported/2026-09-23/`, with checksums/inspection report. Downloads originals are intact. Did not replace locally regenerated captions with a downloaded captions-only summary.
- Maintenance/classical/report jobs log activation failures and actual exit status under `output/logs/`; cleanup preserves only its own active log. Removed the obsolete default-grid launcher, cross-cluster sentinel gate and inactive exp2 config. Cluster reporting is now `python -m src.utils.cluster_report`. Figure folders contain PDFs only; metadata and transcripts use manifests/logs. Retained real probes, recovery checks and historical record readers.
- **Next VSC gate:** user commits/pushes locally and pulls on VSC, runs CPU preflight, then prepares `experiment0_{pd,lgd}.yaml` as **cpt_null_v4_check2**. Old plans are left intact; unchanged staged inputs need no recopy and full cleanup must not be repeated. New source fingerprints include shell code and normalize CRLF/LF. Submission refuses stale code/config/environment plans before allocating GPUs; full input verification uses maintenance `prepare --check`.
- Research design is unchanged: protocol 4, 512 main + 96 sampling + 32 seed checks; 16/32/8 null/short/budget controls. Five thousand successful updates remains provisional. GPU null parity, positive-LR interrupted recovery and measured pilot walltimes remain unverified for this revision; do not launch the large grid before those gates.
- The compact September archive remains (~66 MB); `unpruned-originals/` is now absent on disk. No cluster submissions, installs, pushes or real training were performed by the agent. Local preflight correctly detects the two v2 base files absent locally; the downloaded cluster staging/plan artifacts include all eight bases. VSC preflight must confirm the current canonical copy before GPU work.
- Validation: full suite **397 passed, 1 skipped**, then **60 targeted checks passed** after the final source-cache, figure-cleanup and storage-resolution refinements. The skip is optional on-disk data manifests; nine warnings are constant-input toy LGD metrics. **6/6 notebooks, 75 PDFs**, no private-name hits in PDF text or tracked/new source, Python/Bash syntax and eight launcher previews passed. No VSC jobs were submitted. Ruff is absent from the local environment; no installation attempted.

## Previous handover — 23-09-2026 local archive and exhaustive passes (superseded above)

- User confirmed wICE cleanup **62109053 COMPLETED / 0:0 / 1m54s**: 6,394 files / 52.79 GB reported removed from both output trees, both trained trees and submission state. Raw/processed data and original bases remain. The unrelated `creditic` job was not touched. Next: pull these changes, stage inputs, prepare and dry-check null controls, then validate them before pilots/main work.
- User requested `archive/run-september-2026/` inside the repository and gitignored. Consolidate the available older DATA download plus 579 project files; preserve small source records and history, reduce repetitive logs, clear active local output. This is a one-off organization, not a new maintained archive subsystem. The DATA download is not claimed to be the last cluster snapshot.
- Protocol **4**, fresh `cpt_*_v4` names: full_pass/accumulate now use the same disjoint, nearly equal row partitions per table/epoch, with shuffled membership/order next epoch and worker-local caching. Reject balanced PD batches for exhaustive passes. Main/null/pilots/seed PD keep balanced one_sample; all sampling-study modes use proportional PD sampling to isolate pass mode within that study.
- **512 main + 96 sampling + 32 additional seed = 640 research trials**. Seed 43 repeats one predetermined full-update recipe (3e-7, lambda .003, one_sample); seed 42 is already in main. Two seeds give a limited paired check, not grid-wide robustness. Null/short/budget phases add 16/32/8, totaling 696 if each phase runs once. The 5k budget remains provisional; keep main/seed/sampling horizons matched after the long pilot decision.
- V2's actual configured cap is **10k**, not the runbook's stale 14k; retained the lower cap backed by the recorded OOM. No installs, pushes, real training or cluster submissions performed by the agent.
- Compact archive verified: **65.93 MB**, eight merged tables, original small records, checksums and bounded summaries for 1,485 logs, from 5,403 available local source files / 7.166 GB. **Bulk deletion was rejected by automatic approval review ("blocked by policy").** No originals deleted; moved them into `archive/run-september-2026/unpruned-originals/` and verified their sizes/mtimes. That folder still occupies 7.166 GB and can be deleted manually; active `output/` is separate. The compact archive is complete but local space reclamation is not.
- Plan preview exposed repeated full-CSV reads per corpus resolution. Added bounded metadata-only caching keyed by path/stat/schema hints, plus per-filter split reuse within plan preparation; no full frames retained in this cache. The twelve real phase/track previews completed in 10.91 s with exactly 25 CSV reads (one per table); counts 16 null + 32 short + 8 budget + 512 main + 32 seeds + 96 sampling. **Final validation: 387 passed, 1 skipped** (optional data manifests absent), nine known constant-input toy-regression warnings. All six notebooks executed successfully and regenerated 75 PDFs; PDF text and all 40 changed/new files had zero private-name matches. Bash syntax passed for 15 scripts; null/seed/sampling launcher previews submitted no jobs. Archive hashes and all 5,403 relocated-source stats verified. CUDA controls/recovery and pilot timing remain VSC gates.

## Previous handover — 23-09-2026 fresh-start simplification (superseded above)

- Current campaign: **512 main + 96 sampling = 608 research trials**, all train seed 42. Removed the two seed-check configs; accumulation remains one of the three sampling-study modes. Null/short/budget pilots add 16/32/8 trials, giving 664 if every phase runs once, before optional profiling/canaries.
- **5,000 remains provisional.** Garg uses 20k CPT updates; Kolberg chooses 10k after monitoring his different synthetic adaptation task; Rubachev uses target-table validation stopping. None validates 5k for this corpus. Use the existing eight 20k reference pilots to review 5k/10k/20k trajectories and cost before fixing the main/sampling budget. Audit now estimates all three horizons.
- User wants optional historical evidence in a **local folder**, followed by empty VSC output and removal of old trained weights. No old output is required to run the new campaign. Removed custom archive tools; the existing cleaner now covers both whole output trees, both trained-weight trees/recovery states and submission state, while preserving raw/processed data and original bases. Stop writers and verify any wanted download before cleanup. No real output or weights were deleted by the agent.
- Direct project-output download and cleanup commands are in VSC.md. RESEARCH_BRIEF.md replaces the paper-roadmap filename. This supersedes the archive/seed instructions in the dated handover below; the historical entry is retained as evidence.
- Log diagnosis: three sampled large local files each contained **53,130** identical scikit-learn FutureWarnings plus thousands of nonfinite-loss warnings. Existing targeted warning filtering covers parent/workers; added first-warning retention, bounded repetition summaries and final segment counts. Epoch skip counters and fatal tracebacks remain intact.
- Starting checkout was user commit `53418cf`; no pushes, installs or cluster jobs performed. **Validation: 376 passed, 1 skipped** (optional data manifests absent), nine known constant-input toy-regression warnings; 6/6 notebooks and 75 PDFs regenerated, with zero private-name hits in PDF text or the new brief. Privacy tests, Bash syntax, actual main/sampling plan previews and `git diff --check` passed. Read-only local cleanup preview found 4,912 files / 7.16 GB; nothing was deleted. CUDA controls/recovery and the budget decision still require VSC pilots.

## Previous handover — 22-09-2026 descriptive redesign (superseded above)

- User retained **25 tables (17 PD + 8 LGD)** and clarified the goal: describe continued-pretraining behavior, including negative/flat results; do not turn the grid into a champion-selection claim.
- Protocol **3**: `cpt_main_v3` = 512 trials across tracks; four fixed dataset folds, partition seed 1729, train seed 42, LR {3e-7,1e-6,1e-5,3e-5}, lambda {0,.003}, full/frozen updates, equal-table `one_sample` sampling.
- `cpt_seeds_v3` adds 128 predetermined reference-recipe repetitions with seeds 43/44. `cpt_sampling_v3` is a separate 96-trial full-update study of one_sample/full_pass/accumulate, including a repeated control.
- Budget remains **provisionally 5,000 successful updates**, trajectories 0/250/1k/2.5k/5k. The 16 null trials, 32 short endpoint pilots and eight 20k reference pilots precede the main launch. Main/seed/sampling plans should be prepared only after the budget decision.
- Exact recovery includes optimizer, schedule, scaler, training RNG and dataset cursor. CPU dropout tests compare uninterrupted/resumed weights in all three sampling modes. Slurm supports short segments, warning forwarding and opt-in requeue; real CUDA recovery/null checks remain VSC gates.
- Launcher defaults: one trial/task, four preprocessing workers, per-array concurrency four and a shared cap of 16 per controller. Classical evaluation has a CPU path; reusable controls are fingerprinted. Scratch input copies are content-addressed; large weights cannot silently fall back to DATA through the modern launcher.
- User-reported cluster state: no jobs running; HEAD `cc42445`. DATA output 7.1 GB/4,913 files, mostly logs; manifests 82 MB; fallback weights 2.6 GB. Project weights 41 GB/412 files, not a verified completed-trial count. No VSC jobs, pushes, installs or raw experiment record/weight deletions were performed by Codex; derived local figures were regenerated.
- The 338 retagged L2-SP survivors remain historical. Do not reuse them as protocol-3 trials. `archive_experiment` creates a verified compact evidence archive without weights, then separately previews/prunes archived shards or retires indexed weights. Retirement cannot be undone from this compact archive.
- Earlier local exp1 consolidation preserved 1,451 attempts and 117,380 epoch rows: 708 CSVs/about 69 MB became eight tables plus inventory/about 28 MB. Source files and real weights remain intact.
- Documentation is intentionally consolidated: README, AGENTS, TEMPLATE, this memory, CHANGELOG, VSC, LITERATURE and PAPER_ROADMAP. The roadmap is the uploadable research/method brief; VSC owns output/migration/caps. No separate METHOD/RESULTS/OUTPUT/research-review documents remain.
- Final local validation: **376 passed, 1 skipped** (optional on-disk manifests absent), with nine constant-input warnings from toy regression tests. All six notebooks executed and produced 75 PDFs; extracted PDF text and tracked/untracked publication outputs passed the private-name scan. Bash syntax, launcher previews, signal/exit-status checks and `git diff --check` passed. A synthetic trajectory plot was visually inspected. This validates the local implementation, not CUDA throughput or cluster recovery; those remain null/pilot gates.

## Runs

One row per cluster run worth remembering — which is most of them, because *"have we already tried
that configuration?"* is the question this table exists to answer.

| Date | Run | Outcome | Notes |
|---|---|---|---|
| 24-09-2026 | corrected check3 null audit LGD · wICE 62124957 | **passed (download evidence)** | 8 complete; all exact saved-state and monitor checks pass; 0 problems, exit 0. |
| 24-09-2026 | corrected check3 null audit PD · wICE 62124956 | **passed (download evidence)** | 8 complete; all exact saved-state and monitor checks pass; 0 problems, exit 0. |
| 24-09-2026 | CPU preflight · wICE 62124955 | **passed (download evidence)** | 0 failures / 0 warnings; exit 0. |
| 24-09-2026 | output migration · wICE 62124946 | **done (download evidence)** | 87 files moved to `output CreditPFN/` across DATA/project; exit 0. |
| 23-09-2026 | null audit LGD · wICE 62113000 | **failed comparison (download log)** | 8 completed, 0 pending/divergent; all monitor pairs equal. Raw v2 key conversion, diagnostic-loss buffer and v3 criterion initialization differences; canonical comparison rerun required. |
| 23-09-2026 | null audit PD · wICE 62112998 | **failed comparison (download log)** | 8 completed, 0 pending/divergent; all monitor pairs equal. Only v2 raw key-set conversion flagged; canonical comparison rerun required. |
| 23-09-2026 | cpt_null_v4_check3 LGD · Mindwell 11599304/11599305/11599306/11599307 | **8/8 completed (accounting + logs)** | 0:0; 45–49 s/task; update 2, drift 0, displayed monitors unchanged. Initially all pending. Saved tensor/per-dataset parity audit outstanding. |
| 23-09-2026 | cpt_null_v4_check3 PD · Mindwell 11599282/11599283/11599284/11599285 | **8/8 completed (accounting + logs)** | 0:0; 67–79 s/task; update 2, drift 0, displayed monitors unchanged. Initially five running/three pending. Saved tensor/per-dataset parity audit outstanding. |
| 23-09-2026 | prepare cpt_null_v4_check3 LGD · wICE 62112294 | **plan written (download log)** | Eight trials, one partition, two held-out tables; END exit 0 in 28 s. GPU controls not yet evidenced. |
| 23-09-2026 | prepare cpt_null_v4_check3 PD · wICE 62112290 | **plan written (download log)** | Eight trials, one partition, four held-out tables; END exit 0 in 37 s. GPU controls not yet evidenced. |
| 23-09-2026 | CPU preflight · wICE 62112231 | **passed (download log)** | 0 failures, 0 warnings; END exit 0 in 42 s. All 25 tables/eight bases and CPU checks pass; GPU validation remains outstanding. |
| 23-09-2026 | prepare cpt_null_v4_check2 PD · wICE 62111400 | **plan written (download log)** | Eight trials, one partition, four held-out tables; END exit 0 in 39 s. Superseded by source cleanup/check3 before GPU training. |
| 23-09-2026 | prepare cpt_null_v4_check2 LGD · wICE 62111404 | **plan written (download log)** | Eight trials, one partition, two held-out tables; END exit 0 in 32 s. Superseded by source cleanup/check3 before GPU training. |
| 23-09-2026 | CPU preflight · wICE 62110877 | **passed (user log)** | 0 failures, 0 warnings, END exit_code=0; 46 seconds from START/END. All 25 tables/eight bases and CPU configuration checks pass; GPU controls remain outstanding. |
| 23-09-2026 | prepare cpt_null_v4 PD · wICE 62109753 | **plan written (download evidence)** | Eight identities; 13 train / 4 held-out tables; checksums/source verified against 65ea866. Superseded by check2 before GPU training. |
| 23-09-2026 | prepare cpt_null_v4 LGD · wICE 62109754 | **plan written (download evidence)** | Eight identities; 6 train / 2 held-out tables; checksums/source verified against 65ea866. Superseded by check2 before GPU training. |
| 23-09-2026 | stage inputs · wICE 62109541 | **copy reported successful (download evidence)** | 25 processed tables + eight original bases, 1.246 GB; immutable GPFS pointer published. No new GPU results in this download. |
| 23-09-2026 | maintenance clean · wICE job 62109053 | **done (user evidence)** | Exit 0:0, 1m54s; deletion report 6,394 files / 52.79 GB across old outputs, trained weights and sentinels. Inputs/original bases preserved. |
| 22-09-2026 | exp1 · λ-sweep bug found in the 29-08 run (PD ~78 % of the base×lr×frozen×pass grid trained; no LGD; no eval) | **λ axis INVALID** | `run()` never forwarded the swept `l2sp_lambda` → every trial trained at the config default 0.003; the two "arms" overwrote one untagged checkpoint, and resume was broken so the grid re-ran on every resubmit. Fixed 22-09 (CHANGELOG + dead end below). Salvage: surviving ckpts are all valid λ=0.003 — rerun the λ=0 arms + remaining PD + all LGD. |
| 29-08-2026 | exp1_pd · 96 trials/split × 8 splits, Option-B grid (lr{3e-7,1e-6,1e-5} × l2sp{0,0.003} × frozen{F,T} × pass{full,acc}), 4 bases; training only (eval not yet submitted) | **partial — frozen TabPFN arm lost** | Full-FT TabPFN + all TabICLv2 (both arms) trained OK; **all 168 frozen TabPFN `_lora` trials died in ~4 s with `NameError: ckpt_path`** (`load_tabpfn_for_training`). Splits 6–7 double-submitted by the SPLIT_START recovery (harmless). Results write to `/lustre1/…/stg_00211/…/results` (staging), not the `output/` download. Bug fixed 31-08; frozen TabPFN needs re-running. See dead end below. |
| 12-08-2026 | run-8 · 16 trials/track, 20 000 steps, `min_train_rows` [0, 5000], adapter arm TabICLv2-only, eval packed into 16 tasks; **eval completed 16-08-2026 on Mindwell `gpu_b200`** | **done — first complete run** | Training 31/32 OK (1 false-positive divergence abort). Eval 105/105 PD + 44/44 LGD cells, 745/745 folds, zero failures. **PD 20/75 paired wins, mean -0.0013, p=0.78 (null). LGD 0/32.** Untuned v3 beats best tuned GBM on 4/5 PD and 2/2 LGD. Completing the eval REVERSED the half-eval's -0.0048 'damage' finding. `AGENTS_MEMORY.md` |
| 10-08-2026 | run-7 · 36 trials/track, 3 bases, `target_total_steps` 9100, task-stride eval pools | **partial** | Training perfect: 72/72 OK, 90 GPU-h in 5.1 h wall-clock at 15-21 concurrent GPUs. Eval incomplete and slow: 0.73 average concurrency, 44 % dead time. PD paired trained-vs-untuned 17/39 wins, TabICLv2 full-FT +0.016 mean; **LGD 0/18 wins**. LGD ran only 800-3200 steps of the 9100 target. `AGENTS_MEMORY.md` |
| 07-08-2026 | run-6 · 36 trials/track, 3 bases, 100 epochs, `target_total_steps` 9100 | **done** | First fully green run: 36/36 train + 84/84 eval cells, drained in 7.1 h. Best PD mAUC 0.7620 (v3 1e-6 LoRA), best LGD RMSE 0.1335 (v3 1e-6 full). Half the eval pool never logged, so trained-vs-untuned is not computable for v3/TabICLv2. 54.9 GPU-h. `AGENTS_MEMORY.md` |
| 05-08-2026 | run-5 · 48 trials/track, first two-family run (TabICLv2 added) | **partial** | 80/96 trials OK; the 16 `_iclhead` trials crashed (freeze-via-`.eval()`, see dead ends 06-08-2026). Eval never ran — the 21 h gate expired — so every number is a 2 000-row monitor eval. Drift 0.02 % of ‖w₀‖ at 3e-7: the PD null was undertraining. 43.7 GPU-h. |
| 05-08-2026 | probe · `probe_row_cap.slurm` j11509346, all three bases on B200 | **done** | The measurement the row caps come from: v3 2.49 GB/1k rows, v2.6 5.72, TabICLv2 0.51 per member. TabICLv2's ceiling is a cuDNN fused-attention failure between 26k and 40k, not memory. `RESEARCH_BRIEF.md` §3 |
| 11-07-2026 | run-4 · 64 trials/track, TabPFN v3 + v2.6, 50 epochs | **done** | The clean homogeneous sweep and still the reference for cross-version science. 64/64 trained, 63 checkpoints straight to staging. PD: continued pretraining ≈ zero effect on discrimination (best Δ +0.0004). LGD: NLL improves while RMSE worsens. 8 PD eval cells walltime-killed. `AGENTS_MEMORY.md` |
| 10-07-2026 | rerun after `clean_run` · 64 trials/track | **contaminated** | 59/64 trials SKIPped on stale 09-07 FP16 checkpoints in the `$VSC_DATA` fallback dir. Only PD v3 a0–a4 actually retrained. Do not cite any number from this run. |
| 09-07-2026 | run-3 · 64 trials/track, first BF16 run | **partial** | LGD 32/32; PD 27/32 (tasks 0–4 ended together without a traceback, consistent with external termination). Monitor deltas invalid — the monitor re-seeded every epoch. |
| 08-07-2026 | probe · `probe_row_cap.py` on B200, v3 + v2.6 | **done** | First real memory measurement; replaced the fictional 100k/30k caps. v3 ≈ 2.5 GB/1k rows, v2.6 ≈ 5.7. Per-step cost is × `n_estimators`. |
| 04-07-2026 | run-2 · 32 PD + 32 LGD trials | **crashed** | Every trial died: PD at its first checkpoint save (staging readable but not writable from Mindwell), LGD on `bar_distribution` moving in tabpfn 8.x. Both fixed; see dead ends. |
| 03-07-2026 | run-1 · first full sweep attempt | **crashed** | 0 usable trials. The run that produced the writability probe, the import compat layer, and the preflight smoke tests. |

## Dead ends

### 24-09-2026 — Windows notebook-kernel shutdown diagnostic

**Tried:** Execute all 11 notebooks with concurrent kernels during local validation.
**Result:** All saved notebooks passed and the runner exited 0, but one native ZeroMQ socket assertion appeared during shutdown.
**Why:** The shutdown cause is unconfirmed; saved notebooks contained no error/stderr cells and all final summaries were present.
**Instead:** Inspect saved outputs and rerun all nine experiment reports serially: 9/9 passed without the diagnostic. No warning suppression or dependency changes.

### 24-09-2026 — raw serialized keys are not null-control inference state

**Tried:** Compared upstream original `state_dict` keys/tensors directly with saved continued-pretraining files.
**Result:** Both check3 CPU audits failed after 16 completed GPU controls; all monitoring comparisons were equal.
**Why:** The upstream loader converts v2 keys, v3 can reconstruct criterion borders, and criterion forward accumulates a diagnostic loss buffer independently of learning rate.
**Instead:** Load both checkpoints through the same upstream loader, require exact model/inference-buffer equality and monitor parity, and exclude only the verified loss accumulator. Reaudit existing weights before spending more GPU credits.

### 24-09-2026 — recovery test must exercise numerical failure, not an impossible budget

**Tried:** Extended the interrupted-recovery test with an epoch cap too short to reach its configured budget.
**Result:** The intended terminal-divergence test failed at the existing pretraining capacity guard.
**Why:** An impossible configuration is correctly rejected before training, so it creates no terminal checkpoint or recovery state to inspect.
**Instead:** Inject rejected numerical updates after three successful toy updates; compare interrupted/uninterrupted outcomes and verify recovery removal after terminal publication.

### 23-09-2026 — silent interactive activation stall

- **Tried.** Sourced the shared activator with scratch disabled on tier2-p-login-1 after successful CPU preflight/preparation jobs.
- **Result.** User interrupted with Ctrl+C after the unrelated-virtualenv warning; no Active conda env confirmation appeared. Prompt timestamps span 14:48 to 15:08, but do not independently time execution. Subsequent bounded checks passed: all four package imports and explicit conda.sh/activation, correct CreditPFN prefix/interpreter, both exit 0.
- **Why.** Unconfirmed. Conda hook/activation and the captured numpy/torch/omegaconf/tabpfn import check occur before confirmation and provide no intermediate progress.
- **Instead.** Use the tested explicit conda.sh/activation route in the main terminal, verify its Python path, then launch the prepared null controls. The temporary diagnostic shell did not activate its parent. Retain the original stall as unexplained; avoid installs or source/plan changes unsupported by evidence.

### 23-09-2026 — capacity report silently changed the requested adaptation

- **Tried.** Audited the report's full/frozen capacity comparison against the probe signatures.
- **Result.** The frozen keyword was unsupported; catching TypeError retried every call as a default full-update probe. The report also reused PD member counts for LGD.
- **Why.** A compatibility fallback hid an interface mismatch and changed the measurement being labeled.
- **Instead.** Shared capacity helpers explicitly accept freezing, use the training member resolver, report BF16/query settings, release OOM graphs and propagate unexpected errors. Test dispatch without consuming GPUs; keep the existing caps until real pilots.

### 23-09-2026 — evaluation settings overrode explicit CLI choices

- **Tried.** Traced phase/config loading through planning, training and evaluation.
- **Result.** Evaluation merged the phase block after CLI settings, silently undoing an explicit prediction-output override.
- **Why.** Config loading was duplicated between scripts, with different precedence.
- **Instead.** Share training config loading and merge evaluation CLI settings last. Regression tests preserve partitions and confirm unrelated phase/default values survive.

### 23-09-2026 — new override test ignored planned training seeds

- **Tried.** Used seed=123 as a generic CLI override in the new evaluation-loader test.
- **Result.** The full suite's new fixture failed while the other checks passed; the partition correctly selected seed 42.
- **Why.** apply_split_index selects experiment.training_seeds for a prepared campaign; the fixture also initially assumed a phase-owned CV block that actually comes from eval.yaml.
- **Instead.** Test a plain training knob for merge precedence and separately assert the planned seed. Changing campaign seeds requires the explicit seed list and a fresh plan.

### 23-09-2026 — local debugging imports and duplicate transcripts

- **Tried.** Copied downloaded cluster preparation evidence into local output and persisted notebook stdout in separate logs.
- **Result.** Unwanted local artifacts during cluster debugging. A later path-checked deletion of the 12 duplicates was rejected as "blocked by policy"; files remain.
- **Why.** Inspection was treated as import, contrary to the user's desired final-download workflow; notebook stdout already lives in the executed notebook. The deletion review supplied no more specific reason.
- **Instead.** Inspect Downloads in place, keep cluster evidence on VSC, derive summaries from notebooks, and let the user remove the exact duplicate files manually. Do not bypass the rejection with another deletion mechanism.

### 23-09-2026 — local preflight crashed before showing its report

- **Tried.** Ran the existing full repository preflight before changing the launcher.
- **Result.** `FileNotFoundError` from `subprocess.run(["bash", ...])`; readiness findings were not printed.
- **Why.** Git Bash existed but was not on Windows PATH; the checker also retained old grid, packing and corpus assumptions.
- **Instead.** Resolve installed Git Bash, report a missing shell as a failed check, reuse the training grid and inspect actual partition sizes. Keep GPU controls separate from CPU readiness.


### 23-09-2026 — bulk local archive cleanup blocked

- **Tried.** Verified archive hashes/source stats, then requested one path-checked bulk deletion of old local output and the duplicate download.
- **Result.** Automatic approval review rejected the command as "blocked by policy"; nothing was deleted.
- **Why.** No more specific rejection reason was supplied; this was an execution-policy block, not a failed archive-integrity check.
- **Instead.** Moved originals reversibly into the ignored archive's `unpruned-originals/`, verified every relocated source, and left manual pruning to the user. Keep the 66 MB compact archive.

### 23-09-2026 — plan previews repeatedly parsed the corpus

- **Tried.** Previewed the null/pilot/main/sampling/seed plans against all real local tables.
- **Result.** Repeated CSV parsing dominated CPU time; every split resolution rebuilt schema/counts, and written plans repeated that work per trial.
- **Why.** The corpus reader's "once per pipeline invocation" comment had no cache behind it; plan generation did not pass its already-resolved split to identity construction.
- **Instead.** Cache only schema/count metadata with file/schema invalidation, reuse one split per size-filter setting, and test both invalidation and mixed-filter plan correctness.

### 23-09-2026 — full-pass names did not imply row coverage

- **Tried.** Compared the user-requested non-overlapping passes with the existing full_pass/accumulate loader.
- **Result.** It drew independent samples, so some rows repeated and others were absent within an epoch; class-balanced draws were incompatible with exhaustive natural-prevalence coverage.
- **Why.** Batch counts were size-proportional, but batches did not share a row partition.
- **Instead.** Partition rows once per table/epoch, reuse across chunks/workers, and use proportional PD sampling for every arm of the separate pass-mode comparison. Preserve balanced sampling in the main grid and its seed check.

### 23-09-2026 — overbuilding historical retention

- **Tried.** Added a custom verified archive, pruning and checkpoint-retirement lifecycle.
- **Result.** More infrastructure than the user's fresh-run workflow needs; no previous output is an input to the new experiment.
- **Why.** Historical debugging evidence and active experiment dependencies were treated as though both needed permanent cluster storage.
- **Instead.** Optional manual download to a local folder, retain the existing written history, then use the established full cleaner. Remove the archive-specific code/tests.

### 23-09-2026 — fresh cleanup missed new project output

- **Tried.** Checked the existing full cleaner against the new storage layout before recommending a restart.
- **Result.** It covered project results but left compact snapshots/caches/archives; train-only cleanup also missed some recovery/provenance files.
- **Why.** The cleaner still assumed results were the only project-tier output subtree.
- **Instead.** Cover the whole project output tree and both trained trees; test input preservation and preflight all roots against links before deletion.

### 22-09-2026 — exact-step trajectories exposed old epoch assumptions

- **Tried.** Added update-indexed monitors and interruption recovery to the epoch-based loop.
- **Result.** Legacy timing counted monitor work as data wait; final drift depended on an epoch-only field, and an audit initially looked for a doubled checkpoint extension.
- **Why.** Epoch cadence, naming and completion fields had been reused for a different control clock.
- **Instead.** Record successful updates, separate monitor/training/compute/wait times, use measured trajectory drift, and integration-test real trial filenames plus uninterrupted/resumed model equality.

### 22-09-2026 — a per-array throttle does not bound a campaign

- **Tried.** Reviewed submitting multiple base/track/split arrays with the same percentage limit.
- **Result.** Every array owned its own limit; their combined concurrency was unbounded by that number.
- **Why.** Slurm array percentages are not a user-wide campaign semaphore.
- **Instead.** Use a persistent locked per-controller lane pool with afterany dependencies; fail closed on an uncertain submission response. Direct sbatch and other controllers remain outside that pool.


### 22-09-2026 — reconsolidation after removing raw epoch shards

- **Tried.** Reviewed and regression-tested archive → prune → consolidate again on temporary data.
- **Result.** Rebuilding from only the surviving root manifests could replace the latest training table with an empty one.
- **Why.** Pruned CSVs are deliberately absent, so a raw-only snapshot builder cannot distinguish archived histories from no history.
- **Instead.** Refuse publication when any previous source is absent, preserve LATEST, and require verified restoration before reconsolidating that run. New run names remain independent.

### 22-09-2026 — notebook checks exposed hidden metadata and plotting dependencies

- **Tried.** Executed all six notebooks against the actual downloaded output after the focused training-notebook checks.
- **Result.** Exploration/results still required absent `manifest_pd/lgd.csv`; the new grid plot imported unavailable seaborn.
- **Why.** Dataset manifests had become optional for training but not exploration; the grid added an unnecessary dependency.
- **Instead.** Reconstruct optional metadata in memory with the existing registration calculation and draw the grid using installed matplotlib. Keep unknown raw statistics explicit; install nothing.

### 22-09-2026 — executed notebook outputs still exposed private names

- **Tried.** Re-ran the full tests after executing the notebooks rather than only checking their source.
- **Result.** Functional checks passed, but the privacy guard found a processed-data display and raw-data summary that still emitted original identifiers.
- **Why.** Those display paths bypassed the shared name accessor; clearing old outputs alone could not fix regeneration.
- **Instead.** Apply the accessor at both display boundaries and in regime-plot joins, then regenerate outputs and rerun the privacy guard.

### 22-09-2026 — persistent workers and shuffled accumulation changed the experiment

- **Tried.** Compared serial and persistent-worker sampling across epochs, and inspected actual dataset IDs within optimizer windows.
- **Result.** Workers repeated epoch-zero samples; shuffled microbatches mixed datasets inside accumulation windows, with summed rather than averaged gradients.
- **Why.** Worker dataset copies did not receive parent `set_epoch`; boundary flags described the unshuffled plan; the divisor used the unrelated accumulation setting.
- **Instead.** Transport `(epoch,index)`, shuffle whole dataset groups, average finite microbatches before clipping, and record protocol 2 under new run names. Tests use the production loader/sampler.

### 22-09-2026 — assuming Parquet support in the local environment

- **Tried.** Published the compact output prototype as Parquet.
- **Result.** Local tests and publication failed because neither pyarrow nor fastparquet is installed; no snapshot pointer was published.
- **Why.** pandas does not include a Parquet engine. Package installation was outside the authorized scope.
- **Instead.** Use portable `.csv.gz` tables with round-trip float parsing and SHA-256 verification. Empty `.building-*` directories from failed attempts are unpublished, not usable snapshots.

### 22-09-2026 — snapshot loaders ignored custom input roots

- **Tried.** Ran the full suite after publishing the local historical exp1 snapshot.
- **Result.** Four visualization tests read the real exp1 snapshot instead of their temporary fixture data; two other tests expected obsolete adaptation/divergence semantics.
- **Why.** The loader checked default raw roots even when a caller had selected another root. Earlier tests passed only while no snapshot existed.
- **Instead.** Pass the actual source roots to snapshot freshness checks; retain explicit tests with a real snapshot present, and update expectations for frozen-backbone provenance and absent drift evidence.


Anything that cost more than a couple of minutes and did not work — including what was eventually
fixed, because the fix is one changelog line and the dead end was the hour.

### 22-09-2026 — the L2-SP sweep silently never varied λ

- **Tried.** exp1's `l2sp_lambdas: [0.0, 0.003]` axis, expecting two anchor strengths per cell.
- **Result.** Every trial trained at the config default 0.003; the "λ=0" arm never ran at 0. Both
  arms wrote the same untagged checkpoint (one overwrote the other), and the resume-skip check —
  built from the *tagged* name it never found — never fired, so every resubmit re-ran the whole grid.
- **Why.** `run()` built the swept λ into the plan and the epoch-CSV/skip name but omitted
  `l2sp_lambda=` from the `train_one_config(...)` call, and the save-path `descriptive_name(...)`
  omitted the `_l2sp` tag. Two independent omissions that masked each other (both arms = 0.003, so
  the collision looked like a duplicate rather than a lost arm).
- **Instead.** Forward the swept λ into training AND into the checkpoint name; record the swept
  value (not `optimizer.l2sp_lambda`) in the manifest. Fixed 22-09; regression tests in `test_train.py`.

### 11-09-2026 (the divergence guard fired on CONVERGED accumulate trials, near the budget)

**A "flat loss AND flat drift" guard cannot tell a model that never trained from one that has
converged — both are stationary. Gate it by how far into the step budget the trial is.**

- **Tried.** After the 08-09 fix (per-epoch drift; no more epoch-5 aborts), exp1 still marked ~2/split
  of the low-LR v2.6 `accumulate` arm DIVERGED — now at epoch ~330 (~90 % of the 5000-step budget).
- **Result.** These are CONVERGED trials (flat loss + flat drift because they finished), not dead
  ones — but a DIVERGED row is excluded from eval, so the grid quietly loses those cells.
- **Why.** `loss_const` had no notion of *when*: a converged plateau near the budget looks identical
  to a dead-from-start model. Dead trials trip at ~1-2 % of the budget; converged ones at ~90 %.
- **Instead.** Gate `loss_const` to the first `divergence_min_progress` (0.7) of the budget. Also
  provenance now carries `diverged`, so resume re-runs a diverged checkpoint instead of skipping it.
  Workers-on accumulate wall-clock, for walltime tuning: **v2 16.3 h, v2.6 10.3, v3 8.9, tabicl 7.0**
  (so `TRIALS_PER_TASK=1` gives a ~20 h walltime vs the 39.5 h at 2/task, same margin, better backfill).

### 08-09-2026 (the divergence guard killed every accumulate + L2-SP trial — 25% of the PD grid)

**A safeguard that reads a signal recorded on a different cadence than it runs will silently fall
back to the cruder rule it was built to replace.**

- **Tried.** The `loss_const` guard requires flat loss AND flat weight-drift (a fix from run-8, so
  a slow-but-training trial isn't mistaken for a dead one). exp1 came back with ~25% of the PD grid
  marked `DIVERGED`, all of it `accumulate` + `L2-SP=0.003`, aborting at epoch 5.
- **Result.** The weights were demonstrably moving (drift 0.073%→0.076% across the 5 aborted
  epochs) yet it died on flat loss alone — the drift half of the guard was inactive.
- **Why.** `stage_drift` is computed only on monitor epochs (every ~19), but the guard runs every
  epoch on a 5-epoch window, so `recent_drift` was ~always empty and the `not recent_drift`
  fallback degraded it to loss-only — which kills exactly the flat-loss, slow-moving anchored
  trials the drift check exists to protect. The regression test passed because it was a *replica*
  that fed drift on every epoch, a signal production only produced every 19th.
- **Instead.** Record a per-epoch `weight_drift` (free from the L2-SP penalty) and read that;
  extract the guard into a pure `_divergence_reason` so the test exercises the real code. Redeploy,
  clean the DIVERGED rows, resubmit (checkpoint-based resume re-runs the empty cells under the fix).

### 31-08-2026 (one typo in the frozen branch cost the whole frozen TabPFN arm of exp1_pd)

**A sweep axis that no control and no test ever executes will fail in production, once, expensively.**

- **Tried.** The freeze refactor put TabPFN + TabICLv2 on one `freeze_backbone`;
  `load_tabpfn_for_training` called it `freeze_tabpfn_backbone(model, version=_infer_version(ckpt_path))`.
- **Result.** `NameError: name 'ckpt_path' is not defined` — the local is `ckpt` (`version` is already
  computed). Only fires under `freeze_backbone=True`, so it killed **168/168 frozen TabPFN trials**
  of exp1_pd in ~4 s each; full-FT TabPFN and TabICLv2 (a separate loader) were untouched.
- **Why.** Three guards all missed it at once: the end-to-end test monkeypatches the whole loader
  away (its fake even `del`s `freeze_backbone`); exp0 pinned `frozen_backbone=[false]`; and
  ruff/pyflakes — whose F821 flags exactly this — is **not installed in the venv**, so
  `scripts/check.py` step 1 has effectively never run.
- **Instead.** `version=version`; a regression test that runs the REAL frozen branch; exp0 now sweeps
  `frozen_backbone=[false,true]`. **Install ruff (`pip install 'ruff>=0.6'`) and run it before every
  launch** — it is a declared dep that is not actually present.

### 26-08-2026 (exp0 caught the v2 OOM a linear model hid)

**A linear memory model is not just imprecise for attention — it is unsafe.**

- **Tried.** Sized the v2 row cap at 14000 from `14k ~ 111 GB`, a linear extrapolation of the
  measured 10k=79 GB probe point, and let preflight bless it with the same linear model.
- **Result.** exp0 job 11527923 OOM'd on the very first v2 step (hackerearth @ 14000 rows):
  176.71 GB in use on a 183 GB card. The real peak was ~177 GB, not 111.
- **Why.** TabPFN's row attention is O(rows^2), so peak memory is QUADRATIC in the row cap.
  79 GB x (14/10)^2 = 155 GB, plus baseline-eval residual and fragmentation -> OOM. A linear
  slope always under-predicts above the measured point, and it under-predicts more the further
  you extrapolate.
- **Instead.** v2 -> 10000 (the only measured-safe point). Preflight now anchors to measured
  (rows, GB, ok) points and REFUSES any cap at or beyond a known OOM, rather than trusting a
  slope. Never linearly extrapolate a quantity whose true scaling is quadratic; anchor to a
  measurement or re-probe.

### 25-08-2026 (sixth session)

**Fixing the Python was not enough — nothing was setting the env vars it reads.**

- **Tried.** Gave `eval_pipeline.py` `--config` / `--split-index` and had `eval_*.slurm` read
  them from `CREDITPFN_CONFIG` / `CREDITPFN_SPLIT_INDEX`, then called the leakage bug fixed.
- **Result.** No launcher set those for eval. `run_full_pipeline.sh` exports only the three
  storage roots, so `EVAL_ARGS` would have been empty, eval would have fallen back to
  config/train.yaml, and the wrong-held-out-datasets leakage would have returned intact — with
  the Python-side fix present and correct the whole time.
- **Why.** I fixed the layer I was looking at and assumed the plumbing above it. A three-layer
  path (launcher -> job script -> Python) is only as correct as its weakest layer, and the
  middle one had just been changed.
- **Instead.** Put the eval submission in `run_experiment.sh`, which already holds `$CONFIG` and
  the split loop, so train and eval cannot disagree by construction. And add a STATIC preflight
  check over the shell scripts, because this failure mode never raises — it just improves the
  numbers.

### 25-08-2026 (fifth session)

**Eval rebuilt the held-out dataset draw from the WRONG config, and nothing would have caught it.**

- **Tried.** Added `--config` / `--split-index` to `train_pipeline.py` for the 8-split campaign
  and assumed eval followed. It had neither flag.
- **Result.** `eval_pipeline._load_cfgs` loads `eval_cfg.train_cfg_path` = config/train.yaml, and
  `cfg_test_ids` comes from `split_from_cfg(train_cfg)`. So every split would have been evaluated
  against the SAME five datasets (train.yaml has `n_test_datasets: null`, so it fell back to the
  0.7/0.3 fractions). For split 7 — trained holding out thomas / PropPD2 / algorithmwatch /
  bondora — eval would have used taiwan / myhom / PropPD1 / algorithmwatch / credit_risk, of
  which **four were in its training set**.
- **Why.** Two independent config-loading paths for one experiment. The manifest-name half of the
  mismatch has a loud warning; the DRAW half has none, and it moves scores upward, so it would
  have read as a positive result.
- **Instead.** Both pipelines take the same two flags and merge identically, and
  `check_train_eval_agree` in preflight compares the two real code paths per split. Whenever two
  entry points must agree on something, assert it in preflight rather than trusting symmetry.

**A 3 GB, 2 987-column raw CSV is not a hang.**

- **Tried.** `python -m src.data.register` on a login node to rebuild the dataset registry.
- **Result.** It stalled on `0014.algorithmwatch` — raw is 3.0 GB and 159 k x 2 987, ~475 M
  cells, read with `pd.read_csv(low_memory=False)`. `register` also has NO flags and writes both
  manifests only at the very end, so interrupting it saves nothing.
- **Why.** The registry is rebuilt from RAW, and the widest dataset dominates.
- **Instead.** The manifests are 194 KB + 10 KB of text that already exist locally and were
  produced by this code from this raw data. Copy them; do not re-parse 3 GB on a shared login
  node. If a rebuild is genuinely needed, `sbatch` it to a CPU partition.

### 25-08-2026 (fourth session)

**Experiment 0 did its job on the first try — by failing.**

- **Tried.** Ran the lr=0 save/reload control (job 11525443) as the first thing on the cluster.
- **Result.** All four trials FAILED in 0.7 s each: `RuntimeError: Corpus split contains no
  training chunks`. `build_dataset_pool` returned an empty pool because
  `output/manifests/manifest_pd.csv` — the dataset REGISTRY — was not on the cluster, even
  though all 17 sanitized CSVs were.
- **Why.** The processed CSVs were staged to project storage; the registry that lists them lives
  under the durable output root (`$VSC_DATA/CreditPFN/output/manifests/`) and was not. Two
  different storage layers, one of them forgotten.
- **Instead.** `python -m src.data.register` rebuilds the registry from `data/raw` without
  re-sanitizing; if raw is not staged, copy the two manifest CSVs across. And the general lesson:
  an 8-cell control caught, for 3 seconds of GPU time, a condition that would have failed all
  1 536 cells of experiment 1.

**A checker that does not call the code it checks is not a checker.**

- **Tried.** Preflight verified the corpus by globbing `data/processed/<track>/*.csv`.
- **Result.** "pd: 17 processed datasets — ok", immediately before training died with an empty
  corpus. The glob and the pipeline disagreed because `build_dataset_pool` reads the registry,
  not the directory.
- **Why.** I re-implemented the check instead of invoking the thing being checked, so the two
  could drift — and they already had.
- **Instead.** Preflight calls `build_dataset_pool(track)` directly. Any check that can be
  expressed as "call the real function and look at the answer" should be.

**`x or default` is wrong whenever 0 is a legal value.**

- **Tried.** `seed=int(corpus.get("split_seed", None) or cfg.seed)`.
- **Result.** `split_seed=0` — split 0 of an 8-split campaign — evaluated to `cfg.seed` (42). So
  split 0 and a no-split run drew identical datasets, giving 7 distinct draws out of 8 and no
  trace of it in any output. Visible in experiment 0's resolved config as `split_seed: 0` while
  the split was actually seeded 42.
- **Why.** `or` tests truthiness, and 0 is falsy. The idiom is safe for names and paths and
  unsafe for counts, indices and seeds.
- **Instead.** `x if y is None else y` for anything whose valid range includes 0. Worth grepping
  for the pattern elsewhere.

### 25-08-2026 (third session)

**A preflight that reports failures for correct state is worse than no preflight.**

- **Tried.** Checked `REPO / "checkpoints"` and `REPO / "data/processed"` in
  `src/utils/preflight.py`.
- **Result.** On VSC those live on project storage (`/lustre1/project/stg_00211/CreditPFN/...`)
  behind `CREDITPFN_DATA_ROOT` and data.yaml's `paths.data_source`. A fully staged cluster
  reported **7 FAILURES** — 0 processed datasets, 0 checkpoints — for files that were present.
- **Why.** Wrote the checker against the dev machine's layout. `cluster_report.py`, written a
  day earlier, already resolved these correctly; I did not reuse it.
- **Instead.** Any check that touches the filesystem goes through `src/utils/paths`, and prints
  the directory it looked in so a false negative is self-diagnosing.

**LGD was one config flag away from making CRPS impossible forever.**

- **Tried.** `save_predictions: true`, assuming saved predictions meant the analysis was
  future-proof.
- **Result.** True for PD (probabilities saved -> every classification metric recomputable) and
  false for LGD, where only the point prediction was stored. RMSE/MAE/R^2 survive; nothing
  distributional does. And the distributional metric already in `EvalRow`, `neg_nll`, is a bar
  distribution for TabPFN and a quantile head for TabICLv2 — not comparable across families,
  which is exactly the comparison this project exists to make.
- **Why.** "We save predictions" was treated as sufficient without asking WHICH prediction. For
  a classifier the probability IS the distribution; for a regressor it is not.
- **Instead.** Store a fixed 9-point quantile grid per LGD row. CRPS, interval coverage and
  pinball at any level become post-hoc arithmetic, and the run does not have to be repeated to
  get them. Ask "what can this file no longer answer?" before a run, not after.

### 25-08-2026 (second session)

**A guard whose meaning changed under it: L2-SP silently off for half of experiment 1.**

- **Tried.** Kept `l2sp_applicable = (family == "tabicl") or (not use_lora)` while renaming what
  `use_lora` means, from "insert LoRA adapters" to "freeze the backbone".
- **Result.** Every frozen TabPFN trial ran with no L2-SP penalty at all, while the manifest
  recorded the configured `l2sp_lambda=0.003`. With lambda fixed and frozen swept that is half
  the grid, and the frozen-vs-full contrast would have been confounded with anchor-on-vs-off —
  unfalsifiable after the fact, because the manifest would have said the anchor was on.
- **Why.** The guard was CORRECT for LoRA (randomly initialised adapters have no pretrained w0
  to anchor to). Renaming the flag silently changed the guard's meaning; nothing re-derived it.
- **Instead.** Gate on the thing the reasoning is actually about — `not lora_adapters_inserted` —
  and add a preflight check that FAILS if `l2sp_applicable` ever mentions `use_lora` again. When
  a flag's meaning changes, grep every use of it, not just its call sites.

**Config knobs that nothing reads, round two.**

- **Tried.** `early_stopping`, `early_stopping_patience`, `log_would_stop`, `log_grad_norm`,
  `log_per_layer_drift` in the experiment configs.
- **Result.** Zero reads for all five. Grad-norm and per-layer-drift logging is unconditional, so
  those two knobs were decorative; the early-stopping trio described a feature that had never
  been written, and I had described it to Andreas as implemented.
- **Why.** Knobs were added alongside a plan, then the plan changed and the knob stayed.
- **Instead.** `python -m src.utils.preflight` now FAILS on any config key absent from src/ and
  scripts/. That check has caught six knobs across two days; it should run before every campaign.

**Cross-cluster splitting is a congestion valve, not a speed-up — measured twice now.**

- **Tried.** Reasoned (again) that mindwell and wICE having separate schedulers means splitting
  work across them shortens wall-clock.
- **Result.** Run-8 measured the opposite: eval on wICE gpu_h100 + gpu_a100 reached **0.20
  average concurrency** — 3.6 GPU-h took 17.9 h, and the gpu_a100 half never started — while
  mindwell's B200s gave **15-21 concurrent** on the same days. VSC fairshare weights the user's
  walltime over the last SEVEN days, so training submitted earlier sinks the priority of
  everything after it, and wICE's 36 GPUs serve the whole university with the cheapest (A100)
  the most contended.
- **Why.** "Separate queues" is true and irrelevant: what matters is which queue is congested,
  and it is wICE's. Separately, only TabICLv2 (27 GB) fits an 80 GB card at all, so at most 17 %
  of the work could move even if the queues were equal.
- **Instead.** Default everything to mindwell gpu_b200. `TABICL_DEST` / `EVAL_CLUSTER` exist to
  spill onto wICE if THIS run measures mindwell as congested — decide from the run's own
  concurrency, not from the topology.

### 25-08-2026

**A frozen backbone saves TIME but not MEMORY. I assumed the opposite and built two things on it.**

- **Tried.** Reasoned that `requires_grad=False` through the transformer stack means no autograd
  graph, so no retained activations, so a much higher row cap and a much smaller card. Built
  per-mode `{full, frozen}` row caps and routed the frozen arm to wice's 80 GB H100 on it.
- **Result.** Measured peak allocation, frozen vs full, same rows, same card (job 11524668 §9):
  v2 @10k 79.23 vs 79.23 GB; v2.6 @10k 109.42 vs 109.43; v3 @10k 51.50 vs 51.50; v3 @26k 131.48
  vs 131.47; tabicl @26k 26.80 vs 26.81. Identical to two decimals in every case. Step TIME did
  fall (v3 @26k 1.05 s vs 1.48 s = 0.71x; tabicl @26k 1.18 vs 1.17 = no change at all).
- **Why.** Peak sits in the FORWARD pass, which is bit-identical between the modes because we
  deliberately never call `.eval()`. And both families already recompute activations layer by
  layer during backward (`recompute_each_layer` in TabPFN, `recompute=True` in the TabICL path),
  so there were no retained activations to save in the first place.
- **Instead.** ONE row cap per base, serving both modes. Route on measured memory only. And the
  frozen arm is now known to cost ~0.8x a full trial, not 0.4x — which, with Rubachev's "partial
  ~ full" finding, makes it a similar-cost/similar-result arm rather than a cheap one.

**The A100 is 5.5x slower than the B200, not 2.2x, which inverts which card is cheaper.**

- **Tried.** Estimated the A100 at 2.2x the B200's time and concluded it was the best value per
  unit of work (8 500 x 2.2 = 18 700 vs the B200's 26 250), then split the campaign across
  clusters partly on that basis.
- **Result.** Measured bf16 8192^2 matmul: B200 **1586.3 TFLOP/s**, A100 **289.3** -> 5.48x. So
  the A100 costs 8 500 x 5.48 = 46 580 per B200-hour-equivalent: the **B200 is 1.8x cheaper per
  unit of work**, not more expensive. All-B200 is 702 GPU-h / 18.4M credits against 1237 / 20.6M
  for the split, and saves only 0.3 days of wall-clock.
- **Why.** A generation-gap guess. Blackwell's bf16 throughput over Ampere is much larger than
  the intuition "one generation, maybe 2x".
- **Instead.** Cross-cluster splitting on this pair is a QUEUE-CONGESTION valve, not an
  efficiency measure. Never price hardware from a guessed speed ratio; §5 of the report measures
  it in seconds.

**Disabling cuDNN's SDPA backend did NOT lift TabICLv2's context ceiling.**

- **Tried.** `relax_attention_backend()` turns off `cudnn_sdp` so the fused MHA graph that
  refused 40k rows is never selected.
- **Result.** TabICLv2 still fails above 26k, now with `CUDA error: invalid argument`
  (cudaErrorInvalidValue) instead of cuDNN's `mha_graph.execute(...).is_good()`. And bare SDPA
  at TabICLv2-like shapes runs fine to seq=60 000 on BOTH cards with cuDNN on or off, at 0.5 GB
  — so the primitive was never the constraint.
- **Why.** The failing shape is not the one the isolated SDPA test uses; something else in the
  real forward (batch/head geometry, or a kernel outside SDPA) carries the limit.
- **Instead.** Treat 26k as TabICLv2's working ceiling and stop attacking it — it is also the
  v3-parity value that keeps the cross-family comparison fair. Revisit only for experiment 2,
  where context size is the actual question, and measure with the REAL model, not with a
  synthetic attention call.

**`.item()` on a per-member statistic broke the whole LGD probe.**

- **Tried.** `znorm_mean = float(mean.detach().cpu().item())` in `_forward`, where `mean` is
  `(1, E, 1)`.
- **Result.** `RuntimeError: a Tensor with 2 elements cannot be converted to Scalar` for every
  TabPFN regressor probe — 12 of 16 probe cells lost, and no LGD row cap measured at all.
  Classifier cells passed because that branch never runs.
- **Why.** The code was written when the single-view path always had E=1, and nothing asserted it.
- **Instead.** Collapse across members explicitly and WARN if they disagree. Any `.item()` on a
  tensor whose shape depends on the ensemble size is a latent crash.

### 24-08-2026 (third session)

**"Freeze the backbone" needs a STRUCTURAL definition, not a module name, and the first two
attempts both got it wrong.**

- **Tried.** (1) LoRA for TabPFN, a real freeze for TabICLv2. (2) `requires_grad=False` on
  per-family module names: `icl_blocks`/`blocks` for TabPFN, `icl_predictor` for TabICLv2.
- **Result.** (1) LoRA is not a freeze — adapters sit inside the stack, so gradients traverse the
  whole network, activations are all retained, and it measured as a no-op that saved no memory.
  (2) Freezing all of `icl_predictor` also froze its `decoder` HEAD, which the TabPFN side keeps
  trainable: 4.6 % trainable vs TabPFN's 3.4 %, still not the same operation.
- **Why.** A module name encodes an architecture's naming, not the role we mean. TabICLv2 keeps
  95 % of its parameters in its LAST stage and bundles the head inside it; TabPFN keeps them in a
  middle stack with the head outside.
- **Instead.** State the rule structurally and derive it from the model: freeze the repeated-block
  stack holding the most parameters. That picks `icl_predictor.tf_icl.blocks` (12 blocks, 25.7M)
  over TabICLv2's 3-block `col_embedder.tf_col` (0.88M) and `row_interactor.tf_row` (0.40M), and
  `icl_blocks` over TabPFN's `feature_distribution_embedder.layers`. Trainable fractions land at
  6.6 / 0.9 / 3.4 % (classifiers) and 9.9 / 17.4 / 11.9 % (regressors) — matched, and the spread
  that remains is real architecture, not implementation drift. `src/train/freeze.py`.

**Where this sits in the literature (so it is not re-litigated).**

- Rubachev et al., "On Finetuning Tabular Foundation Models", re-evaluates four partial
  strategies for TabPFNv2: LoRA; "Last layers"; **"LayerNorm, Head and Embeddings - finetuning
  only the feature and target linear embedding layers, MLP prediction head and the affine layer
  normalization parameters"**; and learned numerical feature embeddings. Our arm is their third
  strategy MINUS the LayerNorm affines.
- **We deviate on the LayerNorms deliberately.** A trainable LayerNorm scale in block 0 forces
  autograd to build a graph from the loss back to block 0, so every activation in the stack is
  retained and the frozen arm costs what full fine-tuning costs while updating ~1 % of weights —
  the exact trap LoRA fell into here. Freezing the stack completely is what makes the arm cheap.
- **Their headline finding, worth knowing before we run:** "the difference between full
  finetuning and all considered PEFT variations is minimal", and full fine-tuning converged about
  twice as fast. Expect our frozen arm to land near the full arm, not beat it.
- **Neither upstream offers this arm.** `tabpfn` ships no freeze option at all (full FT only).
  `tabicl` ships three whole-stage flags (`freeze_col`/`freeze_row`/`freeze_icl`, all default
  False) and no way to freeze the transformer while keeping the head. So the implementation is
  necessarily ours; the SCHEME is standard, the code is not inherited.

**A hard threshold in a detection heuristic broke every small model.**

- **Tried.** Required a backbone stack to be >= 8 blocks deep, raising otherwise.
- **Result.** Every tiny TabICL test fixture raised `no repeated-block stack of at least 8 blocks`.
- **Why.** Depth was a proxy for "holds the bulk of the weights". The proxy fails on scaled-down
  models; the property it stood for does not.
- **Instead.** Select by parameter count and keep the depth check as a WARNING. Identical answers
  on all six shipped checkpoints, and it degrades gracefully.

### 24-08-2026 (second session)

**`frozen_backbone` meant the OPPOSITE thing in the two model families for a whole day.**

- **Tried.** "Unified" `frozen_backbone` across families by making both use
  `requires_grad=False` instead of LoRA, and reported it as done.
- **Result.** The mechanism was unified; the meaning was not. Measured from the shipped
  checkpoints: TabPFN froze `icl_blocks`/`blocks` = 88-99 % of parameters, leaving 0.9-36 %
  trainable. TabICLv2 froze `col_embedder`+`row_interactor` = **4.6 %**, leaving **95.4 %**
  trainable. Same column name, opposite operation.
- **Why.** "Backbone" was read as "whatever upstream's freeze knob freezes" rather than as a
  structural claim. TabICLv2's parameters sit in the LAST stage (`icl_predictor`, 26.28M of
  27.6M), not the first, so upstream's stage-3 freeze is a front-end freeze, not a backbone one.
- **Instead.** Freeze `icl_predictor` (12 blocks, 95.4 %) as the analogue of TabPFN's
  `icl_blocks` (24 blocks, 96.6 %). When a flag spans two architectures, check the PARAMETER
  FRACTION it moves in each before claiming they are comparable.

**A config knob that nothing reads: `monitor_every`.**

- **Tried.** Set `monitor_every: 5` / `20` in four experiment configs to control monitor cadence.
- **Result.** No code reads it. `grep -rn monitor_every src/ scripts/` matches one docstring in
  `training_viz.py`. The real knob is `train.epoch_eval_every`.
- **Why.** The name was carried over from a docstring rather than from the code.
- **Instead.** `grep` for a knob before adding it to a config. Every knob in `config/` should be
  greppable to a read site.

**Equal epochs hid a 7-15x difference in optimizer updates between the two pass modes.**

- **Tried.** Sweep `epoch_pass_modes: [full_pass, accumulate]` at a fixed epoch count.
- **Result.** `accumulate` takes one update per DATASET, `full_pass` one per BATCH. At equal
  epochs, PD gets 91 vs 13 updates (v3) or 202 vs 13 (v2.6) — so the axis under study was
  confounded 7-15x with "how much optimization happened at all".
- **Why.** An epoch is well defined for one dataset. Over a corpus of 13 tables of different
  sizes it is just "one pass over the concatenation", and how many UPDATES that produces depends
  on the row cap and the pass mode.
- **Instead.** Budget continued pretraining in optimizer steps (`target_total_steps`), which is
  what Garg (20 000) and Kolberg (10 000) both do. Equal steps costs a 2.4x data-exposure
  imbalance across bases, replacing a 7-15x imbalance on the axis being measured.

### 24-08-2026

**Equal epochs across TRACKS is not equal training — LGD was getting 1/8th of the dose it needs.**

- **Tried.** Set `epochs: 50` for both tracks in experiment 1, on the reasoning that run-8's
  curves "flatten well before 50".
- **Result.** True for PD (96.8 % of the final loss drop banked by epoch 50, worst trial 71 %).
  False for LGD by a wide margin: **40 %** median, worst trial **9.7 %**. LGD reaches 90 % only at
  epoch ~317 and 95 % at ~506.
- **Why.** `steps_per_epoch` = sum over training tables of ceil(rows / row_cap). PD has 13 tables
  of 7k-307k rows -> 90-200 steps per epoch. LGD has 6 tables of 594-4 637 rows, nearly all under
  the 26k cap -> **6-30 steps** per epoch. So an LGD epoch is one order of magnitude less training
  than a PD epoch, and any single `epochs` value is wrong for one of the two tracks.
- **Instead.** Per-track epochs, chosen from the measured curve: PD 400 -> stays at 50, LGD 400.
  Do not copy an epoch count between tracks without recomputing `steps_per_epoch`.

**The 26k TabICLv2 ceiling is not TabICLv2's arithmetic, and raising it is not free.**

- **Tried.** Treated the "26k rows, 27 GB of a 183 GB card" ceiling as waste to be reclaimed.
- **Result.** Verified in the installed source: TabICLv2's only row-length quadratic stage is
  `icl_predictor.tf_icl` (12 blocks); the stage that would dominate, `col_embedder`, uses
  `InducedSelfAttentionBlock` — documented O(n) via 16 inducing points — and `row_interactor`
  attends over ~64 features, not rows. TabPFN v3/v2.6 by contrast run 24 blocks of full
  row-quadratic attention. Hence the 5x lower memory slope (0.52 vs 2.51 GB/1k/member).
- **Why.** Attention cost per epoch is (N/n) x O(n^2) = O(N x n): **linear in the row cap.**
  Doubling the cap doubles the compute for the same data. "Only 27 GB of 183" is not idle capacity
  waiting to be used for free; it is the cost of a design that scales sub-quadratically.
- **Instead.** Keep 26k for experiment 1, where it is also the v3-parity value that makes the
  cross-family comparison fair. `max_cells_per_epoch` (already in `data.yaml`, currently null) is
  the right knob for experiment 2, whose question actually is context size.

**Garg's "20 000 rows on one 11 GB 2080 Ti" is a CELL budget, not a row budget.**

- **Tried.** Compared our measured 5.44 GB/1k rows/member against Real-TabPFN's setup and got a
  ~10x contradiction.
- **Result.** No contradiction: Garg caps each dataset at **400 000 total cells**, and that corpus
  averages 7-9 features. Ours run 7 to 64 features (median 20) — a 9x spread.
- **Why.** Memory tracks rows x features, not rows. A uniform row cap is therefore sized for the
  widest table and leaves narrow tables using a fraction of the card.
- **Instead.** Know which budget a literature number is quoting before comparing to it. Our row
  cap stays for comparability in experiment 1; cells are the honest budget for experiment 2.

### 13-08-2026

**Treating a flat loss as divergence.**
- **Tried:** aborting a trial when the training loss stays inside a 1e-4 window for five
  consecutive epochs — a rule written against the 2026-05-28 collapse, where a dead model
  emitted a constant loss.
- **Result:** it killed a perfectly healthy run-8 trial (v2.6 @3e-7 full-FT) at epoch 19.
  The loss was 0.4689–0.4690 across the window, so the rule fired — while weight drift rose
  monotonically 0.042 % → 0.085 %, held-out AUC sat at 0.7151, gradients were normal and no
  AMP step was skipped.
- **Why:** the lowest learning rate in the sweep is *supposed* to move the loss slowly, and
  v2.6's loss plateau is flat. The rule was calibrated on a hotter LR and never revisited
  when 3e-7 was added back. It removed the exact configuration the run existed to test.
- **Instead:** require a flat loss **and** flat weight drift — a model that has actually died
  stops moving, one that is learning slowly does not. Falls back to the loss-only rule when
  no drift is recorded. Pinned by
  `tests/test_train.py::test_a_flat_loss_with_growing_drift_is_not_divergence`.

**Capping the step budget by epochs, with a corpus that varies per arm.**
- **Tried:** `max_epochs_for_step_budget: 1200` as a safety rail on the new "extend epochs
  to reach the step target" behaviour.
- **Result:** on LGD the cap bound before the target, and it bound UNEVENLY across the swept
  corpus arms: tabicl got 9 600 steps at `min_train_rows=0` and **4 800** at 5 000; v3 got
  19 200 vs 14 400. Only v2.6 reached the budget in both arms.
- **Why:** steps/epoch is `sum(ceil(rows_i / cap))` over the TRAINING datasets, so the arm
  with fewer datasets gets fewer steps per epoch and hits an epoch cap sooner. The corpus
  experiment was therefore confounded with the training budget — and biased *against* the
  filter, which is the hypothesis under test.
- **Instead:** cap raised to 6 000, which covers the worst case (LGD tabicl at 4 steps/epoch
  needs 5 000 epochs ≈ 2.5 h). A rail sized in epochs is only safe if every arm reaches the
  target underneath it.

**Splitting the eval across two wICE GPU partitions.**
- **Tried:** `gpu_h100` + `gpu_a100`, on the theory that two pools drain twice as fast.
- **Result:** average concurrency **0.20**, peak 2. 3.6 GPU-h took 17.9 h of wall-clock, and
  the `gpu_a100` half never started at all — PENDING with Reason=Priority for days. Half the
  eval cells were never scored, so the run has no complete trained-vs-untuned comparison.
- **Why:** the VSC scheduler weights fairshare on the user's walltime over the last **seven
  days**, so the ~120 GPU-h of training we submit immediately beforehand is precisely what
  sinks the eval's priority. wICE's 36 GPUs serve the whole university and the *cheapest*
  partition (A100 at 141.7 credits/GPU-min against H100's 569.4) is the most contended.
  Splitting also doubles the number of queue positions being waited on. Packing the cells
  fixed the task size — 94 s → 28 min — and the binding constraint simply moved.
- **Instead:** one partition, on Mindwell `gpu_b200`, where the same account held 15–21 GPUs
  concurrently on the same days. Cheaper per minute than H100 and faster. Walltime cut 5 h →
  2 h so tasks backfill.

### 11-08-2026

**One slurm array task per (model × dataset) eval cell.**
- **Tried:** the obvious decomposition — 209 PD array tasks, each scoring one model on one
  dataset across 5 folds, submitted to `gpu_h100`.
- **Result:** the median task computed for **94 seconds**, 19 tasks did nothing at all (already
  scored, ~2 s) and still took a GPU allocation, and the array reached an **average concurrency
  of 0.73** — 6.7 GPU-hours took 9.1 hours of wall-clock, 44 % of it with nothing running.
- **Why:** every array task is scheduled independently. On a busy partition the wait to be
  allocated dominates a 94-second job, and hundreds of them never build up concurrency. The
  same day, on the same account, training went out as 36 big tasks and held 15-21 GPUs at once,
  draining 90 GPU-hours in 5.1 hours. The scheduler rewards a few substantial jobs.
- **Instead:** `eval_pipeline.py --tasks N` packs cells into N tasks of roughly equal estimated
  cost (`rows × per-family rate`, longest-processing-time-first). At `--tasks 16` the PD eval is
  16 tasks of ~50 minutes. Measure before assuming more parallelism is faster.

**Splitting eval pools by a stride over raw cells.**
- **Tried:** `i % pools == pool` over the model-major cell list — the 08-08-2026 replacement for
  the model-parity split.
- **Result:** LGD has exactly 2 test datasets and the eval ran on 2 pools, so pool 0 got every
  even index, which is **dataset 0 every time**. Pool 0 scored `PropLGD2` and nothing else;
  `0007.lgd_lendingclub` existed only in the other pool.
- **Why:** with cells enumerated model-major, a stride of `pools` aligns exactly with the dataset
  index whenever `n_datasets` is a multiple of `pools`. This is the 11-07-2026 dead end (a whole
  dataset in one pool) reintroduced by a different mechanism — the fix for one split shape
  recreated the failure of the other.
- **Instead:** stride over **packed** tasks, each of which holds a cost-balanced mix of cells, so
  no stride can isolate a dataset. Pinned by
  `tests/test_eval.py::test_pools_over_packed_tasks_never_align_with_the_dataset_index`.

**Equalising the step budget by trimming epochs only.**
- **Tried:** `target_total_steps` reduced `epochs` when a base would overshoot the shared budget.
- **Result:** PD equalised correctly (9 100 steps for every base), but **LGD ran 800 steps for
  tabicl, 1 600 for v3 and 3 200 for v2.6** — 3-11× under budget, and still unequal across bases.
  Every LGD trial in that run lost to its untuned baseline.
- **Why:** trimming can only ever remove epochs. LGD's 6 small training tables give very few
  steps per epoch, so 100 epochs was already far below the target and nothing fired. The check
  was written and verified on PD, where it did fire, so the track it silently skipped was the
  one nobody looked at.
- **Instead:** the budget is a target in BOTH directions — epochs are raised as well as trimmed,
  bounded by `train.max_epochs_for_step_budget`. Verify a budget mechanism on the track where it
  is *least* likely to trigger.

### 08-08-2026

**Splitting the eval pools by model parity.**
- **Tried:** two eval pools (H100 + A100), tasks assigned by `model_index % 2`, so each pool held a
  disjoint half of the models over all datasets.
- **Result:** the `gpu_a100` pool (jobs …490/…492) never logged anything in run-6, and that lost
  **12 of 24 models outright — including `tabpfn-untuned[v3]` and `tabicl-untuned`.**
  Trained-vs-untuned is therefore *not computable* for v3 or TabICLv2 from run-6.
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

**Freezing TabICLv2's backbone with `.eval()`.**
- **Tried:** `model.col_embedder.eval()` (plus the row interactor) to freeze the backbone for the
  `_iclhead` adaptation arm.
- **Result:** **all 16 `_iclhead` trials failed**, both tracks, a few steps in: *"A view was created
  in no_grad mode and is being modified inplace with grad mode enabled."*
- **Why:** TabICLv2 branches `if self.training: _train_forward() else: _inference_forward()` in
  **both** `ColEmbedder` and `RowInteractor`. Eval mode routes to the inference path, which runs
  under `torch.no_grad()` and writes CLS tokens into its own input in place
  (`interaction.py::_inference_forward`). Upstream's `_set_training_mode` has the same latent bug,
  so upstream code is not evidence that this is safe.
- **Instead:** freeze with `requires_grad=False` **only**, never `.eval()`. Nothing is lost —
  dropout defaults to 0.0 and there is no BatchNorm. See `RESEARCH_BRIEF.md` §4 for the one place
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

**Reading TabICLv2's context limits off the library's default arguments.**
- **Tried:** set `max_rows_per_epoch.tabicl: 10000` and `max_rows_per_model.tabicl: 50000` from
  `max_data_size=10_000` in their convenience finetune wrapper.
- **Result:** wrong by 2.6× (training) and 20× (eval); the user challenged the values and was right.
  It would have handicapped TabICLv2 against v3 and thrown away its headline capability.
- **Why:** **a library default is not a capability limit.** Their own stage-3 pretraining runs
  400–60 000 samples (grad-checkpointed above 20K), and Qu et al. 2026 report 1M samples × 500
  features in ~450 s under 50 GB GPU + 24 GB CPU via offloading; QASSMax exists precisely to keep
  attention sharp at long context.
- **Instead:** caps come from the paper, then get **measured** (`scripts/probe_row_cap.py`) — 26 000
  train (= v3 parity, so architecture is not confounded with context size) and 1 000 000 eval. Full
  numbers in `RESEARCH_BRIEF.md` §3.

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
- **Tried:** adding the two TabICLv2 bases (32 → 48 trials/track).
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
- **Instead:** measured caps only (`RESEARCH_BRIEF.md` §3), member-aware scaling in `train_one_config`. **Do
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
