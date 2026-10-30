# Agents' memory — runs and dead ends

## 30-10-2026

- Numbering/cache/recovery validation: full local pytest **1,263 passed, one existing absent-manifest skip** (10m14s); 49 Slurm shell files pass Bash syntax checks. All **15 notebooks / 133 figures** reran successfully. After repairing UTF-8 separators and rerunning the affected reports, all 133 inline images match the previously reviewed images byte-for-byte. Verified PDF/inline/caption ownership, every final summary in `All_Results.md`, no orphan PDFs, valid notebook schemas and the local schema-2 result layout. The added source-encoding regression passes with the 13-test numbering suite. These reports still use the prior download; current-source VSC runtime acceptance remains pending.
- Deployed source `4ac75c3` through Git into canonical DATA after checking consumers. Both read-only library checkouts now match the recorded pins (TFM `26156f3`, VSC `0ed4ffa`); the frozen refresh evaluator remains `4bf4535`. Cache jobs `11755971` / `62475830` were submitted with a durable receipt; consult their final removal inventories rather than inferring reclaimed bytes from the earlier storage estimate.
- **Tried:** bounded SSH deployment and initial VSC-library checkout.
  **Result:** transport timed out after the parent Git update; the new library worktree was incomplete.
  **Why:** the checkout exceeded the connection lifetime; an SSH timeout did not establish whether the remote Git operation had completed.
  **Instead:** inspect exact commits, processes, locks and worktree contents before retrying. Only the newly created empty library worktree was checked out again; both pins and clean worktrees were then verified. No existing library modifications were discarded.

- Full refresh audit **62474641** completed on wICE at 10:26 CET (9m27s, exit 0): all 445 refresh tasks passed. Independently verified 459 benchmark comparisons / 2,295 folds / 30,967,719 predictions and 125 context tasks / 5,550 cells / 147,493,764 predictions, with unchanged cohorts/targets and complete context test rows. `experiment0/manifests/evaluation_refresh/study20261029/inspection.json` is passing; publication is still staged. No extension research was released.
- Public configs/Slurm wrapper names now follow E1/E2/E5/E6/E7/E8/E9/E10; E3/E4 reuse E1. Compared all eight moved YAML files with Git: scientific values unchanged apart from the relocated control-config reference. Embedded campaign/checkpoint identities stay immutable. YAML now supplies the actual layout, and source identities include it. Combined publication cleanup follows campaign/extension dependencies; shell mappings and real log routing are checked against the registry.
- User approved scoped successful-job cache retirement and node-local cache-policy tests. Completed Slurm cleanup removed **466 GPFS caches / 147,810,082,354 bytes (137.7 GiB)** and two empty Lustre caches, with per-path accounting and immutable inventories; six failed GPFS caches remain protected. A 55-second login-node recursive inventory had timed out; inventory/deletion instead ran in bounded native Slurm allocations. Shared staged inputs, runtime images, recovery state and scientific results were not selected.
- **Tried:** focused tests after replacing folder-number inference with explicit config ownership and renaming public config paths.
  **Result:** older mock launcher fixtures lacked real config files and cleanup fixtures still created obsolete figure folders.
  **Why:** fixtures encoded the removed inference/layout rather than the deployed contract.
  **Instead:** supply real owner-bearing configs and current RQ metadata; test all public-to-recorded identities, both source fingerprints, and successful/failed job EXIT cleanup while preserving peers.
- **Tried:** one ad-hoc source-text edit with Python's Windows default decoding.
  **Result:** four UTF-8 plotting/runner files were re-encoded incorrectly; no numerical evidence changed.
  **Why:** implicit CP1252 decoding rather than explicit UTF-8.
  **Instead:** reverse the exact encoding transformation, use explicit UTF-8 thereafter, and rerun source/notebook validation before publication.

- At 09:54 CET all **445 FP32 tasks** had completed with Slurm exit `0:0`. The old extension's 68 capacity cases were contained in 18 cancelled array submissions; no capacity gate or extension research passed. Recovery must use a fresh extension/source identity and retain the original evaluator for its completed refresh. The recovery coordinator now binds that evaluator's clean Git commit and preparation digest, repeats current-source acceptance, preserves all final artifact gates and avoids dependencies on terminal jobs that have left Slurm's controller.
- RQ notebook compaction validated locally: **15/15 notebooks, 133 figures** (previously 29/261); full suite **1,229 passed, one existing absent-manifest skip** in 10m17s. Visually reviewed all new/changed RQ and overview figures, including full-resolution label checks. Independently verified every PDF/inline/caption identity, all final summaries in `All_Results.md`, notebook schemas, execution counts, no embedded definitions and no orphan publication artifacts. Scientific result/checkpoint paths and the shared factorial are unchanged. These reports use the currently downloaded evidence; they do not claim the pending VSC FP32/context/extension work has passed or been published.
- **Tried:** run the v2/v2.6 extension gate after local adapter-mapping tests passed.
  **Result:** all four DiagScale training-entry checks failed; one LoRA case crashed without a Python traceback.
  **Why:** the training loop retained a v3-only filename check. The native-crash cause remains unresolved; it must not be described as a numerical-prediction failure without evidence.
  **Instead:** validate the actual loaded projection layout through the existing strict injector, add four production-entry regressions, enable fault-handler traces and use a fresh source-bound gate after deployment/acceptance. Never reuse stale successful receipts or change the scientific recipes to bypass the gate.
- **Tried:** combine adapter prediction diagnostics under one RQ.
  **Result:** report generation detected duplicate comparison labels for the three LoRA learning rates.
  **Why:** the old label fallback identified every LoRA recipe as one-sample training; extension readers also depended on a temporary loader scope.
  **Instead:** retain LoRA rate identity and resolve prediction roots in the configuration's explicit extension scope; regression tests cover both. Saved predictions are unchanged.

| Run | Verified outcome | Scope |
|---|---|---|
| Mindwell `11755971`, cache retirement, source `4ac75c3` | Completed, exit 0, 6m34s; 466 completed-job caches / 147,810,082,354 bytes removed | Six failed-job caches protected; final plan-bound removal receipt records every removed path; no research/runtime assets selected |
| wICE `62475830`, cache retirement, source `4ac75c3` | Completed, exit 0, 34 seconds; two empty completed-job caches retired | Exact removal receipt under `experiment0/manifests/storage_review`; no scientific files selected |
| wICE `62474641`, full FP32 inspection, source `4bf4535`, runtime `env20261029f` | Completed, exit 0, 9m27s; all 445 tasks passed | 459 benchmark comparisons / 2,295 folds / 30,967,719 predictions; 5,550 context cells / 147,493,764 predictions. Publication remains staged |
| Mindwell `11753657`, extension smoke gate, source `4bf4535`, runtime `env20261029f` | Seven of twelve passed; cases 4–7 failed before training; case 9 exited 139 | All four context probes and three LoRA cases passed. The four DiagScale cases reached an obsolete v3-only training guard despite successful standalone adapter probes. LoRA LGD v2.6 crashed in a native process after saving the second reference checkpoint; its cause is not established. Inspector `11753862` and research controller `11753901` were cancelled; new extension research was not released. Preserve the failed gate and its recovery evidence. |
| wICE `62473876`, FP32 benchmark preflight, accepted runtime `env20261029f`, source `4bf4535` | Completed, exit 0, 4m24s; **459 comparisons / 2,295 folds / 30,967,719 saved predictions** passed independent metric and cohort checks | Public E7/E8/E9 (historical campaigns 3/6/7): 228/99/132 comparisons. All 320 benchmark shards have successful completion receipts. Original results remain active; context checks and final publication are still pending. Submission and audit receipts are under canonical DATA `experiment0/manifests/study_v2/study20261029/benchmark-preflight*.json`. |

- Final local validation of the migration read guard and runtime-selection/bootstrap changes: **1,211 passed, one existing absent-manifest skip**, with the final full rerun taking 12m30s. The final immutable-runtime seal and bootstrap input-staging checks also passed their focused 40-test run. The default activation now refuses implicit DATA Conda use; the explicit same-GPU comparator remains available, and the sealed Slurm submission bootstrap preserves its acceptance dependencies. These changes are not deployed into active VSC source `4bf4535`.

## 29-10-2026

- Preserved the temporary checkout's three pre-Git build helpers in a **5,051-byte** archive under canonical DATA `experiment0/manifests/runtime_builds/study20261029/pre-git-build-helpers.tar.gz`. Verified every extracted byte against its original Git stash blob; the adjacent JSON records stash/tree IDs and SHA-256 digests. This is historical evidence, not a source deployment. The original stash and active checkout remain intact until their consumers finish.
- The post-migration local rerun passed **29/29 notebooks, 261 figures**. Every complete summary and all 261 inline PNG byte sequences are identical to the reviewed pre-migration revision; PDFs, inline figures and captions have one-to-one ownership, no orphan PDFs, and all 29 summaries appear verbatim (trailing padding normalized) in the ordered combined report. The new VSC results are still pending and are not included. All 47 shell scripts pass Bash syntax checks; two local CRLF copies were restored to their exact already-published LF bytes.
- The extension research controller **11753901** was accepted after inspector **11753862**; the persistent submission helper completed its work. At 23:37 CET, refresh **11753618** had **29 completed / 8 running / 408 pending** tasks, without an exceptional state; all 80 extension GPU checks were still queued behind normal priority/dependencies. No new results have been promoted.
- Migrated the local merged output with the same journaled layout tool: **35,599 files / 19,113,502,239 bytes** passed before/after SHA-256 verification, and `STUDY_LAYOUT.json` now routes historical campaign identities to the public layout. The empty local staging directory was retired. A fresh 29-notebook run is checking the actual migrated readers; this does not include still-running VSC additions.
- Verified both VSC layout markers and removed only the two empty migration staging directories with nonrecursive `rmdir`; the canonical DATA cleanup receipt records their exact paths. Migration journals, scientific files and active runtime caches remain intact. Extension-test admission now includes all **12 smoke cases, 68 capacity cases and inspector 11753862**; queued tests are not passing tests.

- Post-correction validation: full local suite **1,202 passed, one existing absent-manifest skip** in **11m51s**. Runtime receipt isolation, preservation of earlier source receipts, migration byte preservation/recovery and a publication-write-free idempotent rerun all pass. Current VSC consumers remain on `4bf4535`; these local fixes are not yet deployed into their checkout.

- **Tried:** rerun the complete local suite after the runtime-receipt scope correction.
  **Result:** 1,201 passed, one existing skip and one failure in 13m51s; the second application of an already completed migration hit Windows access denied while replacing unchanged `STUDY_LAYOUT.json`.
  **Why:** the migration unnecessarily rewrote publication markers even after recognizing the layout as complete; the precise external Windows file-lock source was not established.
  **Instead:** make the completed path a true no-op and explicitly test that it performs no publication writes. Remove only the empty owned staging directory on successful new migrations, and reject linked staging. The VSC campaign keeps its unchanged deployed source and already verified output; these local housekeeping corrections require validation before publication.

| Run | Verified outcome | Scope |
|---|---|---|
| wICE `62471310`, owned runtime-build cleanup | Completed, exit 0, 1m39s; 348 compact build records preserved and hash checked before removal | Six exact task-owned Lustre build directories and three empty failed PROJECT runtime directories removed; accepted `env20261029f`, active caches, source checkouts and scientific assets retained |

- Migration controller **11753589** completed at 22:26 CET after verifying all **6,695 DATA files and 28,904 PROJECT files**; both output roots now carry `STUDY_LAYOUT.json`. Historical identities/checkpoint locations and scientific file bytes are preserved. FP32 refresh **11753618** has 445 tasks and no running throttle; eight tasks had completed with no recorded failure at 22:36, while the larger tasks were still running. This is progress, not final prediction acceptance/publication.
- **Tried:** extension preparation `11753619` against the already accepted container while the FP32 refresh ran from the same immutable source.
  **Result:** preparation failed before GPU work; its dependent smoke/capacity/inspection/research jobs were cancelled.
  **Why:** runtime acceptance used the experiment-scoped manifests helper and therefore looked inside `experiment0/extensions/study20261029/` for a global runtime receipt.
  **Instead:** preserve failed submission receipts and unchanged deployed source. A temporary, explicitly recorded symlink in that extension's manifests points to the genuine accepted `env20261029f` receipts; no gate or receipt was rewritten. Retry preparation **11753656** passed in 66 seconds. The permanent local fix makes runtime receipts scope-independent and source-specific; 49 focused tests pass. Deploy it only after current source consumers finish, then validate the new source independently and retire the alias.
- **Tried:** resume the remaining extension-test submissions through a bounded interactive SSH call.
  **Result:** the 55-second connection ended while the submitter waited on the site's 500-submitted-task ceiling; accepted job IDs and a clear (not ambiguous) submission state were saved.
  **Why:** queue admission can exceed an interactive transport's lifetime even though computation is submitted to Slurm correctly.
  **Instead:** verify no surviving submitter/lock, then resume the same recorded plan in a persistent lightweight submission process. It also queues the existing research controller after the independent safety inspector; no completed/accepted task is duplicated. Retry smoke **11753657** contains 12 cases; capacity admission continues incrementally as the site permits, with no artificial running throttle.

- Runtime `env20261029f` passed independent live acceptance **11753588** at 22:16 CET. CPU preparation **11753586** verified all 238 distributions and 53 staged input files (1,259,693,608 bytes). All ten B200 cases in **11753587** passed: original and CPT predictions matched the original runtime **exactly** for all four bases × PD/LGD; both TabICL FP32 production-shape probes (50,000 context, 4,096 query, 64 features, eight views) peaked at **77.04 GiB**. These bounded checks validate this recorded runtime/protocol, not arbitrary future packages or shapes. Migration controller **11753589** can now proceed through the gated workflow.
- Study workflow `study20261029` submitted after GitHub and the clean separate DATA checkout both verified `4bf45351075b3722d85a366da22b8df5f2322f75`, with exact library pins and no active CreditPFN consumers. Mindwell accepted runtime CPU preparation **11753586**, ten-case GPU array **11753587**, independent acceptance **11753588** and gated migration/launch controller **11753589**. Arrays have no artificial running throttle. Submission alone does not establish acceptance or authorize publication; the original output/runtime remain preserved pending the gates.
- Current implementation validation: full local suite **1,199 passed, one existing skip** in 10m47s; the final presentation changes and two additional regressions passed all 49 focused tests. All **29 notebooks** ran successfully; four were rerun after visual refinements. Independently verified **261** PDF/inline/caption identities, all final summaries in `All_Results.md` (trailing padding normalized), valid notebook schemas, no embedded definitions and no orphan PDFs. Visually reviewed the full figure collection, including enlarged checks of apparent contact-sheet clipping; paginated dense overview maps and clarified factorial titles/cost labels. This is not VSC runtime acceptance. The user explicitly requested retirement of the temporary DATA checkout and obsolete owned workspaces after completion; maintain the cleanup ledger in VSC.md and verify consumers before removing anything.
- Runtime build **62470828**, `env20261029f`, completed on wICE in **8m48s**, exit 0. Separate DATA checkout pulled and verified Git revision `d6e183e1b61456c298fb4791ff343955bacd5aab` before submission. The container matches Python and all 238 recorded distribution versions plus immutable Git-package source hashes; PROJECT contains `container.sif`, `img.sqfs` and 236 inventoried files. CPU/GPU acceptance, native-scratch execution and scientific-result migration remain pending; the original environment and assets are unchanged. The local and canonical templates now match SHA-256 `0ed244c16cab8ea86afab5ed90eea36eef3a9e218314c38538e227e4f009fd3c` after synchronization commit `5186e75`.
- Runtime recreation `62470745` (`env20261029e`) failed after 28 seconds before any installation. Direct inspection found a pip cache but no `conda-packages` directory in the prior owned workspace; requiring both caused the failure. Reuse either cache independently and resolve both paths before checking containment. The original runtime, checkpoints and results remain unchanged; no new research workload has started.
- Deployment correction: the first isolated build helpers were copied uncommitted to the separate DATA checkout for troubleshooting. The user reiterated the required local-edit → test → commit/push → VSC Git update workflow. Preserve those helper bytes as historical build evidence, deploy subsequent changes through Git, and never update a checkout used by active jobs.
- Runtime recreation: wICE `62468686` (`env20261029c`) stopped after 12m47s because the recorded TALENT version is Git-installed and unavailable on PyPI. The recorder now retains the exact public Git commits and installed Python-source hashes for TALENT and TabPFN. Retry `62468802` (`env20261029d`) restored both sources but failed its final exact-version check after 47m20s; no runtime was accepted or research job started.
- **Tried:** install the recorded `torch==2.12.0` with the CUDA-13 index also available.
  **Result:** pip selected distribution `2.12.0+cu130`; the original distribution metadata is `2.12.0`, and the acceptance check rejected the mismatch.
  **Why:** PEP 440 equality permits a local-version suffix when the requested version omits it; equal import-version descriptions did not establish exact distribution identity.
  **Instead:** require strict `===` equality for recorded index distributions; retain immutable Git pins, source hashes and subsequent same-GPU prediction comparisons. Reuse only verified owned build caches, preserving the original environment and failed-build evidence.
- Runtime recreation: wICE `62468504` failed before installation because the site wrapper has no `frontends/conda-containerize` alias; the private copy now creates it. Retry `62468548` recreated the explicit Conda packages on owned Lustre scratch but stopped at the pre-pip interpreter check. No old environment or scientific evidence was changed, and no runtime was accepted.
- **Tried:** invoke plain `python` after the site wrapper's `conda activate` inside its container build.
  **Result:** the interpreter-prefix assertion failed before pip installation (wICE `62468548`, exit 1, 7m25s).
  **Why:** activation alone did not prove that the selected interpreter belonged to the new prefix; the check correctly refused installation. The exact inherited-path source was not recorded.
  **Instead:** invoke the new prefix's absolute interpreter for every install/validation command, prepend its bin directory and print both prefixes before asserting. Preserve the failed build for diagnosis; retry in a fresh version directory.
- Approved implementation started: direct native Windows SSH verified both queues empty at 16:28 CET. The original DATA checkout and environment are preserved. A separate DATA checkout `CreditPFN-study-v2` hosts the runtime build; wICE `62468504` was accepted for `env20261029` (eight CPUs, partition-default memory, three hours). It records exact Conda/pip versions and recreates them inside the site's container wrapper with unpacked work on Lustre scratch. Submission is not runtime acceptance; CPU/GPU checks and image verification remain required.
- **Tried:** clone the existing DATA repository and fetch the current GitHub revision within a bounded SSH call.
  **Result:** the channel timed out while `git fetch origin` was still running; the clone itself exists at `7988e7a`.
  **Why:** remote fetch did not complete within 55 seconds; no conclusion about authentication or network cause yet.
  **Instead:** inspect process and checkout state before retrying; stage only the three independent, new runtime-build files in the separate checkout. The original checkout remains unchanged.

- Policy adoption: local `docs/TEMPLATE.md` is byte-identical to the canonical copy (SHA-256 `285c7c321b2cd52d8d415e30f41fe775ceb47113f39007c2490fdb0ec23fc8d8`); VSC Library pin is `0ed4ffa89d312b9bb6411681b6a14f81672439f3`. Updated AGENTS, README and VSC runbook to require manual Smallstep authentication with the existing Windows agent, direct verified inspection, DATA Git checkout, PROJECT immutable SIF runtimes/persistent assets and native temporary scratch. Existing Conda/cache placement, root aliases, old throttles and CPU/memory requests remain explicitly pending migration. No runtime, config, scientific output or notebook was changed.
- Direct read-only VSC inspection at 15:39 CET: existing certificates worked; both controller queues were empty. Actual checkout and `.git` resolve to DATA. Remote source is `7988e7a`; TFM checkout `de324ef` differs from its recorded pin and VSC Library is uninitialized. The existing CreditPFN Conda environment and cache remain on DATA. Lustre PROJECT contains `data`, `checkpoints`, `output CreditPFN`; no CreditPFN SIF directory or extension scope was found in the inspected locations. GPFS project `stg_00211` was not visible. Both native scratch tiers retain CreditPFN working directories, including a GPFS input pointer; queue emptiness alone does not authorize deletion. No remote source/submodule update, authentication, installation, migration, deletion or Slurm submission was performed.
- Read-only quota snapshot: DATA 56,233/76,800 MiB and 531k/10,000k files; shared Lustre `stg_00211` 538.2 GiB/1 TiB and 45,343/150,000 files. These are account/allocation totals, not CreditPFN-only usage or proof of a GPFS allocation. Recheck before migration. Local original-campaign manifests contain 768 OK outcomes (512/32/96/48/16/16/48 for E1/2/3/5/6/7/8); interrupted segments are additional history, not extra scientific trials. The 128-trial v2/v2.6 extension has no local or inspected remote scope yet. The proposed ten-study reorganization remains a plan only.
- Validation of this documentation-only adoption: new links/fences and `git diff --check` pass. Full local pytest: 1,173 passed, one existing skip, two failures; both failing cases passed on an isolated rerun (2 passed). No notebook or plotting code changed, so notebooks were not rerun. No VSC computational test was submitted.
- **Tried:** the complete local suite against unchanged executable code during the policy-documentation update.
  **Result:** two exact-budget/recovery tests failed at `checkpoint_io.atomic_save` with Windows access denied during replacement of a temporary recovery checkpoint, before their numerical-equivalence assertions.
  **Why:** `os.replace` was denied on the Windows temporary filesystem; the process or filesystem condition causing that denial was not identified. Both exact cases passed unchanged when rerun separately.
  **Instead:** preserve the failed full-suite result, record the passing focused rerun and keep this separate from scientific validity or VSC runtime acceptance. No production serialization change or weakened test assertion was made under the documentation-only scope.
- **Tried:** native Windows OpenSSH, strict host checking and LF stdin script with separate stdout/stderr markers plus deliberate exit 7.
  **Result:** stdout and exit status arrived, but the error marker was absent from both captured streams.
  **Why:** the observed login transport did not preserve stderr; its exact upstream cause was not established.
  **Instead:** start the remote script with `exec 2>&1`; both markers and exit 7 then verified execution. Subsequent bounded read-only audits verified completion markers and exit 0. Do not call this separate-stderr verification or start another login flow.

- Implemented the authorized v2/v2.6 extension for experiments 4–8 without changing original configurations or result identities. Generated configs and all additional evidence use an explicit `extensions/<ID>` scope; 128 new training trials, 250 context tasks and 896 foundation-benchmark shards are planned. The new experiment-0 gate queues 12 correctness and 68 capacity cases plus independent inspection. No extension job has been submitted, no packages installed and no weights downloaded. Actual B200/model acceptance remains pending. Inference context caps are 10,000/50,000; measured training ceilings remain 10,000/11,000 for v2/v2.6, respectively. E7 retains E6's exact 8,192-row control and fixed large-table membership.
- Validation: full suite **1,173 passed, one skipped** (existing absent-manifest test); all **25 notebooks passed, 227 figures**. Independently checked executed cells, summary text against `All_Results.md`, 227 inline/PDF/caption identities and absence of orphan PDFs. Visually inspected 48 synthetic extension figures outside the publication tree and 22 regenerated historical figures covering affected shared layouts. The final submission-quota refinement passed all 28 extension tests. Normalized three older notebooks to the cell-ID schema without changing their executed content; all 25 documents validate. Notebook extension sections explicitly report missing evidence until the new VSC artifacts are downloaded. Reviewed literature at TFM pin `26156f36eb7676e8d9050547817346db548b0e85` and infrastructure at VSC pin `1c29213086a0ae76b8f95cdd8eccbb9db8d6ca87`; both submodules stayed read-only.
- **Tried:** run extension preparation and then read an unrelated historical context ZIP in the same process during integration testing.
  **Result:** the historical reader searched inside the extension scope and missed its original evidence.
  **Why:** preparation/planning left the extension environment selector active after returning.
  **Instead:** wrap both extension entry points in restoring scopes; regression-test restoration and run the historical-reader tests together with extension tests. The complete suite subsequently passed.

- Refreshed local-output audit: both consolidation logs (`62463049`, `62463050`) now independently confirm exit 0. All 6,890 sources/tables in their current inventories match SHA-256; raw evaluation CSV sets match exactly. Across experiments 1/2/3/5/6/7/8, all 7,770 metric files / 38,850 folds have OK status, finite primary metrics, valid prediction receipts and no duplicate comparison/fold identities. The 1,083 new TabICL files exactly match the re-score log paths and contain only 27–28 October results. Decoded all 57,925,718 new prediction rows; 29,925 independent point/quantile metric checks agree within rounding tolerance (largest scaled error 1.994e-7), with matching test cohorts across 165 table/fold panels and no quantile crossings beyond 1e-7.
- All 25 saved notebooks executed without errors/unexecuted code cells; normalized final stdout matches every ordered `All_Results.md` section. All 227 inline figures match their PDFs and ordered caption metadata, with no orphan PDFs, extra figure folders or old `00_general` output paths. Inspected 20 central main/overview figures visually. The ignored October-13 `All_Results (1).md` remains an obsolete duplicate outside the reader path; automatic approval review rejected its deletion with only a generic policy reason, so it was left untouched. The overview table-map caption's claim of identical original evaluations is inaccurate across differing inference protocols; this is a presentation limitation, not a stale raw-result mixture.
- Current predefined-reference TabICL effects after full FP32 re-scoring: PD equal-table mean ΔAUC −0.00158148 (median −0.00058389), LGD mean ΔR² −0.00548490; seed-43 PD mean −0.00184118. Refreshed the research brief's former automatic-policy means and notebook/figure counts. TabPFN v2 PD reference remains +0.00477250, with 17 positive table effects and 16 above the 0.1-pp reporting band; seed 43 gives +0.00474857. No new scientific experiment was implemented or submitted.

- **Tried:** local `reset_family_benchmark --experiment experiment1 --family tabicl --apply` before downloading the FP32 replacements.
  **Result:** Windows `WinError 5` stopped the first directory removal after its files were deleted; experiment 2 was not attempted.
  **Why:** the downloaded directory had the Windows read-only attribute; bare `shutil.rmtree` did not clear it.
  **Instead:** retry Windows unlink/rmdir once after clearing a verified read-only attribute; keep link, real permission and file-lock errors fatal. A real Windows regression covers nested read-only files/directories and an idempotent rerun. Ten focused tests and the full suite (1,147 passed, one skipped) pass; rerun the scoped local cleanup before downloading.

- Read `Downloads/output CreditPFN.zip` (87,421,150 bytes, downloaded 08:51) in place; CRC and member-path checks pass. All 256 resolved task identities for the TabICLv2 FP32 re-score have matching terminal logs. Experiment 1 array `11742938` completed 128/128 tasks, reporting 969 model–table comparisons and 4,845 successful folds, zero failed folds; its last task ended 28-10 at 19:10:17 CET. Experiment 2 array `11742963` completed 112 nonempty tasks plus 16 expected empty LGD shards, all exit 0, reporting 114 comparisons and 570 successful folds. All task configs retain five-fold evaluation, prediction saving, and TabICL AMP/FA3 disabled. No new crash, OOM, non-finite/numerical warning or failed-fold message appears; the old virtualenv warning is followed by activation of the correct CreditPFN conda environment.
- The logs record 1,083 distinct prediction output paths on PROJECT storage. This DATA-only ZIP contains no result or prediction files, so it establishes logged completion, not independent metric/prediction validity. No post-re-score consolidation is present. Re-consolidate `cpt_main_v5` and `cpt_seeds_v5`, obtain fresh DATA plus PROJECT output without mixing superseded local automatic-precision files, then rerun notebooks and audit the updated results. No benchmark resubmission or new training is indicated.

| FP32 benchmark array | Final downloaded evidence | Remaining work |
|---|---|---|
| Experiment 1 · Mindwell `11742938` | 128/128 successful tasks; 4,845 folds OK | Consolidate and verify PROJECT artifacts |
| Experiment 2 · Mindwell `11742963` | 112 successful-work tasks + 16 expected empty tasks; 570 folds OK | Consolidate and verify PROJECT artifacts |
| Main consolidation · wICE `62463049` | User reports successful log after submitting at 08:56 on 29-10; not independently read | Download refreshed DATA + PROJECT output |
| Seed consolidation · wICE `62463050` | User reports successful log after submitting at 08:56 on 29-10; not independently read | Download refreshed DATA + PROJECT output |

## 28-10-2026

- Read `Downloads/output CreditPFN.zip` (87,148,105 bytes, downloaded 09:13) in place; archive CRC and safe-member checks pass. FP32 TabICL re-score of experiment 2, Mindwell `11742963`, has 128/128 task logs ending exit 0: 112 report completed work and 16 are expected empty LGD shards (four per partition, indices 12–15 of each 16-task grid). The 112 nonempty tasks report 114 comparisons and 570 successful folds with no failed folds. Experiment 1 array `11742938` has 35 completed task logs reporting 302 comparisons and 1,510 successful folds, four partial logs and 89 task indices without a resolved-config record at capture. All 167 captured resolved configs specify TabICL AMP/FA3 disabled; no traceback, OOM or failed-fold message appears in these runs. Some five-fold large-table comparisons take about three hours; the four partial tasks request eight hours. DATA logs are a snapshot, not proof of current scheduler state, and this archive contains no PROJECT metric/prediction CSVs; do not interpret the re-score or resubmit from the ZIP alone. Preserve running work, verify queue, and consolidate only after both arrays finish.

| Cluster run | Outcome at 09:13 download | Next step |
|---|---|---|
| TabICLv2 FP32 experiment 2 `11742963` | 112 completed-work tasks + 16 expected empty shards, all exit 0 | Preserve results; verify PROJECT files after the full run |
| TabICLv2 FP32 experiment 1 `11742938` | 35 completed-work tasks, four partial logs, 89 not yet visible | Let existing array run; inspect live scheduler state and final task outcomes |

## 27-10-2026

| TabICLv2 FP32 re-score (source `7988e7a`) | Outcome | Next step |
|---|---|---|
| Reset exp2 `62444529`, wICE | done, exit 0 | Deleted 4 TabICL result folders (342 files) |
| Reset exp1 `62444528`, wICE | refused, exit 1 | Exp2 reset was running; writer guard blocked it (intended) |
| Reset exp1 `62444623`, wICE | done, exit 0 | Deleted 34 folders (2,907 files); no consolidated pointer existed |
| Exp1 `FAMILY=tabicl` array `11742938`, Mindwell | submitted 14:28, 128 tasks | After both arrays finish: consolidate `cpt_main_v5`, `cpt_seeds_v5`; download; rerun notebooks |
| Exp2 `FAMILY=tabicl` array `11742963`, Mindwell | submitted 14:29, 128 tasks | Shares the bounded pool, so it may wait on exp1 lanes |

- **Tried:** submit the exp1 and exp2 `reset-family --apply` jobs within one minute of each other (and of the exp1 preview).
  **Result:** exp1's apply refused because exp2's maintenance job was running; exp2's succeeded.
  **Why:** the writer guard treats any queued/running CreditPFN job as a possible consumer.
  **Instead:** submit one reset at a time and wait for `END exit_code=0` before the next.
- **Tried:** launch `FAMILY=tabicl run_benchmark.sh` from a login shell with the old TabPFNCredit venv / VSC `Python` module active, then after `deactivate`.
  **Result:** the first launch scanned the not-yet-deleted results on project Lustre for over ten minutes; after `deactivate` the module Python lacked pandas.
  **Why:** the readiness check stats every family file when old results exist; `deactivate` left the VSC `Python/3.12.3` module ahead of conda.
  **Instead:** delete first, then `module purge` and `conda activate CreditPFN`, check `which python`, and launch.

- Local experiment-8 artifact audit: all 627 metric CSVs, matching prediction gzip files and evaluation receipts are present; receipt filenames/byte counts pass, and all 33,535,942 stored prediction rows decode with finite targets and point predictions. Each comparison has five successful folds with unique row identities and expected row counts; SHA-256 comparisons of sorted row/target pairs match across models, rates and partitions for every track/table/fold. Recomputed PD AUC/Brier/log loss and LGD RMSE/MAE/R² match all 3,135 fold records within numerical tolerance. Saved quantiles are finite and have no crossings beyond 1e-7. On the warning-affected retention table, all 16 files (12 LoRA plus four cached copies of the original), 340,208 prediction rows and 480 fold-level quantile/interval checks pass; the uncapped likelihood cannot be recovered without distribution outputs. Finite saved values do not establish the counterfactual without overflow. No scientific files or notebooks were modified by this audit.
- Experiment-8 five-fold interpretation: equal-table mean credit ΔAUC for TabPFN v3 / TabICLv2 is -0.000371 / -0.003237 at 1e-5, -0.009511 / -0.010994 at 1e-4, and -0.024917 / -0.039655 at 1e-3 (17 tables). Mean credit ΔR² is -0.024210 / -0.032969, -0.184804 / -0.127327 and -0.192724 / -0.149143, respectively (eight tables). Higher rates decrease AUC on all 17 PD tables and R² on all eight LGD tables for both models. Non-credit primary outcomes also worsen at higher rates; these are descriptive, single-training-seed results of one rank/target recipe. TabICL uses matched FP32 inference in this experiment. All 23 user-rerun notebooks have no saved errors or unexecuted code cells; both experiment-8 final summaries occur in `All_Results.md`, and all 12 inline figures have matching PDF names in `CAPTIONS.md`. No local suite or notebook rerun was performed during this read-only analysis.

- Final experiment-8 DATA audit: `Downloads/experiment8.zip` (5,016,572 bytes, downloaded 08:34:57) passes CRC and safe-member checks. Both benchmark arrays have all 128 distinct task identities and 256/256 completed logs, each exit 0 and `status=OK`; 627 logged comparisons and 3,135 folds report success with zero failed folds. Both prior 24/24 checkpoint audits still pass. Last benchmark log ends 08:07:57 CET. There are no traceback/OOM/failure messages in the benchmark logs. The archive contains DATA logs/manifests only; PROJECT CSVs and prediction payloads remain unverified locally.
- Numerical follow-up: 13 TabPFN v3 comparisons on non-credit `openml_46961` (12 LoRA checkpoints and the shared original) report 832 inverse-power-transform cast-overflow warnings; all 13 log five successful folds and finite displayed RMSE/R². One split-0 rate-1e-3 comparison clamps one non-finite likelihood value out of 4,252 observations to 100 nats. That capped negative NLL is not the uncapped likelihood; the magnitude of any effect on distribution/quantile outputs and point predictions has not been established from logs. Audit PROJECT artifacts and paired predictions after download; no rerun is justified from the logs alone.

| Completed experiment-8 benchmark | Outcome | Next step |
|---|---|---|
| Mindwell `11740933`, wICE `62435037` | 256/256 task logs successful; 3,135/3,135 recorded folds successful | Download both DATA and PROJECT experiment-8 output, verify prediction/metric artifacts and interpret alongside the numerical caveat |

- Read `Downloads/experiment8.zip` (4,887,500 bytes, downloaded 07:52:46 CEST) in place; archive CRC/path checks pass. The final PROJECT checkpoint/trajectory audits `62435028` (PD) and `62435029` (LGD) each report 24/24 complete, zero pending/diverged/problems, passed true and exit 0. Benchmark arrays are Mindwell `11740933` and wICE `62435037`. All 128 wICE tasks and 76/128 Mindwell tasks have completed logs with `status=OK`, exit 0 and zero failed folds; 2,315 successful folds across 463 logged comparisons. Four further Mindwell tasks had partial logs at the snapshot; the remainder had not appeared. No crash or OOM appears. The remaining live scheduler state is not known from the ZIP.
- Four completed TabPFN v3 LGD retention comparisons on `openml_46961` (original and all three LoRA rates) each emit 64 `safe_power_transformer` cast-overflow warnings; all report five successful folds and finite displayed RMSE/R². One high-rate LoRA comparison clamps one non-finite negative log-likelihood observation among 4,252 rows to 100 nats. Preserve this secondary-metric caveat and check the PROJECT predictions/metrics after the complete download; this DATA-only snapshot does not establish prediction-level validity or a scientific result.

| Experiment-8 benchmark snapshot | Outcome | Next step |
|---|---|---|
| wICE `62435037` | 128/128 task logs report OK | Preserve output; inspect result files after download |
| Mindwell `11740933` | 76/128 task logs report OK; four partial logs | Let queued work continue; inspect final scheduler state and results before analysis |

## 26-10-2026

- Completed-training audit: read `Downloads/experiment8.zip` (4,373,159 bytes, downloaded 20:54:56 CET) in place. CRC/member checks, both plan checksums, all 48 identity hashes and exact planned checkpoint-name coverage pass. All 56 resolved configurations retain native rank-16 Q/V LoRA, alpha 32, zero dropout/decay/L2-SP and the intended evaluation precision. The eight manifest files contain 48 successful completions plus eight planned interruptions, all superseded by successful recovery; every completed trial records exactly 10,000 updates, finite final loss/primary metrics/drift and a unique PROJECT checkpoint path. All 56 job logs end with exit 0; 58,344 epoch records have finite losses and zero recorded AMP/data skips, with no traceback/OOM/divergence error. Train/holdout dataset memberships remain disjoint and cover all 17 PD / eight LGD holdouts. This DATA-only ZIP contains no PROJECT checkpoint bytes, trajectories or benchmark results; final on-cluster audits remain required. No training rerun or cleanup is indicated.

| Experiment-8 completed production, source `730cd20` | Outcome | Evidence / next action |
|---|---|---|
| PD, both models × three rates × four partitions | 24/24 complete | Last log ends 16:00:14 CET; all eight segmented recoveries complete |
| LGD, both models × three rates × four partitions | 24/24 complete | Last log ends 18:37:37 CET; no recorded interruptions |
| Benchmark | No downloaded submission/result evidence | Run both PROJECT audits with `sbatch --wait`; submit both benchmark resource arrays only if both audits succeed. No cluster action performed locally. |

- Experiment-8 progress audit: read `Downloads/experiment8.zip` (1,115,724 bytes, downloaded 14:10:54 CET) in place. CRC and member-path checks pass; both 24-trial plan checksums and all 48 embedded trial identities verify. The DATA snapshot contains 27 PD logs and 19 manifest events: 13 unique trials completed exactly 10,000 successful updates, six deliberate segment interruptions, five already superseded by successful resumed completions. Eight latest logs were still progressing at capture. All 12,717 recorded epoch losses are finite, with no nonzero AMP/data skips, crash traces or OOM messages. Completed trials use source `730cd20`; resolved adaptation mode is native LoRA despite the legacy `lora=False` banner. Trainable counts are 860,160 (TabPFN v3) and 466,944 (TabICLv2); peak allocations 83.683 and 25.476 GB respectively on B200. PROJECT trajectories, weights and benchmark results are absent from this DATA-only ZIP, so their contents were not audited locally.
- Early PD monitor effects are descriptive and incomplete: five completed 1e-5 runs change held-out AUC by -0.003574 to +0.007825; all eight completed 1e-4/1e-3 runs decrease it (largest observed drop 0.073588). These fixed-context monitors do not replace the final five-fold benchmark and cannot rank a deployment recipe. Completed PD trial compute is approximately 83–101 minutes, excluding queue delays.

| Experiment-8 production snapshot, source `730cd20` | Outcome | Evidence / next action |
|---|---|---|
| PD split 0, both models × three rates | 6/6 complete | All 10,000 updates; three TabICL trials recovered after planned segments |
| PD split 1 | 5/6 complete in ZIP | TabICL 1e-3 recovery log reaches about 9,776 updates; no failure recorded |
| PD split 2 | 2/6 complete in ZIP | Other four trial logs progressing |
| PD split 3 | 0/6 complete in ZIP | Three TabICL trials progressing at capture; three TabPFN tasks start before the live queue check |
| User's 14:15 Mindwell queue | 10 PD tasks running; 24 LGD tasks pending | LGD array 11739179 awaits priority; arrays 11739180–11739186 await normal dependency chains. No CreditPFN holds. One PD task left the queue after the ZIP; its success is not inferred from disappearance. Let existing jobs continue; do not duplicate submissions. Wice jobs belong to JEVPFN. |

- Reporting-refinement validation: clean full-suite rerun passed 1,132 tests with one pre-existing missing-local-manifest skip (626.19 s). All 23 notebooks completed, producing 185 PDFs (two additional low-rate views); the three reports executed before the final label/marker adjustment were refreshed again. All 14 new/adjusted panels were rendered and visually reviewed; their text stays inside PDF page bounds. Independently verified 185 inline figures/PDFs/captions, all 23 verbatim final summaries in alphabetical order, and zero orphan PDFs. Production training/evaluation code, manifests and submodule pins remain unchanged.
- **Tried:** run the complete local suite alongside the full notebook rebuild after the reporting refinements.
  **Result:** 1,131 tests passed, one skipped, one mocked LoRA-submission test failed at Windows `os.replace` with access denied on a temporary JSON file; its 18-test launcher/submitter group immediately passed independently.
  **Why:** the observed failure is a local temporary-file replacement denial, not a VSC training or gate assertion. The holder of the Windows file handle was not identified.
  **Instead:** retain the validated runtime code and rerun the complete suite; do not change fingerprinted atomic-write code merely to bypass an unreproduced local file-lock failure.
- Reviewed the 11:08:49 Downloads `experiment0.zip` in place: archive CRC/path checks passed, no duplicate members; all eight `lora2` receipts pass, each raw receipt hash matches independent inspection, and all 56 archived debug config/plan bindings match. The workflow digest is `6574c83543317e1a760b8fb502c716e42679de0b3aad83a09bbe794a60d216e7`; source/environment fingerprint `90ea1cacb84631ab930ea04ac1a01bdcf7d7c6a24e56ccaaed1cd13b2d53a310` matches local source after comparing Windows raw config/script files with LF newlines. The DATA archive references PROJECT evidence, but does not contain those result JSONs or weights; they were independently inspected on VSC, not rehashed locally. All ten logs end with exit 0, no tracebacks/OOM/assertion failures. Forty completed per-projection gradient audits pass; twelve interrupted-trial audits are expected recovery probes, not failures. No experiment-8 production job was submitted by the user or this audit.
- External notebook review: PropLGD1's processed target mean is 0.70442018; the histogram already used it while the profile/descriptor read a stale manifest (0.7049). Recompute displayed processed target descriptors from the loaded targets, leaving campaign manifests/data unchanged. Shared prediction panels retain means/IQRs and add individual table circles. Add only two low-rate detail figures (PD AUC and LGD R²) beside the complete main-grid heatmaps; rates are fixed by their numerical values, not favorable outcomes. Preserve all rate/anchor/scope cells, table averaging, inference-policy qualifications and full summary tables. Only visualization/notebook/docs/tests change; the accepted LoRA fingerprint and experiments 1–8 recipes remain untouched.

| Cluster run, source `cfa9658`, workflow `lora2` | Outcome | Evidence / next action |
|---|---|---|
| Prepare `11738977` | passed, exit 0 | Workflow and both production plans prepared |
| Smoke `11738978_0–3` | four passed, exit 0 | PD/LGD × TabPFN v3/TabICLv2; all three rates, native inference, export, scope and recovery checks |
| Capacity `11738979_0–3` | four passed, exit 0 | Four corpus partitions per model/task; per-projection gradients and effective updates pass on B200 |
| Inspect `11738980` | passed, exit 0 | Eight receipt hashes verified; completed 11:07:30 CEST; use `CREDITPFN_LORA_GATE=lora2` for production |

- Final LoRA-review validation: full suite 1,128 passed / one missing-local-manifest skip, exit 0 (559.63 s); 23 final focused launcher/report checks passed, including the additional one-command shell-dispatch test. All 23 notebooks completed and produced 183 PDFs. Independently verified 183 matching inline images/PDFs, caption coverage, every verbatim final summary in All_Results.md and no orphan PDFs. Experiment 8 remains explicitly pending, with no invented result figures. No changed file belongs to the production training identity; existing unused plans remain valid, but the new gate needs fresh VSC receipts.
- Reviewed the external LoRA assessment against source, configs and TFM pin `702d9ab0b66f6639ba8d443ff9c4a91231964465`. Keep the exploratory recipe unchanged: BF16 AMP on B200, balanced PD contexts, zero factor decay/L2-SP. Experiment 5 differs on these axes, so it is not a controlled adapter-only comparison. Ning et al. Appendix B.1/B.2 uses weight decay 0.01; zero decay is a study choice, not a LoRA requirement. The review's claim about a specific number of early BF16-insensitive steps is unmeasured. The temporary cast allocation in `_apply` is harmless and was left alone.
- Strengthened only the debug path: observe finite A/B gradient paths, nonzero B gradients and nonzero effective BA changes separately for every Q and V in actual smoke/capacity training. Initial zero A gradients are valid. Evidence is bound into the existing gate, with native merged-weight scope/export/recovery checks retained. Production training identity and experiment-8 recipe are unchanged. Native LoRA no longer sets the legacy frozen-arm flag; legacy per-tensor plots reject adapter protocols and the LoRA reports distinguish raw factor statistics from effective Q/V movement.
- `run_experiment0_lora.sh all ID` now queues preparation, smoke, capacity and inspection together on Mindwell using successful dependencies and cancellation of invalid downstream dependencies. CPU work uses `batch_graniterapids`; GPU work uses `gpu_b200` (VSC pin `3bcdf55e576d10161d6189b40ac39380bbba9eca`). The scoped submission record preserves job IDs and partial outcomes; duplicate IDs are refused. No production job is submitted by this chain. The experiment-8 launcher activates CreditPFN itself before checking the accepted gate. The user authorized agent pushes on this date; AGENTS.md now reflects that permission. No cluster submission, cleanup or package installation was performed locally.

| New cluster diagnostic, source `a43a221`, workflow `lora1` | Outcome | Evidence / next action |
|---|---|---|
| Inference-only `11738052`, LGD TabPFN v3 | completed, exit 0; no gate granted | All repeat calls and original/zero-export predictions identical. Changed-export FP32 chunk error max 2.4736e-5, one point 1.0855 times its old allowance; original chunk max 3.5763e-5. Automatic-policy checks pass. Fresh gate needed. |

- Inspected the 09:26 Downloads `experiment0.zip` in place: archive CRC passes, no duplicate/escaping members, all 137 LoRA files stay under `lora/lora1/`, and all 56 archived debug config/plan bindings match. Three smoke receipts still pass, one still fails; no capacity receipts or inspection file. This is DATA evidence, not a download of the PROJECT weights/results. No archive, local campaign output, checkpoint or completed experiment was changed. The zero-byte CSV lock files are normal synchronization files, not notebook logs or proof of active jobs.
- Validation of the context-scaled gate change: 107 focused tests passed; full suite 1,111 passed / one missing-local-manifest skip, exit 0 (749.09 s). Regression tests preserve strict repeat and classification behavior, reject nonfinite/shape corruption and meaningful batch shifts, and check context-scale invariance. No model, training recipe, notebook or visualization source changed. No VSC submission, cleanup, commit, push or submodule change was performed. The new gate still needs VSC receipts.
- **Tried:** apply the same pointwise relative tolerance to regression query-batch changes and same-batch checkpoint identity.
  **Result:** one changed-export FP32 prediction failed by 8.55%, although its maximum batch discrepancy was smaller than the original model's; every repeated-call and zero-export prediction was identical. Disabling TF32 did not change the result.
  **Why:** the pointwise allowance shrinks near zero; different query shapes can change floating-point reductions. The evidence supports a batch-rounding false alarm, not a demonstrated adapter/export corruption; it does not identify a specific kernel or certify the unrun LGD training checks.
  **Instead:** set a separate LGD maximum chunk error of 1e-4 context-target standard deviations (fixed synthetic context SD 0.97675, absolute bound 9.7675e-5), shared across checkpoints and independent of query labels/errors. Keep old pointwise diagnostics and all repeat/export/native attention/recovery limits. The configured bound is an engineering criterion, not an upstream guarantee. Require fresh workflow `lora2`, all eight receipts and independent inspection; do not reinterpret `lora1` as passed. Training identity is unchanged, so existing unused production plans do not need resetting for this patch.

| Cluster run, source `b1d121e`, workflow `lora1` | Outcome | Evidence / next action |
|---|---|---|
| Prepare `62427349` | done, exit 0 | Both experiment-8 plans and 28 scoped debug configs created |
| Smoke `11737731_0` / job `11737732`, PD TabPFN v3 | passed | All three rates and recovery checks; DATA receipt passed |
| Smoke `11737731_1` / job `11737733`, PD TabICLv2 | passed | All three rates and recovery checks; DATA receipt passed |
| Smoke `11737731_2` / job `11737734`, LGD TabPFN v3 | failed, exit 1 | Native prediction repeat/chunk/finite assertion, before training comparisons; exact failing checkpoint/check not logged |
| Smoke `11737731_3` / job `11737731`, LGD TabICLv2 | passed | All three rates and recovery checks; DATA receipt passed |
| Capacity array `11737736`, jobs `11737736–11737739` | blocked, exit 1 | Failed smoke receipt rejected before capacity training |
| Inspect `62427400` | failed, exit 1 | No inspection receipt; production not authorized |

- Read `Downloads/experiment0.zip` and `experiment8.zip` in place. These contain DATA evidence only: 18 successful tiny-training outcomes plus nine deliberate recovery interruptions, three passed smoke receipts and one failed receipt. All 58 bound config/plan file hashes, four receipt identities and both production-plan checksums agree. Experiment 8 contains two plans only. PROJECT evidence/checkpoints are referenced but not included, so their contents could not be independently rehashed locally. The user's 08:40 CET queue was empty on both controllers. The legacy `lora=False` debug-banner flag does not describe the native `train.adaptation_mode: lora` intervention; the separate optimizer/adapter diagnostics and receipts do. No completed experiment-1–7 evidence was changed.
- **Tried:** require finite, repeat-call and query-chunk parity in the released TabPFN v3 LGD inference probe.
  **Result:** one combined assertion failed; the other three model/task smoke cases passed and later gates refused execution.
  **Why:** the old message omitted which checkpoint/comparison failed and its magnitude; numerical sensitivity versus an adapter/export defect cannot be distinguished from this DATA download.
  **Instead:** keep the same tolerances and retain `lora1`; add detailed comparisons plus a single inference-only diagnostic of the saved probe exports under recorded precision/TF32 policies. Diagnostic output cannot replace validation receipts or authorize training. No model or production recipe was changed, and no remote cleanup, submission, install or push was performed.
- Local validation: 82 focused checks passed; full suite 1,086 passed / one optional missing-manifest skip, exit 0 (796.64 s). Diagnostic tests cover separate repeat/chunk/nonfinite/shape failures, historical-checkpoint provenance, failed-arm evidence, policy restoration, no training/receipt replacement and active diagnostic cleanup guards. No notebook or visualization source changed; no notebook rerun was needed. Released-model diagnosis is still pending VSC execution.

## 25-10-2026

- Final local validation: full suite 1,067 passed / one optional missing-manifest skip (851 s); the final evaluation-entry safeguard and five additional regression cases passed in a separate 112-test focused run. All 23 notebooks executed successfully, producing 183 PDFs. Verified every inline figure, caption, PDF ownership/paper width and final summary against the rebuilt indexes; no notebook errors or broken README/literature/results-guide links. Experiment-8 notebooks correctly report pending evidence and draw no scientific figures; all 12 new plotting layouts were separately checked with synthetic fixtures. Actual released TabPFN v3 and production GPU capacity remain subject to the new VSC gate. No install, commit, push, VSC submission or submodule change.
- Experiment-8 benchmark jobs now recheck the independent LoRA inspection and resolved evaluation settings before cache/model construction, including after queue delay. Inference precision, CV, ensemble and prediction-output overrides cannot silently change the validated benchmark; output relocation remains allowed. No VSC jobs were submitted.
- **Tried:** complete the first 23-notebook rerun alongside full LoRA regression validation.
  **Result:** the durable test log finished with 1,063 passes, one optional-manifest skip and one obsolete notebook-count assertion; notebook execution stopped after 13 completed notebooks during session interruption.
  **Why:** the old contract still expected 21 notebooks, and the interrupted full notebook run had not rebuilt the aggregate indexes.
  **Instead:** update the contract to 23, verify the focused LoRA checks, then rerun full validation and all notebooks from stable source; do not treat partial execution or deleted indexes as completion.

## 24-10-2026

- Add experiment 8 as 48 separate native Q/V LoRA trials (v3/v2, PD/LGD, four partitions, three rates), with rank 16/alpha 32/dropout zero and no factor anchoring/decay. TabICL original/adapted inference uses FP32/no-FA3. Native exports merge without mutating live adapters; recovery retains their full state. TFM pin `702d9ab0b66f6639ba8d443ff9c4a91231964465`; VSC pin `3bcdf55e576d10161d6189b40ac39380bbba9eca`, both read-only.
- Add four smoke and four production-shape capacity tasks under `experiment0/lora/<ID>/`, independent inspection and `CREDITPFN_LORA_GATE`. Scoped cleanup covers that workflow or experiment 8 on both durable storage tiers and refuses active/unknown consumers or linked paths. A failed gate followed by a runtime change also needs an explicit reset of unused immutable experiment-8 plans before production; completed earlier experiments remain outside cleanup.
- The local package lacks TabPFN v3: structural v3 fixtures and real tiny TabICLv2 models validate local mathematics/integration, but exact released v3/v2 GPU behavior is not certified until the new VSC receipts pass. Freezing TabICL alone changes CPU output by about 1e-7; zero-adapter exact checks match autograd flags, with declared tolerance against original/native inference. LoRA factors retain FP32 masters; movement reports effective Q/V displacement, not A/B-factor drift. Twelve synthetic report types were visually reviewed outside publication output; pending research reports contain no invented curves.
- **Tried:** use OmegaConf.select unconditionally in the new grid validation branch.
  **Result:** four established lightweight configuration tests failed despite unrelated legacy grids being unchanged.
  **Why:** those supported callers supply SimpleNamespace configurations rather than OmegaConf objects.
  **Instead:** detect the optional adaptation mode with ordinary attribute access and apply strict LoRA validation only to the explicit new mode.
- Final full-suite validation: **959 passed, one existing optional-manifest check skipped**, exit 0 (617.07 seconds). The final sampling-results refresh also passed (12 figures); its compact hour ticks and logarithmic-axis caption are saved and indexed. Existing pandas concatenation FutureWarnings are non-fatal; there are no notebook error outputs.
- Final compaction audit: all 21 notebooks executed successfully; 183 saved PDFs match 183 inline figures and ordered captions, with all 21 complete final summaries reproduced in `All_Results.md`. All PDFs retain native 6.75-inch widths and embedded fonts; no missing/orphan figures, broken guide links or error cells. Experiment 1 now has 12/11 training figures and 11/9 result figures for PD/LGD. Reviewed the complete collection visually, inspected revised/suspicious figures at larger resolution and repaired overlapping narrow-log cost ticks. Complete numerical evidence and scientific campaign outputs remain intact; no Git publication performed.
- **Tried:** recover the final full-suite terminal result after the interrupted session.
  **Result:** its process handle was unavailable; the last observed progress could not prove completion.
  **Why:** terminal state did not survive the interruption.
  **Instead:** repeat validation with a durable test log and XML report outside the repository; retain the completed notebook outputs.

## 23-10-2026

- Follow-up notebook revision: main results now lead with final benchmark choice contrasts and complete rate/anchor/scope maps, with common recipe-complete cohorts across metrics and equal table weighting. Positive choice contrasts are explicitly distinct from CPT beating original weights. Fixed-context monitors use change versus learning rate, separate legend groups and equal table weights rather than equal partition means. The fixed-reference text now filters the same predefined recipe as its figure. No LoRA was used: the main grid uses full updates or dominant-transformer-stack freezing; DiagScale remains a separate experiment.
- Compact training section 3 into a matched credit/retention monitor summary; retain native-scale paired optimizer trajectories as complete tables. Combine model movement and summarize observed time/process memory; retain device counters as tables. Preserve the existing template exceptions for output naming, shared figure tree and native ICML widths. Scientific source outputs and training/evaluation code remain unchanged.
- **Tried:** collect the full suite while adding a seed-caption regression fixture.
  **Result:** the run retained the initial fixture without movement observations and failed its single-page assertion; 956 other tests passed.
  **Why:** tests collected before a source edit keep the old function, while traceback lines can display the newer file.
  **Instead:** freeze source before full validation; the corrected fixture and its training/protocol predecessors pass together (101 tests).
- **Tried:** initial full regression run during notebook redesign.
  **Result:** 954 passed, one optional-manifest skip, an obsolete plot-call contract failed, and Windows denied one temporary atomic file replacement.
  **Why:** the contract still mandated the retired constant-baseline figures; the transient Windows filesystem denial was not reproduced by the focused rerun.
  **Instead:** update the notebook contract to the new saved comparisons, retain pairing/metric checks, and rerun both failures plus all new report tests (14 passed); repeat the complete suite after the repair.
- **Tried:** a temporary real-data preview script used `Page.fig`.
  **Result:** it loaded the real PD cohort but failed before saving preview images.
  **Why:** the publication page stores its figure as `Page.figure`.
  **Instead:** use the normal notebook/PDF pipeline for real-data inspection; synthetic layout checks remain outside the output tree.
- Implement the sibling paper's notebook recommendations using completed campaign evidence only. Factorial maps require the same table cohort across every planned recipe; retention first requires all four matched training partitions. Context effect curves retain individual tables, and numerical-policy contrasts pair the same table/fold before subtraction. Table IQRs are descriptive, not fold/seed confidence intervals. Existing outcomes, checkpoints and run identities are preserved.
- Preserve the selected-table diagnostic and unchanged-save evidence; keep boundary probes as tables rather than adding another small-sample figure. Model-specific cost panels and complete partition-ratio tables describe recorded time/exposure, not equal-compute comparisons. Calibrated log loss joins Brier/ECE on a common complete cohort; older exports without calibrated log loss retain their measured panels.
- Keep the existing shared PDF tree and native 6.75/3.25-inch widths. The current arXiv manuscript preserves these source widths. The paper's local figure copies remain version-owned; the new results guide identifies current notebook sources. User explicitly authorizes Git publication for this notebook revision; no new cluster work, package installation or submodule change is authorized or performed.
- Final local validation: 944 tests passed, one existing optional corpus-manifest check skipped. All 21 notebooks executed successfully after repairing the affected visualization; 249 PDFs match 249 inline figures, captions and complete ordered summaries. Reviewed the full collection visually and inspected all 26 new/adjusted retained figures at larger resolution, iterating labels and legends. Removed two duplicate result cost panels; all 35 results-guide links resolve. Scientific inputs and training/evaluation behavior are unchanged.
- **Tried:** pass the new optional provenance argument to every figure sink.
  **Result:** the full suite caught one legacy sink-signature failure after 942 passing checks.
  **Why:** a compatible sink can accept only the original figure/name/caption arguments.
  **Instead:** forward provenance only when a page supplies it; retain the original interface for ordinary pages and rerun validation.
- **Tried:** compact the sampling-cost panels using direct shared-style subplots.
  **Result:** the full notebook run caught a missing pyplot import in that section; completed scientific measurements were unaffected.
  **Why:** this module previously obtained its figures through the campaign wrapper and had no pyplot import of its own.
  **Instead:** import pyplot explicitly, rerun the visualization regression checks and repair the affected notebook before the final publication audit.
- **Tried:** always read per-table counts when labeling an endpoint dashboard.
  **Result:** a regression fixture with only aggregate endpoints lacked the per-table columns.
  **Why:** aggregate-only callers cannot supply an observed table denominator.
  **Instead:** retain the plot, label table support unavailable and leave its metadata empty rather than inventing counts.

## 21-10-2026

| Cluster run | Outcome | Next step |
|---|---|---|
| `inference_review inspect papercheck1`, wICE 62379301 | User-supplied log: 15/15 task audits passed, overall `passed: true`, exit 0; 40-second inspection | Download DATA receipts and PROJECT prediction journals, then compare policy-specific effects; no repeat indicated by this audit |

- The audit verifies stored prediction metrics, row pairing, journal integrity and unchanged-save parity. It does not establish that precision has no effect on the CPT comparison. The environment activator removed the old virtualenv and selected the correct CreditPFN conda Python. This records the pasted cluster evidence; the new numerical results have not yet been inspected locally.
- Local follow-up inspection: all 15 downloaded journal checksums and 114 cells (1,627,068 saved query predictions, including repeated queries) independently pass metric, pairing, integrity and parity checks. All 30 unchanged-save comparisons have maximum probability difference zero. Vehicle Loan mean ΔAUC is +0.071504 automatic / −0.001748 FP32; Myhom +0.014826 / +0.000064; Taiwan Credit Card −0.001780 / −0.001757. No direct FA3 calls are recorded: attention dispatches through PyTorch SDPA with FP16/FP32 tensors; TF32 is off. Retain the original benchmarks, but do not attribute those large automatic-policy gains to precision-independent credit transfer.
- All 21 user-rerun notebooks have no saved errors or unexecuted code cells; all complete final summaries match `All_Results.md` after the runner's documented trailing-padding normalization. All 243 inline figures, PDFs, metadata entries and captions match, with the required paper dimensions. Visually inspected the four finishing-check figures. The original source/partition audit still verifies all 25 tables. Primary reference summaries agree with the prior audit; its complete standard-prediction checks remain recorded under 19-10-2026. The new diagnostics, source sensitivity and table-bootstrap intervals support drafting with the declared limitations rather than repeating the campaign.
- Full local test suite: **936 passed, one optional missing-manifest skip**, exit 0 (697.58 s). No executable, notebook or visualization source was changed in this inspection; the user's executed notebooks were preserved. No new cluster run, install or Git publication was performed.
- Publication scope: user retains the precision diagnostic in the appendix and defers broader precision research to a possible future paper. No further experiment is adopted. Primary-source review confirms related TabICL inference-setting evidence in Ning et al., Appendix A.1 (about 9 Elo above the published frozen default, not isolated as a precision-only effect). In the saved Vehicle Loan predictions, original FP16/FP32 probabilities have mean absolute difference 0.213043; CPT 0.001332. Vehicle Loan has 233,154 rows, Myhom 7,000 and the Taiwan control 30,000, so this is not confined to a tiny dataset. These are post-hoc summaries of existing predictions, not new runs.

## 20-10-2026

- Implement the approved four finishing tasks without adding a main experiment: checkpoint-only TabICL inference review, LGD endpoint/distribution analysis, source/partition sensitivity and conditional table-bootstrap summaries. Preserve all completed benchmarks and checkpoints; new numerical results belong only to experiment0/inference_review. No new cluster run, install or Git publication was performed. VSC numerical evidence remains pending.
- Local validation: full suite 932 passed/1 optional-manifest skip; final focused checks 41 passed. All 21 notebooks executed, yielding 242 figures. The original-plan audit covers all 25 credit tables without same-table or identical-file train/holdout overlap; it does not establish borrower/population independence. All eight LGD tables have paired reference predictions for boundary-error analysis.
- **Tried:** refresh only the PD results notebook after a layout adjustment.
  **Result:** the result-loading cell exceeded its 1,800-second timeout after the full run had passed it in 128 seconds.
  **Why:** the kernel stopped progressing; Windows recorded a thermal sleep/resume during this attempt, but the exact cause is unproven.
  **Instead:** retry only that notebook with one kernel; it passed in 200 seconds. All 21 saved notebooks now validate with 242 embedded/saved figures and complete matching captions/summaries; scientific results were preserved.
- Added the independent read-only VSC Library submodule and daily pin-update proposals; shared HPC guidance moved out of TFM Library, with project instructions retained in docs/VSC.md. No cluster workload was submitted.

What is worth carrying between sessions: **the cluster runs that have been done**, and **the things
that turned out not to work**. Read it before starting; add to it as you go.

Not the changelog — that records edits to the repository. This records experience: what was run,
what came out, and what is already known to fail.

**Keep it short.** One line per run, four per dead end. Newest first, dates `DD-MM-YYYY`. Never
delete an entry: a run you would otherwise repeat and a dead end you already paid for are both
evidence.

Method and research context live in `RESEARCH_BRIEF.md`; operational/storage details and measured caps live in `VSC.md`. The runs table below retains historical headline measurements.

## Complete local evidence and interpretation audit — 19-10-2026

| Local evidence | Verified outcome | Next step |
|---|---|---|
| Merged DATA + PROJECT, experiments 1/2/3/5/6/7 | 4,047/627/1,083/627/297/462 comparisons; all 35,715 outer folds successful | Preserve completed training and evaluation; no new submission indicated |
| Standard prediction exports | 7,143/7,143 decoded; 403,139,023 saved row predictions checked against identities and independently reproduced primary metrics | Interpret equal-table paired effects; these rows are not independent replications |
| Experiment 4 context journals | 250/250 SQLite tasks; all 11,100 cells validated; 294,987,528 saved query predictions | Context, validation and test indices are disjoint; nested contexts and fixed query cohorts agree across states/families |
| User's complete local notebook execution | 21/21 passed; 239 PDFs match 239 inline figures and the complete ordered summaries/captions | Select a small paper figure set from the larger analysis collection |
| Full local test suite | 916 passed, one optional corpus-manifest smoke test skipped; exit 0, 697.02 s | No runtime repair indicated by the tests |

- Read the locally downloaded scientific artifacts in place. All standard metric/prediction receipts agree in filename and byte size; full gzip decoding, finite values, unique out-of-fold row identities, matching targets and complete folds pass. Recomputed AUC/F1, Brier/MSE, RMSE/MAE/R² agree at the recorded numerical precision. Secondary metric ranges pass; 12,000 fold records provide quantile scores, with no recorded quantile crossings. The previously reported inverse-transform warnings do not produce invalid saved primary predictions in these exports.
- All 11 training-plan and 720 trial-identity checksums verify; all 25 credit-table hashes agree across plans and train/test memberships are disjoint. Every planned research trial has a unique successful 10,000-update record with finite loss, score, drift and cost. The 316 historical interruption rows are retained recovery history, not unfinished trials. The skipped optional test requires a data-pipeline corpus manifest absent from this download; general notebooks successfully reconstruct the corpus summary from local data using the existing fallback.
- Experiment 4's prepared-plan checksum, SQLite integrity, exact context/test hashes, nested prefixes, paired original/CPT inputs, cross-family fold inputs and outer-test row coverage pass. Independent AUC/Brier and RMSE/R² checks reproduce the saved metrics. These checks validate the saved query cohorts; they do not establish independence between source populations.
- Rechecked all 21 final summaries verbatim in `All_Results.md`, every metadata/caption/PDF/inline association, filesystem order, PDF widths/heights and absence of stale PDFs. Zero saved cell errors or notebook-defined functions/classes. Visually reviewed the 28 current experiment-5/6/7 PDFs, including the 19 newly populated figures; the preceding publication review covered the other figures. Both Markdown indexes remain tracked. No scientific input or executable source changed in this audit; the research brief now reflects completed local evidence and the paired findings. Read-only TFM-library pin remains `de324efcca55be392feb0c01830b3b6ce364587f`.
- Interpretation: reference PD ΔAUC is +0.004773/+0.000430 for TabPFN v2/v3; TabICL's +0.003695 mean has a −0.000176 median and is concentrated in a few tables. LGD degradation, rate sensitivity, calibration differences and substantial accumulation cost remain genuine findings under the declared protocol. DiagScale's TabICL inference disables AMP/FA3 for both states; its changed original score is not an unmatched within-experiment control. Avoid raw cross-experiment rankings that ignore this precision difference.
- A targeted interpretation concern remains: Vehicle Loan TabICL AUC is 0.583062 original / 0.654566 full CPT in experiment 1, versus 0.656382 for experiment 5's FP32/no-FA3 original; Myhom is 0.579923 / 0.594749 versus 0.594666. Both comparisons retain matching outer rows/targets. The coincident gain is evidence of numerical sensitivity, not proof that AMP/FA3 explains the entire CPT effect or that the main paired evaluation is invalid. Recommend a small original/full-CPT × precision inference check before a mechanistic TabICL claim; no retraining, full campaign resubmission or automatic new job. Recorded this in the research brief for independent review.

**Tried:** Reproduce every stored R² using float64 arithmetic and tight relative-only tolerances near zero.
**Result:** Flagged 35 standard comparisons and four context tasks despite matching predictions and other metrics.
**Why:** The original sklearn calculations used float32; near-zero R² magnifies relative differences. Maximum float64 discrepancies were 3.03e-7 / 4.58e-7 respectively.
**Instead:** Reproduced every flagged R² in the original float32 arithmetic (machine precision / exact); rechecked all affected folds, context cells and paired row identities. Preserve the scientific outputs; do not resubmit or rewrite them.

## Download paused at a second authentication request — 19-10-2026

**Tried:** The two-source PowerShell scp download for DATA and PROJECT output.
**Result:** The pasted excerpt shows completed DATA manifests followed by another login request; its browser challenge is not recognized. PROJECT transfer completion is not established.
**Why:** One scp invocation can open separate connections for its sources. The exact cause of the missing challenge is unconfirmed; reopening the same invalid URL cannot create a new request.
**Instead:** Start a fresh interactive SFTP connection, complete its live MFA request and retrieve both trees in that session. Preserve copied files; do not append-resume potentially rewritten results or rerun experiments. Commands are checked against official VSC/OpenSSH documentation; provider authentication still requires the user. Updated the runbook; no executable source or scientific output changed.

## Final episode benchmark and notebook publication audit — 19-10-2026

| Run / evidence | Verified outcome | Next step |
|---|---|---|
| Experiment 7, Mindwell 11695432 / wICE 62339667 | 256/256 benchmark tasks successful; 462/462 expected comparisons, all five folds passed | Download both output trees for final prediction-level analysis |
| Recovered wICE task 62339667_14, physical job 62340086 | Restart log ends at 09:18:51 CEST, exit 0; final three comparisons / 15 folds successful | No benchmark resubmission or retraining |
| Local notebook runner, four kernels | 21/21 passed; 220 PDFs and 220 inline figures; complete summaries and captions agree | Refresh the complementary PROJECT input before final conclusions |

- Read the 09:22 Downloads ZIP in place: 79,978,498 bytes, 6,172 unique files, clean CRC; all 3,069 JSON parse. The recovered task fills the preceding entry's three missing comparisons. All planned training and benchmark completion evidence is now present; this archive still contains DATA records, not the new PROJECT metrics, trajectories or prediction exports. Preserve completed work and download both trees; final prediction validation, including the previously noted LGD inverse-transform warnings, needs those exports.
- Reran all 21 notebooks against this DATA ZIP and the existing local PROJECT tree. Main-PD results now use 24 figures instead of 55: a common-cohort Brier/ECE overview plus size-selected reliability examples replace 36 calibration pages. Selection uses input row counts independently of performance; complete per-table metric profiles and selection rosters remain printed. LGD results omit classification-only calls; prediction-figure filenames now identify PD/LGD. Missing PROJECT evidence remains unavailable rather than becoming an empty chart or a zero effect.
- Inspected all 220 generated figures in contact sheets and the six changed main-PD panels at full resolution. Direct checks match every PDF to its inline image and caption, every final summary to `All_Results.md`, and both indexes to filesystem order. All PDFs use ICML widths, height at most 4.8 inches and embedded TrueType fonts; no stale PDFs or extra figure trees remain. Shared `general/manifests/scheduler/` is intentional queue-pool bookkeeping, not an obsolete notebook-output tree. Both Markdown indexes remain tracked. The pinned library, training/evaluation source, environments and downloaded inputs are unchanged.
- Verification: **916 passed, one optional missing-manifest skip (692.86 s)**; all 24 publication-contract tests also pass separately. Six new regressions cover common calibration cohorts, exact paired effects, missing evidence, duplicate folds, size-based reliability selection and LGD section scope. Separately inspected 20 synthetic adapter/corpus/episode plotting panels outside the repository; they are layout checks, never research results. README, research brief and runbook now agree on reporting and unified benchmark submission. Final diff checks pass.

**Tried:** Full 21-notebook execution with four parallel kernels.
**Result:** All notebooks passed; one native Windows Jupyter transport message appeared during shutdown after the affected final cell.
**Why:** The message came from libzmq 4.3.5 with pyzmq 27.1.0; its underlying cause is unconfirmed, and no saved cell/output failed.
**Instead:** Repeated the four-kernel closing batch: 4/4 passed without recurrence; independently verified all saved summaries, inline figures, captions and PDFs. No installation or speculative runtime change.

## Episode benchmark: one environment hold remains — 19-10-2026

| Cluster run | Downloaded outcome | Next step |
|---|---|---|
| Experiment 7 foundation benchmark, Mindwell array 11695432 | 128/128 tasks exit 0; 264 model–table comparisons, all five folds successful | Preserve completed results |
| Experiment 7 classical benchmark, wICE array 62339667 | 127/128 tasks exit 0; 195/198 comparisons complete; task 14 has no log/resolved manifest | Release existing task 62339667_14 once; require COMPLETED / 0:0 |

- Read the 09:02 Downloads ZIP in place: 79,975,868 bytes, 6,170 files, clean CRC and unique members; all 3,068 JSON and 97 CSV files parse. Eleven prepared plans and 720 identity checksums verify. The 512/32/96/48/16/16 completed training records for experiments 1/2/3/5/6/7 each have 10k successful updates and finite endpoint loss, test metric and drift. All 25 credit-table hashes agree across plans, train/test/retention memberships are disjoint, and both episode studies retain their exact experiment-6 reference contracts.
- Experiment 7 has 459/462 expected comparisons and no extra/duplicate scientific comparisons or incomplete folds. The missing shard is 2,048-row PD, partition 0: CatBoost on `openml_46952`, logistic regression on Thomas and HMEQ (15 outer-fold cells). All 255 executed tasks use CreditPFN, with zero failed cells, tracebacks, OOMs, timeout or overflow warnings; all 459 prediction exports are reported. Experiments 1/2/3/5/6 retain 4,047/627/1,083/627/297 successful comparisons; experiment 4 has 250 passed completion receipts and all eight follow-up controls remain passed.
- User's 09:05 live queue confirms this is the only remaining job. `scontrol` maps array task 62339667_14 to physical job 62340086: PENDING, environment-retrieval hold, priority zero, no allocation/runtime, Restarts=1. This failed before application startup; the exact site cause is not established. Recommend one release of that existing task, then inspect accounting. Do not cancel/resubmit the whole benchmark or repeat training/experiment 0. If held again, retain the scheduler evidence and investigate with VSC support.
- This is DATA output only: no new PROJECT metric/prediction exports or training trajectories are present. Download both complementary trees after completion, then regenerate notebooks for final interpretation. Only these operational records changed; runtime, environment, weights, library pin and scientific evidence remain unchanged. No cluster action or new test-suite execution was performed locally.

**Tried:** Count benchmark evidence only from logs ending with exit zero.
**Result:** Experiment 1 appeared to lack 13 comparisons.
**Why:** Its historical timed-out attempt saved 13 complete five-fold cells; the successful resume correctly reused them.
**Instead:** Join persisted cell-level successes across attempts and verify prediction receipts; count array-task completion separately. All 4,047 main comparisons remain evidenced.

## Saved predictions and CPT interpretation — 16-10-2026

- Inspected local PROJECT artifacts: experiments 1/2/3 contain 4,047/627/1,083 prediction exports respectively, each with a matching receipt filename and byte size; these are model–table exports, not training-trial counts. Experiment 4 has 250 SQLite files with 11,100 populated prediction cells, including original and CPT states separately. The 09:14 DATA ZIP reports 627/297 prediction exports for experiments 5/6, but their new PROJECT files are not downloaded locally; scheduler success does not independently validate their contents.
- Read-only theory review at `de324efcca55be392feb0c01830b3b6ce364587f` distinguishes statistical bias, repeated-context sensitivity, training-seed disagreement and calibration. Existing five-fold predictions support paired observed-error analysis, not a true conditional bias/context-variance decomposition. The synthetic-mechanism and row-reservoir proposals remain unadopted; no additional training or VSC job is required for these reporting changes.
- New notebook diagnostics verify five-fold coverage, unique and matching test rows/targets, evaluation budgets, prediction receipt sizes and reproduced Brier/MSE before emitting aggregates. Missing PROJECT evidence stays explicitly unavailable; no raw test rows are published. Original/CPT comparisons use the fixed reference or all prescribed study arms; seed 42/43 disagreement stays descriptive. Training/evaluation source, scientific identities, environments and library pin remain unchanged.
- Verification: full local suite **910 passed / one optional missing-manifest skip** (681.31 s); after the label-processing optimization, all 17 prediction-diagnostic tests pass again. Final full notebook rerun: **21/21 passed, 251 PDFs**, including six new prediction panels inspected visually at 6.75-inch width. All 21 final summaries appear completely and in order in `All_Results.md`; captions cover all 251 PDFs. Prediction-level checks verify 100 experiment-1, 100 seed-pair and 300 sampling-arm credit comparisons. The previously computed experiment-1/2 diagnostic tables remain identical after optimization. Final experiment-1 PD and experiment-2 results notebooks take 147/152 s versus 594/683 s on the first pass; this is a local timing observation, not an isolated benchmark.

**Tried:** Applied display-name/redaction functions separately to every saved prediction row.
**Result:** The first report pass completed 20 notebooks; deliberately stopped only the final experiment-3 results kernel after its new analysis had become unnecessarily slow.
**Why:** Millions of repeated labels triggered the same Python/regex work; a 100k-row local check took 1.029 s versus 0.007 s for an equivalent unique-label lookup.
**Instead:** Resolve each distinct label once per artifact, verify identical aggregates, then rerun all notebooks. The interrupted local report is not a VSC failure.

## Final episode-size audits passed — 16-10-2026

| Cluster check | Outcome | Next gate |
|---|---|---|
| Experiment 7, 2,048-row audit, 62339227 | 8/8 complete; zero pending/diverged/problems; passed, exit 0 at 09:06:20 CEST | Submit the combined experiment-7 benchmark once |
| Experiment 7, 26,000-row audit, 62339228 | 8/8 complete; zero pending/diverged/problems; passed, exit 0 at 09:06:22 CEST | Same submission; retain experiment 6's shared 8,192-row reference |

- Read the 09:14 Downloads ZIP in place: 79,323,969 bytes, 5,660 files, clean CRC and unique members; all 2,813 JSON parse. The two audit logs are the only additions to the earlier download. They verify project checkpoints, identities, update budgets and recorded diagnostics. No experiment-7 benchmark logs or new PROJECT result/prediction tables are included.
- The user's 09:20 queue is empty on Mindwell and wICE. Both audits and the eight accepted follow-up receipts pass, so recommend the existing `run_experiment7.sh benchmark` wrapper with gate `b3197869c5974faf83713e3c9072be2c`: 128 GPU and 128 CPU tasks, without retraining or repeating experiment 0. This records readiness and instructions, not a submitted benchmark.
- Training and benchmark source, environment, submodule and downloaded evidence remain unchanged. Today's preceding full local test run remains applicable: 893 passed, one optional missing-manifest skip. The literature review uses read-only library pin `de324efcca55be392feb0c01830b3b6ce364587f`; proposed additional studies remain proposals. Final scientific interpretation still needs the new PROJECT predictions/results and the source-population audit described in the research brief.

## Adapter/corpus benchmarks complete; episode recovery finished — 16-10-2026

| Campaign / jobs | Downloaded outcome | Next gate |
|---|---|---|
| Experiment 5 benchmark, GPU 11693691 / CPU 62333544 | 256/256 tasks exit 0; 627/627 expected model–dataset–partition comparisons, all five folds successful (3,135 fold records) | Download PROJECT results/predictions for final analysis |
| Experiment 6 benchmark, GPU 11693692 / CPU 62333551 | 128/128 tasks exit 0; 297/297 expected comparisons, all five folds successful (1,485 fold records) | Download PROJECT results; preserve the shared experiment-7 reference |
| Experiment 7 recovery, 11693588_1 / 11693589_0 | TabICL resumes at update 5,500; TabPFN v3 at 6,000; both finish 10,000 updates, exit 0 | Audit both episode configurations, then submit the combined benchmark once |

- Read the 08:32 Downloads ZIP in place: 79,292,538 bytes, 5,658 files, clean CRC, unique members; all 2,813 JSON and 97 CSV files parse. Exact planned-roster joins give 512/32/96/48/16/16 successful training identities for experiments 1/2/3/5/6/7, all at 10k updates with finite endpoint loss, score and drift. All 11 plan and 720 identity checksums verify; credit-table hashes agree and training/test memberships are disjoint. Older experiments retain their historical source fingerprints; their expected difference from today's source is not a failed integrity check. The new 80 identities and experiment-6/7 shared-reference contracts match current source.
- Both new benchmark rosters have no missing, duplicate or extra comparisons, incomplete folds or failed cells. Include restarted CPU logs (`r1`) when joining resolved manifests; assuming `r0` would falsely hide ten successful tasks. Last experiment-5/6 completions are 06:33/02:45 CEST on 16 October. Explicit checkpoint paths still select TabPFN v3 and dated TabICL v2. These logs establish coverage, not independent prediction-level validation: this archive has DATA logs/manifests, not the new PROJECT result tables, diagnostics or physical weights. Its tracked notebook reports are not freshly executed analysis.
- Experiment 7 is now 16/16 complete (eight per episode size). The recoveries end at 19:22/19:06 on 15 October. Across 100 experiment-5/6/7 training logs, no traceback, CUDA OOM or nonzero AMP/data-skip counter appears; all eight accepted follow-up receipts remain passed. No final experiment-7 audit or benchmark log is present. User's 08:42 queue has no CreditPFN jobs on either controller; the remaining CreditICL job is unrelated. Run the two CPU audits, require eight complete checkpoints and no pending/diverged/problems in each, then use the existing gate and `run_experiment7.sh benchmark`. Do not repeat successful training or historical experiment 0.
- Investigated 832 TabPFN target-inverse-transform overflow warnings across 13 experiment-5 LGD benchmark logs, confined to non-credit `openml_46961`; every affected comparison still has five successful folds and finite logged metrics. The pinned primary code dump (`de324efcca55be392feb0c01830b3b6ce364587f`, `TabPFN .txt`, `_yeojohnson_inverse_transform` / `transform_borders_one`) records an upstream 8.2.0 rounding-margin fix and nonfinite/extreme-border masking/repair; the cluster used 8.0.8. A local synthetic float32-boundary reproduction yields identical masks/repaired borders before/after that margin. This is mechanism evidence, not reproduction of unavailable cluster inputs/packages. The same warning existed in older benchmarks; no new CreditPFN implementation defect or wholesale rerun is established. Inspect the downloaded PROJECT predictions before final scientific claims; preserve the current environment for the matched experiment-7 benchmark.
- Rounded log means suggest conservative adapter rates stay near baseline whereas `.03` generally degrades credit performance; the large-only corpus has no consistent AUC advantage across the two bases. These are preliminary descriptions, not final paired estimates or significance claims. No executable code, environment, submodule, downloaded evidence or checkpoint was changed; no cluster job was submitted.
- Verification: full local suite **893 passed / one optional missing-manifest skip (791.49 s)**; diff checks pass. No notebook or visualization source changed. Wait for the new PROJECT tables before regenerating final notebook analysis; the supplied DATA ZIP alone cannot certify saved predictions or publication results.

## Adapter/corpus audits passed; episode recovery queued — 15-10-2026

| Cluster check | Outcome | Next gate |
|---|---|---|
| Experiment 5 PD audit, 62333302 | 24/24 complete, 0 diverged/pending/problems, passed, exit 0 | Submit experiment 5's combined benchmark phase |
| Experiment 5 LGD audit, 62333312 | 24/24 complete, 0 diverged/pending/problems, passed, exit 0 | Same experiment-5 submission covers both tracks |
| Experiment 6 PD audit, 62333317 | 16/16 complete, 0 diverged/pending/problems, passed, exit 0 | Submit experiment 6's benchmark phase |
| Experiment 7 partition-1 recovery, 11693588_1 / 11693589_0 | User's 17:22 queue shows both PENDING on Priority; no new attempt log yet | Keep queued; require 16/16 completion and both episode-size audits before benchmarking |

- Read the 17:17 Downloads ZIP in place: 78,241,964 bytes, 4,884 files, clean CRC and unique members; all 2,427 JSON and 96 CSV files parse. The three new CPU audits ran in CreditPFN at 17:11 and verified PROJECT checkpoint presence/identities, 10k update budgets, trajectory coverage, finite credit/retention scores, parameter diagnostics and GPU resource samples. Their 48/16 completions agree with the current expected training rosters. No new training traceback/OOM or nonzero logged AMP/data skips appear.
- Experiment 7 remains 14/16 in this download (8/8 at 2,048 rows, 6/8 at 26,000); the two accepted recovery jobs account for the missing partition-1 identities. Priority is a scheduling wait, not the previous environment-retrieval hold. The separate running CreditICL job is unrelated. No experiment-5/6/7 benchmark log is present, and the user confirms none was submitted.
- Submit experiments 5/6 once, using the existing accepted gate and checkout; their wrappers queue both foundation and classical arrays (128 tasks per resource for experiment 5; 64 for experiment 6). They may overlap experiment-7 recovery. Do not repeat its training submission or historical experiment 0. No executable, source evidence, checkpoint or environment change was made; this was an output/launcher review, with no cluster action or repeated test run.

## Follow-up training completion and two recoverable timeouts — 15-10-2026

| Campaign | Downloaded outcome | Next gate |
|---|---|---|
| Experiment 5, DiagScale | 48/48 OK, 24 per track, all at 10,000 successful updates | Audit both tracks' project checkpoints/diagnostics, then benchmark |
| Experiment 6, corpus policy | 16/16 OK at 10,000 updates | Audit PD, then benchmark; preserve the shared experiment-7 anchors |
| Experiment 7, episode size | 14/16 OK: all eight 2,048-row trials and six 26,000-row trials | Resume only the two partition-1 26,000-row trials, then audit both configurations |

- Read the 16:49 Downloads ZIP in place: 78,211,320 bytes, 4,881 files, clean CRC and unique members. It contains DATA logs/manifests, not new PROJECT diagnostics, predictions or weights. All 11 research-plan hashes and 720 trial-identity hashes verify; all 25 credit-table hashes agree across plans, train/test membership is disjoint, and both experiment-7 reference mappings match experiment 6. Current training source and all eight accepted follow-up receipt fingerprints still agree after source line-ending normalization.
- Every one of the 78 completed new trials has finite endpoint loss/metrics/drift, positive movement and exactly 10k updates; all logged AMP/data skip counters are zero, with no research traceback/OOM/divergence. Experiment-5 monitors favor conservative adapter rates over .03; experiment-6 large-only versus all-table AUC differences are small and model-dependent. These partition-averaged training monitors are not five-fold benchmark conclusions. No experiments 5–7 benchmark log is present. Earlier experiment-1/2/3 training remains 512/32/96 OK; all 256 sampling benchmark logs now end successfully.
- User accounting confirms `11688058_1` (TabICL) and `11688059_0` (TabPFN v3) timed out at 14:59:48 after 1:40:17 on the same node, r11g12. Neither reached the normal saved-interruption/requeue completion. The live logs still end at 14:49:14/14:48:30 with finite losses and zero skips. User confirms their project `.ckpt.resume.pt` files exist (332,477,180 and 640,132,515 bytes, timestamps 14:45/14:41); contents have not been inspected locally. The loader validates full recovery state/identity before continuing.
- Verified a no-submit preview of `SPLIT_START=1 SPLITS=2 STAGES=train SEGMENT_MINUTES=180 CREDITPFN_AUTO_REQUEUE=1` on `config/experiment7/pd_rows26000.yaml`: exactly two single-element arrays, partition 1 only, 3:10 allocations. `SPLITS` is the exclusive upper partition bound, not a count after `SPLIT_START`. Preserve the original checkout, plans, recovery files and gate; do not rerun either whole experiment-7 wrapper or experiment 0. Experiments 5/6 can be audited and benchmarked while these two finish. No runtime/figure/source-evidence edits or submissions performed; 142 focused launch, recovery, follow-up, size-study and benchmark tests pass.

**Tried:** Finish the two partition-1 26k trials through their usual 90-minute segment and ten-minute save margin.
**Result:** Both reached Slurm TIMEOUT without the normal interruption publication/requeue; earlier periodic recovery files remain.
**Why:** The logs do not establish the underlying stall cause. Both stopped logging on the same node near the stop boundary; neither reports OOM, divergence or a Python exception.
**Instead:** Resume only these two original identities with a longer allocation, letting the loader validate saved state. Inspect resumed progress and final CPU audits; preserve successful trials and do not infer a numerical defect from the timeout.

## Merged results and notebook audit — 15-10-2026

| Downloaded evidence | Outcome | Headline |
|---|---|---|
| Experiment 1, DATA plus PROJECT | Complete | 512/512 trained; 20,235 successful benchmark fold records |
| Experiment 2, DATA plus PROJECT | Complete | 32/32 trained; 3,135 successful benchmark fold records |
| Experiment 3, DATA plus PROJECT | Complete | 96/96 trained; 5,415 successful benchmark fold records, including the previously missing classical comparison |
| Experiment 4 | Complete | 250/250 verified task exports; 11,100 original/CPT context cells |
| Experiment 5 | Partial | 32/48 trained (PD 23, LGD 9); completed trials reach 10,000 updates with finite endpoints |
| Experiment 6 | Partial | 6/16 trained; completed trials reach 10,000 updates with finite endpoints |
| Experiment 7 | Partial | 6/16 new trials trained: four 2,048-row and two 26,000-row trials; the 8,192-row control is reused from experiment 6 |

- The user merged both storage downloads locally. Independent calculations from fold CSVs reproduce trained/original AUC, Brier, R² and relative-RMSE effects for experiments 1–3, including the non-credit panel. The prior DATA-only ZIP was insufficient for these conclusions. No source evidence, checkpoints, training configuration or Slurm code was changed.
- At 11:26 the supplied queue shows six CreditPFN jobs running and 28 pending on resources/dependencies; no CreditPFN environment hold remains and wICE is empty. The separate CreditICL job is not counted. Completed LGD adapter trials take about 39–54 minutes, corpus trials 40–42 minutes, and 26k-row trials 2.1–2.4 hours. Remaining dependencies suggest roughly 4–6 hours with prompt allocation, allowing 6–12 hours for scheduling/requeues, measured from that queue snapshot; this excludes the unsubmitted experiments 5–7 benchmarks. Preserve accepted jobs; audit completed checkpoints before those benchmarks.
- The fixed reference recipe improves PD AUC by 0.00477 for v2 (17/17 tables) and 0.00369 for TabICL (6/17; dominated by Vehicle Loan), but only 0.00043 for v3. LGD R² changes are +0.00410/−0.00477/−0.08321/−0.02843 for v2/TabICL/v2.6/v3. These are equal-table descriptive means, not selected winners or independent-trial inference. PD seed differences are small; LGD v3 is more seed-sensitive. PD accumulation costs about 7–17 times one-sample training without a consistent transfer advantage.
- Final verification: **893 passed / 1 skipped** (optional local data manifests absent); all **21 notebooks** executed successfully. Reviewed the regenerated figure set and all changed layouts: **245 PDFs**, correct ICML widths, embedded text, central figure/caption ownership and exact final summaries in tracked `All_Results.md`. Resource scales and partition labels were corrected; absent follow-up benchmarks remain explicitly unavailable.
- Reconfirmed the 30-09 local-input limitation: 23/25 processed tables match saved cluster hashes after CRLF-to-LF normalization; PropPD2 and PropLGD2 still differ. All available cluster plans agree on their input hashes and have disjoint training/test table membership. The general notebooks explicitly describe local inputs; synchronize canonical processed tables before using their corpus descriptors in the paper. Do not replace the active VSC corpus with local copies.

**Tried:** Compare non-credit benchmarks through the shared follow-up notebook loader.
**Result:** Training monitors were visible, but some completed non-credit benchmark panels were omitted.
**Why:** Stored evaluation rows use `noncredit`; the caller requested `ood`, and an exact string filter dropped those rows.
**Instead:** Normalize the two domain labels at the reporting boundary, test real/legacy labels and incomplete folds, and regenerate notebooks; stored metrics and trained weights require no rerun.

## Overnight progress and one held benchmark task — 15-10-2026

- Read the 09:20 Downloads `output CreditPFN.zip` in place: 74,572,935 bytes, 4,779 files, clean CRC, no duplicate members. This is still DATA logs/manifests; detailed PROJECT results, predictions and physical checkpoints are absent. All 12 research-plan checksums and 720 training-identity checksums verify. Main/seed/sampling remain 512/32/96 unique OK trials at 10,000 updates.
- Experiment 3 has all 128 foundation and 127/128 classical task logs completed with exit 0 and zero failed cells: 1,081 model/dataset comparisons, 5,405 successful fold records. Resolved Slurm contexts identify the sole missing element as wICE `62312109_124` (LGD partition 3, shard 12). User's 09:22 queue/accounting confirms it is held for environment retrieval before startup, elapsed zero; recommend releasing only that existing task once, with no outcome yet supplied. No rerun of completed training or benchmark tasks is warranted.
- Experiment 4 has 250/250 passing receipts bound to its verified plan and 250 successful context logs. Matching each log to its receipt gives 11,100 context-size/state cells, paired original/CPT states and constant test-row counts within every task, with no missing task. This certifies execution/receipt coverage, not a local review of the absent PROJECT CSVs/predictions. Those artifacts are needed before interpreting curves in the notebook.
- New training: experiment 5 **30/48 OK** (21/24 PD, 9/24 LGD), experiment 6 **6/16 OK**, experiment 7 **5/16 OK** (four 2,048-row and one 26,000-row trials); every completed trial has 10,000 successful updates and finite final losses/monitored metrics. All new training logs have zero recorded AMP/data skips and no traceback/OOM. Nine adapter and two episode-size interruption records are recovery events. The open 26k trial explicitly resumes update 7,145; the three open adapter trials record update 1,000 and continue logging near download time.
- User's 09:22 Mindwell queue has four running tasks (three adapters plus one 26k episode), three LGD adapter tasks waiting on priority, and remaining work held by expected shared-capacity dependencies. No Mindwell environment hold appears. All eight 26k trials are now accounted for by the completed trial and seven live queue elements, so the previous submission-headroom pause has cleared. Finish training, then audit checkpoints before submitting experiments 5–7's benchmarks; experiment 7 also needs experiment 6's shared weights. Keep active code/environment/output fixed. No runtime edits, extraction, notebook rerun or cluster action performed here.

**Tried:** Execute the complete sampling benchmark as the accepted 128-task GPU and 128-task CPU arrays.
**Result:** 255 tasks completed successfully; CPU element 62312109_124 has no application log and remains held at zero elapsed time.
**Why:** Slurm reports `user env retrieval failed requeued held`; the underlying site/login-environment cause is not established by these files.
**Instead:** Release that single existing task once, inspect its state, and request VSC investigation if the same hold returns; preserve completed results and unrelated queued work.

## Follow-up training submissions reached queue headroom — 14-10-2026

- User's 11:08 transcript confirms prepared-plan checks and accepted submissions for all 48 experiment-5 trials, all 16 experiment-6 trials and experiment 7's eight 2,048-row trials. Experiment 7's 26,000-row plan passes, but its first submission waits at 450 Mindwell tasks; no 26,000-row array ID appears in this snapshot. Experiment 4's submission is not shown in this excerpt.
- This is the launcher's configured `MAX_QUEUED=450` headroom check, not a training failure or proof that the site's hard quota is 450. It counts all of the user's queued/running array elements and checks again every 60 seconds. Leave the submitter running so it can submit the eight remaining single-trial arrays as space becomes available. Already accepted jobs survive disconnection; the unsent portion does not. Do not restart the experiment-7 wrapper, which would submit the 2,048-row arm again; reconcile any later accepted IDs before recovering an interrupted submission. No executable changes or cluster actions performed.

## Follow-up capacity passed; sampling benchmark progressing — 14-10-2026

- Read the 10:52 Downloads `output CreditPFN.zip` in place: 70,151,085 bytes, 3,692 files, clean CRC and no duplicate members. It contains DATA logs/manifests, not current PROJECT predictions, detailed diagnostics or weights. No extraction, output cleanup or runtime edit was needed.
- Workflow `b3197869c5974faf83713e3c9072be2c` now has all eight passing smoke/capacity receipts under `1908e9b`, with matching source/config fingerprint `3a2b8c6bbee6da1274b6fd188f1cd0b65c677e665e61a74bfd92f94374252c73` and environment versions. User's 10:51 cluster inspection passed, including the referenced PROJECT evidence hashes. Preserve this workflow and export its gate ID for every new preparation/submission.
- Four capacity tasks finish successfully in 448/272/181/90 seconds for PD-v3/PD-TabICL/LGD-v3/LGD-TabICL. Twelve real-corpus 16-update probes cover adapters and the distinct corpus/episode policies; 24 original-model context probes cover six sizes through 50,000 rows at production view counts and 4,096-row queries. No error/OOM or nonzero logged AMP/data-skip counter appears. The context probes use 64 synthetic numeric features; they do not establish capacity for every categorical expansion or predict full benchmark duration.
- Experiment-3 benchmark arrays are already accepted: Mindwell `11687925` and wICE `62312109`, each 128 tasks. The ZIP contains 28 completed classical PD tasks, reporting 37 model/dataset comparisons and 185 successful folds, zero failed cells; four GPU and three CPU logs are unfinished. No benchmark traceback/OOM appears. User's 10:58 queue confirms four GPU and four CPU tasks running, all other displayed tasks waiting on `JobArrayTaskLimit`, with no held jobs or dependency blocks. Keep both arrays; do not resubmit them.
- Next prepare experiments 4/5/6; prepare experiment 7 only after experiment 6's reference plan succeeds. Then queue 250 context tasks and 48/16/16 training trials on the same fixed code/environment. New work can overlap experiment 3, but 330 new GPU tasks plus its 128 accepted tasks may reach the launcher's 450-task queue threshold. Wait for each submitter to finish; do not duplicate partially accepted submissions. No repeat of historical experiment 0 is required by this evidence.

## Follow-up correctness checks passed — 14-10-2026

- Read the five supplied Downloads logs in place: CPU preparation 62312115 and all four smoke tasks for workflow `b3197869c5974faf83713e3c9072be2c` succeeded under `1908e9b`. Tasks 0–3 each publish `passed=true` and exit 0, with the same source/config fingerprint and package versions. Current source reproduces that fingerprint using Linux LF bytes; the direct Windows byte hash differs because YAMLs use CRLF locally. No runtime fingerprint or receipt was changed.
- PD/LGD × TabPFN v3/TabICLv2 each finish three reference/resumed pairs at rates 1e-4, 1e-3 and .03: 24 final six-update checkpoints, 12 intentional update-3 interruptions and 12 recoveries. All logged monitored trajectory scores are finite; no error marker or nonzero AMP/data-skip counter appears. Passing receipts cover zero/nonzero adapter loading, original-weight isolation, inference/cache/query-chunk checks, original/CPT five-fold context checks and recovery state/trajectory tolerances. Detailed project evidence is referenced by receipt hash but was not downloaded; the final cluster gate rechecks its integrity.
- Smoke task wall times were 307/229/223/218 seconds for PD-v3/PD-TabICL/LGD-v3/LGD-TabICL. These 256-row checks do not establish production capacity. Next submit only the four capacity tasks with the same workflow/source/environment, then require the eight-receipt inspection before preparing experiments 4–7. Existing benchmark jobs may continue. No code change, training, submission, installation, cleanup or Git publication was performed here.

## Benchmark compatibility after a source update — 14-10-2026

- User already pulled the follow-up source after the completed sampling-training audits. The earlier instruction to finish every old benchmark before any deployment was too restrictive: `check_prepared(stage="eval")` deliberately accepts changed code/environment while preserving original plan/config/identity checks; training resume remains strict. Do not recreate experiment-3 plans or retrain its completed checkpoints.
- Compared the follow-up changes with `72838a6` and the confirmed prior cluster commit `ac248ae`: experiment-3 and shared scientific YAMLs are unchanged; native full-weight inference paths retain their behavior. Adapter loaders, inference overrides and size-study gates are opt-in. Evaluation fingerprints change, so old cached controls may need recomputing; no claim of cross-version bitwise equality follows.
- User confirms VSC pulled `ac248ae` to `717ee24` before audits 62311793/62311794, which therefore passed under the new source. The **09:44 Mindwell/wICE queues are empty**. No experiment-3 benchmark was submitted in the supplied transcript; no in-flight evaluation needs recovery.
- Deploy the pending experiment-7 shell-routing fix before creating any new follow-up receipt. With one fixed checkout, experiment-3 benchmarking and the isolated experiment-0 follow-up checks can overlap. If jobs were already submitted, inspect the queue and finish them before another shared-code update. New experiments still require all eight smoke/capacity receipts and their own plans. This entry supersedes the blanket old-source launch restriction below; no cluster actions or Git publication were performed.
- Verification: **173 focused tests passed (60.26 s)** across launch safety, evaluation/cache sharing, follow-ups and size studies, plus **5 combined-benchmark dispatch tests (5.24 s)**; diff checks passed. Only operational documentation changed this turn; the previous full-suite result remains 886 passed / one optional missing-manifest skip for the unchanged runtime patch.

## Sampling audits and follow-up launch routing — 14-10-2026

- Downloaded wICE logs **62311793 (PD)** and **62311794 (LGD)** both pass: each has 48 expected/completed checkpoints, 10,000 successful updates per trial, zero divergence/pending/problems, sampled GPU resources for every trial and `END exit_code=0`. These audit physical project checkpoints, prepared identities, trajectories and diagnostics. Experiment 3 is ready for its full benchmark under the existing cluster source.
- Follow-up launch review reproduced the shell logger rejecting experiment 7 (or retaining the wrong inherited log group), while maintenance put its preparation log in `general`. Extend both shell routes through experiment 7. Real batch-entrypoint tests cover maintenance, PD/LGD training/evaluation and combined GPU/CPU benchmarks; experiment-1/3 controls preserve earlier behavior. No research YAML, model code, library pin or saved output changed.
- Launch sequence remains old-source experiment-3 benchmarking, then the new checkout and isolated experiment-0 smoke/capacity gates. Once plans pass, all four new studies can be queued together: 250 context tasks plus 48 DiagScale, 16 corpus-policy and 16 episode-size training trials. Prepare experiment 6 before experiment 7 because the latter binds its shared reference plan; their training can overlap. Their final benchmarks follow completed training. No jobs were submitted here; the user owns commit/push/deployment.
- Verification: full suite **886 passed / one optional missing-manifest skip (625.47 s)**, including all 21 real shell routing regressions. Both changed shell files pass Bash syntax checks and retain LF endings; diff checks pass. No notebook/visualization logic changed, so no notebook regeneration was needed. The local fix remains uncommitted/unpushed for deployment after the old benchmark.

**Tried:** Exercise the real batch entrypoints for experiment 7 using a controlled environment-activation failure, without Python training or Slurm submission.
**Result:** All seven experiment-7 cases failed the routing assertion; all 14 experiment-1/3 controls passed.
**Why:** Shared shell log routing and maintenance config detection still matched only `experiment[0-6]`.
**Instead:** Include experiment 7 in both shell routes and retain batch-entrypoint regression coverage.

## Training complete and main/seed benchmarks finished — 14-10-2026

- Read Downloads `output CreditPFN (1).zip` in place (08:34 download, 68,359,395 bytes): 3,457 files, clean CRC, no duplicate members. All six prepared-plan hashes and 640 trial-identity hashes verify. Exact planned-roster joins give **512/512 main, 32/32 seed and 96/96 sampling OK**, every trial at 10,000 successful updates, with no unexpected identities. Earlier interrupted segments are superseded by successful final outcomes; no research training traceback/OOM is recorded.
- Main and seed each have **256/256 successful benchmark shards** across PD/LGD, foundation/classical and all four training partitions. Joining all attempts, including earlier completed cells reused by shard zero, covers every expected model and credit/non-credit dataset with 5/5 successful folds: **4,047 main comparisons (20,235 fold records)** and **627 seed comparisons (3,135 fold records)**. No incomplete fold summary or nonfinite reported benchmark metric appears. Last main/seed benchmark completion logged at 03:33/04:38 CEST on 14 October. This ZIP is DATA logs/manifests only; current project CSVs, predictions, telemetry and physical checkpoints remain uninspected.
- User's **08:43 queue is empty on both Mindwell and wICE**. No sampling benchmark logs or new experiment-0 follow-up receipts exist in this snapshot. Next: audit the 48 PD and 48 LGD sampling checkpoints on VSC, then submit experiment 3's full benchmark using the existing cluster checkout. Preserve that source until benchmarking completes; only then deploy reviewed follow-up changes and run the isolated experiment-0 gates for experiments 4–7. No evidence here requires repeating old training or historical experiment 0. No submissions, source changes, extraction, cleanup or notebook rerun performed in this inspection.

## Final repository audit — 13-10-2026

- Reproduced explicit-corpus selection adding tables to a pinned side when fold/count defaults were inherited, and unknown IDs silently falling back to another population. Enforce exact explicit lists and fail on unknown IDs; six new regressions failed before and pass after the fix. Existing research YAMLs use fixed folds without these explicit-list cases; isolated generated controls specify both sides. This finding is not evidence that completed research runs need repeating.
- Require five valid calibration-bin records per paired source, not merely five successful AUC folds. A synthetic missing-bin fold now produces an unavailable panel. Combine raw/Platt/isotonic reliability plots, use GiB without hidden additive axis offsets and size resource grids to available models. Preserve all source values, complete summaries and caption ownership. No existing benchmark predictions or training measurements were edited.
- PyTorch 2.2 lacks the imported `torch.amp.GradScaler`; the minimum is now 2.3. Declare direct threadpoolctl/psutil dependencies and exclude the canonical output tree from development-tool discovery. The installed environment passes `pip check`; no installation or upgrade was performed. README now covers notebooks through experiment 7; clarify selection bias/CV dependence and replace stale repeat-control instructions with isolated follow-up gates.
- Local literature remains read-only at `de324efcca55be392feb0c01830b3b6ce364587f`. The study is defensible as a descriptive IID transfer benchmark; source/borrower dependence, feature timing, transductive schema filtering and base-pretraining overlap still limit stronger claims. New GPU API/capacity receipts remain required. Do not pull shared runtime changes into the checkout used by accepted old jobs; finish their benchmarks first. No cluster submission, output cleanup, commit, push or submodule edit was performed.
- Verification: baseline **855 passed / one optional missing-manifest skip**; final suite **865 passed / the same skip (493.01 s)**, plus **six focused plot regressions** after final layout refinements. All **21 notebooks execute**, with the five affected notebooks rerun successfully after the last refinement. All **208 PDFs** are single-page at ICML full width, visually reviewed, captioned and free of private-name text; every ordered final summary is reproduced in the tracked `All_Results.md`. No orphan figure tree remains. Python parsing (139 modules), Bash parsing/LF checks (31 scripts), local Markdown links, Git output exceptions, installed package requirements and diff checks pass. Analysis still uses the supplied DATA ZIP and older local project tree; notebook execution does not certify unreceived benchmark results. The latest VSC snapshot additions concern Sofia, not the Mindwell/wICE deployment.

**Tried:** Explicitly pin only the training or test list while retaining inherited K-fold/count settings.
**Result:** The original implementation expanded the pinned list; unknown IDs could fall back silently.
**Why:** The remaining-table split populated both sides and invalid explicit IDs were removed before mode selection.
**Instead:** Preserve supplied memberships exactly, fill only the unpinned side and reject unknown IDs before splitting.

## Separate corpus policy and episode budget — 13-10-2026

- User asks to separate the two questions previously crossed in the unrun experiment 6. Keep the original four PD holdouts, then filter only each training complement. New experiment 6: all/large (>10k rows), fixed 8,192-row cap, v3/TabICLv2, full LR 3e-7/lambda .003/seed 42/10k updates: 16 fresh trials. Experiment 7: identical large pools at 2,048/8,192/26,000 total episode rows, reusing eight experiment-6 anchors: 16 fresh trials, 24 analyzed cells. Combined fresh training falls from 48 to 32; the reduced design no longer estimates the full corpus × episode interaction.
- Local unfiltered/large training counts are 13/9, 13/8, 12/7, 13/9, with 4/4/5/4 unchanged test tables. All local large tables fill 26k; VSC preparation must verify its own bytes/counts. New guards reject short eligible tables, fewer than three training tables, changed test assignment, silent runtime/cell truncation and incompatible reference data/base/config/code/environment. Reference plan hashes and exact trial mappings protect shared-control reuse; scoped cleanup preserves experiment 6 while experiment 7 references it.
- Reread the full Real-TabPFN primary text and TabICLv2 training section at unchanged read-only pin de324efcca55be392feb0c01830b3b6ce364587f. Large-table eligibility is a composition/exposure policy; episode size changes both context and query counts. Neither experiment isolates a universal table-size effect or holds GPU cost fixed. Literature and research brief now separate these estimands and prioritize completed paired evidence/source-aware uncertainty over more grids for publication.
- New experiment-6/7 training/results notebooks use the shared PDF/caption/summary pipeline and require matching plans, arms and complete five-fold outcomes. Experiment 7 reads its anchor directly from experiment 6. Shared controls also freeze resolved benchmark defaults/code/package versions; evaluation rejects changed views, caps or folds inside the job. Follow-up gates include shared YAML defaults and both benchmark wrappers. Isolated experiment-0 capacity tests now exercise both 8,192-row policies and the two other caps; they verify actual row exposure. No new VSC runs, installs, commits, pushes, output deletion or library edits; old campaign YAMLs preserved. Do not deploy changed shared-runtime fingerprints while accepted old jobs still need their checkout.
- Verification: final full suite **855 passed / one optional missing-manifest skip (504.71 s)**; focused follow-up/design/benchmark tests **75 passed**. All **21 notebooks execute**, including four new size-study notebooks, followed by another successful four-notebook pass after wording refinement. Verified all 21 complete ordered summaries, 220 captioned PDFs, no orphan figure folders/PDFs and no invalid UTF-8 replacement characters; shared reports remain eligible for Git. Visually inspected all ten synthetic size-study figure layouts in temporary storage, separate from research output. New studies correctly show pending evidence with no invented empirical panels. Analysis uses the unchanged 14:13 DATA ZIP plus the older local project tree, not a new complete PROJECT download. Diff checks pass; existing experiment-1/2/3 YAMLs and the library remain unchanged.

## Follow-up notebooks and afternoon progress — 13-10-2026

- Read the new 14:13 Downloads ZIP in place: **3,191 files**, 67,381,555 bytes, clean CRC, no duplicate members. This remains DATA logs/manifests only, not project trajectories/results/weights. Exact planned-roster joins give main **512/512**, seed **32/32**, sampling **90/96** completed, all at 10k updates; sampling is 46/48 PD and 44/48 LGD. The six absent outcomes are all TabPFN v2. No research traceback, CUDA OOM or divergence appears; absent outcomes are not live scheduler states.
- Main and seed classical logs now cover all 64 shards per track successfully, including the previously incomplete seed LGD coverage. Their foundation benchmarks cover all 16 shards in partitions 1/2 per track; partition-0 main probe logs are not full arrays, and partitions 0/3 remain unconfirmed. No sampling benchmark log is present. Scheduler confirmation is recorded below. Do not rerun old training/experiment-0 controls or replace accepted recovery jobs on this evidence.
- User's 14:52 queue confirms one sampling training task and four main benchmark tasks running on Mindwell, with four training tasks and remaining benchmark arrays waiting on dependencies/array limits. No environment-retrieval holds appear; wICE is empty. Retain these jobs. Sampling benchmarking must still follow completed/audited sampling training using the existing cluster checkout before deploying follow-up runtime changes.
- Added experiment4/01_context_curves and experiment5/{01_training,02_results}. Context analysis requires matching plan/task receipts, paired row hashes, complete exports and five folds; fixed-cohort horizons prevent changing dataset composition from masquerading as context gains. DiagScale uses adapter-specific movement/optimizer labels and matched-rate table support. Original/adapter inference comparisons stay inside experiment 5. Empty new campaigns report pending evidence, with no synthetic research results.
- Clarified experiment 6's 48 fresh PD arms and deployment order: finish accepted old jobs before changing their shared runtime; run the new four correctness and four capacity tasks, then prepare and submit follow-ups. No full repeat of historical experiment 0. Local eligible training counts are 13/13/12/13 for all tables and 9/8/7/9 for large tables; VSC preparation remains authoritative because its processed bytes differ locally.
- Checked the adjacent manuscript's ICML 2026 template and current official author/reviewer guidance. Eight main-text pages at submission, not ten; next-edition rules/dates were not confirmed. Literature read only at pin de324efcca55be392feb0c01830b3b6ce364587f; no library, scientific configuration, Downloads file or checkpoint edits, no installation, submission, commit or push.
- Verification: full pytest **825 passed / one optional missing-manifest skip (538.30 s)**; presentation refinements pass **104 focused tests**, followed by all **10 new report tests** after final marker changes. All **17 notebooks execute successfully**, producing **220 captioned PDFs** and a 10,264,597-byte shared summary. All 17 final summaries match verbatim, sections/captions are sorted, no orphan PDFs remain, shared reports are not ignored and notebook/report redaction is stable. Visually checked all 18 synthetic follow-up figure variants in temporary storage and the three actual new coverage/schedule figures. No experiment-4/5 outcomes exist yet: their empirical panels correctly remain unavailable. Notebook inputs combine this DATA ZIP with the older already-local project tree; they are not a fresh complete project-results download.

## Isolated follow-up implementation — 13-10-2026

- User replaces the old numbering with context-only experiment 4, DiagScale experiment 5 and large-PD experiment 6; generic-data and prevalence proposals are excluded. No new cluster jobs, installations, commits, pushes or output deletions were performed. Existing research YAMLs remain unchanged.
- Context curves use the fixed full-parameter experiment-1 recipe (LR 3e-7, lambda .003, seed 42, 10k updates), matching each table's excluded partition; 29 nested context sizes (8–50,000), unchanged complete five-fold test sets and a separate fixed validation-label budget. The 50k cap bounds compute, not v3 capability. Explicit v3/datestamped TabICL paths and hashes prevent default-model substitution; both local v3 headers identify tabpfn_v3, a 1M sample guard and no row subsampling.
- TTC primary text and relevant released adapter/attention/training/cache paths inspected at read-only pin de324efcca55be392feb0c01830b3b6ce364587f. Adapters require optimizer-only clipping, original TabICL autograd flags, full gradient clearing, unchanged bases and inference reload hooks. Experiment 5 uses FP32, provisional 8,192-row episodes, three rates (1e-4, 1e-3, .03), 48 trials and precision-matched TabICL base/adapter inference. This is a new CPT recipe, not a literal target-table fine-tuning reproduction or a one-factor comparison with old full-FT.
- New experiment0/followups workflow isolates four small correctness tasks, then four capacity tasks. Production 4–6 launchers require all eight matching receipts and intact detailed evidence. Capacity tests are unrun; synthetic adapter tests cannot certify the installed VSC package hooks or production GPU memory. Finish accepted old jobs before deploying changes to fingerprinted shared runtime; do not replace old plans to make them pass new hashes.
- Cleanup previews experiment/workflow ownership on DATA and PROJECT, including trained/recovery weights, then requires explicit deletion. It refuses active/unknown writers and linked paths; no cleanup was run. Shared publication summaries need regeneration after a local scoped cleanup.
- Validation: the full scientific implementation passes **814 tests / one optional missing-manifest skip in 552.41 s**; the final cross-experiment cleanup dependency guards additionally pass **40 focused tests**. Bash parsing, Python compilation and diff checks pass; existing experiment-1/2/3 YAMLs and the library pin/tree are unchanged. No notebooks/visualization source were edited or regenerated in this implementation. Adapter tests cover family-specific dtype promotion/casting; VSC probes include nonzero fresh/cached attention and a visible reload perturbation. The initial 802-pass run exposed one stale small-batch assertion, repaired as described below.

**Tried:** Apply the old all-experiment recovery/small-batch test unchanged to the new configs.
**Result:** The valid experiment-6 2,048-row ablation failed the assertion.
**Why:** That historical invariant coupled batch size to recovery instead of the declared experiment design.
**Instead:** Preserve its original campaign scope and separately test the new episode grid, corpus filters, explicit model versions and phase gates.

**Tried:** Infer v3's row guard from generic package defaults during the context-cap review.
**Result:** The actual pinned classifier/regressor headers declare 1,000,000 rows, not 50,000.
**Why:** Post-v2 inference configuration is stored in the checkpoint; the nearby generic 50k configuration belongs to v2.5.
**Instead:** Inspect both explicit v3 checkpoint headers, keep the study's 50k cap as a compute choice, and correct the explanation before any cluster run.

## DATA snapshot and follow-up design review — 13-10-2026

- Read the 09:22 Downloads ZIP in place: 3,161 files, 66,782,546 compressed bytes, clean CRC and no duplicate members. It contains DATA logs/manifests only; no project training histories, result CSVs, predictions or weights. All six research plan checksums and 640 trial identity checksums pass. Planned-roster joins give main 512/512 OK, seeds 32/32 OK, sampling **83/96 OK** (43 PD / 40 LGD), two interrupted and eleven unrecorded. No research traceback, CUDA OOM, nonfinite logged loss or divergence is recorded; historical intentional exit-75 segments remain.
- Main and seed foundation benchmark logs now have all 16 successful shards for partitions 1 and 2 per track, reporting zero failed cells; main classical logs cover all four partitions, seed PD classical covers all four and seed LGD has 59/64 successful shards. Partition-0 main profiling logs are not full-array coverage. Project result artifacts are still required to certify complete paired predictions/receipts; no sampling benchmark logs are present.
- User's 09:52 queue confirms three sampling tasks running and ten dependency-pending, plus four remaining main and four seed GPU benchmark arrays dependency-pending. Five wICE classical tasks are held by environment retrieval (`62295621_3`, `62295622_10`, `62295620_13`, `62295622_9`, `62295619_10`); recommend releasing those existing tasks once, preserving the accepted campaign. No cancellation, resubmission, training, installation, commit or push performed.
- User excludes proposed experiment 4. Refine designs 5–8 in the existing research brief; experiment 6 is favored but not implemented/authorized for launch. New library pin `de324efcca55be392feb0c01830b3b6ce364587f` contains TTC code; verified architecture-specific scaling and TabICL autograd-flag/optimizer/clipping separation. Real-TabPFN's raw mean AUC gain is approximately .003, versus .022 normalized. Local large-PD eligibility is 9/8/7/9; one LGD partition has none, so propose PD-only corpus × episode-size study. Keep all active numerical/configuration code and downloaded evidence unchanged.
- Validation: **787 passed / one skipped in 601.33 s** (optional unbuilt local data manifests); documentation diff checks pass and the submodule working tree/gitlink remain unchanged. Only the existing research brief, literature notes and dated logs were edited; no notebook rerun is required for these documentation-only changes.

## Merged-output publication audit — 12-10-2026

- User-downloaded merged DATA/project tree: main 512/512 and seed 32/32 training OK; sampling 64/96 OK (34 PD, 30 LGD), nine interrupted and 23 unrecorded. Main five-fold benchmark contains 3,650 PD / 2,525 LGD fold records, all OK, but complete paired factorial credit support covers only partition 2 (five PD / two LGD tables). Classical main controls cover all four partitions. Seed contains 15 classical PD folds and no trained benchmark; sampling contains no benchmark. No traceback/OOM is recorded in the 751/60/237 main/seed/sampling logs. File snapshots are not live completion.
- User's 14:47 queue: six sampling training and four GPU eval tasks running on Mindwell; eleven classical tasks running on wICE. No environment-retrieval holds appear; remaining tasks await dependencies/array limits. Keep accepted jobs and their recovery state; do not rerun experiment 0 or restart completed training.
- At the complete partition-2 support, full-update PD mean AUC changes are +0.00064 (LR 3e-7, lambda .003), -0.00932 (1e-5), -0.01176 (3e-5), versus -0.03691 at 3e-5 without anchoring. Low-LR frozen LGD shows a tiny positive RMSE reduction; full-update references are negative on average. These seven tables are not the full benchmark population and do not justify winner/null claims. The full monitor grid also shows increasing deterioration at large learning rates.
- `All_Results.md` was ignored by its entire parent folder; unignore only it and the shared caption index. CSV text preserves indices/rows/columns and eight-significant-digit formatting, reducing the initial 14,564,132-byte report to 9,986,880 bytes (approximately 9.99 MB). Full 14-notebook execution passes, then the affected sampling training notebook passes again after explicit unavailable-cohort annotations. All 215 PDF links/captions and every notebook's final summary match; seven private aliases are available and the reports pass a redaction-stability scan. Scientific outputs/PDFs remain ignored. No raw data, weights, Downloads files or temporary review sheets are added to Git.
- Final validation: **787 passed / one skipped in 587.42 s** (optional unbuilt local data manifests); seven focused publication/workflow checks pass. All 14 notebooks execute, with the affected sampling notebook rerun after the final annotation refinement. Visually reviewed all 215 PDFs through contact sheets and the repaired partial panel at full size. `git diff --check` passes; all 14 saved notebooks and the two shared reports pass private-name redaction checks.
- Read closest CPT/adaptation/split papers at library pin `1ac6e6783f24b2d5cd058af7ad8cad7c1bfaf7ae`; requested October TTC paper is absent from that pin, so read its primary arXiv HTML and GitHub source at `45fb141b2acf15518e70885aa149c8e1898cd15c`. Add proposed appendix controls to the existing research brief/literature document. No new experiments, installs, submissions, numerical source edits, commit or push. Preserve the current cluster checkout while its accepted eval jobs run: the earlier local notebook-runner edits are included in its broad eval fingerprint.

**Tried:** Full local pytest in the default and a fresh dedicated Windows temporary directory.
**Result:** Both full runs hit WinError 5 in one instantaneous mocked callback workflow; the second reported 785 passed / one skipped / one failed. An isolated fresh-root retry and a standalone 100-replacement probe passed.
**Why:** Atomic destination replacement was denied; the destination was not ReadOnly and no holding process was identified. The callback test already mocks locks/state and does not consume the saved bytes.
**Instead:** Spy on complete state publications in that workflow-order unit test, retaining every callback/advance assertion; separately inject denied replacement and verify original bytes survive and temporary files are removed. Do not hide runtime permission failures or alter the active Linux publication code.

## Campaign return audit — 12-10-2026

- Seed checkpoint audits **62295163 (PD)** and **62295164 (LGD)** both pass on wICE: 16/16 completed checkpoints per track, every trial at 10,000 updates, zero divergence/pending/problems and `END exit_code=0`. The cluster resolves weights under project `checkpoints/trained/experiment2/<track>/`; its environment guard removes the inherited virtualenv and activates CreditPFN successfully. Experiment-2 benchmarking is ready; these receipts certify training artifacts, not final benchmark completion. Logs were read in Downloads without copying or extracting them.
- User's 11:13 queue snapshot confirms that all six released sampling roots are **RUNNING**. Release succeeded; retain their recovery state and dependent arrays. The 10:52 balance snapshot reports 42,703,920 available credits. Neither application completion nor a current remaining balance follows from this startup snapshot. Experiment 2 may be benchmarked independently of experiment 3 after both seed checkpoint audits pass; compatible main controls should be published first where possible to avoid duplicate computation.
- Subsequent local verification: **14/14 notebooks execute**, then all eight research notebooks pass again after correcting garbled Unicode labels in the dashboard source. The shared output contains **168 captioned PDFs** and all 14 complete final summaries; inspected all 62 available seed/sampling figures and scanned every PDF for the repaired encoding artifacts. Inputs are the latest DATA ZIP plus the older local project-output copy, explicitly selected without extraction; no follow-up final benchmark results are present. Full pytest now passes **782 tests / 1 skipped in 512.44 s** (optional local data manifests). Five new shell tests check complete benchmark routing, invalid indices and dry/failed readiness without Slurm submission. Training identity remains `86517b7df7d61736c7b026d986dad635868824ed8ef97ff76ff1f257ea248468`. Local combined-launcher/presentation edits remain uncommitted; use the existing cluster launchers while this campaign is active, preserving its evaluation caches and source identity.
- Read the Downloads ZIP in place: 2,555 files, 64,002,641 compressed bytes, clean CRC. This contains DATA logs/manifests, not project trajectories, benchmark artifacts or weights. Latest measured trial outcomes: main **512/512 OK**, seeds **32/32 OK**, sampling **63/96 OK** (33 PD, 30 LGD); sampling also has nine saved interruptions and 24 identities without an outcome. All completed research trials report 10k updates. Main checkpoint audit receipts pass 256/256 per track; no seed checkpoint audit is in this download.
- Main benchmark logs show 64 successful full-array classical task logs per track plus the two earlier probes; foundation arrays completed partition 2 (16 tasks per track), with three other partitions still pending. Final pipeline summaries report zero failed cells. The earlier PD profiling timeout has a successful resumed successor. No Python traceback or CUDA OOM appears in the experiment-1/2/3 logs. Project result receipts/predictions must still be inspected before certifying benchmark coverage or interpreting effects; log counts are not unique scientific comparison counts. No follow-up benchmark logs are present.
- User's 10:45 scheduler snapshot: **129 pending Mindwell tasks**, zero running; wICE empty. Six held PD sampling roots (`11644188_2`, `11644199_5`, `11644200_2`, `11644201_8`, `11644206_9`, `11644225_8`) block 27 later training tasks and 96 main evaluation tasks through whole-array capacity dependencies. Last saved updates at those roots are respectively 1,911 / 2,028 / 1,059 / 3,762 / 8,925 / 1,371. The 10:52 job report for `11644188_2` confirms priority 0, no allocation, restart count 4, 24 CPUs/120 GiB/one B200 and an implicit retrieval request through `--export=ALL,<variables>`. Recommend releasing only the six existing roots once; user action/outcome remains pending. No cancellation, resubmission or experiment-0 repeat is justified by these holds alone.
- Reviewed the updated VSC snapshot's KU Leuven topology, hardware, storage, native-filesystem staging, resource/accounting and Slurm environment/dependency guidance at pin `1ac6e6783f24b2d5cd058af7ad8cad7c1bfaf7ae`. Current routes already use wICE/Mindwell; Genius occurs only as a historical activation anecdote and compatibility alias. Preserve numerical source while jobs remain queued. Current training identity `86517b7df7d61736c7b026d986dad635868824ed8ef97ff76ff1f257ea248468` matches every seed/sampling plan; main plans retain their earlier identity. Existing uncommitted notebook/dashboard/bundled-benchmark edits predate this return audit and remain a separate unfinished task. No runtime changes, Downloads extraction, installation, training, submission, commit or push in this audit.
- Validation: all 640 prepared identity checksums and six plan checksums pass; 23 shell scripts parse in Bash; `git diff --check` passes. Full pytest: **775 passed / 1 skipped / 1 failed in 604.31 s**. The sole failure asserted the superseded `cp.plot_secondary_tradeoff` notebook call. Updated that regression to require the compact paired endpoint pages, PD Brier page and secondary metric profile, adding an actual paired Brier direction check; the entire affected regression file then passed **70/70 in 11.87 s**. The full suite was not rerun after this test-only repair; no notebook or visualization source was edited or re-executed in this audit. The unfinished presentation changes still need their own execution/visual review before deployment.

**Tried:** Continue automatically requeued sampling tasks while later training and main benchmark arrays share the same capacity pool.
**Result:** Six tasks were held before application startup; their dependency chains remained pending through the user's trip.
**Why:** Slurm could not retrieve the login environment. Explicit-variable export implicitly enables retrieval; the exact login/node/site cause is not established from client evidence.
**Instead:** Release the six roots once, preserve their recovery files and downstream IDs, and escalate repeated retrieval holds with the job reports rather than restarting completed experiments.

## Notebook scheduling and refreshed local coverage — 01-10-2026

- The old runner already used a two-process pool with `--workers 2`, but preparation and all execution were silent until the final report. Added flushed preparation/start/cell/completion messages, 15-second running-cell heartbeats, longest-previous-runtime scheduling and one numerical-library thread per kernel by default. Each kernel explicitly uses the invoking Python interpreter. Worker exceptions are reported without losing other completed notebooks; invalid worker counts fail before clearing outputs. No new logs, locks or output folders. Alphabetical summaries/captions and figure ownership are unchanged.
- Read current merged local output, independently of the older morning Downloads ZIP: experiment 1 has 512/512 OK training trials and 100 PD/130 LGD successful benchmark folds; experiment 2 has 32/32 OK training trials and no benchmark rows. Experiment 3 has seven completed PD trials, nine interrupted PD and twelve interrupted LGD records, with 68 planned trials unrecorded locally. This is recorded evidence, not live Slurm state. No trained model weights are downloaded locally; they belong on project storage. Main benchmark logs cover the four initial split-0/task-0 profiling jobs, including successful recovery of PD job 11644185 (six remaining comparisons, 30 folds, 5,715.3 seconds); they do not certify full arrays. No complete paired credit or non-credit benchmark effects exist in this local snapshot.
- Keep these runner edits local while continuing benchmarks: evaluation fingerprints include all of `src/utils/`, so deploying the analysis-only change invalidates old evaluation keys despite unchanged numerical behavior. Training identity is unchanged. No cluster submission, training, installation, commit or push.
- Validation: **776 passed / 1 skipped in 650.28 s** (optional data manifests); all **14 notebooks passed with both two and four workers**, producing the same **365 PDFs**. Four-worker measured wall time was **406.3 s**, versus approximately **620 s** with two on this machine (16 logical CPUs, 63.4 GiB RAM). Real-kernel regression checks verify overlapping execution, distinct kernel PIDs, the invoking interpreter, actual one-thread BLAS pools, heartbeats, failure isolation and longest-first scheduling with sorted reports. Verified every caption link and final notebook summary; no stale extra PDF or notebook error output. Downloaded main checkpoint audits report 256/256 completed and passed per track; no seed checkpoint audit is present locally yet. The user clarified that the four profiling jobs were understood as the full arrays; clarified the 16 shards × four dataset partitions and retained-result reuse in `VSC.md`.

## Publication layout and current output check — 01-10-2026

- User now requests experiment subfolders inside the sole `output CreditPFN/figures/` tree. Corpus figures use `00_general/`; per-notebook caption JSON lives inside `_metadata/` beside the PDFs. Root `All_Results.md` contains all saved notebook summaries; `figures/CAPTIONS.md` contains every caption ordered by notebook folder/name and drawing index. This supersedes the older flat-file and experiment-manifest figure layouts below. Recognized legacy artifacts migrate; reruns prune empty directories, including Windows ReadOnly download folders. Scientific logs, training measurements and benchmark records are preserved. Removed obsolete local PNG review sheets and downloaded scheduler-state/lock copies; the local `general/` tree is gone. VSC may still use `general/` for shared maintenance logs and its live capacity pool.
- Read the latest Downloads ZIP in place (1 October morning snapshot, 1,851 entries, CRC check passed). Main training remains 512/512 OK, seed training 32/32 OK; sampling has 24 interrupted and 72 unrecorded outcomes. All 590 main log files lack traceback/OOM/time-limit markers. Existing local project results contain 100 PD and 130 LGD final fold records, all OK, but no complete paired held-out credit benchmark effects. This is a partial output snapshot, not live scheduler evidence.
- Endpoint monitoring suggests learning-rate-dependent deterioration on credit and non-credit data, usually smaller when the main transformer stack is frozen. Across the grid, PD held-out mean AUC change is +0.00027 at 3e-7 versus -0.03022 at 3e-5; LGD held-out mean fractional RMSE reduction is -0.01405 versus -0.09079. These are equally weighted descriptive monitor aggregates across architectures/adaptation/lambda settings, not final five-fold benchmark estimates or independent-seed inference.
- The notebook runner now routes summaries through publication-only path helpers. Numerical training identity remains `86517b7df7d61736c7b026d986dad635868824ed8ef97ff76ff1f257ea248468`; engines/configs/data/weights are unchanged. The broad evaluation fingerprint includes `src/utils/run_notebooks.py`, so its source identity changes to `d4f7c0d42d361e5aa696253e31a73ed269b1643d85e158b2efcc3dc287d5b85f` despite this being an analysis-only change. Keep the current VSC checkout for ongoing/resumed benchmarks and compatible control reuse; this local plotting refactor requires no new cluster run. Completed result files remain readable. No installation, training, cluster submission, commit or push.

- Validation: full suite **770 passed / 1 skipped in 562.61 s** (optional unbuilt data manifests); **74 focused plotting/migration checks passed** after the final legend refinement. All **14 notebooks** executed; the affected PD results and short-pilot notebooks were rerun after repairs. Confirmed **365 unique PDF files**, all inside the central figure tree; one caption index with every working PDF link; one 16.91-MB summary file containing all 14 final notebook outputs in filesystem order. Every PDF is 486 points wide; Poppler bounding-box checks found no text outside any saved page. Visually reviewed 27 representative updated figures, including the repaired pilot legend. Regression coverage includes migration/idempotence, empty ReadOnly Windows folders, untouched scientific evidence, notebook ownership, parallel-saver directory preservation, sorted summaries and partial calibration availability.

**Tried:** Regenerate every notebook against the current DATA ZIP and existing local project measurements.
**Result:** The PD results notebook failed in its calibration section after drawing the endpoint-monitor plots.
**Why:** No reference-model evaluation rows were yet available; pandas interpreted an empty non-boolean grouping result as a column selector and lost its merge keys.
**Instead:** Return an empty table with its schema intact before grouping and filter complete folds with an explicit boolean selection. Added an absent-reference/invalid-metric regression and reran the notebook; unavailable calibration evidence remains unavailable rather than becoming a fabricated plot.

## Follow-up benchmarks and paired reporting — 01-10-2026

- Final credit/non-credit evaluation already supports all 32 seed and 96 sampling checkpoints with the common five-fold/inner-validation protocol. Added cross-experiment control sharing in the GPU/CPU Slurm wrappers: published cache artifacts are hard-linked without overwriting donors; exact keys, folds and predictions remain validated by the existing reader. Sharing occurs at job start, so wait for main controls to be published to avoid concurrent duplicate computation. This does not replace checkpoint audits or completed-roster gates.
- Rebuilt all four experiment-2/3 notebooks: seed-43 versus the predefined seed-42 reference; three-protocol within-experiment contrasts; separate credit/non-credit monitoring and final benchmarks; secondary/calibration metrics, matched costs and exposure. Match dataset/partition before averaging, require all four OOD partitions and all three sampling arms. Missing evidence generates coverage tables, not placeholder quality plots. Plot logic stays under `src/visualize/`; PDFs remain flat at the requested ICML widths.
- All 14 notebooks executed successfully after repairing the three failures below; affected five notebooks were rerun. Confirmed every final summary in `All_Results.md`, 309 unique captioned PDFs in the flat figure folder, and no embedded error outputs. Visually checked the 14 currently available follow-up figures, 52 synthetic future-result figures across both tasks/four bases, and six varied-effect/cost stress figures. Synthetic review artifacts stayed in temporary storage. Downloads were read in place; follow-up project trajectories and final benchmarks are absent in this snapshot.
- Validation: full suite **763 passed / 1 skipped in 648.26 s** (optional unbuilt local data manifests). The final missing-time guard prevents NaN costs becoming zero-hour efficiency points; **13 focused comparison tests passed** after that guard, followed by the affected results-notebook rerun. Cache tests cover real phase-config key agreement, changed evaluation seeds, hard-link/atomic replacement behavior, incomplete folds and predictions. Training and evaluation engines were not changed.
- Current numerical identities are unchanged: training `86517b7df7d61736c7b026d986dad635868824ed8ef97ff76ff1f257ea248468`, evaluation `fba4a9c9d52512fc8ad1d54f95905a76f61b85b32463028bf96c4a670131a487`. Existing research plans, recovery files and compatible benchmark receipts remain valid. No research training, cluster submission, installation, commit or push. Library pin remains `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`.

**Tried:** Execute the complete notebook collection against the latest DATA ZIP and existing local project measurements.
**Result:** The auxiliary notebook failed selecting a ZIP workflow; both new sampling notebooks failed on the not-yet-completed trial roster.
**Why:** Archive metadata exposed nanosecond time but not `st_mtime`; an empty pandas grouping lost its expected key columns before cost matching.
**Instead:** Expose both timestamp forms, return unavailable costs before grouping an empty completed roster, and cover ZIP selection/incomplete campaigns in regression tests. All repaired notebooks now execute without inventing missing measurements.

## Seed completion, sampling segments and benchmark timeout — 01-10-2026

- Read the 09:18 Downloads ZIP in place: 1,821 files, 57,734,101 bytes, clean CRC and no duplicate members. Experiment 2 has **32/32 completed 10k-update trials** (16 PD + 16 LGD); seven interrupted PD records have completed successors. All identities, filenames and resolved scientific configurations match the prepared plans and current source. The 39,216 logged epoch rows have finite losses and zero recorded AMP/data skips; all 32 trials log the five post-baseline milestones. Recorded trial time totals **33.65 GPU-hours**, excluding allocation overhead. Six known upstream TabICL power-transform warnings occur in PD fold 2 without recorded loss/primary-score failure. Project weights and detailed trajectories are not included in this DATA-only ZIP; run the two existing CPU audits to verify them in place.
- Experiment 3's **24/24 first-fold trials saved their intended recovery segments**, ending with exit 75 and 123–5,145 successful updates; none is a completed research trial yet. All 7,371 logged epoch rows have finite losses and zero recorded AMP/data skips, and peak memory stays within the previously measured caps. Interrupted manifest rows use placeholder step totals; their saved-progress messages carry the actual counters. Resume these 24 and start the remaining 72 with the same prepared configs; do not restart experiment 0.
- Production accumulation is costly: first-fold PD estimates at 10k updates are about **25.3 h v2, 19.1 h v2.6, 9.8 h v3 and 11.3 h TabICL** per trial. Multiplying the complete first-fold training-time projection by four gives roughly **361 GPU-hours for the 96-trial sampling grid**, before fold variation, monitoring and restart overhead; this is not a measured completion time. Use 90-minute segments with auto-requeue and `CREDITPFN_MAX_REQUEUES=40`; the default 20 requeues may be too short for the slowest other-fold accumulation trials. The update budget and scientific schedule remain unchanged.
- User's scheduler check confirms PD benchmark **11643871_0 TIMEOUT**, elapsed 02:00:27 against a two-hour limit. Its log records **13 completed model/table comparisons (65 folds)**; one TabPFN-v3/Bondora comparison alone took 5,686.9 s. Reconstructed the exact 19-comparison task using the current packer and verified its first 14 entries against the log: six remain, comprising one Bondora comparison and five tables of at most 2,584 rows. Resubmit only task zero with `EVAL_TASKS=16` and a three-hour limit; matching complete comparison receipts/predictions are reused, while the interrupted comparison restarts. Full main benchmarking is still pending; the other three initial shards passed as recorded below.
- No runtime/configuration/data/Downloads changes, training, installation, submission, commit or push. The current source identity remains `86517b7df7d61736c7b026d986dad635868824ed8ef97ff76ff1f257ea248468`; the earlier 741 passed / 1 skipped suite applies to that unchanged implementation, and was not rerun for this output/documentation audit.

**Tried:** Run the first full-size packed PD benchmark task with the default two-hour allocation.
**Result:** TIMEOUT after 65 successful folds, during its second large-table comparison.
**Why:** Each five-fold TabPFN-v3/Bondora comparison costs about 95 minutes at the actual context/member settings; the older packing estimate understated this task's runtime.
**Instead:** Preserve completed results, resume the six remaining comparisons with three hours, and size subsequent benchmark allocations from production measurements. This does not invalidate trained weights or require another control run.

## Seed/sampling preparation accepted — 30-09-2026

- Read the 20:30 Downloads ZIP in place: 1,675 files, 53,761,244 bytes, clean CRC and no duplicate members. CPU preparations `62202369`–`62202372` all exit 0 and write the expected four plans: 16 PD + 16 LGD seed trials and 48 PD + 48 LGD sampling trials. All 128 identity checksums, trial-index mappings, folds, local phase configurations and current training source agree.
- Compared every new identity with its matching main reference: **all 25 credit-table hashes, eight research-retention hashes, eight original base-weight hashes and package versions match**. Intended differences are seed 43 for experiment 2, proportional PD sampling/pass mode for experiment 3, and the already validated entropy reporting source change. The earlier local/VSC CSV mismatch does not affect these VSC plans; preserve the cluster inputs. No further preparation or experiment-0 rerun is needed.
- Launch experiment 2 with the usual 90-minute resumable segments. For experiment 3, first submit partition zero on both tracks using `SPLITS=1 SEGMENT_MINUTES=20 CREDITPFN_AUTO_REQUEUE=0`: 24 actual research trials, each limited to a 30-minute allocation, at most 12 GPU-allocation hours in total. Keep the scientific 10k schedule; intended exit 75 means a recovery file was saved. Inspect throughput, memory and recovery publication before resuming those trials and releasing folds 1–3. These segments are part of experiment 3, not additional control trials.
- Initial benchmark probes: CPU PD `62202344` completes 5/5 folds in 228.4 pipeline seconds; CPU LGD `62202345` completes 5/5 in 10.6 seconds; GPU LGD `11643872` completes 125/125 in 637.2 seconds, all exit 0. GPU PD `11643871` has 60 successful folds, then starts the 730,176-row Bondora table at 19:58:36; the 20:30 archive contains no later outcome or recorded error. Check its live scheduler state; do not infer completion or failure from the quiet log. These are initial task shards, not the full main benchmark.
- No runtime/configuration/data/Downloads changes, submission, installation, commit or push. The full suite passed earlier this session on this unchanged implementation (741 passed / 1 skipped); this follow-up checks the new cluster plans and outputs.

## Auxiliary controls completed — 30-09-2026

- Read the updated Downloads ZIP in place (1,665 files, 53,704,707 bytes, clean CRC). Workflow `cc862607c5a24ca0940b4e1eb9a239a9` passed all eight GPU callbacks and final CPU audit `62202341`: **52/52 six-update arms**, 16 deliberate interruption/resume checks, eight seed comparisons and **40/40 smoke evaluation folds** (4,044 out-of-fold predictions). All completed arms have finite primary monitors, positive drift and zero recorded numerical skips. GPU log start/end intervals sum to **0.367 GPU-hours**, excluding unlogged allocation overhead; the workflow reached its final audit at 19:55:44.
- All 16 recovery pairs match predictive tensors, compared monitor scores at 0/3/6 and row exposure exactly. Ten match every saved tensor bitwise; six TabPFN LGD pairs differ only in `criterion.losses_per_bucket`, the upstream detached diagnostic accumulator excluded by the existing recovery comparison. This buffer does not enter the returned loss or predictive calculations. Do not call all saved checkpoint tensors bitwise identical.
- The receipt matches the current workflow fingerprint; all 13 auxiliary plans have valid checksums and current training source `86517b7df7d61736c7b026d986dad635868824ed8ef97ff76ff1f257ea248468`. Their eight base-weight hashes, package versions and six selected credit-table hashes match the original main plans. Deployed commit is `ac248ae`; library pin remains `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`. DATA contains the VSC audit reports; the actual auxiliary trajectories/weights remain on project storage.
- Rechecked the complete research grids and dataset roles: **32 seed-43 trials** and **96 sampling trials**, 10k updates, four bases, four folds and both tasks. All three experiments have identical table partitions: PD held-out counts 4/4/5/4, LGD 2/2/2/2, each table held out once. Experiment 2 retains the main reference recipe; experiment 3 intentionally uses proportional PD sampling in all three arms, including its separate one-sample control. Local tests cover row coverage, fresh epoch partitions, worker consistency, per-table gradient averaging/L2-SP, exact budgets/recovery and outer/inner evaluation separation.
- The main seed-42 reference's 32 completed trials recorded **33.64 GPU-hours** (22.92 PD, 10.72 LGD), a seed-study estimate rather than billed time. Production PD accumulation averages **5.31–16.54 chunks/update** across bases/folds; LGD averages 1–2. These follow the cluster's recorded table sizes and measured caps, not timing extrapolation. The tiny auxiliary batches do not establish full-size runtime/memory. Prepare fresh VSC plans and compare their data/base/environment identities against experiment 1; then run the seed study and bound the first real sampling segments before releasing all 96 trials. Preserve the 10k schedule and resume those same trials. No repeat of experiment 0 is indicated.
- The 19:57 snapshot shows the four initial main benchmark tasks starting: GPU PD/LGD `11643871`/`11643872`, CPU PD/LGD `62202344`/`62202345`. They are not completed in this snapshot; do not resubmit or change runtime code while they run. No source/config/data/Downloads changes, training, installation, cluster submission, commit or push. Full local suite: **741 passed / 1 skipped in 578.88 s**, the skip is optional unbuilt local data manifests.

**Tried:** Use local processed CSV byte hashes to corroborate the cluster's saved main inputs.
**Result:** All raw hashes differed; normalizing only CRLF to LF matched 23/25 tables, with PropPD2 and PropLGD2 still different.
**Why:** The local processed tree is not a byte-identical copy of VSC inputs; the remaining content differences cannot be established from output-only downloads.
**Instead:** Keep the VSC inputs untouched, use cluster-recorded sizes for cost calculations, and compare new VSC plans with the original main plans before research submission. Do not upload or regenerate local processed data for this campaign.

## Project-checkpoint confirmation — 30-09-2026

- Read the new Downloads ZIP in place (1,569 files; CRC check passed). PD audit `62202338` and LGD audit `62202339` both ended with exit 0: each reports 256 completed checkpoints, zero pending/diverged/problems and `passed: true`. All 512 unique checkpoint paths resolve under project `checkpoints/trained/experiment1/{pd,lgd}/`, with matching prepared provenance and 10k-update budgets. This verifies file availability and metadata on VSC; final benchmarking still supplies the actual inference/load check. The four initial benchmark tasks can proceed; none has output in this snapshot.
- Auxiliary workflow `cc862607c5a24ca0940b4e1eb9a239a9` passed CPU preparation (`62202340`) and submitted Mindwell array `11643860`. The 19:49 snapshot shows four PD tasks executing, seed arms completing normally, no recorded errors/callback failures, and no final auxiliary receipt yet. Do not resubmit it or treat this partial snapshot as approval for experiments 2/3. Its logs show deployed commit `ac248ae`, including the entropy repair. No source, configuration, downloaded file or scientific result was changed during this inspection.

## Completed main-training audit — 30-09-2026

- The user's merged local DATA/project download contains **512/512 completed main trials: 256 PD + 256 LGD**, 64 per dataset fold in each track. Every trial matches its original prepared identity, has exactly **10,000 successful updates**, and records all six milestones (0/250/1k/2.5k/5k/10k) with finite primary credit/non-credit scores. All 68 PD interruption records have completed successors; LGD has none. No logged traceback/OOM/time-limit kill or recorded AMP/data skip was found. Actual checkpoint files remain on VSC: run the existing CPU audits there before certifying checkpoint availability/provenance.
- Detailed coverage: 512 epoch histories, 512 trajectories, 512 compressed parameter histories and 512 resource histories; 628,480 epoch rows, 3,072 trajectory rows, 1,018,752 parameter rows and 91,739 successful GPU resource samples. Recorded trial elapsed time sums to **509.06 GPU-hours** (349.83 PD, 159.23 LGD), excluding queue/allocation overhead. The experiment1 tree is about 1.17 GB; no final benchmark results exist yet. Keep the training evidence and original plans.
- Reproduced and repaired the deferred entropy defect: float32 clipping rounds the upper probability bound back to one, giving `0 * log(0)`. Promoting the diagnostic copy to float64 before clipping prevents this without changing the probabilities used for primary scoring, the objective or trained weights. The saved download has **seven missing training-table entropy measurements in six PD trials**, no missing held-out/OOD entropy, and finite other supported metrics. Historical CSVs remain unchanged; their missing entropy cannot be reconstructed from aggregate scores. Separate upstream Yeo-Johnson overflow warnings remain (96, TabICL PD fold 2, including update zero); finite monitor results do not by themselves establish the internal cause.
- Original completed training identity remains `6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28`. The shared-metric repair changes the current training identity to `86517b7df7d61736c7b026d986dad635868824ed8ef97ff76ff1f257ea248468`; do not rewrite old plans or retrain experiment1. Both original plans pass `check_prepared(stage="eval")` under the corrected code, which gives the benchmark its own identity. Prepare fresh auxiliary/experiment2/experiment3 work under the corrected source.
- Training is complete, not final scoring. Use separate GPU foundation and CPU classical benchmark jobs, first measuring actual production evaluation cost. The requested experiment-2/3 gate is `run_experiment0.sh auxiliary`; the historical `part2` budget pilots already passed. No auxiliary receipt is present in this download. The user's notebook rerun finished during review: all 14 saved notebooks have final summary outputs and no error outputs; no notebook code or generated output was changed by this audit.
- Validation after the metric repair: **741 passed / 1 skipped** (optional local data manifests absent), both original evaluation-plan gates passed, and `git diff --check` was clean. Exact zero/one float32 and float64 probabilities are covered by regression tests. No training, cluster submission, installation, commit or push was performed.

**Tried:** Treat every parameter-history relative-change NaN as a missing measurement.
**Result:** The initial ad hoc scan falsely flagged 384 otherwise complete trials.
**Why:** Frozen/unanchored tensors intentionally have no reference drift; only anchored trainable tensors define those diagnostics.
**Instead:** Check availability against anchoring: all anchored absolute/relative changes and all tensor norm/mean/std/absolute-max measurements are finite, and unanchored tensor norms remain unchanged from update zero. Preserve intentional missing values.

## Auxiliary validation and notebook refinements — 30-09-2026

- User requires a fast extra experiment-0 gate before experiments 2/3, rather than using a first research partition as the functional test. `run_experiment0.sh auxiliary` prepares 13 case configs from the research reference recipes, then runs eight GPU tasks with 15-minute limits, at most four concurrent in the shared pool. Across both tracks: 36 continuous six-update arms + 16 interrupted/resumed arms = 52. Interrupt at 3, continue to 6. Tiny 256-row contexts, two small training tables and one original held-out table; no scientific grid changes or cluster submission in this session.
- The final auxiliary audit requires every base/task, zero numerical skips, exact budgets/identities, finite monitors and positive drift, seed-dependent trained weights with fixed initial monitors, matching full-pass/accumulation recovery, expected exposure ordering and five-fold package-data evaluation. Small shapes do not certify production memory/time. New generated identities stay under experiment0; prior accepted controls are retained. The auxiliary notebook shows pending evidence honestly until those runs are downloaded.
- Experiments 2/3 remain 32 seed-43 and 96 sampling trials at 10k updates; all four bases and both tracks. Reviewed fixed partition/monitor/evaluation seeds, context/query separation, row partitions and per-table gradient averaging against local primary sources at library pin `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`. No training-path changes were necessary. Table identities alone do not prove population independence; equal steps do not match exposure/compute, and two seeds only test sensitivity at one recipe.
- Corpus profiles now add predictor counts beside logarithmic row bars; LGD histograms mark means/medians. Exposure labels give row counts, cap and signed color-coded multipliers without connectors. Pilot curves omit partial-pass crosses and retain their values in tables. Budget panels include training-credit minus held-out-credit improvement, explicitly a between-table transfer diagnostic. Paired L2-SP/LR summaries require complete matching partitions/recipes. Zero-penalty panels and all-zero AMP plots are omitted while numerical tables retain them. Clipping shares a task-wide percentage scale, linear to 0.1% then logarithmic. These supersede the earlier presentation notes below.
- The existing main-training source identity remains `6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28`. Runtime training code, research configs, original data/weights and Downloads are unchanged. Evaluation/workflow code identities change with the new utilities; never relabel old receipts. New auxiliary validation is independent of the historical part-1 receipt.
- Validation: full suite **738 passed / 1 skipped** (optional local data manifests absent); **50 focused checks passed** after the final plotting refinements and additional workflow test, followed by **4 privacy checks passed** on regenerated notebooks. All **14 notebooks** executed successfully; affected notebooks were rerun after final edits. All **298 PDFs** were visually reviewed. Automated checks found matching captions and all 14 complete printed summaries, correct 486-point widths, no off-page text or proprietary identifiers, and no PDFs outside the flat figure folder. No research training, installation, cluster submission, commit or push.

**Tried:** Read parameter drift from epoch rows while assembling training summaries and paired factor plots.
**Result:** Empty parameter panels and falsely inflated nonfinite-health counts despite valid saved drift measurements.
**Why:** Drift is measured at fixed trajectory milestones; epoch records intentionally contain NaN placeholders.
**Instead:** Read drift from trajectory records at its exact cadence; use only always-recorded optimizer fields for the epoch health count, and omit unsupported panels. Preserve complete support/partial-loss tables and add a regression check.

## Notebook layout and training-process follow-up — 30-09-2026

- User now requests all publication PDFs in one flat `output CreditPFN/figures/` folder. Experiment/notebook prefixes retain ownership; reruns clear generated PDFs/metadata, and full reruns also retire renamed notebooks' figures. Training records/results are preserved. Caption metadata remains experiment-scoped, with the shared index beside the PDFs. This supersedes the older per-notebook folder descriptions below.
- Publication path helpers are isolated under `src/visualize/paths.py`, using the shared output root. The legacy helpers in runtime-fingerprinted `src/utils/paths.py` are retained for migration so active training plans are not invalidated. Training source identity stays `6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28`.
- Corpus counts/missingness use zero-based bars; geometry combines both tasks. The sampling illustration combines one-sample/accumulate markers and labels the full-pass/equal-table *step-share* ratio, explicitly distinct from absolute updates and row exposure.
- Short PD pilot loss drops at update 250 reflect an incomplete final traversal: three tables versus thirteen at update 247. Plots retain incomplete-coverage losses as separate crosses/tables rather than joining them into a full-corpus learning curve. Configured schedules use the actual training scheduler, share a logarithmic plot across identical tasks, and retain observed epoch rates separately.
- Experiments 1–3 now separate training dynamics and endpoint results; monitor endpoints remain distinct from five-fold benchmarks. Experiments 2/3 have `01_training` and `02_results` notebooks. Resource diagnostics include update-window trajectories. No training implementation, scientific configuration, dataset, checkpoint, cluster job or Downloads file was changed.
- Loss-coverage summaries retain per-trial counts and partial-pass exceptions, with every plotted window still tabulated; they do not duplicate the full epoch archive in notebook stdout. This reduced the executed LGD training notebook from 115 MB to 12 MB on this snapshot.
- Rechecked experiment 2: 32 seed-43 trials; experiment 3: 96 protocol trials, both at 10k updates. Neither needs experiment 1 completion to train. Seed comparisons need matching main reference cells. Experiment 3 still needs production-cap full-pass/accumulation timing and resume evidence; the bounded first-partition launch described in VSC.md preserves its final scientific schedule.
- Validation: **719 passed / 1 skipped** (unbuilt local data manifests), then **67 focused checks passed** after compacting coverage tables. All **13 notebooks** executed successfully, with affected notebooks rerun after final edits. All **303 PDFs** were visually reviewed; final checks found matching captions and verbatim summaries, no off-page text or proprietary names, and no PDFs outside the flat folder. `All_Results.md` is 14.15 MB. Automatic approval review blocked removing empty legacy figure directories and the temporary review images in `general/manifests/figure_review/`; manual cleanup remains. No install, training, submission, commit or push.

**Tried:** Run the full suite after splitting the seed and sampling notebooks into training/results pairs.
**Result:** The notebook-layout guard failed on its historical count of eleven notebooks.
**Why:** The requested separation intentionally expands the collection to thirteen; the guard hard-coded the former count.
**Instead:** Update the expected count while retaining AST checks for thin notebooks, matching saver ownership and final summaries; rerun the affected tests and full suite.

**Tried:** Print the full loss-coverage archive in the training summary while adding partial-traversal diagnostics.
**Result:** Hundreds of thousands of raw epoch rows bloated the notebook and shared summary.
**Why:** Those rows duplicated stored training evidence beyond the window statistics and exceptions shown in the figures.
**Instead:** Print complete plotted statistics, per-trial coverage counts and partial-coverage epoch values; keep full raw histories in their existing training files.

## Notebook presentation follow-up — 30-09-2026

- The user downloaded both output tiers into the local repository. Analysis now reads that merged tree; no Downloads files were moved by the agent. Rechecked 16/16 null audits, the exact corrected 12 PD + 8 LGD short-pilot cohort, and 8/8 historical budget pilots. All selected pilots have finite recorded optimizer diagnostics and zero AMP/data skips; the historical budget-objective caveat remains necessary.
- Added tracked public aliases for Credit Risk, Bondora and the PD/LGD SBA copies, and expanded model labels at display boundaries while preserving join keys/checkpoint names. Balanced dense pages; the 17-table PD holdout and target bars fit one figure each. Removed notebook production notes and explained final-setting averages versus variation across datasets in plain language.
- Raw `target_in_raw=False` is expected for German Credit (last-column rename and 1/2-to-0/1 recoding) and SBA LGD (charge-off/disbursement for defaults). The raw report now verifies target-column presence in processed files and explains these transformations. German's recognized raw label is excluded from its predictor count; no data or preprocessing code changed.
- Compact pilot reports show paired endpoint matrices, model panels for recorded losses, shared measured schedules, and numerical optimizer/resource summaries: **107 to 5** short-pilot figures and **74 to 7** budget figures on the merged local snapshot. Budget reports combine credit and non-credit curves. PD/LGD tags are explicit in exports and captions. Numbered FigureSaver prefixes already prevented physical file collisions; the earlier issue was ambiguous unqualified names, not overwritten exports.
- Training identity remains `6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28`; no training, cluster submission, install or push. The deferred entropy diagnostic repair is outside this presentation change.

- Validation: final full suite **711 passed / 1 skipped** (unbuilt data manifests); **11/11 notebooks executed successfully**, including reruns after the final label and prose fixes. All **296 PDFs** were visually reviewed; automated checks found no off-page text or proprietary names, and every caption/figure summary is present in All_Results.md. Temporary review images were removed. Raw and processed inventories both cover 25/25 datasets, and both expected target transformations have their processed target columns.

**Tried:** Apply the main-grid 50-window curve layout to 250-update pilots and reuse per-recipe plots throughout experiment 0.
**Result:** Short histories had isolated unmarked points and fragmented lines, while repeated one-recipe budget figures obscured valid available measurements.
**Why:** Plot windows were narrower than the epoch recording cadence; splitting every base/adaptation/diagnostic into a separate figure produced excessive repetition.
**Instead:** Bound window width by the recorded epoch interval, show observed-point markers, retain genuine missing-data gaps, and use compact pilot-specific panels/tables. Empty cohort curves are omitted instead of exported as empty axes.

**Tried:** Run the full suite after adding explicit task labels to notebook figure calls.
**Result:** 707 passed, one skipped, and one exact-source-string assertion failed despite the same figures still being saved and displayed.
**Why:** The assertion included the call's closing parenthesis and rejected the new optional `track` keyword.
**Instead:** Inspect the notebook call AST to verify that both expected figure functions are passed to `cp.show`, independently of presentation keywords; retain the actual Brier-field check.

**Tried:** Scan the regenerated notebook JSON verbatim for proprietary names during the full test suite.
**Result:** 708 passed and one skipped; the privacy guard flagged one short identifier inside an embedded PNG's base64 bytes.
**Why:** Random encoded image bytes can match a short text token; the sole match was in image/png, not in source, displayed text or a label.
**Instead:** Parse notebook MIME bundles, omit valid binary encodings from the text scan, retain source/text/SVG/metadata and malformed payloads, and inspect rendered figures plus exported PDF text separately. Regression tests cover both exclusions and retained readable content.

## Notebook/publication review — 30-09-2026

- Rebuilt all 11 notebooks around coverage, paired credit effects, learning trajectories, retention, parameter movement, cost and final five-fold evaluation. Analysis and reporting logic remains in `src/visualize/`; each final printed summary includes all plotted values and displayed tables in section order. `All_Results.md` preserves that text with trailing line whitespace removed. Captions describe the measurement, pairing and aggregation, without claiming results from missing files.
- User explicitly selected the adjacent ICML paper's **6.75-inch full / 3.25-inch column widths** over the generic A4 template. PDF export keeps those widths and embedded TrueType text. Four-LR panels, paginated dataset matrices, shared comparison scales and external legends replace crowded overlays. Figures, caption metadata and summaries remain under `output CreditPFN/<experiment>/`.
- Read the current DATA ZIP directly in Downloads using `CREDITPFN_ANALYSIS_ROOT`; project diagnostics/results can be supplied separately with `CREDITPFN_ANALYSIS_PROJECT_ROOT`. Both accept a folder or ZIP and never extract/copy inputs. Missing project files remain unavailable instead of falling back to unrelated local runs. The snapshot still supports PD **256/256** and LGD **152/256** OK outcomes; it cannot supply actual per-table trajectories or final benchmarks.
- Corrected-pilot analysis follows the latest passed receipt and its saved configs (**12 PD / 8 LGD**), rather than the static historical pilot grid. Aggregate curves require all planned partition trials at each milestone, identical baseline/current dataset support and visible gaps. Final benchmark pairing requires all five exact outer folds and rejects duplicate rows; calibration additionally requires usable bins from both compared models.
- Training/config/Slurm code was not changed. Training source identity stays `6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28`; library pin remains `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`. This notebook work requires no VSC retraining. The separately documented entropy diagnostic repair remains deferred.
- Final verification: **697 passed / 1 skipped in 500.70 s** (optional local data manifests absent), **11/11 notebooks executed**, **88 actual PDFs** and **100 synthetic-schema PDFs** visually reviewed. PDF geometry, embedded fonts, text boundaries, caption coverage and all 11 final summaries were checked. Temporary synthetic PDFs and raster previews were removed; four empty review directories remain because automatic approval review blocked their deletion. Downloads hashes and the training identity are unchanged; no training, installation, submission, commit or push.

**Tried:** Validate only currently available DATA-side notebook figures.
**Result:** Project-side curves, calibration, seed/sampling and resource views had no real data to exercise; the first visual pass also exposed crowded ticks and coincident points.
**Why:** A partial archive cannot validate future populated views, and count-oriented tick formatting is inappropriate for fractions of datasets or updates.
**Instead:** Exercise the native future schemas with explicitly synthetic test fixtures, check complete coverage/missing milestones/duplicate folds, render and inspect every figure, use fractional ticks and visible point offsets, then remove temporary QA exports.

**Tried:** Run the full suite after moving metric selection from notebooks into the source module.
**Result:** 694 tests passed, one skipped and two failed: an obsolete notebook-source assertion and the previously recorded Windows temporary `state.json` replacement error. The latter passed unchanged on a targeted rerun.
**Why:** The source assertion encoded the former code location; the independent temporary-file denial remains intermittent and its cause is not established.
**Instead:** Test the actual selected Brier-score field in the source function while checking notebook delegation. Preserve the active atomic writer and record both full-run and targeted results rather than hiding the Windows failure.

## Current handover — 30-09-2026 main grid 79.7% complete; entropy diagnostic needs repair after training

- Read `Downloads/output CreditPFN.zip` in place: **1,336 files / 40,836,614 bytes**, SHA256 `145889c4f1ddc048a19be9aadc67577a31fc40cc0db42e2b63cb2510c589ff01`; CRC clean, no duplicate entries, **671 JSON / 24 CSV** files parse with consistent CSV row widths. This remains a DATA-side snapshot: project training/trajectory/tensor/resource files, predictions and weights are absent. No extraction into the repository.
- At **09:56 CEST**, **408/512 trials complete (79.7%)**: PD **256/256**, LGD **152/256**. LGD fold counts **64/52/32/4**; another **16 LGD logs are advancing**, four each for v2 fold 2, v2.6 fold 3, v3 fold 1 and TabICLv2 fold 2, with **88 trials unstarted**. All PD finished by **06:14**. These are archive observations, not live scheduler states; all 512 submissions remain accounted for and the shared pool has no uncertain-submission marker.
- **492 attempts / 424 distinct trials**, **476 zero-exit closed logs / 16 open logs**. All **68 saved PD interruptions resumed and completed**, with matching saved/recovered update counters, continuous epoch numbering and no repeated trajectory milestone. Every completed trial has exactly **10,000 successful updates**, one initial baseline, five later credit milestones, a matching final-save message and one OK manifest row. All **471,918 logged epoch summaries** have finite loss/gradient/drift/memory fields and zero AMP/data skips; no traceback, OOM, DIVERGED, disk/publication error or nonfinite logged primary score was found.
- Both plan checksums, **512 trial identities**, **492 resolved configurations** and all **476 manifest events** agree with the current recipes/source hash `6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28`. Runtime: commit `2d36943`, library pin `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`, CreditPFN, BF16, B200, 24 CPUs and 120 GiB host RAM. TabICL baseline differences are tiny (maximum AUC spread **1.10e-6**, RMSE **6.65e-9**) under the documented fast nondeterministic execution. Full-update v3/TabICL models retain eight fixed positional-frequency parameters; total minus trainable equal to eight is expected, not a failed adaptation arm. See `RotaryEmbedding` in the pinned `TabPFN .txt` / `TabICL.txt` code dumps.
- **Confirmed diagnostic bug:** seven entropy warning events across six PD trials originate in `src/eval/metrics.py::_classification_metrics`. Clipping float32 probabilities at `1 - 1e-15` leaves an exact one; `0 * log(0)` then makes `prediction_entropy` NaN. A synthetic CPU reproduction gives NaN entropy while AUC, Brier, log loss and ECE remain finite; float64 yields the expected **0.25020120867**. This calculation is a reporting path, not an optimization objective. Preserve the active source fingerprint; repair after this training campaign and before final evaluation, then recompute affected entropy measurements where required. No retraining is justified by this diagnostic alone. Exact affected per-table cells require the project trajectory files.
- Separate upstream Yeo-Johnson **power-overflow warnings occur 96 times**, once at each of six monitor points for all 16 TabICL PD fold-2 trials (including their unmodified baselines). Logged primary scores remain finite and training has no skipped updates. This regular preprocessing warning is not evidence of optimizer divergence, but the absent per-table diagnostics/transformed inputs prevent independently certifying its numerical effect. Keep it visible for the project-storage audit.
- Preliminary paired **held-out monitor** directions: PD **104/256 improved AUC**, LGD **18/152 reduced RMSE**; these are configuration/fold outcomes, not independent datasets or final benchmark results. Conservative PD full updates show small gains for v2/v2.6/TabICL; larger learning rates often degrade, with v3 particularly sensitive. Do not average raw LGD RMSE across table scales or select a winner from these monitors. Retention/forgetting remains unassessed without project-side measurements.
- Completed trial work: **349.83 PD + 89.27 LGD = 439.10 GPU-hours**, excluding setup, queueing and unfinished work; PD **59–104 min/trial**, LGD **14–60 min/trial**. Remaining work is roughly **56 GPU-hours** by matched base/adaptation timings. The four-task array dependencies matter: remaining v3 work alone is about **5.6 hours at four workers**, so roughly **6–8 further wall-clock hours from this snapshot** is more plausible than dividing all remaining work by 16; unmeasured later-fold runtimes, setup and queues can change this substantially.
- Continue the existing submissions; no cancellation, cleanup or resubmission indicated. Only handover/changelog records changed; no cluster command, training, installation, commit or push. Local full suite: **680 passed / 1 skipped / 1 failed** in 514 seconds; the one Windows temporary-file replacement failure **passed unchanged on its targeted rerun** in 5.82 seconds. Do not describe the initial full run as entirely passing.

**Tried:** Evaluate prediction entropy with float32 probabilities clipped at `1e-15` and `1 - 1e-15`.
**Result:** Exact-one probabilities still produce NaN entropy; reproduced the same warning pair observed at seven production monitor events.
**Why:** The upper bound rounds to one in float32 and the formula evaluates zero multiplied by negative infinity.
**Instead:** After the active campaign, use boundary-safe entropy arithmetic (or suitable precision), cover exact-zero/one probabilities in a focused regression test, and recompute affected diagnostics; preserve training identities and completed weights.

**Tried:** Require exact repeated baselines and equality of total/trainable parameter counts in every full-update arm during the ad hoc archive audit.
**Result:** Flagged tiny TabICL baseline differences and the same eight-parameter difference in every v3/TabICL full arm; investigation found no corresponding training failure.
**Why:** Fast GPU kernels do not promise bitwise inference equality, and upstream stores fixed rotary frequencies as nontrainable parameters.
**Instead:** Quantify baseline spread and compare model-aware parameter counts against the pinned implementation; keep these audit assumptions distinct from confirmed output defects.

**Tried:** Run the full local suite after the archive audit.
**Result:** `test_pilot_only_preparation_and_callbacks_use_selected_counts` hit `PermissionError [WinError 5]` replacing its temporary `state.json`; 680 other tests passed and one skipped. The failed test passed unchanged on immediate rerun.
**Why:** Windows denied `os.replace` in the test's temporary directory; the source of the denial was not identified. No matching production failure appears in the downloaded logs.
**Instead:** Record both results, repeat only the affected test, and avoid altering the active training/serialization code on this unreproduced local evidence.

## Previous handover — 29-09-2026 main training healthy in the 08:58 snapshot

- Read `Downloads/output CreditPFN (1).zip` in place: **684 files / 11,754,907 bytes**, SHA256 `7e47e077245d250f1b1fd4c4e546ff3993da6839bdfb8cc5216a40029255e3ce`; CRC clean, **350 JSON / 19 CSV** files parse. No extraction or copying into the repository. The DATA-side snapshot has no project training CSVs or checkpoint files, so it cannot independently certify their contents or live scheduler state.
- The detached submitter finished: final LGD fold arrays **11624651 / 11626375 / 11626507 / 11626875** were accepted, followed by `Submission complete`. **All 512 trials / 32 arrays are submitted.** The shared scheduler pool has 16 slots, those four final array IDs in its lanes, and no uncertain-submission marker. Do not restart either submitter.
- **116/256 PD trials completed**, all at **10,000 successful updates**: fold 0 **64/64**, fold 1 **41/64**, fold 2 **11/64**, fold 3 **0/64**. Another **16 PD logs were advancing at 08:58**, **15 trials had saved an interruption without a later attempt in the snapshot**, and **109 PD trials had not started**. All **256 LGD trials remain unstarted in this snapshot**; their shared-pool dependencies follow the earlier PD submissions. Whole-grid completion: **116/512 (22.7%)**. These are snapshot classifications, not a live `squeue` result.
- **171 GPU attempt logs** represent **147 distinct scientific trials**. There are **39 normal segment-save events**; **24 have resumed and completed**, with continuous epoch progress and no repeated earlier milestone. The remaining 15 saved trials need only a live queue check for normal pending/requeue status. All 155 closed attempt logs end at zero; the other 16 are actively updating. No traceback, OOM, FAIL/DIVERGED outcome, non-finite logged training loss/primary monitor, or nonzero AMP/data-skip counter was found. Every completed trial has its initial baseline and all five later credit-score milestones, finite endpoint metrics/drift/memory, and no duplicate OK row. Repeated baseline AUCs agree exactly within base/fold.
- Both plans and all **512 identity checksums** match current training source **6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28** and current configs; all **171 resolved configurations** and **155 manifest rows** match their planned recipes. Completed rows have correct partition counts, both adaptation modes and all LR/L2-SP arms represented. All attempts use CreditPFN, BF16, 24 cores and 120 GiB host RAM; recorded commit **2d36943**, library pin **81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba**. Completed trial work totals **158.39 GPU-hours**, **60–104 minutes/trial**, excluding setup/queue and unfinished work; peak allocated GPU memory reaches **181.08 GB**. Rounded console timings imply about **2.9%** residual input/loop overhead, not GPU utilization. Detailed per-table OOD/parameter/resource CSVs remain unverified on project storage.
- At **09:14**, the user's queue snapshot confirms all **15 saved trials are PENDING (JobArrayTaskLimit)**: array **11624545** has 12 waiting and four running; **11624556** has three waiting and four running. These are the configured four-task array limits, not scheduler holds or failed recoveries. The saved tasks are already requeued and can resume as their array slots become available; no release or resubmission is needed. This targeted query covers these two arrays, not the whole campaign.
- Recommendation: let the existing campaign continue; no runtime/config change, cleanup or resubmission indicated. Only handover/changelog notes changed; no training, remote command, installation, commit, push or new local experiment output. The previous full suite remains **681 passed / 1 skipped** on this unchanged runtime; this turn checked output integrity and plan/recipe/recovery consistency, not another test-suite run.

**Tried:** Associate resumed logs using a defaulted `SLURM_RESTART_COUNT` from resolved metadata, and group interrupted manifest events by final-checkpoint path.
**Result:** The ad hoc audit falsely mapped resumed metadata back to restart-0 logs and collapsed distinct interruptions; corrected grouping found 24 clean recoveries and 15 saved trials still awaiting a later attempt.
**Why:** Resolved metadata does not contain that restart field, while INTERRUPTED rows intentionally have an empty final path and a placeholder zero step count. They are attempt events, not completion records.
**Instead:** Pair each job's time-ordered resolved records with its numbered `_rN` logs, group scientifically by run name and trial index/recipe, and read saved progress from the interruption message or recovery state. Do not treat partial-row defaults or absent per-epoch monitors as numerical failures.

## Previous handover — 28-09-2026 main submission underway; LGD waiting for queue room

- User submitted the prepared `cpt_main_v5` grid with 90-minute segments and automatic requeue. **PD: all 256 trials / 16 arrays accepted** (`11624537`, `11624542`–`11624545`, `11624554`–`11624564`); its launcher reported completion. **LGD: 192 trials / 12 arrays accepted** (`11624565`–`11624576`, folds 0–2); the remaining **64 trials / four arrays**, fold 3, are not yet submitted in the supplied transcript.
- The launcher reports **448 submitted tasks** and waits before adding another 16-task array to its configured **450-task headroom**. This is the expected application-level wait, not an `sbatch` rejection; the next array fits at **434 or fewer** active/submitted tasks. Polling is once per minute. Keep this submitter running; do not launch another copy. Concurrent research jobs remain bounded by the shared 16-slot pool.
- At **17:50**, the user's next transcript shows Ctrl+C during that unchanged 448-task wait, with no additional arrays accepted, then a `nohup` background launch restricted to LGD **`SPLIT_START=3`**. Shell PID **3692749** on **tier2-p-login-4**; stdout/stderr go to `output CreditPFN/experiment1/logs/submit_lgd_fold3.log`. This PID confirms a background process was started, not that preparation passed or the final four arrays were accepted. Inspect that log next; do not duplicate the background submitter. Already accepted Slurm jobs are independent of this terminal.
- No main training outcome or actual running count has been observed yet. Check the queue and first PD arrays from a second terminal; do not equate accepted submissions with successful training. Only run records changed; no runtime edit, cluster action, install, commit or push by the agent.

## Previous handover — 28-09-2026 experiment-1 plans and submission previews passed

- Latest Downloads ZIP read in place: **335 files / 2,534,874 bytes**, SHA256 `04f84202d2f558fb564eccad73d0300e5deb7a9166b93f164a61fc8aa6bdb688`, CRC clean; **179 JSON / 16 CSV** files parse. The archive starts directly at the experiment directories. Experiment 1 contains three maintenance logs and two prepared plans, with no training results yet.
- CPU preflight **62175400** passed with **0 failures / 0 warnings**; PD preparation **62175489** and LGD preparation **62175508** both passed. The user's scheduler transcript confirms all three **COMPLETED / 0:0**. Logs use the CreditPFN interpreter and the required experiment-specific output paths. Missing result/training directories are expected before the first research trial; the preflight's generic GPU-control reminder does not require repeating accepted controls.
- Verified both plan checksums and all **512** unique trial identities: exact current configs/source, matching accepted-control package versions and base/credit/retention hashes, disjoint train/test tables and complete four-fold coverage. The transcript's previews both pass the prepared-plan gate, each covering **16 arrays / 256 unique cells**, four workers, shared cap 16, 90-minute segments, **1:40:00** allocations and the correct recovery signal/requeue options. Numerical source remains **6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28** at runtime commit **2d36943**.
- Next: user submits the two experiment-1 training launchers without `DRY=1`, preserving the accepted experiment-0 evidence and fixed code/configs. Use a persistent terminal because the 512-task submission can wait at the 450-task headroom. This snapshot ends at the previews and does not show research submission or live scheduler state. Only these operational notes changed; no runtime edit, extra GPU test, install, commit or push.

## Previous handover — 28-09-2026 corrected production pilots passed; prepare experiment 1

- Latest Downloads ZIP read in place: **330 files / 2,422,405 bytes**, SHA256 `84c4d9b122cfba41659465f1e5e52698445505c3d40c763b34d8c6ddbe12115e`, CRC clean. Pilot workflow **c2656ffd6e57451aa3522468854ce8f7** and its identical receipt pass **12/12 PD + 8/8 LGD**, all at **250 updates**, no pending/diverged trials or audit problems. Trial manifests have finite endpoint losses/scores/memory and code **2d36943**; plans/checksums match current training/workflow identities. The cluster library checkout now matches **81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba**.
- CPU preparation **62174499** and audit **62174759** both end at exit 0; total logged workflow span **16:40:30–16:47:19 CEST**. Recorded trial work totals **0.680 GPU-hours**, **66–180 s/trial**, excluding allocation setup/queue overhead. These are targeted repeats of existing experiment-0 pilots after the objective repairs, not new research experiments. Together with the accepted corrected recovery checks, they clear the remaining training-validation gate; prepare fresh experiment-1 plans and inspect preflight/submission previews next. No complete experiment-0 or long-pilot repeat is indicated by this snapshot.
- Launch/performance review: current training identity remains **6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28**; read-only main previews resolve **256 PD + 256 LGD**, with held-out table counts **4/4/5/4** and **2/2/2/2**. Across the corrected pilots' logged non-initial epochs, `io/(io+compute)` is **4.6–7.1%** by base/task. This uses rounded console timings and measures residual input/loop overhead, not GPU utilization; detailed resource CSVs remain on project storage. Keep the tested four workers, BF16 and 16-slot shared pool; no unmeasured performance rewrite before launch. Use 90-minute resumable work segments with 100-minute allocations. The 512-task launch can wait at the 450-task queue headroom, so keep the submitter in a persistent terminal. CPU main-plan preparation/preflight are still required on VSC; no research jobs have been submitted by the agent.
- This inspection changed only the handover/changelog; runtime code retains the preceding **681 passed / 1 skipped** verification. Six focused design/plan/submission checks pass again (**5.09 s**), as do the proposed launch sequence's Bash syntax and diff checks. No new training, submission, install, commit or push; the workflow explicitly stopped after its pilot audit. The ZIP does not establish unrelated live scheduler activity.

## Previous handover — 28-09-2026 corrected recovery passed; selected production pilots next

- Read Downloads ZIP in place: **275 files**, **2,220,075 bytes**, SHA256 `6ae16b6ff7b3187d640af60e95dbfb9ce0daa55d643e919d858023da4c754ca3`; CRC and all **151 JSON / 14 CSV** records parse. Workflow **f15175a25a7b472f89d70b0e5ba5d8cf** and its identical `recovery_passed.json` report **8/8 pairs passed**, all **16 arms at 12 updates**, nonzero drift, finite endpoint losses/scores. All 16 prepared trial identities/checksums match training source **6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28** and commit **3822d8b**.
- Saved inference/model tensors and monitored metrics at **0/5/12** match exactly in every pair. Only the three TabPFN LGD diagnostic `criterion.losses_per_bucket` buffers differ; this is explicitly reported and does not enter inference or loss computation. All **40** packaged-table benchmark folds and complete prediction checks pass. Eight GPU logs plus CPU preparation/audit end at exit 0, all use the CreditPFN conda interpreter; the inherited-venv warnings show successful cleanup. Overall logged workflow span **16:04:08–16:09:36 CEST**; GPU allocations **59–151 s**, **0.217 GPU-hours** total, excluding CPU jobs/queue/accounting overhead.
- Cluster manifests record library checkout **+52dab0185e3d430b85dae661f4ef37aa2c61c1cf**, despite the superproject's **81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba** pin. Pull fetched objects without updating the submodule working tree. User should run `git submodule update --init tfm-library` before the next submission. The library supplies documentation, not imported training code; this mismatch does not invalidate numerical recovery evidence. Never silently rewrite the recorded pin.
- Added reusable `run_experiment0.sh pilot` with per-track base selection, fresh configs/plans under its workflow folder, actual per-track callback counts and a standalone `pilot_passed.json`. It preserves old recipes/receipts and cannot advance to long pilots or research. Next: user commits/pushes, pulls while writers are stopped, synchronizes the library and launches **12 PD + 8 LGD** pilots (v2/v2.6/v3 PD; v2/v3 LGD; both LR endpoints and adaptations; 250 updates). Old equivalent-size trial times **66–179 s** support **15-minute** requests. Training identity remains unchanged; this launcher-only addition changes workflow identity without requiring another recovery run. No new checked-in configs, local experiment output, training, install, commit, push or remote submission.
- Validation: **681 passed / 1 skipped in 650.41 s** (optional local data manifests absent); eight focused workflow checks and both real-shell submission previews pass. Previews cover **12 PD / 8 LGD**, paths containing spaces, 15-minute requests and no premature Slurm signal; their missing local VSC plans are expected, so these are submission-shape checks only. Training identity stays **6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28**; new workflow identity **fb37c92f29ec99010f469a35d9a23770e52d17cefd2057c42ec6e954ca812aae**. No notebook/visualization change; no notebook rerun required.

## Previous handover — 28-09-2026 TabPFN objectives corrected; targeted GPU validation required

- The requested loss/method review found a substantive mismatch: `_classification_loss` and `_ensemble_step_loss` included unused head columns, justified by an incorrect historical comment. At library pin **81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba**, `TabPFNClassifier.forward` selects active classes before `FinetunedTabPFNClassifier._forward_with_loss`; the installed local classifier also slices before inference softmax. Corrected both paths to active-class CE, preserving class permutations, member/query averaging and deterministic-compatible 2D CE.
- The same loss review found a second mismatch: TabPFN regression context labels can use a fitted `safepower` transform, but `_ensemble_step_loss` repeated the shared untransformed z-scored query across members. The upstream `_targets_in_estimator_space` maps queries with each member's already-fitted transform. The builder now carries member-specific query tensors through device transfer and NLL; missing regression targets fail rather than silently falling back. Two checks reproduced the defect using a context-fitted sklearn power transform and the native bar-distribution loss, and now pass.
- These repairs change positive-LR **TabPFN PD/LGD** gradients (LGD members with non-identity target transforms). A read-only pickle-metadata inspection confirms v3 `(None, 'safepower')`, whereas v2.6 uses `('none',)` (upstream identity `FunctionTransformer`); SHA-256 of both local base files matches the downloaded accepted budget plan. Upstream `_get_v2_config` also specifies `(None, 'safepower')`. Thus the known affected recipes are three TabPFN PD bases and v2/v3 LGD; v2.6 LGD has no target-space mismatch. Do not attribute old affected pilot degradation to CPT under the corrected recipe or use their receipt to certify corrected optimization. **TabICL objectives are unchanged.** Existing null/recovery evidence remains evidence for its original executable. Preserve it rather than rewriting identities or deleting all output.
- The unchanged TabICL PD/LGD and identity-transform v2.6 LGD curves still motivate the **10k** descriptive horizon, six milestones and **512 + 32 + 96** design. No blanket repeat of all eight 20k pilots is required solely for that horizon decision. Before main submission, validate the affected TabPFN paths with short positive-LR/recovery GPU controls under fresh run names/plans, then prepare research plans. The bundled workflow cannot be run over incompatible old plans and its old receipt must not be relabeled.
- New executable identities: training **6d309cc1f4db21d3ff6897f7fb1f64bd69cfd395dbe6c6289b90f87506a3dd28**, workflow **7662f78f5b7f1bb2a0ea17d3baaca72dc1d2a27d71c0e6261093b0fee9013918**. Changed method/source, not a reporting-only waiver. Grid counts and budgets are rechecked; existing Downloads and cluster artifacts are untouched. No foundation-model training, install, push, commit or submission performed.
- Final verification on the completed source: full **676 passed / 1 skipped in 576.68 s** (optional local data manifests absent). Both objective regressions were reproduced before their fixes; the compatibility repair also passed all 24 repeatability tests. Diff/parse checks pass. No notebooks or visualization implementation changed, no local experiment output created, and the read-only library remains at its existing pin.

**Tried:** Validate TabPFN classification CE against a second calculation over all head outputs, trusting the old comment's upstream claim.
**Result:** Existing tests passed while the objective differed from upstream active-class CE; eight corrected loss/gradient checks reproduced the mismatch and now pass after the fix.
**Why:** The oracle repeated the implementation's assumption instead of tracing upstream class selection before loss/softmax; unused logits changed loss and gradients despite being excluded at inference.
**Instead:** Test active-class loss/gradient equivalence, class permutations and invariance to arbitrary unused logits; retain the CUDA-compatible loss shape and revalidate affected PD controls before research training.

**Tried:** Share one context-z-scored regression query tensor across every preprocessed TabPFN ensemble member.
**Result:** A member with transformed context labels optimized NLL against query labels in different units; the regression-loss test and preprocessing/device-transfer test reproduced the mismatch.
**Why:** Only transformed context labels were retained from the upstream preprocessor; its fitted per-member target transform was never applied to query labels.
**Instead:** Transform queries using the returned member's fitted config without refitting, persist the per-member tensor in the batch, and test NLL/gradient alignment plus context-only fitting. Old affected LGD pilots require qualification.

**Tried:** Run the full suite after requiring per-member regression query targets.
**Result:** 675 passed / 1 skipped; the synthetic GPU repeatability helper's LGD batch failed the new guard.
**Why:** That diagnostic constructs its batch directly without the production preprocessing builder, so its identity-transform targets also needed the new field.
**Instead:** Populate the helper's per-member query targets; all 24 repeatability tests pass. Keep the missing-target guard rather than silently restoring the mismatched-loss fallback.

## Previous handover — 28-09-2026 budget trajectories reviewed; 10k research horizon

- Read all **eight** downloaded `cpt_budget_v5_check2*.trajectory.csv` files in place (**578,738 bytes** total); their names match the accepted budget audit exactly. Each has the seven ordered measurements **0/250/1k/2.5k/5k/10k/20k**, finite primary scores and monotone elapsed time/row exposure. Combined SHA256 of sorted `filename:content_sha256\n` records: **f2b75060e2a18994a3c77cd4de5a46bfd85678b425355022cb0f3c7b71305b98**. No new run, numerical failure or missing primary milestone.
- Equal-table **held-out credit PD AUC changes**, in percentage points at **5k / 10k / 20k**, are v2 **+0.727/+0.904/+0.950**, v2.6 **+0.217/+0.211/+0.312**, v3 **-0.007/-0.031/-0.217**, TabICL **+0.539/+0.694/+0.838** (four credit tables). Non-credit TabICL means cross baseline: **+0.490/-0.227/-0.395 pp**; the 10k median is still **+0.071 pp**, with two of four tables improved, so the mean decline is heterogeneous and largely driven by online shopping intention (**-2.366 pp**). PD v2 also loses **0.910 pp** on the mean non-credit panel at 10k; v2.6/v3 gain **0.249/0.804 pp**.
- All four models worsen on both held-out LGD tables at 5k/10k/20k. Mean **per-table relative RMSE increases** at **10k** are v2 **0.665%**, v2.6 **4.914%**, v3 **1.212%**, TabICL **5.710%**. Non-credit increases are **0.480%, 17.209%, 7.171%, 1.650%**, respectively. At 20k these means are **1.109%, 17.243%, 8.914%, 2.205%**; late behavior is not exhausted by 5k or universally by 10k. Do not pool raw RMSE across incompatible target scales or present an aggregate decline as decline on every table.
- Changed only the six research configs: **10,000 successful updates**, milestones **0/250/1k/2.5k/5k/10k**, epoch safety rail **4,000**. The late retention reversal and continued credit changes justify a wider descriptive window than the provisional 5k; 10k is a compute compromise, not convergence or score-maximizing selection. One conservative recipe/fold/seed per base/task and a 20k cosine schedule cannot predict every 10k-grid endpoint. Keep all 20k pilot evidence and disclose reuse of the monitored dataset fold in the descriptive main study. Trial counts remain **512 + 32 + 96**.
- Full-update reference elapsed times through 10k total **8.816 GPU-hours**; applying those base/task rates to 64 main trials each gives **564.2 GPU-hours**, or **543.6** when applying the short-pilot frozen/full timing ratios. These extrapolations do not measure other folds/recipes. Budget roughly **600–800 GPU-hours for main training**, excluding final evaluation and queues, as a planning allowance rather than a measured campaign cost. Counts and all six read-only design previews validate; main/seed scientific settings still agree after restoring seed 42. Main launch previews have 16 arrays/256 trials per track, 90-minute work segments plus a 10-minute margin; they explicitly lack prepared VSC plans. No plans or output are written locally. Current training executable identity stays **a0e59d910da7b66b5449888da9622ef6f7543d89d4b2228791f34352e20acf62**; changed scientific configs require fresh research plans on VSC.
- Experiment 0 remains accepted under its original executable. NaN trajectory `lr_applied` is the already-fixed historical reporting defect; gradient/loss/epoch-duration placeholders are not milestone measurements. Secondary regression MAPE is unavailable for zero-target tables, and these lightweight monitors provide no density NLL; their missingness is unchanged from update zero. No new runtime fix or full experiment-0 rerun is indicated. Next: user commits/pushes and pulls with writers stopped, CPU-prepares fresh main plans and runs preflight, then reviews submission previews before the 512-trial launch. Do not overwrite any incompatible pre-existing plan.
- Validation: full local suite **672 passed / 1 skipped in 513.84 s** (optional local data manifests absent). All six design previews, seed/reference parity, epoch safety rails, both main launch previews, proposed Bash command syntax and diff checks pass. No notebook/visualization implementation changed. Downloads, experiment-0 configs, runtime sources and the read-only library are untouched; no local output, training, installation, push or cluster submission.

## Previous handover — 28-09-2026 experiment 0 fully passed

- Read the refreshed `Downloads/output CreditPFN.zip` in place: **213 files / 18,985,269 uncompressed bytes**, ZIP **2,103,206 bytes**, SHA256 `4758387fc86d9a6ea89b8fe85a902b63885e9f224fabc60f5b2c89b7a61087b5`. CRC, **112 JSON / 10 CSV** files parse. Both passing receipts equal their workflow states and share the original fingerprint **da6b75f824979fc03d9e8fd1c04b3ab60d29a0f27f6270cbb325ddfbf8f2a2fb**; neither has failed tasks.
- CPU audit **62165438**, restart 1, finished **28-09 at 13:25:00 CEST**, application exit 0; supplied scheduler record is **COMPLETED 0:0, 00:01:36**. Both budget reports pass: **4/4 PD + 4/4 LGD**, each **20,000 successful updates**, zero pending/diverged trials, zero problems and zero data/optimizer skips. All **3,159 GPU resource samples** report sampled. The audit checked the project-side checkpoint identities, trajectory milestones, credit/OOD scores and diagnostic files. The prior environment hold is resolved for this job; this does not certify site-wide maintenance restoration.
- **Experiment 0 is complete; do not clean or rerun it.** The ZIP still contains only DATA-side logs/manifests, not the eight project-side budget trajectory CSVs. Obtain those to inspect credit/OOD curves before fixing the common research horizon (provisionally 5,000). Then prepare fresh research plans and inspect CPU preflight/submission previews before launching experiment 1. Experiments 2/3 need not wait for all of experiment 1, subject to the fixed budget and sampling-mode checks described below.
- The user has committed the prior reporting/control-flow fixes as local HEAD **f647304**; no new runtime change was needed here. Existing experiment-0 receipts certify the original executable, not that commit; those fixes do not change successful optimization and do not justify repeating the GPU pilots. The unchanged tested source retains **672 passed / 1 skipped** from the preceding audit; no repeat suite for this documentation-only update. Current library pin is **81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba**. Downloads remain untouched; no training, install, push, cleanup or remote action.

## Previous handover — 28-09-2026 final repository audit; research launch gates remain

- Reproduced and fixed two reporting/control-flow defects: revisiting a fingerprinted DIVERGED checkpoint returned success without incrementing the failure counter; `lr_applied` duplicated the next scheduled rate rather than the last successfully applied rate. Revisited failures now remain nonzero without retraining; applied rates are captured before stepping and preserved through recovery. No optimizer, objective, sampling, schedule or learned-weight change. The earlier GPU-peak reporting repair remains included and uncommitted.
- Validation: the new checks first reproduced **seven failures / one pass**; after repairs, full `pytest -q` reports **672 passed / 1 skipped in 481.74 s** (optional local manifests absent). All **111 tracked Python files / 18 YAML files / 17 shell scripts** parse. Read-only preparation confirms **512 main + 32 seed + 96 sampling** and PD held-out counts **4/4/5/4**, LGD **2/2/2/2**. Seed references match the main scientific settings except training seed. All six launch previews have correct counts/routing and 90-minute segments plus a 10-minute margin; these are submission-shape checks, not prepared cluster plans. No local output tree or cluster job was created; the library pin remains **e5ce01614eebe520af303f2b5bfd212298eab2be**.
- Local identities are now training **a0e59d910da7b66b5449888da9622ef6f7543d89d4b2228791f34352e20acf62**, workflow **4e845f15d0f199f653ba41ab07fa06b535e78085e1e723d6c3a10299d149e027**. Do not pull these changes into the old-source audit's checkout or relabel existing plans. The successful experiment-0 weights remain scientific evidence; these reporting fixes do not warrant a full GPU rerun.
- Remaining: confirm Lustre restoration (live Leuven page still lists the incident), finish original CPU audit **62165438** and obtain `part2_passed.json`; inspect the eight project-storage budget trajectories, including per-dataset credit/OOD behavior, and fix the common horizon (currently 5k). Then the user commits/pushes, pulls with writers stopped, prepares new research plans and runs CPU preflight; inspect quota and submission previews before releasing experiment 1. Production-size final evaluation still needs its own capacity/timing profile.
- Experiments 2/3 initialize from base weights and may train before experiment 1 finishes. Experiment-2 analysis needs its matching seed-42 references; experiment 3 includes its own one-sample controls. Share fixed code/data/budget and the per-controller concurrency pool. Profile full-pass/accumulation on GPU, including a saved segment, before releasing all 96 sampling trials; experiment 0's production pilots covered one_sample. Launching another experiment after all main arrays are queued may leave it behind shared-pool dependencies, not start it immediately.

**Tried:** Revisit an existing fingerprinted numerical failure through the actual training entry point in a synthetic regression.

**Result:** The manifest said DIVERGED but the process returned zero and printed overall OK.

**Why:** The skip branch preserved the checkpoint but omitted the divergence counter used by the final exit status.

**Instead:** Count the retained failure, preserve its checkpoint, and test both failed and successful existing outcomes without retraining.

**Tried:** Compare epoch/trajectory learning-rate measurements with actual successful optimizer calls, including interrupted recovery.

**Result:** Epoch `lr_applied` was one scheduler step ahead; trajectory applied rates were missing, and skipped-only epochs still reported a rate.

**Why:** Reporting read the optimizer's rate after `scheduler.step()` rather than remembering the rate used for the successful update.

**Instead:** Capture before stepping, retain only successful applied rates in recovery, and leave skipped-only epochs unknown; assert exact synthetic model-state recovery in every sampling mode.

## Previous handover — 28-09-2026 confirmed Lustre maintenance; wait for restoration

- User supplied the [official status page](https://status.vscentrum.be/); its [Leuven detail](https://status.vscentrum.be/tier2_leuven.html), updated 09:52:56, confirms Lustre scratch/project storage unavailable on all login/compute nodes **28-09, 09:00–17:00 scheduled**. Genius/wICE are reserved for maintenance. This explains today's transfer-path failures and supports maintenance as the reason audit **62165438** remains pending after release; the original 26-09 environment-retrieval hold predates this window and is not explained by it.
- Supersedes the filesystem-search steps below: wait for confirmed restoration, then retry the original trajectory download and inspect the existing audit's accounting/log/receipt. Do not relocate output, delete anything, resubmit training or pull the local telemetry fix while the old-source audit is pending. CreditPFN requires Lustre for durable weights/diagnostics, so it is ineligible for the temporary Mindwell `no-lustre` reservation. **17:00 is an announced endpoint, not a verified restoration time.** No new jobs or code changes; documentation/diff checks only.

**Tried:** Diagnose the missing project directory through repeated transfer/path checks before checking the live VSC status page.

**Result:** scp could not enumerate training output; a follow-up search was silent. The user then identified the scheduled storage outage.

**Why:** Lustre project storage is deliberately unavailable during maintenance; the pinned documentation describes normal operation, not live outages.

**Instead:** Check the relevant site's live incident page first, preserve completed experiment evidence and pending jobs, and verify restoration before retrying downloads.

## Previous handover — 28-09-2026 memory reporting repaired; released audit still pending

- The user's follow-up SSH `find` returned only the authentication banner and then the PowerShell prompt, with no matches or visible error; no exit status was supplied. This does not establish why the files were not found. Current routing/writer code still resolves the logged project path and `.trajectory.csv` suffix. Inspect directories and symlinks directly in an established VSC shell next; do not infer deletion from a silent search or patch the writer without runtime evidence.
- The subsequent user scp attempt authenticated but SFTP could not enumerate `/lustre1/project/stg_00211/CreditPFN/output CreditPFN/experiment0/training/`. Preparation log **62164623** confirms that exact directory existed with two entries on 26-09; current visibility is unresolved. `login.hpc.kuleuven.be` is the documented load-balanced endpoint, so neither a wrong host nor deleted output is established. An agent-side noninteractive SSH read was rejected for missing authentication; no remote command ran. Next: authenticated, read-only `find` through the user's SSH session on that endpoint. Do not blindly repeat the same scp command, invent another hostname, or infer a training failure. This follow-up changes documentation only; the preceding **670-pass / 1-skip** code verification remains applicable.
- User released CPU audit **62165438** with `scontrol -M wice release` at 09:43. The supplied 09:54 queue snapshot reports **PENDING (None)**; the earlier environment-retrieval hold reason is absent, but execution/completion is not established. Account output reports **70,702,118 available / zero reserved**; these accounting units are not GPU-hours. Keep the original cluster checkout until the audit finishes; do not resubmit the eight completed budget trials.
- Fixed the reset-counter defect in `src/train/telemetry.py`, `src/train/loop.py` and `scripts/train_pipeline.py`: independent trial maxima survive epoch resets and resumptions, partial-epoch peaks survive recovery, and epoch CSVs now receive the allocated/reserved byte measurements previously present only in console output. Successful manifests use the returned trial peak; unmeasured SKIP/FAIL/INTERRUPTED attempts report unknown rather than a misleading counter. No optimizer, loss, sampling or model-state change. Existing downloads/manifests are unchanged; the old manifest peaks remain invalid for resource analysis, and historical console maxima are rounded evidence, not exact reconstructed measurements.
- Runtime identities after this local telemetry change: training **c7f91887ba9064d1916e1098d295166da6eff043500cb409976cf7a276dc35da**, workflow **bd8e9b00254f74d330b036921bf186ae8eb52dcb28f77d33394c2d875c500145**. The completed pilots used the earlier identities recorded below. Do not relabel their plans or run their pending audit under changed source. Preserve their scientific evidence and prepare future research plans with the new identity after the audit; a reporting-only repair does not justify repeating paid training.
- Experiment 1 estimate at the provisional **5,000 updates / 512 trials**: actual first-5k long-pilot intervals are **38.1–52.6 min PD / 18.7–26.2 min LGD** for full updates. Scaling each base/task by the short-pilot frozen/full throughput ratio, then by 32 recipes/folds per adaptation mode, gives **272.3 GPU-hours**. Allow approximately **300–400 GPU-hours** for planning, with unmeasured fold-composition variation; at 16 continuously occupied GPUs that is **19–25 hours before queue delays**. Five-fold final evaluation/HPO is additional and has not been profiled at production settings. Summed padded short-pilot walltime requests would be **502.9 hours**, not measured usage. Do not equate requested limits, allocated GPU-hours and SAM billing units.
- Provisional horizon recommendation remains **5,000** for the broad sweep, retaining the existing 20k reference trajectories. Logged held-out PD AUC at baseline/5k/20k: v2 **.76257/.76984/.77207**, v2.6 **.76829/.77046/.77141**, v3 **.77422/.77415/.77205**, TabICL **.68512/.69051/.69350**. LGD RMSE: v2 **.22031/.22116/.22184**, v2.6 **.21993/.22840/.22738**, v3 **.22058/.22253/.22311**, TabICL **.28327/.29624/.30202**. These support neither uniform improvement nor universal saturation at 5k. Only one recipe/fold was run; the per-table/OOD files are still needed. A 20k cosine schedule's update-5k state is not the endpoint of a 5k cosine schedule. The pilot informs affordable observation duration, not an optimal-step claim; no configuration changed on these aggregates alone.
- Next data needed: the eight project-side `experiment0/training/{pd,lgd}/cpt_budget*.trajectory.csv` files, downloaded directly to Downloads using SFTP/scp. OpenSSH for Windows **9.5p2** is installed locally, so default scp uses SFTP and accepts the quoted remote path with its space. Review credit/OOD trajectories and the final CPU audit before settling the horizon or releasing experiment 1. No deployment, push, cluster job, cleanup, output copying or submodule edit performed here.
- Validation: focused telemetry/training/design checks **102 passed**; complete `pytest -q` **670 passed / 1 skipped** in **600.33 s** (optional local data manifests absent). Regression coverage includes reset counters, device selection, independent trials, partial-epoch resume, final monitoring peaks, and exact synthetic model-state recovery in all three sampling modes. Source/diff checks pass; no local output tree was created. Fix remains local and uncommitted.

**Tried:** Download budget trajectories through scp using the project path in the completed-job logs.

**Result:** Authentication succeeded, but SFTP returned `No such file or directory` for the training directory.

**Why:** Current path/storage visibility through that transfer connection is unverified; the available evidence does not distinguish a changed path from a mount/access issue.

**Instead:** Locate the files with a read-only `find` in an authenticated SSH shell on the same endpoint before issuing another transfer command.

**Tried:** Use the final manifest's GPU-memory counter to summarize the completed long pilots.

**Result:** PD v2 reported 0.183 GB while epoch console measurements reached 181.08 GB; ordinary epoch CSV memory columns were zero.

**Why:** Epoch logging reset CUDA peak counters before the manifest read them, and epoch records never received the measured byte values.

**Instead:** Retain per-trial and partial-epoch maxima in recovery state, return the cumulative peak to the manifest, and write interval peaks before resetting. Use historical console measurements for old runs without modifying their original records.

## Previous handover — 28-09-2026 budget training complete; CPU audit held by Slurm

- Read `Downloads/output CreditPFN.zip` in place: **210 files / 18,789,984 uncompressed bytes**, 2,090,695-byte ZIP; SHA256 `f535daec48f86873cd8ffce42b5443aa9df69c8af17e08d0c514100ec0bceda6`, CRC valid, all **110 JSON** parse. **Ten plans / 72 trial identities** verify against unchanged runtime **8e7e5a23d24ce1eddcf7ad90809fd4f5cc6dfc6d1f3a267857f46ede03ff37ed**. DATA-side records only; no project training/result files. Downloads remain unchanged.
- Part2 workflow **22a3be3a479b4ed99b6c9701dcc896fb** has **8 done / 0 failed**, phase `budget`, status **auditing**, matching workflow fingerprint **da6b75f824979fc03d9e8fd1c04b3ab60d29a0f27f6270cbb325ddfbf8f2a2fb**. Preparation **62164623** finished 26-09 at 12:22:22 CEST. All **8 final manifest rows are OK at 20,000 updates**, and all 12 budget allocation logs end with exit 0; no OOM, traceback or nonfinite-loss warning, and **zero data/optimizer skips**. Four PD interruptions are deliberate saved segments, each followed by a successful single requeue. Part1 remains accepted; **no part2 receipt yet**.
- The user reports CPU audit **62165438**, wICE, PENDING with reason **user env retrieval failed requeued held**, zero runtime. Its log is absent and no budget audit report exists: Slurm blocked startup before the script ran. `_cpu` submits `--export=ALL,CREDITPFN_EXPERIMENT=experiment0`, a form that implicitly invokes login-environment retrieval in Slurm. Official sbatch docs describe this hold on retrieval failure/timeout; they do not identify this instance's underlying login/node cause. VSC's pinned documentation also describes environment reconstruction. No evidence of another model-training failure.
- **Next:** inspect/release existing audit once with `scontrol -M wice release 62165438`, then check `sacct`/its new log. If release is denied or the same hold recurs, inspect the job details before arranging an audit-only replacement; do not start part2 again, clear claims, cancel unrelated work, clean outputs or retrain. Keep source/configs unchanged until the pending audit has finished. The existing audit alone can publish the final part2 receipt; release is not itself a pass. No remote operation was performed here.
- GPU jobs finished **26-09 at 15:54:25 CEST**, **3 h 32 min 35 s** after preparation began. Sum of budget GPU allocation log durations **17.63 h**; cumulative training-loop durations **17.56 h**. Per-model PD v2/v2.6/v3/TabICL training minutes **174.4/150.4/184.7/208.5**, LGD **74.2/103.9/80.4/77.0**. PD resumed at **13,667 / 15,840 / 12,884 / 11,391** updates respectively. Job IDs: PD v2 **11619746**, v2.6 **11619745**, v3 **11619747**, TabICL **11619744**; LGD v2 **11619750**, v2.6 **11619749**, v3 **11619751**, TabICL **11619748**.
- Logged held-out monitor endpoints are mixed: three PD AUC increases and one decrease; all four LGD RMSE endpoints exceed baseline. These are one-fold training monitors, not final five-fold benchmark estimates. All six positive-update milestones are logged per trial; full seven-row trajectories, per-dataset credit/OOD measurements and parameter/resource histories are on project storage under `output CreditPFN/experiment0/training/{pd,lgd}/`. Obtain the eight `cpt_budget*.trajectory.csv` files before deciding the main horizon; do not choose it from final aggregate scores alone.
- **Nonblocking reporting defect to correct before resource analysis:** `RunRow.peak_gpu_gb` is read after epoch logging resets CUDA peak counters. It reports **0.183–1.000 GB**, while the budget epoch logs record **11.00–181.08 GB**. The PD v2 example is **0.183 vs 181.08 GB**. Use the detailed measurements, not this summary field, for capacity/cost plots. This does not alter updates or weights; retain the current executable while the audit is pending and avoid retraining to repair metadata. The full-suite result remains **667 passed / 1 skipped** for unchanged runtime; no new code/test changes this inspection. Documentation/command syntax and diff checks only; no install, training, push, cleanup, output copying or library change.

**Tried:** Automatically submit the final budget audit after all eight training callbacks completed.

**Result:** Slurm held audit 62165438 during user-environment retrieval; the workflow remains `auditing`, with no script log.

**Why:** Slurm environment reconstruction failed before application startup; the underlying site/login cause is not identifiable from the downloaded files.

**Instead:** Release this existing CPU audit once; if it repeats, diagnose its scheduler details and replace only the audit, preserving trained models and their identities.

## Previous handover — 26-09-2026 fresh part 1 accepted; release part 2

- Read `Downloads/output CreditPFN (1).zip` in place: **176 files / 3,269,485 uncompressed bytes**, 539,066-byte ZIP; SHA256 `c0c49a0406c3848bc1b23ad75ac7ae8d9f4a00519420488f6bed5fc20835fa9d`. CRC, all **94 JSON / eight CSV** records, **eight plan checksums / 64 trial identities** verify. DATA-side logs/audits only: project weights and diagnostics were checked by the cluster audits, not downloaded here. Downloads remain unchanged.
- Fresh workflow **d8e878653d3e46dc9201772b88e50275** and `part1_passed.json` both say **passed**, with no failed tasks and fingerprint **da6b75f824979fc03d9e8fd1c04b3ab60d29a0f27f6270cbb325ddfbf8f2a2fb**, matching the current executable. Preparation **62164595** started **11:10:47 CEST**; final audit **62164601** ended **11:25:52 CEST**, **15 min 05 s** elapsed. All **60 job logs** end with exit 0.
- **16/16 null controls**: two zero-LR updates each, canonical state equality and identical monitored scores. **32/32 pilots**: 250 successful updates each, no divergence, **zero data skips / zero optimizer skips** in every audit and printed epoch. PD table 0011 is included in the pilot plan; its former systematic exclusion is absent. All required trajectories, parameter measurements and retention scores pass the cluster audits; **230 resource samples** all report `sampled`.
- **8/8 recovery pairs**: zero learned/inference tensor differences, equal monitors at **0/5/12**, passing selected-arm audits and **40 benchmark folds / 4,044 predictions**. Three LGD TabPFN detached criterion-diagnostic buffers differ as documented; they are not model drift. The eight intentional update-5 interruptions and `[no-monitor]` placeholders are expected. No traceback, error, nonfinite-loss warning or OOM appears.
- Timing: null GPU jobs **28–111 s**, short pilots **81–202 s**, recovery jobs **54–141 s**. The full-update pilot estimates for **20k** are PD v2/v2.6/v3/TabICL **265/232/274/291 min**, LGD **120/165/129/122 min**; total **26.63 GPU-hours**. These are short-pilot extrapolations, not guarantees. With eight allocations promptly available, the longest estimate is about **4.9 h**, plus staging, monitor/save/requeue overhead and queue delays. Budget configs remain **3e-7 / L2-SP 0.003 / full updates / one_sample / seed 42 / fold 0**, seven trajectory measurements through 20k.
- **Next:** run `bash scripts/slurm/run_experiment0.sh part2` in the active VSC CreditPFN environment. **Do not clean or rerun part1, rebuild data, or change runtime/configs.** Part2 has not run; its eight long pilots use automatic two-hour work segments plus a ten-minute allocation margin. Review their trajectories before settling experiment 1's horizon. No new runtime fix or deployment is needed. The supplied ZIP omits the general data manifests present in the prior download; their absence here does not prove cluster deletion and does not affect corpus selection (code + processed CSVs are authoritative). Preserve that preparation provenance for final analysis; do not launch a redundant data rebuild.
- Read-only verification and documentation only. Runtime identity remains **8e7e5a23d24ce1eddcf7ad90809fd4f5cc6dfc6d1f3a267857f46ede03ff37ed**; the unchanged runtime's last full local suite was **667 passed, 1 skipped** on 25-09-2026. No extra suite run for status documentation, no local output, training, install, push, cleanup or VSC submission. Library remains read-only at **e5ce01614eebe520af303f2b5bfd212298eab2be**. The part2 command parses; it was not executed locally.

## Previous handover — 26-09-2026 corrected corpus rebuilt; fresh part 1 next

- Read `Downloads/output CreditPFN.zip` in place: **182 files / 3,502,927 uncompressed bytes**, 572,210-byte ZIP; SHA256 `fe2b9fe97623f82ca4857c8aec2e4f2f90e7dd0e518a318a8334cdd3077dfc02`, CRC valid, all **95 JSON** parse. Its only new job log is CPU data preparation **62164553**, **10:24:38–10:54:08 CEST**, **29 min 30 s**, exit 0. All **25 processed tables (17 PD, 8 LGD)** were rebuilt on project storage. The two dataset manifests parse and contain complete unit-conversion records: **48 columns in PD table 0011, five in table 0014**, none in the other 23 tables. No preparation error is logged.
- The ZIP contains **no new GPU probe or training run**. The old part-1 receipt **5433645ed42e42f4825d441c1c2e6309**, fingerprint `e852f032...`, remains historical and cannot certify corrected inputs. Part 2 has never run. Local HEAD **bb4248c** contains the previously tested fix; training identity remains **8e7e5a23d24ce1eddcf7ad90809fd4f5cc6dfc6d1f3a267857f46ede03ff37ed**, workflow **da6b75f824979fc03d9e8fd1c04b3ab60d29a0f27f6270cbb325ddfbf8f2a2fb**.
- **Next:** with CreditPFN writers stopped, clear only experiment 0's output and trained-weight subtrees on both tiers, preserving `general/` manifests and canonical processed data. Then run a fresh full **part1** with measured 10/15/10-minute null/pilot/recovery requests. This is the next GPU validation; a separate zero-update diagnostic need not add another feedback round. The new zero-skip gates remain enforced. Inspect the fresh receipt and reports before part2; no GPU success is inferred from CPU preparation. General scheduler state has no uncertain submission and ignores completed job IDs when reserving lanes.
- Documentation-only inspection: **no runtime/config changes**, no local or cluster deletion, submission, training, install, push, library change or Downloads mutation. The unchanged runtime's full-suite result remains **667 passed, 1 skipped** from 25-09-2026; not rerun merely for documentation. Corrected the runbook's stale skip-audit wording and distinguished an experiment-0 reset from a full campaign wipe. The latter would unnecessarily discard the just-created general provenance. Bash syntax of the proposed commands is checked.

## Previous handover — 25-09-2026 probe isolates range and clipping defects

- Read `Downloads/output CreditPFN.zip` in place: **178 files / 3,284,185 uncompressed bytes**, 543,040-byte ZIP; SHA256 `ad7d49a6e1590921fd6915ac58573e3eb03e15ce910d25afa8d9dc9defcca6c3`, CRC valid; all **94 JSON** records parse. New maintenance job **11618877** ran **18:09:26–18:09:44 CEST**, **18 s**, exit 0, after user commit **e151e7f**. Zero optimizer updates/checkpoint writes; old part-1 receipt **5433645ed42e42f4825d441c1c2e6309** is unchanged. No part 2 exists.
- All six comparisons completed: current clipping fails under both precisions; upstream float32 clipping yields finite losses **0.654919 BF16 / 0.653881 FP32** but adds **1,888 NaNs per member**. Float64 clipping preserves the large finite values and still produces a nonfinite model loss. Inputs reach **3.392e38**. This isolates numerical preprocessing; changing BF16 alone does not cure it. Upstream's finite loss is not evidence of preserved information. Non-bitwise gradients under the default fast kernel profile are not a failed deterministic recovery test.
- Checked the two affected public raw columns: each has **104,773 finite values**, with maxima up to **8.993e41**. The old float32 cast lost **53,049 + 58,304 = 111,353 finite cells**; surviving values then overflow clipping/encoder statistics. Sanitization now changes extreme column units by recorded powers of two before casting, preserves actual missingness, and rejects underflow in rescaled columns. The rule uses whole-table range metadata and is explicitly part of the existing transductive schema preparation; it is not claimed to be an inductive fitted normalization.
- Corrected training clipping to the upstream two-pass logarithmic rule with sample standard deviation and **context-only** fitted bounds. It now refuses overflowing bounds rather than silently erasing features. Experiment-0 audits require valid zero data/optimizer skip counts; successful-update budgets alone cannot hide systematic exclusions. Factors go into the existing general dataset manifests; no new output directory or per-trial metadata shard. Fixed the data CLI's stale claim that existing tables were skipped.
- In-memory preparation of the complete affected raw table completed in **42.625 s**, with **105,471 rows, 64 selected features, identical targets** and **48 unit conversions before selection**. The two previously corrupted retained features are no longer selected after correcting the underlying values; this is a data/feature identity change, not just a logging fix. Local raw/processed files and Downloads remain unchanged. Production GPU behavior still requires validation.
- **Next:** user commits/pushes and pulls with writers stopped; rebuild all processed inputs on the CPU using the refreshed data pipeline, then run the five-minute real-table probe after that job succeeds. Preserve current evidence meanwhile. Once accepted, perform the requested clean restart and a fresh full part 1; a complete output wipe also removes the general dataset manifests, so rerun CPU data preparation after cleanup to restore that provenance before part 1. Do not reuse old plans, checkpoints or receipts, or launch part 2 early. Training identity **8e7e5a23d24ce1eddcf7ad90809fd4f5cc6dfc6d1f3a267857f46ede03ff37ed**; workflow **da6b75f824979fc03d9e8fd1c04b3ab60d29a0f27f6270cbb325ddfbf8f2a2fb**.
- Validation: the three initial regression tests reproduced finite-data loss, unintended in-bound shrinkage and the missing context boundary. **40 focused tests pass** after correcting a missing default in the extended manifest schema; the full suite passes **667 tests, 1 skipped** (optional local manifests absent), **513.59 s**. Bash syntax and diff checks pass. Source edits, tests and existing docs only; no VSC actions, real-model training, installs, push, notebook/output files or library changes. Library pin remains **e5ce01614eebe520af303f2b5bfd212298eab2be**; numerical source anchors are in LITERATURE.md.

## Previous handover — 25-09-2026 real-table probe stopped before its comparisons

- Read `Downloads/output CreditPFN.zip` in place: **177 files / 3,274,266 uncompressed bytes**, 541,171-byte ZIP; SHA256 `42ede171922b02ec2016e5f41ccb7827ad62434ca16b9cacc8a5cee46e196c13`, CRC valid. The only addition to the preceding part-1 bundle is maintenance log **11618868**. Downloads and local output remain untouched.
- On deployed **16ffeec**, the zero-update PD v2/table-0011 probe failed in **16 seconds**, **17:47:23–17:47:39 CEST**, exit 1. TabPFN raised `ValueError` for NaNs in encoded inputs during `current_bf16`; the probe caught only runtime/import errors and never reached FP32 or upstream comparisons. This confirms an encoder failure, not its preprocessing/precision root cause. No training updates or checkpoint/receipt writes occurred.
- Reproduced the same early exception locally. The probe now records `ValueError` alongside expected runtime/import errors, prints aggregate inputs before the first forward and each result immediately, then continues. Six settings compare current clipping, upstream clipping, and upstream clipping with float64 calculations, each with BF16/FP32 model arithmetic. Wider clipping is diagnostic only; model inputs retain their original dtype. State/RNG restoration remains tested; unexpected programming exceptions still propagate. Reuse the exact known-warning filter without hiding model errors.
- **Next: rerun only the five-minute maintenance probe after deployment.** Keep part 2 blocked by the systematic PD v2 table exclusion documented below. Once the real training correction is established and verified, use a fresh part 1 to certify one consistent executable; part 2 has never run. The diagnostic edit changes the broad workflow fingerprint, not training identity `9a85958d0285...`; do not rewrite the previous receipt to authorize longer jobs. No production training/config/preprocessing edits, cleanup, submissions, installs, push or library changes this turn.
- Validation: **653 passed, 1 skipped** (optional local manifests absent), **453.01 s**, including 24 diagnostic tests. The encoder-error regression failed before the fix and passes afterward. CLI help, proposed Bash syntax and diff checks pass. Training identity is unchanged; new workflow identity **208f1d7936d883ec635a6ef33f676c3e8e22c3f4d1b27bc06047d3f2a93f6938**. Real-table GPU comparisons still require the user-run probe; local tests do not certify that outcome.

## Previous handover — 25-09-2026 part 1 passes its gates; systematic PD v2 skips need diagnosis

- Read `Downloads/output CreditPFN.zip` in place: **176 files / 3,270,286 uncompressed bytes**, 539,594-byte ZIP; SHA256 `d3473aa17724c1b2e542fef79cfb0202c4819005852e4f99b6d05c6a535cb222`. CRC, all **94 JSON / eight CSV** files and all **eight plans / 64 trial identities** verify. DATA-side output only; project diagnostics/weights were inspected by cluster audits, not downloaded here.
- Deployed **a50c38d**, workflow **5433645ed42e42f4825d441c1c2e6309**: **16/16 null controls, 32/32 short pilots and 8/8 recovery pairs passed the automated gates**. `part1_passed.json` matches that checkout's workflow identity `e852f032cab5...`. All 60 logs end with exit 0; final audit **62154541** finished **17:17:44 CEST**. No part-2 run exists. Nulls have exact canonical state/monitor parity; pilots each reached 250 successful updates; 228 null/pilot GPU samples all report `sampled`.
- Recovery checks report zero model-tensor differences, matching 0/5/12 trajectories, and eight passing five-fold benchmark smoke checks (**4,044 predictions**). Three LGD TabPFN detached loss-diagnostic buffers differ as documented. Expected interruption rows and `[no-monitor]` placeholders are not failures.
- **Hold part 2 despite the automated receipt:** every PD v2 pilot skipped **21 non-finite losses on table 0011**, across both LRs and adaptation modes. With 13 one-sample visits per epoch and 21 epochs, this means every visit to that table was skipped; the 250 updates came from the other tables. PD v2 recovery also skipped it once per comparison arm. Update-count/parity gates do not certify per-table training coverage. Other pilots log no numerical skips.
- Local table bytes match the cluster plan after CRLF-to-LF normalization. The table has two numeric columns above 1e30 and a maximum magnitude about **3.403e38**. These are candidate numerical triggers, not a proved GPU root cause. Reproduced a separate source discrepancy: `apply_outlier_clip` changes in-bound values using a rational shrinkage, whereas upstream `TorchSoftClipOutliers` uses two-pass context-fitted logarithmic clipping. The local helper also restores extreme inputs after overflowing its variance calculation. Library pin **e5ce01614eebe520af303f2b5bfd212298eab2be**; production clipping remains unchanged pending the targeted diagnostic.
- Extended the existing `gpu_repeatability` utility with `--table-number`: keep the original corpus index/seed, use at most 2,048 rows, compare current/upstream clipping under BF16/FP32, restore state/RNG, and print numerical aggregates only. **Zero optimizer updates, no checkpoint or workflow writes.** User commits/pushes, pulls with writers stopped, then runs the bounded maintenance probe documented in VSC.md. Keep all current output/weights; do not clean, rerun part 1, or launch part 2 before reviewing that probe. Training identity stays `9a85958d0285...`; the diagnostic utility changes the broader workflow identity, so the old receipt must not be bypassed.
- Measured job times: null **27–57 s**, pilots **82–202 s**, recovery **55–142 s**. Audit estimates for the eight 20k full-update recipes total **26.5 GPU-hours**: PD about **3.9–4.8 h/trial**, LGD **2.0–2.7 h/trial**, plus queue/segment delays. These extrapolate 250-update measurements and do not justify launching with unresolved table exclusion.
- Validation: **650 passed, 1 skipped** (optional local data manifests absent), **529.34 s**; includes 21 diagnostic tests. CLI help, proposed Bash commands, maintenance syntax and diff checks pass. New workflow identity `c0e523e4fd77f2a03b3ced6709efa212c1c42d33cbe3a04aca3362249006faf5`; training identity remains unchanged. The old receipt is retained as evidence of the deployed run, not rewritten for the diagnostic revision. No production loss/preprocessing/config edits, real-model training, VSC submission, install, cleanup, push, submodule change or Downloads mutation. Local output still has zero files. Actual real-table GPU diagnosis is the next user-run step.

## Previous handover — 25-09-2026 signal retry reaches training; four old checkpoints block part 1

- Read `Downloads/output CreditPFN.zip` in place: **54 files / 852,196 uncompressed bytes**, 127,533-byte ZIP; SHA256 `4ab77c4d955562349841832ed2c8be0850cb2c005e858071ab50e337315a16fa`. CRC, 27 JSON records, two CSVs and all eight plan checksums pass. DATA-side output only; project diagnostics/weights are not included. Downloads were not modified or copied.
- Workflow **588ddf3d0790427690dc8cc742c259f2** on deployed **f914f29** stopped at `null`, status **failed**. Its ledger has **12 done / 4 failed**, matching all 16 task logs. Preparation **62152033** exited 0 with zero preflight failures/warnings. No null audit, short pilots, recovery jobs or `part1_passed.json` followed; no budget pilots ran.
- All eight LGD controls and PD v2/v3 in both adaptation modes completed: **two successful zero-LR updates, zero drift, finite scores and exact reported before/after monitor parity**. Job log durations were **90–156 seconds**, training-loop durations 10–35 seconds. No new numerical/OOM/signal failure appears in these logs; tensor-state/resource audits were not released, so completion is not a full null-control pass.
- Four PD v2.6/TabICL tasks refused existing checkpoint identities before training (30–65 s, exit 1). These are exactly the four arms completed before the signal fix. The guard prevents mixing versions; project checkpoint sidecars themselves were not supplied, so their old exact hash is not independently re-read here. Prepared training code is `97d4de5e28ae...`, workflow `dd97c09b06c7...`; local pending VSC fixes remain `9a85958d0285...` / `e852f032cab5...` and were not deployed.
- Next: commit/push pending local fixes, confirm stopped CreditPFN writers, pull on VSC, preview then perform the complete two-tier cleaner (including trained weights), and wait for successful cleanup before launching **part1** with 10/15/10-minute null/pilot/recovery requests. Do not bypass identities or delete only `output CreditPFN/`. Review the new part-1 receipt before the **first** part-2 run; no long run needs repeating.
- This inspection changes documentation only. **48 cleanup/path tests passed** in 1.67 s; downloaded-record checks and Bash syntax checks of the proposed commands pass. Runtime identity is unchanged since the preceding full **642-pass/1-skip** audit; that full suite was not repeated for this documentation-only update. No cluster actions, installs, runtime edits, local output files or submodule changes.

## Previous handover — 25-09-2026 VSC execution audit; resubmission outcome pending

- User reports resubmitting part 1 after `f914f29` (signal fix); no new job IDs/results supplied. Leave those jobs and their checkout unchanged until they finish. The earlier 4/16 snapshot below is not the status of this resubmission.
- Inventoried the 322-file VSC documentation dump and screened its requirements across sites; read the applicable Leuven, job submission, storage and Conda/Python sections in full. Library pin remains `e5ce01614eebe520af303f2b5bfd212298eab2be`, unchanged. Cross-checked current official Leuven storage/Slurm pages. Mindwell one-GPU training requests of 24 cores/120 GiB are within documented limits; live QOS, project quota and GPU runtime remain cluster measurements.
- Reproduced native CPU over-allocation with a simulated two-core task (Torch retained 16 threads; boosting defaults could use all host cores). Batch activation now resets native pools per allocation; training reserves worker cores and respects affinity; boosting fits, CatBoost pools and predictions have explicit ceilings. Thread settings must be reset rather than merely inherited across GPU → CPU audit → GPU stages.
- Frequent training diagnostics previously opened/updated project-Lustre files from Mindwell. They now accumulate under GPFS `output CreditPFN/<experiment>/training_work/<attempt>`, then publish atomically per file to the existing project `training/<track>` folder on completion/interruption/error. Publication failure retains working files and fails the trial. A hard kill can leave unpublished resource samples on GPFS; inspect before retrying/cleanup. No permanent third output archive or Downloads copy was created.
- Corrected the default scratch pointer for optional wICE GPU routes; capped queue/submission response waits at 45 s without allowing duplicate retries after uncertain acceptance. Added a login-shell header and explicit defaults to the standalone hardware report, plus CPU/GPU/memory fields in job logs. Production GPU caps/settings and all scientific configs remain unchanged.
- These local runtime changes alter prepared training/workflow identities (training `9a85958d0285...`, workflow `e852f032cab5...`). Do not pull while the reported resubmission runs. Review its results first; with writers stopped, deploy and use the requested clean part-1 validation before part 2. Never bypass fingerprints or manually reset active workflow/pool state.
- Validation: **642 tests passed, 1 skipped** (optional local manifests absent), in **468.56 s**; all 17 shell/Slurm files parse and retain LF endings; `git diff --check` passes. Focused checks cover actual Bash setup, per-stage CPU budgets, GPFS publication/interruption/failure and uncertain scheduler responses. No notebooks/visualization changes. No VSC jobs, installs, cleanup, push, local output files or submodule changes by this audit.

## Previous handover — 25-09-2026 short-job signal defect confirmed; part 1 needs a corrected retry

- Read the new `Downloads/output CreditPFN.zip` directly: **38 files / 695,716 uncompressed bytes**, ZIP SHA256 `53a673f4dde69edf37ac484fb1dbdda7aedd7a448e4fa29e598c27aecfae27e4`. CRC, 15 JSON files, the one CSV and all eight prepared-plan checksums pass. Plans match the deployed `20bc8a8` training identity. DATA records only; project training/results/weights are not in this download. Downloads remain unchanged.
- Workflow **9278d539651d4342add4a6e1495e5091**, fingerprint `4f3952f7...`, reached `null`. CPU preparation **62151928** passed with zero failures/warnings. Only **4/16 null trials** completed: PD v2.6/TabICL, full and frozen, each two updates and reported zero drift. No null audit, short pilots, recovery comparisons, part-1 receipt or budget pilots were released. The ledger still says `running` because the other jobs died before their callbacks; it is not a live scheduler status.
- User's `sacct` confirms the other **12 array tasks FAILED with ExitCode 0:10 (SIGUSR1)** after 27–30 s; both queues were empty at 15:39. Ten downloaded startup-only logs misleadingly say exit 0; two failed tasks left no included log. The launcher unconditionally requested `--signal=B:USR1@600` even with the recommended `00:10:00` null allocation. The resulting immediate warning killed jobs before the training handler was installed. Synthetic Bash tests reproduce the misleading footer; this is a launch defect, not evidence of a model or CUDA numerical failure.
- Request warnings/requeue only for positive `SEGMENT_MINUTES`, whose walltime adds the ten-minute publication margin; install nonzero signal handlers before activation and restore them after training children. No scientific config, model, dataset or notebook changes. New training identity `97d4de5e28ae3ee4ac23b21f1ba75f3a6ce0bfdef27fd6ed8143b3a1e0d1bf8f`; workflow identity `dd97c09b06c7e0c3f270b99feddc8b10213772aaec399102096d4272f67aa5fb`. User commits/pushes/pulls after validation, confirms no writers, performs the requested clean restart, then retries **part1** with the same 10/15/10-minute requests. Do not reuse the old immutable plans or launch part2 yet.
- Preliminary part-2 planning estimate: historical 250-update pilots took **83–207 s**. Multiplying by 80 for 20k gives **1.8–4.6 GPU-hours per trial**, roughly **15–40 GPU-hours for eight**, before additional segment/queue overhead. This is a coarse extrapolation across prior recipes, including fixed startup/monitor time, not a measured long-run ETA. Fresh full-update short pilots should refine it. Default segments request 2 h 10 min; completed work resumes without resetting the 20k schedule.
- Validation: **627 tests passed, 1 skipped** (optional local manifests absent), in **479.07 s**. The old code failed ten focused cases; repaired tests cover short/segmented submission arguments, startup signals, restored handlers and warning forwarding that preserves recovery exit 75. All **17 shell/Slurm files** and proposed retry commands parse; LF endings are intact. Notebooks/visualization were untouched. No VSC submissions, cleanup, installs, pushes or local output generation by the agent.

**Tried:** Shorten null allocations to ten minutes while retaining the launcher's unconditional ten-minute warning.

**Result:** Twelve of sixteen controls terminated with SIGUSR1 before training; old log footers incorrectly displayed exit 0 and callbacks never ran.

**Why:** The warning lead time equaled the allocation; the shell had no startup signal handler. Earlier syntax checks and longer allocations did not exercise this interaction.

**Instead:** Emit the warning only for resumable segments with an added margin; test real submission arguments, startup signals, handler restoration and forwarding to the training child. Confirm success through audits and scheduler accounting.

## Previous handover — 25-09-2026 final local audit before the clean experiment-0 rerun

- Reviewed splitting, preprocessing, sampling/gradients, frozen modules, update budgets/recovery, monitoring, five-fold scoring, tuning/calibration, output publication, analysis and VSC launch/cleanup boundaries. Confirmed primary literature at unchanged library pin `e5ce01614eebe520af303f2b5bfd212298eab2be`. The descriptive design remains **512 main + 32 seed + 96 sampling trials**, with 5k still provisional pending the eight 20k reference pilots. Equal updates do not equal exposure/compute; IID/transductive schema preparation, source dependence, a small retention panel and non-independent folds remain explicit limits.
- Fixed reproduced defects: query-batch-dependent TabICL zero-filling; evaluation completion published before required predictions; ambiguous class-column padding; sentinel/delimiter collisions in feature deduplication; missing experiment layer in checkpoint relocation. Linear controls now use context-fitted one-hot categories, with a category-renumbering regression test. Requested HPO no longer silently falls back when Optuna is absent; result rows retain selected settings and actual trial counts, and CPU preflight checks declared baseline dependencies. Removed unused private AMP/CSV helpers and the mechanistic interpretation/fitted trend from gain-versus-base plots.
- Read-only comparison on **all 25 local raw tables (3.79 GiB of CSV bytes)** found identical duplicate-column decisions before/after the dedup correction; the current processed corpus need not be rebuilt for this fix. No raw/processed data, weights, Downloads, archive or library content was changed. The downloaded ZIP hash remains `e2ed5330e1e6da5ac7c8138c327c7f56815ccee1050aafc6e35139b5ec78833d`.
- Changes are local/uncommitted on top of `3037f36`; no install, push, VSC submission or cleanup. New training identity `0806e3c76754542556bebe1459c2cec9021a88e5bae5f099c14879387aed6305`, workflow identity `4f3952f74e0993df515081a070154f41ba8d992ec72153deb95047e72421133a`. Old successful GPU controls remain historical evidence and do not certify this changed source. After local validation, user commits/pushes/pulls, checks for active CreditPFN writers, cleans output/trained weights on both tiers, then reruns part1. Review its receipt before part2; settle the budget before preparing experiment1. Preserve the accepted fresh experiment0 when advancing.
- Validation: **615 tests passed, 1 skipped** (optional local manifests absent), in 551.66 s. All **11 notebooks passed**, producing **35 PDFs** in temporary output; their exact original bytes were restored. The changed descriptive plot also rendered with synthetic paired results. All **17 shell/Slurm scripts** and four proposed VSC commands parse; 108 Python syntax/global-name checks and local baseline dependency checks passed. No local output tree remains. This establishes local validation, not a guarantee of defect-free CUDA execution or production-size evaluation capacity; the fresh cluster controls remain required.

**Tried:** Inject a prediction-write storage failure after five successful folds and compare single-row/mixed-batch TabICL sanitization.

**Result:** The old evaluation was skipped as complete despite missing predictions; a missing query value changed to zero only when scored alone.

**Why:** Completion preceded artifact publication; the query batch recomputed the training-derived dead-column mask.

**Instead:** Publish predictions atomically and the receipt last; check required artifacts on resume and retain the context-derived mask.

## Previous handover — 25-09-2026 all eight recovery pairs pass; clean full validation next

- Read `Downloads/output CreditPFN.zip` directly, unchanged: **358 files / 5,666,579 uncompressed bytes** (933,425-byte ZIP), containing 113 logs, 243 experiment-0 manifests and two general scheduler files. ZIP CRC and all **188 JSON / 18 CSV** records pass parsing. This download contains DATA-side records; project-side training/results and weights were not downloaded.
- User deployed **3037f36**. Isolated workflow **22af75990edf457bbeb43698e17e947f** and its `recovery_passed.json` both say **passed**: eight completed pairs, zero failed. CPU preparation **62149265** and final audit **62149461** exited 0; the final audit finished **25-09-2026 13:41:40 CEST**. Its fingerprint **05f6a5a282be997e56865bde5fd7d2d8912217244f02eb6afb7512c4a88629fa** matches the repaired source.
- All eight cluster comparison reports show **zero model-tensor difference**, identical monitored scores at updates **0/5/12**, and passing selected-arm audits. Each resumed arm deliberately stopped after update 5 and finished at 12. Three LGD TabPFN detached criterion diagnostics differ; the documented exclusion remains explicit. All eight five-fold benchmark smoke checks passed (**40 folds, 4,044 predictions**, according to cluster reports). The ten new job logs all end with exit 0; GPU allocations took **63–153 seconds** each.
- PD v2 skipped one non-finite-loss batch in each comparison arm; both reached 12 successful updates and matched exactly. The same warning appeared in earlier PD v2 pilots. Preserve and assess numerical skip counts in the fresh production-size pilots; a passing recovery comparison is not a claim of warning-free training. Expected `INTERRUPTED` rows and `[no-monitor]` epoch placeholders are not failed controls.
- Completed job-log runtimes support shorter fresh-part-1 requests: null **27–57 s**, short pilots **83–207 s**, recovery **63–153 s**. Use `NULL_WALLTIME=00:10:00 PILOT_WALLTIME=00:15:00 RECOVERY_WALLTIME=00:10:00` as submission overrides, without changing budgets, caps or source identity; queue waiting is additional.
- There is still **no full `part1_passed.json` or part-2 run**. The previously passed 16 null and 32 short controls used earlier code. Next: confirm no active CreditPFN writers, inspect cleanup targets, clear prior output/trained weights on both tiers as requested, then run fresh **part1** (16 null, 32 short, eight recovery pairs). Review that receipt before the **eight part2 budget pilots**, settle the training horizon, then release experiment 1. Keep the accepted clean experiment-0 evidence when starting experiment 1. No cleanup, submission, Downloads change or implementation change was performed in this inspection.
- Local verification: **599 passed, 1 skipped** (optional local manifests absent; 449.95 s). Additional read-only assertions match the current source fingerprint, all eight reports, 16 finite completed training records and eight deliberate interruption records. No notebook or visualization code changed; no notebook execution was needed.

## Previous handover — 25-09-2026 five recovery pairs pass; PD ensemble loss and review fixes

- User committed the previous repair as **d0c3929**. Read `Downloads/CreditPFN-20260925-111056/output CreditPFN/` in place: **694 files / 18,457,489 bytes**, comprising 103 logs, 191 manifest/workflow files, 372 training files, 26 result/prediction files and two general scheduler files. All **149 JSON and 412 CSV/compressed CSV files parse**. Both storage tiers are represented; no Downloads files were modified or copied into the repository.
- Isolated workflow **1338dabf6e9541d4b4c20813f2e7e065** is finished but **failed**, with five successful and three failed tasks. All four LGD bases and PD TabICL pass recovery, selected-arm audits and five-fold smoke scoring. Their actual model tensors and monitored values at updates 0/5/12 match exactly. LGD TabPFN's detached `criterion.losses_per_bucket` differs and remains separately reported; it is not learned-state drift. The three PD TabPFN reference arms fail at their first training loss with `nll_loss2d_forward_out_cuda_template` lacking a deterministic implementation.
- Experiment 0 is **not complete**: the earlier 16 null and 32 short pilots passed, but no accepted full-part-1 receipt or part-2 budget run exists in this snapshot. Read-only downloaded records are not a live scheduler query. Experiment 1 remains unreleased.
- Flatten ensemble member/query pairs before classification CE, preserving the objective/all output columns and loss gradients while using the non-spatial reduction. Strictness/tolerances, budgets, row caps and the research grid are unchanged. The earlier synthetic probe used the single-forward loss and missed this shape-specific refusal; it now follows the production ensemble forward/loss. Confirmed against PyTorch 2.12 `nll_loss2d_forward_out_cuda_template` (atomic mean/sum reduction).
- Independently reproduced both Claude findings: evaluation dropped saved trial identity from model handles, and unfinished frozen manifest rows were reconstructed with legacy adapter tags. Also reproduced an earlier evaluation failure from a function-local OmegaConf import. Carry the saved identity into roster membership checks, import OmegaConf at module scope, and reconstruct absent adaptation metadata from the fingerprinted phase configuration. Regression cases use the actual `RunRow` schema, both families/tracks and valid/foreign/missing checkpoint identities.
- Download also exposes **13 one-line orphan recovery summaries**: the first child renamed the shell-owned file, while later children reused its old path. Keep the active job filename stable; trial names remain logged. Completed historical logs are untouched.
- Next: user commits/pushes and pulls on VSC, runs the isolated recovery check with the corrected source, and supplies its result. Only after all eight pass: inspect/stop CreditPFN writers, preview and clear both output tiers plus trained weights, rerun full part 1 and then part 2, review the horizon, prepare experiment 1. Training source is **4de2b3ad4ba05cbb7cbe399779cdc0308d5aa0d0474b1a455230fd45e0a94488**; new training/evaluation fingerprints invalidate old plans, so never bypass or relabel them. Preserve original data/base weights and downloaded historical evidence. No agent training, installation, push, cleanup or VSC submission.
- Validation: **599 passed, 1 skipped** (optional local manifests absent) in **461.16 s**, plus **142 focused tests passed**. Coverage includes numerical loss/gradient equivalence, both reported regressions, actual ensemble probe wiring and shared child-log paths. All **11 notebooks passed**, using the download in place and producing **113 temporary PDFs**; original notebook bytes were restored. All **17 shell files** parse with LF endings; a read-only symbol-table scan finds no unresolved global candidates in src/scripts. A 120-second diagnostic timer printed a slow existing large-CSV test stack; the test and full suite subsequently passed. Downloads still contains exactly 694 files / 18,457,489 bytes; no local output tree or new debug files. Library remains read-only at **e5ce01614eebe520af303f2b5bfd212298eab2be**. Actual B200 validation of the corrected PD loss remains pending.

## Previous handover — 25-09-2026 GPU variation reproduced; deterministic recovery validation next

- User committed the preceding diagnostic as **d7dff7c**. Read `Downloads/output CreditPFN.zip` directly, without extracting: **240 files / 4,694,936 uncompressed bytes**, ZIP CRC validation passed. New diagnostic: **Mindwell 11615820**, 09:24:45–09:24:55, correct CreditPFN environment, B200/Torch 2.12.0+cu130, exit 0. The preceding extracted directory is absent; no byte-for-byte comparison against it is claimed. No Downloads files were changed or imported.
- Confirmed PD v2 GPU arithmetic variation: identical state/RNG/batch produced **110/129 different gradient tensors**, maximum absolute delta **0.000244140625**, in both default-kernel comparisons. Both deterministic profiles had **zero** loss/gradient differences across the three repeats. This identifies one real source of non-repeatability; it does not prove the cause of every previous pair mismatch or establish correct resumption. The small first-call timings are not throughput benchmarks. Previous full-workflow recovery is still failed; 16 null/32 short pilots and eight benchmark smoke checks remain historical passed evidence.
- Recovery configs now request strict deterministic kernels and a **2,048-row cap**, with `cpt_recovery_v5_check3` identities. Configure the CUDA workspace before device initialization; unsupported deterministic operations fail. Seed Python's RNG alongside NumPy/Torch, and log/store numerical execution settings. Normal pilots and research grids retain fast kernels and measured production caps; the lower cap cannot raise capacity. Scientific fingerprints cover both new settings. Model/trajectory tolerances are unchanged.
- Added **`bash scripts/slurm/run_experiment0.sh recovery`**: CPU preparation of uniquely named probe arms, eight GPU pairs (four concurrent, 30-minute requests), and a diagnostic-only `recovery_passed.json`. Benchmark files now live under their workflow ID to prevent retry collisions. It never reuses the failed ledger or releases part 2. User must commit/push and pull before submitting. Training source identity is **9df9c258be3cdb6979aad5f45bbd132b3d3e179884c72e7aba39f79e2c183a26**; prior plans are historical and must not be relabeled or bypassed.
- **User's new cleanup preference supersedes the earlier retention recommendation:** after debugging passes, clear the old output and trained weights on both tiers, then rerun **all of experiment 0**, part 1 followed by the eight long part-2 pilots. Stop CreditPFN writers and inspect the cleanup preview first; preserve original data/base weights and reusable processed/retention inputs. Keep the completed clean experiment-0 evidence when advancing to experiment 1. No cleanup has been performed or newly authorized to execute before the recovery result.
- Validation: full retry **567 passed, 1 skipped** (optional local manifests absent), **72 focused tests passed**, and **38 observability tests passed** after benchmark-output scoping. Recovery integration cases cover all three sampling modes, terminal divergence, Python/Torch randomness, state restoration, actual lowered row caps and saved execution provenance. Workflow tests verify isolated preparation and that its receipt cannot release budget pilots. A mocked Bash launch confirms inherited experiment-1 config is cleared and both commands keep experiment-0 routing; no job was submitted. Shell/CLI/diff checks pass. ZIP stays 714,860 bytes; no local output tree or notebook changes. No installs, real-model training or VSC submissions by the agent. Library pin unchanged: **e5ce01614eebe520af303f2b5bfd212298eab2be**.

## Previous handover — 25-09-2026 CPU inspection complete; isolate GPU repeatability

- User committed the preceding changes as **ccaa866**. Read the new `Downloads/output CreditPFN/` in place: **239 files / 4,691,286 bytes**; only one additional maintenance log. Job **62143430** ran on wICE in the correct environment, 08:54:19–08:54:37, **exit 0**. Inspection completed; it did not change workflow **720955dab94c457fa23e23d8eba06c50**, which remains failed at recovery. The **16 null + 32 short pilots** and eight packaged five-fold smoke checks remain passed; no long pilots or experiment 1 were launched by that workflow.
- All eight selected checkpoint identity/diagnostic audits passed, with no missing or extra tensor keys. Maximum learned-state differences: PD **6.79e-6 / 7.58e-6 / 9.00e-6 / 8.33e-6** and LGD **5.71e-6 / 5.24e-6 / 6.21e-6 / 7.27e-6**, for v2/v2.6/v3/TabICL. The three much larger LGD deltas (**0.0167 / 0.0365 / 0.0254**) are confirmed as `criterion.losses_per_bucket`; they are not weight drift.
- Every pair differs at update 5, before the pause. Seven pairs have identical update-zero monitor values; PD TabICL already differs in three isotonic-calibration score fields at update zero (primary scores match). The scalar maximum over all score fields mixes counts and metric units; do not describe it as an AUC or RMSE difference. This evidence isolates a repeatability problem but does not identify its numerical cause or prove correct resumption.
- Added **`maintenance.slurm probe-repeatability`**: one synthetic batch through the actual loader/loss, three identical-state/RNG forward/backward repeats under default, deterministic, and deterministic math-attention settings. Default target: **PD v2, 512 rows, 16 features, two members**, one B200 allocation capped at five minutes. No optimizer, model updates, checkpoint writes, plans, receipts or submissions from the diagnostic itself; only the usual maintenance log. Deterministic-kernel refusals are recorded as errors, never passes. A successful small-batch probe is not full recovery validation.
- Training code identity is still **f234507aef0bc738f9178085dc56878c6b6b5bd1b23a934bd327d0182f3a24a7**; no training/config changes or repeated completed trials. User must commit/push, pull on VSC and submit the one diagnostic. No local GPU verification, install, push or real training by the agent. A later recovery retry must deliberately verify/reuse passed stages, not erase claims or bypass source guards.
- User's project-storage inventory confirms **legacy `/lustre1/project/stg_00211/CreditPFN/output` is absent**; named output contains only **experiment0**, about **12 MB training diagnostics + 320 KB results** (rounded total 12 MB). DATA download contains about **1.88 MB logs / 2.81 MB manifests**, plus two tiny general scheduler files. Checkpoints are separate and not included in those totals. Experiment 1 is already empty; no cleanup is needed now. Preserve stage receipts/plans, parameter/resource/trajectory measurements and budget evidence. Consider obsolete failed-attempt weights/logs only after experiment 0 is accepted, with exact inventories and no active writers; do not run the whole-tree cleaner to start experiment 1.
- ZIP input can be read directly without extracting; user need not unzip future transfers. Downloads were not changed or imported. The library remains read-only at pin **e5ce01614eebe520af303f2b5bfd212298eab2be**; storage guidance checked against `VSC Documentation.txt`, symbols `Managing storage usage` and `KU Leuven storage`.
- Validation: full suite **554 passed, 1 skipped** (optional local manifests absent), **49 focused tests passed**, and **13 diagnostic tests passed** after the final report-wording refinement. Coverage includes RNG/buffer/weight restoration, detection of uncontrolled variation and nonfinite/missing gradients, restoration after kernel errors, CPU refusal, and both model families' batch axes/loss paths. Maintenance/submission syntax, CLI help and diff checks passed. Downloads still contains exactly 239 files / 4,691,286 bytes; no local output tree was created. No notebook changes; actual GPU repeatability remains unverified.

## Previous handover — 24-09-2026 controls/pilots passed; recovery comparison failed

- Read `Downloads/output CreditPFN/` in place: **238 files / 4,662,669 bytes**. Workflow **720955dab94c457fa23e23d8eba06c50** is stopped at `recovery`, status `failed`, eight failed pairs. CPU prepare **62141115**, null audit **62141138** and pilot audit **62141239** passed. All **16 null controls** (two updates) and **32 short pilots** (250 updates) completed; both track audits have zero pending/divergent/problem entries. GPU sampling now works in training: 24 null and 211 pilot samples. Preserve these successful trials.
- All eight recovery jobs reached 12 updates and passed the packaged **five-fold benchmark smoke** (569 PD / 442 LGD predictions per model), but every state/trajectory equivalence comparison failed. This is not the full held-out credit benchmark. No part-1 receipt, long budget pilot or main sweep was released.
- Crucial distinction: the independent arms already differ at update **5 before interruption**, with matching row counts. PD v2's held-out AUC is 0.762554 versus 0.762875; its first rounded gradient norms also differ. A final mismatch alone therefore does not isolate a checkpoint-resume defect. CUDA/backward nondeterminism or uncontrolled randomness are hypotheses, not confirmed causes; the current loop does not request deterministic kernels. See [PyTorch 2.12 reproducibility guidance](https://docs.pytorch.org/docs/2.12/notes/randomness.html).
- Downloaded maximum state differences: approximately 6.8e-6–9.0e-6 for PD and 7.3e-6 for LGD TabICL; LGD TabPFN reports 0.017–0.037 across **all state tensors**, including a changed `criterion.losses_per_bucket` diagnostic. These maxima cannot be described as learned-weight drift without identifying their tensors. The read-only library snapshot confirms that `FullSupportBarDistribution.forward` updates this buffer separately from its returned loss (`tfm-library/repositories/TabPFN .txt`, pin `e5ce01614eebe520af303f2b5bfd212298eab2be`). Actual weights and detailed trajectories remain on project storage, outside this DATA download.
- Added **CPU-only `maintenance.slurm inspect-recovery --id <workflow>`** to compare the existing pairs: separate diagnostic-buffer deltas, name the largest tensor differences and compare monitors at 0/5/12. It reads existing data, prints a bounded normal experiment-0 log, and never trains, writes a receipt or alters the workflow. Existing model/trajectory tolerances remain intact; selected-arm identity errors now also fail the GPU check. The detached `criterion.losses_per_bucket` lives outside the saved model recovery state and resets on resume; report its difference without treating it as an inference/training parameter, consistently with null audits. Whole-saved-state equality remains explicitly reported. Actual model/trajectory mismatches still fail. Do not rerun part 1 or launch part 2 until the recovery issue is resolved.
- Training code identity remains **f234507aef0bc738f9178085dc56878c6b6b5bd1b23a934bd327d0182f3a24a7**; no training/config/weight changes. The workflow/evaluation fingerprint changes with the diagnostic utility, so a later recovery retry must explicitly preserve and verify the successful earlier stages; do not bypass the old ledger's source guard. User must commit/push and pull before the CPU inspection. No agent installs, pushes, VSC submissions or real training.
- Removed an untracked reappearance of obsolete `tests/test_output_migration.py` only after its Git blob exactly matched the deleted predecessor (`d5a5378...`, before `6777eb1`); current coverage already lives in `tests/test_consolidate_output.py`. `Downloads/output/` is a separate **CreditICL** download, not a CreditPFN output-naming regression; do not mix or remove it. No Downloads files were moved or deleted; no local output tree was created.
- Validation: full suite **541 passed, 1 skipped** (optional manifests absent), followed by **40 focused tests passed** after the final diagnostic-buffer comparison refinement. The maintenance wrapper and proposed submission command parse; CLI help and unchanged training fingerprint checked. Tests cover missing/different state keys, dtype/shape/nonfinite values, criterion borders, pre/post-pause mismatches, incomplete monitor records, selected identity failures, bounded logs and inspection without training/callbacks/writes. Notebooks were untouched. Local CPU tests do not establish GPU recovery equivalence.

## Previous handover — 24-09-2026 B200 counter check passed; corrected part 1 ready

- **B200 check passed (20:22):** Read `Downloads/maintenance_11614332_r0.log` in place. Mindwell job **11614332**, node `r11g22`, used the CreditPFN environment, returned `gpu_status: sampled`, empty GPU error fields and `END exit_code=0`. Counters: 0% idle utilization, 4 MiB device memory, 196.41 W, 32 C. This verifies the corrected query on the cluster; no training occurred. Next: `bash scripts/slurm/run_experiment0.sh part1` from the activated CreditPFN environment. No new code, pull or repeat standalone check is needed; only handover documentation changed locally.
- **Submission correction (20:19):** Mindwell rejected the standalone resource check's `--gpus=1` option before creating a job. Use **`--gpus-per-node=1`**, as the existing training wrappers do. No GPU allocation occurred; no cleanup, code change, new identity or pull is needed to retry the corrected command. Shell syntax validation does not validate the site's Slurm submission policy.
- Read `Downloads/output CreditPFN/` in place: **56 files / 1,125,299 bytes**, from user commit `c74610c`. Workflow `0865da430fbe433ca16ad6118017bdb7`: CPU prepare **62140639** passed with zero preflight failures/warnings; all **16 null GPU jobs exited 0**, each with two successful updates. CPU audit **62140691** then exited 1 and left phase `null`, status `failed`. No short pilots, recovery checks, budget pilots or main sweep were released.
- Both downloaded null audits report **8 complete, 0 pending, 0 divergent**; all 16 canonical saved states match exactly and all 16 credit/non-credit monitor comparisons pass. The only 16 audit problems are missing successful GPU resource samples: **25 `CalledProcessError` samples** in total. Detailed CSVs remain on project storage and were not part of this DATA download. Output/weight paths in the logs use the intended experiment folders and storage tiers.
- Confirmed a device-selector bug: PyTorch 2.12's `CUuuid.__str__` uses `uuid_to_string`, which emits a bare UUID, but the sampler passed it directly to NVIDIA's tool. A local read-only NVIDIA query reproduced exit 6 (`No devices were found`) for the bare UUID and exit 0 with `GPU-`. Primary evidence: PyTorch `registerCudaDeviceProperties` and [`uuid_to_string`](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/csrc/utils.cpp). The standalone query now passes on B200; collection during training remains part of the fresh null-control audit.
- Normalize the GPU prefix, preserve explicit prefixed selectors, record bounded subprocess details/exit codes and warn once per trial. A sample requires actual utilization and memory-use values; unavailable counters cannot pass. `preflight --gpu-resources` checks the same sampler without loading datasets/models or writing measurement files. Its Slurm log remains under experiment 0.
- Experiment-0 configs now use **`cpt_*_v5_check2`** so corrected source cannot overwrite immutable v5 plans/checkpoints. Only those run names changed; grids, budgets and experiments 1–3 are unchanged. The standalone GPU check passed; rerun part 1 to verify the corrected measurement path with fresh records. Keep the existing successful null evidence; do not bypass its failed resource gate or relabel old weights. No Downloads copies, output deletions, installs, commits, pushes or VSC submissions by the agent.
- Validation: reproduced the old selector failure in a regression test, then **16 resource tests passed**. Full suite **525 passed, 1 skipped** (local manifests absent). The standalone entry point returned real local NVIDIA counters using a mocked PyTorch device identifier because local Torch is CPU-only; this is not B200 validation or model training. All **17 shell/Slurm files** and the proposed standalone submission command parse. All eight config payloads differ only in `run_name`; no local output tree was generated and notebooks were untouched.

## Previous handover — 24-09-2026 named output and CPU preflight repair

- The user reconfirmed **`output CreditPFN/<experiment>/`** on DATA, project storage and locally. This supersedes the bare `output/` name in the preceding handover. Python resolvers, shell logs, configuration and current documentation use the name with its space; relative legacy aliases cannot create a second output tree. `docs/TEMPLATE.md` remains the generic template, and AGENTS/README/VSC document the deliberate naming difference. Checkpoints retain their existing experiment directories.
- Read `Downloads/output/` in place: **8 files / 6,384 bytes**. Workflow `fedcc296871349599a8a6015f0ae9065`, wICE preparation job **62134902**, failed with `NameError: OmegaConf` in the newly added retention preflight check; ledger phase `prepare`, status `failed`, only the CPU job recorded. No GPU stage or plan preparation was reached. The ten public inputs were prepared according to the user's login-node output; the rerun reuses verified downloads.
- Reproduced the exact error in two CLI regression cases, then consolidated the dependency import at module scope. Fresh part-1 submission after the user commits/pushes and pulls on VSC creates plans and a new workflow under the correct directory. The failed workflow is not resumed or migrated; no successful v5 weights from this attempt need preserving. Experiment 4 remains deferred, and experiment 1's scientific grid is unchanged.
- Local output directories were already absent at the start of this repair. Downloads and historical archives remain untouched. Verification writes to temporary storage; no repository output clutter, VSC submissions, installs or pushes by the agent.
- Validation: **509 passed, 1 skipped** (optional absent manifests); **93 focused checks passed** including shell logging with spaces and legacy aliases on both storage tiers. All **17 shell/Slurm files parse**. All **11 notebooks passed**, producing **35 PDFs** under temporary `output CreditPFN/`; original notebook bytes were restored, and no bare `output/` tree was created. The real local preflight completes all 14 configurations and verifies both retention panels; its only failures are the two v2 original weights absent locally, repeated across the relevant configs. VSC must verify the actual project-storage weights before GPU release.

## Previous handover — 24-09-2026 fresh protocol 5 and bundled experiment 0

- User explicitly requested a fresh rerun and `output/<experiment>/{logs,manifests,...}` on both tiers. This supersedes the earlier instruction to reuse v4 preparation/null controls. Every current phase is `cpt_*_v5`; do not rename/reuse old v4 weights as current controls. No v5 VSC jobs have been submitted by this agent.
- `bash scripts/slurm/run_experiment0.sh part1` downloads verified public inputs on the login node, then stages/prepares on CPU and releases 16 null controls, 32 short pilots and eight uninterrupted/resumed GPU pairs with CPU audit gates. Each recovery pair also checks five-fold final scoring/prediction output. `part2` separately submits eight 20k-update pilots after a current part-1 receipt; two-hour resumable work segments. Research counts remain 512 + 32 + 96 = 640; experiment 0 adds 72 training arms, not counting resumed segments twice.
- Fixed research retention panel: four classification and four regression OpenML versions from TabArena-v0.1; two packaged datasets for null/recovery. All ten inputs downloaded and checksum-verified locally under gitignored `data/retention/`. They are excluded from the credit registry and staged explicitly on VSC. This is our declared subset/protocol, not the full official benchmark; base-pretraining contamination is unverified. Library pin remains `e5ce01614eebe520af303f2b5bfd212298eab2be`.
- Project storage now holds per-trial epochs, trajectories, compressed parameter summaries and periodic resources under each experiment, plus final metrics/predictions and narrow consolidated tables. DATA holds job logs/small manifests/cluster locks. Weights use `checkpoints/trained/<experiment>/`. Downloads were not moved. Local notebook verification artifacts go to a temporary directory; notebooks are cleared afterwards for the fresh start.
- Monitoring now separates context/validation/query; query labels cannot select F1 thresholds or calibrators. Fixed-milestone records retain complementary metrics. Final scoring retains five outer folds, validation-only HPO/thresholds/calibration and original row indices. Quantiles use the explicit native APIs (`TabPFNRegressor.predict` full output; `TabICLRegressor.predict` with `alphas`), sharing one regression forward.
- Local deletion of `output CreditPFN/` was rejected by automatic approval review. Its 51 files / 742,235 bytes remain pending manual user removal; no alternate deletion route was attempted. The new active `output/` is absent until its first actual run; verified notebook cell outputs were cleared for the fresh start. The obsolete `src.utils.migrate_output` was removed; old output is unnecessary for a fresh v5 run.
- Validation: full suite **504 passed, 1 skipped** (optional absent manifests); **167 focused checks passed** after the quantile/monitor/recovery changes; **41 audit/launch/consolidation checks passed** after the final resource gate. **295 entry-point/data/training/report checks passed, 1 optional skip**, after moving experiment selection ahead of the first log; **24 consolidation/observability checks passed** after the test-file rename. All 17 shell/Slurm files parse and Python compilation/diff checks passed. All 11 notebooks executed successfully to temporary output; one previously observed Windows ZeroMQ shutdown assertion appeared after successful cell execution. Local preflight cannot certify the two absent v2 base files; VSC part-1 preflight checks the actual eight bases before any GPU submission. No real-model training or package installation was performed locally.

## Previous handover — 24-09-2026 experiment folders and analysis notebooks

- Read Downloads in place: **93 files / 1,306,137 bytes**. Migration **62124946** moved 87 files on both tiers and exited 0; preflight **62124955** reports zero failures/warnings and exit 0. Corrected null audits **62124956 (PD)** and **62124957 (LGD)** each report eight complete, zero pending/diverged/problems, eight exact canonical saved-state matches, eight monitor matches and `passed: true`, exit 0. The 16 check3 controls are now audited; do not spend GPUs repeating them merely because configs moved. Two-update cost extrapolations are not production walltime evidence.
- User requested experiment 0 = debugging/pilots; 1 = main sweep; 2 = seed sensitivity; 3 = sampling/accumulation. Moved all 12 phase configs into those four directories and verified their parsed payloads are identical to the preceding HEAD. Shared data/train/eval defaults remain at the config root. Existing run names remain intact. Main/seed/sampling counts remain **512 / 32 additional / 96**; experiment 0 retains 16 null + 32 short + 8 budget pilots.
- Replaced six flat notebooks with **11 ordered notebooks** under `00_general/` and experiments 0–3. New reports cover corpus geometry/quality/partitions/exposure, operational gates, update-indexed trajectories, matched factor/seed/sampling effects, complete-fold benchmark comparisons, complementary metrics and cost. Figures use bounded pages, shared A4 style, PDF-only FigureSaver and final section-ordered text summaries. No result PDFs are fabricated for unavailable measurements. Runner discovery and figure/caption paths support nested notebook names.
- Added optional **CREDITPFN_ANALYSIS_ROOT**, naming the downloaded output directory itself, for read-only analysis without import. Only visualization readers use it; generated figures/summaries still go to the repository output. Normal cluster storage variables and checkpoint locations are unchanged.
- Validation: full suite **479 passed, 1 skipped** (optional on-disk manifests), nine known constant-input toy-regression warnings; **102 focused checks passed** after the final analysis corrections. All **11 notebooks passed**, producing **39 PDFs** from local corpus data and downloaded null-control evidence; unstarted phases retain planned coverage without fabricated result figures. The publication scan covered 61 artifacts with zero private-name matches. Actual notebook figures and synthetic full-grid heatmaps/trajectories were visually inspected at A4 width. All nine experiment notebooks passed a final serial rerun. Downloads' 93-file content digest is unchanged. No install, real model training, cluster submission, push or Downloads mutation by the agent.
- **Update — pilot preparation complete:** downloaded CPU jobs **62129268 (PD)** and **62129269 (LGD)** each wrote 16-trial `cpt_pilot_v4` plans and exited 0. Both downloaded plan checksums, all 32 trial checksums and training-source hashes match the current checkout `fbf7041`. The download contains 97 files / 1,575,241 bytes: 33 logs, 62 manifest/history/plan files and two summary files. Read in place; no files moved or removed. Pilot training has not yet been evidenced.
- **Next operational phase:** launch the existing 32 short, positive-LR experiment-0 pilots (250 updates each), using the prepared plans. Do not repeat preparation or null controls. A one-hour request per pilot is the current provisional launcher default. CUDA recovery verification and measured budget/walltime decisions still precede the main sweep. Output grouping by experiment is a proposed cleanup, not implemented in this runtime review; changing hashed routing before launch would invalidate the newly prepared plans.

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
| 15-10-2026 | Sampling benchmark · Mindwell 11687925 / wICE 62312109 | **255/256 done** | All 128 GPU and 127 CPU tasks exit 0, 5,405 folds OK; CPU 62312109_124 held before startup. Release recommendation pending user action. |
| 15-10-2026 | Context curves · context_v1 | **250/250 done** | All receipts match plan/logs, 11,100 paired-state cells, constant test sizes within tasks, all exit 0. PROJECT artifacts not downloaded. |
| 15-10-2026 | DiagScale · cpt_diagscale_v1 | **30/48 done** | 21 PD + 9 LGD at 10k updates; three PD tasks running, LGD pending on priority/capacity. No recorded numerical skips or errors. |
| 15-10-2026 | Corpus policy · cpt_corpus_v1 | **6/16 done** | All completed trials at 10k updates; remaining jobs pending on capacity dependencies. |
| 15-10-2026 | Episode size · cpt_episode_rows2048_v1 / rows26000 | **5/16 done** | Four small + one large complete at 10k; one large trial resumed update 7,145 and is running; ten other trials pending. |
| 14-10-2026 | DiagScale PD · cpt_diagscale_v1 | **submitted** | All 24 trials; arrays 11688015/16/17/18/19/24/25/26, four partitions, 90-minute recovery segments. No execution outcome supplied yet. |
| 14-10-2026 | DiagScale LGD · cpt_diagscale_v1 | **submitted** | All 24 trials; arrays 11688027–11688034, four partitions, prepared-plan check passed. |
| 14-10-2026 | Corpus policy PD · cpt_corpus_v1 | **submitted** | All 16 trials; arrays 11688035/36/38/39/40/41/42/43; prepared-plan check passed. |
| 14-10-2026 | Episode size PD · cpt_episode_rows2048_v1 / rows26000 | **partly submitted** | Eight 2,048-row trials accepted as 11688044–11688051; all eight 26,000-row trials still unsent at the 11:08 snapshot, waiting at 450 tasks. |
| 14-10-2026 | Follow-up capacity PD/TabPFN v3 · Mindwell 11687975 (task 0) | **pass** | Five 16-update training probes and six context sizes through 50k; exit 0, 448 s. |
| 14-10-2026 | Follow-up capacity PD/TabICLv2 · Mindwell 11687976 (task 1) | **pass** | Five 16-update training probes and six context sizes through 50k; exit 0, 272 s. |
| 14-10-2026 | Follow-up capacity LGD/TabPFN v3 · Mindwell 11687977 (task 2) | **pass** | One 16-update adapter probe and six context sizes through 50k; exit 0, 181 s. |
| 14-10-2026 | Follow-up capacity LGD/TabICLv2 · Mindwell 11687974 (task 3) | **pass** | One 16-update adapter probe and six context sizes through 50k; exit 0, 90 s. |
| 14-10-2026 | Follow-up gate · b3197869c5974faf83713e3c9072be2c | **pass** | All eight receipts accepted by cluster inspection at 10:51; retain receipts and PROJECT evidence for experiments 4–7. |
| 14-10-2026 | Sampling benchmark · Mindwell 11687925 / wICE 62312109 | **running** | 28 classical tasks/185 folds OK in 10:52 DATA ZIP; 10:58 queue has four running per controller, remaining jobs array-throttled, no holds. |
| 14-10-2026 | Follow-up prepare · wICE 62312115 | **done** | Workflow b3197869c5974faf83713e3c9072be2c created; four smoke tasks submitted; exit 0, 71 s. |
| 14-10-2026 | Follow-up smoke PD/TabPFN v3 · Mindwell 11687933 (task 0) | **pass** | Three learning-rate recovery pairs, six final six-update checkpoints, matching gate receipt; exit 0, 307 s. |
| 14-10-2026 | Follow-up smoke PD/TabICLv2 · Mindwell 11687934 (task 1) | **pass** | Three learning-rate recovery pairs, six final six-update checkpoints, matching gate receipt; exit 0, 229 s. |
| 14-10-2026 | Follow-up smoke LGD/TabPFN v3 · Mindwell 11687945 (task 2) | **pass** | Three learning-rate recovery pairs, six final six-update checkpoints, matching gate receipt; exit 0, 223 s. |
| 14-10-2026 | Follow-up smoke LGD/TabICLv2 · Mindwell 11687932 (task 3) | **pass** | Three learning-rate recovery pairs, six final six-update checkpoints, matching gate receipt; exit 0, 218 s. |
| 14-10-2026 | Sampling checkpoint audits · wICE 62311793/62311794 | **96/96 pass** | PD/LGD each 48 completed at 10k updates, no divergence/pending/problems; both logs exit 0. Ready for experiment-3 benchmarking on the existing cluster source. |
| 14-10-2026 | Main/seed/sampling DATA snapshot, downloaded 08:34; queue 08:43 | **640/640 training OK; main/seed benchmarks complete in logs** | Main 512, seed 32, sampling 96 at 10k updates; main/seed each 256 successful benchmark shards and full logged five-fold coverage; both queues empty. Sampling checkpoint audit/benchmark and fresh project-artifact download remain. |
| 13-10-2026 | Main/seed/sampling DATA snapshot, downloaded 14:13 | **512/512 main; 32/32 seed; 90/96 sampling** | 46 PD/44 LGD sampling completed; all main/seed classical shards logged successful; GPU partitions 0/3 and project results still unconfirmed; no new runs submitted. |
| 12-10-2026 | Merged main/seed/sampling output + 14:47 queue | **512/512 main; 32/32 seed; 64/96 sampling** | Partial five-fold benchmarks; six train/four GPU eval/eleven CPU eval tasks running, no retrieval holds; no new runs submitted. |
| 12-10-2026 | Sampling hold recovery · six existing Mindwell roots | **6/6 RUNNING at 11:13** | User released the exact held tasks; preserve downstream jobs and check subsequent outcomes. No replacement runs submitted. |
| 12-10-2026 | Seed checkpoint audits · wICE 62295163/62295164 | **32/32 pass** | PD/LGD each 16 completed 10k-update checkpoints; zero divergence/pending/problems; both job logs exit 0. Ready for the experiment-2 benchmark. |
| 12-10-2026 | Main/seed/sampling campaign · Downloads + live queue audit | **512/512 main; 32/32 seed; 63/96 sampling complete** | Six held sampling tasks block 27 training and 96 evaluation tasks; main CPU benchmark logs passed, GPU partition 2 passed; no new cluster runs submitted. |
| 01-10-2026 | Seed study · cpt_seeds_v5 · PD/LGD four folds | **32/32 training complete** | Every trial reaches 10k; seven PD interruptions recovered; 33.65 recorded GPU-hours; project CPU audits still required. |
| 01-10-2026 | Sampling fold-zero segments · Mindwell 11643936–11643959 | **24/24 recovery saves** | Intended exit 75 after 123–5,145 updates; zero recorded skips/OOM; resume these and release the other 72 trials. |
| 01-10-2026 | Initial PD benchmark · Mindwell 11643871_0 · scheduler confirmation | **TIMEOUT** | Two-hour limit; 13/19 comparisons (65 folds) saved; six remain, only one large table. Resume task zero with three hours. |
| 30-09-2026 | Sampling LGD preparation · wICE 62202372 | **passed** | 48 identities, four folds, original data/retention/base/environment fingerprints retained. |
| 30-09-2026 | Sampling PD preparation · wICE 62202371 | **passed** | 48 identities; proportional PD control and both exhaustive modes; original input fingerprints retained. |
| 30-09-2026 | Seed LGD preparation · wICE 62202370 | **passed** | 16 seed-43 identities; all other scientific reference settings and input fingerprints retained. |
| 30-09-2026 | Seed PD preparation · wICE 62202369 | **passed** | 16 seed-43 identities; all other scientific reference settings and input fingerprints retained. |
| 30-09-2026 | Initial GPU LGD benchmark shard · Mindwell 11643872 | **passed** | 25 model/table pairs, 125/125 folds, zero failures, 637.2 pipeline seconds; full benchmark still pending. |
| 30-09-2026 | Initial CPU PD benchmark shard · wICE 62202344 | **passed** | XGBoost/Bondora, 5/5 folds, zero failures, 228.4 pipeline seconds. |
| 30-09-2026 | Initial CPU LGD benchmark shard · wICE 62202345 | **passed** | Linear regression/non-credit table, 5/5 folds, zero failures, 10.6 pipeline seconds. |
| 30-09-2026 | Initial GPU PD benchmark shard · Mindwell 11643871 · 20:30 snapshot | **incomplete snapshot** | 60 folds passed, then processing large Bondora table; no END or error, live state unverified. |
| 30-09-2026 | Initial main benchmark tasks · GPU 11643871/11643872, CPU 62202344/62202345 | **started** | Logs begin at 19:57; no final outcomes in this snapshot. Continue existing jobs. |
| 30-09-2026 | Auxiliary final audit · wICE 62202341 | **passed** | Eight matching reports, no failures; accepted receipt matches current source/workflow. No experiment-0 rerun needed. |
| 30-09-2026 | Auxiliary GPU controls · Mindwell array 11643860 | **8/8 passed** | 52 six-update arms, 16 successful recovery comparisons, eight seed checks and 40 smoke folds; 0.367 GPU-hours by log intervals. Production accumulation cost remains unmeasured. |
| 30-09-2026 | Auxiliary controls · Mindwell array 11643860 · 19:49 snapshot | **running** | Four PD tasks started normally; completed seed arms, no recorded errors, no final receipt yet. Existing workflow continues; no resubmission. |
| 30-09-2026 | Auxiliary preparation · wICE 62202340 | **done** | CPU preflight 0 failures / 0 warnings; exit 0, eight-task GPU array submitted. |
| 30-09-2026 | Main LGD checkpoint audit · wICE 62202339 | **passed** | All 256 expected project checkpoints present with matching provenance and 10k updates; zero pending/diverged/problems; log exit 0. |
| 30-09-2026 | Main PD checkpoint audit · wICE 62202338 | **passed** | All 256 expected project checkpoints present with matching provenance and 10k updates; zero pending/diverged/problems; log exit 0. |
| 30-09-2026 | Completed main training · cpt_main_v5 · merged DATA/project download | **512/512 complete** | 256 PD + 256 LGD at 10k; all six milestones/diagnostic files present, 509.06 recorded GPU-hours, zero skips or nonfinite primary scores; 68 interrupted PD attempts recovered. Seven entropy-only missing measurements repaired for future scoring. Final benchmark and auxiliary controls still pending. |
| 30-09-2026 | Main training progress · cpt_main_v5 · 09:56 snapshot | **running** | 408/512 complete at 10k (256 PD, 152 LGD); 16 LGD advancing, 88 unstarted; all 68 saved interruptions recovered. No observed training/primary-score failure; confirmed entropy-only NaN diagnostic and separate upstream power warnings, with project diagnostics still absent. |
| 29-09-2026 | Main training progress · cpt_main_v5 · 08:58 snapshot, queue follow-up 09:14 | **running** | All 512 submitted; 116 PD complete at 10k, 16 active logs, 24 completed recoveries, no observed training failures/skips. All 15 saved trials confirmed pending at normal four-task array limits. LGD not started in the archive. |
| 29-09-2026 | Final LGD fold submission · Mindwell 11624651 / 11626375 / 11626507 / 11626875 | **submitted** | Detached submitter accepted the remaining 64 trials and exited normally; all 256 LGD trials are now submitted. |
| 28-09-2026 | Main LGD training submission · Mindwell 11624565–11624576 · config/experiment1/lgd.yaml | **partly submitted** | Folds 0–2: 192/256 trials accepted; fold 3 waits at 448 submitted tasks against 450-task headroom. No training outcome observed yet. |
| 28-09-2026 | Main PD training submission · Mindwell 11624537, 11624542–11624545, 11624554–11624564 · config/experiment1/pd.yaml | **submitted** | All 256 trials / 16 arrays accepted, launcher complete; 90-minute work segments, 100-minute allocations, automatic requeue. No training outcome observed yet. |
| 28-09-2026 | Main LGD preparation · wICE 62175508 · config/experiment1/lgd.yaml | **done** | 256 immutable trial identities across four folds; scheduler 0:0, plan/source/input checks and submission preview passed. |
| 28-09-2026 | Main PD preparation · wICE 62175489 · config/experiment1/pd.yaml | **done** | 256 immutable trial identities across four folds; scheduler 0:0, plan/source/input checks and submission preview passed. |
| 28-09-2026 | Main CPU preflight · wICE 62175400 · both experiment-1 configs | **done** | 0 failures / 0 warnings; scheduler 0:0, 512 planned cells and eight verified non-credit monitoring tables. |
| 28-09-2026 | Corrected-loss production pilots · wICE prepare 62174499 / audit 62174759 | **done** | 12/12 PD + 8/8 LGD at 250 updates, no pending/divergent trials or audit problems; 0.680 GPU-hours of trial work, 6 min 49 s workflow span. |
| 28-09-2026 | Corrected-loss recovery · Mindwell array 11623721; wICE prepare 62174259 / audit 62174270 | **done** | 8/8 pairs, 16 completed 12-update arms, exact model/trajectory parity and 40 benchmark folds passed; 0.217 GPU-hours, 5 min 28 s workflow span. |
| 28-09-2026 | Budget audit · wICE 62165438 | **done** | Released after the environment-retrieval hold; restart 1 passed, scheduler 0:0 in 1 min 36 s. Eight 20k-update pilots accepted; part2 receipt published. |
| 26-09-2026 | Budget PD TabICL · Mindwell 11619744 | **done** | 20k updates, zero skips; resumed once from 11,391; 208.5 training min. |
| 26-09-2026 | Budget PD v3 · Mindwell 11619747 | **done** | 20k updates, zero skips; resumed once from 12,884; 184.7 training min. |
| 26-09-2026 | Budget PD v2 · Mindwell 11619746 | **done** | 20k updates, zero skips; resumed once from 13,667; 174.4 training min. |
| 26-09-2026 | Budget PD v2.6 · Mindwell 11619745 | **done** | 20k updates, zero skips; resumed once from 15,840; 150.4 training min. |
| 26-09-2026 | Budget LGD v2.6 · Mindwell 11619749 | **done** | 20k updates, zero skips; one allocation, 103.9 training min. |
| 26-09-2026 | Budget LGD v3 · Mindwell 11619751 | **done** | 20k updates, zero skips; one allocation, 80.4 training min. |
| 26-09-2026 | Budget LGD TabICL · Mindwell 11619748 | **done** | 20k updates, zero skips; one allocation, 77.0 training min. |
| 26-09-2026 | Budget LGD v2 · Mindwell 11619750 | **done** | 20k updates, zero skips; one allocation, 74.2 training min. |
| 26-09-2026 | Part2 preparation · wICE 62164623 | **done** | Matching part1 plans checked; budget plans/staging ready; 32 s, exit 0. |
| 26-09-2026 | Fresh part 1 final audit · wICE 62164601 | **passed** | Current-fingerprint receipt d8e878653d3e46dc9201772b88e50275; 8 recovery pairs accepted; 8 s, exit 0. Part 2 released for user launch. |
| 26-09-2026 | Fresh recovery array · Mindwell 11619710–11619717 | **passed** | 8/8 pairs; zero model drift, matching 0/5/12 monitors, 40 scoring folds / 4,044 predictions; 54–141 s/job. |
| 26-09-2026 | Fresh pilot phase / audit · wICE 62164600 | **passed** | 32/32 at 250 updates; zero data/optimizer skips or divergence; GPU jobs 81–202 s, audit 11 s. |
| 26-09-2026 | Fresh null phase / audit · wICE 62164597 | **passed** | 16/16 at two zero-LR updates; exact state/monitor parity, zero skips; GPU jobs 28–111 s, audit 26 s. |
| 26-09-2026 | Fresh part 1 preparation · wICE 62164595 | **done** | CPU checks, eight immutable plans and staging completed; 35 s, exit 0; released null controls. |
| 26-09-2026 | Full corrected data preparation · wICE 62164553 | **done** | 25/25 processed CSVs; 53 numerical unit conversions across two PD tables; 29 min 30 s, exit 0. No new GPU run in download. |
| 25-09-2026 | PD v2/table 0011 six-setting probe · Mindwell 11618877 | **diagnosed** | 6/6 settings completed; only upstream float32 clipping had finite losses, while adding 1,888 NaNs/member. Zero updates/writes; 18 s, exit 0. |
| 25-09-2026 | PD v2/table 0011 precision/clipping probe · Mindwell 11618868 | **failed** | Uncaught encoder ValueError in first setting; comparisons incomplete. Zero updates/writes; 16 s, exit 1. |
| 25-09-2026 | Part 1 recovery audit · wICE 62154541 | **passed** | 8/8 pairs and five-fold smoke passed; wrote part1_passed.json. 19 s, exit 0. |
| 25-09-2026 | Recovery LGD TabICL full · Mindwell 11618827 | **passed** | Exact model/0-5-12 monitors; five-fold smoke, 442 predictions. 69 s, exit 0. |
| 25-09-2026 | Recovery LGD v3 full · Mindwell 11618834 | **passed** | Exact model/0-5-12 monitors; five-fold smoke, 442 predictions. 85 s, exit 0. |
| 25-09-2026 | Recovery LGD v2.6 full · Mindwell 11618833 | **passed** | Exact model/0-5-12 monitors; five-fold smoke, 442 predictions. 66 s, exit 0. |
| 25-09-2026 | Recovery LGD v2 full · Mindwell 11618832 | **passed** | Exact model/0-5-12 monitors; five-fold smoke, 442 predictions. 55 s, exit 0. |
| 25-09-2026 | Recovery PD v2 full · Mindwell 11618828 | **passed** | Exact model/0-5-12 monitors; five-fold smoke, 569 predictions. One nonfinite batch skipped per arm. 112 s, exit 0. |
| 25-09-2026 | Recovery PD v2.6 full · Mindwell 11618829 | **passed** | Exact model/0-5-12 monitors; five-fold smoke, 569 predictions. 112 s, exit 0. |
| 25-09-2026 | Recovery PD TabICL full · Mindwell 11618831 | **passed** | Exact model/0-5-12 monitors; five-fold smoke, 569 predictions. 105 s, exit 0. |
| 25-09-2026 | Recovery PD v3 full · Mindwell 11618830 | **passed** | Exact model/0-5-12 monitors; five-fold smoke, 569 predictions. 142 s, exit 0. |
| 25-09-2026 | Part 1 pilot audit · wICE 62154202 | **passed** | 32/32 reached 250 updates; recorded trajectories/parameters/resources; skips require review. 13 s, exit 0. |
| 25-09-2026 | Pilot LGD v3 full · Mindwell 11618826 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 98 s, exit 0. |
| 25-09-2026 | Pilot LGD v3 frozen · Mindwell 11618825 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 89 s, exit 0. |
| 25-09-2026 | Pilot LGD v3 full · Mindwell 11618824 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 101 s, exit 0. |
| 25-09-2026 | Pilot LGD v3 frozen · Mindwell 11618759 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 90 s, exit 0. |
| 25-09-2026 | Pilot LGD v2 frozen · Mindwell 11618758 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 82 s, exit 0. |
| 25-09-2026 | Pilot LGD v2 full · Mindwell 11618822 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 89 s, exit 0. |
| 25-09-2026 | Pilot LGD TabICL frozen · Mindwell 11618755 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 82 s, exit 0. |
| 25-09-2026 | Pilot LGD TabICL full · Mindwell 11618821 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 88 s, exit 0. |
| 25-09-2026 | Pilot LGD TabICL frozen · Mindwell 11618820 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 83 s, exit 0. |
| 25-09-2026 | Pilot LGD TabICL full · Mindwell 11618803 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 90 s, exit 0. |
| 25-09-2026 | Pilot LGD v2 frozen · Mindwell 11618802 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 82 s, exit 0. |
| 25-09-2026 | Pilot LGD v2.6 frozen · Mindwell 11618757 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 101 s, exit 0. |
| 25-09-2026 | Pilot LGD v2 full · Mindwell 11618801 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 85 s, exit 0. |
| 25-09-2026 | Pilot LGD v2.6 full · Mindwell 11618800 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 112 s, exit 0. |
| 25-09-2026 | Pilot LGD v2.6 frozen · Mindwell 11618799 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 99 s, exit 0. |
| 25-09-2026 | Pilot LGD v2.6 full · Mindwell 11618798 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 111 s, exit 0. |
| 25-09-2026 | Pilot PD v3 frozen · Mindwell 11618751 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 187 s, exit 0. |
| 25-09-2026 | Pilot PD v3 full · Mindwell 11618797 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 201 s, exit 0. |
| 25-09-2026 | Pilot PD v3 frozen · Mindwell 11618794 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 192 s, exit 0. |
| 25-09-2026 | Pilot PD v3 full · Mindwell 11618781 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 200 s, exit 0. |
| 25-09-2026 | Pilot PD TabICL frozen · Mindwell 11618742 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 188 s, exit 0. |
| 25-09-2026 | Pilot PD v2 full · Mindwell 11618754 | **completed; skips** | 250 updates, lr=3e-05; 21 nonfinite losses skipped on table 0011. 178 s, exit 0. |
| 25-09-2026 | Pilot PD v2 full · Mindwell 11618752 | **completed; skips** | 250 updates, lr=3e-07; 21 nonfinite losses skipped on table 0011. 180 s, exit 0. |
| 25-09-2026 | Pilot PD v2.6 full · Mindwell 11618749 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 166 s, exit 0. |
| 25-09-2026 | Pilot PD v2.6 full · Mindwell 11618747 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 165 s, exit 0. |
| 25-09-2026 | Pilot PD v2.6 frozen · Mindwell 11618748 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 153 s, exit 0. |
| 25-09-2026 | Pilot PD v2 frozen · Mindwell 11618753 | **completed; skips** | 250 updates, lr=3e-07; 21 nonfinite losses skipped on table 0011. 176 s, exit 0. |
| 25-09-2026 | Pilot PD v2 frozen · Mindwell 11618750 | **completed; skips** | 250 updates, lr=3e-05; 21 nonfinite losses skipped on table 0011. 175 s, exit 0. |
| 25-09-2026 | Pilot PD TabICL full · Mindwell 11618746 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 200 s, exit 0. |
| 25-09-2026 | Pilot PD TabICL full · Mindwell 11618744 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 202 s, exit 0. |
| 25-09-2026 | Pilot PD TabICL frozen · Mindwell 11618745 | **passed** | 250 updates, lr=3e-07; no numerical skips logged. 195 s, exit 0. |
| 25-09-2026 | Pilot PD v2.6 frozen · Mindwell 11618743 | **passed** | 250 updates, lr=3e-05; no numerical skips logged. 154 s, exit 0. |
| 25-09-2026 | Part 1 null audit · wICE 62153798 | **passed** | 16/16 exact saved-state and monitor checks; all GPU counters sampled. 36 s, exit 0. |
| 25-09-2026 | Null LGD v3 full · Mindwell 11618741 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 32 s, exit 0. |
| 25-09-2026 | Null LGD v3 frozen · Mindwell 11618738 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 31 s, exit 0. |
| 25-09-2026 | Null LGD v2 full · Mindwell 11618740 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 27 s, exit 0. |
| 25-09-2026 | Null LGD v2 frozen · Mindwell 11618737 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 28 s, exit 0. |
| 25-09-2026 | Null PD v3 frozen · Mindwell 11618732 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 56 s, exit 0. |
| 25-09-2026 | Null PD v2.6 frozen · Mindwell 11618727 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 43 s, exit 0. |
| 25-09-2026 | Null PD v2 full · Mindwell 11618731 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 44 s, exit 0. |
| 25-09-2026 | Null PD TabICL frozen · Mindwell 11618726 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 41 s, exit 0. |
| 25-09-2026 | Null LGD v2.6 full · Mindwell 11618739 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 31 s, exit 0. |
| 25-09-2026 | Null LGD TabICL full · Mindwell 11618735 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 28 s, exit 0. |
| 25-09-2026 | Null LGD v2.6 frozen · Mindwell 11618736 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 29 s, exit 0. |
| 25-09-2026 | Null PD v3 full · Mindwell 11618734 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 57 s, exit 0. |
| 25-09-2026 | Null LGD TabICL frozen · Mindwell 11618733 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 31 s, exit 0. |
| 25-09-2026 | Null PD v2.6 full · Mindwell 11618729 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 45 s, exit 0. |
| 25-09-2026 | Null PD v2 frozen · Mindwell 11618730 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 48 s, exit 0. |
| 25-09-2026 | Null PD TabICL full · Mindwell 11618728 | **passed** | 2 updates; exact saved-state/monitor parity; zero drift. 44 s, exit 0. |
| 25-09-2026 | Part 1 preparation · wICE 62153693 | **passed** | Preflight 0 failures/warnings; prepared plans and released nulls. 50 s, exit 0. |
| 25-09-2026 | Null LGD v3 full · Mindwell 11618316 | **completed; unaudited** | Two updates, zero drift/monitor change; log 133 s, exit 0. |
| 25-09-2026 | Null LGD v2 full · Mindwell 11618315 | **completed; unaudited** | Two updates, zero drift/monitor change; log 126 s, exit 0. |
| 25-09-2026 | Null LGD v2.6 full · Mindwell 11618314 | **completed; unaudited** | Two updates, zero drift/monitor change; log 128 s, exit 0. |
| 25-09-2026 | Null LGD v3 frozen · Mindwell 11618313 | **completed; unaudited** | Two updates, zero drift/monitor change; log 133 s, exit 0. |
| 25-09-2026 | Null LGD v2 frozen · Mindwell 11618312 | **completed; unaudited** | Two updates, zero drift/monitor change; log 156 s, exit 0. |
| 25-09-2026 | Null LGD v2.6 frozen · Mindwell 11618311 | **completed; unaudited** | Two updates, zero drift/monitor change; log 129 s, exit 0. |
| 25-09-2026 | Null LGD TabICL full · Mindwell 11618310 | **completed; unaudited** | Two updates, zero drift/monitor change; log 127 s, exit 0. |
| 25-09-2026 | Null LGD TabICL frozen · Mindwell 11618309 | **completed; unaudited** | Two updates, zero drift/monitor change; log 156 s, exit 0. |
| 25-09-2026 | Null PD v3 full · Mindwell 11618308 | **completed; unaudited** | Two updates, zero drift/monitor change; log 90 s, exit 0. |
| 25-09-2026 | Null PD v2 full · Mindwell 11618307 | **completed; unaudited** | Two updates, zero drift/monitor change; log 91 s, exit 0. |
| 25-09-2026 | Null PD v2.6 full · Mindwell 11618306 | **blocked by checkpoint identity** | Old checkpoint occupied the same filename; log 57 s, exit 1. |
| 25-09-2026 | Null PD TabICL full · Mindwell 11618305 | **blocked by checkpoint identity** | Old checkpoint occupied the same filename; log 62 s, exit 1. |
| 25-09-2026 | Null PD v3 frozen · Mindwell 11618304 | **completed; unaudited** | Two updates, zero drift/monitor change; log 129 s, exit 0. |
| 25-09-2026 | Null PD v2 frozen · Mindwell 11618303 | **completed; unaudited** | Two updates, zero drift/monitor change; log 93 s, exit 0. |
| 25-09-2026 | Null PD v2.6 frozen · Mindwell 11618302 | **blocked by checkpoint identity** | Old checkpoint occupied the same filename; log 30 s, exit 1. |
| 25-09-2026 | Null PD TabICL frozen · Mindwell 11618301 | **blocked by checkpoint identity** | Old checkpoint occupied the same filename; log 65 s, exit 1. |
| 25-09-2026 | Corrected part-1 preparation · wICE 62152033 | **done** | Workflow 588ddf3d; preflight zero failures/warnings; released 16 null tasks; log 86 s, exit 0. |
| 25-09-2026 | Corrected experiment-0 part 1 · workflow 588ddf3d | **failed at null** | Previously user-reported resubmission now verified: 12 done, four identity conflicts; later stages not released. |
| 25-09-2026 | Null LGD v3 array · Mindwell 11618184 | **failed before training** | Both tasks 0:10/SIGUSR1 after 27 s; ten-minute warning on ten-minute allocation. |
| 25-09-2026 | Null LGD v2 array · Mindwell 11618183 | **failed before training** | Both tasks 0:10/SIGUSR1 after 27 s. |
| 25-09-2026 | Null LGD v2.6 array · Mindwell 11618182 | **failed before training** | Both tasks 0:10/SIGUSR1 after 27 s. |
| 25-09-2026 | Null LGD TabICL array · Mindwell 11618181 | **failed before training** | Both tasks 0:10/SIGUSR1 after 27 s. |
| 25-09-2026 | Null PD v3 array · Mindwell 11618178 | **failed before training** | Both tasks 0:10/SIGUSR1 after 30 s. |
| 25-09-2026 | Null PD v2 array · Mindwell 11618176 | **failed before training** | Both tasks 0:10/SIGUSR1 after 30 s. |
| 25-09-2026 | Null PD v2.6 array · Mindwell 11618173 | **completed; unaudited** | Two updates in each adaptation mode; reported drift 0; 67–68 s, exit 0. |
| 25-09-2026 | Null PD TabICL array · Mindwell 11618172 | **completed; unaudited** | Two updates in each adaptation mode; reported drift 0; 67–71 s, exit 0. |
| 25-09-2026 | Fresh part-1 preparation · wICE 62151928 | **done** | Workflow 9278d539; preflight 0 failures/warnings; 42 s in log, exit 0; released 16 null tasks. |
| 25-09-2026 | Isolated recovery final audit · wICE 62149461 | **passed** | Workflow 22af7599; 8/8 pairs, diagnostic recovery receipt only; 20 s, exit 0. |
| 25-09-2026 | Deterministic recovery LGD TabICL · Mindwell 11617313 | **passed** | Exact state/0-5-12 monitors; five-fold smoke, 442 predictions; 83 s, exit 0. |
| 25-09-2026 | Deterministic recovery LGD v3 · Mindwell 11617323 | **passed** | Exact model/monitors; detached criterion diagnostic differs; five-fold smoke, 442 predictions; 87 s, exit 0. |
| 25-09-2026 | Deterministic recovery LGD v2.6 · Mindwell 11617322 | **passed** | Exact model/monitors; detached criterion diagnostic differs; five-fold smoke, 442 predictions; 70 s, exit 0. |
| 25-09-2026 | Deterministic recovery LGD v2 · Mindwell 11617321 | **passed** | Exact model/monitors; detached criterion diagnostic differs; five-fold smoke, 442 predictions; 63 s, exit 0. |
| 25-09-2026 | Deterministic recovery PD v3 · Mindwell 11617316 | **passed** | Exact state/0-5-12 monitors; five-fold smoke, 569 predictions; 153 s, exit 0. |
| 25-09-2026 | Deterministic recovery PD v2 · Mindwell 11617314 | **passed** | Exact state/monitors; one non-finite batch skipped per arm; 12 updates, 569 smoke predictions; 133 s, exit 0. |
| 25-09-2026 | Deterministic recovery PD v2.6 · Mindwell 11617315 | **passed** | Exact state/0-5-12 monitors; five-fold smoke, 569 predictions; 122 s, exit 0. |
| 25-09-2026 | Deterministic recovery PD TabICL · Mindwell 11617317 | **passed** | Exact state/0-5-12 monitors; five-fold smoke, 569 predictions; 119 s, exit 0. |
| 25-09-2026 | Isolated recovery preparation · wICE 62149265 | **done** | Workflow 22af7599, corrected 3037f36 source; preflight zero failures/warnings; 50 s, exit 0. |
| 25-09-2026 | Deterministic recovery LGD TabICL · Mindwell 11615916 | **passed** | Exact saved state and 0/5/12 monitors; five-fold smoke, 442 predictions; exit 0. |
| 25-09-2026 | Deterministic recovery LGD v3 · Mindwell 11615923 | **passed** | Exact model/monitors; only detached criterion diagnostic differs; 442 predictions; exit 0. |
| 25-09-2026 | Deterministic recovery LGD v2.6 · Mindwell 11615922 | **passed** | Exact model/monitors; only detached criterion diagnostic differs; 442 predictions; exit 0. |
| 25-09-2026 | Deterministic recovery LGD v2 · Mindwell 11615921 | **passed** | Exact model/monitors; only detached criterion diagnostic differs; 442 predictions; exit 0. |
| 25-09-2026 | Deterministic recovery PD TabICL · Mindwell 11615920 | **passed** | Exact saved state and 0/5/12 monitors; five-fold smoke, 569 predictions; exit 0. |
| 25-09-2026 | Deterministic recovery PD v3 · Mindwell 11615919 | **crashed** | Spatial CUDA CE reduction lacks deterministic implementation; reference stops before its first update; exit 1. |
| 25-09-2026 | Deterministic recovery PD v2.6 · Mindwell 11615918 | **crashed** | Same spatial CUDA CE refusal at first training loss; exit 1. |
| 25-09-2026 | Deterministic recovery PD v2 · Mindwell 11615917 | **crashed** | Same spatial CUDA CE refusal at first training loss; exit 1. |
| 25-09-2026 | Isolated recovery preparation · wICE 62144804 | **done** | Four uniquely named reference/resumed plans, workflow 1338dabf; 64 s, exit 0. |
| 25-09-2026 | PD v2 repeatability probe · Mindwell 11615820 | **GPU variation isolated** | 10 s, exit 0; default: 110/129 gradient tensors differ, max 2.44e-4; deterministic and deterministic math: exact repeats. No optimizer steps/checkpoints. |
| 25-09-2026 | inspect-recovery · wICE 62143430 | **inspection passed; recovery still failed** | 18 s, exit 0; all eight identities valid, model deltas 5.24e-6–9.00e-6, pre-pause differences in all pairs; larger LGD deltas are diagnostic buffers. |
| 24-09-2026 | v5 check2 recovery LGD TabICL · Mindwell 11614392 | **comparison failed** | Both arms reached 12 updates; state/trajectory mismatch, five-fold smoke passed; exit 1. |
| 24-09-2026 | v5 check2 recovery LGD v3 · Mindwell 11614838 | **comparison failed** | Both arms reached 12 updates; state/trajectory mismatch, five-fold smoke passed; exit 1. |
| 24-09-2026 | v5 check2 recovery LGD v2.6 · Mindwell 11614834 | **comparison failed** | Both arms reached 12 updates; state/trajectory mismatch, five-fold smoke passed; exit 1. |
| 24-09-2026 | v5 check2 recovery PD v3 · Mindwell 11614824 | **comparison failed** | Both arms reached 12 updates; state/trajectory mismatch, five-fold smoke passed; exit 1. |
| 24-09-2026 | v5 check2 recovery PD TabICL · Mindwell 11614829 | **comparison failed** | Both arms reached 12 updates; state/trajectory mismatch, five-fold smoke passed; exit 1. |
| 24-09-2026 | v5 check2 recovery LGD v2 · Mindwell 11614830 | **comparison failed** | Both arms reached 12 updates; state/trajectory mismatch, five-fold smoke passed; exit 1. |
| 24-09-2026 | v5 check2 recovery PD v2 · Mindwell 11614817 | **comparison failed** | Both arms reached 12 updates; state/trajectory mismatch, five-fold smoke passed; exit 1. |
| 24-09-2026 | v5 check2 recovery PD v2.6 · Mindwell 11614818 | **comparison failed** | Both arms reached 12 updates; state/trajectory mismatch, five-fold smoke passed; exit 1. |
| 24-09-2026 | v5 check2 pilot audit · wICE 62141239 | **passed** | 32 complete, 0 pending/divergent/problems; exit 0; recovery released. |
| 24-09-2026 | v5 check2 pilot LGD v3 full LR 3e-05 · Mindwell 11614385 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v3 full LR 3e-07 · Mindwell 11614383 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v3 frozen LR 3e-05 · Mindwell 11614363 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v3 frozen LR 3e-07 · Mindwell 11614384 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v2 full LR 3e-05 · Mindwell 11614382 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v2 frozen LR 3e-05 · Mindwell 11614362 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v2 frozen LR 3e-07 · Mindwell 11614381 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v2.6 full LR 3e-05 · Mindwell 11614379 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v2 full LR 3e-07 · Mindwell 11614380 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v2.6 frozen LR 3e-05 · Mindwell 11614361 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v2.6 full LR 3e-07 · Mindwell 11614375 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD v2.6 frozen LR 3e-07 · Mindwell 11614378 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD TabICL full LR 3e-05 · Mindwell 11614374 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v3 full LR 3e-05 · Mindwell 11614371 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD TabICL frozen LR 3e-05 · Mindwell 11614360 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v3 frozen LR 3e-05 · Mindwell 11614359 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v3 full LR 3e-07 · Mindwell 11614369 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v3 frozen LR 3e-07 · Mindwell 11614370 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD TabICL full LR 3e-07 · Mindwell 11614372 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot LGD TabICL frozen LR 3e-07 · Mindwell 11614373 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v2 full LR 3e-05 · Mindwell 11614367 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v2 frozen LR 3e-05 · Mindwell 11614358 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v2 full LR 3e-07 · Mindwell 11614365 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v2 frozen LR 3e-07 · Mindwell 11614366 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v2.6 full LR 3e-05 · Mindwell 11614364 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v2.6 frozen LR 3e-05 · Mindwell 11614352 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD TabICL full LR 3e-07 · Mindwell 11614353 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD TabICL full LR 3e-05 · Mindwell 11614355 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD TabICL frozen LR 3e-05 · Mindwell 11614351 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD TabICL frozen LR 3e-07 · Mindwell 11614354 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v2.6 full LR 3e-07 · Mindwell 11614356 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 pilot PD v2.6 frozen LR 3e-07 · Mindwell 11614357 | **passed audit** | 250 updates, complete diagnostics, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null audit · wICE 62141138 | **passed** | 16 exact state/monitor matches, resource audit passed; exit 0; pilots released. |
| 24-09-2026 | v5 check2 null LGD v3 full LR 0e00 · Mindwell 11614349 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null LGD v3 frozen LR 0e00 · Mindwell 11614344 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null LGD v2 frozen LR 0e00 · Mindwell 11614343 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null LGD v2 full LR 0e00 · Mindwell 11614348 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null PD v3 frozen LR 0e00 · Mindwell 11614338 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null PD v3 full LR 0e00 · Mindwell 11614345 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null LGD v2.6 full LR 0e00 · Mindwell 11614347 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null LGD v2.6 frozen LR 0e00 · Mindwell 11614342 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null LGD TabICL frozen LR 0e00 · Mindwell 11614341 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null LGD TabICL full LR 0e00 · Mindwell 11614346 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null PD v2 full LR 0e00 · Mindwell 11614340 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null PD v2 frozen LR 0e00 · Mindwell 11614337 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null PD v2.6 full LR 0e00 · Mindwell 11614339 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null PD v2.6 frozen LR 0e00 · Mindwell 11614336 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null PD TabICL frozen LR 0e00 · Mindwell 11614334 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 null PD TabICL full LR 0e00 · Mindwell 11614335 | **passed audit** | 2 updates, exact state/monitor parity, valid GPU samples; exit 0. |
| 24-09-2026 | v5 check2 part1 prepare · wICE 62141115 | **passed** | CPU preparation passed, null controls released; exit 0. |
| 24-09-2026 | standalone GPU resource check · Mindwell 11614333 | **passed** | Second supplied counter check returned sampled GPU counters, exit 0; no training. |
| 24-09-2026 | standalone GPU resource check · Mindwell 11614332 | **passed (download verified)** | Correct env, `gpu_status: sampled`, valid utilization/memory/power/temperature, exit 0; ready for corrected part-1 rerun. |
| 24-09-2026 | standalone GPU resource check · Mindwell, 20:19 | **submission rejected; no job** | Site plugin rejects `--gpus=1`; retry with `--gpus-per-node=1`. No GPU work occurred. |
| 24-09-2026 | v5 null audit · wICE 62140691 | **resource gate failed (download verified)** | 16 exact state/monitor matches; only missing GPU counters, 25 CalledProcessError samples; exit 1 stopped part 1. |
| 24-09-2026 | v5 null LGD v3 full · Mindwell 11614316 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null LGD v2 full · Mindwell 11614315 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null LGD v2.6 full · Mindwell 11614314 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null LGD TabICL full · Mindwell 11614313 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null PD v3 full · Mindwell 11614312 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null LGD v3 frozen · Mindwell 11614309 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null LGD v2 frozen · Mindwell 11614308 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null LGD v2.6 frozen · Mindwell 11614307 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null LGD TabICL frozen · Mindwell 11614306 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null PD v2 full · Mindwell 11614305 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null PD v3 frozen · Mindwell 11614304 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null PD v2 frozen · Mindwell 11614303 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null PD v2.6 full · Mindwell 11614302 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null PD TabICL full · Mindwell 11614301 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null PD v2.6 frozen · Mindwell 11614300 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | v5 null PD TabICL frozen · Mindwell 11614299 | **model checks passed** | 2 updates, exact state/monitor parity; GPU counters unavailable. |
| 24-09-2026 | experiment-0 part1 prepare · wICE 62140639 | **passed (download verified)** | Zero CPU failures/warnings; plans/staging completed, 16 null GPU controls released; exit 0. |
| 24-09-2026 | experiment-0 part1 prepare · wICE 62134902 | **crashed (download verified)** | Exit 1: missing `OmegaConf` import in retention preflight; failed ledger still at preparation, no GPU work released. |
| 24-09-2026 | prepare cpt_pilot_v4 LGD · wICE 62129269 | **plan written (download verified)** | 16 trials, 250 updates, one partition; exit 0. Plan/trial checksums and current training source match; no pilot training evidenced. |
| 24-09-2026 | prepare cpt_pilot_v4 PD · wICE 62129268 | **plan written (download verified)** | 16 trials, 250 updates, one partition; exit 0. Plan/trial checksums and current training source match; no pilot training evidenced. |
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

### 25-09-2026 — A finite loss can conceal overflow-induced feature erasure

**Tried:** Compare the current clip, upstream float32 clip and upstream float64 clip under BF16/FP32 arithmetic.
**Result:** Only the float32 upstream clip produces finite loss, while introducing 1,888 extra missing values per member. Retaining the values with float64 clipping still fails in subsequent model arithmetic.
**Why:** Finite raw values exceed float32 storage and encoder statistical range; the old cast erased 111,353 cells across two columns, and upstream float32 statistics can turn additional observations into NaNs that the model then imputes.
**Instead:** Preserve finite inputs through recorded numerical unit changes before dtype casting, match the actual context-fitted clipping algorithm, reject overflowing bounds, and require zero numerical skips in debugging gates. Rebuild data and validate before accepting fresh controls.

### 25-09-2026 — Adding a manifest column requires a complete row schema

**Tried:** Persist numerical unit exponents through the existing dataset manifest columns.
**Result:** The focused suite reports 146 passes, one skip and one schema test failure before GPU work.
**Why:** The new column was listed in MANIFEST_COLUMNS but absent from the registration row constructor.
**Instead:** Initialize it as empty until sanitization succeeds; verify the full registration/sanitization/CSV round trip and rerun validation on the completed source.

### 25-09-2026 — An encoder validation error aborted the numerical diagnostic

**Tried:** Compare current/upstream clipping with BF16/FP32 on the real table that training repeatedly skipped.
**Result:** The first profile raised `ValueError`; the job stopped before printing its input aggregates or trying the other profiles.
**Why:** The diagnostic caught `RuntimeError` and `ImportError`, while TabPFN uses `ValueError` for encoded-input NaNs; the final-only report hid partial evidence.
**Instead:** Record expected validation failures per profile, stream aggregates/results before continuing, and test the exact exception path locally. Add wider clipping arithmetic in the same bounded allocation; do not interpret this diagnostic as a passing training control.

### 25-09-2026 — Successful-update budgets can conceal a missing training table

**Tried:** Accept null/pilot/recovery gates based on finite monitors, exact recovery and completed update budgets.
**Result:** All automated gates passed, but each of four PD v2 pilots skipped all 21 visits to one credit table.
**Why:** The skip guard protects weights from non-finite gradients but the current audit does not require successful exposure to every training table; aggregate update counts conceal systematic exclusions.
**Instead:** Hold the long pilots; compare real-table preprocessing and precision with zero optimizer updates, fix the verified cause, and verify per-table coverage before accepting a production run. Do not hide warnings or bypass identities.

### 25-09-2026 — New plans do not remove incompatible project checkpoints

**Tried:** Retry part 1 with the corrected launcher and newly prepared identities while four existing final checkpoint filenames remained.
**Result:** Twelve null controls complete; four PD v2.6/TabICL tasks refuse mismatched identities and stop workflow progression.
**Why:** Prepared plans describe the new source but do not delete existing weights; cleaning output folders alone cannot clear `checkpoints/trained/` on project storage.
**Instead:** Keep identity checks, preview the complete two-tier cleaner, wait for successful cleanup, and start one fresh part 1 after deploying the pending runtime fixes. Original data and base weights stay intact.

### 25-09-2026 — Editing loaded code invalidated a local full-suite run

**Tried:** Continue moving the GPFS path helper and adding tests while pytest was already running.
**Result:** Four failures after 635 passes: an already-imported module lacked the new helper, and source inspection used line positions from the previous file contents.
**Why:** The process mixed old imported objects with edited source. This was an invalid local validation run, not evidence of those failures on VSC.
**Instead:** Finish runtime edits before starting the full suite; the 55 focused checks passed on the finished files, then restart the full suite with runtime code held fixed.

### 25-09-2026 — A repeatable single-forward probe missed the ensemble loss refusal

**Tried:** Apply strict kernels to all eight recovery pairs after a successful small PD v2 repeatability probe.
**Result:** Five pairs pass exactly for model tensors/monitors; all three PD TabPFN arms abort at the first loss.
**Why:** The probe flattened logits to 2D, while the production ensemble used 3D logits and selected CUDA's nondeterministic spatial NLL reduction. It was a loss-shape coverage gap, not evidence of another resume-state mismatch.
**Instead:** Reuse the 2D CE path after correctly aligning all member/query samples, test loss/gradient equivalence, and make the probe use the ensemble path. Revalidate on B200 without relaxing determinism.

### 25-09-2026 — Mocked analysis metadata and direct benchmark smoke bypassed integration failures

**Tried:** Rely on a frozen-failure fixture with adaptation_mode and on benchmark smoke calls that bypass the final evaluation roster.
**Result:** Actual unfinished manifests still crashed notebooks; final evaluation had an undefined OmegaConf and rejected valid planned checkpoint identities.
**Why:** RunRow never writes adaptation_mode; a local import is invisible to another function; loaded handles omitted provenance used by the guard. The original fixtures did not exercise these boundaries.
**Instead:** Reproduce from real RunRow fields and the actual manifest/sidecar loader, test valid/foreign/missing identities, and keep both the roster guard and strict notebook membership check.

### 25-09-2026 — Local full-suite stall at the persistent-worker test

**Tried:** Run the full Windows CPU suite after the deterministic-recovery changes.
**Result:** No assertions failed, but the process stopped progressing before the final 20 tests and remained idle for several minutes; stopped only that verified pytest process. All final 20 tests then passed in isolation in 16.57 s.
**Why:** The stall did not reproduce in isolation; its cause is unconfirmed. Do not describe an interrupted suite as a passing full run or infer a VSC failure from it.
**Instead:** Reran the complete suite with a 90-second faulthandler traceback timer: **567 passed, 1 skipped in 553.20 s**. The worker stall did not recur; one timer dump showed a slow large-CSV read in an existing data test, which subsequently passed.

### 25-09-2026 — Matching seeds do not make the GPU recovery comparison deterministic

**Tried:** Compare independently started uninterrupted/resumed training with matching seeds and default B200 kernels.
**Result:** All eight recovery pairs differed before the pause; the targeted PD v2 probe reproduced unequal gradients despite restoring state and every RNG.
**Why:** The default GPU calculation is nondeterministic in the measured case; checkpoint serialization alone cannot explain differences already present before interruption.
**Instead:** Enforce strict kernels for the small recovery control, complete Python RNG seeding and test all eight pairs in an isolated workflow. Keep production-sized pilots for memory/throughput evidence; perform the user's full clean rerun only after recovery validation.

### 25-09-2026 — CPU state inspection cannot establish GPU kernel repeatability

**Tried:** Read the existing B200 checkpoints/trajectories on wICE with per-tensor and per-milestone comparisons.
**Result:** All identities pass, the misleading LGD maximum is explained, but all eight actual model/trajectory comparisons still fail before the pause.
**Why:** The two independent training paths were not repeatable before interruption; seven initial monitors match, while TabICL PD already differs in three calibrated fields. GPU kernels or other uncontrolled variation remain hypotheses.
**Instead:** Run one bounded synthetic forward/backward repeatability diagnostic from identical weights/buffers/RNG under three kernel settings, with no optimizer steps or checkpoint writes. Keep the successful controls/pilots and the failed recovery gate.

### 24-09-2026 — recovery comparison conflates independent-run and resume differences

**Tried:** Eight B200 pairs, 12 uninterrupted updates versus stop at 5 and resume to 12, then strict final-state/monitor comparison.
**Result:** All eight comparisons failed; all eight five-fold benchmark smoke checks passed. The earlier 16 null controls and 32 short pilots passed their audits.
**Why:** Logs show the independent arms already differ before the interruption. Final differences cannot isolate a resume defect, and the largest LGD state delta mixes weights and loss-diagnostic buffers. The numerical root cause is not yet confirmed.
**Instead:** Inspect existing project-storage checkpoints/trajectories on CPU with per-tensor deltas and per-milestone comparisons. Preserve the passed stages; establish repeatability before interpreting or repeating the GPU recovery test. Never loosen tolerances simply to release the long pilots.

### 24-09-2026 — generic Slurm GPU option rejected by Mindwell

**Tried:** Submit the standalone GPU resource check with `--gpus=1` from the agent's command.
**Result:** Mindwell's submission plugin rejected it before assigning a job ID or allocating a GPU.
**Why:** The site requires `--gpus-per-node`; Bash syntax checks cannot validate cluster-specific submission restrictions.
**Instead:** Use `--gpus-per-node=1`, matching the repository's working training wrappers; retry only this unsubmitted check.

### 24-09-2026 — GPU counters received PyTorch's bare UUID

**Tried:** Run v5 bundled experiment 0 with periodic NVIDIA counters alongside the null controls.
**Result:** All 16 training jobs and state/monitor comparisons passed, but every resource sample failed; the CPU gate stopped progression.
**Why:** PyTorch supplies an unprefixed UUID; NVIDIA's selector requires `GPU-`. The sampler retained only the exception class, hiding the actual command failure, and unit tests had not exercised resource collection.
**Instead:** Normalize the selector, retain bounded error details with one console warning, test actual sampler/CLI behavior, and verify a standalone B200 resource query before fresh `check2` controls.

### 24-09-2026 — preflight orchestration was not exercised by helper tests

**Tried:** Start bundled experiment 0 after the public monitoring inputs were downloaded.
**Result:** CPU job 62134902 exited 1 on `NameError: OmegaConf`; the workflow correctly stopped before GPU submission.
**Why:** Other preflight helpers imported OmegaConf locally, but the new retention check in `main()` lacked an import. Existing helper tests missed the CLI path.
**Instead:** Import once at module scope; test the actual CLI loop with available and missing retention inputs, and execute the full local preflight before handing over the rerun.

### 24-09-2026 — local output deletion rejected

**Tried:** Remove only the resolved repository-local `output CreditPFN/` tree with PowerShell after the user's fresh-start request.
**Result:** Automatic approval review rejected the bounded removal with “blocked by policy”; the files remain.
**Why:** No more specific reason was supplied. Filesystem permissions did not authorize bypassing the review decision.
**Instead:** Stop deletion attempts, keep active verification output in temporary storage and give the user the manual removal command.

### 24-09-2026 — sparse quantiles never reached the native API

**Tried:** Record a fixed nine-quantile grid through the model wrappers during the extended evaluation audit.
**Result:** Wrappers accepted only `X`, so distribution requests were swallowed as unavailable; TabICL's fallback default grid also differed from our declared levels.
**Why:** TabPFN uses `quantiles`, TabICL uses `alphas`, and TabPFN returns a list of quantile arrays rather than a row-major matrix.
**Instead:** Use explicit native multi-output calls, reuse one forward, test square-matrix orientation/levels and exercise distribution scoring in each GPU recovery canary.

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
