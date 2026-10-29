# CreditPFN — research brief

This document is intended to stand alone when shared with a collaborator or uploaded into another chat. Operational commands live in [VSC.md](VSC.md), primary evidence in [LITERATURE.md](LITERATURE.md), and historical runs in [AGENTS_MEMORY.md](AGENTS_MEMORY.md).

## Additional model coverage: TabPFN v2 and v2.6

Authorized on 29-10-2026 after inspecting the completed main results. This is an
exploratory coverage extension, not a retrospectively preregistered confirmation.
Experiments 1–3 already include both models and are not retrained. New experiment-0
checks and experiments 4–8 use an isolated extension ID on both storage tiers.

| Experiment | Additional design for each version | New training trials, both versions |
|---|---|---:|
| 4 | Original versus the fixed E1 full-update reference (LR 3e-7, L2-SP 0.003, seed 42); five row folds, nested context budgets, all test rows | 0 |
| 5 | DiagScale, PD/LGD, three original adapter rates, four unchanged dataset partitions | 48 |
| 6 | PD all-table versus >10,000-row training-table pool; same held-out tables and 8,192-row episode cap | 16 |
| 7 | Fixed large-table pool: 2,048 rows, shared E6 8,192-row control, model-specific training ceiling | 16 |
| 8 | Native Q/V LoRA, PD/LGD, rank 16, alpha 32, original three rates; no weight decay or L2-SP | 48 |

Inference context caps are **10,000 for v2 and 50,000 for v2.6**. Larger nominal
context settings are removed; the native cap is included, and every dataset/fold also
retains its full available context when smaller. The fixed validation budget and all
outer-test rows remain unchanged. The benchmark does not silently replace v2/v2.6
with the newest package default.

The separately measured training episode ceilings are **10,000 for v2 and 11,000
for v2.6** total context-plus-query rows. E5/E6 retain 8,192; E7 reaches the applicable
ceiling; E8 uses the applicable ceiling, as its existing design uses each model's
measured training cap. The four dataset partitions, update budget, sampling, seeds
and rates remain those of the corresponding experiment. E7's upper endpoints are
model-specific operating points, not equal-row cross-model comparisons. Compare
changes within a model, with E6's exact shared 8,192-row control.

The extension comprises 128 training trials, 250 context-evaluation tasks and 896
foundation-benchmark shards (some may legitimately be empty). Original-model controls
are evaluated with each study's same protocol; completed classical baselines are not
repeated. Benchmarks retain predictions and five-fold evaluation. The source-bound
experiment-0 gate tests 12 correctness cases and 68 capacity cases, including every
training partition/arm, both tasks, both model versions, every adapter rate in short
recovery tests, native export/reload, cached inference and context/test separation.
These are bounded validation probes, not proof that every possible failure is absent.
No extension VSC results are available at implementation time.

## Question and intended contribution

**How does continued pretraining on a small credit-domain corpus change the behavior of tabular foundation models?** The aim is a descriptive account of learning rate, anchoring, adaptation scope and training progress across architectures and tasks. Improvements, degradation, negligible movement and numerical instability are all possible findings.

The primary object is the response over the specified grid, not a winning model. Describe effect sizes, trajectories, variation across datasets and compute cost. The maximum observed score is an exploratory result; it is not an unbiased estimate for a selected deployment recipe. The current design does not need a separate selection corpus to describe its prespecified grid. A later champion claim, a redesigned grid chosen from these results, or a deployment recommendation needs fresh evaluation or explicitly exploratory wording.

Compare each adapted model with its exact starting checkpoint. The paper can establish behavior on this corpus and budget; it cannot establish a universal benefit/null effect, domain specificity versus generic real-data adaptation, or absence of forgetting without the corresponding controls.

### From theory to the empirical questions

The organizing question is **when credit-domain specialization helps or impairs the in-context
learning procedure of a tabular foundation model**. Prior-data NLL learns predictions under a
training-task distribution; credit CPT changes that distribution and can either specialize useful
structure or narrow it excessively. A fixed model can also become less sensitive to random context
draws without becoming less systematically wrong. These mechanisms motivate competing expectations,
not a guarantee of positive transfer. Primary sources and their assumptions are in [LITERATURE.md](LITERATURE.md).

| Question | Relevant evidence | Interpretation boundary |
|---|---|---|
| Does stronger credit adaptation improve excluded credit tables while preserving non-credit behavior? | Experiment 1: learning rates, L2-SP, update scope, credit/retention trajectories and paired final scores | Parameter displacement is not predictive bias; finite-panel retention is not universal absence of forgetting. |
| Is the reference result sensitive to CPT training randomness? | Experiment 2: seed 42 versus 43, with fixed inference folds and context policy | Two seeds describe sensitivity at one recipe, not grid-wide variance. |
| Does revisiting more rows or changing update weighting make CPT more useful per unit cost? | Experiment 3: paired sampling arms, row exposure and compute | Equal updates do not equal compute or joint all-row context. |
| Does CPT change how much an unseen credit table needs labeled context? | Experiment 4: original/CPT curves on identical nested prefixes and test rows | Context size dependence is not repeated-context variance; validation labels remain an additional fixed budget. |
| Does restricting adaptation to attention query scaling preserve useful behavior? | Experiment 5: all three adapter rates against their matched original | Cross-table CPT differs from target-table fine-tuning; comparisons with experiment 1 also change recipe details. |
| Is a large-table-only training policy helpful, and does episode size help within that fixed pool? | Experiment 6: corpus eligibility; experiment 7: context-plus-query rows | The first changes corpus composition; the second changes context, query count and compute. Neither isolates all notions of “more data.” |
| Can low-rank changes inside attention transfer useful credit-domain structure? | Experiment 8: fixed query/value LoRA design at three adapter rates, paired with original weights | Added after earlier results; exploratory. No claim that its one rank/target policy represents all LoRA designs. |

The framing was sharpened after results were visible. Present these as theory-informed exploratory
questions; do not retroactively claim preregistered directional hypotheses. The contribution is a
bounded account of domain CPT and its failure/success conditions, not a generic TFM leaderboard.
An empirical paper can be coherent without a positive mean gain or an additional numbered experiment.

The [results guide](RESULTS_GUIDE.md) maps these questions to notebook sections and exact saved
figures. Factorial final-benchmark response maps retain all planned recipes on a common table
cohort within base and domain. Their bands are table interquartile ranges, not uncertainty over
training seeds or independent folds. Context curves retain individual table effects without
estimating repeated-context variability. Numerical-policy effects subtract original from CPT
within the same diagnostic table, policy and outer fold. These are summaries of existing outputs,
not additional experiments or test-based recipe selection.

### Saved test predictions and post-hoc diagnostics

`config/eval.yaml` enables `results.save_predictions`. Standard benchmarks write the metric CSV,
its `.csv.evaluation.json` receipt and a sibling prediction file (`.parquet`, or `.csv.gz` when no
Parquet engine is installed) under project `output CreditPFN/<experiment>/results/<TRACK>/<method>/`.
The receipt records the prediction filename and byte size; it is not a prediction-content hash.
Each prediction has `test_dataset_id`, `model_name`, `fold_idx`, processed-table `row_idx`, `y_true`
and `y_pred`. PD stores raw positive-class probabilities; LGD stores point predictions and the
configured nine quantiles when supported. Per-row recalibrated PD probabilities are not stored;
validation-calibrated fold metrics and reliability bins are stored separately. A sparse quantile
grid does not support exact CRPS or arbitrary quantiles. Experiment 4 instead stores compressed
arrays in each task's SQLite `cells.predictions`, including exact context/validation/test indices,
test targets/predictions and validation targets/predictions. Metric CSV exports alone omit those arrays.

These artifacts support the following analysis **without additional GPU work**:

- Within each held-out credit table, pair original and CPT by partition, outer fold and row;
  require matching targets, all five successful folds, expected row counts and valid prediction
  exports. Report mean prediction minus mean observed target, Brier/MSE, and RMS/mean absolute
  prediction change. Positive signed error denotes overprediction. A lower absolute mean error
  is not necessarily lower Brier/MSE, so show them together.
- Use the predefined experiment-1 reference, all prescribed arms in experiments 3/5/6/7, and
  matched seed-42/43 reference predictions in experiment 2. Do not select favorable recipes or
  subgroups from test performance. Seed disagreement is descriptive training-seed sensitivity.
- Pool out-of-fold rows **within a table** for these residual diagnostics; give tables equal
  weight across the corpus. This can differ slightly from averaging five fold metrics when fold
  sizes differ. Keep coverage and per-table values, not only an aggregate. Predictions from a
  changing or incomplete arm cohort do not enter a shared comparison.

The new reports read exports in place and expose aggregates only. They do not publish per-borrower
targets or predictions. Missing project artifacts produce an explicit unavailable status, not
zero bias or failure. Mismatched identities, invalid probabilities or corrupted exports stop the
analysis. Completed scheduler tasks alone cannot establish prediction-level validity.

**What is not identifiable from the existing design:** each test row has one out-of-fold prediction
per checkpoint and context size. The five folds contain different test rows, and experiment 4 has
one nested context sequence per fold. These do not provide repeated contexts at fixed size for the
same query. Moreover, the true conditional default probability or conditional mean LGD is unknown.
Thus neither variance across borrowers nor average signed residual is the theoretical model
bias/context-variance term. Even a perfectly calibrated model need not be conditionally unbiased.

If further explanation becomes necessary, an optional inference-only extension would retain the
existing excluded credit tables, fixed CPT checkpoints, outer test rows and validation partitions;
resample multiple contexts at a few fixed sizes; pair draws between original and CPT; and freeze
inference-view randomness. This directly tests whether **credit CPT changes context sensitivity**.
It would estimate finite-pool sensitivity, not prove true-bias reduction or deployment robustness.
It needs its own saved context identities and fast correctness gate before any VSC submission.
No such run is currently implemented or authorized by this analysis update. The broad synthetic
mechanism and row-reservoir proposals remain unadopted; finish and interpret the existing study first.

## Corpus, partitions and evaluation population

- **25 registered tables: 17 PD classification and 8 LGD regression.** Every registered processed file must exist before a corrected run starts. Keep the corpus fixed throughout the campaign.
- Four dataset folds, partition seed **1729**. PD held-out counts are **4, 4, 5, 4**; LGD counts are **2, 2, 2, 2**. Training uses the complement: 12/13 PD tables or 6 LGD tables. Balance means table count, not equal rows, default rates, countries or institutions.
- Every dataset is held out exactly once per recipe/training seed. It may be a training dataset in the other folds. The four fitted models overlap substantially in their training corpora; they are not four independent statistical replications.
- Distinct dataset names do not establish independent populations. Related releases, loans, borrowers and institutions require a source-level grouping audit before stronger independence or out-of-time claims. No automatic cross-dataset deduplication system is reinstated.
- Held-out dataset evaluation currently uses five row-level outer CV folds, plus an inner validation split for classical tuning, decision thresholds and calibration. Evaluation seed **99** is fixed across training seeds. These are IID table benchmarks; they do not simulate future origination cohorts.
- All outer test rows are scored. Inference context caps depend on the base, but are identical for that base's trained and untuned arms. Cross-base rankings therefore combine architecture, prior and native context capacity.

Explicit train/test lists used by isolated controls are exact: unknown IDs fail, and specifying
only one side fills only its complement. Inherited fold/count defaults cannot expand a pinned
side. Research experiments use the fixed folds above; corpus-policy filters apply only after
those held-out assignments are fixed.

Only schema-compatible, legally usable data belong in the corpus. Raw private data, names, contents and credentials do not belong in the paper or Git. Figures use the private display-name mapping; the temporary tracked preprocessing-slug exception is a source-code exception, not permission to publish those names in results.

## Experiment organization

Experiment **0** owns debugging and readiness: null controls, positive-LR/recovery checks, short timing pilots and long budget pilots. Experiment **1** is the full descriptive sweep. Experiment **2** repeats one predefined reference at an additional training seed. Experiment **3** compares one-sample, full-pass and accumulated-gradient training. The general notebooks describe the corpus independently of these outcomes.

Recovery equivalence uses strict deterministic execution and a 2,048-row training cap to separate serialization correctness from GPU arithmetic variation. Null/short/budget pilots and the research grid retain production row caps and fast kernels. A fixed seed controls stochastic inputs; it does not guarantee bitwise equality on nondeterministic GPU kernels. Numerical execution settings are recorded. Small recovery controls establish state restoration under their tested conditions, not capacity or cross-platform reproducibility.

Configs are grouped under `config/experiment0/` through `config/experiment7/`; notebooks use experiment groups plus `general/`. Experiments 1–3 retain their `cpt_*_v5` identities; experiments 4–7 have separate run names and the additional validation gate described below. Isolated auxiliary pilots use unique `cpt_pilot_aux_*` names. Earlier v4 controls remain historical evidence, not passing controls for v5.

## Experiment 1: implemented main grid

| Factor | Levels / rule | Meaning |
|---|---|---|
| Base | TabPFN v2, v2.6, v3; TabICLv2 | Separate classifier/regressor weights, identified by SHA-256 |
| Peak learning rate | `3e-7`, `1e-6`, `1e-5`, `3e-5` | From a conservative real-table CPT reference to a stress level |
| L2-SP | `0`, `0.003` | Add `0.5 × lambda × sum((w - w0)^2)` over trainable pretrained parameters |
| Adaptation | Full parameter updates; frozen backbone | Exact trainable names/counts recorded in checkpoint provenance |
| Dataset sampling | `one_sample` | One sampled context/query batch per table visit; equal table visitation and a new random table order each round |
| Dataset partitions | Four fixed folds | Every table is a held-out dataset once |
| Training seed | `42` | Common randomness across recipes where compatible; different architectures/caps do not imply identical numerical computation |
| Budget | 10,000 successful optimizer updates | A finite gradient that is actually applied increments the counter; rejected updates do not |
| Trajectories | 0, 250, 1,000, 2,500, 5,000, 10,000 updates | Diagnostics during the same training trajectory, under one fixed schedule |

The count is **4 × 4 × 2 × 2 × 4 × 2 tracks = 512 training trials**. PD and LGD remain separate tasks and analyses. Configs are `config/experiment1/pd.yaml` and `config/experiment1/lgd.yaml`, run family `cpt_main_v5`.

A partial round at the last update is permitted; visitation counts then differ by at most one. One update is not equal FLOPs, rows or GPU seconds across models or sampling modes. Record processed rows, successful updates, measured training time, monitoring time and peak memory alongside model quality.

### Adaptation is explicit

Full updates optimize every normally trainable parameter. Architectural constants can remain nontrainable: TabICLv2's RoPE frequency parameter is one example.

The frozen arm detects the repeated transformer stack containing the most parameters and freezes that stack, including its normalization affine parameters. It leaves the surrounding input/target embeddings and prediction machinery trainable. For the current architectures this selects TabPFN v3 `icl_blocks`, v2.6 `blocks`, the v2 transformer encoder stack, and TabICLv2 `icl_predictor.tf_icl.blocks`. TabICLv2's column embedder and row interactor remain trainable. The null/pilot audit must confirm the actual names/counts for each loaded checkpoint.

This is **not LoRA**, not only a final linear-head update, and not a literal replication of Rubachev's LayerNorm/head/embedding recipe. The inherited internal flag is still `use_lora`; modern filenames use `_frozen`, and provenance uses `adaptation_mode`. Legacy `_lora`/`_iclhead` names remain readable.

Freezing changes `requires_grad`, never `.training`. TabICLv2 uses different algorithms in training and inference mode. Gradients still propagate through a frozen stack to trainable embeddings, so the frozen parameter fraction does not predict activation-memory savings. Retain measured row caps.

L2-SP is an unnormalized sum, matching the real-table reference form. The same lambda does not give equal effective regularization across architectures, task-loss scales or trainable subsets. Report its measured penalty and relative trainable-weight drift, including drift when lambda is zero; interpret interactions rather than treating lambda as a universal strength.

### Training objectives

PD uses cross-entropy over the dataset's active classes; binary credit tables therefore use two logits. TabPFN regression uses its native bar-distribution negative log-likelihood on context-normalized targets, followed by each member's context-fitted target transform where configured. TabICLv2 regression uses mean pinball loss across 999 native quantiles on normalized targets. Each objective is averaged over query rows and the two preprocessing members, then adds the selected L2-SP penalty. Epoch records keep the pure data loss and penalty separate.

These objectives retain each model family's probabilistic prediction task. They do not directly optimize AUC, thresholded F1 or point RMSE, and no reviewed evidence establishes one universally best credit CPT loss. Reducing a distributional loss need not reduce RMSE. Compare effects within a family and dataset; raw loss magnitudes across families/tasks are not comparable.

The 28-09-2026 audit corrected TabPFN PD's previous loss over all output-head columns. Upstream `TabPFNClassifier.forward` selects active classes before fine-tuning CE and inference softmax, contrary to the old CreditPFN comment. It also corrected regression query targets: members with a fitted target transform previously saw transformed context labels but untransformed z-scored query labels in their loss. Query labels now receive that same already-fitted transform, following upstream `_targets_in_estimator_space`; query rows never fit it. These are optimization changes: earlier TabPFN budget curves remain historical evidence under the old objectives. TabICL losses are unchanged. The main campaign uses separately prepared identities and the subsequent GPU controls recorded in agent memory; earlier pilots must not be relabeled as measurements of the corrected objective.

### Sampling inside a table

The main grid and its seed/null/pilot configurations use **class-balanced batch subsampling for PD**, capped by available rows, then splits that sampled batch into disjoint 60% context and 40% query rows. Despite its inherited name `context_sampling`, this affects **both** context and query prevalence. It is not a context-only intervention. LGD uses uniform row subsampling. Sampling is without replacement within a draw; rows can recur across visits.

This is a documented departure from Garg et al.'s uniform-row recipe. Hold it fixed in the main grid and treat proportional versus balanced sampling as a separate follow-up if calibration or prevalence effects become central. Evaluate probabilities at the natural prevalence of the outer test folds.

Two preprocessing ensemble members are used for training; their objectives are averaged. Categorical/transformation fits that are defined on the context must remain context-only. TabICL's training adapter ordinal-encodes and imputes the sampled episode's features before its internal context/query split, following that upstream training interface; no held-out credit table enters those episodes. Dataset-level sanitization includes data-dependent feature filtering over the full table: the benchmark is consequently transductive with respect to that schema preparation, not a completely inductive preprocessing protocol. An inductive claim requires moving that selection inside the relevant folds. Credit LGD uses a predefined target clipped to [0,1]; this describes bounded loss prediction, not every economic LGD definition including costs or recoveries above exposure.

Finite measurements must not become missing solely because their units exceed float32 range. During schema preparation, numerical columns with maximum finite magnitude above 2**40 are divided by a power of two that brings that maximum below one, before float32 conversion. This positive linear unit change preserves signs, ordering and ratios up to rounding; genuine NaN/Inf values stay missing. Factors are recorded as `numeric_unit_shifts` in the existing dataset manifests and applied to the common data seen by every model. Like feature selection, choosing these factors from the complete table is part of the declared transductive schema preparation, not context-fitted statistical normalization. A rescaled column with an unrepresentable finite dynamic range is rejected for review. TabPFN training then uses the upstream two-pass logarithmic outlier rule, with sample standard deviations and bounds fitted on context rows only; ordinary values inside the bounds remain unchanged.

### Budget and trajectories

The common research budget is **10,000 updates**, increased from the provisional 5,000 after reviewing the eight 20k reference pilots. The extra observation window matters: on the tested schedule, TabICL's mean non-credit classification AUC changes from above baseline at 5k to below it at 10k; several credit and non-credit regression curves also continue moving. Ten thousand is a compute-conscious descriptive horizon, not a convergence claim or a budget selected to maximize credit scores. Deterioration remains part of the result; all completed trials use the same endpoint.

The TabICL PD reversal and LGD curve use objectives unchanged by the subsequent TabPFN corrections. They still motivate the 10k window; old TabPFN trajectories cannot certify the corrected losses' behavior. A repeat of all eight long pilots is not required solely to retain this descriptive horizon, which makes no architecture-wide convergence claim.

The literature establishes no universal update count: Real-TabPFN uses 20k; TabPFN-Wide uses 10k based on its own monitoring; target-table fine-tuning has different stopping rules. [LITERATURE.md](LITERATURE.md) gives the source evidence. These pilots cover one reference recipe, one dataset fold and one training seed per base/task. Their monitored fold is reused in the descriptive main grid, so the horizon choice is pilot-informed rather than independent of the reporting corpus. They do not establish the behavior of every grid recipe. Preserve the 20k pilot trajectories as longer-run evidence; some changes continue after 10k, so the main study cannot claim to characterize eventual convergence or all late forgetting.

AdamW uses zero ordinary weight decay, gradient norm clipping at 1, a 10% warmup and cosine decay to 5% of peak. The complete successful-update budget fixes the schedule once. Recovery restores its position; a resubmission does not restart warmup.

A trajectory point is a measurement of the same evolving weights, not a separately trained model or a new grid trial. For example, the point at update 5,000 of a 10,000-update run has followed the 10,000-update schedule; it is not equivalent to an independently trained 5,000-update run. Likewise, the pilot's 10k point follows a 20k schedule and is not a numerical prediction of the main endpoint. Intermediate weights are transient. Only final weights and the latest optimizer recovery state persist.

Monitoring uses fixed seed **31415**, up to 2,000 rows/table and four inference members. Every table has fixed, disjoint context/validation/query rows (48%/12%/40% before rounding). Thresholds and calibrators use only validation rows. This differs deliberately from the training batch's 60% context/40% query. It records per-table discrimination, calibration and error metrics, plus timing and movement from initialization. Monitoring preserves training RNG and module modes. The final benchmark uses 32 TabPFN or 8 TabICLv2 members and full outer test folds. Keep the two evaluation fidelities separate in plots and text.

A finite but disappointing/flat curve does not stop training. Only numerical failure or inability to reach the budget produces a divergent outcome. Completed numerical failures are retained as outcomes for the descriptive grid instead of repeatedly rerunning them until one succeeds. Infrastructure failures and interruptions are recoverable work, not scientific successes.

## Sampling comparison and training randomness

**Experiment 2 adds 32 seed trials** repeating the predefined full-update reference (LR `3e-7`, lambda `0.003`, `one_sample`) with training seed **43**: one recipe × one additional seed × four bases × four dataset folds × two tasks. Its seed-42 counterpart already exists in experiment 1. Configs are `config/experiment2/{pd,lgd}.yaml`, run family `cpt_seeds_v5`. Keep dataset partitions, monitor/evaluation seeds, row caps and all other settings identical to the main reference.

This is a small paired sensitivity check: did changing the training randomness materially alter the reference behavior? Two seeds do not reliably estimate a seed distribution, and this check says nothing about robustness at every other recipe. Show paired differences and curves; do not count the repetitions as extra independent datasets. A future expansion to 64 additional trials would add seed 44 for the same recipe, only if the initial comparison warrants it.

**Experiment 3**, `config/experiment3/{pd,lgd}.yaml`, defines a separate **96-trial** study at LR `3e-7`, lambda `0.003`, full updates, seed 42: three sampling modes × four bases × four folds × two tasks. **All three modes use proportional PD sampling** (`train.context_sampling: stratified`), with uniform LGD sampling. Covering each row exactly once cannot force balanced class prevalence without dropping or repeating rows. This proportional `one_sample` control is therefore deliberately separate from the main grid's balanced-PD reference; their PD difference is a prevalence-policy change, not a pass-mode effect.

The research campaign contains **512 main + 96 sampling + 32 additional seed trials = 640 trials**. All three experiments initialize from the original base weights; neither experiment 2 nor experiment 3 requires an experiment-1 checkpoint or a selected winner. Their training can overlap after fixing the shared budget, inputs and code. The seed comparison needs the matching seed-42 main references before analysis, not before seed-43 training. Experiment 3 has its own one-sample controls. Use the same pilot-selected budget and trajectory points throughout; measure full-pass/accumulation GPU behavior and cost before launching that grid, because experiment 0's production pilots use one_sample.

| Mode | Rows and update rule | Interpretation |
|---|---|---|
| `one_sample` | One capped draw and update per table visit; rows can recur between visits | Equal table visitation |
| `full_pass` | Partition each table into `ceil(rows / cap)` non-overlapping chunks; update after each chunk | Larger tables receive more optimizer updates |
| `accumulate` | The same partitions, kept together by table; average finite chunk gradients before clipping and update once per table | More rows per update while keeping table-level update balance |

**Protocol 4 makes full passes exhaustive.** For each completed epoch, shuffle and partition each table anew. Chunks differ in size by at most one, stay below the measured row cap and contain both context and query rows. Classification partitions spread each class approximately proportionally; extremely rare classes need not appear in every chunk. Within a chunk, context and query are disjoint. Workers cache one partition per table/epoch, and deterministic indices preserve partitions across worker counts and interruptions. Invalid sampling policies fail rather than silently changing the experiment.

Full-pass and accumulation use identical partitions for a fixed seed/epoch, but their update order, model states and progress per update differ. Accumulation averages **finite chunk gradients**, not one joint forward over the whole table; slightly unequal query counts are not reweighted by row. Every row is assigned once in a completed epoch, not necessarily used successfully: numerical skips remain recorded, and an exact update budget may stop mid-epoch. Report row exposure and compute alongside update counts. Historical pre-protocol-4 `full_pass` used independent draws and did not guarantee coverage; keep those records in the local archive.

## Phase gates before the main run

**Experiment 0 part 1 is one submission command**, implemented by `run_experiment0.sh part1`. Public monitoring inputs download on the login node; CPU preparation verifies and stages all inputs, writes immutable plans and runs preflight. It then releases:

1. **16 zero-LR controls**: four bases × two adaptations × two tasks, two optimizer updates. CPU audits require exact canonical inference-state equality and fixed-monitor parity.
2. **32 positive-LR pilots**: conservative/high LR endpoints × both adaptations × four bases × two tasks, 250 updates, first dataset fold. Record real training, monitoring, memory and preparation costs. These pilots are operational checks, not a recipe-selection exercise.
3. **Eight GPU recovery pairs**: one full-update reference per base/task. Compare 12 uninterrupted updates with stop/save at update 5 and resume to update 12. Report bitwise equality as well as predefined tensor tolerances (`rtol=1e-6`, `atol=1e-7`) and score tolerances (`rtol=1e-5`, `atol=1e-7`). A tolerance pass is not a claim of bitwise CUDA determinism. Each pair also exercises five-fold final scoring, distribution metrics and prediction persistence on its packaged non-credit table. These 16 tiny training arms are additional controls, not main-grid observations.

Only successful completions release a CPU audit, and only a passing audit releases the next stage. No GPU allocation waits for another job. Failure stops progression; the workflow never silently retries numerical failures. An ambiguous submission requires inspection, not another launch.

**Experiment 0 part 2 is separate**, via `run_experiment0.sh part2`: eight full-update reference pilots, one per base/task, through 20k successful updates with 0/250/1k/2.5k/5k/10k/20k measurements. A current part-1 receipt and unchanged input/environment identities are required. These longer jobs use resumable two-hour work segments plus a save/monitor margin. They follow a 20k schedule; their early points are not substitutes for a 5k schedule.

The reviewed pilots inform the common **10k budget and six milestones** in experiments 1–3, recorded in their immutable plans. The epoch safety rail is 4,000 rounds, preserving headroom above the required successful updates. The research design remains **512 main + 32 seed + 96 sampling = 640 trials**, plus **124 control/pilot training arms** if every experiment-0 stage including the auxiliary checks runs once (72 original arms + 52 auxiliary arms). Interrupted execution adds a segment, not a new scientific trial.

### Auxiliary gate for experiments 2 and 3

Run `run_experiment0.sh auxiliary` after local validation, before submitting the seed and sampling studies. Eight GPU tasks (four bases × two tasks) run 52 six-update arms: 36 continuous cases and 16 interrupted/resumed counterparts. Seed 42/43 checks hold the monitor and corpus split seeds fixed. Sampling checks use the experiment-3 stratified policy in every arm; LGD reuses its identical seed-42 one-sample case, while PD needs a separate stratified control.

Only the two smallest eligible training tables and smallest held-out table in partition zero are used, preserving their original roles. Selection uses row count, not outcomes. The 256-row cap forces multiple disjoint chunks; total training rows are capped at 8,192 before GPU submission. Recovery checks stop after update 3, resume to 6 and compare weights, fixed monitor scores and exposure against continuous arms. Every completed arm must pass identity, finite-monitor, exact-budget, zero-skip and positive-parameter-movement checks. The seed-43 endpoint also passes five-fold package-data scoring. A final CPU audit requires all eight task reports.

This is an engineering gate, not a performance or capacity estimate. It neither reruns the historical budget pilots nor changes the main grid. Deterministic small-shape recovery does not certify production-shape speed or bitwise reproducibility. Experiments 2 and 3 start from original weights, so training can overlap experiment 1 after this gate; seed comparison still needs the matching seed-42 reference outputs. Keep row exposure, time and failed updates in the sampling analysis. Slightly unequal chunk query counts are equally weighted as chunk means, not exactly row weighted.

## Retention and measurement protocol

`config/retention.yaml` fixes a research panel of eight non-credit datasets from the curated **TabArena-v0.1 OpenML versions**: Amazon employee access, online shopping intention, QSAR biodegradation and seismic bumps (classification); airfoil noise, concrete strength, protein structure and superconductivity (regression). Null/recovery controls use the small packaged breast-cancer/diabetes pair. Data identifiers, versions, source/license metadata and byte hashes are recorded; no panel table enters the 25-table adaptation corpus.

This panel is not the full TabArena benchmark. We apply CreditPFN's own fixed monitoring and five-fold final evaluation, not TabArena's official benchmark protocol. Prior exposure of base checkpoints to these datasets is not established. Report **change from each base on a declared panel**, not universal retention, benchmark leadership or guaranteed novel-data generalization.

At milestones record trainable parameter names/counts, relative movement of the trainable weights, and per-tensor norm, mean, standard deviation, maximum absolute value, bounded sampled quantiles, anchored absolute/relative movement, cosine similarity and optimizer-moment norms. Unanchored frozen-tensor movement remains unavailable rather than being invented as zero. Epoch records retain data loss, regularization, gradient/clipping/skips, update/row counts and training/data-wait/monitor time. Every 20 seconds record process memory and allocated-device utilization, memory, power and temperature when available. Counters describe sampled device behavior, not exact process energy or billed allocation time.

Final evaluation uses **five outer row folds** on each held-out credit table and each non-credit panel table. Inside each outer training fold, a disjoint 20% validation split supplies classical HPO, maximum-F1 threshold selection and Platt/isotonic calibration. No outer test labels enter those choices. F1 at threshold 0.5 is retained for context. Foundation models use their capped context; classical controls use the full inner training fold. This compares practical native-context settings, not equal row exposure.

Inference precision (decided 27-10-2026): TabICLv2's original and adapted checkpoints are scored in FP32 without FlashAttention 3 in experiments 1, 2, 5 and 8, because the automatic FP16/FA3 policy mis-scored the original checkpoint on some PD tables (experiment-0 numerical review). Experiments 1 and 2 adopt this by re-scoring their existing checkpoints; training is unchanged. Experiments 3, 4, 6 and 7 retain the automatic TabICLv2 policy. TabPFN bases use the package's automatic policy (mixed-precision autocast on GPU) for both weight states throughout.

Logistic and ridge regression use context-fitted one-hot categorical encoding, numerical median imputation and scaling. CatBoost uses native categorical features; the declared XGBoost recipe uses ordinal numeric codes. These are specific tuned controls, not an exhaustive comparison of every preprocessing recipe. Requested tuning requires Optuna. Each result records the requested/completed trial counts and selected parameters, since a time limit can end a search before all requested trials finish. The outer test folds remain untouched by both the search and validation-fitted calibration/threshold selection; reusing validation for those choices can overfit validation, but does not turn it into the reported test score.

Record fold-level discrimination/error scores, confusion counts, prevalence, raw/calibrated proper scores, ECE and reliability-bin sufficient statistics, and fitting versus prediction/scoring time. Compressed row predictions preserve the original dataset row index. Supported regression distributions add 80%/90% interval coverage/width, a fixed-grid mean pinball loss and quantile-crossing rate. Unsupported quantities remain missing. Nine quantiles do not produce an exact CRPS. Untuned/classical scores and predictions are reusable only with matching data/model/config/code/environment identities.

Detailed measurements live on project storage under each experiment. Resource and tensor records consolidate into separate narrow tables so they do not expand the wide trajectory table into a large mostly-empty matrix. Notebook credit and non-credit averages remain separate.

## Reporting and remaining evidence

Experiments 2 and 3 have the same final credit/non-credit benchmark as experiment 1. Experiment 2 compares each seed-43 checkpoint with its predefined seed-42 main reference; experiment 3 compares its three protocols with each other and their original base. Each experiment retains its own results and predictions. Compatible untuned/classical controls can be reused across experiments without changing their evaluation identities; trained checkpoints always require their own evaluation.

The follow-up notebooks pair observations on dataset, training partition, base and recipe before aggregation. Credit datasets contribute their single held-out partition; each non-credit dataset requires all four matched training partitions. Protocol contrasts require all three modes on that same support. Missing counterparts exclude a comparison rather than comparing different surviving cohorts. The seed notebook reports paired differences, not a two-seed variance estimate. Sampling notebooks report row exposure and recorded training time alongside performance; equal updates do not imply equal compute. Training notebooks show dynamics; results notebooks distinguish final five-fold benchmarks from supporting small-context monitor endpoints.

- Show PD AUC changes and LGD fractional RMSE reductions **within dataset**, paired to the same base and row fold. Aggregate row folds/repetitions within dataset before giving datasets equal weight. Do not average raw LGD RMSE across incompatible target scales.
- Show individual datasets and coverage, with failures and missing milestones. Trajectory plots leave a point missing when observations from known baseline trials are absent at that update, rather than silently averaging only survivors.
- Report calibration/log loss/Brier and class-imbalance metrics for PD, RMSE/MAE and supported probabilistic diagnostics for LGD. Existing nine-quantile summaries do not establish an accurate full-distribution CRPS. TabPFN density scores are clipped for numerical stability; report that convention and do not treat unlike density/quantile objectives as interchangeable.
- Do not count training trials, overlapping dataset folds or many CV rows as independent datasets. Any dataset-level bootstrap is descriptive and conditional on this small, dependent corpus; avoid unqualified significance claims or post-hoc winner tests.
- Starting score versus improvement shares the starting score on both axes. Noise, mathematical coupling and ceiling effects can create a trend without a causal adaptation mechanism. Treat that plot as descriptive; do not use its correlation as mechanistic evidence.
- Compare endpoint and trajectory effects for learning rate, anchoring and adaptation, including interactions. Separate step budget, row exposure, monitoring overhead and GPU allocation time.
- A provenance audit of base pretraining and related data sources is still required for contamination/independence claims. The intended starting weights are upstream synthetic-pretrained bases, not Real-TabPFN adapted weights; content identity matters more than filename labels.
- Temporal/grouped validation requires valid origination-time and loan/borrower keys, excluded from features. It is a future sensitivity study, not a property of current IID folds. Historical date-availability observations must be rechecked against the actual raw version.
- A fixed public non-credit retention panel is monitored and evaluated separately as described above. A matched generic-data adaptation control would still be required to isolate a specifically credit-domain effect; it is not part of the current grid.
- Additional baselines such as RealMLP, context-prevalence sensitivity, longer horizons and more lambda levels are optional follow-ups, clearly separated from the main descriptive grid.

## Reporting the descriptive study

The notebook sequence follows the paper's questions: corpus geometry and partitions; engineering
controls and budget evidence; main-grid dynamics and final paired effects; focused seed sensitivity;
sampling, exposure and cost. Coverage accompanies every experiment. DATA-only partition-mean
monitors remain separate from per-dataset project trajectories and five-fold final evaluation.
Corrected short pilots use their saved generated configuration; affected historical budget curves
retain their objective-provenance limitation.

Main results lead with final five-fold effects of CPT choices: matched anchoring, update-scope
and learning-rate contrasts, followed by complete rate × anchoring × scope maps for each metric.
All recipes within a model use the same complete table cohort across the displayed metrics.
Contrasts fix every other setting, average matched recipe differences within table, then weight
tables equally. A favorable choice contrast can mean less deterioration; only a positive effect
relative to original weights indicates an observed CPT gain. Models are never pooled to hide
opposing responses. AUC/Brier axes may rescale changes by 100 for readability, with explicit units;
RMSE gains divide the paired error reduction by original RMSE. Metrics are not averaged together.

Credit/non-credit recipe comparisons, fixed-reference uncertainty/source checks, probability
quality and prediction shifts complete the descriptive account. Table spread is not a confidence
interval over independent seeds. Complete numeric tables retain every plotted value and per-table
positive-effect counts. Test-grid maxima remain exploratory, rather than independently validated
recipe selections. Fixed-context monitor changes appear separately against learning rate, with
equal table weights and distinct legends for rate, anchoring and update scope. Initial-score
scatterplots are omitted because their x-coordinate is constant within a starting model.

Training notebooks retain architecture-specific native objectives and performance trajectories.
Matched monitor-choice effects, combined parameter movement and completed-trial time/memory
summaries replace repeated model-specific diagnostic figures. Detailed native loss/optimizer
contrasts and device telemetry remain in tables. Aggregate trajectory points require every planned
trial and unchanged baseline support; missing milestones remain gaps. Device capacity is not a
predictive result, and recorded training time excludes queue wait.

Calibration compares raw, Platt and isotonic Brier/ECE/log loss on the same complete table cohort
within model; means weight tables equally and quartiles describe table spread. Calibrators and
F1 thresholds are fitted on validation rows. Detailed reliability/threshold plotting helpers remain
available, but the default main notebook favors the compact overview and complete reference
metric tables. Absent calibrated metrics are unavailable rather than imputed. Cross-validation
folds are dependent; their standard deviation divided by the square root of five is not a justified
standard error. Primary methodological support remains in the literature notes.

The final notebook text records every displayed table, plotted value and caption. Figures use the
actual ICML dimensions rather than the generic A4 template; the adjacent manuscript is not rewritten
by the notebook runner. Literature motivation and limits of these presentation choices are recorded
in LITERATURE.md.

## State of the evidence

As of **29-10-2026**, both complementary storage trees have been downloaded locally, including
the completed TabICLv2 FP32 re-score of experiments 1 and 2. All planned research training is
complete: experiments 1/2/3/5/6/7/8 contain 512/32/96/48/16/16/48 completed trials.
Their final benchmarks contain 4,047/627/1,083/627/297/462/627 model–table–partition comparisons,
respectively: **7,770 comparisons and 38,850 successful outer-fold records**. Experiment 4 has
250 completed inference tasks and 11,100 original/CPT context cells. Experiment 7 also reuses
eight experiment-6 controls; these are not additional training runs.

The 29-10 local audit checks all standard metric exports for complete folds, finite primary
metrics, receipt/prediction byte counts and duplicate identities. Every raw result CSV in the
new experiment-1/2 consolidated inventories is present with no additional CSVs; 6,890 recorded
source/snapshot files match their checksums. All 1,083 refreshed TabICL prediction exports
decode (57,925,718 rows), with matching outer test-row/target identities and 29,925 independently
recalculated metric values agreeing within rounding tolerance. Earlier audits covered the
other standard prediction exports, including experiment 8 on 27-10, and all experiment-4
context SQLite tasks for nested contexts, context/validation/test disjointness, matched query
cohorts and reproduced primary scores. Those older audits are distinct from the new re-score.
All 25 executed notebooks have complete ordered summaries and 227 matching inline figures/PDFs
and captions, with no saved execution errors or orphan figures. The current indexes are tracked.
Detailed evidence and historical recoveries remain in `AGENTS_MEMORY.md`; no further benchmark
or training submission is indicated by this audit.

### Current descriptive findings

For the **predefined** full-update, LR `3e-7`, L2-SP `0.003`, seed-42 reference, equal-table
five-fold mean effects against each exact starting checkpoint are:

| Base | PD mean ΔAUC, 17 tables | LGD mean ΔR², 8 tables |
|---|---:|---:|
| TabPFN v2 | +0.004773 | +0.004101 |
| TabPFN v2.6 | −0.000024 | −0.083212 |
| TabPFN v3 | +0.000430 | −0.028429 |
| TabICLv2 | −0.001581 | −0.005485 |

These effects do not establish a universal CPT benefit. TabPFN v2's reference PD AUC improves
on all 17 tables (16 exceed the descriptive 0.1-percentage-point reporting band). TabICL's
FP32 mean and median effects are negative (median −0.000584); six tables have a positive
point estimate, and only one exceeds that reporting band. The former automatic-policy
TabICL mean was +0.003695. The historical **paired inference-precision diagnostic**,
independently rechecked locally on 21-10-2026, explains why those former gains needed re-scoring:

| Diagnostic table | Original / CPT AUC, automatic | Original / CPT AUC, FP32 with FA3 disabled |
|---|---:|---:|
| Vehicle Loan | 0.583062 / 0.654566 | 0.656382 / 0.654633 |
| Myhom | 0.579923 / 0.594749 | 0.594666 / 0.594730 |
| Taiwan Credit Card | 0.790154 / 0.788374 | 0.790129 / 0.788372 |

These are means of five matched outer folds, using identical contexts and complete test
sets in every policy/state comparison. Vehicle Loan's ΔAUC changes from +0.071504 to
−0.001748; Myhom's from +0.014826 to +0.000064. Unchanged-save predictions agree exactly
with the original in all 30 policy/fold checks. The automatic path records FP16 attention
on the native benchmark contexts; the FP32 path records FP32. Both dispatch through
PyTorch SDPA, no direct FA3 call is observed, and TF32 is disabled. The bounded Vehicle Loan
probes also show automatic-policy boundary sensitivity. These observations establish
numerical-policy dependence; they do not identify a particular internal SDPA kernel or
estimate the prevalence across other tables, recipes or models.

The current main and seed-check benchmarks now use matched FP32/no-FA3 TabICL inference
on every credit and retention table, rather than the former automatic-policy results.
The diagnostic remains an appendix qualification, not a prevalence estimate across all
models. Experiments 3, 4, 6 and 7 retain their recorded automatic TabICL inference policy;
their within-study comparisons must not be interpreted as isolated contrasts against
the FP32 studies. Keep each study's own original-model control.

Across the balanced grid, larger learning rates generally hurt and freezing
or anchoring attenuates degradation; this is a response over the specified recipes, not a
test-selected deployment recommendation. Calibration and discrimination need separate
interpretation: for example, the current reference TabICL mean raw Brier reduction is
−0.000845, and its corresponding Platt-calibrated reduction is −0.000808.

- **Experiment 2:** mean seed-43 minus seed-42 PD AUC changes range from −0.000260 to −0.000017
  across bases. LGD is less uniformly stable: TabPFN v3's mean ΔR² is +0.019034, driven mainly
  by one table (maximum +0.147777, median +0.000018). Two seeds cannot estimate general variance.
- **Experiment 3:** revisiting/accumulating more rows has no consistent predictive advantage.
  For PD, median recorded accumulation cost is approximately 7–17 times one-sample training
  across bases at the same successful-update budget. More updates, more rows and more compute
  must not be conflated.
- **Experiment 4:** labeled context strongly affects performance; CPT is a much smaller,
  family- and context-dependent change. On the fixed nine-table PD cohort, 50,000-row mean
  ΔAUC is +0.006051 for TabICLv2 and +0.000004 for TabPFN v3. The latter has +0.002478 at eight
  rows on that same cohort. Do not compare absolute levels across different horizon cohorts.
- **Experiment 5:** the conservative adapter rate stays closest to the original; `0.03` hurts
  mean credit AUC/R² for both families. Restricting trainable parameters alone does not prevent
  deterioration. Its TabICL FP32/no-FA3 inference requires its own matched original controls.
- **Experiment 6:** excluding small training tables has no consistent advantage: PD mean ΔAUC
  changes from +0.003572 to +0.002057 for TabICLv2 and from −0.001003 to −0.000776 for TabPFN v3.
- **Experiment 7:** larger episodes are not uniformly better. TabPFN v3 mean PD ΔAUC is
  −0.004316/−0.000776/−0.000114 for 2,048/8,192/26,000 rows; TabICL changes less and nonmonotonically.
  Episode size also changes query count and cost, and the 8,192-row reference is shared.

The research retention panel contains **eight non-credit tables**, four per task; the two
packaged smoke datasets are separate debugging controls. This is not broad evidence about
all out-of-domain tasks. Dataset/population overlap, corpus-level transductive schema selection,
IID row folds, two training seeds and exploratory hypotheses remain limitations; passing local
checks does not eliminate them. The bounded precision, source/partition, LGD-distribution
and table-uncertainty checks are complete. Draft the paper around the supported conditional
findings; no additional run is required to begin writing. A claim of precision-independent
TabICL gains across the full corpus would require a broader matched FP32 evaluation.

The notes below distinguish the earlier, invalid pre-redesign evidence.

On 22-09-2026 the user reported no running VSC jobs. The old experiment has an invalid swept L2-SP axis and defects in worker sampling/accumulation. Retagging 338 surviving checkpoints recovered their lambda-0.003 identity, not a corrected training history. The user reported 412 checkpoint files on project storage; that is not a verified completed-trial count.

Historical headline measurements, including the completed run-8 evaluation, remain in the agent-memory runs table as historical observations. They are not results of this redesigned experiment and cannot establish its conclusions. A clean rerun means new training and evaluation under new phase names; it does not require deleting the raw corpus or redownloading validated base weights.

The current implementation uses training protocol **5**. Debugging outcomes are recorded in the agent-memory runs table; they are not results of the main scientific grid. Local tests exercise sampling, finite mean gradients, exact budgets, recovery, cleanup boundaries, identity checks and analysis. Accepted cluster controls, the pilot budget decision, real CUDA recovery and measured walltimes are prerequisites for the full campaign. Preparation logs establish that inputs and plans were written; they are not completed training or evaluation results.

A passing experiment 0 establishes the tested controls, not the capacity of every final evaluation task. Before scheduling all final benchmarks, measure a representative large-table evaluation with the intended inference ensemble/context cap and a classical HPO task. Training-cap pilots and the small packaged recovery benchmark do not validate million-row inference or its walltime.


## Follow-up experiments 4–7

These implementations require new experiment-0 validation receipts before research submission.
Accepted validation receipts and subsequent VSC outcomes are recorded in `AGENTS_MEMORY.md`.
Experiments 1–3 retain their configurations and identities.
The earlier generic-data control and prevalence proposal are excluded. The remaining designs
are renumbered consecutively as follows; historical dated records retain their original numbers.

| Experiment | Fixed question | Design |
|---|---|---|
| 4: labeled context | Does credit CPT change the learning curve as labeled context becomes scarce? | Original versus fixed CPT, TabPFN v3 and TabICLv2, 25 tables, five outer folds, 29 nested context sizes; no new CPT. |
| 5: DiagScale CPT | Can small query-scaling adapters produce useful cross-table adaptation? | Two modern bases × three rates × four dataset partitions × PD/LGD = 48 training trials. |
| 6: training-corpus policy | Does excluding small training tables change transfer at a fixed episode cap? | Two modern bases × four partitions × all/large-only training tables = 16 PD trials, capped at 8,192 rows. |
| 7: episode size | Does a larger training episode change transfer on an identical large-table corpus? | Two modern bases × four partitions × three episode sizes = 24 PD comparisons: 16 new trials and eight shared controls from experiment 6. |

### Experiment 4: context learning curves

The fixed CPT reference is experiment 1 (`cpt_main_v5`), full-parameter adaptation,
peak LR `3e-7`, L2-SP `0.003`, seed 42 and the 10,000-update endpoint. Each table uses
only the checkpoint whose training partition excluded it. This gives eight reference
checkpoints per task (two families × four partitions), not a checkpoint selected by its
held-out score. Original-weight, dataset and saved-provenance hashes are checked before use.
TabPFN uses explicit `tabpfn-v3-*-v3_default.ckpt` paths; package default-version selection
is never used. TabICL uses the existing dated v2 checkpoints.

Contexts contain 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1,024,
1,536, 2,048, 3,072, 4,096, 6,144, 8,192, 12,288, 16,384, 20,480, 24,576, 32,768,
40,960, 49,152 and 50,000 rows when available. Include the
available-pool endpoint up to the declared cap, deduplicating coincident sizes. Never replace
missing rows by duplicates. The shared 50,000-row maximum is a compute cap for this study,
not v3's architectural limit: both pinned local v3 checkpoints declare a one-million-row
inference limit and no internal row subsampling. Keep the package's size guard enabled.
The upper end still requires the separate GPU capacity gate at the configured inference-view
count. Classification prefixes preserve class coverage and approximately
proportional prevalence; their order uses context-pool labels only. Regression prefixes are random.

Five outer folds cover every test row exactly once. The outer-training fold is divided into a
context pool and a fixed 20% inner-validation subset for threshold/calibration measurements.
Thus the x-axis counts **context labels**, not total labels acquired: validation labels are an
additional fixed resource. Each context budget and both weight states use exactly the same
validation/test rows. All test rows are predicted in memory-bounded chunks. No test-row cap,
inference-view sweep, target-table weight training or held-out checkpoint selection is involved.
Use 32 TabPFN and eight TabICL inference views throughout. Numerical/categorical preprocessing
fits the selected context, retaining the existing corpus-level sanitization limitation.

One task per dataset/family/fold gives 250 tasks. Each task commits its metrics and compressed
row-index/prediction arrays to one project-storage SQLite file and exports a compact metrics CSV.
Interrupted cells can be retried without repeating committed cells. The experiment asks whether
CPT substitutes for some labeled context; interpret the paired AUC/R² and proper-score changes,
not only the highest-context endpoint. Large-context aggregate curves require a common dataset
cohort or explicit support counts, because small tables cannot populate the rightmost points.

### Experiment 5: restricted query-scale adaptation

Use the TTC paper's DiagScale mechanism, not its complete target-table fine-tuning protocol:
TabPFN wraps `SoftmaxScalingMLP` after the original transform; TabICL scales final queries after
SSMax on fresh and cached paths. Learn FP32 zero-initialized multipliers `1 + delta`. Store only
adapter state plus an immutable original-weight reference. Only adapter tensors enter the
optimizer/clipping operation. TabICL retains original backbone autograd flags for CUDA parity,
clears their gradients, and never steps those weights.

The rate grid is `1e-4`, `1e-3`, `3e-2`. The additional `1e-4` is a conservative cross-table
hypothesis; the source paper does not establish an optimal rate for this regime. AdamW uses
weight decay 0.01 on zero-centered deltas, with original-weight L2-SP disabled. Training is FP32,
with a provisional 8,192 total context-plus-query row cap, two training views, proportional
one-sample table visitation, 60/40 context/query and 10,000 successful updates. Use the existing
warmup/cosine schedule, four dataset partitions, seed 42 and fixed credit/non-credit monitors.
No validation-best epoch selection or fallback to original weights is used in cross-table CPT.

TabICL evaluation fixes `use_amp=False` and `use_fa3=False` for **both base and adapter**.
The benchmark uses the existing five-fold metrics, calibration and prediction writers within
experiment 5. Comparing against the base measures this adapter recipe. Cross-experiment
comparisons with full/frozen CPT also change rate, regularization, precision, sampling or episode
size; they are descriptive recipe comparisons, not a pure causal effect of parameter count.
The FP32 episode cap and deployment precision need the new capacity gate; old BF16 tests do not
validate them. Do not describe PEFT as computationally free based on optimizer parameter count.

### Experiment 6: training-corpus policy at a fixed episode cap

First partition the **complete 17-table PD registry**, with the existing seed 1729 and four
balanced whole-dataset folds. For each fold, exclude its test tables, then compare the full
training complement against its subset with `n_rows >= 10001`. Never filter the test set or
redraw a fold after eligibility changes. Every PD dataset, including every small dataset,
therefore remains a test dataset exactly once. These corpus partitions are distinct from the
five row-level CV folds used later within each held-out table.

Use TabPFN v3 and TabICLv2, full updates, LR `3e-7`, L2-SP `0.003`, seed 42, proportional
one-sample visitation, two training views, 60/40 context/query and 10,000 successful updates.
Fix the total episode cap at **8,192 rows** in both arms. A traversal visits each eligible table
once in random order; one sampled episode gives one optimizer update. Small tables use their
available rows without padding or duplicates. This gives 2 bases × 2 corpus policies × 4 folds
= **16 new trials**, run `cpt_corpus_v1` in `config/experiment6/pd.yaml`.

The local processed corpus gives full/large training counts **13/9, 13/8, 12/7, 13/9**;
held-out counts remain **4, 4, 5, 4**. All currently eligible large tables exceed 26,000 rows.
These are local inventory checks, not evidence about VSC's exact bytes: cluster preparation
records and validates its own membership and row counts. LGD is excluded because its
large-only corpus is empty in one partition. Require at least three eligible training tables
per partition, preserve the complete registry and reject explicit dataset/split overrides.

The estimand is the **effect of this training-corpus policy**, not the isolated causal effect
of table size. Removing small tables also changes identities, diversity, revisit frequency
and actual rows per update. Record each table's available rows, episode/query rows and visit
share in the prepared plan, plus actual exposure and cost during training. This limitation
cannot be removed simply by naming the study a large-data ablation.

### Experiment 7: episode size on one fixed large-table corpus

Keep experiment 6's large-only training membership, test assignments, bases, recipe, seed,
update budget and schedule. Vary **total context-plus-query rows**: 2,048, 8,192 or 26,000.
With the same 60/40 split, both labeled context and loss-evaluated query counts increase.
This tests the training-episode budget, not context length alone or exposure at equal GPU time.
Experiment 4 separately varies inference context while keeping the complete test sets fixed.

Reuse the eight 8,192-row large-arm trials of experiment 6 as the shared control. Train only
the 2,048- and 26,000-row levels: 2 bases × 2 new sizes × 4 folds = **16 additional trials**.
The configurations are `config/experiment7/pd_rows{2048,26000}.yaml`, with run names
`cpt_episode_rows2048_v1` and `cpt_episode_rows26000_v1`. There are 24 analyzed cells in this
experiment but only **32 distinct new training trials across experiments 6 and 7**. The
previous, unrun 48-cell factorial design is retired. The smaller design does not estimate
corpus-policy × episode-size interactions.

Prepare experiment 6 first. Experiment 7 binds its reference plan hash and exact trial mapping;
data bytes/membership, base-weight hashes, code, environment, seed, update budget, schedule,
sampling and evaluation settings must match. Resolved benchmark defaults, implementation and
package versions are frozen too, checked again inside each evaluation task; CLI overrides
cannot change inference views, context caps or folds. Only the declared episode cap changes in the
matched large-arm trials. Reference files stay in experiment 6; no copying, hidden rerunning
or reuse of differently configured experiment-1/3 checkpoints occurs. Cleanup refuses to
delete experiment 6 while a saved experiment-7 plan still depends on it.

Every eligible training table must supply every requested episode size. If a future dataset
version cannot fill 26,000 rows, preparation fails rather than dropping another table,
padding rows or changing test assignments. Runtime member/cell caps must not silently shorten
the declared episode. The 26k cap is the existing measured ceiling for these bases, extending
Real-TabPFN's 20k recipe; it is not a literal reproduction of that paper. Its safety under the
new recipe still requires the experiment-0 capacity gate.

Both studies monitor updates 0/250/1k/2.5k/5k/10k and use the same final five-fold credit and
non-credit benchmark with original/classical controls. Train and benchmark phases remain
separate. After all plans are prepared, their new training trials may run concurrently;
experiment-7 benchmarking additionally requires experiment-6 training completion. Analysis
needs both studies' results and all three matched arms, with complete paired folds. Report
AUC/proper-score effects alongside rows, queries, GPU allocation time, memory and throughput;
equal successful updates do not equalize compute or unique-row exposure.

### Validation and publication scope

The isolated experiment-0 follow-up workflow first checks all four base/task combinations:
zero-adapter identity, original-weight immutability, nonzero save/reload, cached prediction,
three learning rates and uninterrupted versus interrupted/resumed updates. A tiny synthetic
five-fold context test loads the actual original/CPT reference pair and verifies complete test
coverage. A separate capacity array tests FP32 adapter episodes, both corpus policies at 8,192
rows, the other two sizes on the identical large-table pool, and long-context inference. Its
episode checks verify recorded row exposure as well as completion. Eight successful, current
code/config/environment receipts with unchanged detailed evidence are required before experiments 4–7 can launch. A failed workflow
can be cleaned by its identifier without deleting earlier experiment-0 controls or research runs.

The main manuscript should emphasize the factorial behavior, credit/non-credit trade-off and
cost. Follow-up curves and mechanism/corpus sensitivities mainly belong in the appendix. Finish
matched benchmarks before drawing conclusions from missing cells. Treat tables as the units of
comparison and CV folds/configurations as dependent observations. Audit shared source populations,
feature timing and temporal/grouped applicability; the corpus is not 25 independent lending
populations. Without the excluded generic-data adaptation control, avoid a causal claim that
credit-domain specialization itself is responsible. Retain the documented transductive schema
sanitization limitation. None of the proposed extra experiments removes these limitations.

### Manuscript and reproducible release

#### Final scope and interpretation checks — 21-10-2026

Keep experiments 1–7; no new main experiment is added. The final inference check uses
Vehicle Loan, Myhom and Taiwan Credit Card, each with five outer folds and the fixed
experiment-1 reference checkpoint. Original, unchanged-save and CPT weights are compared
under automatic settings and FP32 tensors/no FA3 on identical native benchmark contexts
and all test rows. One Vehicle Loan fold additionally probes 1,023/1,024/10,239/10,240
context rows on a fixed 256-row query subset, with an AMP/no-FA3 arm. These small probes
diagnose a numerical boundary; they are not a replacement context curve or prevalence estimate.
Actual backend calls, dtypes, TF32 settings, data/weight hashes and per-row predictions are
recorded. All 15 tasks and the independent wICE audit pass. The downloaded 15 journals
match their receipts and independently reproduce all 114 cells (90 full-test comparisons
plus 24 bounded probes), including 1,627,068 saved prediction rows. This count includes
repeated evaluation of the same queries, not independent observations. The controlled
results and their interpretation are reported above; no save/reload defect was found.

The user fixed the publication scope on 21-10-2026: keep the detailed precision diagnostic
in the appendix and defer a broader precision investigation to a potential future paper.
The main results still need a brief cross-reference identifying the numerical-policy
dependence of the large TabICL gains. No additional experiment is adopted for this paper.
Ning et al.'s Appendix A.1 already documents related inference-setting differences in a
frozen TabICL baseline; this is not an established first-of-kind precision discovery.

The main results notebooks add 95% conditional percentile table-bootstrap intervals and
leave-one-table-out mean ranges for the predefined reference. Five folds are averaged
within a table, never counted as five independent replications. Shared CPT corpora and
unresolved population dependence limit the inferential interpretation; these intervals
do not include training-seed uncertainty. A documented-public-source sensitivity changes
the population summarized and is not a contamination correction.

The source ledger identifies the documented LendingClub and SBA PD/LGD source links,
checks original plan memberships and exact file hashes, and leaves restricted/local
source independence unresolved. Separate PD/LGD training prevents those cross-task twins
from being same-track training/holdout overlap. Different byte hashes cannot rule out
partially overlapping borrowers, transformed copies or base-pretraining contamination.

For TabPFN v2.6 LGD, inspect R²/RMSE together with pinball loss, coverage, width and the
historical **clipped** log score. Saved predictions support separate errors at targets
zero, one and the interior. The log-score implementation replaces nonfinite per-row NLL
and clips at ±100 nats, including finite extremes. Its improvement is not evidence that
uncapped likelihood improved; the original logits needed to resolve that question were
not exported. The paper can report the point/quantile deterioration without claiming a
validated density-improvement mechanism.

After this bounded check, write the paper around observed adaptation benefits, harms,
context dependence and cost. Generic-data CPT, target-table fine-tuning, temporal
evaluation, additional seeds and new hyperparameter grids remain outside this paper.

The number of experiments is not a publication criterion. The contribution should be
a supported account of when cross-table adaptation helps or harms, how this depends on labeled
context and adaptation capacity, and whether corpus/episode size explains the behavior. A negative
average alone is insufficient evidence of general impossibility. Describe the chosen training
horizon, evaluated corpus and inference protocol explicitly; avoid selecting a favorable recipe
from the reported test grid and presenting it as an independently validated winner.

Before drafting conclusions, reconcile every planned trial and benchmark cell with its actual
outcome, preserve failed/unfinished cells in coverage, and quantify paired effects at the table
level. Five CV folds, recipes and overlapping training partitions are not independent samples.
Audit related dataset populations and base-pretraining overlap; report uncertainty and robustness
to source grouping where that audit identifies dependence. The limited two-seed reference and
small non-credit panel bound the claims even if every planned run finishes.

Sharpening the contribution means organizing completed evidence around a few answerable claims,
not adding experiments until an average becomes positive:

| Question for the main text | Required evidence | Interpretation boundary |
|---|---|---|
| How do the adaptation knobs change transfer, retention and cost? | Complete experiment-1 paired benchmarks; prespecified factor contrasts; trajectories and failure coverage; experiments 2/3 as sensitivity checks | Describe this grid, corpus and horizon; do not promote the best tested recipe into an independently selected winner. |
| Does CPT help when labeled context is scarce? | Experiment-4 paired curves with unchanged test rows and explicit dataset support at each context size | The fixed validation-label budget and inference views remain part of the evaluated protocol. |
| What explains the observed response? | Adapter recipe (5), corpus policy (6) and episode budget (7), each compared with its matched control | Different recipes do not isolate only parameter count; corpus filtering changes composition; episode growth changes query exposure and compute too. |

Before inspecting these new outcomes, keep the endpoint, reference arms and signs fixed: large
minus all for experiment 6, and 2,048/26,000 minus 8,192 for experiment 7. Present paired AUC
changes and Brier improvements, the table distribution, non-credit retention and resource costs.
Show table/source-aware uncertainty rather than treating 5 CV folds or 4 overlapping training
partitions as independent replications. A statistically inconclusive difference is not evidence
of equivalence; an eventual negligible-effect claim needs an explicit, justified practical margin.
Do not collapse unlike task metrics or unequal table support into one favorable headline.

The current scope is broad enough to support a focused empirical paper if the completed results
yield defensible insight. Prioritize completeness, provenance/source audit and synthesis over
new grids. For example, a corpus-policy benefit without an episode-size benefit points to the
eligibility/composition policy at this budget; the converse points to the episode budget on this
fixed large-table corpus. Neither identifies a universal mechanism. If both are inconclusive,
report the detectable effect range and tested limits rather than claiming CPT never helps.

The verified [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
allow eight main-text pages at submission (nine for camera-ready), with references/appendices
outside that limit; they require the official LaTeX format, anonymous submission and anonymous
supplementary code. Reviewers need not read appendices, so the central design, main evidence and
limitations belong in the main text. The [reviewer guidance](https://icml.cc/Conferences/2026/ReviewerInstructions)
explicitly recognizes insight from evaluating existing methods as originality. These are 2026
rules checked on 13-10-2026, not confirmation of the next edition's dates or requirements.

Use the existing adjacent paper project: introduce the question and related work, specify the
dataset-level transfer design, present the main response/retention/cost results and whichever
follow-up changes the interpretation, then discuss limits. Detailed seed, sampling, capacity and
secondary-metric evidence can occupy the appendix. Freeze a reproducible artifact with exact
configs, checkpoint/data identities, environment, code/library pins, aggregate results, captions
and regeneration commands. Audit private aliases, paths and data/weight redistribution terms
before preparing the anonymous submission copy. Keep proprietary rows and unlicensed weights
out of the public release. Include a credit-risk-specific impact discussion (miscalibration,
fairness, privacy and inappropriate deployment), rather than claiming these benchmarks validate
automated lending decisions. Authors remain responsible for validating all analysis and text.


## Experiment 8: low-rank query/value continued pretraining

The completed five-fold benchmark compares original and LoRA weights on the same test rows.
Equal-table credit mean ΔAUC (LoRA minus original, 17 PD tables) for TabPFN v3 / TabICLv2 is
-0.000371 / -0.003237 at 1e-5, -0.009511 / -0.010994 at 1e-4 and -0.024917 / -0.039655 at
1e-3. Equal-table credit mean ΔR² (eight LGD tables) is -0.024210 / -0.032969, -0.184804 /
-0.127327 and -0.192724 / -0.149143, respectively. At both higher rates every credit table
loses primary-metric performance for both models; low-rate outcomes vary by table. The four
non-credit tables per task also show lower mean primary scores at both higher rates. These are
descriptive, single-training-seed results for this Q/V rank-16 recipe, not a universal claim
about LoRA or an independently validated rate selection.

All 627 result/prediction/receipt triplets were read locally, with matched row/target identities
and reproduced primary metrics across 3,135 folds. One non-credit LGD table generated inverse
power-transform overflow warnings for original and LoRA TabPFN. Its saved point predictions and
quantiles are finite, ordered and reproduce the recorded scores; its one capped likelihood value
remains a reported numerical limitation, and an overflow-free counterfactual was not measured.

This follow-up asks whether structured additive changes inside the pretrained attention
layers can produce useful cross-table credit adaptation. It was adopted after experiments
1–7 were observed and is explicitly exploratory. It does not replace their grid or outputs.

For each targeted weight, `W = W0 + (alpha/r) B A`. `W0` is fixed; A is randomly initialized
and B starts at zero, so the initial effective weight equals the original. Rank `r=16`
limits the update to at most sixteen independent directions per projection, not sixteen
trainable parameters. `alpha=32` gives multiplier two. Dropout is zero, making the
weight-parametrized and native merged computations equivalent. Only learning rate varies:
`1e-5`, `1e-4`, `1e-3`; these are a prospective logarithmic sensitivity grid, not a
literature-established optimum or rates selected using the held-out test results.

Both TabPFN v3 and TabICLv2 adapt query and value projections in all three transformer
stages. Keys, attention output projections, feed-forward weights, embeddings, biases and
prediction heads stay fixed. TabPFN's label-valued retrieval decoder has no learned V
projection and is excluded. This conventional Q/V restriction is not the all-linear LoRA
recipe of Ning et al. The question is about this stated update family; a negative result
would not rule out other ranks or target modules. The registered adapter inventory records
every module, weight, shape and Q/V slice; unsupported layouts fail before training.

There are two families × two tasks × four existing dataset partitions × three rates =
**48 training trials**. They retain seed 42, 10,000 successful optimizer updates, the
existing training losses, per-family measured row/cell caps, two training ensemble members,
40% query rows, one-sample scheduling, warmup/cosine schedule and AMP policy. PD uses the
main grid's balanced contexts; LGD uses its stratified policy. Neither heads nor base
weights enter the optimizer. L2-SP and AdamW weight decay are zero: penalizing A/B is not
an effective-weight-displacement penalty. Such a penalty is possible but is outside this
experiment. Rank, scaling, dropout and target modules are fixed, not extra swept axes.
Zero factor decay is a design choice, not a claim that weight decay is invalid for LoRA:
Ning et al.'s target-table LoRA recipe uses AdamW decay 0.01. Experiment 8 is not a
controlled LoRA-versus-DiagScale comparison. BF16 forward operations can round small
updates, but how long this affects a particular run is unmeasured; it cannot be inferred
from the learning rate alone or dismissed because experiment 1 also uses AMP.

The five-fold benchmark pairs original and adapted predictions on the same held-out table
rows. Keep AUC/Brier/log loss for PD and R²/RMSE for LGD, equal table weighting, all three
rates, non-credit monitoring, and measured cost. Do not select a best test recipe and label
its score an unbiased generalization result. TabICL original and adapted inference both
use FP32 and disable FA3 to avoid the already measured automatic-precision confound. This
also applies to the fixed-context monitors. TabPFN stays explicitly pinned to v3 weights;
no constructor may silently choose a newer default. Training AMP remains a distinct policy
from inference precision. Experimental results stay separate from experiment 1 and from
experiment 5, whose optimizer, sampling and precision settings differ. Cross-experiment
comparisons need a verified common protocol and cohort, not just similar filenames.

Training reports show native loss, optimization health, held-out/retention trajectories,
cost and the norm of the effective Q/V update relative to its original targeted weights.
This movement is invariant to reciprocal rescaling of A and B; factor-space distance is
not described as backbone drift. Raw per-tensor diagnostics remain parameter/factor statistics:
zero-initialized B has undefined relative change/cosine to its initial value. These are not
per-layer effective-weight measurements. Results reports show paired final changes and coverage
with complete numerical summaries. Empty research output produces an explicit pending
report. No artificial measurements are supplied to publication notebooks.

Implementation uses PyTorch weight parametrizations: TabICL reads packed QKV tensors
functionally, so wrapping a Linear.forward call alone would miss its actual computation.
Native exports merge the update into a new state dictionary without changing the live
adapter, optimizer or random state. Recovery preserves unmerged factors and optimizer,
scheduler, scaler, RNG and sampling progress. LoRA reduces optimizer state, but does not
remove backbone activation/backpropagation costs; no speed or memory saving is assumed.

The separate experiment-0 LoRA workflow checks each family/task at every production rate:
identity insertion, frozen base weights, actual adapter gradients, nonzero predictions,
fresh/cached attention, merged native loading, and uninterrupted/interrupted recovery.
Every Q and V projection is observed during the actual debug training pipeline. The audit
requires finite gradient paths to A and B, a nonzero B gradient and nonzero effective BA
movement for each projection. A's initially zero gradient is expected when B starts at zero.
This catches a dead packed-QKV slice without confusing FP32 export rounding with an absent
training gradient. Hooks only observe gradients and are never installed in production runs.
After the `lora1` LGD batch-splitting failure, the query-chunk acceptance bound was changed
to 1e-4 synthetic context-target standard deviations; same-batch identity, merge/cache and
recovery criteria stayed unchanged. This is a post-diagnostic engineering tolerance,
not a preregistered scientific result. `lora1` remains failed and a fresh gate is required;
[VSC.md](VSC.md#experiment-8-lora-workflow) records the evidence and exact scope.
Production-shape probes visit all four dataset partitions with their actual feature widths.
An independent inspection binds receipts to code, environment, configurations, bases and
data before production launch. Local checks include real tiny TabICLv2 models; the local
TabPFN installation lacks v3, so exact v3 execution is a required VSC gate, not a local claim.
