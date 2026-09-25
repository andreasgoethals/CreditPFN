# CreditPFN — research brief

This document is intended to stand alone when shared with a collaborator or uploaded into another chat. Operational commands live in [VSC.md](VSC.md), primary evidence in [LITERATURE.md](LITERATURE.md), and historical runs in [AGENTS_MEMORY.md](AGENTS_MEMORY.md).

## Question and intended contribution

**How does continued pretraining on a small credit-domain corpus change the behavior of tabular foundation models?** The aim is a descriptive account of learning rate, anchoring, adaptation scope and training progress across architectures and tasks. Improvements, degradation, negligible movement and numerical instability are all possible findings.

The primary object is the response over the specified grid, not a winning model. Describe effect sizes, trajectories, variation across datasets and compute cost. The maximum observed score is an exploratory result; it is not an unbiased estimate for a selected deployment recipe. The current design does not need a separate selection corpus to describe its prespecified grid. A later champion claim, a redesigned grid chosen from these results, or a deployment recommendation needs fresh evaluation or explicitly exploratory wording.

Compare each adapted model with its exact starting checkpoint. The paper can establish behavior on this corpus and budget; it cannot establish a universal benefit/null effect, domain specificity versus generic real-data adaptation, or absence of forgetting without the corresponding controls.

## Corpus, partitions and evaluation population

- **25 registered tables: 17 PD classification and 8 LGD regression.** Every registered processed file must exist before a corrected run starts. Keep the corpus fixed throughout the campaign.
- Four dataset folds, partition seed **1729**. PD held-out counts are **4, 4, 5, 4**; LGD counts are **2, 2, 2, 2**. Training uses the complement: 12/13 PD tables or 6 LGD tables. Balance means table count, not equal rows, default rates, countries or institutions.
- Every dataset is held out exactly once per recipe/training seed. It may be a training dataset in the other folds. The four fitted models overlap substantially in their training corpora; they are not four independent statistical replications.
- Distinct dataset names do not establish independent populations. Related releases, loans, borrowers and institutions require a source-level grouping audit before stronger independence or out-of-time claims. No automatic cross-dataset deduplication system is reinstated.
- Held-out dataset evaluation currently uses five row-level outer CV folds, plus an inner validation split for classical tuning, decision thresholds and calibration. Evaluation seed **99** is fixed across training seeds. These are IID table benchmarks; they do not simulate future origination cohorts.
- All outer test rows are scored. Inference context caps depend on the base, but are identical for that base's trained and untuned arms. Cross-base rankings therefore combine architecture, prior and native context capacity.

Only schema-compatible, legally usable data belong in the corpus. Raw private data, names, contents and credentials do not belong in the paper or Git. Figures use the private display-name mapping; the temporary tracked preprocessing-slug exception is a source-code exception, not permission to publish those names in results.

## Experiment organization

Experiment **0** owns debugging and readiness: null controls, positive-LR/recovery checks, short timing pilots and long budget pilots. Experiment **1** is the full descriptive sweep. Experiment **2** repeats one predefined reference at an additional training seed. Experiment **3** compares one-sample, full-pass and accumulated-gradient training. The general notebooks describe the corpus independently of these outcomes.

Recovery equivalence uses strict deterministic execution and a 2,048-row training cap to separate serialization correctness from GPU arithmetic variation. Null/short/budget pilots and the research grid retain production row caps and fast kernels. A fixed seed controls stochastic inputs; it does not guarantee bitwise equality on nondeterministic GPU kernels. Numerical execution settings are recorded. Small recovery controls establish state restoration under their tested conditions, not capacity or cross-platform reproducibility.

Configs are grouped under `config/experiment0/` through `config/experiment3/`; notebooks use the same groups plus `00_general/`. All current run families use `cpt_*_v5`, including `cpt_recovery_v5`. This is a fresh protocol: it adds disjoint validation rows for monitoring, a fixed public retention panel and experiment-scoped output. Earlier v4 controls remain historical evidence, not passing controls for v5.

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
| Budget | Provisionally 5,000 successful optimizer updates | A finite gradient that is actually applied increments the counter; rejected updates do not |
| Trajectories | 0, 250, 1,000, 2,500, 5,000 updates | Diagnostics during the same training trajectory, under one fixed schedule |

The count is **4 × 4 × 2 × 2 × 4 × 2 tracks = 512 training trials**. PD and LGD remain separate tasks and analyses. Configs are `config/experiment1/pd.yaml` and `config/experiment1/lgd.yaml`, run family `cpt_main_v5`.

A partial round at the last update is permitted; visitation counts then differ by at most one. One update is not equal FLOPs, rows or GPU seconds across models or sampling modes. Record processed rows, successful updates, measured training time, monitoring time and peak memory alongside model quality.

### Adaptation is explicit

Full updates optimize every normally trainable parameter. Architectural constants can remain nontrainable: TabICLv2's RoPE frequency parameter is one example.

The frozen arm detects the repeated transformer stack containing the most parameters and freezes that stack, including its normalization affine parameters. It leaves the surrounding input/target embeddings and prediction machinery trainable. For the current architectures this selects TabPFN v3 `icl_blocks`, v2.6 `blocks`, the v2 transformer encoder stack, and TabICLv2 `icl_predictor.tf_icl.blocks`. TabICLv2's column embedder and row interactor remain trainable. The null/pilot audit must confirm the actual names/counts for each loaded checkpoint.

This is **not LoRA**, not only a final linear-head update, and not a literal replication of Rubachev's LayerNorm/head/embedding recipe. The inherited internal flag is still `use_lora`; modern filenames use `_frozen`, and provenance uses `adaptation_mode`. Legacy `_lora`/`_iclhead` names remain readable.

Freezing changes `requires_grad`, never `.training`. TabICLv2 uses different algorithms in training and inference mode. Gradients still propagate through a frozen stack to trainable embeddings, so the frozen parameter fraction does not predict activation-memory savings. Retain measured row caps.

L2-SP is an unnormalized sum, matching the real-table reference form. The same lambda does not give equal effective regularization across architectures, task-loss scales or trainable subsets. Report its measured penalty and relative trainable-weight drift, including drift when lambda is zero; interpret interactions rather than treating lambda as a universal strength.

### Sampling inside a table

The main grid and its seed/null/pilot configurations use **class-balanced batch subsampling for PD**, capped by available rows, then splits that sampled batch into disjoint 60% context and 40% query rows. Despite its inherited name `context_sampling`, this affects **both** context and query prevalence. It is not a context-only intervention. LGD uses uniform row subsampling. Sampling is without replacement within a draw; rows can recur across visits.

This is a documented departure from Garg et al.'s uniform-row recipe. Hold it fixed in the main grid and treat proportional versus balanced sampling as a separate follow-up if calibration or prevalence effects become central. Evaluate probabilities at the natural prevalence of the outer test folds.

Two preprocessing ensemble members are used for training; their objectives are averaged. Categorical/transformation fits that are defined on the context must remain context-only. TabICL's training adapter ordinal-encodes and imputes the sampled episode's features before its internal context/query split, following that upstream training interface; no held-out credit table enters those episodes. Dataset-level sanitization includes data-dependent feature filtering over the full table: the benchmark is consequently transductive with respect to that schema preparation, not a completely inductive preprocessing protocol. An inductive claim requires moving that selection inside the relevant folds. Credit LGD uses a predefined target clipped to [0,1]; this describes bounded loss prediction, not every economic LGD definition including costs or recoveries above exposure.

### Budget and trajectories

The literature does not validate 5,000 as a universal budget. Real-TabPFN uses 20k; TabPFN-Wide uses 10k based on its own monitoring; target-table fine-tuning has different stopping rules. Keep 5k provisional until the long reference pilot and measured cost are reviewed; [LITERATURE.md](LITERATURE.md) gives the evidence and decision criteria. A bounded descriptive horizon is valid without proving convergence, provided that limitation is explicit.

AdamW uses zero ordinary weight decay, gradient norm clipping at 1, a 10% warmup and cosine decay to 5% of peak. The complete successful-update budget fixes the schedule once. Recovery restores its position; a resubmission does not restart warmup.

A trajectory point is a measurement of the same evolving weights, not a separately trained model or a new grid trial. For example, the point at update 1,000 of a 5,000-update run has followed the 5,000-update schedule; it is not equivalent to an independently trained 1,000-update run. Intermediate weights are transient. Only final weights and the latest optimizer recovery state persist.

Monitoring uses fixed seed **31415**, up to 2,000 rows/table and four inference members. Every table has fixed, disjoint context/validation/query rows (48%/12%/40% before rounding). Thresholds and calibrators use only validation rows. This differs deliberately from the training batch's 60% context/40% query. It records per-table discrimination, calibration and error metrics, plus timing and movement from initialization. Monitoring preserves training RNG and module modes. The final benchmark uses 32 TabPFN or 8 TabICLv2 members and full outer test folds. Keep the two evaluation fidelities separate in plots and text.

A finite but disappointing/flat curve does not stop training. Only numerical failure or inability to reach the budget produces a divergent outcome. Completed numerical failures are retained as outcomes for the descriptive grid instead of repeatedly rerunning them until one succeeds. Infrastructure failures and interruptions are recoverable work, not scientific successes.

## Sampling comparison and training randomness

**Experiment 2 adds 32 seed trials** repeating the predefined full-update reference (LR `3e-7`, lambda `0.003`, `one_sample`) with training seed **43**: one recipe × one additional seed × four bases × four dataset folds × two tasks. Its seed-42 counterpart already exists in experiment 1. Configs are `config/experiment2/{pd,lgd}.yaml`, run family `cpt_seeds_v5`. Keep dataset partitions, monitor/evaluation seeds, row caps and all other settings identical to the main reference.

This is a small paired sensitivity check: did changing the training randomness materially alter the reference behavior? Two seeds do not reliably estimate a seed distribution, and this check says nothing about robustness at every other recipe. Show paired differences and curves; do not count the repetitions as extra independent datasets. A future expansion to 64 additional trials would add seed 44 for the same recipe, only if the initial comparison warrants it.

**Experiment 3**, `config/experiment3/{pd,lgd}.yaml`, defines a separate **96-trial** study at LR `3e-7`, lambda `0.003`, full updates, seed 42: three sampling modes × four bases × four folds × two tasks. **All three modes use proportional PD sampling** (`train.context_sampling: stratified`), with uniform LGD sampling. Covering each row exactly once cannot force balanced class prevalence without dropping or repeating rows. This proportional `one_sample` control is therefore deliberately separate from the main grid's balanced-PD reference; their PD difference is a prevalence-policy change, not a pass-mode effect.

The research campaign contains **512 main + 96 sampling + 32 additional seed trials = 640 trials**. Run the seed check once the matching main references are available; it does not repeat the full grid. Use the same pilot-selected budget and trajectory points in all three phases.

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

Review these curves and measured costs, then fix the common budget/milestones in experiments 1–3 before preparing their plans. A longer budget also needs a sufficient epoch safety rail. The default research design remains **512 main + 32 seed + 96 sampling = 640 trials**, plus **72 control/pilot training arms** if every experiment-0 stage runs once. Interrupted execution adds a segment, not a new scientific trial.

## Retention and measurement protocol

`config/retention.yaml` fixes a research panel of eight non-credit datasets from the curated **TabArena-v0.1 OpenML versions**: Amazon employee access, online shopping intention, QSAR biodegradation and seismic bumps (classification); airfoil noise, concrete strength, protein structure and superconductivity (regression). Null/recovery controls use the small packaged breast-cancer/diabetes pair. Data identifiers, versions, source/license metadata and byte hashes are recorded; no panel table enters the 25-table adaptation corpus.

This panel is not the full TabArena benchmark. We apply CreditPFN's own fixed monitoring and five-fold final evaluation, not TabArena's official benchmark protocol. Prior exposure of base checkpoints to these datasets is not established. Report **change from each base on a declared panel**, not universal retention, benchmark leadership or guaranteed novel-data generalization.

At milestones record trainable parameter names/counts, relative movement of the trainable weights, and per-tensor norm, mean, standard deviation, maximum absolute value, bounded sampled quantiles, anchored absolute/relative movement, cosine similarity and optimizer-moment norms. Unanchored frozen-tensor movement remains unavailable rather than being invented as zero. Epoch records retain data loss, regularization, gradient/clipping/skips, update/row counts and training/data-wait/monitor time. Every 20 seconds record process memory and allocated-device utilization, memory, power and temperature when available. Counters describe sampled device behavior, not exact process energy or billed allocation time.

Final evaluation uses **five outer row folds** on each held-out credit table and each non-credit panel table. Inside each outer training fold, a disjoint 20% validation split supplies classical HPO, maximum-F1 threshold selection and Platt/isotonic calibration. No outer test labels enter those choices. F1 at threshold 0.5 is retained for context. Foundation models use their capped context; classical controls use the full inner training fold. This compares practical native-context settings, not equal row exposure.

Logistic and ridge regression use context-fitted one-hot categorical encoding, numerical median imputation and scaling. CatBoost uses native categorical features; the declared XGBoost recipe uses ordinal numeric codes. These are specific tuned controls, not an exhaustive comparison of every preprocessing recipe. Requested tuning requires Optuna. Each result records the requested/completed trial counts and selected parameters, since a time limit can end a search before all requested trials finish. The outer test folds remain untouched by both the search and validation-fitted calibration/threshold selection; reusing validation for those choices can overfit validation, but does not turn it into the reported test score.

Record fold-level discrimination/error scores, confusion counts, prevalence, raw/calibrated proper scores, ECE and reliability-bin sufficient statistics, and fitting versus prediction/scoring time. Compressed row predictions preserve the original dataset row index. Supported regression distributions add 80%/90% interval coverage/width, a fixed-grid mean pinball loss and quantile-crossing rate. Unsupported quantities remain missing. Nine quantiles do not produce an exact CRPS. Untuned/classical scores and predictions are reusable only with matching data/model/config/code/environment identities.

Detailed measurements live on project storage under each experiment. Resource and tensor records consolidate into separate narrow tables so they do not expand the wide trajectory table into a large mostly-empty matrix. Notebook credit and non-credit averages remain separate.

## Reporting and remaining evidence

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

## State of the evidence

On 22-09-2026 the user reported no running VSC jobs. The old experiment has an invalid swept L2-SP axis and defects in worker sampling/accumulation. Retagging 338 surviving checkpoints recovered their lambda-0.003 identity, not a corrected training history. The user reported 412 checkpoint files on project storage; that is not a verified completed-trial count.

Historical headline measurements, including the completed run-8 evaluation, remain in the agent-memory runs table as historical observations. They are not results of this redesigned experiment and cannot establish its conclusions. A clean rerun means new training and evaluation under new phase names; it does not require deleting the raw corpus or redownloading validated base weights.

The current implementation uses training protocol **5**. Debugging outcomes are recorded in the agent-memory runs table; they are not results of the main scientific grid. Local tests exercise sampling, finite mean gradients, exact budgets, recovery, cleanup boundaries, identity checks and analysis. Accepted cluster controls, the pilot budget decision, real CUDA recovery and measured walltimes are prerequisites for the full campaign. Preparation logs establish that inputs and plans were written; they are not completed training or evaluation results.

A passing experiment 0 establishes the tested controls, not the capacity of every final evaluation task. Before scheduling all final benchmarks, measure a representative large-table evaluation with the intended inference ensemble/context cap and a classical HPO task. Training-cap pilots and the small packaged recovery benchmark do not validate million-row inference or its walltime.
