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

## Implemented main grid

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

The count is **4 × 4 × 2 × 2 × 4 × 2 tracks = 512 training trials**. PD and LGD remain separate tasks and analyses. Configs are `experiment1_pd.yaml` and `experiment1_lgd.yaml`, run family `cpt_main_v3`.

A partial round at the last update is permitted; visitation counts then differ by at most one. One update is not equal FLOPs, rows or GPU seconds across models or sampling modes. Record processed rows, successful updates, measured training time, monitoring time and peak memory alongside model quality.

### Adaptation is explicit

Full updates optimize every normally trainable parameter. Architectural constants can remain nontrainable: TabICLv2's RoPE frequency parameter is one example.

The frozen arm detects the repeated transformer stack containing the most parameters and freezes that stack, including its normalization affine parameters. It leaves the surrounding input/target embeddings and prediction machinery trainable. For the current architectures this selects TabPFN v3 `icl_blocks`, v2.6 `blocks`, the v2 transformer encoder stack, and TabICLv2 `icl_predictor.tf_icl.blocks`. TabICLv2's column embedder and row interactor remain trainable. The null/pilot audit must confirm the actual names/counts for each loaded checkpoint.

This is **not LoRA**, not only a final linear-head update, and not a literal replication of Rubachev's LayerNorm/head/embedding recipe. The inherited internal flag is still `use_lora`; modern filenames use `_frozen`, and provenance uses `adaptation_mode`. Legacy `_lora`/`_iclhead` names remain readable.

Freezing changes `requires_grad`, never `.training`. TabICLv2 uses different algorithms in training and inference mode. Gradients still propagate through a frozen stack to trainable embeddings, so the frozen parameter fraction does not predict activation-memory savings. Retain measured row caps.

L2-SP is an unnormalized sum, matching the real-table reference form. The same lambda does not give equal effective regularization across architectures, task-loss scales or trainable subsets. Report its measured penalty and relative trainable-weight drift, including drift when lambda is zero; interpret interactions rather than treating lambda as a universal strength.

### Sampling inside a table

The current fixed preprocessing configuration uses **class-balanced batch subsampling for PD**, capped by available rows, then splits that sampled batch into disjoint 60% context and 40% query rows. Despite its inherited name `context_sampling`, this affects **both** context and query prevalence. It is not a context-only intervention. LGD uses uniform row subsampling. Sampling is without replacement within a draw; rows can recur across visits.

This is a documented departure from Garg et al.'s uniform-row recipe. Hold it fixed in the main grid and treat proportional versus balanced sampling as a separate follow-up if calibration or prevalence effects become central. Evaluate probabilities at the natural prevalence of the outer test folds.

Two preprocessing ensemble members are used for training; their objectives are averaged. Categorical/transformation fits that are defined on the context must remain context-only. Dataset-level sanitization includes data-dependent feature filtering over the full table: the benchmark is consequently transductive with respect to that schema preparation, not a completely inductive preprocessing protocol. An inductive claim requires moving that selection inside the relevant folds.

### Budget and trajectories

The literature does not validate 5,000 as a universal budget. Real-TabPFN uses 20k; TabPFN-Wide uses 10k based on its own monitoring; target-table fine-tuning has different stopping rules. Keep 5k provisional until the long reference pilot and measured cost are reviewed; [LITERATURE.md](LITERATURE.md) gives the evidence and decision criteria. A bounded descriptive horizon is valid without proving convergence, provided that limitation is explicit.

AdamW uses zero ordinary weight decay, gradient norm clipping at 1, a 10% warmup and cosine decay to 5% of peak. The complete successful-update budget fixes the schedule once. Recovery restores its position; a resubmission does not restart warmup.

A trajectory point is a measurement of the same evolving weights, not a separately trained model or a new grid trial. For example, the point at update 1,000 of a 5,000-update run has followed the 5,000-update schedule; it is not equivalent to an independently trained 1,000-update run. Intermediate weights are transient. Only final weights and the latest optimizer recovery state persist.

Monitoring uses fixed seed **31415**, up to 2,000 rows/table and four inference members. It records per-dataset primary metrics, aggregate diagnostics, elapsed time and movement from the initialization. Monitoring preserves training RNG and module modes. The final benchmark uses 32 TabPFN or 8 TabICLv2 members and full outer test folds. Keep the two evaluation fidelities separate in plots and text.

A finite but disappointing/flat curve does not stop training. Only numerical failure or inability to reach the budget produces a divergent outcome. Completed numerical failures are retained as outcomes for the descriptive grid instead of repeatedly rerunning them until one succeeds. Infrastructure failures and interruptions are recoverable work, not scientific successes.

## Sampling comparison and training randomness

**No seed-sensitivity campaign is included.** All main and sampling trials use training seed 42. The earlier proposal added 128 trials to 512, for 640 combined; it never meant 640 additional trials. Those extra trials would estimate variability from row draws, preprocessing and training randomness at two reference recipes. That is useful for a robustness question, but it is not the current research priority. Results are consequently conditional on the chosen training seed; dataset folds do not replace seed repetitions.

`sampling_{pd,lgd}.yaml` defines an independent **96-trial** study at LR `3e-7`, lambda `0.003`, full updates, seed 42: three sampling modes × four bases × four folds × two tracks. Its `one_sample` control is intentionally rerun under the separate phase name.

The current research campaign therefore contains **512 main + 96 sampling = 608 trials**, plus the separate null controls and pilots. Accumulation remains in the sampling comparison; it is not part of every main-grid recipe.

| Mode | Data draws and update rule | Interpretation |
|---|---|---|
| `one_sample` | One draw and update per table visit | Equal table visitation |
| `full_pass` | `ceil(table_rows / cap)` independently resampled draws per table/round; update after each draw | Larger tables receive more optimizer updates |
| `accumulate` | Same number of draws as `full_pass`, kept together by table; mean finite gradients, then one update per table | More examples per update while keeping table-level update balance |

**The historical name `full_pass` does not guarantee exhaustive, non-overlapping row coverage.** Each draw is a fresh subsample; rows can repeat or remain unseen in a round. Do not describe these modes as merely equivalent implementations. At equal updates they differ in row exposure, weighting and compute. Report both update-indexed and exposure/compute-indexed results; a separate equal-compute comparison would need its own fixed budget rule.

## Phase gates before the main run

1. Optionally download historical measurements to a local archive folder; clear old output and trained checkpoints from both VSC tiers once the wanted copy is verified. No legacy output is required by the new run. Keep canonical raw/processed data and original model weights, and verify quotas.
2. Stage inputs, record content fingerprints and freeze the environment.
3. **16 zero-LR trials**, both adaptation arms across bases/tasks: verify original versus saved tensors and fixed-monitor parity. This checks the actual installed save/reload paths.
4. **32 short pilot trials**, 250 successful updates, conservative and high LR endpoints, both adaptations, one dataset fold, both tasks. Profile workers 0/4/8 only if needed (96 short trials for all three settings). Compare throughput, CPU memory, data wait and monitoring cost; worker count is a performance setting, not a scientific factor.
5. **8 longer reference pilots**, one full-update reference per base/task on the first fold, up to 20,000 updates with measurements through 5k/10k/20k. This checks whether 5k would truncate substantial behavior. These pilots follow a 20k schedule, so their early points are not substitutes for the 5k-grid curves. Choose and document the main budget before preparing its immutable plans. Keep the same final budget/trajectory definitions in main and sampling configs.
6. Main grid, separate sampling comparison, final evaluation and consolidation. The 16 null controls + 32 short pilots + 8 budget pilots are additional to the 608 research trials: **664 scheduled training trials** if all phases run once, before optional worker profiling or recovery canaries.

Short jobs can resume from a successful-update boundary. Validate one interrupted/requeued positive-LR pilot on VSC before enabling automatic requeue for the campaign. Synthetic CPU tests establish the software contract; they do not establish CUDA determinism or throughput on B200 hardware.

## Reporting and remaining evidence

- Show PD AUC changes and LGD fractional RMSE reductions **within dataset**, paired to the same base and row fold. Aggregate row folds/repetitions within dataset before giving datasets equal weight. Do not average raw LGD RMSE across incompatible target scales.
- Show individual datasets and coverage, with failures and missing milestones. Trajectory plots leave a point missing when observations from known baseline trials are absent at that update, rather than silently averaging only survivors.
- Report calibration/log loss/Brier and class-imbalance metrics for PD, RMSE/MAE and supported probabilistic diagnostics for LGD. Existing nine-quantile summaries do not establish an accurate full-distribution CRPS. TabPFN density scores are clipped for numerical stability; report that convention and do not treat unlike density/quantile objectives as interchangeable.
- Do not count training trials, overlapping dataset folds or many CV rows as independent datasets. Any dataset-level bootstrap is descriptive and conditional on this small, dependent corpus; avoid unqualified significance claims or post-hoc winner tests.
- Compare endpoint and trajectory effects for learning rate, anchoring and adaptation, including interactions. Separate step budget, row exposure, monitoring overhead and GPU allocation time.
- A provenance audit of base pretraining and related data sources is still required for contamination/independence claims. The intended starting weights are upstream synthetic-pretrained bases, not Real-TabPFN adapted weights; content identity matters more than filename labels.
- Temporal/grouped validation requires valid origination-time and loan/borrower keys, excluded from features. It is a future sensitivity study, not a property of current IID folds. Historical date-availability observations must be rechecked against the actual raw version.
- A fixed non-credit panel is required to measure forgetting; a matched generic-data adaptation control is required to isolate a specifically credit-domain effect. Neither is part of the current 25-table grid.
- Additional baselines such as RealMLP, context-prevalence sensitivity, longer horizons and more lambda levels are optional follow-ups, clearly separated from the main descriptive grid.

## State of the evidence

On 22-09-2026 the user reported no running VSC jobs. The old experiment has an invalid swept L2-SP axis and defects in worker sampling/accumulation. Retagging 338 surviving checkpoints recovered their lambda-0.003 identity, not a corrected training history. The user reported 412 checkpoint files on project storage; that is not a verified completed-trial count.

Historical headline measurements, including the completed run-8 evaluation, remain in the agent-memory runs table as historical observations. They are not results of this redesigned experiment and cannot establish its conclusions. A clean rerun means new training and evaluation under new phase names; it does not require deleting the raw corpus or redownloading validated base weights.

The current implementation uses training protocol **3**. Local tests exercise sampling, finite mean gradients, exact budgets, recovery, cleanup boundaries, identity checks and analysis. Cluster null controls, the pilot budget decision, real CUDA recovery and measured walltimes remain gates before launching the full campaign.
