# Literature — continued pretraining and adaptation behavior

Initial review against primary text at library pin **`e5ce01614eebe520af303f2b5bfd212298eab2be`**; closest-work and appendix review refreshed on 12-10-2026 at **`1ac6e6783f24b2d5cd058af7ad8cad7c1bfaf7ae`**. This is a focused reading list for CreditPFN, not a systematic proof of novelty. A library folder year/date can describe a later revision rather than the first publication date. The library is read-only in this repository.

The training-objective check below additionally uses the updated pin **`81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`** (28-09-2026).

## Closest evidence

| Primary study | What was evaluated | Implication for CreditPFN |
|---|---|---|
| [Garg et al., Real-TabPFN](../tfm-library/papers/text/2025/07_Garg_et_al._Real_TabPFN_Improving_Tabular_Foundation_Models_via_Continued_Pre_training_With_Real_World_Data.txt) ([paper page](https://arxiv.org/abs/2507.03971)) | Continued pretraining of TabPFN v2 on 71 curated general-purpose classification tables; comparisons of real-data sources and context sizes, evaluated on separate benchmark tables | Direct precedent for real-table corpus adaptation. It supports a conservative reference recipe and corpus/context sensitivity, not an assertion that credit-domain CPT must help or that its settings transfer unchanged to LGD/new architectures |
| [Rubachev et al., On Finetuning Tabular Foundation Models](../tfm-library/papers/text/2025/06_Rubachev_et_al._On_Finetuning_Tabular_Foundation_Models_1.txt) ([paper page](https://arxiv.org/abs/2506.08982)) | Target-dataset fine-tuning of TabPFN v2, comparisons of full and partial updates, convergence and representation/retrieval changes; IID and temporal tasks | Directly relevant to adaptation behavior, but the task is different from transfer to an unseen table after corpus-level CPT. Full updates are an important reference; partial updating does not automatically save time or improve quality |
| [Tanna et al., Exploring Fine-Tuning for Tabular Foundation Models](../tfm-library/papers/text/2026/04_Tanna_et_al._Exploring_Fine_Tuning_for_Tabular_Foundation_Models.txt) | Per-dataset adaptation experiments across tabular foundation models, including original TabICL; sensitivity to adaptation procedure | Motivation to report failures, architecture-specific mechanisms and stability. Its TabICL results do not establish collapse or the correct freeze scheme for **TabICLv2** |
| [Qu et al., TabICLv2](../tfm-library/papers/text/2026/02_Qu_et_al._TabICLv2_A_better_faster_scalable_and_open_tabular_foundation_model.txt) | A newer model with classification/regression support, staged architecture and scalable inference | Use the actual v2 training/inference interfaces and distinguish front-end freezing from freezing its dominant ICL stack. Inference capacity demonstrations do not establish train-time memory capacity |
| [Tanna et al., Data Presentation Over Architecture](../tfm-library/papers/text/2026/05_Tanna_et_al._Data_Presentation_Over_Architecture_Resampling_Strategies_for_Credit_Risk_Prediction_with_Tabular_Foundation_Models.txt) | Context construction and imbalance on two credit datasets, without corpus-level continued pretraining; Lending Club uses a temporal split | Context and prevalence are substantive factors. It does not establish that balancing the **query training loss distribution** is equivalent to balancing only the inference context |
| [Purucker et al., Beyond IID](../tfm-library/papers/text/2026/06_Purucker_et_al._Beyond_IID_How_General_Are_Tabular_Foundation_Models_Really.txt) | Tabular model evaluation under IID, temporal and grouped splits; finetuning/CPT is outside the evaluated adaptation scope | IID rankings should not be promoted into claims about future lending or new borrowers. Proper split keys and source grouping are prerequisites for those questions |
| [Kolberg et al., TabPFN-Wide](../tfm-library/papers/text/2026/03_Kolberg_et_al._TabPFN_Wide_Continued_Pre_Training_for_Extreme_Feature_Counts.txt) | Continued pretraining toward extreme feature counts using an adapted synthetic training setting, with retention/generalization checks | CPT can target a data regime as well as a real-data domain. It motivates retention checks, but does not measure real credit-corpus specialization |
| [Grinsztajn et al., TabPFN-3 Technical Report](../tfm-library/papers/text/2026/05_Grinsztajn_et_al._TabPFN_3_Technical_Report.txt) | New architecture, synthetic prior training and broader inference capacity/evaluation | Changes in base prior and context capacity can change adaptation behavior. A published inference envelope is not a measured continued-pretraining row cap for our GPU/configuration |

## Recipe choices and their limits

**Reference rate and anchoring.** Garg et al. use LR `3e-7`, AdamW, warmup/cosine, L2-SP strength `0.003` in the form `0.5 × lambda × ||w - w0||²`, 20,000 optimizer steps, one dataset batch per step and a 60/40 context/query split. They sample up to 20,000 rows uniformly. CreditPFN's equal table visitation is an explicit design choice; do not misread uniform **row** sampling as a documented equal-table schedule in that paper.

**Training objectives.** Classification uses cross-entropy over the active dataset classes. In the [TabPFN snapshot](<../tfm-library/repositories/TabPFN .txt>), `TabPFNClassifier.forward` selects/permutates those columns before `FinetunedTabPFNClassifier._forward_with_loss` applies CE; the installed local classifier also selects classes before softmax. The former CreditPFN explanation that unused outputs steal inference probability mass was incorrect. The 28-09 correction removes their unintended contribution to the training loss and gradients. Earlier positive-LR TabPFN PD pilots describe the previous full-head objective, not this corrected recipe.

TabPFN regression uses the loaded bar-distribution negative log-likelihood. Its upstream `FinetunedTabPFNRegressor._forward_with_loss` also supports auxiliary objectives; CreditPFN does not enable those. The upstream `_targets_in_estimator_space` applies a member's context-fitted target transform to its query targets before the loss. CreditPFN previously repeated the shared z-scored query across all members even when a member's context used `safepower`; the 28-09 correction carries each member's transformed query targets into NLL without fitting on query rows. Old TabPFN LGD pilots with non-identity target transforms therefore also used a different objective. TabICL's objectives are unaffected by these repairs.

Scope matters: the hash-matched v3 regressor metadata and upstream `_get_v2_config` specify `(None, 'safepower')`. The hash-matched v2.6 regressor specifies `('none',)`, whose factory is an identity `FunctionTransformer`; that pilot has no target-space mismatch. Do not assume every LGD base was affected merely because it shares a training function.

TabICLv2 regression uses mean pinball loss across its 999 native quantiles, matching the [TabICL snapshot](../tfm-library/repositories/TabICL.txt) `_pinball_loss` and `_compute_batch_loss`; its paper describes the quantile formulation. Native losses provide a defensible controlled starting point, not proof of optimal credit AUC/F1/RMSE. NLL and pinball assess predictive distributions; neither directly minimizes point RMSE. Loss scales and the relative strength of the same L2-SP lambda differ across these objectives.

**Learning-rate range.** The four-level main range extends from this conservative CPT reference through `3e-5`. Rubachev et al. study higher target-specific fine-tuning rates, including a search spanning roughly `5e-6` to `5e-4`; those rates are not automatically appropriate when preserving a transferable prior across unrelated tables. The upper CreditPFN level is a useful stress condition to examine, not a literature-established optimum. A short endpoint pilot checks numerical behavior before the large campaign.

**Partial updates.** Rubachev's LayerNorm/head/embedding strategy includes trainable normalization affine parameters. CreditPFN freezes the selected stack completely and trains modules around it. This is a distinct ablation. Measure the fraction and names of trainable parameters, movement from initialization, time and memory; avoid applying a shared label to opposite freeze schemes in different families.

**Regularization interpretation.** On/off L2-SP answers whether this particular reference anchor changes behavior. Two levels cannot identify an optimal lambda or a regularization response curve. Because the penalty is a sum over trainable weights and the data losses differ across tasks, equal lambda is not equal effective constraint. Penalty-to-data-loss behavior and relative movement help interpret architecture/adaptation interactions.

**Budget: the literature provides precedents, not a universal update count.** The closest real-table CPT study, Garg et al., ran 20,000 updates. Kolberg et al. fixed 10,000 updates after their monitored omics/SNP curves plateaued (TabPFN-Wide, Appendix J); that used a different synthetic adaptation task. Rubachev et al. evaluated target-table fine-tuning every ten updates with patience of sixteen non-improving checks. That is a stopping rule, not a 160-update total or evidence for a universal CPT budget.

No reviewed study establishes that 5k, 10k or 20k updates captures all relevant behavior for all four CreditPFN bases, both tasks and both adaptation modes. With no skipped steps, the selected 10k equal-table budget corresponds to roughly 769–833 visits per PD training table or 1,667 per LGD training table. This smaller corpus repeats tables much more often than a large corpus; row draws, losses, rates and anchoring still determine what those visits do.

**CreditPFN's pilot-informed decision.** Review of the eight 20k full-update reference pilots supports extending the provisional 5k window to **10k**: credit effects and non-credit retention still change after 5k, including a classification retention mean crossing below its baseline. This is project evidence, not a conclusion from the cited literature. Ten thousand balances a wider descriptive window against compute; it does not establish saturation, and the preserved 20k pilots contain later changes too. One recipe/fold/seed per base/task cannot establish convergence of every main-grid recipe. The monitored fold is reused in the descriptive study; disclose this horizon selection rather than presenting it as independent validation. Historical measurements live in AGENTS_MEMORY.md, and the implemented protocol in RESEARCH_BRIEF.md.

Following the loss corrections, the unaffected TabICL classification/regression and v2.6 regression trajectories still support observing beyond 5k. Keep 10k as a descriptive horizon, but do not use affected old TabPFN pilots to certify the corrected objectives' score curves. Short GPU validation of those changed paths remains required; retaining a common descriptive horizon is not a claim that every architecture has converged.

A point on a 20k-schedule trajectory is not equivalent to a fresh run whose entire cosine schedule ends at 10k. Longer pilots diagnose whether appreciable late behavior exists; they do not isolate duration from schedule shape. Keep the chosen budget and milestones fixed in main, seed and sampling configs before preparing their plans, disclose the pilot decision, and do not select each recipe's best checkpoint for the main comparison.

**Sampling.** Equal table visitation prevents the largest table from owning most updates. Exhaustive partitioning with size-proportional updates and within-table gradient accumulation address different questions. Equal steps alone do not match data exposure or compute. For PD, the existing class-balanced batch draw changes both context and query prevalence; hold it fixed in the main grid and its seed check. The separate pass-mode study uses proportional sampling in all three arms so exhaustive coverage is possible without changing class prevalence between those arms. Non-overlapping partitions are a controlled design choice here, not a literature-established improvement in predictive quality.

**Seed and sampling controls.** Rechecked the primary Real-TabPFN recipe (single-table batches, uniform capped row draws and 60/40 context/query), Rubachev's fine-tuning analysis, and the TabICLv2 objectives at pin `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`. None establishes that two seeds quantify grid-wide robustness or that disjoint full passes dominate random sampling. Experiment 2 is a paired two-seed sensitivity check at one prespecified recipe. Experiment 3 deliberately changes both update weighting and exposure, so equal updates must be accompanied by row and compute measures. Its averaged chunk objective is not a joint all-row context and slightly unequal query counts are not exactly row weighted. The new six-update auxiliary controls verify execution, not those empirical research hypotheses.

**What the study can conclude.** A fixed descriptive grid can map behavior without claiming a winner. It must still show incomplete coverage and numerical failures, preserve paired base controls, and avoid treating dependent folds/seeds as independent datasets. A generalization claim about a selected recipe, no-forgetting claim, specifically domain-driven benefit, or temporal credit deployment needs corresponding additional evidence.

**Baseline preprocessing and descriptive correlations.** The linear controls one-hot encode nominal categories using only their training context, rather than imposing the arbitrary ordering of ordinal codes. This is a declared baseline recipe, consistent with the [scikit-learn encoder's intended use for linear models](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), not a claim that it dominates all categorical encodings. For a higher-is-better score, the change-versus-base plot has `Cov(base, adapted - base) = Cov(base, adapted) - Var(base)`. A negative association can therefore arise without an adaptation mechanism. Show paired per-table effects and factor contrasts; do not interpret that correlation causally.

## Figures and interpretation

Visualization review: 30-09-2026, library pin **`81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`**.
These are presentation choices for this descriptive design, not claims that the literature prescribes
one sufficient figure set.

- **Corpus geometry, exposure and learning curves:** Real-TabPFN's corpus/context comparisons,
  Rubachev et al.'s adaptation/convergence analysis, and TabPFN-Wide's retention checks motivate
  showing table sizes, optimizer progress, parameter movement, credit/non-credit changes and cost
  together. They do not establish a universal update budget or broad retention from a small panel.
- **Paired dataset effects:** [Demšar (2006), Statistical Comparisons of Classifiers over Multiple
  Data Sets](https://www.jmlr.org/papers/v7/demsar06a.html) analyzes comparisons across datasets.
  CreditPFN therefore keeps the dataset as its reporting unit and pairs folds before aggregating.
  We show absolute AUC changes, fractional RMSE reductions, matched factor contrasts and the full
  dataset spread. This normalization is our design choice. Ranks are secondary context; no critical
  difference diagram or independent-fold confidence interval is claimed for this correlated grid.
- **Probability quality:** [Guo et al. (2017), On Calibration of Modern Neural
  Networks](https://proceedings.mlr.press/v70/guo17a.html) motivates inspecting calibration separately
  from classification accuracy. Its experiments concern image/document models; our binary
  positive-class reliability panels use observed event rates and predicted probabilities within
  each credit dataset. They are not a reproduction of multiclass top-confidence diagrams.
  Brier/log loss, bin counts and validation-selected F1 thresholds accompany ECE; a small ECE alone
  does not establish probability quality. Raw, Platt and isotonic curves are separated, with the
  fixed adapted recipe paired to its base on the same outer folds.
- **Honest partial reports:** at most four LR curves share a panel; dataset matrices are paginated;
  missing measurements remain gaps; complete planned coverage is required for aggregate learning
  curves. Quartiles describe observed spread rather than sampling uncertainty. A DATA-only
  partition-mean endpoint plot is explicitly distinct from a project-storage per-table result.

## Upstream implementation anchors

Use symbol names when referring to code snapshots because line numbers change on refresh:

- [TabPFN snapshot](<../tfm-library/repositories/TabPFN .txt>): `load_model_criterion_config`, `DatasetCollectionWithPreprocessing`, `TabPFNEnsemblePreprocessor`, `FinetunedTabPFNClassifier`. Preserve architecture/configuration and criterion metadata in the project's save path.
- Numerical preprocessing: `TorchSoftClipOutliers`, `TorchSoftClipOutliersStep`, `torch_nanstd`, `TorchStandardScaler` and `TabPFNV2._embed_features` in that snapshot establish the two-pass logarithmic clipping, context fitting and subsequent imputation/scaling path. Rational shrinkage of every value is not the upstream algorithm. The snapshot's float32 statistics can overflow on extreme units; a finite model loss after overflow has turned feature values into NaNs is not evidence of information-preserving preprocessing. CreditPFN's recorded unit conversion before dtype casting is an explicit engineering safeguard, not a paper-prescribed adaptation method. Reference pin: `e5ce01614eebe520af303f2b5bfd212298eab2be`.
- [TabICL snapshot](../tfm-library/repositories/TabICL.txt): `TabICL`, `ColEmbedder.forward`, `RowInteractor.forward`, `_build_meta_batch`. Training flags choose materially different forward algorithms; freezing must not switch the forward path into inference mode.
- Distribution scoring: `TabPFNRegressor.predict(output_type="full", quantiles=...)` returns point outputs, a list of quantile arrays and criterion/logits; `TabICLRegressor.predict(output_type=["mean", "quantiles"], alphas=...)` returns a row-major quantile matrix. Specify the levels explicitly and reuse these multi-output calls; their default nine levels are not CreditPFN's chosen grid.
- [VSC documentation snapshot](<../tfm-library/repositories/VSC Documentation.txt>): `KU Leuven storage`, `Transferring data between Lustre and GPFS`, quota and job-scheduling sections. Use the live official pages linked in [VSC.md](VSC.md) to verify current operational rules; generic defaults do not identify our allocation's actual quota.

Before publication, refresh the closest-work search and verify the exact upstream weight provenance. Avoid global “first” claims based only on this reading list. Keep paper-evaluated behavior distinct from functionality merely supported by an implementation.

## Fixed non-credit retention panel

The TabPFN-Wide retention study and Real-TabPFN's evaluations motivate measuring the starting model and adapted model on the same external tables; neither prescribes CreditPFN's eight-table panel. We use a fixed non-credit subset of [TabArena-v0.1, OpenML study 457](https://www.openml.org/search?type=study&id=457), with exact dataset/version IDs in `config/retention.yaml`. The [TabArena primary paper and supplement](https://openreview.net/forum?id=jZqCqpCLdU) describe the benchmark's curated tasks. [TALENT's official repository](https://github.com/LAMDA-Tabular/TALENT) was also examined; its much broader collection is not necessary for the initial bounded retention panel.

The panel choice is a project design decision, not a literature-validated sufficient sample of all non-credit domains. We use our own five outer folds and fixed milestone subsets; this is not an official TabArena benchmark score. The two sklearn packaged datasets used for null/recovery controls provide a small inference/debugging check only. Base-checkpoint exposure remains unverified; paired changes are interpretable without claiming these are entirely new datasets to every base.


## October test-time-compute paper: relevance and limits

Read on 12-10-2026. Local library pin: `1ac6e6783f24b2d5cd058af7ad8cad7c1bfaf7ae`.
The requested paper is absent from this pinned library; the primary paper and its public
implementation were inspected directly, without changing the submodule.

[Ning et al., Test-Time Compute for Tabular Foundation Models: Mechanisms, Gains, and Limits](https://arxiv.org/html/2610.12005v1)
separates adaptation, aggregation and context construction. Its adaptation is **target-table
fine-tuning**, with inner validation, early stopping and fallback to the original weights.
That differs from CreditPFN's cross-table domain adaptation and fixed endpoint budget.

The released [implementation](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/tree/45fb141b2acf15518e70885aa149c8e1898cd15c)
was inspected at commit `45fb141b2acf15518e70885aa149c8e1898cd15c`:

- `DiagScale`, `inject_diagscale`, `select_trainable` in the TabPFN patches implement
  zero-initialized query-scaling adapters and freeze the original weights. The TabICL
  `finetune_tabicl`/`_finetune` runner restores the validation-best state, including the
  untouched starting state. This is not our current frozen-backbone mode.
- `configs/tabpfn3/full_ft.yaml` uses peak LR `1e-6`, at most 30 epochs, two training/validation
  views and eight deployment views. DiagScale configurations use `3e-2`; that rate is for
  tiny adapters, not a recommendation for full-weight credit CPT.
- The TabICL KV-cache patch and its `test_ia3_cached_kv_matches_fresh_path` illustrate why
  a nonzero adapter must be tested on cached and fresh prediction paths, not only at
  zero initialization. Any future integration needs save/reload and resumed-update controls.
- Study-level LR selection uses aggregate benchmark test performance (Appendices A.4/B.3),
  with a separate across-dataset sensitivity analysis. Task-level checkpoint/ensemble
  selection uses training-side validation. These two selection levels must not be conflated.
  Our post-hoc ablations likewise remain exploratory unless evaluation is reserved separately.

Implications for this project: compare inference budgets for original and adapted checkpoints;
consider a narrowly scoped similarity-adapter study; and separate domain adaptation from
subsequent target-table adaptation. Do not transfer its positive results into a claim that
credit-corpus CPT must help. The context-retrieval evidence motivates a bounded large-table
inference sensitivity, not an untested assertion that CPT benefits only large datasets.

### Local primary-source refresh — 13-10-2026

Library pin: `de324efcca55be392feb0c01830b3b6ce364587f`. The earlier absence statement above
describes the 12-10-2026 pin only; both the TTC primary text and its released code are now
available locally in `papers/text/2026/10_Ning_et_al._Test_Time_Compute_for_Tabular_Foundation_Models_Mechanisms_Gains_and_Limits.txt`
and `repositories/Test-Time Compute for Tabular Foundation Models.txt`.

Verified symbols: TabPFN `DiagScale`, `inject_diagscale`, `select_trainable`; TabICL
`sdpa_with_flattened_batch`, `inject_diagscale`, `select_trainable`, `finetune_tabicl`/`_finetune`.
TabPFN scales the output of each frozen `SoftmaxScalingMLP`; TabICL applies the multiplier
after SSMax. The latter runner uses `freeze_others=False`, clears all model gradients and
clips only optimizer-selected tensors, preserving CUDA forward flag parity. This is not
equivalent to putting the backbone in inference mode. The snapshot patches TabPFN 8.0.7 and
a post-2.1.1 TabICL source; current VSC plans record TabPFN 8.0.8 / TabICL 2.1.1, so compatibility
must be tested rather than assuming a drop-in install. No library files or environments were changed.

Real-TabPFN Section 4 reports TabPFNv2, 71 curated OpenML/Kaggle tables, AdamW peak LR
3e-7, linear warmup/cosine schedule, L2-SP alpha 0.003, 20k updates, one table per batch,
up to 20k total rows split 60/40 context/query and a 400k-cell cap. Appendix C reports
mean raw ROC-AUC 0.929 → 0.932 (about +0.003), distinct from Section 5's normalized
0.954 → 0.976 (+0.022). The episode-size ablation reports normalized gains +0.010 at
2,048/4,096, +0.016 at 8,192/16,384 and +0.022 at 20,000. Source comparisons additionally
change curation, features and diversity; they do not isolate table size alone.

The Real-TabPFN primary text's Section 3 explicitly selects training tables above 10k rows as
part of its contamination safeguards while evaluating smaller tables. It is not a controlled
large-versus-small training-corpus experiment. The source/grouped split warnings in Beyond IID
also apply to credit tables that share loan populations or releases.


### Implemented follow-ups and deliberate differences — 13-10-2026

At the same `de324efcca55be392feb0c01830b3b6ce364587f` pin, experiment 4 is now the
context-only curve; experiment 5 is DiagScale CPT; experiment 6 tests PD training-corpus
eligibility at a fixed episode cap; experiment 7 tests episode size on the same large-table
pool. The formerly numbered generic-data and prevalence proposals are excluded.

The independent adapter implementation follows `DiagScale`, `inject_diagscale`,
`sdpa_with_flattened_batch` and `finetune_tabicl` in the local TTC snapshot. It does not
install or alter the authors' forks. Preserve query-scaling order, TabICL autograd parity,
optimizer-only clipping and nonzero cached/reloaded inference. TabPFN's wrapper uses tensor
dtype promotion; TabICL explicitly casts the multiplier to the query dtype. Keep that
distinction, while preserving FP32 adapter storage. Match TabICL FP32 deployment
for original and adapted weights so precision is not a hidden within-experiment confound.
The paper uses target-table validation-best selection/base fallback; our cross-table experiment
uses fixed endpoints. Its additional `1e-4` rate and provisional 8,192-row FP32 episodes are
new choices to be validated, not published optimal CPT settings. Comparing this recipe with
old full-weight CPT does not isolate only parameterization. Numerical and protocol tests
precede any cluster claim of equivalence or capacity.

### Separating training-corpus eligibility from episode size — 13-10-2026

Reread the complete Real-TabPFN primary text and TabICLv2's training-schedule section at
`de324efcca55be392feb0c01830b3b6ce364587f`. Real-TabPFN's above-10k training-table rule is
part of its contamination safeguards. Its smaller held-out tables are still evaluated: the
paper does not say CPT helps only large downstream datasets. Its episode-size ablation is
more direct evidence for testing a larger context-plus-query budget than its data-source
comparison is for attributing gains to table size alone.

TabICLv2 Section 4.1 uses 500k updates with 1,024-row tasks, then 40k with 400–10,240 rows,
then 10k with 400–60k rows. Learning rates change from 8e-4 to 1e-4 to 2e-5; the context
fraction changes from 30–90% to 80%. That staged recipe motivates studying episode size,
but it does not isolate its effect or establish a credit-CPT optimum.

Accordingly, experiment 6 changes only the eligibility policy within each fixed training
complement, with an 8,192-row cap. It measures the combined policy effect, including changed
table identities/diversity, exposure and revisit frequency. Experiment 7 keeps that large-table
pool fixed and changes total rows to 2,048/8,192/26,000, preserving the 60/40 split and update
schedule. It changes both context and query counts; equal updates do not match compute.
These experiments test a conservative reference on modern bases, not a literal replication
of Real-TabPFN's v2/20k-update/20k-row/cell-capped recipe. The matching 8,192-row control is
shared by identity, reducing 48 previously proposed fresh trials to 32, at the deliberate
cost of no longer estimating the full corpus-policy × episode-size interaction.
