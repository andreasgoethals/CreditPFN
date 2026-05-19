# Literature on Tabular Foundation Models

A chronological tour of the 27 PDFs in this folder. The arc:
PFNs (in-context Bayesian inference for arbitrary priors) → TabPFN
(PFNs with a tabular prior) → TabPFNv2 (production-grade — the
model we build on) → a Cambrian explosion of variants (continued
pretraining, drift, fairness, causal inference, time series, many
classes, scalability) → TabPFN-2.5 / 2.6 (the unified successors
that themselves ship continued-pretraining as default) →
TabPFN-3 (May 2026; new three-stage architecture, scales to 1M
rows, test-time-compute "Thinking" mode).

For every paper:

* **Where it fits** in the picture above.
* **What it actually contains** — methods, datasets, headline result.
* **For CreditPFN** — concrete relevance to this project.

The six most directly relevant papers for CreditPFN are
[Grinsztajn 2026 — TabPFN-3](#tabpfn-3),
[Garg 2025 — Real-TabPFN](#real-tabpfn),
[Hollmann 2025 — Accurate predictions on small data](#tabpfn-v2-nature),
[Grinsztajn 2026 — TabPFN-2.5](#tabpfn-25),
[Rubachev 2025 — On Finetuning Tabular Foundation Models](#on-finetuning), and
[Kolberg 2026 — TabPFN-Wide](#tabpfn-wide).

## Overview table

| Year | Authors | Title | One-line contribution | PDF |
|------|---------|-------|-----------------------|-----|
| 2021 | Müller et al. | Transformers Can Do Bayesian Inference | Foundational PFN paper: a transformer trained on synthetic samples from a prior approximates the prior's posterior in-context. | [pdf](../papers/2021_Muller_et_al._Transformers_Can_Do_Bayesian_Inference.pdf) |
| 2023 | Hollmann et al. | TabPFN — A Transformer That Solves Small Tabular Classification Problems in a Second | First TabPFN: PFN trained on a tabular SCM prior, beats AutoML baselines on small datasets. | [pdf](../papers/2023_Hollmann_et_al._TabPFN_A_Transformer_That_Solves_Small_Tabular_Classification_Problems_in_a_Second.pdf) |
| 2023 | Müller et al. | PFNs4BO — In-Context Learning for Bayesian Optimization | PFN as a drop-in surrogate for Gaussian-Process BO. | [pdf](../papers/2023_Muller_et_al._PFNs4BO_In_Context_Learning_for_Bayesian_Optimization.pdf) |
| 2024 | Breugel and Schaar | Why Tabular Foundation Models Should Be a Research Priority | Position paper: tabular FMs are an under-invested high-leverage area. | [pdf](../papers/2024_Breugel_and_Schaar_Why_Tabular_Foundation_Models_Should_Be_a_Research_Priority.pdf) |
| 2024 | Helli et al. | Drift-Resilient TabPFN | Trains TabPFN with a drift-injecting synthetic prior, generalises better under distribution shift. | [pdf](../papers/2024_Helli_et_al._Drift_Resilient_TabPFN_In_Context_Learning_Temporal_Distribution_Shifts_on_Tabular_Data_1.pdf) |
| 2024 | Hoo et al. | The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models | Show that TabPFN beats ARIMA / classical TS baselines when forecasting is framed as tabular regression on lag-features. | [pdf](../papers/2024_Hoo_et_al._The_Tabular_Foundation_Model_TabPFN_Outperforms_Specialized_Time_Series_Forecasting_Models_Based_on.pdf) |
| 2024 | Rundel et al. | Interpretable Machine Learning for TabPFN | Adapts SHAP / partial-dependence / interaction analysis to TabPFN's in-context inference path. | [pdf](../papers/2024_Rundel_et_al._Interpretable_Machine_Learning_for_TabPFN.pdf) |
| 2025 | Garg et al. | **Real-TabPFN** — Improving Tabular Foundation Models via Continued Pre-training With Real-World Data | **The recipe we follow.** Continue-pretrains TabPFNv2 on 71 curated real datasets; +0.022 ROC-AUC on the OpenML AutoML benchmark. | [pdf](../papers/2025_Garg_et_al._Real_TabPFN_Improving_Tabular_Foundation_Models_via_Continued_Pre_training_With_Real_World_Data.pdf) |
| 2025 | Hollmann et al. | Accurate predictions on small data with a tabular foundation model | The TabPFNv2 paper (Nature). Production-grade architecture with alternating-attention, NaN handling, ensemble preprocessing. | [pdf](../papers/2025_Hollmann_et_al._Accurate_predictions_on_small_data_with_a_tabular_foundation_model.pdf) |
| 2025 | Liu and Ye | TabPFN Unleashed — A Scalable and Effective Solution to Tabular Classification Problems | Inference-time tricks (stratified context, bootstrap, query subsampling) that push v2 past its 10k-row limit. | [pdf](../papers/2025_Liu_and_Ye_TabPFN_Unleashed_A_Scalable_and_Effective_Solution_to_Tabular_Classification_Problems.pdf) |
| 2025 | Müller et al. | Position — The Future of Bayesian Prediction Is Prior-Fitted | Position paper: PFNs as a unifying framework for approximate Bayesian inference. | [pdf](../papers/2025_Muller_et_al._Position_The_Future_of_Bayesian_Prediction_Is_Prior_Fitted.pdf) |
| 2025 | Pfefferle et al. | nanoTabPFN — A Lightweight and Educational Reimplementation of TabPFN | TabPFN training loop in <900 lines; the cleanest reference implementation publicly available. | [pdf](../papers/2025_Pfefferle_et_al._nanoTabPFN_A_Lightweight_and_Educational_Reimplementation_of_TabPFN.pdf) |
| 2025 | Qu et al. | TabICL — A Tabular Foundation Model for In-Context Learning on Large Data | Hierarchical attention TabPFN-competitor scaling to 500 k-row tables. | [pdf](../papers/2025_Qu_et_al._TabICL_A_Tabular_Foundation_Model_for_In_Context_Learning_on_Large_Data.pdf) |
| 2025 | Robertson et al. | Do-PFN — In-Context Learning for Causal Effect Estimation | PFN trained to predict ``do``-interventions; in-context causal effect estimation. | [pdf](../papers/2025_Robertson_et_al._Do_PFN_In_Context_Learning_for_Causal_Effect_Estimation.pdf) |
| 2025 | Robertson et al. | FairPFN — A Tabular Foundation Model for Causal Fairness | PFN with explicit protected-attribute structure for counterfactual fairness audits. | [pdf](../papers/2025_Robertson_et_al._FairPFN_A_Tabular_Foundation_Model_for_Causal_Fairness.pdf) |
| 2025 | Rubachev et al. | **On Finetuning Tabular Foundation Models** | Empirical study: fine-tuning TabPFN with full / LoRA / prefix-tuning. Hyperparameter ranges that work. | [pdf](../papers/2025_Rubachev_et_al._On_Finetuning_Tabular_Foundation_Models_1.pdf) |
| 2025 | Tanna et al. | TabTune — A Unified Library for Inference and Fine-Tuning Tabular Foundation Models | Common API across TabPFN, TabICL, TabDPT for fair head-to-head comparison. | [pdf](../papers/2025_Tanna_et_al._TabTune_A_Unified_Library_for_Inference_and_Fine_Tuning_Tabular_Foundation_Models.pdf) |
| 2025 | Ye et al. | A Closer Look at TabPFN v2 — Understanding Its Strengths and Extending Its Capabilities | Empirical analysis identifying v2 weaknesses and proposing patches that became v2.5 specialist checkpoints. | [pdf](../papers/2025_Ye_et_al._A_Closer_Look_at_TabPFN_v2_Understanding_Its_Strengths_and_Extending_Its_Capabilities.pdf) |
| 2025 | Zhang et al. | Mitra — Mixed Synthetic Priors for Enhancing Tabular Foundation Models | A "mixed" synthetic prior interpolating between TabPFN's and ForestPFN's priors. | [pdf](../papers/2025_Zhang_et_al._Mitra_Mixed_Synthetic_Priors_for_Enhancing_Tabular_Foundation_Models.pdf) |
| 2025 | Zhang et al. | TabPFN — One Model to Rule Them All | Survey-style win aggregation across many domains. | [pdf](../papers/2025_Zhang_et_al._TabPFN_One_Model_to_Rule_Them_All.pdf) |
| 2026 | Grinsztajn et al. | **TabPFN-2.5** — Advancing the State of the Art in Tabular Foundation Models | Successor architecture (18–24 layers, 50 k×2000 limit) and the family of v2.5 checkpoints. | [pdf](../papers/2026_Grinsztajn_et_al._TabPFN_2.5_Advancing_the_State_of_the_Art_in_Tabular_Foundation_Models.pdf) |
| 2026 | Hoo et al. | From Tables to Time — Extending TabPFN-v2 to Time Series Forecasting | Native time-axis attention version of TabPFN. | [pdf](../papers/2026_Hoo_et_al._From_Tables_to_Time_Extending_TabPFN_v2_to_Time_Series_Forecasting.pdf) |
| 2026 | Klein and Hoffart | Position — Foundation Models for Tabular Data within Systemic Contexts Need Grounding | Position paper from SAP: tabular FMs trained on isolated tables miss the operational context (business rules, code, data models) that gives data meaning. Proposes Semantically Linked Tables (SLT) and FMSLT as a new model class. | [pdf](../papers/2026_Klein_and_Hoffart_Position_Foundation_Models_for_Tabular_Data_within_Systemic_Contexts_Need_Grounding.pdf) |
| 2026 | Kolberg et al. | **TabPFN-Wide** — Continued Pre-Training for Extreme Feature Counts | Continued-pretraining recipe for high-dim datasets; FeatureAgglomeration template. | [pdf](../papers/2026_Kolberg_et_al._TabPFN_Wide_Continued_Pre_Training_for_Extreme_Feature_Counts.pdf) |
| 2026 | Ma et al. | Foundation Models for Causal Inference via Prior-Data Fitted Networks | Unified causal-PFN framework; Do-PFN + FairPFN at scale. | [pdf](../papers/2026_Ma_et_al._Foundation_Models_for_Causal_Inference_via_Prior_Data_Fitted_Networks.pdf) |
| 2026 | Ma et al. | TabDPT — Scaling Tabular Foundation Models on Real Data | Real-data-only TabPFN competitor; retrieval-based self-supervision on OpenML. | [pdf](../papers/2026_Ma_et_al._TabDPT_Scaling_Tabular_Foundation_Models_on_Real_Data.pdf) |
| 2026 | Qu et al. | TabICLv2 — A better, faster, scalable, and open tabular foundation model | Improved TabICL with bigger context limit and open weights. | [pdf](../papers/2026_Qu_et_al._TabICLv2_A_better_faster_scalable_and_open_tabular_foundation_model.pdf) |
| 2026 | Grinsztajn et al. | **TabPFN-3** — Technical Report | **The successor we will eventually re-base on.** New three-stage architecture (column-wise → row-wise → ICL), scales to 1M rows on a single H100, many-class attention decoder, "Thinking" test-time-compute mode. Synthetic-prior only, +200 Elo over TabPFN-2.6 on TabArena-medium. | [pdf](../papers/2026_Grinsztajn_et_al._TabPFN_3_Technical_Report.pdf) |

---

## 2021 — Müller et al. — Transformers Can Do Bayesian Inference

**arXiv:** [2112.10510](https://arxiv.org/abs/2112.10510) ·
**PDF:** [open](../papers/2021_Muller_et_al._Transformers_Can_Do_Bayesian_Inference.pdf)

**Where it fits.** The bedrock of the entire PFN / TabPFN line.
Müller and colleagues introduce *Prior-Fitted Networks* (PFNs):
transformers that, after a single training phase on synthetic
datasets sampled from a Bayesian prior, perform approximate
posterior inference *in context* on any new dataset drawn from
that prior — without any parameter update.

**What it contains.** The training recipe: at each step, sample a
prior dataset (an arbitrary stochastic process specified by the
researcher), give the transformer the labelled context plus an
unlabelled query point, and use cross-entropy loss to push the
network's prediction toward the true label sampled from the
prior. The authors prove (under capacity and prior-coverage
assumptions) that the trained network's predictive distribution
converges to the true Bayesian posterior of the underlying prior.
They demonstrate this on synthetic Gaussian-Process regression,
on Bayesian neural-network regression, and on small classification
problems sampled from a structured-causal-model prior — the
beginnings of what would become TabPFN.

The crucial conceptual contribution: there is no learning at
*inference* time. Once trained, the network is a fixed function;
calling it on a new dataset is one forward pass. This sidesteps
both the brittleness of MCMC sampling for non-conjugate priors
and the slowness of variational fitting for every new task.

**For CreditPFN.** This paper is *why* TabPFN works at all and
*why* continued pretraining changes the implicit prior rather
than the inference algorithm. When we continue-pretrain on
credit-risk data, we are not running a new inference algorithm;
we are nudging the synthetic prior the network has internalised
toward credit-risk-flavoured DGPs. Cite this paper when an
external reviewer asks "but is this still Bayesian?". Yes —
it is approximate posterior inference under a learned prior.

---

## 2023 — Hollmann et al. — TabPFN

**arXiv:** [2207.01848](https://arxiv.org/abs/2207.01848) ·
**PDF:** [open](../papers/2023_Hollmann_et_al._TabPFN_A_Transformer_That_Solves_Small_Tabular_Classification_Problems_in_a_Second.pdf)

**Where it fits.** The first TabPFN. Applies the PFN recipe to
small tabular classification with a structured-causal-model
prior tailored for tabular data.

**What it contains.** A transformer takes
``(X_context, y_context, X_query) → ŷ_query`` in one forward
pass. The prior generator samples small structural causal models
that specify a joint distribution over features and a target;
millions of these are sampled and the transformer is trained to
predict held-out targets given context. At inference time, no
gradient updates: a held-out test set is just appended to the
training set in the input, the transformer marginalises over its
implicit prior, and predictions come out one forward pass later.

Empirical headline: on the OpenML-CC18 benchmark restricted to
small datasets (≤ 1k rows, ≤ 100 features, ≤ 10 classes),
TabPFN beats well-tuned XGBoost, Random Forest, and a tuned
AutoGluon stack — in a *second* of inference where AutoGluon
takes minutes. The paper also discusses calibration (TabPFN's
predictive probabilities are sharp and well-calibrated) and
robustness to small-sample noise.

**For CreditPFN.** The architectural ancestor of v2 / v2.5 / v2.6.
The 1k-row / 100-feature limits of v1 are why the field needed v2;
those limits would not have allowed our PD/LGD datasets (some are
hundreds of thousands of rows, some are thousands of features).
Read the prior-generator section of this paper for the
intuition about *what* the synthetic prior is — that's the
distribution we're nudging during continued pretraining.

---

## 2023 — Müller et al. — PFNs4BO

**arXiv:** [2305.17535](https://arxiv.org/abs/2305.17535) ·
**PDF:** [open](../papers/2023_Muller_et_al._PFNs4BO_In_Context_Learning_for_Bayesian_Optimization.pdf)

**Where it fits.** Sister paper to TabPFN — applies PFNs to a
different domain (Bayesian-optimisation surrogate models).

**What it contains.** A PFN trained on synthetic
(configuration, value) pairs from a hyperparameter-search-style
prior. At inference time, given a few past evaluations of an
unknown objective, the PFN predicts the expected value at
unseen configurations. Used as a drop-in replacement for
Gaussian-Process surrogates inside an acquisition-function loop.
The advantages over GPs: no kernel choice, no quadratic-time
fitting, and the ability to encode arbitrary prior knowledge by
designing the prior generator.

**For CreditPFN.** Tangential. Useful only if we later wrap a
PFN around our own hyperparameter search for the
continued-pretraining loop (e.g. searching over learning rate,
number of epochs, and choice of base checkpoint).

---

## 2024 — Breugel and Schaar — Why Tabular Foundation Models Should Be a Research Priority

**arXiv:** [2405.01147](https://arxiv.org/abs/2405.01147) ·
**PDF:** [open](../papers/2024_Breugel_and_Schaar_Why_Tabular_Foundation_Models_Should_Be_a_Research_Priority.pdf)

**Where it fits.** Position / agenda paper. Pre-dates Real-TabPFN
and TabPFN-2.5; argues that tabular ML is an under-invested
research area despite tabular data being economically dominant.

**What it contains.** A diagnosis of why tabular ML lags behind
NLP and computer vision in foundation-model adoption. Five
proposed obstacles: (1) data heterogeneity (every tabular
dataset has its own schema), (2) the no-pretraining norm
(researchers train models from scratch per dataset), (3) the
absence of a TabArena-equivalent that the field treats as
canonical, (4) the absence of a public, large, high-quality
real-data corpus comparable to ImageNet or LAION, (5) the
absence of a commercial player whose interests align with
publishing such a corpus. The paper sketches research
directions for each.

**For CreditPFN.** Useful framing for the introduction of a
thesis or paper. The "no public real-data corpus" point is the
exact gap our 3000-dataset purchase will close in the credit-risk
sub-domain.

---

## 2024 — Helli et al. — Drift-Resilient TabPFN

**arXiv:** [2411.10634](https://arxiv.org/abs/2411.10634) ·
**PDF:** [open](../papers/2024_Helli_et_al._Drift_Resilient_TabPFN_In_Context_Learning_Temporal_Distribution_Shifts_on_Tabular_Data_1.pdf)

**Where it fits.** TabPFN variant trained for distribution shift
— realistic temporal evolution of the data-generating process,
rather than i.i.d. test data.

**What it contains.** Modifies TabPFN's synthetic prior generator
to inject continuous distribution shifts between context rows
and query rows: covariate drift (input distribution moves over
time), prior drift (label-marginal moves), and concept drift
(conditional ``P(y|x)`` moves). A neural-network sub-component
parametrises the drift dynamics, and the sampler interleaves
"early" rows (context) with "late" rows (query) so the model
learns to extrapolate the drift function from the context.

The empirical message: a TabPFN trained with this drift-aware
prior generalises noticeably better when the test distribution
differs from training by a continuous shift. The vanilla TabPFN
prior is implicitly i.i.d.; under drift, it cannot recover.

**For CreditPFN.** Highly relevant. Credit-risk data drifts hard
across macroeconomic regimes — the 2008 crisis and 2020 pandemic
both produced massive shifts in PD distributions. A
CreditPFN that combines (i) Real-TabPFN-style continued
pretraining on a credit corpus with (ii) drift-aware training-time
augmentation would, in principle, generalise better to
out-of-cycle defaults than a plain Real-TabPFN replication.
Worth folding the drift-aware augmentation into ``src/train/``
behind a config flag.

---

## 2024 — Hoo et al. — TabPFN Outperforms Specialized Time Series Forecasting Models

**arXiv:** [2407.05393](https://arxiv.org/abs/2407.05393) ·
**PDF:** [open](../papers/2024_Hoo_et_al._The_Tabular_Foundation_Model_TabPFN_Outperforms_Specialized_Time_Series_Forecasting_Models_Based_on.pdf)

**Where it fits.** Application paper rather than a method paper.
Shows that TabPFN beats classical time-series forecasters when
forecasting is reframed as cross-sectional regression on
lag-features.

**What it contains.** Pipeline: take a univariate time series,
build features (lags, calendar features like day-of-week,
rolling mean and rolling std with various window sizes), and
feed each (features → next-step value) pair to TabPFNRegressor.
On M4 / M5-style benchmarks, this beats ARIMA and several deep
TS baselines (DeepAR, N-BEATS) — without any TS-specific
architecture.

The narrative is that TabPFN's flexibility on small datasets
and absence of architectural priors lets it discover the right
forecast structure from a few hundred lag-feature rows, where
deep TS models need much more data to converge.

**For CreditPFN.** Mostly out of scope — credit-risk modelling
is typically cross-sectional (per loan / per borrower) rather
than time-series forecasting. But the result validates the
"TabPFN as a default tabular regressor" narrative we are
building on.

---

## 2024 — Rundel et al. — Interpretable Machine Learning for TabPFN

**arXiv:** [2403.10923](https://arxiv.org/abs/2403.10923) ·
**PDF:** [open](../papers/2024_Rundel_et_al._Interpretable_Machine_Learning_for_TabPFN.pdf)

**Where it fits.** Interpretability tooling for TabPFN.

**What it contains.** Adapts the standard interpretability
toolkit — SHAP values, partial-dependence plots, feature
interaction analysis — to TabPFN's in-context inference path.
Key challenge addressed: SHAP for TabPFN cannot use the standard
"model-as-black-box" approach because querying with a single row
plus the entire training set is the unit of inference, not just
the test row. The paper proposes adaptations that account for
the in-context structure and shows that the resulting
attributions are stable and consistent with expert intuition on
several benchmark datasets.

The headline argument is that TabPFN is *no harder* to
interpret than a GBDT once the right tooling is in place.

**For CreditPFN.** Important downstream. Credit-risk modelling
has regulatory interpretability requirements (Basel III, EBA
guidelines on internal ratings-based approaches), and any
production deployment of CreditPFN must produce
loan-by-loan attributions that map cleanly to credit-bureau
inputs. For the thesis evaluation chapter, we will need to
show that CreditPFN explanations are at parity with what
production banks currently report from logistic-regression /
GBDT scorecards.

---

<a id="real-tabpfn"></a>

## 2025 — Garg et al. — Real-TabPFN

**arXiv:** [2507.03971](https://arxiv.org/abs/2507.03971) ·
**PDF:** [open](../papers/2025_Garg_et_al._Real_TabPFN_Improving_Tabular_Foundation_Models_via_Continued_Pre_training_With_Real_World_Data.pdf)

**Where it fits. The recipe we are following.** Continued
pretraining of TabPFNv2 on a curated set of real-world tables
from OpenML and Kaggle. The closest published methodology to
CreditPFN — same pipeline, different domain.

**What it contains.** *(I read pages 1–6 of this PDF directly via*
*the* ``pdftotext`` *command, so the details below are quoted from*
*the paper rather than reconstructed from memory.)*

* **Method.** Take the synthetic-only TabPFNv2 checkpoint and
  continue pretraining on a hand-curated corpus of 71 datasets
  (≥ 10 000 rows each, mixture of OpenML and Kaggle).
  *Minimal* preprocessing: ``OrdinalEncoder`` for categoricals,
  and if the target has more than 10 classes, retain the nine
  most common and merge the remainder into a tenth "other".
  No imputation, no scaling beyond what TabPFN does internally.
* **Data contamination protocol** (their §3) — five-tier filter:
  (1) only datasets > 10k samples (every evaluation dataset is
  smaller, so size alone separates pretrain from eval);
  (2) cross-reference IDs / names / shapes;
  (3) cross-reference column names;
  (4) row hashes;
  (5) column hashes;
  (6) manual metadata inspection.
  This is exactly what our :mod:`src.data.dedup` implements.
* **Headline result.** On 29 datasets from the OpenML AutoML
  Benchmark, Real-TabPFN improves normalised ROC-AUC by +0.022
  vs. default TabPFNv2 (Wilcoxon signed-rank ``p = 0.0045``).
  The Wilcoxon test confirms this is a per-dataset improvement,
  not just an average bump driven by a few outliers.
* **Ablations.** OpenML alone: +0.019. Kaggle alone: +0.015.
  Union: +0.022 — heterogeneous sources are complementary.
  CommonCrawl as a continued-pretraining corpus *hurts*
  performance (the average dataset there is only ~100 rows ×
  7 features — too small to give TabPFN useful signal).
  GitTables (avg ~1000 rows × 9 features) helps, but less than
  OpenML+Kaggle (avg 10k–100k rows × 10s of features).
* **Context-size scaling.** Continued-pretraining gains grow with
  the context size used during continued pretraining: 2k → 20k
  context yields a monotonic improvement curve (Figure 4 in
  the paper).

**For CreditPFN.** The blueprint. Our contributions versus
Real-TabPFN are:

1. **Domain.** Credit risk instead of generic. Their corpus is
   "any real-world table they could find that wasn't in the eval
   set". Ours is "credit-risk tables specifically".
2. **Scale.** 3000 datasets vs. their 71. Whether the in-context
   prior shift is a function of corpus size is exactly what our
   ablations will measure.
3. **Two parallel tracks.** PD (classification) and LGD
   (regression) instead of one classification objective.
4. **Dedup protocol.** Replicated faithfully — Stage 1 of our
   pipeline implements all five of their checks plus three
   extras (rounded-row hash, subset detection, fuzzy column-name
   matching).

The paper's source code is **not** public; the methods section
plus this summary is what we have to work from. When a detail is
ambiguous (e.g. exact context-size schedule), we err on the side
of replicating their reported headline numbers as a sanity check
before scaling up to credit-specific corpora.

---

<a id="tabpfn-v2-nature"></a>

## 2025 — Hollmann et al. — Accurate predictions on small data with a tabular foundation model

**Journal:** *Nature*, 2025 ·
**PDF:** [open](../papers/2025_Hollmann_et_al._Accurate_predictions_on_small_data_with_a_tabular_foundation_model.pdf)

**Where it fits.** The TabPFNv2 paper. Production-grade release
that 100×s the scaling limits of v1 and ships in *Nature*.

**What it contains.** A re-architected v2 with alternating
sample-attention / feature-attention layers, scaling to
~10 000 rows × 500 features (orders of magnitude more than v1).
Headline ingredients:

* **Synthetic prior expansion.** Far broader and deeper than v1's
  prior; covers richer noise distributions, more diverse SCMs,
  and inputs with realistic categorical / numerical mixes.
* **Architecture.** Stacked transformer with two attention
  patterns interleaved per layer: "sample attention" treating
  each row as a token, and "feature attention" treating each
  column as a token. The interleaving lets the model reason
  about both row-level patterns (similarity, density) and
  column-level patterns (interactions, redundancy) in one pass.
* **NaN handling.** Built into the encoder via
  ``NanHandlingEncoderStep`` — explicit indicator features plus
  learned default replacements. Categorical handling via an
  internal ``OrdinalEncoder`` fitted at ``.fit(X, y)`` time.
* **Inference-time ensemble.** A pool of preprocessing
  configurations (``PowerTransformer``, ``QuantileTransformer``,
  ``RobustScaler``, ``SquashingScaler``, ``"none"``); each
  estimator in the ensemble cycles through one configuration,
  yielding diverse predictions that are averaged at inference.
* **Headline results.** Beats AutoGluon, CatBoost, XGBoost on
  the TabArena benchmark across the full ≤ 10k-row range.

**For CreditPFN.** The architecture we instantiate. Most of our
``sanitize.py`` design decisions flow directly from how v2 is
engineered:

* Don't pre-winsorise (``OUTLIER_REMOVAL_STD = 12.0`` is internal).
* Don't pre-apply power / quantile transforms (the inference
  ensemble does that, with diversity).
* Preserve NaNs (``NanHandlingEncoderStep`` handles them).
* Don't pre-normalise the regression target
  (``RegressorBatch.znorm_space_bardist_`` does that, and inverts
  at predict time).

---

## 2025 — Liu and Ye — TabPFN Unleashed

**arXiv:** [2502.02527](https://arxiv.org/abs/2502.02527) ·
**PDF:** [open](../papers/2025_Liu_and_Ye_TabPFN_Unleashed_A_Scalable_and_Effective_Solution_to_Tabular_Classification_Problems.pdf)

**Where it fits.** Adaptation method for TabPFN. Re-frames TabPFN's
weaknesses through a formal **bias–variance decomposition** of its
generalisation error, then proposes a method (**BETA**: Bagging
and Encoder-based Fine-tuning for TabPFN Adaptation) that attacks
both sides simultaneously.

**What it contains.** The authors observe that previous TabPFN-
improvement papers each address either bias or variance, never
both — and the resulting methods leave performance on the table.
BETA combines two complementary mechanisms.

* **Bias reduction via a lightweight learned encoder.** A small
  parameter-efficient adapter sits between the raw features and
  TabPFN's internal embedding, mapping datasets of arbitrary
  dimensionality into multiple fixed-dimensional latent
  representations. This both addresses TabPFN's ~500-feature soft
  cap and lets the model adapt to the downstream task during
  fine-tuning.
* **Variance reduction via Batch Ensemble plus bootstrapped
  sampling.** Multiple lightweight encoders run in parallel with
  parameter sharing (à la Wen 2020 / Gorishniy 2025), each fed a
  bootstrap-sampled context. Their predictions are averaged,
  smoothing out idiosyncrasies of any single sampled context set.

The method also integrates with Error-Correcting Output Codes
(ECOC) to handle multiclass tasks with > 10 classes (which v2
struggles with). Evaluated on 200+ benchmark classification
datasets where BETA either outperforms or matches state of the
art while remaining computationally lightweight.

**For CreditPFN.** Tangential to our continued-pretraining stage.
The parameter-efficient adapter pattern would be interesting if
our 3000-dataset corpus ever hits feature-count ceilings that
``FeatureAgglomeration`` doesn't smoothly handle, and the
variance-reduction tricks complement what the official TabPFN
package already does at inference time (`AutoTabPFN` ensembling).
We may pick up the bootstrap-context idea for evaluation.

---

## 2025 — Müller et al. — Position: The Future of Bayesian Prediction Is Prior-Fitted

**arXiv:** [2502.05489](https://arxiv.org/abs/2502.05489) ·
**PDF:** [open](../papers/2025_Muller_et_al._Position_The_Future_of_Bayesian_Prediction_Is_Prior_Fitted.pdf)

**Where it fits.** Position paper / manifesto from the original
PFN authors.

**What it contains.** Argues that PFNs are the natural successor
to MCMC and variational Bayes for *predictive* Bayesian inference
(as opposed to posterior estimation per se). The argument:

* Inference algorithms for non-conjugate priors are slow and
  brittle.
* PFNs amortise inference into one forward pass, with arbitrary
  user-specified priors.
* The remaining bottlenecks are (a) prior design and (b) corpora
  of real data to validate against synthetic priors — *not*
  inference algorithms.

The paper sketches research directions: prior-design libraries,
PFN-as-baseline for new probabilistic methods, and a unified
benchmark across regression / classification / forecasting / BO.

**For CreditPFN.** Useful for thesis context. Frames the
"design a better prior for credit-risk" project as a fundamentally
Bayesian-inference contribution rather than just hyperparameter
tuning.

---

## 2025 — Pfefferle et al. — nanoTabPFN

**arXiv:** [2510.13129](https://arxiv.org/abs/2510.13129) ·
**PDF:** [open](../papers/2025_Pfefferle_et_al._nanoTabPFN_A_Lightweight_and_Educational_Reimplementation_of_TabPFN.pdf)

**Where it fits.** Educational reference implementation —
TabPFN distilled to ~900 lines of clear PyTorch.

**What it contains.** A complete working PFN training loop
(synthetic prior dump, model, optimiser, training loop, eval),
designed to be readable end-to-end. Specifically:

* A synthetic-data HDF5 prior dump (300k×150×5 in the demo).
* The TabPFN-style model — alternating attention between samples
  and between features, plus a target-embedding head.
* A ``train()`` function that iterates over the prior dump,
  computes cross-entropy on held-out query labels, and applies
  AdamW with gradient-norm clipping.
* A small inference wrapper that exposes a sklearn-style
  ``fit`` / ``predict_proba`` interface.

The paper accompanies the code with an exposition of the PFN
recipe at a level appropriate for a graduate ML class.

**For CreditPFN.** Critical resource. The training loop in
``repositories/NanoTabPFN.txt`` is the structural template for
our ``src/train/train.py`` — we'll adapt the loop to iterate
over our cached real datasets instead of a synthetic prior dump.

---

## 2025 — Qu et al. — TabICL

**arXiv:** [2502.05564](https://arxiv.org/abs/2502.05564) ·
**PDF:** [open](../papers/2025_Qu_et_al._TabICL_A_Tabular_Foundation_Model_for_In_Context_Learning_on_Large_Data.pdf)

**Where it fits.** Direct competitor to TabPFN. Different
architectural approach; scales to ~500 k-row tables natively.

**What it contains.** TabICL ("Tabular In-Context Learning")
adopts a **two-stage** architecture instead of TabPFNv2's flat
column-then-row alternation. First, each row is collapsed into a
single dense vector via a column-then-row attention block:
distribution-aware column-wise embedding (a Set-Transformer-style
operation that treats each column as a permutation-invariant set
of cell values, enabling cross-table transferability), then
within-row attention to model feature interactions, then a
[CLS]-token aggregation that produces a fixed-dimensional row
embedding. Second, ICL runs over these row embeddings — collapsing
the column dimension *before* in-context learning sidesteps
TabPFNv2's `O(n²m + nm²)` cost.

The pretraining adds a tree-based synthetic prior (decision-tree
ensembles) on top of the standard SCM prior to inject GBDT-flavoured
inductive biases, plus curriculum learning that scales the
pretraining dataset size from 1k → 60k rows. To handle > 10 classes
(the pretraining limit) the model uses hierarchical classification.
Empirically, on TabArena's 53 datasets above 10 k rows TabICL
**surpasses both TabPFNv2 and CatBoost**, while on smaller datasets
it matches TabPFNv2 at up to 10× faster inference.

**For CreditPFN.** Comparison baseline only — our continued-
pretraining work commits to the TabPFN family for continuity with
Real-TabPFN. But TabICL is the right reference architecture for
"what should the pipeline look like" if we ever push past v2's
50 000-row context cap. The set-transformer column embedding is
also a promising pattern for handling the 3000-dataset
heterogeneity at scale.

---

## 2025 — Robertson et al. — Do-PFN

**arXiv:** [2506.06039](https://arxiv.org/abs/2506.06039) ·
**PDF:** [open](../papers/2025_Robertson_et_al._Do_PFN_In_Context_Learning_for_Causal_Effect_Estimation.pdf)

**Where it fits.** First PFN published for causal-effect
estimation. Predates the unified CausalFM framework (Ma 2026)
by ~6 months and focuses specifically on **conditional
interventional distributions** — the answer to "what would the
outcome distribution look like if we intervened to set
treatment t?", given only observational data.

**What it contains.** Pretraining recipe:

* **Sample millions of SCMs.** Each SCM has explicit treatment
  variables, outcome variables, and confounders. Importantly,
  the prior covers a wide variety of causal structures including
  cases where unconfoundedness *fails*.
* **For each SCM, sample two datasets simultaneously**: one
  observational (no interventions) and one interventional (forced
  ``do(T = t)``). The transformer is given the *observational*
  dataset as context and is trained to predict the *interventional*
  outcomes — meta-learning the act of causal inference itself.
* **At inference time**, the model sees only an observational
  dataset and a query intervention; the network outputs an
  estimate of the causal effect with calibrated uncertainty.

The crucial property: Do-PFN does **not** require the user to
specify the causal graph, and does **not** rely on the
unconfoundedness assumption that classical methods (IPW, DR,
causal forests) need. The breadth of the SCM-prior means the
model has, in expectation, "seen" enough confounded cases during
training that it auto-corrects.

Beats classical IPW and Doubly Robust estimators on synthetic and
semi-synthetic causal-inference benchmarks.

**For CreditPFN.** Out of scope for the immediate work but
methodologically interesting follow-up. The killer question for
credit-risk regulation is "what's the causal effect of an
intervention (APR cut, forbearance, collateral tightening) on
PD/LGD?". Standard predictive CreditPFN cannot answer this
because correlations ≠ causation. A Do-PFN head trained on top
of CreditPFN's representations *could* — and would unlock entire
new classes of regulatory and policy-evaluation use-cases.

---

## 2025 — Robertson et al. — FairPFN

**arXiv:** [2506.07049](https://arxiv.org/abs/2506.07049) ·
**PDF:** [open](../papers/2025_Robertson_et_al._FairPFN_A_Tabular_Foundation_Model_for_Causal_Fairness.pdf)

**Where it fits.** Causal-PFN sibling of Do-PFN, this time aimed
at the **causal-fairness** problem rather than treatment-effect
estimation. Addresses the limitation that current causal-fairness
frameworks require the user to specify the correct causal model —
a tall ask in practice and a source of "fairwashing" when the
specified graph is wrong.

**What it contains.** Pretraining recipe:

* **Synthetic causal-fairness data prior.** Sparse MLPs
  represent SCMs whose root nodes include exogenous protected
  attributes (binary: race / gender / age). For each SCM, the
  paper samples both a *biased* dataset (with the protected
  attribute's full causal influence) and a *fair* counterpart
  obtained by removing the outgoing edges of the protected
  attribute — i.e. by simulating the counterfactual world in
  which the protected attribute had no causal influence on the
  outcome.
* **Training objective.** The transformer sees the biased
  observational dataset as context and is trained to predict
  the *fair* outcomes from the counterfactual world. Loss is
  computed against the fair targets, so the network learns to
  internally identify and remove the causal effect of the
  protected attribute.
* **At inference time**, FairPFN takes only the biased
  observational data and produces fairer predictions, integrating
  over the simplest causal explanations consistent with the
  observed bias. **No user-specified causal graph is needed.**

The paper demonstrates strong performance on hand-crafted SCMs
and several real-world fairness benchmarks (Adult, COMPAS,
German Credit, Law School Admissions), beating robust baselines
across most settings.

**For CreditPFN.** ★ Highly relevant for the regulatory chapter.
Credit-risk models are bound by anti-discrimination law (ECOA
in the US, the EU AI Act, the Basel-III IRB framework's fair-
lending guidance). A FairPFN-style audit on top of CreditPFN
would let us certify that any disparate impact in our model's
PD/LGD predictions is causally due to legitimate underwriting
factors — and not to a leaked proxy of a protected attribute via
ZIP code, employment sector, or similar. This is the exact
audit pattern banks are increasingly being asked to produce by
supervisors.

---

<a id="on-finetuning"></a>

## 2025 — Rubachev et al. — On Finetuning Tabular Foundation Models

**arXiv:** [2506.08982](https://arxiv.org/abs/2506.08982) ·
**PDF:** [open](../papers/2025_Rubachev_et_al._On_Finetuning_Tabular_Foundation_Models_1.pdf)

**Where it fits.** Empirical study of finetuning tabular FMs on
downstream datasets. The closest published reference for our
training-stage hyperparameter choices.

**What it contains.** Systematically benchmarks three finetuning
strategies — full gradient updates, LoRA-style low-rank
adapters, and prefix tuning — on TabPFN, TabICL, and TabDPT
across dozens of downstream tasks. Headline findings:

* **Full finetuning helps on datasets large enough to overcome
  the overfitting risk** (typically > 1000 rows). Below that,
  zero-shot in-context learning is hard to beat.
* **LoRA recovers most of the gain at a fraction of the parameter
  cost** — ~1–5 % of full-model parameters trained, typically
  90–95 % of the full-finetune improvement.
* **Stable learning rates** are in the 1e-5 to 1e-4 range. Higher
  rates destabilise; lower rates barely move from the pretrained
  point.
* **Epoch budgets** of 10–50 epochs are typical; longer
  schedules overfit on most datasets.

The paper also identifies pathological cases where finetuning
*hurts* (extremely small datasets, datasets with severe label
noise) and proposes early-stopping protocols to detect them.

**For CreditPFN.** Our continued pretraining is technically a
*multi-task* finetuning regime, which is a strict generalisation
of the single-dataset finetuning studied here. The hyperparameter
ranges (LR ~1e-5, ~30 epochs) reported in this paper are the
starting point for ``config/training.yaml`` once ``src/train/``
is implemented.

---

## 2025 — Tanna et al. — TabTune

**arXiv:** [2511.02802](https://arxiv.org/abs/2511.02802) ·
**PDF:** [open](../papers/2025_Tanna_et_al._TabTune_A_Unified_Library_for_Inference_and_Fine_Tuning_Tabular_Foundation_Models.pdf)

**Where it fits.** Software / benchmark paper from Lexsi Labs. A
unified, scikit-learn-compatible Python library that standardises
the entire workflow — inference, fine-tuning, evaluation — across
TabPFN, TabICL, ContextTab, OrionMSP, OrionBix, and other tabular
FMs.

**What it contains.** TabTune addresses four operational frictions
that have been slowing TFM adoption in practice:

* **Diverse preprocessing requirements** — each TFM expects its
  own data encoding (TabPFN wants numerically encoded categoricals
  consistent with its synthetic priors, TabICL wants set-transformer
  embeddings for categoricals, etc.). TabTune handles the per-model
  preprocessing internally.
* **Fragmented APIs and training protocols** — some models are
  zero-shot only, some support full SFT, some support PEFT
  (LoRA / prefix tuning). TabTune exposes a unified
  ``.fit()`` / ``.predict()`` / ``.evaluate()`` interface across
  all of them.
* **Evaluation gaps in deployment-relevant metrics** — beyond
  accuracy, the library ships built-in calibration (Expected
  Calibration Error, Maximum Calibration Error, Brier score) and
  fairness diagnostics (Statistical Parity Difference, Equalised
  Odds Difference, Equalised Opportunity Difference).
* **Model selection complexity** — a benchmarking module
  consistently ranks models on accuracy, calibration, fairness
  and resource efficiency on standard suites (TALENT, OpenML-CC18).

The library targets the "experimental bed" use-case: study how
zero-shot vs. PEFT vs. SFT trades off across calibration,
fairness, and compute, all under one harness.

**For CreditPFN.** Strong candidate for our `src/eval/` layer.
The ECE / MCE / Brier metrics directly answer the regulatory
question of whether CreditPFN's PD probabilities can be used as
calibrated risk scores in a Basel-III IRB context. The fairness
metrics partially overlap with what FairPFN-style auditing
provides. We will likely wrap our evaluation through TabTune
when comparing CreditPFN vs. TabPFN-2.6, TabICL, TabDPT, and
the published Real-TabPFN-2.5 checkpoint.

---

## 2025 — Ye et al. — A Closer Look at TabPFN v2

**arXiv:** [2502.17361](https://arxiv.org/abs/2502.17361) ·
**PDF:** [open](../papers/2025_Ye_et_al._A_Closer_Look_at_TabPFN_v2_Understanding_Its_Strengths_and_Extending_Its_Capabilities.pdf)

**Where it fits.** Mechanistic analysis of TabPFNv2 (rather than
a new architecture). Asks two questions: *how* does v2 cope so
well with heterogeneous tabular data, and *how* can its known
limits — 10 000 samples × 500 features × 10 classes — be lifted
without retraining? Three concrete findings.

**What it contains.**

1. **TabPFN v2 internalises attribute-token learning.** Unlike
   prior tabular methods that rely on word-vector–style attribute
   semantics or learn dataset-specific attribute tokens, v2 uses
   **randomly resampled attribute tokens at every inference call**.
   The paper's analysis shows that v2 still consistently infers
   attribute relationships *through ICL itself* — effectively
   integrating "what this column means" learning into the same
   forward pass that produces predictions. This is what lets v2
   transfer across datasets with different schemas and
   dimensionalities without per-dataset adaptation.
2. **TabPFN v2 can be repurposed as a feature extractor.** Using
   a leave-one-fold-out strategy that aligns training and test
   embeddings, the authors show v2 maps tabular instances into a
   *near-linearly-separable* embedding space. Training a plain
   linear model on those embeddings recovers most of v2's
   accuracy — meaning v2 is implicitly learning a strong general-
   purpose tabular representation, not just a classifier.
3. **Test-time divide-and-conquer mitigates the
   10 k × 500 × 10 limits.** Instead of retraining v2 on bigger
   data, the paper proposes post-hoc strategies analogous to LLM
   test-time scaling: hierarchical class decomposition for
   many-class tasks, feature-subset chunking for wide tables,
   bootstrap-context aggregation for large-N data. Empirical
   gains across all three limit regimes.

Several of these patches were rolled into the official Prior Labs
package as non-default v2.5 specialist checkpoints —
``_low-skew``, ``_quantiles``, ``_large-features-L``,
``_large-features-XL``, ``_large-samples`` — per the catalogue
in ``repositories/Huggingface TabPFN.txt``.

**For CreditPFN.** Two concrete uses. First, the feature-extractor
finding gives us a free tool: at evaluation time we can extract
v2 embeddings of credit-risk features and pass them to a downstream
linear / GBDT classifier as a sanity-check baseline ("how much of
CreditPFN's value-add is just better embeddings vs. better
in-context learning?"). Second, LGD targets are heavy-tailed (most
mass at LGD = 0 and LGD = 1, with sparse interior) — exactly the
distribution the ``_low-skew`` and ``_quantiles`` v2.5 specialist
checkpoints target. We should benchmark both as alternative LGD
bases.

---

## 2025 — Zhang et al. — Mitra: Mixed Synthetic Priors

**arXiv:** [2510.21204](https://arxiv.org/abs/2510.21204) ·
**PDF:** [open](../papers/2025_Zhang_et_al._Mitra_Mixed_Synthetic_Priors_for_Enhancing_Tabular_Foundation_Models.pdf)

**Where it fits.** Alternative pretraining recipe from Amazon /
AutoGluon team. Argues that the right way to improve TFMs is *not*
to add architectural complexity but to **design better synthetic
priors** — and proposes a principled framework for mixing them.
Pretrained model is open-source on HuggingFace as
``autogluon/mitra-classifier`` and ``autogluon/mitra-regressor``.

**What it contains.** The paper formalises three properties a good
synthetic prior should have, and operationalises each as a
measurable score:

1. **Standalone performance** — a TFM pretrained *only* on this
   prior should already do well on real data.
2. **Diversity** — a TFM pretrained on this prior should not
   easily overfit to its own distribution (i.e. the prior covers
   a wide range of generative mechanisms).
3. **Distinctiveness within a mixture** — data sampled from this
   prior should be hard for TFMs trained on *other* priors to
   predict, meaning it adds something the other priors miss.

The first is captured by a per-prior performance vector P; the
latter two by a "Generalisability Matrix" G with diagonal entries
measuring overfitting and off-diagonal entries measuring inter-
prior overlap. Using these criteria the authors select a final
mixture of **SCMs + tree-based priors** (gradient boosting,
random forest, decision tree, extra-tree). SCMs deliver
standalone performance and diversity; tree-based priors are
distinctive (TFMs pretrained on SCMs alone struggle on TBP-
generated data, so adding TBPs covers ground SCMs miss).

The mixture is model-agnostic: it improves both flat row-attention
architectures (à la TabPFN) and 2D cell-attention architectures.
**Mitra outperforms TabPFNv2 and TabICL on both classification and
regression benchmarks**, with better sample efficiency.

**For CreditPFN.** Complementary direction we are *not* taking —
we're enriching with real data, not redesigning the synthetic
prior. But:

* The "mixed synthetic prior + continued pretraining on credit
  data" combination is a clean ablation if we want to see whether
  prior-design and data-augmentation are additive.
* Mitra's open weights make it a strong **third comparison
  baseline** in our `src/eval/` (alongside TabPFN-2.6, TabICL,
  TabDPT). Particularly interesting for credit-risk: tree-based
  priors are exactly the inductive bias GBDTs use, so Mitra may
  be unusually competitive on credit-risk benchmarks where GBDTs
  are the default production model.

---

## 2025 — Zhang et al. — TabPFN: One Model to Rule Them All

**arXiv:** [2505.20003](https://arxiv.org/abs/2505.20003) ·
**PDF:** [open](../papers/2025_Zhang_et_al._TabPFN_One_Model_to_Rule_Them_All.pdf)

**Where it fits.** A statistician's appraisal of TabPFNv2,
written for a *Statistical Science*-flavoured audience rather
than an ICML one. Provides a careful re-derivation of how
TabPFN works as **approximate Bayesian inference**, and uses
that lens to identify several application domains in which
out-of-the-box TabPFN matches or beats domain-specific
state-of-the-art methods.

**What it contains.** Three concrete applications evaluated:

* **Semi-supervised parameter estimation** — using TabPFN with
  a small labelled set plus unlabelled rows as context. Beats
  specialised semi-supervised baselines on several benchmarks.
* **Prediction under covariate shift** — TabPFN automatically
  handles a moderate degree of distribution shift through its
  in-context calibration, with no covariate-shift-specific
  modifications. Competitive with importance-weighting and
  domain-adaptation baselines.
* **Heterogeneous treatment-effect estimation** — TabPFN
  predictions plug into S-learner / T-learner causal-inference
  frameworks and are competitive with Causal Forest, X-learner,
  and DR-learner.

A key theoretical observation: **TabPFN can adapt to both
nonparametric *and* parametric structure simultaneously**.
Sometimes outperforms LASSO even when the data is genuinely
sparse-linear (i.e. when LASSO's modelling assumptions are
correctly specified) — because TabPFN's implicit prior over
SCMs already covers sparse-linear data, plus everything else.
This blurs the classical bias-variance trade-off and makes
TabPFN attractive as a *default* tabular regressor / classifier
even in regimes where simple parametric models are usually
preferred.

**For CreditPFN.** Strong rhetorical citation: when we argue in
the thesis that TabPFN-as-baseline is a more principled
reference than XGBoost-as-baseline, this paper makes the case
for us in clean Bayesian language. The covariate-shift result
also lines up with what we want from a credit-risk model:
robustness across macroeconomic regimes without explicit
domain-adaptation engineering.

---

<a id="tabpfn-25"></a>

## 2026 — Grinsztajn et al. — TabPFN-2.5

**arXiv:** [2511.08667](https://arxiv.org/abs/2511.08667) ·
**PDF:** [open](../papers/2026_Grinsztajn_et_al._TabPFN_2.5_Advancing_the_State_of_the_Art_in_Tabular_Foundation_Models.pdf)

**Where it fits. The architecture we instantiate.** Successor to
v2: deeper (18–24 layers), bigger context limit (50 000 samples
× 2000 features), and crucially **ships the real-data-finetuned
variant as a default option**.

**What it contains.**

* **Architecture.** Transformer with TabPFNv2-like alternating
  attention with 18–24 layers, varying across the family of
  checkpoints (the small-features specialist is shallower, the
  large-features specialist is deeper).
* **Training data.** Synthetic-only base + a Real-TabPFN-style
  real-data continued-pretraining variant. The Real-TabPFN-2.5
  checkpoint uses 43 curated datasets listed in the paper's
  Appendix C.1 (the same recipe as Garg 2025 above, refined and
  scaled).
* **Checkpoints.** Per
  ``repositories/Huggingface TabPFN.txt:91-106``: ``_default`` is
  real-finetuned; ``_default-2`` is synthetic-only (the
  methodologically clean base for our continued pretraining);
  multiple specialist variants (``_large-features-L``,
  ``_large-features-XL``, ``_large-samples``, ``_low-skew``,
  ``_quantiles``, ``_real``, …).
* **Evaluation.** New SOTA on a proprietary benchmark, on
  TabArena, and on RealCause (a causal-inference benchmark
  where the regression variant of the model is repurposed).

Note that v2.6 (the immediate successor described on the
HuggingFace card and the TabPFN docs) was released after v2.5
*without* a corresponding paper. v2.6 reverts the
"default-is-real-finetuned" naming convention: its single
``_default`` checkpoint is again *synthetic-only*. The full
checkpoint provenance is in
[`docs/CHECKPOINTS.md`](CHECKPOINTS.md).

**For CreditPFN.** Our base architecture. The checkpoint-choice
question (v2.5 ``_default-2`` vs. v2.6 ``_default``) is a
training-stage hyperparameter we will benchmark, not a decision
fixed at the data-pipeline stage.

---

## 2026 — Hoo et al. — From Tables to Time

**arXiv:** [2501.02945](https://arxiv.org/abs/2501.02945) ·
**PDF:** [open](../papers/2026_Hoo_et_al._From_Tables_to_Time_Extending_TabPFN_v2_to_Time_Series_Forecasting.pdf)

**Where it fits.** A more thorough version of the 2024 forecasting
paper by the same authors. Reframes time-series forecasting as a
*tabular regression problem* and shows that the **unmodified**
pretrained TabPFN-v2 — paired with a lightweight temporal
featurisation — beats specialised time-series foundation models.
Released as **TabPFN-TS** at github.com/PriorLabs/tabpfn-time-series.

**What it contains.** The construction is simple but the result
is striking. Each time step becomes one row in a tabular
regression problem: features encode time progression
(running index), multi-scale seasonality (year, day-of-week,
hour-of-day, …), plus optional covariates (weather, economic
indicators, control inputs). The target is the observed value at
that time step. Forecasting then reduces to predicting future
rows whose temporal features are known in advance — and TabPFN-v2
predicts the entire forecast horizon **in one forward pass**.

Headline results:

* On the **fev-bench** benchmark (covariate-informed forecasting),
  TabPFN-TS achieves **state-of-the-art** at 11 M parameters.
* On **GIFT-Eval** (univariate forecasting), it is competitive
  with Chronos-Mini and **matches Chronos-Large** despite having
  ~65× fewer parameters.
* No time-series–specific pretraining. No fine-tuning. The
  pretrained TabPFN-v2 weights are used as-is.

The paper also includes mechanistic studies: how the model
exploits temporal structure (lag features matter most, calendar
features add a calibration boost), how forecasting quality
varies across tabular backbones (TabPFN-v2 ≫ TabPFNv1 ≫ tree
ensembles given the same featurisation), and how the model
handles distribution shifts at long horizons.

**For CreditPFN.** Out of scope for our PD/LGD pretraining work.
But it is the cleanest demonstration that **TabPFN-v2 generalises
beyond i.i.d. tabular regression** — useful when defending the
choice of TabPFN-v2 as our base architecture against reviewers who
might argue it's "only" a tabular regressor. Also relevant if we
ever extend CreditPFN to **default-curve forecasting** — predicting
the survival curve of a loan across months — where the same
featurisation trick would map directly onto our cached `.npz`
format.

---

## 2026 — Klein and Hoffart — Position: Foundation Models for Tabular Data within Systemic Contexts Need Grounding

**arXiv:** [2505.19825](https://arxiv.org/abs/2505.19825) ·
**PDF:** [open](../papers/2026_Klein_and_Hoffart_Position_Foundation_Models_for_Tabular_Data_within_Systemic_Contexts_Need_Grounding.pdf)

**Where it fits.** A *contrarian* position paper from SAP that
challenges the entire "isolated tables" framing of current tabular
foundation models — TabPFN, TabICL, TabDPT, Mitra all included.
Argues these models trained on individual tables (or even on
schema-level multi-table relations via GNNs) fundamentally miss
the **operational context** — the procedural logic, declarative
rules, and domain knowledge — that gives tabular data its meaning
in real-world enterprise systems.

**What it contains.**

* **The diagnosis.** Current tabular FMs assume "information
  completeness within tables" — that the information needed to
  predict an outcome is in the rows themselves. In an enterprise
  setting that's almost never true. A row like
  `(amount=4999, ..., approved=True)` next to
  `(amount=5000, ..., approved=False)` is fully explained by a
  rule somewhere in the codebase: ``if amount >= 5000:
  require_manager_approval()``. A purely statistical model trained
  on historical decisions might learn an *approximate* threshold
  near $4 800 and misclassify edge cases — while the actual rule
  is a hard `>=` boundary, sitting in source code that the model
  never sees.
* **The proposal: SLT + FMSLT.** *Semantically Linked Tables* —
  the relational data plus three layers of explicitly-modelled
  context: declarative business knowledge (data models, business
  objects, business rules, process models), procedural knowledge
  (agent logic in natural language, application logic as code),
  and world knowledge (general business concepts, types,
  relationships, implicit assumptions). A *Foundation Model for
  SLT* (FMSLT) is then a new model class that ingests **code as
  logic**, not as text — distinguishing "branching paths
  define decision boundaries" from mere token co-occurrence.
* **Two-phase training recipe.** (i) Pre-train on open-source
  code-data pairs and synthetic systems to learn business-logic
  mechanics. (ii) Apply zero-shot to proprietary enterprise data
  via in-context retrieval of the relevant code/rules.
* **Operational Turing Test.** A new benchmark proposed in the
  paper: an FMSLT passes if, given an enterprise dataset, it can
  predict outcomes that depend on rules expressed only in code, at
  parity with a system that has access to the explicit rule
  engine. The paper reviews recent enterprise-agent benchmarks
  (WorkArena++, TheAgentCompany, CRMArena-Pro, AgentArch, MLGym,
  COMPASS) and shows current LLM-based agents top out at 30–35%
  on tasks requiring implicit business knowledge — motivating the
  need for explicit operational grounding.

**For CreditPFN.** A *very* relevant cautionary tale, even though
we are not building an FMSLT. Two takeaways:

* **The honesty argument.** Credit-risk decisions inside production
  banks are *also* governed by code and rules — Basel-III IRB
  formulae, internal scorecards, risk-appetite thresholds set by
  the credit committee, ECOA fair-lending guardrails. A CreditPFN
  trained purely on historical PD/LGD outcomes will absorb the
  *consequences* of these rules but not the rules themselves. For
  any production deployment we should be honest about what
  CreditPFN *can* and *cannot* do: it predicts under the
  distribution of past underwriting decisions; it does not
  represent the underwriting policy itself.
* **The follow-up direction.** A natural extension of CreditPFN
  would be an FMSLT-style augmentation: feed in the documented
  underwriting policy (as code or as structured rules) alongside
  the tabular data, and train the model to use both. That's a
  thesis-extension topic, not the current data-pipeline work — but
  worth bookmarking for the discussion section of the eventual
  paper.

---

<a id="tabpfn-wide"></a>

## 2026 — Kolberg et al. — TabPFN-Wide

**arXiv:** [2511.21002](https://arxiv.org/abs/2511.21002) ·
**PDF:** [open](../papers/2026_Kolberg_et_al._TabPFN_Wide_Continued_Pre_Training_for_Extreme_Feature_Counts.pdf)

**Where it fits.** Continued-pretraining sibling to Real-TabPFN,
focused on the extreme-feature-count regime.

**What it contains.** Modifies TabPFN's training recipe to
include synthetic datasets with hundreds-to-thousands of
features (matching multi-omics, wide bureau-data, etc.), then
fine-tunes on real wide-feature datasets. Specifically:

* **Synthetic prior augmentation** — the SCM generator is
  extended to produce datasets where the number of features
  vastly exceeds the number of rows, which v2's prior almost
  never sampled.
* **Preprocessing pipeline** — uses
  ``FeatureAgglomeration(metric='euclidean', linkage='ward')``
  inside their preprocessing chain (their Appendix B is the
  direct source for the design we adopted in our ``sanitize.py``).
* **Released checkpoints** — ``_large-features-L`` (≤ 500
  features) and ``_large-features-XL`` (≤ 1000 features).

**For CreditPFN.** Very relevant. Two of our raw datasets exceed
2000 features (``0014.algorithmwatch`` has 2987,
``0011.loan_default`` has 770). Two options for handling them:
(a) cluster down to 128 with our own ``FeatureAgglomeration``
(current default), or (b) use the ``_large-features-XL``
checkpoint directly. We'll benchmark both in ``src/train/``.

---

## 2026 — Ma et al. — Foundation Models for Causal Inference via Prior-Data Fitted Networks

**arXiv:** [2506.10914](https://arxiv.org/abs/2506.10914) ·
**PDF:** [open](../papers/2026_Ma_et_al._Foundation_Models_for_Causal_Inference_via_Prior_Data_Fitted_Networks.pdf)

**Where it fits.** Unified causal-PFN framework — published at
ICLR 2026, supersedes Do-PFN and FairPFN by generalising both.
Introduces **CausalFM**: a general recipe for training PFN-based
foundation models that perform Bayesian causal inference across
multiple identification strategies in a single forward pass.

**What it contains.** Three layers of contribution.

1. **Theory: necessary criteria for valid SCM-priors for causal
   inference.** The paper formalises how to construct prior
   distributions over structural causal models such that the
   resulting PFN's in-context predictions are valid estimates of
   the causal quantity of interest. Identifies what breaks if you
   naïvely take TabPFN's SCM-prior and ask it to estimate a
   treatment effect.
2. **Method: CausalFM priors.** A novel family of SCM-priors
   parameterised by Bayesian neural networks, structured to
   respect the underlying causal-inference setting. Concretely:
   different priors for **back-door adjustment** (the
   conditioning-on-confounders case), **front-door adjustment**
   (the mediator case), and **instrumental-variable adjustment**
   (the natural-experiment case).
3. **Empirics: trained PFN models for CATE estimation.**
   Conditional Average Treatment Effect estimation across diverse
   benchmarks. CausalFM **outperforms current state-of-the-art
   CATE estimators** that are specifically trained for the task,
   while requiring no per-dataset training itself.

Key advantages over classical causal inference: (i) no retraining
per new dataset (in-context inference), (ii) principled
uncertainty quantification, (iii) the model auto-selects the
identification formula based on the observed data structure,
(iv) identifiability guarantees baked into the prior design.

**For CreditPFN.** Out of scope for the immediate work. Bookmarked
for follow-up causal questions in credit-risk: "what is the causal
effect of a 100 bp APR cut on probability of default?",
"would tightening collateral requirements have prevented these
defaults?", "does forbearance during a downturn causally improve
recovery rates?". A CausalFM-style head trained on top of a
CreditPFN backbone would let us answer those without retraining
per question.

---

## 2026 — Ma et al. — TabDPT

**arXiv:** [2410.18164](https://arxiv.org/abs/2410.18164) ·
**PDF:** [open](../papers/2026_Ma_et_al._TabDPT_Scaling_Tabular_Foundation_Models_on_Real_Data.pdf)

**Where it fits.** From Layer 6 AI (Toronto). Pretrained at the
**opposite extreme of the synthetic-vs-real spectrum** from
TabPFN: a transformer pretrained on *real* OpenML data only, with
no synthetic prior. Open weights on
[HuggingFace](https://huggingface.co/Layer6/TabDPT) and
inference + training code on
[GitHub](https://github.com/layer6ai-labs/TabDPT-inference).

**What it contains.** Two methodological pillars:

* **ICL retrieval + self-supervised learning.** At each training
  step the model samples a real OpenML table, masks part of it,
  and is trained to fill the masked part — a column-masking SSL
  objective in the spirit of BERT's masked-language-model. Each
  forward pass also retrieves an in-context support set from the
  same table, so the model learns to use the retrieved context
  as evidence (like LLM RAG). Trained model handles both
  classification and regression.
* **Scaling laws on real data.** The paper's empirical contribution
  is showing that **scaling both model size and pre-training data
  size yields consistent gains following power-laws** (Figure 1
  of the paper). This is the first demonstration of LLM-style
  scaling behaviour for tabular foundation models — and crucially,
  it works on *real* data, not just synthetic.

Evaluated on **OpenML-CC18** (classification) and **OpenML-CTR23**
(regression). TabDPT consistently matches or surpasses specialised
per-dataset baselines that get hyperparameter-tuned, at a fraction
of the deployment cost. Particularly strong in the few-shot
regime: with minimal semi-supervised modifications, TabDPT
outperforms specialised baselines on **10-shot classification**.

**For CreditPFN.** Important comparison baseline because it brackets
our project from the opposite end. Real-TabPFN (and CreditPFN) is
"synthetic-only pretraining + real-data continued pretraining";
TabDPT is "real-data pretraining from scratch with retrieval";
TabPFN-v2 is "synthetic-only pretraining". By comparing all three
on credit-risk data we can disentangle the contributions of
synthetic priors vs. real data vs. retrieval-augmentation. The
local code dump is at ``repositories/TabDPT.txt``; the published
training-data list (`tabdpt_datasets/data_splits/{cls,reg}_datasets.csv`)
must be cross-checked against our held-out evaluation set to
prevent contamination.

---

## 2026 — Qu et al. — TabICLv2

**arXiv:** [2602.11139](https://arxiv.org/abs/2602.11139) ·
**PDF:** [open](../papers/2026_Qu_et_al._TabICLv2_A_better_faster_scalable_and_open_tabular_foundation_model.pdf)

**Where it fits.** Direct successor to TabICL — full state-of-the-
art tabular FM, **fully open** (inference code, model weights,
synthetic-data engine, and pretraining code), explicitly framed
as a "let's democratize the recipe so the field can build on top"
release. Beats RealTabPFN-2.5 on TabArena even after RealTabPFN-2.5
is hyperparameter-tuned, ensembled, and fine-tuned on real data.

**What it contains.** Three pillars of contributions, each with
substantial detail.

1. **A novel synthetic-data generation engine** focused on
   *diversity*. Builds on TabICL's distribution-aware feature
   embeddings and adds new SCM construction primitives,
   tree-ensemble priors (à la Mitra), and a controllable
   difficulty curriculum. The release of this generator is part
   of the paper's open-science contribution.
2. **Architectural innovations.** The most interesting one for
   our work: a **scalable softmax** in the attention layers that
   solves the "attention fading" problem — as context length n
   grows, vanilla softmax's denominator grows too, flattening
   the attention distribution and preventing the model from
   focusing sharply. TabICLv2 scales attention logits by `s log n`
   (where s is a learnable per-head parameter), maintaining
   discriminative attention even at very large context sizes.
   This is what unlocks generalisation to **million-scale
   datasets in under 50 GB GPU memory**. Concurrent theoretical
   work (Chen 2025) confirms `log n` scaling is *necessary* to
   maintain attention sharpness as n grows.
3. **Optimisation protocol.** Replaces AdamW with the **Muon**
   optimizer (Jordan 2024), which converges faster on the
   transformer-style training loop. Combined with the new prior
   and architecture, this delivers TabICLv2's headline result:
   surpassing the strongest previously-published TFM
   (RealTabPFN-2.5) on TabArena and TALENT benchmarks **without
   any hyperparameter tuning**.

Comprehensive ablations quantify each contribution.

**For CreditPFN.** Comparison baseline. Three concrete uses:

* **Eval baseline.** Open weights mean we can drop TabICLv2 into
  our `src/eval/` benchmarking with no licence friction.
* **Open prior generator.** If we ever want to test "credit-risk
  augmented synthetic prior", TabICLv2's open generator is a
  better starting point than Prior Labs' (closed) v2.5/v2.6
  generators.
* **Scalable-softmax idea.** Transferable to TabPFN-v2 with
  modest engineering — could let our continued-pretrained
  CreditPFN handle larger downstream credit-risk datasets at
  inference time.

---

<a id="tabpfn-3"></a>

## 2026 — Grinsztajn et al. — TabPFN-3 (Technical Report)

**arXiv:** [2605.13986](https://arxiv.org/abs/2605.13986) ·
**PDF:** [open](../papers/2026_Grinsztajn_et_al._TabPFN_3_Technical_Report.pdf) ·
**Date:** May 12, 2026

**Where it fits.** Prior Labs' next-generation tabular
foundation model after TabPFN-2.5 / 2.6. A clean architectural
reset rather than a continued-pretraining variant: a three-stage
transformer that scales in-context learning to **one million
training rows on a single H100**, adds native many-class
classification, and introduces test-time-compute ("Thinking
mode") for accuracy at the cost of latency. Released under the
TABPFN-3.0 license (permissive for research and internal
evaluation); enterprise / API tier ships TabPFN-3-Plus.

**What it contains — architecture.** TabPFN-3 abandons the
TabPFN-2.x alternating row/feature attention in favour of a
three-stage design borrowed from Qu et al.'s TabICL line:

1. **Feature distribution embedding (column-wise).** Each
   column independently embedded by a transformer with
   inducing-point attention — sidesteps the O(n²) cost of full
   cross-row attention.
2. **Feature aggregation (row-wise).** Learned `cls` tokens
   attend to all features within each row; concatenated
   `cls`-hidden states form a single fixed-dimensional row
   embedding decoupled from the input feature count.
3. **In-context learning.** A TabPFN-v1-style ICL transformer
   on the row embeddings — sequence length proportional to rows
   only, so scaling to 1M rows is feasible.

On top of this they layer (i) an **attention-based many-class
decoder** that frames class prediction as soft nearest-neighbour
retrieval over the in-context training rows (non-parametric in
class count); (ii) **row-chunking inference** that precomputes
the inducing-vector summary once and then streams rows through
the column-aggregator in fixed-size chunks, decoupling peak
activation memory from dataset size; (iii) **multi-query
attention** in the ICL stage — test rows share a single KV head,
shrinking the per-estimator KV cache to ~7 GB at 1M rows.
Pretrained purely on synthetic data from an improved SCM prior;
no real-data continued pretraining in the base release.

**What it contains — results.** On TabArena-medium
(10k–100k rows), a single forward pass of TabPFN-3 beats every
other model — including tuned-and-ensembled baselines — by a
significant Elo margin and pareto-dominates the speed/accuracy
frontier. TabPFN-3-Plus (Thinking) beats AutoGluon 1.5 extreme
(4 hours of tuning) in <1/10 the runtime, with no LLMs, no real
data, no internet search. Up to **20× faster than TabPFN-2.5
inference**. SOTA on many-class classification, on the RelBenchV1
relational benchmark via a relational checkpoint, on TabSTAR
(tabular-text) via Plus, and 2nd on the fev-bench time-series
benchmark via a TabPFN-TS-3 checkpoint.

**For CreditPFN.** Important and slightly destabilising. Three
threads to think about:

* **Successor question.** Our current pipeline targets
  TabPFN-v3 *classifier-v3_default* and *regressor-v3_default*
  checkpoints (see `config/train.yaml:27-37` and
  `docs/CHECKPOINTS.md`). Those v3 weights ARE this paper's
  release — the v3 line in our checkpoint inventory and this
  paper's "TabPFN-3" are the same model. So we are already on
  the latest generation; no re-base required.

* **Continued-pretraining still applies, but the recipe shifts.**
  TabPFN-3 ships synthetic-prior only — no Real-TabPFN-style
  variant yet. The Real-TabPFN recipe (Garg 2025) was designed
  against v2's alternating-attention architecture; the three
  stages here mean the continued-pretraining target surface is
  different. In practice the loop should still work (gradients
  through the column-aggregator + ICL stages are the same idea),
  but the LoRA target modules in
  [config/train.yaml:99-107](../config/train.yaml) (currently
  `q_projection`, `k_projection`, `v_projection`,
  `out_projection`) may need adjusting for v3's named stages —
  especially the new column-aggregator and the many-class
  decoder. Worth re-checking which layers exist on v3 with
  `peft.utils.other.transpose` style introspection before
  trusting the existing LoRA wrap on v3 ckpts.

* **Memory & chunk-size implications.** The paper's row-chunking
  inference scheme is the formalisation of what our cached
  `chunk_*.npz` files already approximate at the dataset level —
  but our training step still loads `n_finetune_ctx_plus_query_samples`
  rows into GPU memory at once. v3's three-stage forward has a
  *very* different memory profile from v2.6 (the column-wise
  inducing-point attention plus the cls-token aggregation
  consume significant VRAM at training time). The
  100 000-row default in [config/train.yaml:122](../config/train.yaml)
  was tuned for v2.6 on a 96 GB H100 NVL; on v3 with LoRA on an
  80 GB H100 it OOMs (observed 2026-05-19, job 66726904). Drop
  the value (50 k or lower) for v3 base checkpoints until we
  re-tune.

* **Test-time compute for risk scoring.** The "Thinking" mode is
  a research-grade idea: trade inference time for accuracy on the
  forward pass alone, no extra training. For credit-risk
  scorecards the wall-clock budget per applicant is typically
  hundreds of milliseconds to seconds at most, so Thinking may
  be marginal for online scoring — but for batch portfolio
  revaluation (overnight job, millions of rows) it could move
  the AUC needle without retraining. Park this as a downstream
  evaluation idea once the continued-pretrained CreditPFN
  checkpoints exist.
