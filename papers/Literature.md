# Literature on Tabular Foundation Models

A chronological tour of the 26 PDFs in this folder. The arc is:
PFNs (in-context Bayesian inference for arbitrary priors) → TabPFN
(PFNs with a tabular prior) → TabPFNv2 (production-grade, the model
we build on) → a Cambrian explosion of variants (continued
pretraining, drift, fairness, causal inference, time-series,
many-classes, scalability) → TabPFN-2.5 (the unified successor that
itself ships continued-pretraining as the default).

For every paper:

* **Where it fits** in the picture above.
* **What it actually contains** — methods, datasets, headline result.
* **What it means for CreditPFN.**

The most directly relevant papers for our project are
[Garg 2025 (Real-TabPFN)](#2025--garg-et-al--real-tabpfn),
[Hollmann 2025 (TabPFNv2)](#2025--hollmann-et-al--accurate-predictions-on-small-data),
[Grinsztajn 2026 (TabPFN-2.5)](#2026--grinsztajn-et-al--tabpfn-25),
[Rubachev 2025 (Finetuning)](#2025--rubachev-et-al--on-finetuning-tabular-foundation-models),
[Helli 2024 (Drift-Resilient TabPFN)](#2024--helli-et-al--drift-resilient-tabpfn), and
[Kolberg 2026 (TabPFN-Wide)](#2026--kolberg-et-al--tabpfn-wide).

---

## 2021 — Müller et al. — Transformers Can Do Bayesian Inference

**Where it fits:** the bedrock paper. Introduces *Prior-Fitted
Networks* (PFNs): train a transformer on synthetic tasks sampled
from a prior and the network learns to do approximate Bayesian
posterior inference *in-context*, in a single forward pass.

**What it contains.** Defines the meta-learning recipe: at training
time, sample a dataset from a Bayesian prior, give the transformer
the labelled context plus an unlabelled query point, and train it to
predict the query label. At inference time, the network never trains
on real data — it just consumes the new dataset as context. The
authors prove that with enough capacity and prior coverage, the
trained network's posterior approaches the true Bayesian posterior.

**For CreditPFN.** This is *why* TabPFN works at all. Continued
pretraining doesn't change the PFN inference paradigm; it shifts the
implicit prior from "abstract synthetic SCMs" toward "real
credit-risk data-generating processes". Read this paper before
arguing about identifiability or statistical guarantees with a
reviewer.

## 2023 — Hollmann et al. — TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second

**Where it fits:** the original TabPFN — applies the PFN recipe to
tabular classification with a structured-causal-model prior.

**What it contains.** A transformer that takes (X_context, y_context,
X_query) → ŷ_query in one forward pass. Pretrained on millions of
synthetic tabular datasets sampled from a prior over Bayesian neural
networks and small SCMs. Beats well-tuned XGBoost / Random Forest /
AutoGluon on the OpenML-CC18 benchmark for small datasets (≤ 1k
rows, ≤ 100 features, ≤ 10 classes), in a *second* of inference.

**For CreditPFN.** v1 of the model family. Architecture is the
direct ancestor of v2 / v2.5 / v2.6. The 1k-row / 100-feature
limits of v1 are why we use v2.6 instead.

## 2023 — Müller et al. — PFNs4BO: In-Context Learning for Bayesian Optimization

**Where it fits:** sibling of TabPFN — applies PFNs to a different
domain (Bayesian optimisation surrogate models) instead of tabular
classification.

**What it contains.** Trains a PFN to predict objective values given
a context of (configuration, value) pairs. Used as a drop-in
replacement for Gaussian-Process surrogates in BO.

**For CreditPFN.** Tangential. Useful only if we later use a PFN as
the surrogate inside our own hyperparameter search.

## 2024 — Breugel and Schaar — Why Tabular Foundation Models Should Be a Research Priority

**Where it fits:** position paper arguing the field needs tabular
analogues of GPT/CLIP. Pre-dates TabPFN-2.5 / Real-TabPFN.

**What it contains.** Surveys the gap between LLM/CV foundation
models (which dominate their fields) and tabular ML (still ruled by
GBDTs). Identifies five obstacles: heterogeneity, no-pretraining
norm, no big benchmark, no public real-data corpus, no commercial
push. Argues that solving these is high-leverage research.

**For CreditPFN.** Useful framing for the introduction of a thesis
or paper. The "no public real-data corpus" point specifically
motivates buying / curating the 3000-dataset corpus we will work on.

## 2024 — Helli et al. — Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data

**Where it fits:** TabPFN variant trained for distribution shift —
realistic temporal evolution rather than i.i.d. test data.

**What it contains.** Modifies TabPFN's synthetic-data generator to
inject temporal drift into the context-vs-query split (e.g. shifting
covariates, changing label functions). The resulting model
generalises better when the test distribution differs from the
training distribution by a continuous shift.

**For CreditPFN.** Highly relevant. Credit-risk data drifts hard
across macroeconomic regimes (2008, 2020). Borrowing their
drift-aware training augmentation as part of our continued
pretraining loop is a concrete avenue to explore — a CreditPFN
trained with drift augmentation should generalise better to
out-of-cycle defaults than a plain Real-TabPFN replication.

## 2024 — Hoo et al. — The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models Based on Simple Features

**Where it fits:** application paper showing TabPFN beats classical
forecasters when you frame forecasting as a tabular regression on
lag-features.

**What it contains.** Take a univariate time series, build features
(lags, calendar, rolling stats), feed to TabPFNRegressor. Beats
ARIMA and several deep TS baselines on M-style benchmarks.

**For CreditPFN.** Mostly out of scope — credit-risk modelling is
typically a cross-sectional default/LGD task per loan. But validates
that TabPFN-style in-context learning generalises beyond the
"classic" tabular setup.

## 2024 — Rundel et al. — Interpretable Machine Learning for TabPFN

**Where it fits:** interpretability tooling for TabPFN.

**What it contains.** Adapts SHAP, partial-dependence and feature
interaction analysis to TabPFN's in-context inference path. Argues
TabPFN is no harder to interpret than a GBDT once these tools are
adapted.

**For CreditPFN.** Important downstream. Credit-risk modelling has
regulatory interpretability requirements (Basel, EBA guidelines).
For the thesis evaluation chapter, we'll need to demonstrate that
CreditPFN's predictions can be explained at the same level a PD
model is currently explained at production banks.

## 2025 — Garg et al. — Real-TabPFN

> Improving Tabular Foundation Models via Continued Pre-training With Real-World Data
> arXiv:2507.03971

**Where it fits:** ★★★ **the recipe we are following.** ★★★ Continued
pretraining of TabPFNv2 on a curated set of 71 real-world tables
from OpenML + Kaggle.

**What it contains in detail.**

* *Method:* take the synthetic-only TabPFNv2 checkpoint, continue
  pretraining on a hand-curated corpus of 71 datasets (≥ 10 000 rows
  each, mixture of OpenML and Kaggle). Minimal preprocessing:
  `OrdinalEncoder` for categoricals; if > 10 classes, keep top-9 +
  "other".
* *Data contamination protocol* (their §3) — five-tier filter:
  size > 10k (every eval dataset is < 10k, so size alone separates
  pretrain from eval); cross-reference IDs / names / shapes;
  cross-reference column names; row hashes; column hashes; manual
  metadata inspection. **This is exactly what our `dedup.py`
  implements.**
* *Results:* +0.022 normalised ROC AUC on the 29-dataset OpenML
  AutoML Benchmark vs. default TabPFNv2 (Wilcoxon p = 0.0045). Gains
  grow with context size — 2k → 20k context yields a monotonic
  improvement curve.
* *Ablations:* OpenML alone +0.019, Kaggle alone +0.015, union +0.022
  (heterogeneous sources help). CommonCrawl as a corpus *hurts*
  performance (datasets too small, ~100 rows each). GitTables helps
  but less than OpenML+Kaggle.

**For CreditPFN.** This is our blueprint. Our contributions vs.
Real-TabPFN:
1. Different domain — credit risk instead of generic.
2. Different scale — 3000 datasets vs. their 71.
3. Two parallel tracks (PD / LGD) instead of one.
4. The dedup protocol is replicated faithfully (Stage 1 of our
   pipeline implements all five of their checks plus three extras).

The paper's source code is **not** public; this Markdown summary
plus the methods section we just lifted is what we have to work
from.

## 2025 — Hollmann et al. — Accurate predictions on small data with a tabular foundation model

**Where it fits:** the TabPFNv2 paper (Nature). Production-grade
release that 100×s the scaling limits of v1.

**What it contains.** A re-architected v2 with alternating
sample-attention / feature-attention layers, scales to ~10 000 rows
× 500 features. Synthetic prior expanded to include richer
distributions, NaN handling baked into the encoder
(`NanHandlingEncoderStep`), categorical handling via an internal
`OrdinalEncoder`, an inference-time ensemble of preprocessing
configurations (`PowerTransformer`, `QuantileTransformer`, etc.).
Beats AutoGluon, CatBoost, XGBoost on TabArena.

**For CreditPFN.** This is the architecture we continue-pretrain.
Most of our `sanitize.py` design decisions (don't winsorise,
preserve NaNs, don't standardise the regression target) flow
directly from how v2 is engineered.

## 2025 — Liu and Ye — TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems

**Where it fits:** scaling TabPFN to large datasets via
context-subsampling tricks rather than re-pretraining.

**What it contains.** Strategies for handling > 10 000-row datasets
with TabPFNv2 at inference time: stratified context selection,
bootstrap aggregation, query-level subsampling. Closes the gap to
GBDTs on medium-large data without retraining.

**For CreditPFN.** Operational. Useful at evaluation time when
scoring CreditPFN on 100k-row downstream credit datasets.

## 2025 — Müller et al. — Position: The Future of Bayesian Prediction Is Prior-Fitted

**Where it fits:** position paper — a manifesto for PFNs as a unifying
framework for "approximate Bayesian inference at scale".

**What it contains.** Argues that PFNs subsume GP regression, BNNs,
and tabular models; the limiting factor is no longer inference
algorithms but priors and dataset corpora.

**For CreditPFN.** Useful for thesis context. Frames why "design a
better prior for credit-risk" is a fundamentally Bayesian-inference
contribution rather than a hyperparameter tuning exercise.

## 2025 — Pfefferle et al. — nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN

**Where it fits:** educational reference implementation — TabPFN in
~900 lines of clear Python.

**What it contains.** A working PFN training loop, a synthetic-data
HDF5 prior dump, a minimal model with alternating attention,
inference utilities. Designed to be readable end-to-end.

**For CreditPFN.** ★★ Excellent for understanding. The `train(model,
prior, ...)` function at ~ line 758 of the source dump
(`repositories/NanoTabPFN.txt`) is the structural template for our
training loop, and the `PriorDumpDataLoader` shows the canonical
batch-format we must produce in our cache (X (B, N, F) + y (B, N) +
single_eval_pos).

## 2025 — Qu et al. — TabICL: A Tabular Foundation Model for In-Context Learning on Large Data

**Where it fits:** TabPFN competitor. Different architecture, scales
to 500k-row tables natively.

**What it contains.** Hierarchical attention over rows-then-features,
allowing the context to span much larger N than v2's 10k limit.
Trained on a mix of synthetic and real data.

**For CreditPFN.** Comparison baseline only. Not our pretraining
target.

## 2025 — Robertson et al. — Do-PFN: In-Context Learning for Causal Effect Estimation

**Where it fits:** PFN extension to causal effect estimation rather
than predictive modelling.

**What it contains.** Trains a PFN with synthetic data sampled from
SCMs to predict do-interventions (Pearl's do-calculus). Outperforms
classical IPW / DR estimators on synthetic and semi-synthetic
benchmarks.

**For CreditPFN.** Out of scope but interesting for follow-up work
("what's the causal effect of an APR cut on default risk?").

## 2025 — Robertson et al. — FairPFN: A Tabular Foundation Model for Causal Fairness

**Where it fits:** another causal PFN, this time aimed at counterfactual
fairness audits.

**What it contains.** Trains a PFN where the synthetic prior includes
explicit protected-attribute structure, then audits a model by
comparing factual and counterfactual predictions in-context.

**For CreditPFN.** Relevant for the regulatory chapter — credit-risk
models are subject to fair-lending rules (ECOA in the US). FairPFN's
auditing primitive could be used to certify that a CreditPFN does not
discriminate by protected attribute beyond what is statistically
unavoidable.

## 2025 — Rubachev et al. — On Finetuning Tabular Foundation Models

**Where it fits:** ★★ **directly relevant.** ★★ Studies what happens
when you fine-tune TabPFN (and other tabular FMs) on a downstream
dataset.

**What it contains.** Empirical study of fine-tuning TabPFN with full
gradient updates vs. parameter-efficient methods (LoRA, prefix
tuning) on dozens of downstream datasets. Headline findings:
fine-tuning *helps* on datasets large enough to overcome the
overfitting risk (typically > 1000 rows); LoRA-style methods recover
most of the gain at a fraction of the parameter cost; learning rates
in the 1e-5 to 1e-4 range are stable.

**For CreditPFN.** Our continued-pretraining is technically a
multi-table generalisation of the fine-tuning setup studied here.
The hyperparameter ranges (epochs ~30, LR ~1e-5) reported in this
paper match what we'll use in `src/train/`.

## 2025 — Tanna et al. — TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models

**Where it fits:** software / benchmark paper. Unified API for
running and finetuning TabPFN, TabICL, TabDPT, etc.

**What it contains.** Common interface to load any tabular FM, apply
common preprocessing, run inference, fine-tune. Useful for
side-by-side comparisons.

**For CreditPFN.** Could provide a clean wrapper for our evaluation
harness when we compare CreditPFN against TabICL / TabDPT baselines.

## 2025 — Ye et al. — A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities

**Where it fits:** empirical analysis of TabPFNv2 weaknesses.

**What it contains.** Identifies specific dataset characteristics
(many classes, very wide tables, very long-tailed targets) where
v2's defaults under-perform; proposes patches for each. Several of
those patches were rolled into the official package as the
non-default v2.5 checkpoint variants (`_low-skew`, `_quantiles`,
`_large-features-XL`, etc., per
`repositories/Huggingface TabPFN.txt`).

**For CreditPFN.** Useful as a checklist of "things we should
benchmark" — credit-risk datasets often have heavy-tailed loss
distributions for LGD (relevant to `_low-skew`).

## 2025 — Zhang et al. — Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models

**Where it fits:** alternative pretraining recipe — a richer
synthetic prior rather than continued pretraining on real data.

**What it contains.** Proposes a "mixed" prior that interpolates
between several published priors (TabPFN's SCM prior, ForestPFN's
decision-tree prior, etc.). Trained model outperforms each
single-prior baseline.

**For CreditPFN.** A complementary direction we are *not* taking
(we're enriching with real data instead). But the prior-mixing recipe
could be combined with continued pretraining in a follow-up.

## 2025 — Zhang et al. — TabPFN: One Model to Rule Them All

**Where it fits:** survey-style position paper covering TabPFN's
empirical wins across many domains.

**What it contains.** Aggregated benchmark numbers across health,
finance, ecology, etc. Argues TabPFN-class models replace much of
the "classic ML toolbox" for tabular problems below ~10k rows.

**For CreditPFN.** Useful citation for the introduction of a paper.

## 2026 — Grinsztajn et al. — TabPFN-2.5

> Advancing the State of the Art in Tabular Foundation Models
> arXiv:2511.08667

**Where it fits:** ★★★ **the architecture we instantiate.** ★★★
Successor to v2: deeper (18–24 layers), bigger context limit (50 000
samples × 2000 features), and crucially **ships the
real-data-finetuned variant as the default**.

**What it contains.**
* *Architecture:* transformer with TabPFNv2-like alternating attention
  with 18–24 layers (varies across the family of checkpoints).
* *Training data:* synthetic-only base + a Real-TabPFN-style
  real-data continued pretraining variant. The Real-TabPFN-2.5
  checkpoint uses 43 curated datasets listed in their Appendix C.1.
* *Checkpoints* (`Huggingface TabPFN.txt:91-106`): `_default` is
  real-finetuned; `_default-2` is synthetic-only; multiple specialist
  variants (`_large-features-L`, `_large-features-XL`,
  `_large-samples`, `_low-skew`, `_quantiles`, …).
* *Evaluation:* new SOTA on a proprietary benchmark, on TabArena,
  and on a causal benchmark (RealCause).

**For CreditPFN.** Our base architecture. Important detail: v2.5's
`_default` is real-finetuned (Prior Labs' generic 43-dataset corpus);
the methodologically clean base for *our* continued pretraining is
`_default-2` (synthetic-only). For v2.6 the convention flipped and
`_default` is again synthetic-only — see
[`checkpoints/CHECKPOINTS.md`](../checkpoints/CHECKPOINTS.md).

## 2026 — Hoo et al. — From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting

**Where it fits:** more thorough version of the 2024 forecasting
paper by the same authors — proper architectural extension instead
of just feature-engineering on top of TabPFN.

**What it contains.** Native time-axis attention in TabPFN, multi-horizon
forecasting, TS-specific preprocessing (de-trending, de-seasonalising
in-context). Beats GPT-style TS models on M5/M4.

**For CreditPFN.** Out of scope. Worth knowing about if we ever
extend CreditPFN to default-curve forecasting (one curve per loan
across months).

## 2026 — Kolberg et al. — TabPFN-Wide: Continued Pre-Training for Extreme Feature Counts

**Where it fits:** ★★ **directly relevant.** ★★ Continued pretraining
of TabPFN to handle datasets with more features than the v2 prior was
trained on.

**What it contains.** Modifies TabPFN's training recipe to include
synthetic datasets with hundreds–thousands of features, then
fine-tunes on real wide-feature datasets (genomics, multi-omics).
Uses `FeatureAgglomeration(metric='euclidean', linkage='ward')`
inside their preprocessing pipeline (their Appendix B is the
direct source for the `FeatureAgglomeration` design in our
`sanitize.py`). Releases `_large-features-L` (≤ 500 features) and
`_large-features-XL` (≤ 1000 features) checkpoints.

**For CreditPFN.** Very relevant. Two of our raw datasets exceed
2000 features (`0014.algorithmwatch` has 2987, `0011.loan_default`
has 770). Two options: cluster down to 128 with our agglomeration
(current default) or use the `_large-features-XL` checkpoint
directly. We'll likely benchmark both in `src/train/`.

## 2026 — Ma et al. — Foundation Models for Causal Inference via Prior-Data Fitted Networks

**Where it fits:** unified causal-PFN framework — generalises
Do-PFN and FairPFN.

**What it contains.** Theoretical and empirical groundwork for
causal in-context learning at scale.

**For CreditPFN.** Out of scope; bookmarked for follow-up work on
causal credit-risk modelling.

## 2026 — Ma et al. — TabDPT: Scaling Tabular Foundation Models on Real Data

**Where it fits:** competitor to TabPFN — a transformer pretrained on
*real* OpenML data rather than synthetic priors.

**What it contains.** Retrieval-augmented training: at every step,
sample a real OpenML table, mask part of it, and train the model to
fill the masked part. Discriminative architecture rather than
generative.

**For CreditPFN.** Comparison baseline. Notable for being an
existence proof that real-data-only pretraining works (which is the
opposite extreme of TabPFN's pure-synthetic recipe — Real-TabPFN
sits in the middle).

## 2026 — Qu et al. — TabICLv2: A better, faster, scalable, and open tabular foundation model

**Where it fits:** v2 of TabICL; closes more of the gap to TabPFN-2.5
and ships open weights.

**What it contains.** Improved hierarchical attention, bigger
training corpus, native large-context support.

**For CreditPFN.** Comparison baseline.
