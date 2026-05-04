# Repositories — context corpus

This folder is a **read-only** reference corpus. Every `.txt` file
here is a flat dump of an upstream repository or webpage, kept
locally so the pipeline code can be grepped against canonical
implementations without round-tripping to the internet. **Never edit
these files.** If a snapshot is stale, replace the whole file with a
fresh dump using the same filename, so existing greps in the
codebase keep working.

Listed alphabetically below. Where a repo corresponds to a paper in
[`papers/`](../papers/), the paper is linked. Where the file is a
fresh dump of a public GitHub repo, the upstream URL is linked.

## Overview table

| File | Lines | GitHub | Paper | What it gives us |
|------|-------|--------|-------|------------------|
| `Huggingface TabPFN.txt` | 494 | [tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5), [tabpfn_2_6](https://huggingface.co/Prior-Labs/tabpfn_2_6) | [Hollmann 2025](../papers/2025_Hollmann_et_al._Accurate_predictions_on_small_data_with_a_tabular_foundation_model.pdf), [Grinsztajn 2026](../papers/2026_Grinsztajn_et_al._TabPFN_2.5_Advancing_the_State_of_the_Art_in_Tabular_Foundation_Models.pdf) | Primary citation source for checkpoint provenance (synthetic vs. real-finetuned, layer counts, intended limits, licence). |
| `NanoTabPFN.txt` | 895 | [automl/nanoTabPFN](https://github.com/automl/nanoTabPFN) | [Pfefferle 2025](../papers/2025_Pfefferle_et_al._nanoTabPFN_A_Lightweight_and_Educational_Reimplementation_of_TabPFN.pdf) | Cleanest end-to-end reference of a PFN training loop. Structural template for `src/train/`. |
| `PFNS.txt` | 20 743 | [automl/PFNs](https://github.com/automl/PFNs) | [Müller 2021](../papers/2021_Muller_et_al._Transformers_Can_Do_Bayesian_Inference.pdf) | Implementations of every encoder step that runs *inside* every TabPFN forward pass (NaN handling, normalisation, …). Tells us what `sanitize.py` should *not* duplicate. |
| `PFNs4BO.txt` | 6 488 | [automl/PFNs4BO](https://github.com/automl/PFNs4BO) | [Müller 2023](../papers/2023_Muller_et_al._PFNs4BO_In_Context_Learning_for_Bayesian_Optimization.pdf) | PFN-as-Bayesian-optimisation surrogate. Tangential to credit-risk; useful only if we wrap a PFN around our own HP search. |
| `TabDPT.txt` | 2 874 | [layer6ai-labs/TabDPT-inference](https://github.com/layer6ai-labs/TabDPT-inference) | [Ma 2026](../papers/2026_Ma_et_al._TabDPT_Scaling_Tabular_Foundation_Models_on_Real_Data.pdf) | Inference code for the real-data-only competitor to TabPFN. Comparison baseline. |
| `TabPFN .txt` | 63 974 | [PriorLabs/tabPFN](https://github.com/PriorLabs/tabPFN) | [Hollmann 2023](../papers/2023_Hollmann_et_al._TabPFN_A_Transformer_That_Solves_Small_Tabular_Classification_Problems_in_a_Second.pdf), [Hollmann 2025](../papers/2025_Hollmann_et_al._Accurate_predictions_on_small_data_with_a_tabular_foundation_model.pdf), [Grinsztajn 2026](../papers/2026_Grinsztajn_et_al._TabPFN_2.5_Advancing_the_State_of_the_Art_in_Tabular_Foundation_Models.pdf) | Canonical sklearn-style API, all checkpoint metadata, the multi-table finetuning machinery (`get_preprocessed_dataset_chunks`, `DatasetCollectionWithPreprocessing`, `FinetunedTabPFN*`). Primary code reference for `src/train/`. |
| `TabPFN Client.txt` | 8 916 | [PriorLabs/tabpfn-client](https://github.com/PriorLabs/tabpfn-client) | — | Hosted-API HTTP client. Not used in our self-hosted pretraining; only for benchmarking against the API. |
| `TabPFN Docs.txt` | 7 797 | [PriorLabs/docs](https://github.com/PriorLabs/docs) | — | The docs.priorlabs.ai source. Documents *intent* of every config knob; faster to grep than the implementation in `TabPFN .txt`. |
| `TabPFN Drift-Resilient.txt` | 17 844 | [automl/Drift-Resilient_TabPFN](https://github.com/automl/Drift-Resilient_TabPFN) | [Helli 2024](../papers/2024_Helli_et_al._Drift_Resilient_TabPFN_In_Context_Learning_Temporal_Distribution_Shifts_on_Tabular_Data_1.pdf) | Drift-aware training augmentation. Highly relevant for credit-risk's macro-cycle drift; consider folding into `src/train/`. |
| `TabPFN Extensions.txt` | 17 415 | [PriorLabs/tabpfn-extensions](https://github.com/PriorLabs/tabpfn-extensions) | — | `AutoTabPFN` post-hoc ensembling, RF-PFN, embeddings, HPO. Source of evaluation baselines. |
| `TabPFN V2 Finetuning.txt` | 3 697 | [PriorLabs/TabPFN/examples](https://github.com/PriorLabs/TabPFN/tree/main/examples) | [Rubachev 2025](../papers/2025_Rubachev_et_al._On_Finetuning_Tabular_Foundation_Models_1.pdf) | The `finetune_classifier.py` and `finetune_regressor.py` reference scripts. Canonical "load checkpoint → backward pass → save checkpoint" sequence. |
| `TabPFN Wide.txt` | 2 388 | [automl/TabPFN-Wide](https://github.com/automl/TabPFN-Wide) | [Kolberg 2026](../papers/2026_Kolberg_et_al._TabPFN_Wide_Continued_Pre_Training_for_Extreme_Feature_Counts.pdf) | The continued-pretraining recipe for extreme-feature-count regimes. Source of our `FeatureAgglomeration` design. |
| `TransformersCanDoBayesianInference.txt` | 6 869 | [automl/PFNs](https://github.com/automl/PFNs) (early) | [Müller 2021](../papers/2021_Muller_et_al._Transformers_Can_Do_Bayesian_Inference.pdf) | Code for the original PFN paper. Mostly historical; useful for explaining what a PFN is. |
| `VSC Documentation.txt` | 39 358 | [hpcleuven/VscDocumentation](https://github.com/hpcleuven/VscDocumentation) | — | Full Sphinx source of the VSC supercomputer documentation. SLURM job scripting, A100 partitions, storage tiers, account / VO management. The reference when writing the SLURM scripts under `scripts/`. |

## Layout

```
repositories/
├── REPOSITORIES.md                          (this file)
├── Huggingface TabPFN.txt                   (   494 lines)
├── NanoTabPFN.txt                           (   895 lines)
├── PFNS.txt                                 (20 743 lines)
├── PFNs4BO.txt                              ( 6 488 lines)
├── TabDPT.txt                               ( 2 874 lines)
├── TabPFN .txt                              (63 974 lines)
├── TabPFN Client.txt                        ( 8 916 lines)
├── TabPFN Docs.txt                          ( 7 797 lines)
├── TabPFN Drift-Resilient.txt               (17 844 lines)
├── TabPFN Extensions.txt                    (17 415 lines)
├── TabPFN V2 Finetuning.txt                 ( 3 697 lines)
├── TabPFN Wide.txt                          ( 2 388 lines)
├── TransformersCanDoBayesianInference.txt   ( 6 869 lines)
└── VSC Documentation.txt                    (39 358 lines)
```

---

## `Huggingface TabPFN.txt`

**Upstream:** HuggingFace model cards for
[`Prior-Labs/tabpfn_2_5`](https://huggingface.co/Prior-Labs/tabpfn_2_5)
and
[`Prior-Labs/tabpfn_2_6`](https://huggingface.co/Prior-Labs/tabpfn_2_6).

**Related papers:**
[2025 — Hollmann et al. — Accurate predictions on small data with a tabular foundation model](../papers/2025_Hollmann_et_al._Accurate_predictions_on_small_data_with_a_tabular_foundation_model.pdf),
[2026 — Grinsztajn et al. — TabPFN-2.5](../papers/2026_Grinsztajn_et_al._TabPFN_2.5_Advancing_the_State_of_the_Art_in_Tabular_Foundation_Models.pdf).

**What it is.** Concatenation of the public HuggingFace model-card
READMEs for `Prior-Labs/tabpfn_2_5` and `Prior-Labs/tabpfn_2_6` (the
source URLs are written out as comments inside the file). Each card
is included twice in the dump (with and without `# Source:` headers);
the content is identical and any grep against the file will still
hit.

**Why it matters here.** The model cards are the *primary published
source* for which checkpoint is real-finetuned vs. synthetic-only,
the layer counts (v2.5 = 18–24, v2.6 = 24), the licence terms, and
the citation. Every fact in
[`checkpoints/CHECKPOINTS.md`](../checkpoints/CHECKPOINTS.md) is
cross-checked against this file.

**Contents in detail.**

- **YAML front matter** — licence (`tabpfn-2.5-license-v1.1` /
  `tabpfn-2.6-license-v1.0`), pipeline tag, gated-access fields,
  thematic tags including `finance`.
- **Model overview** — TabPFN-2.x = transformer-based foundation
  model, in-context learning, single forward pass.
- **Architecture** — v2.5: "TabPFNv2-like alternating attention with
  18-24 layers"; v2.6: "TabPFNv2-like alternating attention with 24
  layers".
- **Training data and priors** —
  - v2.5: "TabPFN-2.5: trained purely on synthetic tabular tasks /
    Real-TabPFN-2.5: continued pre-training on real-world datasets
    (for details please see Appendix C.1 of the model tech report)."
  - v2.6: "TabPFN-2.6 is trained purely on synthetic tabular tasks."
    (No real-finetuned variant. Decisive evidence that v2.6 default
    *is* the synthetic-only base.)
- **The complete v2.5 checkpoint catalogue** with one-line
  descriptions per checkpoint, identical to what you'll find inside
  `TabPFN .txt:736-751`. The 🌍 emoji marks the real-finetuned
  variants. v2.6 has only the `_default` checkpoints listed.
- **Intended use / limitations** — ≤ 50 000 samples and ≤ 2000
  features; not for unstructured data.
- **Licensing** — research-only with an enterprise option for
  commercial use.

**When to grep this file:** when you need a primary citation for any
checkpoint provenance claim (v2.5/v2.6, real vs synthetic, intended
sample/feature limits). For pipeline implementation, prefer
`TabPFN .txt` (more detailed) or `TabPFN Docs.txt` (more recent
prose).

---

## `NanoTabPFN.txt`

**Upstream:** [github.com/automl/nanoTabPFN](https://github.com/automl/nanoTabPFN).

**Related paper:**
[2025 — Pfefferle et al. — nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN](../papers/2025_Pfefferle_et_al._nanoTabPFN_A_Lightweight_and_Educational_Reimplementation_of_TabPFN.pdf).

**What it is.** A minimal reference implementation of TabPFN — the
"how-it-works-in-under-1000-lines" educational version, trimmed of
the production package's ergonomics (sklearn API surface, ensembles,
distributed training, autocast handling, multiple checkpoints) so
that the *core training and inference loop* is visible at a glance.

**Why it's the highest-signal file in this folder for our work:**
Real-TabPFN's source is not public, so the closest thing we have to
"a continued-pretraining loop in a few hundred lines that we can
read end-to-end" lives here.

**Contents in detail:**

- **Lines ~150–260 — Data loading and preprocessing.** Defines
  `get_feature_preprocessor(X)` which fits a `ColumnTransformer` that
  separates numerical from categorical columns by checking, for each
  column, whether `pd.to_numeric(errors='coerce').notna().sum()`
  equals the non-NaN count. Numerical columns get coerced to numeric
  arrays; categoricals get an `OrdinalEncoder(handle_unknown=
  'use_encoded_value', unknown_value=np.nan)`. Constant columns (≤ 1
  unique non-NaN value) are dropped. This is the *minimum* feature
  preprocessing TabPFN inputs need, and it matches the rules we are
  putting into `src/data/sanitize.py`.
- **`get_openml_datasets(...)`** — illustrates the OpenML download
  path for evaluation (TabArena task IDs hardcoded), with stratified
  subsampling via `train_test_split(stratify=y)`.
- **Lines ~520–700 — Model definition.** The full TabPFN-style
  alternating-attention transformer in pure PyTorch: alternating
  attention "between samples" and "between features" within the same
  layer, plus a target embedding head.
  `forward(src, train_test_split_index)` shows exactly how the
  context-vs-query split is consumed: the model sees one tensor of
  shape `(B, N, F)` and an integer index that says "rows
  [0:split_idx) are context with labels, rows [split_idx:] are query
  with masked labels".
- **Lines 758–832 — Training loop.** The pretraining loop in 70
  lines: `schedulefree.AdamWScheduleFree`, learning rate `4e-3`,
  cross-entropy loss reshaped to `(B*N_query, n_classes)`,
  gradient-norm clip at `1.0`, periodic eval. This is the structural
  reference for `src/train/train.py` once we get there.
- **Lines 835–880 — `PriorDumpDataLoader`.** Loads pre-baked
  synthetic prior datasets from an HDF5 file. Fields:
  `X (B, N_max, F_max)`, `y (B, N_max)`, `num_features`,
  `num_datapoints`, `single_eval_pos` (= train/test split index),
  `max_num_classes`. Padding-aware: each batch slices to the
  per-batch max feature count and max sequence length. **This format
  is exactly what our `data/cached/` directory needs to match** for
  continued pretraining, except we'll use `.npz` per dataset instead
  of one monolithic HDF5 (rationale: Real-TabPFN-style continued
  pretraining loops over real datasets, not over a synthetic prior).

**When to grep this file:** any time you need the "what does the
training loop actually look like" answer — model `forward`
semantics, loss shape, gradient handling, optimizer setup, batch
layout.

---

## `PFNS.txt`

**Upstream:** [github.com/automl/PFNs](https://github.com/automl/PFNs).

**Related papers:**
[2021 — Müller et al. — Transformers Can Do Bayesian Inference](../papers/2021_Muller_et_al._Transformers_Can_Do_Bayesian_Inference.pdf)
(the foundational PFN framework that this code implements).

**What it is.** The full Prior-Labs `PFNs` repo — the underlying
"Prior-fitted Network" framework that TabPFN is built on top of.
Contains the canonical implementations of the *internal* encoder
steps that run inside every TabPFN forward pass.

**Why it matters here:** It tells us what `sanitize.py` should *not*
duplicate. The model already handles a lot of preprocessing
internally; if our pipeline does it a second time we either waste
work or (worse) double-transform features in ways that drift the
distribution off the model's training prior.

**Contents in detail:**

- **Lines ~5700–5800 — Encoder-step composition.** Shows how the
  standard TabPFN-2.x model assembles its input encoder as a
  sequence:
  - `NanHandlingEncoderStep` — handles missing values (emits an
    explicit indicator and replaces NaN with a learned default).
  - `LinearInputEncoderStep` — linear projection of features into
    embedding space.
  - `VariableNumFeaturesEncoderStep` — pads/shuffles the feature
    dimension so a model trained on F-max features can ingest any
    F ≤ F-max.
  - `ConstantNormalizationInputEncoderStep` and
    `InputNormalizationEncoderStep` — per-column normalisation
    fitted on the *context* rows only and applied to query rows.
- **Lines ~6048–6310 — Each encoder step's full implementation.**
  - `LinearInputEncoderStep` (line 6048).
  - `ConstantNormalizationInputEncoderStep` (6108).
  - `NanHandlingEncoderStep` (6143): replaces `NaN` with
    `nan_indicator`, `+inf` with `inf_indicator`, `-inf` with
    `neg_inf_indicator`, then concatenates a binary "was-this-NaN"
    flag along the feature dimension. Confirms that our sanitize
    step (h) — replace ±inf with NaN — is correct.
  - `VariableNumFeaturesEncoderStep` (6211).
  - `InputNormalizationEncoderStep` (6290) — per-column mean/std
    computed on the context split only.
- **Lines ~16335–16415 — Standalone preprocessing transforms** that
  run *before* the model, as ensemble members at inference time:
  `PowerTransformer(method='yeo-johnson')` /
  `PowerTransformer(method='box-cox')` /
  `QuantileTransformer(output_distribution='normal')` /
  `RobustScaler(unit_variance=True)`. **These are inference-time
  ensemble preprocessing, not training-time preprocessing.**

**When to grep this file:** to confirm whether something is the
model's internal job or our pipeline's job; to look up what the
encoder steps actually do to NaNs / infs / constants; to see the
canonical ensemble of preprocessing options at inference.

---

## `PFNs4BO.txt`

**Upstream:** Same repo as PFNs (BO branch /
[github.com/automl/PFNs4BO](https://github.com/automl/PFNs4BO)).

**Related paper:**
[2023 — Müller et al. — PFNs4BO: In-Context Learning for Bayesian Optimization](../papers/2023_Muller_et_al._PFNs4BO_In_Context_Learning_for_Bayesian_Optimization.pdf).

**What it is.** PFNs adapted to Bayesian optimisation. Implements an
acquisition function on top of a PFN posterior — useful for
hyperparameter search but unrelated to credit-risk pretraining.

**When to grep this file:** only if you later want to use a PFN as
a surrogate model for tuning your own continued-pretraining
hyperparameters. No relevant preprocessing or training-loop
machinery beyond what's already in `PFNS.txt`.

---

## `TabDPT.txt`

**Upstream:** [github.com/layer6ai-labs/TabDPT-inference](https://github.com/layer6ai-labs/TabDPT-inference)
(inference code; full training code is in the sibling repo
[`layer6ai-labs/TabDPT-training`](https://github.com/layer6ai-labs/TabDPT-training)).

**Related paper:**
[2026 — Ma et al. — TabDPT: Scaling Tabular Foundation Models on Real Data](../papers/2026_Ma_et_al._TabDPT_Scaling_Tabular_Foundation_Models_on_Real_Data.pdf).

**What it is.** TabDPT is the *real-data-only* counterpart to
TabPFN: a transformer trained on real tables sampled from OpenML
via retrieval-augmented self-supervision, with no synthetic
prior. The dump in this folder is the *inference* repo — load
weights from HuggingFace and predict on a new dataset via a
sklearn-style API.

**Why it matters.** TabDPT and TabPFN bracket the
synthetic-vs-real spectrum from opposite ends. Real-TabPFN (and
hence CreditPFN) sits in the middle. Having TabDPT locally lets
us include it as an *inference baseline* in the eval harness:
compare CreditPFN's scores against TabPFN-2.6, TabDPT, TabICL,
and the published Real-TabPFN-2.5 weights, all in one place.

**Contents in detail.**

* `src/tabdpt/classifier.py`, `regressor.py`, `estimator.py`,
  `model.py`, `utils.py` — the inference path.
* `tabdpt_datasets/openml.py` — OpenML dataset loaders that they
  used at training time and ship for reproducibility.
* `tabdpt_datasets/data_splits/{cls,reg}_datasets.csv` — the
  exact CSV manifest of which OpenML datasets they trained on.
  Useful for *contamination checking* — any dataset on this list
  must NOT appear in our held-out evaluation set if we want a
  fair comparison.
* `tests/cls_example.py`, `reg_example.py` — minimum working
  example.

**When to grep this file:** when implementing the TabDPT
baseline in `src/eval/`, or when checking that our credit-risk
held-out splits don't overlap with TabDPT's training corpus.

---

## `TabPFN .txt`

**Upstream:** [github.com/PriorLabs/tabPFN](https://github.com/PriorLabs/tabPFN)
(the main `tabpfn` Python package).

**Related papers:**
[2023 — Hollmann et al. — TabPFN](../papers/2023_Hollmann_et_al._TabPFN_A_Transformer_That_Solves_Small_Tabular_Classification_Problems_in_a_Second.pdf),
[2025 — Hollmann et al. — Accurate predictions on small data](../papers/2025_Hollmann_et_al._Accurate_predictions_on_small_data_with_a_tabular_foundation_model.pdf),
[2026 — Grinsztajn et al. — TabPFN-2.5](../papers/2026_Grinsztajn_et_al._TabPFN_2.5_Advancing_the_State_of_the_Art_in_Tabular_Foundation_Models.pdf).

**What it is.** The user-facing sklearn-style API
(`TabPFNClassifier`, `TabPFNRegressor`), the checkpoint-loading
infrastructure, the inference-time ensembling, and the documented
list of every released `.ckpt` and what it's good for. Plus the
finetuning wrappers (`FinetunedTabPFNClassifier` /
`FinetunedTabPFNRegressor`) and the multi-table machinery
(`get_preprocessed_dataset_chunks`,
`DatasetCollectionWithPreprocessing`, `fit_from_preprocessed`) that
our `src/train/` will compose on.

**Why it matters:** this is the single source of truth for
checkpoint provenance (which files are synthetic-only, which are
real-finetuned), the public configuration knobs
(`PreprocessorConfig`, `ModelInterfaceConfig`), and the supported
input shapes / dtypes / NaN handling at the API boundary.

**Contents in detail:**

- **Lines 649–650** — README block listing the *default* v2.5
  classifier and regressor checkpoint URLs.
- **Lines 736–751 — The complete TabPFN-2.5 checkpoint catalogue
  with one-line descriptions:** which is real-finetuned (🌍 emoji),
  which is synthetic, which specialises for "large features"
  (`large-features-L` up to 500 features, `large-features-XL` up
  to 1000), which for "large samples" (>30K), which for low-skew
  regression targets, etc. **This is what
  `checkpoints/CHECKPOINTS.md` is built from — when in doubt, this
  section of `TabPFN .txt` is ground truth.**
- **Lines 2606–2628 — Programmatic checkpoint name registry** used
  inside the package to validate `model_path` arguments.
- **Lines 6952–6985 — `tabpfn-v2-` (v2.0) checkpoint registry** for
  the older v2.0 model family.
- **Lines 17301–17400 — `DatasetCollectionWithPreprocessing`**, the
  `torch.utils.data.Dataset` subclass that lazily preprocesses each
  dataset on `__getitem__`, returning a `ClassifierBatch` or
  `RegressorBatch`.
- **Lines 17702–17761 — `shuffle_and_chunk_data`**, TabPFN's own
  chunking utility (the package equivalent of our `dataset.py`'s
  chunking). Stratified for multiclass, non-stratified for
  regression.
- **Lines 17764–17881 — `get_preprocessed_dataset_chunks`**, the
  helper that accepts a *list* of datasets and produces a
  `DatasetCollectionWithPreprocessing` ready for a multi-table
  training loop.
- **Lines 18062, 18999, 19616 — `FinetunedTabPFNBase`,
  `FinetunedTabPFNClassifier`, `FinetunedTabPFNRegressor`** — the
  official sklearn-compatible finetuning wrappers.
- **`PreprocessorConfig` definitions** — exposes
  `name='none' | 'safepower' | 'quantile_uni_coarse' | 'quantile_uni'
  | 'robust_scaler' | …`. These are inference-time ensemble names.
- **`save_tabpfn_model` utility** (around line 710) — the reverse
  direction: how to dump a model object back to a `.ckpt`.

**When to grep this file:** checkpoint names, public API surface,
inference-time preprocessing names, validation / error messages
the package raises, the multi-table finetuning machinery.

---

## `TabPFN Client.txt`

**Upstream:** [github.com/PriorLabs/tabpfn-client](https://github.com/PriorLabs/tabpfn-client).

**Related paper:** none.

**What it is.** The HTTP client for Prior Labs' hosted inference
API.

**When to grep this file:** only if you're benchmarking against the
hosted API. Not relevant for our self-hosted pretraining workflow on
VSC.

---

## `TabPFN Docs.txt`

**Upstream:** [github.com/PriorLabs/docs](https://github.com/PriorLabs/docs)
(the source for [docs.priorlabs.ai](https://docs.priorlabs.ai)).

**Related papers:** indirectly all of the TabPFN papers — the docs
sit on top of the package described above.

**What it is.** Flat dump of the TabPFN documentation repository —
the Markdown sources that the docs.priorlabs.ai site is built from.
~7 800 lines covering every documented capability, hyperparameter,
integration, and fine-tuning recipe.

**Why it's the highest-signal recent addition:** it contains the
*official* documentation of TabPFN's preprocessing knobs, the
fine-tuning wrappers shipped inside the package, and the design
intent behind every configurable parameter. For Stage 1–4 design,
this is the most authoritative non-code reference.

**Contents in detail (with line numbers worth bookmarking):**

- **`overview.mdx`** (around lines 50–200) — what TabPFN is and what
  its capabilities are.
- **`models.mdx`** (around lines 1100–1180) — version comparison
  table.
- **`improving-performance/preprocessing.mdx`** (lines 6331–6404) —
  **most important section for our Stage 3 (`sanitize.py`)
  design.** Documents:
  - `PREPROCESS_TRANSFORMS`: ensemble preprocessing names
    (`"quantile_uni"`, `"squashing_scaler_default"`, `"safepower"`,
    `"quantile_uni_coarse"`, `"kdi"`, `"robust"`, `"none"`).
  - `categorical_name`: encoding options.
  - `max_features_per_estimator`: default `500`.
  - `REGRESSION_Y_PREPROCESS_TRANSFORMS`: target transforms for
    regression (`"none"`, `"safepower"`, `"quantile_norm"`,
    `"quantile_uni"`, `"1_plus_log"`).
  - `OUTLIER_REMOVAL_STD`: **default `"auto"` which resolves to
    `12.0` for classification and `None` for regression** — see the
    "Outlier handling" section below.
  - `POLYNOMIAL_FEATURES`, `FINGERPRINT_FEATURE`,
    `SUBSAMPLE_SAMPLES`. None of these belong in
    `config/data.yaml` — they are inference-time levers.
- **`capabilities/fine-tuning.mdx`** (lines 4224–4450) —
  documentation of the official `FinetunedTabPFNClassifier` and
  `FinetunedTabPFNRegressor` wrappers. Important caveat at line
  4246: "The fine-tuning process decouples the preprocessing
  pipeline to generate transformed tensors that mirror the
  preprocessing configurations used during inference, ensuring the
  model optimizes on the exact same data variations it encounters
  when making predictions."
- **`improving-performance/feature-engineering.mdx`,
  `feature-selection.mdx`, `model-parameters.mdx`** — softmax
  temperature, balanced-probability handling, imbalance handling.
- **`extensions/*.mdx`** — `hpo`, `many-class`, `post-hoc-ensembles`,
  `rf-pfn`. Reference for the evaluation protocol later.
- **`api-reference/*.mdx`** — hosted-API only.
- **`integrations/*.mdx`** — Databricks, Azure Foundry, SageMaker,
  MLflow, n8n.
- **`use-cases/*.mdx`** — including `finance.mdx`.

**When to grep this file:** for the *intent* and *contract* of any
documented TabPFN configuration option. Faster than grepping
`TabPFN .txt` (which is the implementation) when you just need
"what does this parameter do".

---

## `TabPFN Drift-Resilient.txt`

**Upstream:** [github.com/automl/Drift-Resilient_TabPFN](https://github.com/automl/Drift-Resilient_TabPFN).

**Related paper:**
[2024 — Helli et al. — Drift-Resilient TabPFN](../papers/2024_Helli_et_al._Drift_Resilient_TabPFN_In_Context_Learning_Temporal_Distribution_Shifts_on_Tabular_Data_1.pdf).

**What it is.** A research repo that fine-tunes / specialises TabPFN
for distribution-shift robustness — explicitly modelling "training
distribution drifts at inference time" rather than assuming i.i.d.

**Why it's interesting for credit risk specifically:** credit-risk
data is *defined* by macroeconomic regime drift. PD and LGD
distributions shift dramatically between expansion and recession.
A model that's been continued-pretrained on a credit-risk corpus
might benefit from drift-aware training-time augmentations
borrowed from this repo. Not a Stage 1–4 concern, but worth
grepping during the training-loop design.

**Contents in detail:** drift-aware loss formulations,
distribution-shift simulation as a training-time augmentation,
extra evaluation protocols (eval on artificially shifted test
sets).

**When to grep this file:** when designing training-time
augmentations or evaluation under regime shift.

---

## `TabPFN Extensions.txt`

**Upstream:** [github.com/PriorLabs/tabpfn-extensions](https://github.com/PriorLabs/tabpfn-extensions).

**Related paper:** none directly; references the `AutoTabPFN`
ensemble idea documented across multiple TabPFN papers.

**What it is.** `tabpfn-extensions` — official add-on package
(post-hoc ensembling, RF-PFN hybrids, embeddings, hyperparameter
search via OpenAutoML, a "many-class classifier" wrapper for >10
target classes, a fingerprint-feature tool, etc.).

**Why it's relevant:** at evaluation time, the standard reporting
package compares plain TabPFN vs. `AutoTabPFN` (post-hoc ensemble).
We will probably compare *our* CreditPFN against both, so the
ensemble mechanics matter for fair comparison.

**Contents in detail:** post-hoc ensemble definition (uses
AutoGluon under the hood), RF-PFN tree-based hybrid (`rf_pfn`
extension — good baseline since classical PD/LGD modelling
traditionally uses gradient-boosted trees and random forests).

**When to grep this file:** when designing the evaluation protocol
or building strong baselines.

---

## `TabPFN V2 Finetuning.txt`

**Upstream:** [github.com/PriorLabs/TabPFN/tree/main/examples](https://github.com/PriorLabs/TabPFN/tree/main/examples)
(the `finetune_classifier.py` and `finetune_regressor.py`
example scripts).

**Related paper:**
[2025 — Rubachev et al. — On Finetuning Tabular Foundation Models](../papers/2025_Rubachev_et_al._On_Finetuning_Tabular_Foundation_Models_1.pdf).

**What it is.** Sebastian Pineda's TabPFN-V2-Finetuning recipe — the
closest public analog of Real-TabPFN's continued-pretraining, but
with one key difference: this is dataset-specific finetuning
("finetune TabPFN-v2 to *one* downstream dataset") rather than
continued pretraining on a whole corpus.

**Why it's a critical reference:** it shows the exact mechanics of
loading a v2 checkpoint, making a forward/backward pass through it,
and saving the result — which is the unit-of-work for continued
pretraining too. Just call the same loop in a sweep over many
datasets.

**Contents in detail:**

- **Lines ~85, 367, 418, 718** — `save_path_to_fine_tuned_model =
  "./fine_tuned_model.ckpt"` and surrounding
  `torch.save({"state_dict": …, "optimizer_state": …,
  "scheduler_state": …}, path)` — the canonical checkpoint format
  for finetuning runs.
- **Lines ~407–441** — `from tabpfn.config import
  ModelInterfaceConfig, PreprocessorConfig` and the
  `no_preprocessing_inference_config` pattern.
- **Lines 468, 562** — `OrdinalEncoder(handle_unknown=
  'use_encoded_value', unknown_value=-1)`. Confirms the categorical
  encoding contract.
- **Lines 538–700** — `preprocess_dummy_data(...)` end-to-end:
  load OpenML task → split → fit OrdinalEncoder on train →
  transform query → cast to torch tensors → device placement.
- **Loss / backward / optimizer setup** (search for
  `loss.backward`).

**When to grep this file:** for the canonical "load checkpoint →
attach optimizer → forward pass → backward pass → save checkpoint"
sequence for *real* TabPFN-2 weights, not the toy model.

---

## `TabPFN Wide.txt`

**Upstream:** [github.com/automl/TabPFN-Wide](https://github.com/automl/TabPFN-Wide).

**Related paper:**
[2026 — Kolberg et al. — TabPFN-Wide](../papers/2026_Kolberg_et_al._TabPFN_Wide_Continued_Pre_Training_for_Extreme_Feature_Counts.pdf).

**What it is.** TabPFN-Wide — modifications for high-dimensional
inputs, e.g. multi-omics or wide credit-bureau datasets with
hundreds to thousands of columns.

**Why it matters:** several credit datasets in our raw corpus
already have >100 columns. The 128-column ceiling we apply via
`FeatureAgglomeration` matches the TabPFN-2.x training prior, but
we should know how the Wide variant addresses the same problem —
and the `FeatureAgglomeration(metric='euclidean', linkage='ward')`
idiom we adopted lives in this codebase first.

**Contents in detail:**

- **Lines ~170–890 — Multiple `load_state_dict` patterns** for
  loading both the standard TabPFN checkpoint and Wide-modified
  checkpoints.
- **Lines 1746–1801 — `load_checkpoint(self)` method** showing how
  the Wide trainer loads a *training-state* checkpoint
  (state_dict + optimizer state + scheduler state). Template for
  resumable training in our SLURM jobs.
- **Lines 2244, 2294, 2312 — `FeatureAgglomeration` usage.** Two
  variants: `FeatureAgglomeration(n_clusters=n_features)` (default
  Euclidean+Ward) at line 2294, and `FeatureAgglomeration(
  n_clusters=n_features, metric='precomputed', linkage='complete')`
  at line 2312. **Confirms the idiom we adopt for sanitize step
  (i).**

**When to grep this file:** dimensionality-reduction strategies,
checkpoint resumption, anything wide/high-feature.

---

## `TransformersCanDoBayesianInference.txt`

**Upstream:** [github.com/automl/PFNs](https://github.com/automl/PFNs)
(early version, repo since superseded by the modern PFNs framework).

**Related paper:**
[2021 — Müller et al. — Transformers Can Do Bayesian Inference](../papers/2021_Muller_et_al._Transformers_Can_Do_Bayesian_Inference.pdf).

**What it is.** Code accompanying Müller et al. 2021 — the original
PFN paper. Mostly historical context for understanding what a
"Prior-fitted Network" *is*: a transformer trained to perform
posterior inference for a particular Bayesian prior, by sampling
synthetic datasets from that prior and training the model to map
context → query predictions.

**When to grep this file:** when you need to write a paragraph
explaining PFNs in your thesis / a defence / a report. Not relevant
for pipeline implementation.

---

## `VSC Documentation.txt`

**Upstream:** [github.com/hpcleuven/VscDocumentation](https://github.com/hpcleuven/VscDocumentation)
(the source of the [VSC documentation site](https://docs.vscentrum.be)).

**Related paper:** none — this is supercomputer infrastructure
documentation, not a research artefact.

**What it is.** Flat dump of the entire Sphinx-generated VSC
(Vlaams Supercomputer Centrum / Flemish Supercomputer Centre)
user documentation. ~39 000 lines of `.rst` source covering
account management, SSH/MFA setup, Genius / wICE / Tier-1 cluster
hardware, SLURM job scripting, storage tiers, scientific software
modules, and the various ways to acknowledge the VSC in
publications.

**Why it matters here.** Our continued pretraining will run on
VSC A100 nodes via SLURM. When we write the SLURM job scripts
under `scripts/` — partitions, GPU allocation, time limits,
memory, scratch storage — this dump is the canonical reference.

**Sections most relevant to CreditPFN:**

* **`source/leuven/tier_2_hardware/`** — KU Leuven Tier-2 cluster
  documentation (Genius and wICE). Includes the GPU partitions
  with NVIDIA A100s and the SLURM partition / QOS names.
* **`source/jobs/`** — SLURM job submission, partitions, time
  limits, GPU requests (e.g. `--gres=gpu:1`,
  `--partition=gpu_a100`), array jobs (useful for the
  3000-dataset parallel preprocessing case).
* **`source/software/`** — module-system documentation for
  loading specific Python / CUDA / cuDNN versions, plus how to
  build custom virtual environments on the cluster filesystem.
* **`source/data_storage/`** — `$VSC_DATA`, `$VSC_SCRATCH` and
  the project storage tiers. Important for deciding where the
  3000-dataset corpus and the cached `.npz` files live (they
  shouldn't all be in `$VSC_HOME`).
* **`source/accounts/`** — initial SSH key setup, MFA, VO
  membership, requesting more quota.

**When to grep this file:** when writing or updating a SLURM
script (`scripts/*.slurm`), debugging a job-submission error,
choosing a partition or GPU type, or sizing storage allocations.
Less relevant during the data-pipeline implementation since that
work is done on a laptop.

---

## `TabPFN V2 Finetuning.txt` ↔ `NanoTabPFN.txt` ↔ `TabPFN Wide.txt` — how they relate

| | NanoTabPFN | Wide | V2 Finetuning |
|---|---|---|---|
| Model definition | toy reimpl | full + wide modifications | uses real package |
| Loads real `.ckpt`? | no (trains from scratch) | yes | yes |
| Has training loop? | yes (synthetic prior) | yes (sweep over real datasets) | yes (single-dataset finetune) |
| Closest to *our* use case | training-loop structure | dataset-sweep structure | checkpoint mechanics |

We will end up combining all three: V2-Finetuning's checkpoint
loading + Wide's resumable training-state pattern + NanoTabPFN's
clear training-loop scaffolding, applied to a multi-dataset corpus
in the Real-TabPFN spirit.

---

### Outlier handling: what TabPFN actually does (verified)

This is important enough to factor out into its own subsection,
because it directly determines how `sanitize.py` should treat
extreme values. The implementation lives in
`TabPFN .txt:15795-15828` (function `remove_outliers`) and
`TabPFN .txt:6273-6314` (the public knob `OUTLIER_REMOVAL_STD`).

```python
# TabPFN .txt:6273-6274
_REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD: float | None = None
_CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD: float = 12.0
```

```python
# TabPFN .txt:15795 (paraphrased)
def remove_outliers(X, n_sigma=4, normalize_positions=-1, ...):
    # 1. Compute per-column mean/std using ONLY the context split.
    # 2. Mark cells outside [mean ± n_sigma·std] as NaN.
    # 3. Re-compute mean/std from the now-cleaned data.
    # 4. Re-derive the [lower, upper] bounds from those robust stats.
    # 5. Apply a SOFT log-squash (not hard clip):
    X = max(-log(1+|X|) + lower, X)
    X = min( log(1+|X|) + upper, X)
```

**Three takeaways for our pipeline:**

1. **TabPFN's z-score normalization is *not* a substitute for
   outlier handling.** Z-scoring is sensitive to the very outliers
   it's trying to normalise — a single `1e9` value pins the mean
   and std to `~1e9` and crushes everything else to ~0. That's why
   the package itself ships an outlier removal step *before* /
   *alongside* the normalization step.
2. **The default threshold is `12σ` for classification, and outlier
   removal is *disabled* by default for regression.** A `[0.5%,
   99.5%]` quantile cut would be ~ `±2.6σ` for Gaussians, which is
   far more aggressive than what TabPFN does at inference and would
   create a train-vs-inference distribution mismatch.
3. **`OUTLIER_REMOVAL_STD` is an *inference-time* parameter** — when
   we pretrain by feeding cached tensors directly to the underlying
   torch model, this step is not automatically applied for us. Our
   `sanitize.py` therefore only normalises `±inf → NaN`; we delegate
   true outlier handling to the package's own machinery at training
   time.

### Fine-tuning wrappers: official package machinery

Independent of the data pipeline, `TabPFN Docs.txt:4224-4450` and
`TabPFN .txt:2035-2188` document `FinetunedTabPFNClassifier` /
`FinetunedTabPFNRegressor`. These are the supported entry points
for gradient-based adaptation of TabPFN. Their existence narrows
our `src/train/` design choices (later turn): we either compose
multiple `FinetunedTabPFN*.fit()` calls in a sweep over our 3 000
datasets, *or* we use the package's own internal multi-table
machinery (`get_preprocessed_dataset_chunks` +
`DatasetCollectionWithPreprocessing` +
`fit_from_preprocessed`) — the second path is what our cached
`.npz` triples (`X`, `y`, `categorical_idx`) are shaped for.

---

## Refreshing this folder

- For the upstream code repos (`TabPFN .txt`, `PFNS.txt`, etc.),
  re-grab a fresh dump from GitHub (e.g. via `code2txt` or a manual
  concat of `find . -name "*.py" -exec cat {}`) and **overwrite the
  existing file with the same filename** so existing greps in the
  codebase keep resolving.
- For the docs and HuggingFace cards, refresh
  `TabPFN Docs.txt` and `Huggingface TabPFN.txt` manually from
  their upstream sources.
