# Checkpoints

Local copies of TabPFN model weights, used as starting points for
continued pretraining on credit-risk data. **Do not edit or commit
new checkpoints without updating this file.**

All facts below are sourced from:

- The upstream `tabpfn` package README (mirrored at
  `repositories/TabPFN .txt`, lines 649–751 and 2606–2623).
- Prior Labs' HuggingFace model cards (verbatim mirror at
  `repositories/Huggingface TabPFN.txt`).
- Hollmann et al. 2025 (*Nature*) and Grinsztajn et al. 2026
  (TabPFN-2.5 paper, arXiv:2511.08667), Appendix C.

The inventory lists every `.ckpt` in this folder, what training
data Prior Labs used to produce it, and a brief note on what role
each plays in **our** continued pretraining experiments.

## Synthetic vs. real-finetuned — what the labels mean

Each Prior Labs checkpoint was produced in one of two ways:

* **Synthetic-only.** Trained from scratch on millions of *synthetic*
  tabular datasets sampled from a structural-causal-model prior. No
  real-world data has touched the weights. **TabPFN-v3 and all v2.6
  variants** ship synthetic-only; the v2.5 family uses the suffix
  `_default-2` for this and reserves `_default` for the real-finetuned
  variant (naming conventions disagree across major versions — see
  the section below).
* **Real-finetuned (Real-TabPFN-2.5).** Took the synthetic-only
  checkpoint and continued pretraining it on a Prior Labs-curated
  corpus of **43 real-world OpenML/Kaggle datasets** (TabPFN-2.5
  paper, Appendix C.1). Only the v2.5 family has a released
  real-finetuned variant (`_default`, `_real`); v2.6 and v3 do **not**.

Neither flavour is intrinsically "right" as a base for our project.
Both are valid starting points and the project deliberately sweeps
over both — see the "What we sweep over" section.

## Inventory (verified against upstream)

| File | Size | Origin | Training data | Role in this project |
|---|---|---|---|---|
| `tabpfn-v3-classifier-v3_default.ckpt`         | 213 MB | HF `Prior-Labs/tabpfn_3` | **Synthetic-only.** The v3 HF card states *"TabPFN-3 is trained purely on synthetic tabular tasks."* New multi-stage transformer architecture (24 main layers); ≤1 M samples × ≤2 000 features (vs. 50 k for v2.5/v2.6). | **Default sweep base.** Latest released checkpoint with the strongest published benchmarks (SOTA on TabArena, TALENT). |
| `tabpfn-v3-regressor-v3_default.ckpt`          | 233 MB | HF `Prior-Labs/tabpfn_3` | **Synthetic-only.** Same v3 card statement applies; no real-finetuned v3 regressor yet. | **Default sweep base** for LGD. |
| `tabpfn-v2.5-classifier-v2.5_default.ckpt`     | 43 MB  | HF `Prior-Labs/tabpfn_2_5` | **Real-finetuned** (🌍 in upstream README). Default classifier since `tabpfn` v2.1.0. | Sweep base: "what does a v2.5 stack of Prior Labs' real corpus + our credit corpus get us?" |
| `tabpfn-v2.5-classifier-v2.5_default-2.ckpt`   | 43 MB  | HF `Prior-Labs/tabpfn_2_5` | **Synthetic-only** ("best classification synthetic checkpoint"). | Sweep base: methodologically cleanest v2.5 ablation against Real-TabPFN-2.5. |
| `tabpfn-v2.5-classifier-v2.5_real.ckpt`        | 43 MB  | HF `Prior-Labs/tabpfn_2_5` | Real-finetuned, alternative variant. "Pretty good overall but weaker on >100–200-feature tasks." | Comparison baseline only — not in the default sweep. |
| `tabpfn-v2.5-regressor-v2.5_default.ckpt`      | 41 MB  | HF `Prior-Labs/tabpfn_2_5` | **Synthetic-only** ("trained on synthetic data only"). | Sweep base: clean v2.5 regressor ablation. |
| `tabpfn-v2.5-regressor-v2.5_real.ckpt`         | 41 MB  | HF `Prior-Labs/tabpfn_2_5` | Real-finetuned. Best among real-finetuned regressors on average. | Sweep base: "real-finetuned starting point" for LGD. |
| `tabpfn-v2.5-regressor-v2.5_real-variant.ckpt` | 41 MB  | HF `Prior-Labs/tabpfn_2_5` | Real-finetuned, alternative variant. | Comparison baseline only. |
| `tabpfn-v2.6-classifier-v2.6_default.ckpt`     | 43 MB  | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** — the v2.6 card states *"TabPFN-2.6 is trained purely on synthetic tabular tasks"*; no real-finetuned v2.6 variant has been released. | Sweep base: the cleanest v2.6 base available. |
| `tabpfn-v2.6-regressor-v2.6_default.ckpt`      | 51 MB  | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** (same card statement). No real-finetuned v2.6 regressor yet. | Sweep base: cleanest v2.6 regressor base. |

## How to read the naming conventions

For **v3** the naming is: only `_default` (synthetic-only). No
specialist or real-finetuned variants have been released yet.

For **v2.5** the naming is: `_default` = real-finetuned,
`_default-2` = synthetic-only. (The suffix-2 means "second-best on
a generic benchmark but synthetic-only", per the HF model card.)

For **v2.6** the naming is *different*: `_default` = synthetic-only
(no real-finetuned variant published yet).

All three conventions are confirmed verbatim by the HuggingFace
cards mirrored at `repositories/Huggingface TabPFN.txt`.

## What we sweep over

The training config (`config/train.yaml::tunable`) treats the base
checkpoint as a tuneable knob and sweeps over a deliberate mix of
versions and flavours:

| Track           | Sweep includes (default)                                                       | What each tells us                                                                                            |
|-----------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| PD (classifier) | `v3_default` · `v2.6_default` · `v2.5_default-2` · `v2.5_default`              | v3 synthetic-only · v2.6 synthetic-only · v2.5 synthetic-only ablation · v2.5 real-finetuned                  |
| LGD (regressor) | `v3_default` · `v2.6_default` · `v2.5_default` · `v2.5_real`                   | v3 synthetic-only · v2.6 synthetic-only · v2.5 synthetic-only · v2.5 real-finetuned                           |

The three flavours answer different questions, and we don't pick a
winner up-front:

- **Synthetic-only base + our continued pretraining.** Methodologically
  cleanest ablation: any downstream gain on credit-risk benchmarks
  is attributable purely to our credit-risk corpus. This is the
  exact recipe Real-TabPFN-2.5 followed (Garg et al. 2025): take the
  synthetic-only checkpoint, continue-pretrain on real data.
  Applies to **all v3 and v2.6 bases**, plus v2.5's `_default-2`.
- **Real-finetuned base + our continued pretraining.** Higher
  starting point on most benchmarks (Prior Labs already exposed the
  model to 43 generic real datasets). Downstream gains then reflect
  *both* their real-data exposure and our credit-risk specialisation,
  so the attribution is muddier — but if our goal is just maximum
  end-task accuracy on credit-risk problems, this might win
  empirically. Only available via v2.5 (`_default`, `_real`).

The eval pipeline (`scripts/eval_pipeline.py`) scores all of these
side-by-side against XGBoost / CatBoost / LogReg / LinReg plus the
*untuned* versions of each base, so the question of "which base
wins" gets answered empirically on the held-out test split.

## Architecture differences across versions

|                                       | v2.5                                            | v2.6                          | v3                                       |
|---------------------------------------|-------------------------------------------------|-------------------------------|------------------------------------------|
| Layers                                | 18–24 (varies across checkpoints)              | 24 (fixed)                    | 24 main layers (multi-stage transformer) |
| Attention pattern                     | TabPFNv2-style alternating                     | TabPFNv2-style alternating    | Multi-stage transformer-based            |
| Sample limit (intended)               | ≤ 50 000                                       | ≤ 50 000                      | ≤ 1 000 000                              |
| Feature limit (intended)              | ≤ 2 000                                        | ≤ 2 000                       | ≤ 2 000                                  |
| Real-finetuned variant published?     | Yes (`_default`, `_real`, `_real-variant`)     | No (only synthetic `_default`)| No (only synthetic `_default`)           |
| Model technical report                | arXiv:2511.08667 (Grinsztajn et al.)            | Same paper                    | Not yet published (HF card only)         |
| Approximate checkpoint size           | ~41–43 MB                                       | ~43–51 MB                     | ~213–233 MB                              |
| License                               | `tabpfn-2.5-license-v1.1`                       | `tabpfn-2.6-license-v1.0`     | `tabpfn-3-license-v1.0`                  |

## Licence

All weights are released under Prior Labs' research-only licences
(`tabpfn-2.5-license-v1.1`, `tabpfn-2.6-license-v1.0`,
`tabpfn-3-license-v1.0`). Testing, evaluation, and internal
benchmarking are explicitly allowed; commercial use, client
deliverables, or commercial decision-making based on the model's
outputs are not. Full text in the licence files inside each HF
repo.
