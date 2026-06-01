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
  (arXiv:2511.08667), Appendix C.

The inventory lists every `.ckpt` in this folder, what training
data Prior Labs used to produce it, and a brief note on what role
each plays in **our** continued pretraining experiments.

Only **v2.6** and **v3** synthetic-only bases are used; both the PD
classifier and the LGD regressor sweep over these two.

## All our bases are synthetic-only

Every checkpoint we sweep was trained from scratch on millions of
*synthetic* tabular datasets sampled from a structural-causal-model
prior — no real-world data has touched the weights. **TabPFN-v3 and
all v2.6 variants** ship synthetic-only, and neither has a released
"real-finetuned" variant. That is exactly the clean starting point the
Real-TabPFN recipe (Garg et al. 2025) assumes: begin from the synthetic
prior, then continue-pretrain on real data — in our case the curated
credit-risk corpus — so any downstream gain is attributable purely to
that corpus.

## Inventory (verified against upstream)

| File | Size | Origin | Training data | Role in this project |
|---|---|---|---|---|
| `tabpfn-v3-classifier-v3_default.ckpt`         | 213 MB | HF `Prior-Labs/tabpfn_3` | **Synthetic-only.** The v3 HF card states *"TabPFN-3 is trained purely on synthetic tabular tasks."* New multi-stage transformer architecture (24 main layers); ≤1 M samples × ≤2 000 features (vs. 50 k for v2.6). | **Default sweep base.** Latest released checkpoint with the strongest published benchmarks (SOTA on TabArena, TALENT). |
| `tabpfn-v3-regressor-v3_default.ckpt`          | 233 MB | HF `Prior-Labs/tabpfn_3` | **Synthetic-only.** Same v3 card statement applies; no real-finetuned v3 regressor yet. | **Default sweep base** for LGD. |
| `tabpfn-v2.6-classifier-v2.6_default.ckpt`     | 43 MB  | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** — the v2.6 card states *"TabPFN-2.6 is trained purely on synthetic tabular tasks"*; no real-finetuned v2.6 variant has been released. | Sweep base: the cleanest v2.6 base available. |
| `tabpfn-v2.6-regressor-v2.6_default.ckpt`      | 51 MB  | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** (same card statement). No real-finetuned v2.6 regressor yet. | Sweep base: cleanest v2.6 regressor base. |

## How to read the naming conventions

For **v3** the naming is: only `_default` (synthetic-only). No
specialist or real-finetuned variants have been released yet.

For **v2.6** the naming is: `_default` = synthetic-only (no
real-finetuned variant published yet).

Both conventions are confirmed verbatim by the HuggingFace cards
mirrored at `repositories/Huggingface TabPFN.txt`.

## What we sweep over

The training config (`config/train.yaml::tunable`) treats the base
checkpoint as a tuneable knob and sweeps over the released
synthetic-only bases for v2.6 and v3:

| Track           | Sweep includes (default)                  | What each tells us                                          |
|-----------------|-------------------------------------------|-------------------------------------------------------------|
| PD (classifier) | `v3_default` · `v2.6_default`             | v3 synthetic-only · v2.6 synthetic-only                     |
| LGD (regressor) | `v3_default` · `v2.6_default`             | v3 synthetic-only · v2.6 synthetic-only                     |

The total grid per track is then `2 bases × 3 LRs × 2 LoRA × 1 qf ×
1 acc × 2 epoch-pass-modes = 24 trials` (the LR / qf / accumulate axes
were narrowed and an epoch-pass-mode A/B added after the first sweep).

Every base in this sweep follows the methodologically clean
ablation recipe of Real-TabPFN (Garg et al. 2025): start from
the synthetic-only checkpoint, continue-pretrain on real data —
in our case our curated credit-risk corpus. Any downstream gain on
credit-risk benchmarks is attributable purely to that corpus.

The eval pipeline (`scripts/eval_pipeline.py`) scores all of these
side-by-side against XGBoost / CatBoost / LogReg / LinReg plus the
*untuned* versions of each base, so the question of "which base
wins" gets answered empirically on the held-out test split.

## Architecture differences across versions

|                                       | v2.6                          | v3                                       |
|---------------------------------------|-------------------------------|------------------------------------------|
| Layers                                | 24 (fixed)                    | 24 main layers (multi-stage transformer) |
| Attention pattern                     | TabPFNv2-style alternating    | Multi-stage transformer-based            |
| Sample limit (intended)               | ≤ 50 000                      | ≤ 1 000 000                              |
| Feature limit (intended)              | ≤ 2 000                       | ≤ 2 000                                  |
| Real-finetuned variant published?     | No (only synthetic `_default`)| No (only synthetic `_default`)           |
| Model technical report                | Grinsztajn et al. 2026 (arXiv:2511.08667, same architecture family) | Not yet published (HF card only)         |
| Approximate checkpoint size           | ~43–51 MB                     | ~213–233 MB                              |
| License                               | `tabpfn-2.6-license-v1.0`     | `tabpfn-3-license-v1.0`                  |

## Licence

All weights are released under Prior Labs' research-only licences
(`tabpfn-2.6-license-v1.0`, `tabpfn-3-license-v1.0`). Testing,
evaluation, and internal benchmarking are explicitly allowed;
commercial use, client deliverables, or commercial decision-making
based on the model's outputs are not. Full text in the licence
files inside each HF repo.
