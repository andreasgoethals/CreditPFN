# Checkpoints

Local copies of TabPFN model weights used as the starting point for
continued pretraining on credit-risk data. **Do not edit or commit
new checkpoints to this folder without updating this file.**

All facts below are sourced from:

- The upstream `tabpfn` package README (mirrored at
  `repositories/TabPFN .txt`, lines 649–751 and 2606–2623).
- The Prior Labs HuggingFace model cards, mirrored verbatim by the
  scraper at `repositories/scraped/hf__tabpfn_2_5.txt` and
  `repositories/scraped/hf__tabpfn_2_6.txt`.
- The TabPFN-2.5 paper (Grinsztajn et al., 2025, arXiv:2511.08667),
  Appendix C.

## Why this matters: the synthetic-vs-real distinction

The Real-TabPFN recipe (Garg et al., 2025, arXiv:2507.03971) is
**continued pretraining of a TabPFN whose initial pretraining used
only synthetic priors**, on a curated corpus of real-world tabular
datasets. The base must be the *synthetic-only* checkpoint.

If you start from a checkpoint that Prior Labs has already real-data
finetuned, you stack their generic-real corpus underneath your
credit-risk corpus. You can no longer cleanly attribute downstream
gains to "credit-risk specialization" — they could be leaking from
Prior Labs' own real-data exposure (43 datasets, listed in TabPFN-2.5
paper Appendix C.1).

Methodologically clean replication therefore needs a synthetic-only
base. The table below labels exactly which of the local files are
synthetic-only and which are not.

## Inventory (verified against upstream)

| File | Size | Origin | Training data | OK as Real-TabPFN base? |
|---|---|---|---|---|
| `tabpfn-v2.5-classifier-v2.5_default.ckpt` | 43 MB | HF `Prior-Labs/tabpfn_2_5` | **Real-finetuned** (🌍 in upstream README). Default classifier since `tabpfn` v2.1.0 | No — confounds the ablation |
| `tabpfn-v2.5-classifier-v2.5_default-2.ckpt` | 43 MB | HF `Prior-Labs/tabpfn_2_5` | **Synthetic-only** ("best classification synthetic checkpoint") | **Yes** — clean v2.5 classifier base |
| `tabpfn-v2.5-classifier-v2.5_real.ckpt` | 43 MB | HF `Prior-Labs/tabpfn_2_5` | Real-finetuned, alternative variant. "Pretty good overall but bad on large features (>100–200)" | Use only as comparison baseline |
| `tabpfn-v2.5-regressor-v2.5_default.ckpt` | 41 MB | HF `Prior-Labs/tabpfn_2_5` | **Synthetic-only** ("trained on synthetic data only") | **Yes** — clean v2.5 regressor base |
| `tabpfn-v2.5-regressor-v2.5_real.ckpt` | 41 MB | HF `Prior-Labs/tabpfn_2_5` | Real-finetuned. Best among real-finetuned regressors on average | Comparison baseline |
| `tabpfn-v2.5-regressor-v2.5_real-variant.ckpt` | 41 MB | HF `Prior-Labs/tabpfn_2_5` | Real-finetuned, alternative variant | Comparison baseline |
| `tabpfn-v2.6-classifier-v2.6_default.ckpt` | 43 MB | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** — the v2.6 model card states "TabPFN-2.6 is trained purely on synthetic tabular tasks" (no real-finetuned v2.6 yet) | **Yes** — the cleanest v2.6 classifier base available |
| `tabpfn-v2.6-regressor-v2.6_default.ckpt` | 51 MB | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** (same model-card statement) | **Yes** — clean v2.6 regressor base |

### Important correction vs. earlier discussion

For v2.5 the naming convention is:
`_default` = real-finetuned, `_default-2` = synthetic-only.

For v2.6 the naming convention is *different*:
`_default` = synthetic-only (no real variant published yet).

So the two v2.6 files in this folder are *already* the
methodologically correct base for Real-TabPFN-style continued
pretraining — no need to chase down a `_default-2` for v2.6.

This is confirmed by the v2.6 HuggingFace model card:

> TabPFN-2.6 is trained purely on synthetic tabular tasks.
> *(repositories/scraped/hf__tabpfn_2_6.txt, "Training Data and Priors")*

vs. the v2.5 card:

> TabPFN-2.5: trained purely on synthetic tabular tasks
> Real-TabPFN-2.5: continued pre-training on real-world datasets
> *(repositories/scraped/hf__tabpfn_2_5.txt, "Training Data and Priors")*

## Architecture differences between v2.5 and v2.6

| | v2.5 | v2.6 |
|---|---|---|
| Layers | 18–24 (varies across checkpoints) | 24 (fixed) |
| Attention pattern | TabPFNv2-like alternating | TabPFNv2-like alternating |
| Real-finetuned variant published? | Yes (`_default`, `_real`, `_real-variant`) | No (only synthetic-only `_default`) |
| Model tech report | arXiv:2511.08667 (Grinsztajn et al.) | Same paper version | None published yet |

## Recommended use in this project

| Track | Base for continued pretraining | Comparison baseline |
|---|---|---|
| PD (classification) — v2.5 ablation | `tabpfn-v2.5-classifier-v2.5_default-2.ckpt` | `tabpfn-v2.5-classifier-v2.5_default.ckpt` (Real-TabPFN-2.5) |
| PD (classification) — v2.6 main | `tabpfn-v2.6-classifier-v2.6_default.ckpt` | (no Real-TabPFN-2.6 baseline available) |
| LGD (regression) — v2.5 ablation | `tabpfn-v2.5-regressor-v2.5_default.ckpt` | `tabpfn-v2.5-regressor-v2.5_real.ckpt` |
| LGD (regression) — v2.6 main | `tabpfn-v2.6-regressor-v2.6_default.ckpt` | (none) |

The actual checkpoint choice belongs in `config/train/*.yaml` (when
the training config is fleshed out), not in `config/data.yaml` —
data-stage code does not touch the checkpoints.

## Licence

All weights are released under Prior Labs' `tabpfn-2.5-license-v1.1`
or `tabpfn-2.6-license-v1.0`. These are research-only licences:
testing, evaluation, and internal benchmarking are explicitly
allowed; commercial use, client deliverables, or commercial
decision-making based on the model's outputs are not. Full text in
the licence files inside each HF repo. For commercial use, contact
`sales@priorlabs.ai`.
