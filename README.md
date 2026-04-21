# CreditPFN

Continued pretraining of TabPFNv2.6 on a curated corpus of real-world
credit-risk datasets. The goal is to specialise the tabular foundation
model's in-context-learning prior toward the structures, feature
distributions, and label noise characteristic of credit-risk data,
and to evaluate whether a credit-specialised foundation model
outperforms the generalist TabPFN on downstream credit-risk tasks.

## Background

**TabPFN** is a transformer-based tabular foundation model that
performs in-context learning over entire tabular datasets in a single
forward pass. Version 2.6 ships two separate checkpoints:

- a **classifier** used here for **Probability of Default (PD)**
  prediction, and
- a **regressor** used here for **Loss Given Default (LGD)**
  estimation.

These two checkpoints have different weights and must be adapted
independently during continued pretraining.

**Continued pretraining** — as introduced for tabular foundation
models in *Real-TabPFN* (Garg et al., 2025,
[arXiv:2507.03971](https://arxiv.org/abs/2507.03971)) — extends the
synthetic-prior pretraining of TabPFN with additional training on a
curated corpus of real tabular datasets from a target domain. This
project applies the same methodology, but to a different domain:
credit risk.

**Credit risk modelling** has two primary quantitative use cases that
map directly onto the TabPFN checkpoints above:

1. **Probability of Default (PD)** — binary/multiclass classification
   of whether an obligor will default within a given horizon.
2. **Loss Given Default (LGD)** — regression of the fraction of
   exposure lost conditional on default.

By continuing to pretrain each TabPFN checkpoint on a corpus of real
credit datasets, we aim to shift the model's prior toward realistic
credit-risk data-generating processes while retaining its
in-context-learning capability on unseen downstream credit tasks.

## Compute

Training is run on the VSC supercomputer using A100 GPUs with SLURM
job scheduling. Job scripts and SLURM templates live under `scripts/`.

## Intended repository layout

```
CreditPFN/
├── README.md
├── .gitignore
├── requirements.txt
├── src/                  # Python package: data, training, eval, models
├── configs/              # Hydra/OmegaConf configs for runs and sweeps
├── scripts/              # SLURM job scripts and CLI entry points
├── notebooks/            # Exploratory analysis and result inspection
├── data/                 # Raw and processed credit-risk datasets (gitignored)
└── checkpoints/          # TabPFN base weights and adapted checkpoints (gitignored)
```

## References

- Hollmann et al., *TabPFN: Accurate predictions on small data with a
  transformer*, and the TabPFNv2 release.
- Garg et al., 2025. *Real-TabPFN: Improving Tabular Foundation Models
  via Continued Pre-training With Real-World Data.*
  [arXiv:2507.03971](https://arxiv.org/abs/2507.03971)
