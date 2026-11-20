# CreditPFN

A descriptive study of **domain adaptation by continued pretraining (CPT)** of tabular foundation
models, with credit risk as the domain. PD is binary default classification; LGD is
loss-given-default regression. Four base models: TabPFN v2, v2.6 and v3, and TabICLv2. The study
asks whether CPT on real credit datasets improves predictions on *unseen* credit datasets, when
and why it does, what is forgotten on non-credit tables, and how the adaptation recipe changes
the outcome.

## Design

The study grid is [`config/design.yaml`](config/design.yaml): one reference recipe plus declared
deviations, materialized into `config/experiments/E<NN>_<slug>/` (and the Experiment 0 pilots in
`config/experiments/E00_pilots/`) by `src/utils/design.py`. Tests fail on any undeclared
difference or stale generated config.

| Research question | Experiments |
|---|---|
| RQ1 Does domain CPT pay off? Effect on unseen credit tables against seed noise, numerical precision, a matched non-credit corpus and target fine-tuning | E1 reference · E2 seeds · E3 training precision · E4 matched non-credit domain control · E5 CPT vs per-table fine-tuning |
| RQ2 When and why? Does in-context learning already supply the domain signal, and which sources transfer to which targets | E6 labelled-context curves · E7 single-source transfer matrix · per-table analysis |
| RQ3 What changes it? Update size, anchoring and trainable scope against forgetting; exposure, table size and episode size | E8 learning rate × L2-SP · E9 frozen / DiagScale / LoRA · E10 exposure (full pass, 10,000 updates) · E11 small vs large tables · E12 episode size |

800 standard training runs plus 100 single-source runs; E1 is the shared control of E2–E4 and
E8–E12. Every table is evaluated by a model that never saw it (four partitions of the 17 PD and
8 LGD tables). Models are evaluated in FP32 on complete five-fold test sets; forgetting is measured
on 19 binary and 12 regression TabArena tables; baselines are CatBoost, XGBoost and
logistic/linear regression. One comparison is pre-specified as primary (E1 against the original
model; PD AUC and Brier score, LGD R² and CRPS; Holm correction over 16 tests). PD is also scored
with the credit-scoring measures KS, H-measure and expected maximum profit. A TabArena-style
leaderboard (Elo anchored at the linear baseline, normalized score, improvability) ranks every
method on the same folds, and an appendix recomputes every result on the public tables only.

Reference CPT uses FP16 autocast with loss scaling. TabPFN v2 PD computes its transformer
normalizations in FP32, and TabICLv2 computes its column embeddings in FP32. Experiment 0
compares the gradients of the recorded training policy with FP32 on identical episodes.
TabPFN LGD trains with a 5-bucket window NLL (robust to the v2.6 regressor's comb-shaped density);
LoRA and DiagScale use L2-SP on their effective weight change.

## Compute and storage

Training and evaluation run on the VSC Mindwell B200 GPUs in a sealed container. The Git checkout
and compact logs live on VSC DATA; datasets, checkpoints, predictions and detailed results on the
GPFS project storage. The output folder is `output CreditPFN/` on both tiers.

Before E1, the Experiment 0 controller resolves the full grid, checks each reference benchmark
against the current weights and input data, and times a full target fine-tuning fold per model
and task. Its versioned receipts bind the campaign release to the validated source, runtime and
input snapshot. Each new full validation preserves earlier pilot outputs.
PD table 0004 retains its early LendingClub loans. Table 0006 uses terminal lender outcomes for
506,724 uniquely matched 2012–2015 challenge loans. Of these, 415,941 were fully paid and
90,783 were charged off. Its input key can be rebuilt with `python -m src.data.lendingclub_outcomes`
from the four original lender archives. CPU preflight verifies the processed cohort by checksum.
Both tables belong to one source for train/test separation, including the E7 transfer matrix.
[2007–2011](https://resources.lendingclub.com/LoanStats3a.csv.zip) · [2012–2013](https://resources.lendingclub.com/LoanStats3b.csv.zip) · [2014](https://resources.lendingclub.com/LoanStats3c.csv.zip) · [2015](https://resources.lendingclub.com/LoanStats3d.csv.zip)

E5 saves each completed baseline and candidate fit plus the latest completed training epoch
on PROJECT. Retrying the same fold restores optimizer, scheduler, loss scale, early stopping,
weights and random state. The quick pilot deliberately interrupts and resumes a candidate.

## Local tests and notebooks

From this repository in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 2
```

The local `.venv` matches the cluster container's exact pins (`config/runtime/`,
`tests/test_environment_parity.py`). Notebooks live in
`notebooks/{rq1,rq2,rq3,experiment0,general,appendix}`, contain no functions or classes, call
`src/` only and end with a complete text summary. The appendix notebooks are generated from the
results notebooks (`python -m src.utils.appendix_notebooks`). Figures are
drawn for Springer LNCS (122 mm wide, vector PDF, no figure-level title) in
`output CreditPFN/figures/<group>/`; `CAPTIONS.md` holds their captions and titles, and
`output CreditPFN/All_Results.md` the notebook summaries.

## Repository layout

Reusable code lives in `src/`; entry points are `scripts/{data,train,eval}_pipeline.py` and
`scripts/probe_row_cap.py`, submitted through `scripts/slurm/`. `tfm-library/` (literature) and
`vsc-library/` (VSC infrastructure) are pinned read-only submodules. Datasets, checkpoints and
credentials are never committed; the project documentation (`docs/`) is kept outside the
repository.

## Based on the repository template

Based on [andreasgoethals/0.-Template](https://github.com/andreasgoethals/0.-Template), which
describes the starting layout and utilities. The shared figure tree (`output CreditPFN/figures/`)
and the output directory name are deliberate project-specific deviations.
