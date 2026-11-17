# CreditPFN

A descriptive study of **domain adaptation by continued pretraining (CPT)** of tabular foundation
models, with credit risk as the domain. PD is binary default classification; LGD is
loss-given-default regression. Four base models: TabPFN v2, v2.6 and v3, and TabICLv2. The study
asks whether CPT on real credit datasets improves predictions on *unseen* credit datasets, how the
adaptation recipe changes that, what is forgotten on non-credit tables, and when and why it works.

## Status (16-11-2026): second campaign, Experiment 0 running

The first study (896 runs) is complete and archived locally in `archive/study-2026-11.zip`
(gitignored; `archive/study-2026-09.zip` holds the September run). Its main lessons: the reference
learning rate 3e-7 was the best tested (PD +0.0009 AUC on held-out credit tables, mostly from
TabPFN v2), LGD got slightly worse for every model, and curves were flat by 2,500 updates.

Project documentation (`docs/`: research brief, VSC runbook, literature, logs) is kept locally and
is not part of the repository. The second campaign is defined in [`config/design.yaml`](config/design.yaml): one reference recipe
plus declared deviations, materialized into `config/experiments/E<NN>_<slug>/` and
`config/experiments/E00_pilots/` by `src/utils/design.py` (tests fail on any undeclared difference or stale generated config).

## Three questions, twelve experiments

| RQ | Experiments |
|---|---|
| RQ1 Does domain CPT pay off? Effect on unseen credit tables against seed noise, numerical precision, a matched non-credit corpus and target fine-tuning | E1 reference · E2 seeds · E3 training precision · E4 matched non-credit domain control · E5 CPT vs per-table fine-tuning |
| RQ2 When and why? Does in-context learning already supply the domain signal, and which sources transfer to which targets | E6 labelled-context curves · E7 single-source transfer matrix · per-table analysis |
| RQ3 What changes it? Update size, anchoring and trainable scope against forgetting; exposure, table size and episode size | E8 learning rate × L2-SP · E9 frozen / DiagScale / LoRA · E10 exposure (full pass, 10,000 updates) · E11 small vs large tables · E12 episode size |

800 standard training runs plus 100 single-source runs; E1 is the shared control of
E2–E4 and E8–E12. Every model is evaluated in FP32 on complete five-fold test sets; forgetting is measured
on 19 binary and 12 regression TabArena tables; baselines are CatBoost, XGBoost and logistic/linear
regression. TabPFN LGD trains with a 5-bucket window NLL (robust to the v2.6 regressor's
comb-shaped density); LoRA and DiagScale use L2-SP on their effective weight change. One
comparison is pre-specified as primary (E1 against the original model; PD AUC and Brier, LGD R²
and CRPS; Holm over 16 tests); PD is also scored with the credit-scoring measures KS, H-measure
and expected maximum profit.

## Storage and workflow

The Git checkout and compact logs/manifests live on VSC **DATA**; datasets, checkpoints,
predictions, detailed results and the sealed container live on the Mindwell-native GPFS
**PROJECT** `/gpfs1/project/prj_00013/CreditPFN`. The required output name is `output CreditPFN/`
on both tiers. Edit and test locally, commit and push, then `git pull --ff-only` on DATA and
verify the commit. Container jobs require a runtime acceptance for the current code identity.
Authentication is manual (`step ssh login --context VSC`, run by Andreas).

## Local tests and notebooks

From this repository in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 2
```

The local `.venv` matches the cluster container's exact pins (`config/runtime/`,
`tests/test_environment_parity.py`). Notebooks live in `notebooks/{rq1,rq2,rq3,experiment0,general}`,
contain no functions or classes, call `src/` only and end with a complete text summary. Figures are
LNCS-width (122 mm) PDFs in `output CreditPFN/figures/<group>/`, without a figure-level title (the
caption carries it; the title is kept beside the caption in `CAPTIONS.md`);
`output CreditPFN/All_Results.md` collects the summaries.

## Repository boundaries

Reusable code lives in `src/`; entry points are `scripts/{data,train,eval}_pipeline.py` and
`scripts/probe_row_cap.py`, submitted through `scripts/slurm/`. `tfm-library/` and `vsc-library/`
are pinned read-only references. Never commit datasets, checkpoints, credentials or raw private
rows. The shared figure tree and the output directory name are deliberate project-specific
choices.
