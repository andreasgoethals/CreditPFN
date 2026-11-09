# CreditPFN

A descriptive study of **domain adaptation by continued pretraining (CPT)** of tabular foundation
models, with credit risk as the domain. PD is binary default classification; LGD is
loss-given-default regression. Four base models: TabPFN v2, v2.6 and v3, and TabICLv2. The study
asks whether CPT on real credit datasets improves predictions on *unseen* credit datasets, how the
adaptation recipe changes that, what is forgotten on non-credit tables, and when and why it works.

## Status (04-11-2026): second campaign, before Experiment 0

The first study (896 runs) is complete and archived locally in `archive/study-2026-11.zip`
(gitignored; `archive/study-2026-09.zip` holds the September run). Its main lessons: the reference
learning rate 3e-7 was the best tested (PD +0.0009 AUC on held-out credit tables, mostly from
TabPFN v2), LGD got slightly worse for every model, and curves were flat by 2,500 updates.

The second campaign is defined in [`config/design.yaml`](config/design.yaml): one reference recipe
plus declared deviations, materialized into `config/experiments/E<NN>_<slug>/` and
`config/experiments/E00_pilots/` by `src/utils/design.py` (tests fail on any undeclared difference or stale generated config). The
method authority is [RESEARCH_BRIEF.md](docs/RESEARCH_BRIEF.md); operations are in [VSC.md](docs/VSC.md).

## Four questions, twelve experiments

| RQ | Experiments |
|---|---|
| RQ1 Transfer: does CPT help on unseen credit tables, how does it develop, how much is randomness, is it domain-specific, is it worth it versus target fine-tuning? | E1 reference · E2 seeds · E3 matched non-credit domain control · E4 CPT vs per-table fine-tuning |
| RQ2 Weight change: update size, anchoring and trainable scope versus forgetting | E5 learning rate × L2-SP · E6 frozen / DiagScale / LoRA |
| RQ3 Training data: which tables, how much of each, episode size, class balance | E7 balanced PD sampling · E8 full pass · E9 small vs large tables · E10 episode size |
| RQ4 When and why: which target tables gain, what they share, which sources help, context scarcity | E11 labelled-context curves · E12 single-source transfer matrix · per-table analysis |

712 standard training runs plus 100 short single-source runs; E1 is the shared control of E2,
E3 and E5–E10. Every model is evaluated in FP32 on complete five-fold test sets; forgetting is measured
on 19 binary and 12 regression TabArena tables; baselines are CatBoost and logistic/linear
regression. TabPFN LGD trains with a 5-bucket window NLL (robust to the v2.6 regressor's
comb-shaped density); LoRA and DiagScale use L2-SP on their effective weight change.

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
`tests/test_environment_parity.py`). Notebooks live in `notebooks/{rq1,rq2,rq3,rq4,experiment0,general}`,
contain no functions or classes, call `src/` only and end with a complete text summary. Figures are
A4-width PDFs in `output CreditPFN/figures/<group>/`; `output CreditPFN/All_Results.md` collects
the summaries.

## Repository boundaries

Reusable code lives in `src/`; entry points are `scripts/{data,train,eval}_pipeline.py` and
`scripts/probe_row_cap.py`, submitted through `scripts/slurm/`. `tfm-library/` and `vsc-library/`
are pinned read-only references. Never commit datasets, checkpoints, credentials or raw private
rows. Substantive changes go in [CHANGELOG.md](docs/CHANGELOG.md), cluster runs and dead ends in
[AGENTS_MEMORY.md](docs/AGENTS_MEMORY.md). The inherited template is
[docs/TEMPLATE.md](docs/TEMPLATE.md); the shared figure tree and the output directory name are
deliberate project-specific choices.
