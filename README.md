# CreditPFN

CreditPFN studies **how continued pretraining on credit data changes tabular foundation models**. It follows learning curves, adaptation choices, stability, predictive quality and compute cost across TabPFN v2, v2.6, v3 and TabICLv2.

The corpus contains **25 datasets: 17 probability-of-default (PD) classification tables and 8 loss-given-default (LGD) regression tables**. Entire datasets are held out from continued pretraining. At evaluation, each held-out table supplies its own labeled in-context examples and disjoint prediction rows. Thus the task is transfer to an unseen dataset with an available labeled context, not label-free prediction.

The study is descriptive: report the behavior of a specified grid, including unfavorable outcomes and failures. Selecting the best observed recipe and claiming its score as an unbiased generalization estimate would require additional evaluation.

- [Research brief](docs/RESEARCH_BRIEF.md): question, design, implementation choices and remaining evidence; self-contained for sharing.
- [Literature](docs/LITERATURE.md): the closest primary studies and what they support.
- [VSC runbook](docs/VSC.md): phases, storage, download, fresh-start and recovery commands.
- [Agent memory](docs/AGENTS_MEMORY.md) and [changelog](docs/CHANGELOG.md): historical runs, failures and changes.

## Workflow

1. Register and sanitize the supplied raw datasets on a CPU node.
2. Stage verified processed tables and original weights into site-local scratch; prepare an immutable experiment plan.
3. Validate save/reload behavior and profile throughput with null controls and pilots.
4. Train the fixed grid, recording successful optimizer updates and trajectories.
5. Score trained weights against their own untuned bases and tuned classical baselines on held-out datasets.
6. Consolidate small concurrent output files into compressed analysis tables; build notebooks and figures locally.

Each checkpoint carries the effective recipe, corpus membership, base/data/code fingerprints, environment versions and actual trainable modules. Resubmissions reject incompatible checkpoints. Recovery checkpoints additionally preserve optimizer, scheduler, random generators and the sampling cursor.

## Repository

| Path | Responsibility |
|---|---|
| `archive/` | Gitignored historical measurements, compact source records and log summaries |
| `config/` | Data preparation, shared training/evaluation defaults, named experiment phases |
| `scripts/{data,train,eval}_pipeline.py` | Experiment entry points |
| `scripts/probe_row_cap.py` | GPU capacity experiment entry point |
| `scripts/slurm/` | Resource wrappers, shared job bodies and bounded submission launcher |
| `src/data/` | Registration, sanitization and display names |
| `src/train/` | Phase/grid resolution, sampling, objectives, adaptation, recovery and capacity measurements |
| `src/model/`, `src/eval/` | Inference wrappers, evaluation configuration, row splits, baseline tuning and metrics |
| `src/utils/` | Plan preparation, staging, auditing, consolidation and cleanup |
| `src/visualize/`, `notebooks/` | Shared plotting logic and thin notebooks |
| `tests/` | Unit, integration and synthetic training checks |
| `tfm-library/` | Read-only pinned literature and upstream implementation snapshots |

Raw data and weights are never committed. On VSC, project storage holds canonical data, current weights and consolidated results; DATA holds the repository and live small output shards. The [runbook](docs/VSC.md) explains how to download finished results and clear a previous experiment. Historical output is optional local evidence, not an input to a fresh run.

Reusable Python logic belongs in `src/`; it does not import experiment entry points from `scripts/`. Utilities run as `python -m src.utils.<name>`. The stable `experiment0` and `experiment1` config names mean null controls and the main sweep; `pilot`, `budget_pilot`, `sampling` and `seeds` name the other phases.

During cluster debugging, output stays on VSC. Download finished output for local analysis once the campaign is complete; files supplied for inspection in Downloads remain there. The two cluster output trees are complementary and are combined under local `output CreditPFN/` at that final download.

## Local inspection and validation

Use the existing local environment. In PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.utils.prepare_experiment --config config/experiment1_pd.yaml
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
```

The first command previews the design without training. A complete source corpus is required. Plans for actual VSC runs must be written in the VSC environment after the pilot decisions, because data, weights and package versions form part of the identity.

Historical output lives under the gitignored `archive/` directory. Its README describes the merged tables and original records. Active notebooks read `output CreditPFN/`; keeping these trees separate prevents historical trials from being mistaken for fresh results. This local archive is a deliberate extension to the repository template.

Notebooks use A4-sized PDF figures through `FigureSaver` and finish with a text summary. They handle missing results, retain failed attempts and prefer verified consolidated tables. The private display-name mapping must accompany private data when generating publication output.

## Environment and contribution rules

The project uses Python 3.11–3.13 and the dependencies in [pyproject.toml](pyproject.toml). It depends on specific upstream training APIs; record and validate the working environment rather than upgrading it during a campaign. The cluster environment is named `CreditPFN`; the local environment is `.venv`.

Read [AGENTS.md](AGENTS.md), [the template](docs/TEMPLATE.md) and [agent memory](docs/AGENTS_MEMORY.md) before editing. Do not push, install packages or start cluster training on the user's behalf without authorization. Never modify the library submodule here. Record substantive changes in both logs and run the relevant validation before reporting completion.

This repository's code is MIT licensed. Dataset permissions and the individual foundation-model weight licenses are separate from the code license.

## Based on the repository template

The layout follows [Andreas' repository template](docs/TEMPLATE.md), with the user-requested output directory name `output CreditPFN/` replacing the template's `output/`. Cluster/debugging logs live in `output CreditPFN/logs/`, configurations and figure metadata in `output CreditPFN/manifests/`, metrics in `output CreditPFN/results/`, and PDFs in `output CreditPFN/figures/<notebook>/`. Notebook stdout stays in the executed notebook; `All_Results.md` is rebuilt from its final summary cell, without a separate transcript. Model weights use the template's `checkpoints/` extension: locally in this repository, permanently on project storage on VSC. The user-requested local `archive/` is the deliberate additional folder; cleanup also removes trained weights when explicitly requested, while preserving original bases.
