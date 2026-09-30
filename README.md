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

Reusable Python logic belongs in `src/`; it does not import experiment entry points from `scripts/`. Utilities run as `python -m src.utils.<name>`.

| Experiment | Configs | Purpose | Trials across PD + LGD |
|---|---|---|---|
| 0 | `config/experiment0/{null,pilot,recovery,budget}_{pd,lgd}.yaml` | Debugging, zero-LR controls, timing and horizon decisions | Part 1: 16 null + 32 short + 8 recovery pairs; part 2: 8 budget pilots |
| 1 | `config/experiment1/{pd,lgd}.yaml` | Full learning-rate × L2-SP × adaptation sweep | 512 |
| 2 | `config/experiment2/{pd,lgd}.yaml` | One additional training seed at a predefined reference | 32 additional; seed 42 is reused from experiment 1 |
| 3 | `config/experiment3/{pd,lgd}.yaml` | One sample versus full pass versus accumulation | 96 |

`data.yaml`, `train.yaml`, `eval.yaml` and `retention.yaml` hold shared defaults and the fixed public monitoring panel. Named configs provide only experiment differences. Fresh `cpt_*_v5` identities distinguish this protocol from earlier output.

From an active CreditPFN environment on a VSC login node, `bash scripts/slurm/run_experiment0.sh part1` prepares inputs and runs the first three controls with automatic audit gates. Run `part2` separately for the long budget pilots. The launcher never starts experiment 1 automatically; see [the runbook](docs/VSC.md).

For targeted debugging, `bash scripts/slurm/run_experiment0.sh recovery` runs only the eight deterministic recovery pairs in a separate workflow. It cannot replace the complete part-1 validation or authorize budget pilots. Recovery uses smaller batches; the ordinary pilots check production sizes and performance.

`bash scripts/slurm/run_experiment0.sh pilot` runs the short production-size pilots alone with fresh run names and an automatic audit. Optional `--pd-bases` and `--lgd-bases` select `v2`, `v2.6`, `v3` or `tabicl`; omitted selections retain all bases. Generated configs stay with the workflow in manifests. This diagnostic preserves earlier output and stops after its own audit.

During cluster debugging, output stays on VSC. Download finished output for local analysis once the campaign is complete; files supplied for inspection in Downloads remain there. The two cluster output trees are complementary and are combined under local `output CreditPFN/` at that final download.

## Local inspection and validation

Use the existing local environment. In PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.utils.prepare_experiment --config config/experiment1/pd.yaml
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.utils.run_notebooks
```

The first command previews the design without training. A complete source corpus is required. Plans for actual VSC runs must be written in the VSC environment after the pilot decisions, because data, weights and package versions form part of the identity.

Historical output lives under the gitignored `archive/` directory. Its README describes the merged tables and original records. Active notebooks read `output CreditPFN/`; keeping these trees separate prevents historical trials from being mistaken for fresh results. This local archive is a deliberate extension to the repository template.

The notebook reading order is `00_general/` (raw inputs and processed corpus), `experiment0/` (null, short and budget pilots), `experiment1/` (training then results, each task separately), `experiment2/` (seed-check training then results), and `experiment3/` (sampling/accumulation training then results). Training notebooks follow losses, monitored performance, parameter movement and device counters over updates. Results notebooks distinguish fixed-context monitor endpoints from final five-fold evaluation. The runner discovers these folders recursively; `--only experiment2` selects one study. Each notebook fixes its own run, so experiments cannot silently pool through a global run selector.

Notebooks export PDFs through `FigureSaver` at the actual ICML paper widths: **6.75 inches full width or 3.25 inches single column**, with embedded TrueType fonts. This is a deliberate, user-approved departure from the template's A4 dimensions. All PDFs share **`output CreditPFN/figures/`**, with experiment/notebook prefixes and drawing-order numbers. Each rerun clears its own previous PDFs; a full run also removes generated figures from retired notebooks and the previous caption index, then rebuilds them. Scientific logs, training diagnostics and evaluation results are preserved. Coverage precedes effects; at most four learning-rate curves share a panel and dataset matrices are paginated. Every caption is displayed below its figure and indexed in `output CreditPFN/figures/CAPTIONS.md`.

Each notebook's final text repeats **every plotted numerical value and displayed table**, without truncation and in section order. The runner copies that text verbatim into `output CreditPFN/general/All_Results.md`. All report logic lives in `src/visualize/`. The private display-name mapping must accompany private data when generating publication output.

For supplied downloads, set `CREDITPFN_ANALYSIS_ROOT` to the **DATA output folder or ZIP**. Optionally set `CREDITPFN_ANALYSIS_PROJECT_ROOT` to the complementary **project output folder or ZIP**. Both accept experiment folders at their root; ZIPs may also contain an enclosing `output CreditPFN/` directory. Nothing is copied, extracted or changed in those inputs. Generated PDFs and summaries remain in this repository's `output CreditPFN/`. A selected external source never silently fills missing evidence from unrelated local output. Unset both variables to read local campaign output; these are analysis settings, separate from cluster storage-root settings.

DATA manifests provide coverage and partition-level endpoint checks. Per-dataset trajectories, loss histories, tensor/resource diagnostics and final benchmark results require project-storage records. Missing sections report unavailable evidence explicitly. The short-pilot notebook uses the latest successful pilot receipt's exact generated configuration, keeping corrected objectives separate from earlier pilots.

## Environment and contribution rules

The project uses Python 3.11–3.13 and the dependencies in [pyproject.toml](pyproject.toml). It depends on specific upstream training APIs; record and validate the working environment rather than upgrading it during a campaign. The cluster environment is named `CreditPFN`; the local environment is `.venv`.

Read [AGENTS.md](AGENTS.md), [the template](docs/TEMPLATE.md) and [agent memory](docs/AGENTS_MEMORY.md) before editing. Do not push, install packages or start cluster training on the user's behalf without authorization. Never modify the library submodule here. Record substantive changes in both logs and run the relevant validation before reporting completion.

This repository's code is MIT licensed. Dataset permissions and the individual foundation-model weight licenses are separate from the code license.

## Based on the repository template

The layout follows [Andreas' repository template](docs/TEMPLATE.md), with the user-requested folder name and experiment layer: `output CreditPFN/<experiment>/{logs,manifests,training,results,consolidated}`. All publication PDFs instead share `output CreditPFN/figures/`, a deliberate departure from the template's per-notebook figure folders. General exploration metadata and shared summaries use `output CreditPFN/general/`. Detailed training diagnostics, predictions and benchmark results use project storage on VSC; logs and small manifests use DATA. Locally these complementary trees share one `output CreditPFN/` directory. Model weights remain in `checkpoints/`, with trained weights grouped by experiment. The gitignored `archive/` remains a separate historical reference. Notebook stdout stays in the executed notebook; the runner rebuilds one `output CreditPFN/general/All_Results.md` and one shared caption index.
