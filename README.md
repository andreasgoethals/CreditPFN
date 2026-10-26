# CreditPFN

CreditPFN studies **how continued pretraining on credit data changes tabular foundation models**. It follows learning curves, adaptation choices, stability, predictive quality and compute cost across TabPFN v2, v2.6, v3 and TabICLv2.

The corpus contains **25 datasets: 17 probability-of-default (PD) classification tables and 8 loss-given-default (LGD) regression tables**. Entire datasets are held out from continued pretraining. At evaluation, each held-out table supplies its own labeled in-context examples and disjoint prediction rows. Thus the task is transfer to an unseen dataset with an available labeled context, not label-free prediction.

The study is descriptive: report the behavior of a specified grid, including unfavorable outcomes and failures. Selecting the best observed recipe and claiming its score as an unbiased generalization estimate would require additional evaluation.

- [Research brief](docs/RESEARCH_BRIEF.md): question, design, implementation choices and remaining evidence; self-contained for sharing.
- [Literature](docs/LITERATURE.md): the closest primary studies and what they support.
- [Results guide](docs/RESULTS_GUIDE.md): research questions, exact figure identities, controls and interpretation limits.
- [VSC runbook](docs/VSC.md): phases, storage, download, fresh-start and recovery commands.
- [Agent memory](docs/AGENTS_MEMORY.md) and [changelog](docs/CHANGELOG.md): historical runs, failures and changes.

## Workflow

1. Register and sanitize the supplied raw datasets on a CPU node.
2. Stage verified processed tables and original weights into site-local scratch; prepare an immutable experiment plan.
3. Validate save/reload behavior and profile throughput with null controls and pilots.
4. Train the fixed grid, recording successful optimizer updates and trajectories.
5. Score trained weights against their own untuned bases and tuned classical baselines on held-out datasets.
6. Consolidate small concurrent output files into compressed analysis tables; build notebooks and figures locally.

The final paper checks preserve experiments 1–7: conditional table-level uncertainty,
source/partition sensitivity and LGD point-versus-distribution diagnostics are in the
main results notebooks. A separate, inference-only TabICL check lives in experiment 0;
see [the finishing-check commands](docs/VSC.md#final-inference-check-before-writing).
It does not retrain models or replace completed benchmark files.

Experiment 8 has a separate LoRA safety gate. A failed inference probe can be inspected
without retraining through `run_experiment0_lora.sh diagnose ID`; preserve the failed
workflow until diagnosis, and follow the [LoRA runbook](docs/VSC.md#experiment-8-lora-workflow).

Each checkpoint carries the effective recipe, corpus membership, base/data/code fingerprints, environment versions and actual trainable modules. Resubmissions reject incompatible checkpoints. Recovery checkpoints additionally preserve optimizer, scheduler, random generators and the sampling cursor.

## Repository

| Path | Responsibility |
|---|---|
| `archive/` | Gitignored historical measurements, compact source records and log summaries |
| `config/` | Data preparation, shared training/evaluation defaults, named experiment phases |
| `scripts/{data,train,eval,context}_pipeline.py` | Experiment entry points |
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
| 0 | `config/experiment0/` | Debugging, zero-LR controls, timing and horizon decisions | Part 1: 16 null + 32 short + 8 recovery pairs; part 2: 8 budget pilots; auxiliary: 52 tiny arms in 8 tasks; follow-ups: 4 correctness + 4 capacity tasks; LoRA: 4 smoke + 4 capacity tasks |
| 1 | `config/experiment1/{pd,lgd}.yaml` | Full learning-rate × L2-SP × adaptation sweep | 512 |
| 2 | `config/experiment2/{pd,lgd}.yaml` | One additional training seed at a predefined reference | 32 additional; seed 42 is reused from experiment 1 |
| 3 | `config/experiment3/{pd,lgd}.yaml` | One sample versus full pass versus accumulation | 96 |
| 4 | `config/experiment4/context.yaml` | Original/CPT learning curves across 29 context sizes, five folds, complete test sets | 250 inference tasks; no CPT |
| 5 | `config/experiment5/{pd,lgd}.yaml` | DiagScale cross-table adaptation at three learning rates | 48 |
| 6 | `config/experiment6/pd.yaml` | All versus large PD training tables at a fixed 8,192-row episode cap | 16 PD |
| 7 | `config/experiment7/pd_rows*.yaml` | 2,048 / 8,192 / 26,000-row episodes on the same large-table corpus | 16 new PD + 8 shared controls from experiment 6 |
| 8 | `config/experiment8/{pd,lgd}.yaml` | Rank-16 query/value LoRA on TabPFN v3 and TabICLv2 at three learning rates | 48; new LoRA correctness/capacity gate required |

`data.yaml`, `train.yaml`, `eval.yaml` and `retention.yaml` hold shared defaults and the fixed public monitoring panel. Named configs provide only experiment differences. Experiments 1–3 use `cpt_*_v5` identities; follow-ups use separate named plans and output groups.

From an active CreditPFN environment on a VSC login node, `bash scripts/slurm/run_experiment0.sh part1` prepares inputs and runs the first three controls with automatic audit gates. Run `part2` separately for the long budget pilots. The launcher never starts experiment 1 automatically; see [the runbook](docs/VSC.md).

For targeted debugging, `bash scripts/slurm/run_experiment0.sh recovery` runs only the eight deterministic recovery pairs in a separate workflow. It cannot replace the complete part-1 validation or authorize budget pilots. Recovery uses smaller batches; the ordinary pilots check production sizes and performance.

`bash scripts/slurm/run_experiment0.sh pilot` runs the short production-size pilots alone with fresh run names and an automatic audit. Optional `--pd-bases` and `--lgd-bases` select `v2`, `v2.6`, `v3` or `tabicl`; omitted selections retain all bases. Generated configs stay with the workflow in manifests. This diagnostic preserves earlier output and stops after its own audit.

`bash scripts/slurm/run_experiment0.sh auxiliary` checks experiments 2 and 3 with six-update seed/sampling/recovery controls. The eight tasks stop at their own audit; research submissions remain separate. The small limits in `config/experiment0/auxiliary.yaml` generate immutable per-case configs in the workflow manifest folder. Notebook `04_auxiliary_experiments` reports the evidence.

Experiments 4–7 require the separate `run_experiment0_followups.sh` correctness/capacity workflow.
Its receipts bind the code, configuration, environment and detailed evidence. Production launchers
refuse missing or stale receipts. See the research brief for fixed recipes and the VSC runbook
for the preparation and launch sequence. Existing experiments keep their original outputs.
Experiments 6 and 7 preserve the original four held-out dataset partitions. Eligibility filters
only their training complements. Experiment 7 binds its 8,192-row control to experiment 6's
immutable plan; it does not duplicate those weights or benchmark results.

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

The notebook reading order is `general/` (raw inputs and processed corpus), `experiment0/` (null, short and budget pilots, then auxiliary controls), `experiment1/` (training then results, each task separately), `experiment2/` (seed-check training then results), `experiment3/` (sampling/accumulation training then results), `experiment4/01_context_curves` (paired context-budget evaluation), `experiment5/` (DiagScale training then results), `experiment6/` (training-corpus policy), `experiment7/` (episode size, including its shared control) and `experiment8/` (native LoRA training then results). Training notebooks follow losses and monitored performance, then compact matched-choice, parameter-movement and cost summaries. Detailed optimizer/device counters remain in tables. Main results notebooks lead with final five-fold choice contrasts and complete recipe maps; fixed-context monitors are a separate later section. The runner discovers these folders recursively; `--only experiment2` selects one study. Each notebook fixes its own run, so experiments cannot silently pool through a global run selector.

Experiment 4 reads project `task_*.csv.gz` metric exports only when their DATA completion receipts match the immutable plan. Unexported SQLite cells remain recovery evidence rather than completed publication records. Its context curves require all five paired folds and keep the table cohort fixed over each displayed range. Experiment 5 labels query-scale movement separately from backbone drift and requires matched original/adapter evaluation within its own inference protocol. Notebooks explicitly report missing PROJECT measurements even when DATA logs establish campaign completion; synthetic plot tests never supply research results.

The runner prints preparation, worker starts, each executing cell/section, completion counts and a heartbeat every 15 seconds. `--workers 2` runs two independent kernels using this Python environment; each kernel defaults to one numerical-library thread to avoid nested CPU oversubscription. Increase `--workers` for more simultaneous notebooks, subject to available memory; `--threads-per-worker` controls numerical threading. Previously expensive notebooks start first, while summaries and captions retain filesystem order. Cell counts show activity, not an estimate of remaining time.

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 2
```

Notebooks export PDFs through `FigureSaver` at the actual ICML paper widths: **6.75 inches full width or 3.25 inches single column**, with embedded TrueType fonts. This is a deliberate, user-approved departure from the template's A4 dimensions. All PDFs live under **`output CreditPFN/figures/<experiment>/`** (corpus exploration uses `figures/general/`), with notebook prefixes and drawing-order numbers. Caption metadata lives beside them in `_metadata/`. Each rerun clears its own previous PDFs; a full run also removes generated figures from retired notebooks and the previous caption index, then rebuilds them. Scientific logs, training diagnostics and evaluation results are preserved. Coverage precedes effects; at most four learning-rate curves share a panel and dataset matrices are paginated. Every caption is displayed below its figure and indexed in `output CreditPFN/figures/CAPTIONS.md`.

Each notebook's final text repeats **every plotted numerical value and displayed table**, without truncation and in section order. Numerical tables use compact CSV rather than wide alignment padding. The runner copies that text verbatim into `output CreditPFN/All_Results.md`. This report and `figures/CAPTIONS.md` are eligible for Git tracking; scientific output and generated PDFs remain ignored. All report logic lives in `src/visualize/`. The private display-name mapping must accompany private data when generating publication output.

For supplied downloads, set `CREDITPFN_ANALYSIS_ROOT` to the **DATA output folder or ZIP**. Optionally set `CREDITPFN_ANALYSIS_PROJECT_ROOT` to the complementary **project output folder or ZIP**. Both accept experiment folders at their root; ZIPs may also contain an enclosing `output CreditPFN/` directory. Nothing is copied, extracted or changed in those inputs. Generated PDFs and summaries remain in this repository's `output CreditPFN/`. A selected external source never silently fills missing evidence from unrelated local output. Unset both variables to read local campaign output; these are analysis settings, separate from cluster storage-root settings.

DATA manifests provide coverage and partition-level endpoint checks. Per-dataset trajectories, loss histories, tensor/resource diagnostics and final benchmark results require project-storage records. Missing sections report unavailable evidence explicitly. The short-pilot notebook uses the latest successful pilot receipt's exact generated configuration, keeping corrected objectives separate from earlier pilots.

Notebooks are the full analysis record, not the proposed number of manuscript figures. Main-grid
results first show matched effects of learning rate, L2-SP and update scope, then all recipe
combinations on held-out credit tables. AUC, Brier score and log loss remain separate; LGD uses
R² and relative RMSE. A favorable choice contrast can mean less damage, while a positive
CPT-versus-original effect means an observed improvement. Maps retain small printed effects and
positive-table counts in their full summaries. No test-selected winner is claimed.

The results guide links the paper's questions to exact figures. Main-grid comparisons require the
same recipe-complete table cohort across displayed metrics. Fixed-reference uncertainty,
probability calibration, prediction shifts and classical controls provide context. Repeated
reliability/threshold panels are no longer emitted by the main notebooks; the complete reference
metric tables retain PR-AUC, F1 and other complementary scores. Calibration summarizes raw,
Platt and isotonic Brier/ECE/log loss on one paired table cohort. LGD keeps point-error and
historical distributional-score diagnostics separate.

Context panels retain individual table effects beneath the equal-table mean. Precision diagnostics
show policy-specific CPT contrasts directly. Cost ratios retain model/partition identities.
Monitor captions and figure metadata distinguish fixed-context measurements from final evaluation;
monitors show change against learning rate instead of repeating a constant initial-score axis.
Main-grid update modes are all-parameter updates or a frozen transformer backbone, **not LoRA**;
experiment 5 separately evaluates DiagScale.

Experiment 8 is a separate, exploratory LoRA follow-up, preserving experiments 1–7.
It freezes every original weight and learns rank-16 additive query/value updates
in all three transformer stages, with alpha 32, zero adapter dropout and no L2-SP
or factor weight decay. Only learning rate varies: `1e-5`, `1e-4`, `1e-3`.
Both original and adapted TabICL benchmarks use FP32 without FA3; raw comparisons
with experiment 1's automatic-policy TabICL results are therefore not isolated
adapter comparisons. Native merged checkpoints use the original inference API;
recovery retains the unmerged factors and optimizer state. No new PEFT package is needed.
The [LoRA workflow](docs/VSC.md#experiment-8-lora-workflow) checks the exact VSC
packages and weights before production; local tests alone do not release it.
Its training/results notebooks follow the same PDF, caption and final-summary conventions.
Before results exist they show the plan and missing-evidence tables, not synthetic curves.

## Environment and contribution rules

The project uses Python 3.11–3.13 and the dependencies in [pyproject.toml](pyproject.toml), including PyTorch 2.3 or later for its AMP API. Dependency ranges are not an environment lock. It depends on specific upstream training APIs; record and validate the working environment rather than upgrading it during a campaign. The cluster environment is named `CreditPFN`; the local environment is `.venv`. Local tests and notebook execution do not replace the GPU correctness/capacity gates for a new experimental protocol.

Read [AGENTS.md](AGENTS.md), [the template](docs/TEMPLATE.md) and [agent memory](docs/AGENTS_MEMORY.md) before editing. Do not push, install packages or start cluster training on the user's behalf without authorization. Never modify the library submodule here. Record substantive changes in both logs and run the relevant validation before reporting completion.

This repository's code is MIT licensed. Dataset permissions and the individual foundation-model weight licenses are separate from the code license.

## Based on the repository template

The layout follows [Andreas' repository template](docs/TEMPLATE.md), with the user-requested folder name and experiment layer: `output CreditPFN/<experiment>/{logs,manifests,training,results,consolidated}`. All publication artifacts instead share `output CreditPFN/figures/`, grouped by experiment, a deliberate departure from the template's per-notebook figure folders. `general/` is reserved for shared cluster maintenance logs and scheduler state, not notebook output. Detailed training diagnostics, predictions and benchmark results use project storage on VSC; logs and small manifests use DATA. Locally these complementary trees share one `output CreditPFN/` directory. Model weights remain in `checkpoints/`, with trained weights grouped by experiment. The gitignored `archive/` remains a separate historical reference. Notebook stdout stays in the executed notebook; the runner rebuilds one `output CreditPFN/All_Results.md` and one shared caption index.
