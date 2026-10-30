# CreditPFN

A descriptive study of **domain adaptation by continued pretraining (CPT)** of tabular foundation models, using credit risk as the case study. PD is binary default classification; LGD is loss-given-default regression. The study asks when transfer helps, how the adaptation recipe changes that effect, and what is retained on non-credit tables. It does not assume CPT is uniformly successful or unsuccessful.

The method authority is [RESEARCH_BRIEF.md](docs/RESEARCH_BRIEF.md); primary-source support is in [LITERATURE.md](docs/LITERATURE.md). Start the analysis with the [PD overview](notebooks/general/03_overview_pd.ipynb) or [LGD overview](notebooks/general/04_overview_lgd.ipynb). [RESULTS_GUIDE.md](docs/RESULTS_GUIDE.md) maps questions to the detailed notebooks.

## Four questions, ten analyses

| Question | Experiment | Contrast |
|---|---|---|
| RQ1: When does CPT help, and how stable is it? | 1 | Learning rate and training progress, with full updates and L2-SP 0.003 |
| | 2 | A second training seed at the predefined reference |
| RQ2: How do restrictions on updates change transfer? | 3 | L2-SP at both full and frozen update scopes, including their interaction |
| | 4 | Backbone freezing at both L2-SP levels, including their interaction |
| | 5 | The specified DiagScale recipe at three learning rates |
| | 6 | The specified native query/value LoRA recipe at three learning rates |
| RQ3: How do data and exposure affect adaptation? | 7 | One sample, full pass and accumulation at equal successful updates |
| | 8 | All versus large PD training tables, fixed episode size |
| | 9 | Episode size on a fixed large-PD-table corpus |
| RQ4: Does CPT change the need for target-table labels? | 10 | Nested context sizes; original/CPT performance and paired differences; complete fixed five-fold test sets |

Experiment 0 contains correctness, recovery, capacity and runtime checks. Experiments 8–9 are PD-only. All other scientific studies cover PD and LGD. The original model coverage is TabPFN v2/v2.6/v3 and TabICLv2 in the shared factorial, seeds and sampling studies; the approved isolated extension fills v2/v2.6 coverage in the follow-ups. New results appear only after their actual VSC gates and audits pass.

The **512 unique factorial training runs are stored once** and support experiments 1, 3 and 4. The original campaigns contain 768 unique completed training runs; the approved model extension adds 128, for 896 if all succeed. Reused controls are never counted as new runs. DiagScale and LoRA differ in several recipe settings: these results cannot identify an intrinsically superior adapter mechanism.

## Stable campaign identities and storage

`config/study.yaml` defines the new presentation. Historical config directories, run names, trial IDs and checkpoint paths remain immutable:

| Historical campaign/config | Public analysis / output payload |
|---|---|
| `experiment1` / `cpt_main_v5` | E1/E3/E4; `shared/main_grid` |
| `experiment2` / `cpt_seeds_v5` | E2; `experiment2` |
| `experiment3` / `cpt_sampling_v5` | E7; `experiment7` |
| `experiment4` / context | E10; `experiment10` |
| `experiment5` / DiagScale | E5; `experiment5` |
| `experiment6` / corpus policy | E8; `experiment8` |
| `experiment7` / episode size | E9; `experiment9` |
| `experiment8` / LoRA | E6; `experiment6` |

The hash-verified migration publishes `STUDY_LAYOUT.json` only after both supplied storage tiers pass. Readers support old downloads and migrated output; never manually merge those two layouts. Checkpoints retain `checkpoints/trained/<historical campaign>/`, with additional models isolated under `extensions/<ID>/`. Old controls and superseded inference evidence remain recoverable. Commands accepting historical config/campaign IDs retain that meaning; public notebook numbers are not interchangeable with those arguments.

On VSC, the actual Git checkout and compact logs/manifests live on **DATA**. Canonical datasets, checkpoints, predictions, consolidated results and the immutable versioned container live on **PROJECT**. Native cluster scratch is temporary working storage. The required output name remains `output CreditPFN/` on both tiers. [VSC.md](docs/VSC.md) owns exact paths, deployment, gates, submission, collection and cleanup.

## Development and cluster workflow

Edit and test locally, commit and push, then use `git pull --ff-only` in a clean DATA checkout and verify the deployed commit. Never replace files used by active or recoverable jobs; use a separate DATA checkout when necessary. Untracked source uploads and direct remote source patches are not deployments.

Computational jobs must explicitly select an accepted PROJECT container with `CREDITPFN_RUNTIME_VERSION`; they refuse silent fallback to the old DATA environment. The preserved original environment is used only by an explicit compatibility-comparison arm. The Slurm submission bootstrap can queue a fresh acceptance graph from the sealed container itself, without project dependencies installed on a login node.

Authentication uses the existing Windows SSH agent. When its certificate is missing or expired, Andreas manually runs `step ssh login --context VSC` and approves KU Leuven Authenticator. Agents reuse that access; they do not initiate authentication, install a second agent or create renewal reminders. Direct inspections use the tested stdin-script transport in [TEMPLATE.md](docs/TEMPLATE.md#vsc-access-windows-and-agents).

The approved migration workflow gates actual work in this order: recorded container build → CPU and same-GPU compatibility/capacity checks → two-tier hash-verified output migration → independent FP32 re-evaluation and the extension's experiment-0 checks → extension research → independent prediction/cohort audits and publication. Independent arrays have no artificial running-task throttle; actual Slurm admission limits and scientific dependencies still apply. Submission is not success. Preserve the old environment until acceptance and rollback requirements are satisfied.

## Local notebooks and tests

The 15 notebooks are organized by question rather than by individual experiment:
`rq1/` combines E1/E2, `rq2/` E3–E6, `rq3/` E7–E9, and `rq4/` E10.
RQ1–RQ3 each have `01_training` and `02_results`; RQ4 has `01_context_curves`.
`experiment0/` retains four validation notebooks and `general/` the two corpus notebooks
and PD/LGD overviews. This deliberate template deviation changes presentation folders only;
scientific output, configuration and checkpoint identities keep the mapping above.

From this repository in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run every notebook in parallel, with live progress:

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 2
```

The runner prevents overlapping runs, regenerates each notebook's own publication output, retires renamed notebooks' figures, embeds the figures in the executed notebooks and produces:

- `output CreditPFN/figures/<notebook group>/`: publication-width PDFs and ownership metadata, grouped as `rq1`–`rq4`, `experiment0` and `general`.
- `output CreditPFN/figures/CAPTIONS.md`: paper-ready descriptions, ordered by notebook and figure occurrence.
- `output CreditPFN/All_Results.md`: every notebook's complete final printed summary, in filesystem order.

Every notebook ends with a section-ordered text summary containing all plotted values and displayed tables. Figures use shared styles and actual ICML widths. PDF files and raw metadata stay local; executed notebooks, captions and the combined summaries are tracked. Missing measurements are explicit, never fabricated. Training notebooks describe trajectories; result notebooks distinguish monitoring from final five-fold evaluation.

For additional downloaded evidence, use the supported input options documented by:

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --help
```

## Repository boundaries

Reusable implementation lives in `src/`; scientific entry points are `scripts/{data,train,eval,context}_pipeline.py` and `scripts/probe_row_cap.py`. Slurm wrappers live in `scripts/slurm/`; notebook code is thin and defines no functions/classes. The standard benchmark dispatcher is reused by the independently audited re-evaluation controller.

`tfm-library/` and `vsc-library/` are pinned read-only references. Never lint, format or test them. Never commit datasets, checkpoints, credentials or raw private rows. Record substantive changes and cluster outcomes in [CHANGELOG.md](docs/CHANGELOG.md) and [AGENTS_MEMORY.md](docs/AGENTS_MEMORY.md). The inherited template is [docs/TEMPLATE.md](docs/TEMPLATE.md); the shared figure tree, output directory name and ICML widths are deliberate project-specific choices.
