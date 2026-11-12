# AGENTS.md — CreditPFN

Instructions for AI agents working in this repository.

## 0. Before you start

1. [`docs/TEMPLATE.md`](docs/TEMPLATE.md) — the layout and rules this project started from.
   **Follow it**, and when you deviate — which is allowed, it is a starting point rather than a
   contract — **say so in your reply**. Never silently.
2. [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — the cluster runs already done and the dead
   ends already hit. Reading it is not optional: it is how you avoid resubmitting a configuration
   that failed last month, or spending an hour on a known dead end.
3. [`README.md`](README.md) — what this project actually is.

## 1. `tfm-library/` IS READ-ONLY. NO EXCEPTIONS BUT ONE.

A **pinned git submodule** holding the shared TFM literature. This repository does not track its
contents.

It is here so you can answer *"what does the literature say?"* and *"how does the official
implementation do this?"* **by reading and grepping files in this repository** — offline, no web
search, nothing from memory. Use it. A claim you can point at a path for beats a confident
sentence.

**Never create, edit, move, or delete anything inside it** — not a typo fix, not a note, not a
reformat. Anything you write there is lost when the pin moves, or corrupts a resource every other
project shares. **The one exception** is `tfm-library/PROJECT_SPECIFIC.md`, gitignored by the
library for exactly this purpose and created from `PROJECT_SPECIFIC.template.md`. If a library
document is wrong, report it to Andreas rather than patching it — the fix belongs in the
library's own checkout, where it flows down to every consumer. Never lint, format or test it.

Cite papers by path (`tfm-library/papers/<year>/...`, full text under `papers/text/`), and **code
dumps by symbol name, never by line number** — the dumps are re-snapshotted and line numbers drift
by thousands. Record the pin (`git submodule status`) next to any result that depends on it. Bump it with
`python -m src.utils.update_tfm_library`.

## 2. Never commit data or checkpoints

`data/` holds datasets under varying licences; `checkpoints/` holds multi-hundred-MB weights. Both
are gitignored. Do not `git add -f` them, and do not paste raw rows into commits, issues or docs.

## 3. Verify before you assert

This project's value is careful measurement. If a claim cannot be confirmed from the library, the
upstream source, or a primary reference, **say so** rather than filling the gap plausibly.
Distinguish what a paper *evaluated* from what its code merely *supports*; what a mechanism *can*
represent from how *often* it occurs; a library annotation from the primary source it summarises.

## 4. Do not train or install without asking

Cluster runs cost real VSC credits. Installs change a shared environment. Ask first.
On 26-10-2026 Andreas authorized agents to push reviewed and validated changes themselves.
Keep scientific recipe changes subject to explicit agreement; a push is not permission to submit cluster work.

### VSC access, storage and campaign changes

Follow the current [template access and storage policy](docs/TEMPLATE.md#vsc-access-windows-and-agents)
and the CreditPFN runbook [docs/VSC.md](docs/VSC.md) (access, storage, runtime, deployment).
Authentication is manual: Andreas runs `step ssh login --context VSC` in Windows PowerShell
and approves KU Leuven Authenticator. Reuse the existing Windows SSH agent and native Windows
OpenSSH. Check certificate validity when access is needed; request that command only if the
certificate is unavailable or expired. Never initiate authentication, install another SSH setup,
export credentials or create renewal reminders.

For relevant authorized work, inspect VSC files, logs, results, quotas and jobs directly.
Use the template's LF-script-over-stdin connection with strict host-key checking and no trailing
remote command. Verify an output marker and the actual exit status; empty exit-zero output is
not execution evidence. The 29-10 gateway check needed `exec 2>&1` inside the remote script to
retain error output; verify both test markers through the combined stream. Keep bounded metadata
reads on login nodes and authorized computational debugging in Slurm allocations. Distinguish
local validation from real CPU/GPU acceptance.

Deploy through Git: edit and validate locally, commit and push, then update a clean DATA
checkout with `git pull --ff-only` and verify its commit before submission. Use a separate
DATA checkout if existing jobs need the old revision. Do not copy uncommitted source files
or patch VSC source directly; a cluster-discovered fix repeats the same local/Git loop.

The actual checkout and `.git` belong on DATA. Large persistent assets, model caches and immutable,
versioned SIF runtime images belong on approved PROJECT storage; temporary work belongs on native
scratch. Do not move an installed environment: recreate it from recorded specifications in a
container and validate before cutover. Match intensive I/O to the compute filesystem, stage only
the required working set, publish recovery state durably, and never blanket-delete scratch.

The old Conda-on-DATA runtime is retained only as the runtime-acceptance comparator; the sealed
PROJECT container `env20261029f` serves the campaign. Computational activation requires
`CREDITPFN_RUNTIME_VERSION`; only the explicit comparison arm may select the old environment.
Preserve source, runtime,
inputs, receipts and recovery state used by active, queued or resumable jobs. Explain and obtain
agreement for migration, installations, cleanup or new workloads before executing them. Future
launchers submit independent work without artificial running caps; real validation and data
dependencies, live site admission limits and measured per-task resources still apply.

## 5. Windows PowerShell 5.1 — no `&&`

No `&&`, no ternary, no `??`. One command per line, or `;` with `if ($?) { ... }`:

```powershell
python -m venv .venv
if ($?) { .\.venv\Scripts\Activate.ps1 }
```

SLURM job scripts are a separate world — bash on Linux, normal POSIX syntax. Keep the two straight.

## 6. Write both logs

Newest first, dates `DD-MM-YYYY`.

- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — every substantive change, one bullet, **as short as
  possible**: what, and why only if it is not obvious. Usually one line; longer only when the change
  genuinely is. The detail belongs in the commit.
- [`docs/AGENTS_MEMORY.md`](docs/AGENTS_MEMORY.md) — two things. **Every cluster run**, one row in
  the table: config, outcome, headline number. And **every failure** that cost more than a couple of
  minutes, four lines: **Tried**, **Result**, **Why**, **Instead** — even when the eventual fix
  worked, because the dead end is the expensive part.

## 7. Notebooks and figures

- A notebook contains **no `def` and no `class`** — logic goes in `src/` — and its **last code
  cell prints a text summary**, section by section, in the notebook's own section order.
- Notebooks follow the four RQs and eleven experiments of `config/design.yaml`:
  RQ1 E1 reference, E2 seeds, E3 matched non-credit domain control, E4 CPT vs target
  fine-tuning; RQ2 E5 update size × L2-SP, E6 trainable scope; RQ3 E7 full
  pass, E8 small vs large tables, E9 episode size; RQ4 E10 context scarcity (context curves),
  E11 single-source transfer matrix and the per-table analysis.
  Experiment 0 owns validation (`experiment0/`); corpus notebooks use `general/`. Folders are
  `rq1/`–`rq3/` (one `01_training` and one `02_results` notebook each), `rq4/`, `experiment0/`
  and `general/`. E1 is the shared control of E2, E3 and E5–E9: never count it twice. Combine
  experiments in a figure when their quantities and comparison units agree; retain
  individual-table evidence.
- **Never pick a colour or a size.** `src/visualize/style.py` owns both, so every notebook here
  looks the same. Add a new one there, once, not in the notebook.
- Save through `src/visualize/figures.FigureSaver`: **PDF only**, all in
  `output CreditPFN/figures/<notebook group>/` (`rq1`–`rq4`, `experiment0`, `general`). Notebook filename prefixes identify ownership;
  each notebook clears only its own files before drawing. Full reruns also retire figures
  from renamed notebooks. This shared publication tree deliberately overrides the template's per-notebook folders.
  The notebook displays each figure inline.
- Use `style.figsize(style.WIDTH_FULL)` or `style.figsize(style.WIDTH_HALF)`: every figure is drawn
  at the width it will occupy in the **A4 paper** (160 mm full width, 77 mm single column), and never
  rescaled afterwards. On 03-11-2026 the user replaced the ICML US Letter geometry chosen on
  30-09-2026 with this A4 article geometry, matching the template.
- Title, caption and legend complement each other: the title names what is compared, the caption
  states how it was measured, the legend decodes every colour and line style. Merge panels that share
  a quantity and comparison unit into one figure rather than repeating near-identical figures.
- Captions are the **paper's** captions: pure description, ready to paste under the figure.
- Bind `NotebookReport(..., sink=sink)` and display tables through `report.table`. The final
  summary must contain every plotted value and displayed table without truncation, not just
  aggregate conclusions.

## 8. Say you are done only when it runs

```powershell
python -m pytest -q
```

And `python -m src.utils.run_notebooks` if you touched a notebook or anything under
`src/visualize/`. Nothing runs these for you — no CI, no hook — so run them before you claim
anything is finished.

## 9. CreditPFN specifics

- **Push validated changes when completing authorized implementation work.** The user granted this
  permission on 26-10-2026. Verify the remote result and tell Andreas to pull on VSC before submission;
  a local commit alone is not deployed there. Do not change cluster files used by active jobs.
- **No secrets, credentials, private dataset contents or large log excerpts** in any document —
  both dated logs included, since both are tracked.
- **Second campaign (from 03-11-2026).** The first study is archived locally in
  `archive/study-2026-11.zip` (verified pack, `PACK_MANIFEST.json` inside); its VSC outputs were
  deleted and no code serves it any more.
  `config/design.yaml` is the grid authority: one reference recipe plus declared deviations,
  materialized by `src/utils/design.py` into `config/experiments/E00_pilots/` (Experiment 0) and
  `config/experiments/E<NN>_<slug>/` (E1–E11). Never hand-edit a generated config;
  `tests/test_design.py` fails on undeclared differences or stale files. Raise open decisions
  with Andreas in chat, not in planning documents.
- **Check which environment you are in.** Locally `.venv/Scripts/python.exe` (Python 3.13, pinned
  to the accepted container lock in `config/runtime/`; `tests/test_environment_parity.py`). Historical cluster
  evidence uses the `CreditPFN` Conda environment, printed as `Active conda env:`; preserve it as
  a comparator/rollback while the accepted PROJECT-container campaign runs. New jobs print
  `Active sealed runtime:`. Runtime acceptance must
  record the SIF digest, actual interpreter and package versions. An active virtualenv can
  silently shadow Conda (`AGENTS_MEMORY.md`, 05-08-2026); never infer the interpreter from a prompt.
- **Output directory:** `output CreditPFN/<group>/` is the user's required name on both storage
  tiers, a deliberate deviation from the template's generic `output/`. Preserve the space and the
  group layer; quote shell paths. Groups are `general`, `experiment0` and the experiment slugs of
  `config/design.yaml` (`paths.CAMPAIGN2_GROUPS`), set by each generated config's
  `experiment.output_group`. Always build paths through `src/utils/paths.py`. Training
  diagnostics and results go to project storage, logs and small manifests to DATA; trained
  weights to `checkpoints/trained/<group>/`.
- **Keep cluster debugging on VSC.** Read Downloads in place when asked; do not copy its files
  into the repository. The user downloads the completed campaign at the final analysis stage.
  Local notebook stdout stays in the executed `.ipynb`, with its final summary in
  `output CreditPFN/All_Results.md`; no duplicate notebook logs or locks. Caption metadata remains in
  `output CreditPFN/figures/<notebook group>/_metadata/`. Give VSC commands in ordinary messages, one command per code block.
- Publication paths live in `src/visualize/paths.py`, using the output root from
  `src/utils/paths.py`.
- Training notebooks cover trajectories, optimizer and resource diagnostics; result notebooks
  cover endpoint comparisons and final benchmarks, explicitly distinguishing monitor outcomes
  from five-fold evaluation. RQ1–RQ3 combine their respective experiments by phase; RQ4 is evaluation only.
- LoRA uses `train.adaptation_mode: lora` (native query/value adapters). Adapter scopes (LoRA,
  DiagScale) apply L2-SP to the effective weight change (`src/train/l2sp.py`); never weight decay
  on LoRA factors. TabPFN LGD trains with the 5-bucket window NLL (`regression_objectives.py`).
- **Storage:** PROJECT is the Mindwell-native GPFS project `/gpfs1/project/prj_00013/CreditPFN`
  (moved from Lustre on 04-11-2026); all jobs run on Mindwell. `prj_00013` also holds other
  projects of Andreas: never touch anything outside `CreditPFN/`.
- **Where the knowledge lives.** `docs/RESEARCH_BRIEF.md` is the self-contained research brief and
  method authority: descriptive goals, partitions, sampling, adaptation and limitations.
  `docs/VSC.md` owns operations, storage, checkpoint formats and measured caps (do not raise one
  without a new probe). `docs/LITERATURE.md` records primary-source support and differences.
  Historical measurements and expensive dead ends stay in `docs/AGENTS_MEMORY.md`.
- **Entry points:** `scripts/{data,train,eval}_pipeline.py` and `scripts/probe_row_cap.py`,
  submitted through `scripts/slurm/`: `campaign_cpu.slurm` (Experiment 0 controller, per-experiment
  submission, summary), `pilot_gpu.slurm`, `run_experiment.sh` (train/eval chains over `train_pd`/`train_lgd`
  and `eval_pd`/`eval_lgd`/`eval_classical`), `campaign_eval_gpu.slurm` (E4 fine-tuning and E10
  context curves via `src/eval/campaign_tasks.py`), `runtime_{build,cpu,gpu}.slurm`, `data.slurm`
  and `probe_row_cap.slurm`. Everything else is a utility under `src/utils/`, run with `python -m`.

## Shared VSC Library

`vsc-library/` is a pinned, entirely read-only infrastructure submodule. Read its
README, source snapshot, guides and Slurm examples; never edit, refresh, lint,
format or test inside it. There is no PROJECT_SPECIFIC exception. Keep this
project's VSC choices and operating instructions in `docs/VSC.md`, outside the
submodule. Shared corrections belong in the standalone VSC Library checkout.
Record the library pin when operational evidence depends on it.
