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
and [CreditPFN migration status](docs/VSC.md#access-storage-policy-and-migration-status).
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

The old Conda-on-DATA runtime is retained as a comparator/rollback; the accepted PROJECT container
`env20261029f` now serves the approved migration campaign. The old environment and throttled launchers
are not defaults for a new campaign. Computational activation requires `CREDITPFN_RUNTIME_VERSION`;
only the explicit compatibility-comparison arm may select the old environment. Read the migration
status before deployment. Preserve source, runtime,
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
- Public notebooks use four RQs and ten analyses: E1 learning rate/progress, E2 seeds,
  E3 L2-SP, E4 freezing, E5 DiagScale, E6 LoRA, E7 sampling, E8 corpus composition,
  E9 episode size, E10 target context. E0 owns validation; corpus/overview notebooks use
  `general/`. Notebook folders are `rq1/`–`rq4/`, `experiment0/` and `general/`: one combined
  training notebook and one combined results notebook per RQ1–RQ3, and an evaluation-only
  context notebook for RQ4. Combine experiments in a figure when their quantities and
  comparison units agree; retain explicit recipe differences and individual-table evidence.
  `config/study.yaml` maps immutable recorded campaign/checkpoint IDs
  to the public layout. E1/E3/E4 reuse the same 512-run factorial stored once in
  `shared/main_grid`; never count those analyses as independent runs or drop the interaction.
- **Never pick a colour or a size.** `src/visualize/style.py` owns both, so every notebook here
  looks the same. Add a new one there, once, not in the notebook.
- Save through `src/visualize/figures.FigureSaver`: **PDF only**, all in
  `output CreditPFN/figures/<notebook group>/` (`rq1`–`rq4`, `experiment0`, `general`). Notebook filename prefixes identify ownership;
  each notebook clears only its own files before drawing. Full reruns also retire figures
  from renamed notebooks. This shared publication tree deliberately overrides the template's per-notebook folders.
  The notebook displays each figure inline.
- Use `style.figsize(style.WIDTH_FULL)` or `WIDTH_HALF`: every figure is drawn at the width it
  will occupy in the **ICML paper** (6.75 inches full width, 3.25 inches single column), and never
  rescaled afterwards. The user explicitly chose the actual ICML US Letter geometry over the
  template's generic A4 dimensions on 30-09-2026.
- Captions are the **paper's** captions: pure description, ready to paste under the figure.
- Bind `NotebookReport(..., sink=sink)` and display tables through `report.table`. The final
  summary must contain every plotted value and displayed table without truncation, not just
  aggregate conclusions. Read supplied DATA/project folders or ZIPs through `src/visualize/inputs.py`.

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
- **Check which environment you are in.** Locally `.venv/Scripts/python.exe`. Historical cluster
  evidence uses the `CreditPFN` Conda environment, printed as `Active conda env:`; preserve it as
  a comparator/rollback while the accepted PROJECT-container campaign runs. New jobs print
  `Active sealed runtime:`. Runtime acceptance must
  record the SIF digest, actual interpreter and package versions. An active virtualenv can
  silently shadow Conda (`AGENTS_MEMORY.md`, 05-08-2026); never infer the interpreter from a prompt.
- **Output directory:** `output CreditPFN/<experiment>/` is the user's required name on both storage tiers, a deliberate deviation from the template's generic `output/`. Preserve the space and experiment layer; quote shell paths. Use `src/utils/paths.py`. Detailed training diagnostics and results go to project storage; logs/small manifests to DATA. Weights retain `checkpoints/trained/<recorded campaign>/`. Config directories and launcher names use public numbers; embedded `output_group` and internal `CREDITPFN_EXPERIMENT` are recorded campaign IDs, mapped by the YAML authority. Never infer an output owner from a public config folder number.
- **Keep cluster debugging on VSC.** Read Downloads in place when asked; do not copy its files
  into the repository. The user downloads the completed campaign at the final analysis stage.
  Local notebook stdout stays in the executed `.ipynb`, with its final summary in
  `output CreditPFN/All_Results.md`; no duplicate notebook logs or locks. Caption metadata remains in
  `output CreditPFN/figures/<notebook group>/_metadata/`. Give VSC commands in ordinary messages, one command per code block.
- Publication paths live in `src/visualize/paths.py`, using the output root from
  `src/utils/paths.py`. Its legacy figure helpers remain for migration: changing that
  runtime-fingerprinted file during active training would invalidate prepared plans.
- Training notebooks cover trajectories, optimizer and resource diagnostics; result notebooks
  cover endpoint comparisons and final benchmarks, explicitly distinguishing monitor outcomes
  from five-fold evaluation. RQ1–RQ3 combine their respective experiments by phase; RQ4 is evaluation only.
- Public experiment 6 (recorded checkpoint campaign 8) uses `train.adaptation_mode: lora`, not the legacy frozen/PEFT switch.
  Its gate is `CREDITPFN_LORA_GATE`; debugging output belongs to `experiment0/lora/<ID>/`
  on both tiers. Preserve old experiment-0 receipts and never substitute follow-up gates.
- **Where the knowledge lives.** `docs/RESEARCH_BRIEF.md` is the self-contained research brief and
  method authority: descriptive goals, partitions, sampling, adaptation and limitations.
  `docs/VSC.md` owns operations, storage, checkpoint formats and measured caps (do not raise one
  without a new probe). `docs/LITERATURE.md` records primary-source support and differences.
  Historical measurements and expensive dead ends stay in `docs/AGENTS_MEMORY.md`.
- **The experiments are `scripts/{data,train,eval,context}_pipeline.py` and `probe_row_cap.py`**, submitted
  through `scripts/slurm/`. Everything else is a utility under `src/utils/`, run with `python -m`.

## Shared VSC Library

`vsc-library/` is a pinned, entirely read-only infrastructure submodule. Read its
README, source snapshot, guides and Slurm examples; never edit, refresh, lint,
format or test inside it. There is no PROJECT_SPECIFIC exception. Keep this
project's VSC choices and operating instructions in `docs/VSC.md`, outside the
submodule. Shared corrections belong in the standalone VSC Library checkout.
Record the library pin when operational evidence depends on it.
