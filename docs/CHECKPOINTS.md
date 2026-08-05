# Checkpoints

Tabular-foundation-model weights used as starting points for continued
pretraining on credit-risk data, plus the trained checkpoints our
sweeps emit. Two families are swept: **TabPFN** (v2.6, v3) and
**TabICL v2** (added 2026-08-04). **Do not edit or commit new
checkpoints without updating this file.**

> **Where the weights actually live.** On VSC, both the **base**
> checkpoints and the **trained** outputs live in project *staging*,
> not in the git tree. Paths in this repo (e.g.
> `checkpoints/tabpfn-v3-classifier-v3_default.ckpt`) are resolved
> through `resolve_staging_path()` (`src/utils/paths.py`) to
> `/lustre1/project/stg_00211/CreditPFN/checkpoints/...` (override base
> with `$CREDITPFN_STAGING_ROOT` / `$TABPFN_STAGING_ROOT`). Both
> clusters (wICE for eval, Mindwell for training) can see this tier, so
> it is the hand-off point for trained weights. Off-VSC (laptop),
> `resolve_staging_path` falls back to the repo root, so the same
> relative paths resolve under `<repo>/checkpoints/`. The base weights
> are read once at job start and cached in RAM. See
> [VSC_GUIDE.md](VSC_GUIDE.md) for the storage-tier topology.

All facts below are sourced from:

- The upstream `tabpfn` package README (mirrored at
  `tfm-library/repositories/TabPFN .txt`, lines 649–751 and 2606–2623).
- Prior Labs' HuggingFace model cards (verbatim mirror at
  `tfm-library/repositories/Huggingface TabPFN.txt`).
- Hollmann et al. 2025 (*Nature*) and Grinsztajn et al. 2026
  (arXiv:2511.08667), Appendix C.

The inventory lists every base `.ckpt`, what training data Prior Labs
used to produce it, and a brief note on what role each plays in
**our** continued pretraining experiments. Trained outputs and their
on-disk format are documented further below.

From the TabPFN family only the **v2.6** and **v3** synthetic-only
bases are used (the older **v2.5** base has been dropped from the sweep
entirely). Alongside them the sweep includes the **TabICL v2** base of
the matching head — so both tracks run three bases.

TabICL sources:

- Upstream code + finetuning internals (mirrored at
  `tfm-library/repositories/TabICL.txt`); the pip package is pinned
  `tabicl[finetune]>=2.1.1,<3` because our shim imports its private
  `tabicl._finetune.data` helpers.
- Weights: HuggingFace `jingang/TabICL`.
- Qu et al., *TabICL: A Tabular Foundation Model for In-Context
  Learning on Large Data* (see `tfm-library/SUMMARIES.md`).

## All our bases are synthetic-only

Every checkpoint we sweep was trained from scratch on millions of
*synthetic* tabular datasets sampled from a structural-causal-model
prior — no real-world data has touched the weights. **TabPFN-v3 and
all v2.6 variants** ship synthetic-only, and neither has a released
"real-finetuned" variant. That is exactly the clean starting point the
Real-TabPFN recipe (Garg et al. 2025) assumes: begin from the synthetic
prior, then continue-pretrain on real data — in our case the curated
credit-risk corpus — so any downstream gain is attributable purely to
that corpus.

## Inventory (verified against upstream)

| File | Size | Origin | Training data | Role in this project |
|---|---|---|---|---|
| `tabpfn-v3-classifier-v3_default.ckpt`         | 213 MB | HF `Prior-Labs/tabpfn_3` | **Synthetic-only.** The v3 HF card states *"TabPFN-3 is trained purely on synthetic tabular tasks."* New multi-stage transformer architecture (24 main layers); ≤1 M samples × ≤2 000 features (vs. 50 k for v2.6). | **Default sweep base.** Latest released checkpoint with the strongest published benchmarks (SOTA on TabArena, TALENT). |
| `tabpfn-v3-regressor-v3_default.ckpt`          | 233 MB | HF `Prior-Labs/tabpfn_3` | **Synthetic-only.** Same v3 card statement applies; no real-finetuned v3 regressor yet. | **Default sweep base** for LGD. |
| `tabpfn-v2.6-classifier-v2.6_default.ckpt`     | 43 MB  | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** — the v2.6 card states *"TabPFN-2.6 is trained purely on synthetic tabular tasks"*; no real-finetuned v2.6 variant has been released. | Sweep base: the cleanest v2.6 base available. |
| `tabpfn-v2.6-regressor-v2.6_default.ckpt`      | 51 MB  | HF `Prior-Labs/tabpfn_2_6` | **Synthetic-only** (same card statement). No real-finetuned v2.6 regressor yet. | Sweep base: cleanest v2.6 regressor base. |
| `tabicl-classifier-v2-20260212.ckpt`           | 110 MB | HF `jingang/TabICL` | **Synthetic-only.** TabICL is pretrained on synthetic tabular tasks; 3-stage architecture (column embedder → row interactor → ICL predictor), ~27 M params. Classifier head emits 10 logit columns. | **Second-family sweep base (PD).** Tests whether the CPT result generalises beyond one architecture/prior. |
| `tabicl-regressor-v2-20260212.ckpt`            | 114 MB | HF `jingang/TabICL` | **Synthetic-only**, same architecture; regression head emits 999 quantiles on context-z-normalised targets (no bar distribution). | **Second-family sweep base (LGD).** |

### Getting the TabICL weights onto VSC (one-time, from a LOGIN node)

Compute nodes have **no outbound network**, and our loaders pass
`allow_auto_download=False` so a missing file fails loudly instead of
stalling a GPU job. Stage both checkpoints once:

```bash
python -c "
from huggingface_hub import hf_hub_download
import shutil, os
dest = os.environ.get('CREDITPFN_STAGING_ROOT', '/lustre1/project/stg_00211/CreditPFN') + '/checkpoints'
os.makedirs(dest, exist_ok=True)
for f in ('tabicl-classifier-v2-20260212.ckpt', 'tabicl-regressor-v2-20260212.ckpt'):
    p = hf_hub_download('jingang/TabICL', f)
    shutil.copy2(p, dest + '/' + f)
    print('staged', f)
"
```

## How to read the naming conventions

For **v3** the naming is: only `_default` (synthetic-only). No
specialist or real-finetuned variants have been released yet.

For **v2.6** the naming is: `_default` = synthetic-only (no
real-finetuned variant published yet).

Both conventions are confirmed verbatim by the HuggingFace cards
mirrored at `tfm-library/repositories/Huggingface TabPFN.txt`.

## What we sweep over

The training config (`config/train.yaml::tunable`) treats the base
checkpoint as a tuneable knob and sweeps over the released
synthetic-only bases for v2.6 and v3:

| Track           | Sweep includes (default)                        | What each tells us                                                        |
|-----------------|-------------------------------------------------|---------------------------------------------------------------------------|
| PD (classifier) | `v3_default` · `v2.6_default` · `tabicl-v2`     | v3 synthetic-only · v2.6 synthetic-only · second family (different prior + architecture) |
| LGD (regressor) | `v3_default` · `v2.6_default` · `tabicl-v2`     | same three, regression heads                                              |

The total grid per track is then `3 bases × 4 LRs × 2 adapt-modes ×
1 qf × 1 acc × 2 epoch-pass-modes = 48 trials`. The current LR grid is
`[3e-7, 1e-6, 1e-5, 3e-5]` — the bottom rung `3e-7` matches
Real-TabPFN, `1e-5` is TabICL's own finetuning default, and `3e-5`
approaches Rubachev's separate single-dataset finetuning median
(~3.9e-5); `1e-4` is excluded because it diverged on the no-LoRA +
`qf=0.20` setting (revisit now that `weight_decay=0.0`). See the
hyperparameter-rationale table in the README for the full literature
comparison.

**The adapt-mode axis is family-specific.** `use_lora=true` means LoRA
for TabPFN, and **freeze-backbone** (train the ICL module only) for
TabICL — that family's own pretraining stage-3 regime. The reason is
empirical: full SFT collapsed TabICL in two independent reports
(TabZilla accuracy 0.873 → 0.567 in Tanna 2026; "failed to train
TabICL" in Kolberg 2026), so the freeze-backbone arm is TabICL's
safe-adaptation arm rather than a parameter-count experiment. Its
checkpoints are tagged `_iclhead` instead of `_lora`.

Every base in this sweep follows the methodologically clean
ablation recipe of Real-TabPFN (Garg et al. 2025): start from
the synthetic-only checkpoint, continue-pretrain on real data —
in our case our curated credit-risk corpus. Any downstream gain on
credit-risk benchmarks is attributable purely to that corpus.

The eval pipeline (`scripts/eval_pipeline.py`) scores all of these
side-by-side against XGBoost / CatBoost plus a linear baseline
(LogReg for PD, LinReg for LGD) and the *untuned* versions of each
base, so the question of "which base wins" gets answered empirically
on the held-out test split. Eval reads the trained checkpoints from
staging and only scores those present on disk at submit time — see
[the trained-checkpoints section](#trained-checkpoints-our-outputs)
below.

## Architecture differences across versions

|                                       | v2.6                          | v3                                       |
|---------------------------------------|-------------------------------|------------------------------------------|
| Layers                                | 24 (fixed)                    | 24 main layers (multi-stage transformer) |
| Attention pattern                     | TabPFNv2-style alternating    | Multi-stage transformer-based            |
| Sample limit (intended)               | ≤ 50 000                      | ≤ 1 000 000                              |
| Feature limit (intended)              | ≤ 2 000                       | ≤ 2 000                                  |
| Real-finetuned variant published?     | No (only synthetic `_default`)| No (only synthetic `_default`)           |
| Model technical report                | Grinsztajn et al. 2026 (arXiv:2511.08667, same architecture family) | Grinsztajn et al. 2026, *TabPFN-3 Technical Report* |
| Approximate checkpoint size           | ~43–51 MB                     | ~213–233 MB                              |
| License                               | `tabpfn-2.6-license-v1.0`     | `tabpfn-3-license-v1.0`                  |

## Trained checkpoints (our outputs)

Each sweep trial writes a finetuned checkpoint to:

```
resolve_staging_path(checkpoint.trained_dir) / <track> / <descriptive_name>.ckpt
```

With the default `checkpoint.trained_dir: "checkpoints/trained"` in
[`../config/train.yaml`](../config/train.yaml), that resolves on VSC to
`/lustre1/project/stg_00211/CreditPFN/checkpoints/trained/<pd|lgd>/…`.
(Training actually resolves via `resolve_writable_staging_path`, which
probes staging writability first: if the compute node can't write staging
— the 2026-07-03 Mindwell failure mode — checkpoints are saved under
`$VSC_DATA/CreditPFN/checkpoints/trained/` instead, and the eval gate
archives them into staging afterwards. Either way the durable copy ends
up in project storage; see docs/VSC_GUIDE.md §0.2.)
Alongside each `.ckpt` we write a `<name>.ckpt.provenance.json`
sidecar (full training-time hyperparameters, the train/test dataset
lists, walltime, GPU, library versions) so a checkpoint can be
inspected without `torch.load`. The training manifest
`output/training/manifests/<run_name>_<track>.csv` records one row per
trial with a `status ∈ {OK, FAIL, SKIP, DIVERGED}`; the eval pipeline
rosters only `OK`/`SKIP` rows whose `.ckpt` exists on disk (it
excludes `FAIL` and `DIVERGED`).

### Descriptive filename schema

The basename encodes the tunable hyperparameters
(`descriptive_name()` in `src/train/loop.py`):

```
<run_name>_<track>_<base-stem>_lr<lr>_seed<seed>[_qf<qf>][_acc<K>][_fullpass][_lora].ckpt
```

- `<base-stem>` — the base checkpoint's filename stem (e.g.
  `tabpfn-v3-classifier-v3_default`).
- `lr<lr>` — learning rate in `%.0e` form with the `+` stripped (e.g.
  `lr1e-05`).
- `qf<qf>` — query fraction × 100, zero-padded (e.g. `qf20`); omitted
  when `query_fraction` is `None`.
- `acc<K>` — `accumulate_grad_batches`; omitted when `None`.
- `_fullpass` — present only for `epoch_pass_mode == "full_pass"`; the
  default `one_sample` adds no tag.
- `_lora` / `_iclhead` — present only when `use_lora` is true;
  `_iclhead` for TabICL bases (freeze-backbone), `_lora` for TabPFN.

Optional segments are dropped when their value is the default/`None`,
so the default one-step-per-dataset, full-FT sweep produces the same
short names as the original (pre-2026-06-01) runs.

### On-disk save format (Prior Labs 4-key dict)

`save_finetuned()` (`src/train/model.py`) persists a single
`torch.save` dict with the **four mandatory keys** Prior Labs'
`load_model` expects:

| Key | Contents |
|---|---|
| `state_dict` | the model weights (LoRA adapters merged back into the base via `merge_and_unload()`, so a LoRA trial is indistinguishable from a full-FT save and needs no PEFT at load time) |
| `config` | `asdict()` of the pydantic `ArchitectureConfig` |
| `architecture_name` | one of `tabpfn_v2`, `tabpfn_v2_5`, `tabpfn_v2_6`, `tabpfn_v3` — tells the loader which architecture class to instantiate |
| `inference_config` | `asdict()` of the `InferenceConfig` (pydantic `extra="forbid"`) |

The file round-trips through `TabPFNClassifier(model_path=…)` /
`TabPFNRegressor(model_path=…)`. We also add a `provenance` key plus
the JSON sidecar described above.

> **Previously-fixed bug (do not regress).** An earlier
> `save_finetuned` wrote only `{state_dict, config}`. With
> `architecture_name`/`inference_config` missing, `load_model` falls
> back to the v2 architecture and v3/v2.6 weights fail to load with
> *"Missing key(s) in state_dict"* — which broke **every** eval. All
> four keys are now mandatory. `architecture_name` is resolved via the
> upstream `_resolve_architecture_name`; if that private symbol is
> unavailable it falls back to a config-class-name match and **raises**
> on an unknown class rather than guessing `"base"` (which would
> silently mislabel a v3 checkpoint as v2.5 and make it unloadable).

### v2.6 vs v3 criterion handling (regressor / LGD)

The two architectures differ in how the bar-distribution criterion is
restored at load time, and our save format is compatible with both:

- **v3** — `model.forward` takes `test_targets_MB`, so the loader
  **strips** the `criterion.*` keys from the state-dict and rebuilds
  the `FullSupportBarDistribution` from the model's own
  `regression_borders` buffer.
- **v2.6** — has no `test_targets_MB`, so the loader **requires** the
  `criterion.*` keys to be present in the state-dict.

`save_finetuned` always merges the regressor's `criterion.*`
parameters into the saved state-dict. Those keys are *required* by
v2.6 and *harmlessly stripped* by v3, so the same write path
round-trips correctly for both. (Classifiers carry no bar-distribution
criterion, so this only concerns the LGD regressor checkpoints.)

### TabICL save format (upstream 2-key dict)

TabICL checkpoints use a different, simpler schema, written by
`save_finetuned_tabicl()` (`src/train/tabicl_model.py`):

| Key | Contents |
|---|---|
| `config` | the `TabICL(**kwargs)` init dict, with `recompute` reset to `False` so inference doesn't pay for gradient checkpointing |
| `state_dict` | CPU model weights (no adapter merging exists for this family — freeze-backbone trials simply have fewer changed tensors) |

Plus our `provenance` key and the same `.provenance.json` sidecar. The
file round-trips through `TabICLClassifier(model_path=…)` /
`TabICLRegressor(model_path=…)`; upstream's loader reads only the two
keys above with `weights_only=True` and ignores extras, so provenance
must stay JSON-safe primitives.

Two family differences worth remembering when reading results:

- **No exact predictive density.** The regressor emits 999 quantiles,
  not a bar distribution, so `neg_nll` is `NaN` for TabICL rows. Never
  compare density metrics across families; the planned cross-family
  density metric is CRPS (computable from the quantiles).
- **`recompute=True` during training.** Gradient checkpointing is the
  main VRAM lever. It is also what upstream does: TabICLv2's own stage-3
  pretraining enables it above 20K samples.

### TabICL context sizes — use the paper, not the library defaults

A correction worth recording, because the first values were wrong:

| | Value | Where it comes from |
|---|---|---|
| Eval fold cap | **1 000 000** (same as v3) | Qu et al. 2026: 1M samples × 500 features in ~450 s under 50 GB GPU + 24 GB CPU (hierarchical CPU/disk offloading); QASSMax scalable softmax keeps attention sharp at long context, so million-scale tables are handled "natively, without retrieval and distillation" |
| Training rows/step | **26 000** (same as v3) | Inside TabICLv2's own stage-3 pretraining range (400–60 000 samples, log-uniform); equal to v3 so a cross-family difference is not confounded with context size |

The earlier values (10 000 train / 50 000 eval) came from
`max_data_size=10_000`, a **default argument of their convenience
finetuning wrapper**, and from mirroring v2.6's cap. Neither is a
capability limit — million-scale in-context inference is TabICLv2's
headline result, and a 50 000-row eval cap would have handicapped it 20×
against v3 while discarding the model's main advantage.

Both numbers are still **unmeasured on our hardware**. Run
`scripts/probe_row_cap.py` (it has a TabICL branch, grid 10k–60k) before
committing a full sweep to them, and watch eval walltime on the first
run — a large-context model overruns the clock rather than OOM-ing.

## Licence

All weights are released under Prior Labs' research-only licences
(`tabpfn-2.6-license-v1.0`, `tabpfn-3-license-v1.0`). Testing,
evaluation, and internal benchmarking are explicitly allowed;
commercial use, client deliverables, or commercial decision-making
based on the model's outputs are not. Full text in the licence
files inside each HF repo.
