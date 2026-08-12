# Code that looks wrong but is deliberate

Every entry here is something a reader — human or agent — would reasonably flag as a bug, redundancy
or leftover, and which breaks something real if "cleaned up". Each one also carries a `why` comment
at its site; this file is the index, so a reviewer can check before editing rather than after.

Dead ends live in `AGENTS_MEMORY.md`; what changed and when in `CHANGELOG.md`.

## Adaptation modes and regularisation

- **`use_lora=True` means *freeze-backbone* for TabICL, LoRA for TabPFN.** One grid axis, two family
  meanings; the run tags are `_iclhead` vs `_lora`. Do not unify them — the literature says full SFT
  breaks TabICL, so this is that family's safe-adaptation arm. `descriptive_name()` derives the tag
  from the base filename, so every call site stays consistent without call-site changes.
- **L2-SP applies to TabICL even in freeze-backbone mode**
  (`l2sp_applicable = (family == "tabicl") or (not use_lora)`) because the trainable ICL head still
  drifts from its pretrained values. For TabPFN + LoRA the base weights are frozen and the adapters
  have no pretrained anchor, so L2-SP is genuinely inert there — hence the asymmetry.
- **`model.col_embedder.eval()` is re-applied after every `model.train()`** in the epoch loop.
  TabICL routes its forward pass on module training flags, so `train()` would silently restore the
  backbone's train-time behaviour. Looks redundant; is not. (Distinct from using `.eval()` *to
  freeze*, which is the 06-08-2026 bug in `AGENTS_MEMORY.md` — freezing is `requires_grad=False`.)
- **TabICL ships one non-trainable `Parameter`** (`row_interactor.tf_row.rope.freqs`, RoPE
  constants), so "every parameter requires grad" is a false assertion even in full-FT mode.

## Memory, capacity and context size

- **Member-aware row scaling** in `train_one_config`: the `max_rows_per_epoch` values in
  `config/data.yaml` are the **PD 2-member** caps, and the code divides by `n_estimators / 2` for
  LGD. Removing this apparently redundant scaling re-introduces the LGD OOM.
- **The LGD context asymmetry is a known confound, not a bug.** On LGD, TabPFN uses 8 ensemble
  members so its caps auto-scale to v3 6 500 / v2.6 2 750, while TabICL keeps 2 members and 26 000.
  Since every LGD dataset is ≤16k rows, TabICL sees full tables and v3 sees a 6.5k sample. Kept
  deliberately — each family runs at its own official member count, and changing TabPFN's would
  break run-4 comparability — but the LGD cross-family comparison is confounded with context size
  and must be stated as such in the paper.

## Paths, saving, eval

- **`resolve_writable_staging_path` and `resolve_staging_path` are distinct on purpose** (probe +
  fallback vs plain resolution). Trained checkpoints use the former, because staging has been
  unwritable from compute nodes before.
- **`neg_nll` is clamped to ±100 nats** (`tabpfn_models.py`) to guard the v2.6 regressor density
  underflow that produced `-inf` and poisoned every aggregate that touched it.
- **`epoch_eval_every=5`:** the monitor eval runs on every 5th epoch, and the divergence detector's
  metric window uses only *monitored* epochs (`monitored_metrics`) — otherwise the NaNs from skipped
  epochs would look like a collapse.

## Environment and docs

- **"Genius login node" in the docs and scripts is correct.** Neither wICE nor Mindwell has its own
  login node; you always SSH to Genius and submit cross-cluster. A past audit flagged this as a bug;
  it is not.
- **Citations into `tfm-library/repositories/*.txt` are file-level, with no line numbers,** on
  purpose: those dumps are periodically refreshed and line numbers drift by thousands. Cite symbol
  names.
