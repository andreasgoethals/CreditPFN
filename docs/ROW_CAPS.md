# Context-size caps (rows per step, rows per eval fold)

Single authority on **how many rows each model sees at a time**, for both
families and both stages. The config files carry the numbers; this file carries
the reasoning, the measurements, and the traps.

Settings that point here:

| Setting | File | Stage |
|---|---|---|
| `finetuning.max_rows_per_epoch` | `config/data.yaml` | training |
| `finetuning.max_cells_per_epoch` | `config/data.yaml` | training (off) |
| `train.n_estimators_finetune{,_tabicl}` | `config/train.yaml` | training |
| `max_rows_per_model` | `config/eval.yaml` | evaluation |

Re-run `scripts/probe_row_cap.py` (via `scripts/slurm/probe_row_cap.slurm` —
**never bare on a login node**) whenever the GPU, `sanitize.max_columns`, or an
ensemble size changes. Nothing here is valid for a different configuration.

---

## 1. Training: rows per step

A training step forwards **all `n_estimators_finetune` ensemble members** and
holds every member's graph for one backward pass. So:

```
peak memory  ≈  n_estimators  ×  rows  ×  (per-member cost per row)
```

The values in `config/data.yaml` are the **PD (2-member) caps**. `loop.py`
scales them down by `2/n_estimators` for LGD's 8 members, so both tracks hold
roughly the same GPU memory ("member-aware row-cap scaling" in
`train_one_config`). That scaling is **TabPFN-only** — it was calibrated on
TabPFN's memory slope, and TabICL is pinned to 2 members on both tracks anyway.

### Measured — B200 (183 GiB), 2026-08-05, job 11509346

64 features, `query_fraction=0.20`, real forward+backward. TabPFN probed at
**1 member**; TabICL at **2 members with `recompute=True`** (its actual training
configuration), so compare the per-member column, not the raw peaks.

| Base | Measurements | Slope /1k rows | Per member | First failure |
|---|---|---|---|---|
| TabPFN v3 | 20k → 50.3 GB · 50k → 124.9 GB | 2.49 GB | 2.49 GB | OOM at 100k |
| TabPFN v2.6 | 9k → 49.2 GB · 20k → 108.8 GB · 30k → 169.3 GB | 5.72 GB | 5.72 GB | OOM at 50k |
| TabICL v2 | 10k → 10.5 GB · 26k → 26.8 GB | 1.02 GB | **0.51 GB** | cuDNN at 40k |

Memory is ~linear in rows; the intercept is negligible (weights are tiny next
to activations). Step times are 0.5–2.5 s throughout — **except** the first
timed step of a job, which includes CUDA warm-up (v3 read 7.58 s at 20k and
2.53 s at 50k; don't misread that as superlinear).

The 2026-07-08 probe measured v3 and v2.6 within ~2 % of the above. An even
earlier "0.93 GB per 1k rows" figure was a **bad measurement** — it timed the
lightweight monitor eval, not a training step. Never trust the monitor's
`gpu_peak_alloc` as the training peak.

### Derived caps

Target ~130 GB peak, leaving ~53 GiB for optimizer state, fragmentation and the
rolling eval snapshot.

| Base | PD (2 members) | LGD (8 members, auto-scaled) | Predicted peak |
|---|---|---|---|
| v3 | **26 000** | 6 500 | ~129 GB |
| v2.6 | **11 000** | 2 750 | ~126 GB |
| TabICL | **26 000** | 26 000 (2 members, not scaled) | ~27 GB |

PD's large datasets (hackerearth 532k, home_credit 307k, vehicle_loan 233k) bind
on the cap, so 26k buys real context — the lever Real-TabPFN attributes gains
to. LGD datasets are all ≤16k rows, so the LGD caps truncate only lgd_freddie.

Lookup key is the leading `v<MAJOR>[.<MINOR>]` of the base filename, or
`tabicl` for that family; `default` is the fallback.

### Why TabICL is set to 26 000

1. **Parity.** If TabICL trained on 10k rows/step while v3 trained on 26k,
   every TabICL-vs-v3 difference would confound **architecture** with **context
   size**, making the cross-family comparison — the whole reason the family was
   added — uninterpretable.
2. **It matches TabICL's own pretraining.** Qu et al. 2026 §B.1: stage 1 = 1 024
   samples, stage 2 = 400–10 240, **stage 3 = 400–60 000** (log-uniform), with
   gradient checkpointing above 20k "to avoid the out-of-memory error" — exactly
   the `recompute=True` we force. The paper credits large-sample exposure for
   large-data generalisation (>10k-row rank 5.50 → 4.71 from stage 2 to 3).

An earlier value of 10 000 came from `max_data_size=10_000`, a **default
argument of tabicl's convenience finetuning wrapper** — a library default, not a
capability limit, and far below their own pretraining regime.

### TabICL's ceiling is a cuDNN kernel limit, not memory

At 40 000 and 60 000 rows the step dies with:

```
RuntimeError: Expected mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good()
              to be true, but got false
```

That is cuDNN's **fused multi-head-attention graph** failing on a long attention
sequence — **not** an OOM. There were ~140 GB still free. At 26k, TabICL uses
26.8 of 183 GB (15 %), and linear extrapolation says memory alone would allow
~180k rows.

So 26 000 happens to be both the parity value and safely under a hard ceiling
somewhere in 26k–40k. **Do not raise it** without first moving the attention
backend off cuDNN (e.g. `torch.backends.cuda.enable_cudnn_sdp(False)`) *and*
re-probing. Raising it blind produces that opaque error mid-sweep.

### Known confound: LGD context is not symmetric

TabPFN uses 8 members on LGD, so its caps auto-scale to v3 6 500 / v2.6 2 750
rows per step, while TabICL keeps 2 members and therefore 26 000. Because every
LGD dataset is ≤16k rows, **TabICL sees complete tables while v3 sees 6 500-row
samples.**

This is deliberate: each family runs at its own official member count (the fair
"as its authors intended" protocol), and changing TabPFN's LGD member count
would break comparability with the run-4 results. But it means the **LGD
cross-family comparison is entangled with context size and must be reported as
such**, not presented as a pure architecture comparison.

### The cell budget (`max_cells_per_epoch`, currently off)

When non-null for a base, per-step rows become
`min(max_rows_per_epoch, max_cells // n_features)` — narrow datasets get more
rows, wide ones fewer, at roughly constant cell count.

This fits **v3 only**: TabPFN-3's capacity is a cell-budget frontier (its report
§2.4 treats 1M rows × 200 features as equivalent to 100k × 2000), and its
3-stage design decouples the ICL stage from feature count while row-chunking
activation memory. It is **wrong for v2.6**, whose dual attention costs
`O(r²·c + r·c²)` — quadratic in rows — so v2.6 stays on a pure row cap.

To enable for v3: set a cell budget *and* raise `max_rows_per_epoch.v3` to the
row ceiling narrow datasets may reach (e.g. 8 000 000 cells, 100 000 rows), then
validate against OOM before a full sweep.

---

## 2. Evaluation: rows per fold

`max_rows_per_model` caps the **training partition of each CV fold** only. The
held-out test partition is never capped — the model predicts on every test row
in one call.

| Base | Cap | Why |
|---|---|---|
| v3 | 1 000 000 | TabPFN-3's published envelope (report §2.4) |
| v2.6 | 50 000 | Its published design envelope ("up to 50,000 data points"). Reduced from 100k on 2026-07-13: dual attention is O(rows²), and one v2.6 × algorithmwatch fold took ~40 min on an A100, blowing 8 cells past the old 2 h walltime. 50k stays in-envelope and cuts the quadratic term 4×. |
| TabICL | 1 000 000 | Million-scale in-context inference is TabICLv2's headline capability: Qu et al. report 1M samples × 500 features in ~450 s under 50 GB GPU + 24 GB CPU via hierarchical CPU/disk offloading, and QASSMax (scalable softmax, logits × `s·log n`) exists to keep attention sharp at long context. |

Our corpus tops out at 532k rows and ≤64 features, so 1M means "no cap in
practice" for both TabPFN v3 and TabICL — which is also what keeps their
trained-vs-untuned comparisons paired.

The cap resolves from the **base checkpoint** for both the trained and untuned
handle, so a family's two arms always see the same fold size. (A 2026-08-04 bug
had the untuned branch miss its key and silently fall through to `default`,
scoring untuned-v3 on 50k-row folds while trained-v3 got 1M.)

**Untested at scale by us:** TabICL's *training* path fails above ~26k rows via
the cuDNN issue above. Inference is a different code path — upstream demonstrates
1M and their wrapper chunks and offloads — so the cap is left at 1M. If TabICL
eval cells fail with that same `mha_graph` message on the large PD datasets, the
fix is to disable the cuDNN SDPA backend, **not** to lower the cap. Such
failures surface as FAIL rows carrying the error text, not silently.

Classical baselines (XGBoost / CatBoost / LogReg / LinReg) are never capped —
they see the full training fold. Their HPO uses a separate
`hpo.<model>.max_rows` subsample.
