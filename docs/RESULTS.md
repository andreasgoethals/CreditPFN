# Results — what each run measured

The per-run measured record. Newest first. Numbers here are what the logs and `output/results/`
actually contained; interpretation and novelty framing live in `PAPER_ROADMAP.md`, the dead ends in
`AGENTS_MEMORY.md`, the capacity measurements in `METHOD.md`.

## How to write an entry

**Every run gets a `Configuration` table before its findings.** A run is only interpretable
against the settings and the corpus that produced it, and both change constantly: the sweep
is reshaped between runs, and the corpus will grow from 25 datasets toward thousands. "PD
mAUC 0.7620" with no configuration beside it is a number nobody can use or reproduce.

Copy this block. Everything in it is recorded automatically — the per-track manifest
(`output/manifests/<run>_<track>.csv`) carries one column per field, the resolved config is
dumped to `output/manifests/resolved/`, and the training log prints the corpus with row
counts — so filling it in is transcription, not archaeology.

| field | where it comes from |
|---|---|
| trials/track, bases, LRs, adaptation, qf | `config/train.yaml` → manifest columns |
| steps/trial, epochs, steps/epoch | manifest `total_optimizer_steps`, `epochs_run`, `steps_per_epoch` |
| corpus: n datasets + total rows, train and test | manifest `n_train_datasets`, `train_rows_total`, `train_dataset_ids` |
| `min_train_rows` | manifest column of the same name (swept since run-8) |
| row caps per base, eval caps | manifest `max_rows_per_epoch`, `config/eval.yaml` |
| L2-SP λ, warmup, LR floor | manifest `l2sp_lambda`, `warmup_fraction`, `min_lr_fraction` |
| code + literature version | manifest `git_commit`, `tfm_library_pin` |
| cost | GPU-hours and wall-clock from the logs |

Two standing comparability rules, both learned the hard way:

- **Never compare `neg_nll` across architectures.** v3, v2.6 and TabICLv2 use different output
  parameterisations (v3-vs-v2.6 differ by ~3.3 nats of criterion granularity; v2.6 values carry ~1 %
  clamped rows at +100 nats; TabICLv2's LGD head is a 999-quantile pinball, so `neg_nll` is NaN).
  Use **CRPS** for cross-family density comparison.
- **Never compare bases on `epochs`.** Steps/epoch depends on the per-base row cap, so equal epochs
  meant unequal steps until `train.target_total_steps` was introduced (08-08-2026).

---

## Run-7 — 10/11-08-2026 — training clean, eval incomplete, LGD undertrained

Training was the best it has been; the eval was not finished when the logs were collected, and
the numbers below come from **one of the two eval pools**, so coverage is partial by design.

### Configuration

| | |
|---|---|
| trials/track | 36 = 3 bases × 3 LRs × 2 adaptation × 2 query_fraction × 1 acc × 1 pass-mode |
| bases | TabPFN v3 default, TabPFN v2.6 default, TabICLv2 (`-v2-20260212`) |
| learning rates | 3e-7, 1e-6, 3e-6 (AdamW, warmup 0.10 → cosine to 5 % of peak) |
| adaptation | full-FT and adapter (LoRA for TabPFN, freeze-backbone for TabICLv2) |
| query_fraction | 0.20 and 0.40 · `accumulate_grad_batches` 1 · `epoch_pass_mode` full_pass |
| step budget | `target_total_steps` 9 100, `epochs` 100 — **realised**: PD 9 100/9 135, **LGD only 800–3 200** |
| L2-SP λ | 0.003 · weight_decay 0.0 · grad_clip 1.0 · BF16 autocast |
| row caps / step | v3 26 000 · v2.6 11 000 · TabICLv2 26 000 (PD, 2 members; LGD auto-scaled for TabPFN's 8) |
| corpus PD | 17 datasets: 12 train (2 158 000 rows), 5 test. `min_train_rows` not yet implemented |
| corpus LGD | 8 datasets: 6 train (78 500 rows, **4 of them under 3 000 rows**), 2 test |
| eval | 5-fold CV, inner val 0.20; caps v3/TabICLv2 1M rows, v2.6 50k; ensembles TabPFN 32, TabICLv2 8; baselines 50-trial Optuna |
| eval layout | 209 PD + 84 LGD array tasks (one per model × dataset), 2 pools by task stride |
| cost | PD 76.2 GPU-h, LGD 13.5 GPU-h; training wall-clock 5.1 h at 15–21 concurrent GPUs |

- **72/72 trials OK on both tracks.** 90 GPU-h drained in 5.1 h of wall-clock at 15–21 concurrent
  GPUs. 0 AMP skips, 0 data skips, no divergence, no tracebacks in 219 logs.
- **`target_total_steps` equalised PD but silently skipped LGD.** PD: tabicl 9 100, v3 9 100,
  v2.6 9 135 steps (45 epochs — correctly trimmed). LGD: tabicl **800**, v3 **1 600**, v2.6
  **3 200** — 3–11× under the 9 100 target, because trimming can only remove epochs. Fixed
  11-08-2026; see `AGENTS_MEMORY.md`.
- **Weight drift is real but small**, and scales with LR as expected: PD v3 ‖Δw‖ 0.0013 → 0.0054
  across 3e-7 → 3e-6; LGD v2.6 the largest at 0.0198.
- **PD, paired against each model's own untuned base on the same dataset — 17/39 wins:**

  | family | adaptation | mean Δ mAUC | best Δ |
  |---|---|---|---|
  | TabICLv2 | full-FT | **+0.0163** | **+0.0353** |
  | TabICLv2 | freeze-backbone | +0.0036 | +0.0295 |
  | TabPFN v3 | full-FT | +0.0010 | +0.0016 |
  | TabPFN v3 | LoRA | +0.0002 | +0.0007 |
  | TabPFN v2.6 | either | −0.0007 | +0.0005 |

  TabICLv2 starts from a much weaker base (myhom 0.5582) and continued pretraining moves it a long
  way (0.5935); TabPFN v3 starts strong and barely moves. **The gain is inversely related to how
  good the starting model already was.**
- **LGD: 0 wins out of 18 paired comparisons.** Every trained model is worse on RMSE than its own
  untuned base; v2.6 full-FT worst at −0.0138. Note this is at 800–3 200 steps, so it is evidence
  that the LGD recipe is wrong, not that LGD cannot be improved.
- **`qf=0.40` beats `qf=0.20` at the top of every PD dataset leaderboard.** More query rows per
  step means more of each batch contributes gradient, and it costs nothing extra.
- **Monitor vs real eval disagree in a specific way:** the 2 000-row monitor showed TabICLv2
  *losing* AUC (−0.005) while the real 5-fold eval shows it gaining the most. Never judge a
  large-context model on the monitor.
- Cost: PD 76.2 GPU-h (v3 2.6 h/trial, v2.6 1.9, tabicl 1.8), LGD 13.5 GPU-h.

## Run-6 — 07-08-2026 — first fully green run, first real eval

- **36/36 training trials + 84/84 eval cells OK, 0 failed folds.** The freeze-backbone fix held. The
  training array drained in **7.1 h** (run-5: 47 h) — the shorter walltime plus the smaller grid
  worked. The gate fired correctly and submitted eval.
- **Steps went 600 → 9 100 (v3 / TabICLv2) and 20 300 (v2.6),** so drift is finally real:
  v3 @1e-6 `l2sp` 5.3e-4 (was 2.7e-5), v2.6 @3e-5 `l2sp` **0.61**. The LR floor works (final LR = 5 %
  of peak).
- **PD (5 test sets, real 5-fold CV, higher mAUC better):**

  | model | mAUC |
  |---|---|
  | trained v3 1e-6 lora | **0.7620** |
  | trained v3 1e-5 full | 0.7576 |
  | trained v3 3e-5 full | 0.7552 |
  | trained tabicl 1e-6 full | 0.7552 |
  | untuned v2.6 | 0.7516 |
  | trained v2.6 | 0.7514 |
  | xgboost (tuned) | 0.7494 |
  | logreg (tuned) | 0.7018 |

  v2.6 continued pretraining ≈ no gain (0.7514 vs 0.7516 untuned).
- **LGD (2 test sets, lower RMSE better):** trained v3 1e-6 full **0.1335** ≈ tabicl 1e-6 full
  0.1338 ≈ untuned v2.6 0.1342 > xgboost 0.1413 ≫ linreg 0.1982. Higher LR clearly hurts
  (v3 3e-5 → 0.1423; v2.6 1e-6 full 0.1490).
- **Lower LR wins on both tracks** now that the step count is real. 1e-6 is best everywhere and 3e-5
  degrades — the next sweep should go **down** (3e-7…1e-6 at the full step budget), not up.
- **Calibration degrades under continued pretraining:** untuned v2.6 ECE 0.0159 → trained
  0.0169–0.0224; trained v3 0.0194–0.0213; xgboost best at 0.0137. But trained v3 has the best
  **Brier** (0.1526), so it is better probabilistically overall. This is what motivated the post-hoc
  Platt/isotonic columns.
- **TabICLv2 is competitive in the real eval** (PD 0.7552, LGD 0.1338). The 2 000-row monitor had it at
  0.694 and was badly underselling it — **never judge TabICLv2 on the monitor**; large context is its
  design point.
- Cost: 54.9 GPU-h for 18 PD trials (v2.6 4.5 h/trial, v3 2.8 h, TabICLv2 1.9 h).
- **Caveat that limits this run:** half the eval pool never logged, so trained-vs-untuned is not
  computable for v3 or TabICLv2 (see `AGENTS_MEMORY.md`, 08-08-2026).

## Run-5 — 05/06-08-2026 — first two-family run: trained, but undertrained

- **80/96 trials OK; the 16 failures were all TabICLv2 `_iclhead`** (see `AGENTS_MEMORY.md`,
  06-08-2026). **Eval never ran** — the gate expired — so every number below is a **2 000-row
  monitor** eval, not the real 5-fold CV.
- **The headline finding was that the models never moved.** From the final-epoch `l2sp`,
  ‖w−w₀‖² = l2sp / 0.0015:

  | lr | one_sample | full_pass |
  |---|---|---|
  | 3e-7 | 3.7e-6 (‖Δw‖ = 0.05) | 4.3e-5 |
  | 1e-6 | 2.7e-5 | 2.0e-4 |
  | 1e-5 | 7.7e-4 | 1.7e-3 |
  | 3e-5 | 3.0e-3 (‖Δw‖ = 1.4) | 2.8e-3 |

  At 3e-7 that is ~0.02 % of ‖w₀‖. Real-TabPFN pairs 3e-7 with **20 000 steps**; we ran 600
  (one_sample) / 4 550 (v3 full_pass) / 10 150 (v2.6 full_pass). **The PD null result was "we did not
  train", not "continued pretraining does not work"** — which is why run-6 raised the step budget.
- Monitor deltas (test): PD v3 mean −0.0008 / best +0.0001; PD v2.6 mean +0.0005 / best +0.0021;
  PD TabICLv2 mean −0.0047 (all negative). LGD RMSE worse for all three (TabICLv2 worst, +0.0075 mean,
  r² −0.06).
- TabICLv2's untuned baseline looked much weaker here (PD AUC 0.694 vs v3 0.728; LGD RMSE 0.187 vs
  0.132) — an artefact of the 2 000-row monitor, as run-6 showed.
- Cost: 43.7 GPU-h / 96 trials. PD v2.6 is the hog (19.5 h). The monitor is only 4–7 % of PD epoch
  time; **I/O was 20–33 %** at `dataloader_workers=0`.

## Run-4 — 11–13-07-2026 — the clean homogeneous 64-trial sweep

The reference sweep for cross-version comparison.

- **Pipeline fully green:** 64/64 trials genuinely trained (0 SKIPs), BF16, 0 amp/data skips, 0
  divergence; 63 checkpoints written straight to staging, 1 fallback (LGD a24, node-transient)
  auto-archived by the gate. Gate: pd 32/32, lgd 32/32 sentinels, drain 3 h 58 m; both eval pools
  covered all datasets.
- **PD: continued pretraining had ≈ zero effect on discrimination.** Untuned v3 mAUC 0.7622 (already
  beating tuned GBM at 0.749). Best trained delta +0.0004 (v3 3e-7); high LR harms (worst −0.0055 at
  v3 3e-5 full_pass); **LoRA was a no-op at every LR** (|Δ| ≤ 0.0007 on eval). A definitive negative
  at 12-dataset corpus scale — later reframed by run-5's drift measurement as *undertrained*.
- **LGD: the NLL-vs-RMSE trade-off replicates on clean checkpoints.** v3 full-FT improves `neg_nll`
  monotonically with LR/exposure (loss2 0.975 → 0.657 = −0.32 nats @3e-5 full_pass; lendingclub
  −0.21) while RMSE worsens (+0.009 / +0.003). v3 @1e-6 full_pass is a near-free density gain
  (−0.056 nats, +0.001 RMSE). **v2.6 degrades on both metrics at all LRs** — version matters.
  Untuned v3 still has the best RMSE (0.1399 / 0.1253).
- **The scientific result as of run-4:** credit-domain continued pretraining sharpens the LGD
  predictive **density** but not point accuracy, and does nothing for PD discrimination — consistent
  with Purucker and Tanna. Next levers: density metrics (CRPS/PIT), temporal splits, more datasets,
  seeds.
- Eval gaps: 8 PD cells walltime-killed (all v2.6 × algorithmwatch on A100, ~40 min/fold) → eval
  walltime 2 h → 5 h and the v2.6 eval cap 100k → 50k (its published envelope); 1 LGD cell (a48)
  produced no log at all (one-off).

## 09/10-07-2026 snapshot — provisional, superseded by run-4

From 183 logs: LGD 32/32 training complete, PD 27/32 (tasks 0–4, the low-LR v3 variants, ended
together without traceback — consistent with external termination). Eval: 102 PD pairs complete, 13
partial, no LGD eval. Untuned v3 averaged AUC 0.7622 over five datasets; LoRA mostly
prediction-identical; the evaluated full-FT / high-exposure trials generally reduced AUC and worsened
ECE. Logs also exposed 97 877 repeated sklearn `FutureWarning`s, 32 non-contiguous `searchsorted`
warnings and 302 FP16 AMP-skipped optimiser steps — fixed by an exact multiline filter, contiguous
targets, and BF16-auto with explicit skip counts.

## 10/11-07-2026 fresh-trial numbers — contaminated run, read with care

Only PD v3 a0–a4 retrained on the new BF16 code (the rest SKIPped on stale checkpoints — see
`AGENTS_MEMORY.md`, 11-07-2026). Those were healthy: 0 amp skips, no divergence, full_pass survived
50 epochs, GPU peak 125.5 GB (within the 130 GB member-aware target). Fixed-sample monitor: train
AUC +0.001…+0.0025 but **test** AUC −0.0004…−0.0017 → mild overfitting even at 3e-7/1e-6; LoRA
@3e-7 an exact no-op to 4 dp. Eval on mostly old checkpoints: no trained variant beat the untuned
base on PD AUC or LGD RMSE, and v3 3e-5 full-FT **collapsed** (AUC 0.50, ECE 0.42 on bank).

---

## State of the world

Updated **08-08-2026**.

- **Run-6 is the current reference** for behaviour and cost; **run-4** remains the clean homogeneous
  64-trial sweep for cross-version science. Staging has been writable since the user chmod'd it on
  11-07-2026, so the fallback path is a safety net rather than the norm.
- **The next sweep should go down in LR, not up** (3e-7…1e-6 at the full step budget), and should
  keep `target_total_steps` fixed across bases.
- **Best-epoch selection is deliberately not implemented** — there is no validation set yet. It
  becomes possible once the corpus is large enough to hold one out; until then every trial reports
  its final epoch.
- **Balance was ~9.7 M credits** (`sam-balance`, 08-07-2026). B200 credit weight is 437.5 per
  GPU-minute; run-4 spent ~4 h across up to 24 B200s plus two eval pools.
