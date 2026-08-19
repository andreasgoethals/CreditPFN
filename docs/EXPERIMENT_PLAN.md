# Experiment plan — where this project goes next

Written 19-08-2026, after run-8 produced the project's first complete evaluation and a null
result. This document is the argument for what to run next and why. The measured record of what
each run *did* is [`RESULTS.md`](RESULTS.md); the pipeline is [`METHOD.md`](METHOD.md).

---

## 1 · The finding that reframes everything: we only measured the left tail

The question is whether continued pretraining (CPT) on in-domain credit data improves a tabular
foundation model. Run-8 says no. But there are two ways to get a null, and they are not the same
paper:

* **A plateau at zero** — the model *can* be moved, we moved it, and performance did not improve.
* **No dose delivered** — the model barely changed, so of course nothing happened.

Run-8 is currently the second one, and the evidence is unambiguous.

### 1.1 The optimiser works. The models do learn the corpus.

Training loss, first epoch → last:

| base | PD | LGD |
|---|---|---|
| TabICLv2 | 1.51 → 0.43 (**−71 %**) | −6 % … −31 % |
| TabPFN v3 | −5 % … −8 % | 1.77 → −0.79 (**−145 %**) |
| TabPFN v2.6 | **−0.7 % … −1.9 %** | 4.02 → 0.33 (**−92 %**) |

So "the training is broken" is ruled out. Gradients flow, the loss falls, and in several cells it
falls dramatically. Note *what* is falling, though: a 71 % or 92 % drop in the first few dozen
epochs is the **output head and loss scale** re-fitting, not the representation learning
transferable structure. The two bases that start already matched to the task (v2.6 and v3 on PD,
at loss ≈ 0.47) move −1 % to −8 % and nothing else happens.

### 1.2 But the weights barely move, and the dose is monotone in the learning rate

Final drift, ‖w − w₀‖ / ‖w₀‖:

| base | lr 3e-7 | lr 1e-6 | ratio for a 3.33× LR step |
|---|---|---|---|
| TabICLv2 | 0.00287 | 0.00469 | 1.64× |
| TabPFN v2.6 | 0.00268 | 0.00685 | 2.56× |
| TabPFN v3 | 0.00239 | 0.00514 | 2.15× |

**The weights moved by a quarter to seven tenths of one percent.** Drift rises monotonically with
LR and shows no sign of saturating, i.e. every trial in run-8 sits in the near-initial linear
regime. Fitting `drift = a·lrᵇ` through the two measured points gives b ≈ 0.4–0.8 and:

| base | LR for 5 % drift | 10 % | 30 % |
|---|---|---|---|
| TabICLv2 | 3.3e-4 | 1.8e-3 | 2.7e-2 |
| TabPFN v2.6 | 1.3e-5 | 3.1e-5 | 1.3e-4 |
| TabPFN v3 | 3.6e-5 | 1.1e-4 | 6.0e-4 |

(Extrapolation from two points — indicative of the order of magnitude, not a prediction.)

### 1.3 Our learning-rate range is below everything in the literature

| source | LR | notes |
|---|---|---|
| Garg 2025, Real-TabPFN | 3e-7 | **never tuned**; + L2-SP λ = 0.003 |
| Kolberg 2026, TabPFN-Wide | 1e-5 | + weight decay 1e-4, **no L2-SP**, 10k steps |
| Tanna 2025/26, TabTune SFT | 1e-5 | + wd 1e-4, ≤ 10–25 epochs |
| Rubachev 2025 | **5e-6 … 5e-4** | the only paper that actually *tuned* the LR |
| **CreditPFN run-8** | **3e-7 … 1e-6** | — |

Our maximum, 1e-6, is **5× below the bottom of the only tuned search in the field**, and 33×
below the value two independent papers converged on. `tfm-library/SYNTHESIS.md` states the
position plainly: the correct CPT hyperparameters "remain genuinely unsettled", with the two
published recipes differing by a factor of 33 and neither tuning them.

We inherited Garg's 3e-7 as if it were a law. It is one untuned choice by one paper, and it was
made for a corpus of 71 large tables where conservatism against forgetting is the concern.

### 1.4 There is also no stopping criterion

Rubachev uses patience-16 early stopping; Tanna uses ≤ 10–25 epochs. We run a fixed step budget
with no stopping rule, which on LGD becomes **625–1 200 passes over six tables** — four times
Garg's exposure (~280 passes over 71 tables) to a twelfth of the data. At our LR that
overtraining does little; at a useful LR it would matter a great deal.

### 1.5 What this means for the paper

**The current result cannot support the claim we want to make.** "CPT does not help modern TFMs"
requires showing the dose–response curve: LR too low → nothing changes; LR useful → the model
changes measurably; LR too high → it degrades. If performance is flat across *that whole arc*,
the null is a real finding about the method. Right now we have only the leftmost point, three
times over.

This is good news for the project. It converts an under-powered negative into a well-defined
experiment with a genuinely publishable outcome either way.

---

## 2 · Dataset splits: stop depending on one draw

### 2.1 What we do today

* **PD** — unpinned in `train.yaml`, so `split_corpus` sorts the 17 ids, permutes with
  `np.random.default_rng(cfg.seed)` and takes 70 % / 30 %. **One draw at seed 42.**
* **LGD** — the 6 train and 2 test ids are hand-pinned in `train.yaml`.

Both are single configurations, and the hand-pinned one cannot even be resampled. Every number in
`RESULTS.md` is conditional on one arbitrary partition of a 17- and an 8-dataset corpus.

### 2.2 How much that costs us

The unit of inference is the **test dataset**, and the between-dataset standard deviation of the
effect is what sets the power:

| track | n_test now | between-dataset SD | smallest effect detectable at 80 % power |
|---|---|---|---|
| PD | 5 | 0.0100 | **0.0126 ROC-AUC** |
| LGD | 2 | 0.0010 | 0.0020 (SD from n = 2 — do not trust it) |

To detect a +0.005 effect on PD we would need ~32 test datasets. We have 17 in total.

### 2.3 The fix: dataset-level K-fold, so every dataset is tested

Resampling the split does not create datasets, but it does recover most of the lost power: with
K-fold over datasets, **every dataset appears in a test set**, so the effect is estimated on all
17 rather than on 5, and the split-choice variance becomes something we can measure instead of
something we hope is small.

Recommended design:

| track | folds | test per fold | train per fold | repeats | configurations |
|---|---|---|---|---|---|
| PD | 4 | 4–5 | **13** | 3 | **12** |
| LGD | 4 | 2 | 6 | 4 | **16** |

Two things improve at once: coverage (each dataset tested 3–4×) *and* corpus size (PD trains on
13 tables instead of 12). This is the 10–15 combinations asked for, but arranged as proper
cross-validation rather than 15 independent random draws — with random draws some datasets never
get tested and others get tested five times.

Analysis rule: **average within a hyperparameter cell across dataset configurations**, and report
the spread across configurations as its own quantity. That spread *is* the "how much does this
depend on the split" number, and it belongs in the paper.

### 2.4 Implementation

`corpus.split_seed`, separate from `cfg.seed`. Today one seed controls both the dataset partition
and weight initialisation, so changing the split also changes the init and the two effects are
confounded. Splitting them is a small change and a precondition for this design.

---

## 3 · Every remaining design decision: settled vs open

### 3.1 Settled — fix these, do not spend sweep budget on them

| decision | value | why it is settled |
|---|---|---|
| adaptation, TabPFN | **full fine-tune** | Rubachev 2025: full FT matches every PEFT variant on accuracy and converges fastest; Tanna finds LoRA *unstable* for TabPFN specifically. Our runs 4, 6 and 7 measured LoRA as a no-op. |
| adaptation, TabICLv2 | **keep both arms** | Tanna 2026: full SFT is near-catastrophic for TabICL (TabZilla 0.873 → 0.567) while TabPFN survives intact. This is a family difference, not a hyperparameter. |
| query_fraction | 0.40 | Garg's 60/40 context/query split; our run-7 A/B was confounded with the adaptation axis and showed nothing. |
| row caps per step | v3 26k · v2.6 11k · TabICLv2 26k | measured on B200, `METHOD.md` §3. |
| monitor cadence | every 5th epoch | run-7/8; cheap and sufficient. |
| epoch cap | 6 000 | so no arm hits the rail before the step target (fixed 13-08). |
| eval protocol | 5-fold CV, caps paired within family | verified comparable in the 17-08 audit. |

### 3.2 Open — these belong in the sweep

**A. Learning rate — now the primary axis.**
`[3e-7, 1e-6, 1e-5, 1e-4]`, a log grid spanning Garg's value to the top of Rubachev's tuned
range. This is the axis that decides whether our null means anything. Add 1e-3 as a deliberate
"break it" point if budget allows: a configuration that visibly destroys the model is *evidence*
that the middle of the range was a fair test.

**B. The anchor — L2-SP vs weight decay vs nothing.**
Garg uses L2-SP λ = 0.003 toward w₀; Kolberg uses plain weight decay 1e-4 toward the origin;
neither tuned it, and they disagree. Sweep `l2sp_lambda ∈ {0, 0.003, 0.03}` crossed with
`weight_decay ∈ {0, 1e-4}` — reduced to the three combinations that are actually distinct
recipes (Garg, Kolberg, unconstrained).

On **L1 instead of L2 toward w₀**: not worth a slot. L1 on `(w − w₀)` induces *sparsity in the
update* — it would keep most weights exactly at their initial value and move a few a long way.
There is no result in the TFM literature motivating that, no implementation to compare against,
and it answers a question nobody is asking. L2-SP is the anchor the field uses; the open question
is its *strength*, and λ = 0 vs 0.003 vs 0.03 covers that.

**C. Stopping rule.**
Replace the fixed budget with patience-based early stopping on the monitor split (Rubachev:
patience 16), and **log the step at which each trial stopped** — that number is itself a result
("the useful budget is N steps, and we gave it 20 000"). Keep a coarse budget arm
`target_total_steps ∈ {2 000, 20 000}` so "more training is worse" can be tested cleanly rather
than inferred.

**D. Corpus size filter.** `min_train_rows ∈ {0, 5000}` — keep. It is the only lever that moved
the result in run-8, and it is the one Garg's ablation predicts.

**E. Base generation — the new axis.** `TabPFN v1, v2, v2.6, v3` at the Phase-A optimum. See §5.

---

## 4 · Phasing, so this fits in the credit budget

The full cross-product of §3.2 across 12–16 dataset configurations is thousands of runs. It has
to be staged, and each stage answers one question.

| phase | question | design | scale |
|---|---|---|---|
| **A · dose–response** | Does a higher LR move the model, and does anything help? | LR (4) × anchor (3) × 3 bases, **one canonical split**, full-FT only | ~36 trials |
| **B · the main result** | Is the null robust to the dataset split? | the 2–3 most informative LRs × anchor best × 3 bases × **12–16 splits** | ~100–150 trials |
| **C · generation ladder** | Does the need for CPT vanish as bases improve? | 4 base generations at Phase-A optimum × 12 splits | ~50 trials |
| **D · corpus scaling** | Does more in-domain data change the answer? | corpus size {12, 25, 40} at fixed budget, once the new datasets land | ~30 trials |

Phase A is the gate. If drift at 1e-5/1e-4 is material and performance still does not improve,
the paper is written. If performance *does* improve at a higher LR, the finding inverts — and
that is a better paper still, because it would mean the field's canonical recipe is
mis-specified.

Do not start Phase B before Phase A has reported. Phase A is ~36 trials ≈ 1 GPU-day and decides
the shape of everything after it.

---

## 5 · The base-generation ladder

Run-8 already contains the seed of this:

| base | what any adaptation scheme does to it (PD, Δ ROC-AUC) |
|---|---|
| TabICLv2 | −0.021 … **+0.034** |
| TabPFN v2.6 | ±0.002 |
| TabPFN v3 | −0.014 … +0.002 |

The effect shrinks as the base improves. With four generations this becomes a curve, and the
claim becomes falsifiable and mechanistic rather than merely negative:

> **The value of domain-specific continued pretraining decays as synthetic priors improve, and at
> the current frontier it is indistinguishable from zero.**

This is directly supported by the literature: Qu 2026 (TabICLv2) and TabPFN-3 both report a
better prior erasing generic real-data gains, and Hoo 2024/2026 show the synthetic prior winning
zero-shot out-of-domain while real-data pretraining pays only in-domain. Our contribution is the
in-domain test, on a domain nobody has run it on, across generations.

Caveat to state in the paper: with 17 PD datasets this is a 4-point trend, and the confidence
interval on each point is wide. It needs Phase D to carry real weight.

---

## 6 · What the notebooks must show for this

* **Weight distance from initialisation** — already plotted (`plot_weight_drift`), but it needs to
  become a first-class result rather than a diagnostic: drift vs LR on log–log axes, per base,
  with the literature's LR values marked. That figure *is* §1.
* **Drift against effect** — the scatter that answers "did the models that moved most do best".
  If it is flat, that is the plateau; if it slopes, we under-trained.
* **Split-variance** — effect per hyperparameter cell with the spread across dataset
  configurations shown explicitly.
* **Stopping step** — where early stopping fired, against the budget given.

---

## 7 · Honest risks

* **Higher LR may simply destroy the models.** That is a *result* (it bounds the useful range),
  but it means the "CPT does not help" claim rests on the middle of the range being genuinely
  tested rather than on a single lucky point. Sweep densely enough to show the arc.
* **17 and 8 datasets is small for a scaling claim**, whatever we do with the splits. K-fold
  recovers power; it does not manufacture independence. The paper must say so.
* **Dataset-level CV inflates compute linearly.** The phasing in §4 is what keeps it affordable;
  skipping Phase A to "just run everything" is the failure mode.
* **The LGD track has no published recipe at all** — every reference paper is classification-only.
  That is our clearest novelty and our largest source of uncertainty at the same time.
