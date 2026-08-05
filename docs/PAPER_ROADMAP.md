# From this repo to a paper — status and roadmap

_Written 2026-08-04. Plain-language companion to the README: where the project
stands, what the experiments have actually shown, whether anyone has already
published this, and what is still missing before writing._

---

## 1. The short version

1. **The pipeline works.** The July 11–13 run ("run-4") trained all 64
   configurations cleanly and evaluated almost all of them. Nine evaluation
   cells are still missing and can be recovered with one command.
2. **The PD result is a clean negative.** Continued pretraining on 12 real
   credit datasets does **not** improve default-risk discrimination. The
   untuned TabPFN-v3 base already beats tuned XGBoost/CatBoost, and no
   learning rate, adapter setting, or exposure level moved held-out AUC by
   more than ±0.0006.
3. **The LGD result is the interesting one.** Continued pretraining
   consistently **sharpens the predictive density** for loss-given-default
   (up to −0.32 nats of NLL) while making the **point estimate slightly
   worse** (RMSE +0.003 to +0.009). That trade-off replicated across two
   independent runs, and it is a real finding, not noise.
4. **Nobody has published this.** No paper does domain-specific continued
   pretraining of a tabular foundation model on real domain data — in credit
   or anywhere else — and none does it for regression at all. The nearest
   neighbours are listed in §5; each differs on an axis that matters.
5. **What is missing is mostly evidence, not ideas.** Two model families
   (now implemented), temporal splits, several seeds, a contamination audit,
   and a proper density metric. §6 is the ordered list.

---

## 2. What changed in this session

**Added TabICL v2 as a second continued-pretraining family, for both tracks.**
Until now every result came from one architecture and one synthetic prior, so
the obvious reviewer question — *"is your PD null result just a TabPFN
quirk?"* — had no answer. It does now: the sweep runs TabPFN v3, TabPFN v2.6
and TabICL v2 side by side, 48 configurations per track.

Design decisions worth knowing:

- **TabICL is trained with its own objectives**, copied from its own
  finetuning code: cross-entropy over its ten logit columns for PD, and mean
  pinball loss over its 999-quantile head for LGD. Each family is trained
  with the loss it was pretrained on — that is what makes it *continued*
  pretraining rather than a retrofit.
- **The "LoRA on/off" grid axis means something different for TabICL.** Two
  independent papers report that aggressive full finetuning breaks TabICL
  (accuracy 0.873 → 0.567 in Tanna et al. 2026; "failed to train TabICL" in
  Kolberg et al. 2026), while TabICL's own pretraining freezes everything
  except its in-context-learning module in its final stage. So for TabICL
  that axis freezes the backbone and trains the ICL head only. Those
  checkpoints are tagged `_iclhead` rather than `_lora`, and the provenance
  file records which adaptation actually ran.
- **One number will never be comparable across families.** TabICL's regressor
  emits quantiles, not a probability density, so its `neg_nll` column is
  empty by construction. The cross-family density metric has to be CRPS
  (computable from those quantiles) — that is item 5 in §6.

**Six bugs found and fixed along the way.** Two of them would have quietly
biased results:

| Bug | Why it mattered |
|---|---|
| Untuned models silently used the wrong evaluation row cap | Untuned-v3 was scored on 50 000-row folds while trained-v3 got 1 000 000 — so the headline trained-vs-untuned comparison was not apples-to-apples on large datasets |
| `__fullpass` was folded into the base tag when reading results | The notebooks averaged `one_sample` and `full_pass` runs of the same configuration into a single point, hiding the exposure effect |
| Evaluation SLURM arrays capped at 200 tasks | The bigger grid needs ~275 for PD; the tail would have been dropped without any error |
| The trial-count call was unchecked in the launcher | A failure would have produced `--array=0--1` and a cryptic sbatch error |
| ±inf reached the model; TabICL also rejects all-NaN columns | Both are reachable from real credit features (zero-denominator ratios, near-empty columns in one fold) |
| Provenance did not record the per-step ensemble size | A checkpoint could not be reproduced from its own metadata |

**Verification.** 275 tests pass, including a genuine end-to-end
`train_one_config` run on a miniature TabICL checkpoint — no mocks — for PD
full-finetuning, PD freeze-backbone, and LGD. That exercises the loader,
batch construction, both losses, the L2-SP anchor, the per-epoch monitor, the
save format and the reload through TabICL's own inference wrappers.

---

## 3. State of the repo

**Ready to run.** Everything below is committed locally and needs your push.

| Area | Status |
|---|---|
| Data pipeline | Stable. 25 datasets (17 PD / 8 LGD), sanitize + dedup + manifests |
| Training | Two families, 48 trials/track, 50 epochs, per-epoch monitor every 5th epoch |
| Evaluation | 5-fold CV per dataset, 50-trial Optuna for all four classical baselines, ~40 metric columns |
| Tests | 275 passing (~6.5 min locally) |
| Docs | README, VSC_GUIDE, CHECKPOINTS, DATA_PIPELINE all updated for two families |
| Literature | `tfm-library/` submodule (shared, read-only) — 40+ paper summaries plus a synthesis |

**Two things must happen on the cluster before the next run:**

1. `pip install 'tabicl>=2.1.1,<3'` in the CreditPFN environment.
2. Stage both TabICL checkpoints **from a login node** — compute nodes have no
   outbound network, and the loaders deliberately refuse to auto-download so
   a missing file fails in seconds instead of hanging a GPU job. The command
   is in [CHECKPOINTS.md](CHECKPOINTS.md).

**TabICL's row caps were corrected on 2026-08-04** after checking Qu et al.
2026 rather than the library's defaults. Evaluation is now 1 000 000 rows,
matching v3 — million-scale in-context inference is TabICLv2's headline
capability (they report 1M samples × 500 features in ~450 s under 50 GB GPU),
and the earlier 50 000 would have handicapped it 20× against v3. Training is
now 26 000 rows/step, also matching v3, so a cross-family difference cannot
be confounded with context size; that sits inside TabICLv2's own stage-3
pretraining range (400–60 000 samples). **Neither number is measured on our
hardware yet** — `scripts/probe_row_cap.py` has a TabICL branch; run it
before the full sweep. The failure mode to watch is walltime, not memory —
that was the v2.6 lesson.

---

## 4. State of the experiments

### Run-4 (11–13 July, 64 trials, the clean one)

**PD — continued pretraining does nothing for discrimination.**

| | mean AUC over 5 held-out datasets |
|---|---|
| Untuned TabPFN-v3 (no finetuning at all) | **0.7622** |
| Tuned XGBoost / CatBoost | 0.749 |
| Best continued-pretrained variant | 0.7626 (+0.0004) |
| Worst | 0.7567 (−0.0055, high learning rate) |

LoRA was a no-op at every learning rate (|Δ| ≤ 0.0007). This is a definitive
negative *at a 12-dataset corpus scale* — which is a legitimate result to
report, not a failure.

**LGD — continued pretraining sharpens the density and blunts the point estimate.**

| Variant | Log-likelihood (higher better) | RMSE (lower better) |
|---|---|---|
| Untuned v3 | baseline | **best** (0.1399 / 0.1253) |
| v3, LR 1e-6, full pass | +0.056 nats | +0.001 (essentially free) |
| v3, LR 3e-5, full pass | **+0.32 nats** | +0.009 |
| v2.6, any LR | worse | worse |

The improvement is monotone in learning rate and exposure, replicated across
two runs, and version-specific (v2.6 degrades on both metrics). Never compare
`neg_nll` across architectures — the histogram granularity differs by ~3.3
nats and v2.6's values carry ~1 % clamped rows.

### Still open from run-4

Nine evaluation cells never produced results: eight PD cells (v2.6 ×
`algorithmwatch`, killed by the old 2-hour walltime) and one LGD cell. The
enabling fixes are already in the code. Recovery scores only the missing
cells:

```bash
STAGES="eval" bash scripts/slurm/run_full_pipeline.sh
```

---

## 5. Has someone already done this?

**No.** The closest work in the literature, and how it differs:

| Paper | What it did | Why ours is different |
|---|---|---|
| **Garg et al. 2025 — Real-TabPFN** | Continued pretraining of TabPFN-v2 on 71 *general-purpose* OpenML/Kaggle tables | General-purpose corpus, not a domain. Classification only. Our recipe follows theirs; our question (does *domain* specialisation help?) is the one they left open |
| **Kolberg et al. 2026 — TabPFN-Wide** | Continued pretraining for extreme feature counts | The target is a *data shape*, not a domain. Classifier only — no wide regressor exists |
| **Rubachev et al. 2025 — On Finetuning TFMs** | 342 single-dataset finetuning runs | Per-dataset finetuning, not corpus-level continued pretraining. No domain specialisation |
| **Tanna et al. 2026 — Data Presentation Over Architecture** | TFMs on Home Credit + Lending Club, 7 context-construction strategies | **Closest applied work, and it uses two of our datasets.** But it is in-context learning only — no finetuning, no continued pretraining — classification only, no temporal splits |
| **Purucker et al. 2026 — Beyond IID** | 142 datasets, IID / temporal / grouped splits, 3 TFMs vs 8 tuned baselines | Tests in-context learning only. The paper **explicitly lists TFM finetuning and continued pretraining as untested future work** — which is precisely our contribution |

**The defensible claims for a paper**, in descending strength:

1. **First domain-specific continued pretraining of a tabular foundation
   model on real domain data** — in any domain, and first in credit risk.
2. **First continued pretraining for tabular regression.** Every prior CPT
   paper is classifier-only. The LGD density result exists nowhere else.
3. **Two architectures, two priors.** With TabICL alongside TabPFN, a null
   result becomes a statement about the approach rather than about one model.
4. **Answers a named open question** from Purucker et al. — does adaptation
   rescue TFMs where in-context learning loses? — with the split protocol
   they showed to be decisive.
5. **Density and calibration treated as first-class outcomes**, not
   afterthoughts. Regulators care about the loss distribution, not just the
   mean; the LGD finding is only visible if you measure likelihood.

**Three objections a reviewer will raise, and the honest position:**

- *"The synthetic prior already covers credit-like data, so there was no
  headroom."* This is the strongest counter to the PD null, and it is partly
  supported by Purucker's finding that TabPFN-2.6's native calibration is
  already so good that post-hoc calibration makes it worse. Address it by
  reporting corpus scale explicitly and by showing the LGD result — where
  headroom clearly did exist.
- *"Context construction, not the model, drives credit AUC."* (Tanna et al.:
  balanced sampling is worth 3–4 AUC points, more than the spread between
  TFM families.) Our per-step sampler is deliberately identical across
  families so this axis is held constant — but it needs an explicit ablation,
  not just a claim.
- *"12 training and 5 test datasets is small."* True. Per-dataset results
  and paired statistics matter more than pooled means here.

---

## 6. What is missing before writing

Ordered by how much each one strengthens the paper per unit of effort.

### Must have

1. **Recover the 9 missing evaluation cells.** One command, no new code.
2. **Run the two-family sweep.** 48 trials/track. This is the single biggest
   addition to the paper's claim.
3. **Temporal splits — but scope them first; the corpus mostly cannot support
   them.** Purucker et al. show that scoring grouped/temporal tasks with IID
   splits distorts model rankings badly (Kendall τ ≈ 0.5). Credit data is
   inherently temporal, and this is the axis where in-context learning loses
   to tuned RealMLP — so it is exactly where adaptation might win.

   **Measured 2026-08-04, and it is the binding constraint:** only 5 of our 25
   raw datasets carry a parseable date column at all — PD `vehicle_loan`
   (`DisbursalDate`) and `bondora_peer2peer` (outcome-side dates only, so
   leakage-prone); LGD `loss2` (`Origination_Date`, `date_vintage_year` — the
   cleanest one we have), `base_model` (`DEAL_TransactionStartDate`) and
   `base_modelisation` (`DATE`). `sanitize.py` currently drops them, and the
   pipeline has no datetime handling anywhere.

   Worse, of the **held-out** datasets: **none of the 5 PD test sets has a
   date**, and only 1 of the 2 LGD test sets does (`loss2`). Both dated PD
   datasets sit in the *training* split.

   So there are three options, in increasing cost:
   - **(a) A temporal case study on `loss2`** — one held-out dataset, LGD
     only. Cheap, honest, and enough to say "the density gain survives a
     time-ordered split on the one dataset where we can test it."
   - **(b) Re-pin the corpus split** so `vehicle_loan` (and possibly
     `bondora`) become PD test sets. Costs a full retrain, and shrinks the
     already-small training corpus.
   - **(c) Re-source fuller raw files** — the complete Lending Club has
     `issue_d`, Home Credit has relative day offsets. Most scientifically
     satisfying, most work, and it changes the corpus mid-project.

   Either way this needs: date-column preservation in `sanitize.py` as a
   **split key, not a feature** (adding it as a feature would change the
   feature space and break comparability with run-4), plus a time-ordered
   fold generator in the eval. Recommendation: do (a) now, and treat (b)/(c)
   as a decision to make deliberately rather than a task to schedule.
4. **Multiple seeds.** Currently one seed per configuration, so "no effect"
   and "effect smaller than seed noise" are indistinguishable. Three seeds on
   the best few configurations is enough to state that properly.
5. **CRPS for LGD.** The only density metric comparable between TabPFN's
   histogram head and TabICL's quantile head. Without it the flagship result
   cannot be stated across families.
6. **A contamination audit.** Our test datasets are public Kaggle/OpenML
   tables, and the base checkpoints are claimed synthetic-only. Garg et al.
   run a tiered audit; reproduce it, because a reviewer will ask whether the
   base model has seen Home Credit or Lending Club.
7. **Paired statistical tests.** Wilcoxon signed-rank per dataset, Friedman
   across models. With 5 test datasets, a raw mean difference is not evidence.

### Should have

8. **RealMLP as a baseline.** It is the model that beat every TFM on
   non-IID splits in BeyondArena. Omitting it invites the obvious objection.
9. **A no-forgetting check.** Kolberg et al. report ρ = 0.9935 between their
   continued-pretrained model and the base on out-of-domain data. Showing
   credit specialisation did not damage general performance is cheap and
   pre-empts a real concern.
10. **A context-construction ablation** (uniform vs balanced sampling) —
    directly addresses the Tanna confound.
11. **A recalibration ablation.** Does post-hoc calibration on top of the
    continued-pretrained model help or hurt? Purucker found TabPFN is one of
    the few models it *hurts*; confirming that in credit is a small, quotable
    result.
12. **Compute cost reporting.** Continued pretraining is not free; a
    cost-per-gain figure makes the negative PD result actionable rather than
    merely disappointing.

### Nice to have

13. **More training datasets.** The clearest test of "was the corpus too
    small?" — the EDW ABS panel data was investigated for this, but its
    January 2026 terms appear to prohibit AI training unless the university
    agreement overrides them. That legal question is unresolved.
14. **A recipe-sensitivity study** (L2-SP on/off, query fraction, warmup) to
    show the null is not an artefact of one hyperparameter choice.

---

## 7. Concrete next steps

```bash
# 1. On your machine — push the work (I never push).
git add -A && git commit -m "TabICL v2 continued pretraining + eval fixes"
git push origin main
```

```bash
# 2. On VSC (Genius login node), in the CreditPFN env.
pip install 'tabicl>=2.1.1,<3'
python -c "from src.train.tabicl_compat import smoke_test; smoke_test('pd')"
```

```bash
# 3. Stage the TabICL weights once — LOGIN node only (see CHECKPOINTS.md).
python -c "
from huggingface_hub import hf_hub_download
import shutil, os
dest = os.environ.get('CREDITPFN_STAGING_ROOT', '/lustre1/project/stg_00211/CreditPFN') + '/checkpoints'
os.makedirs(dest, exist_ok=True)
for f in ('tabicl-classifier-v2-20260212.ckpt', 'tabicl-regressor-v2-20260212.ckpt'):
    shutil.copy2(hf_hub_download('jingang/TabICL', f), dest + '/' + f); print('staged', f)
"
```

```bash
# 4. Recover run-4's 9 missing eval cells (cheap, no training).
STAGES="eval" bash scripts/slurm/run_full_pipeline.sh
```

```bash
# 5. Then the full two-family run.
STAGES="data train eval" bash scripts/slurm/run_full_pipeline.sh
```
