# From this repo to a paper

Two things that live nowhere else in `docs/`: **whether this has already been
published** (§1) and **what evidence is still missing before writing** (§2).

Trimmed on 11-08-2026. What used to be here as "state of the repo", "state of
the experiments" and "next steps" was duplicated and going stale — the run
record is now `AGENTS_MEMORY.md` (the Runs table), the numbers are
`RESULTS.md`, and what changed is `CHANGELOG.md`.

---

## 1. Has someone already done this?

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
3. **Two architectures, two priors.** With TabICLv2 alongside TabPFN, a null
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

## 2. What is missing before writing

Ordered by how much each one strengthens the paper per unit of effort.

### Must have

1. **A complete two-family eval.** Every run so far has finished training and
   then lost part of the eval to walltime or queueing. Nothing else on this
   list can be judged until one full trained-vs-untuned grid exists.
2. **Run the two-family sweep to the full step budget on BOTH tracks.**
   36 trials/track. This is the single biggest
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
   histogram head and TabICLv2's quantile head. Without it the flagship result
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
