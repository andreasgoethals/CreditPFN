# CreditPFN — research brief (second campaign)

Method authority for the second campaign, agreed with Andreas on 04-11-2026. The grid itself is
[`config/design.yaml`](../config/design.yaml); this page explains it. The first study's brief is
in Git history (commit `2c8ef18`) and its results in `archive/study-2026-11.zip` (local, gitignored).

## Question

Does continued pretraining (CPT) of a tabular foundation model on real datasets from one domain
(credit risk) improve its predictions on *unseen* datasets of that domain, and when and why?
PD is binary default classification, LGD is loss-rate regression. Four base models: TabPFN v2,
v2.6 and v3, and TabICLv2. The study is descriptive: it does not assume CPT helps.

Credit risk is the case study, not the limit of the contribution. The ablations (update size and
anchoring, trainable scope, class balance, exposure, table and episode size, context scarcity,
source-target transfer, CPT versus target fine-tuning) are run on four architectures and two task
types, so they also describe how continued pretraining and fine-tuning of tabular foundation
models behave in general; the domain control (E3) separates what is credit-specific from what
any comparable real-data corpus would do.

## Research questions

| RQ | Question |
|---|---|
| RQ1 Transfer | Does CPT on credit datasets improve prediction on credit datasets the model has never seen, for which base models and which task, how does the effect develop during training, how much is training randomness, and is it worth it compared with fine-tuning on the target table? |
| RQ2 Weight change | How do update size, anchoring to the pretrained weights and which parameters are trained trade gains on credit data against forgetting on non-credit data? |
| RQ3 Training data | Which training tables (small vs large), how much of each table, how large each training episode, and how class balance affects transfer. |
| RQ4 When and why | On which target tables CPT helps or hurts, what those tables have in common, how this depends on the source tables, and whether the benefit is larger when the target has few labelled rows. |

## Data, partitions and evaluation

* 17 PD and 8 LGD tables. Each task's tables are split once (seed 1729) into four groups; every
  run trains on three groups and is evaluated on the fourth, so every table is evaluated by a
  model that never saw it. Single-source runs (E12) train on one table and evaluate on all others.
* Final evaluation: five outer folds per held-out table; the fold's training rows are the
  labelled context (20% kept for validation, thresholds and calibration); the original and the
  CPT model see identical context and test rows. All models are evaluated in FP32 with 8
  inference views (preprocessing/permutation ensemble members): for TabPFN v2, 8 versus 16
  views changes performance by 0.01 percentage points (Ye et al. 2025, Fig. 9); TabICLv2 captures
  most ensembling gains at 8 (Qu et al. 2026, App. L.1); TabArena uses 8 for both families.
  Training monitors keep the training's 2 views. TabPFN v2's context is capped at its 10,000-row
  limit.
* Retention (forgetting): 19 binary and 12 regression TabArena-v0.1 tables below 50,000 rows,
  without the four corpus overlaps and three credit-like tables (`config/retention.yaml`).
* Domain control (E3): one non-credit twin per credit table from the curated OpenML suites,
  chosen by an optimal one-to-one assignment on rows, features and categorical features and then
  subsampled to its credit table's shape (`src/data/control_matching.py`,
  `config/domain_control.yaml`). Each partition's control corpus is the twins of that partition's
  credit training tables. Twins exclude credit, banking, insurance and finance tables and every
  table of the retention panel, and have at most 30 million cells (download size). Their targets
  are deliberately not matched. Where no comparable table is as large (the biggest PD tables), the
  twin has fewer rows; training episodes are capped at 10,000-26,000 rows, so this mainly reduces
  row diversity across visits.
* Baselines: CatBoost and logistic (PD) / linear (LGD) regression, 50 seeded Optuna trials
  without a time limit.
* Metrics. PD: AUC, log loss, Brier, ECE, calibration slope and intercept, calibration-in-the-
  large and bias. LGD: RMSE, normalised RMSE, R², MAE, bias, NLL, CRPS (from TabPFN's bucket distribution, with its two half-normal tails approximated as uniform bins; from the
  distributions, quantile-based otherwise), pinball loss and 80% interval coverage. Results are
  averaged within a table before aggregation; LGD aggregates use R²/normalised RMSE.

## Reference recipe

AdamW at peak learning rate 3e-7, L2-SP 0.003 (no weight decay), 10% warm-up then cosine to 5%,
gradient clip 1, all weights trainable, 10,000 updates, episodes up to 10,000 rows (v2, v2.6) or
26,000 (v3, TabICLv2) split 60/40 into context and query, proportional PD sampling, one episode
per table visit, all training tables, two preprocessing views per episode, BF16 training (TabICLv2: FP16 with loss scaling, its upstream recipe; BF16 corrupts its forward), seed 42.

Monitoring at 0, 50, 100, 150, 200, 250, every 250 updates to 2,500 and every 500 after, plus when 1, 2, 5, 10, 20 and 50
million rows have been processed (equal-data comparisons). Each point scores the held-out credit
tables, the run's own training tables and the retention panel on one fixed context/query split,
with the same two preprocessing views as training and up to 10,000 rows per table (Experiment 0
times 2,000, 10,000 and full tables before this is fixed).

### Mechanisms specific to this campaign

* **LGD objective (TabPFN): window NLL.** The negative log density of the target over a window of
  5 of the 5,000 output buckets. For smooth densities it equals the usual bar NLL to within
  ~0.01 nats; for the released v2.6 regressor, whose density is comb-shaped, it no longer rewards
  filling empty buckets (which had moved point predictions and cost R²). TabICL keeps its native
  quantile objective. `src/train/regression_objectives.py`.
* **L2-SP for adapters on the effective weight change.** LoRA: `0.5·λ·Σ‖(α/r)BA‖²`, exactly L2-SP
  of the adapted weights, zero with zero gradient at initialisation. DiagScale: the equivalent
  weight change of the query scaling, `0.5·λ·Σ_j δ_j²(‖W_q[j]‖² + b_j²)`, exact for v2/v2.6 and the
  weight-space equivalent for v3/TabICLv2 (whose query scaling is query dependent). The same λ
  therefore means the same distance from the pretrained weights in every scope.
  `src/train/l2sp.py`; plain weight decay on LoRA factors would instead be a nuclear-norm penalty
  that is non-zero at initialisation.
* **Gradient clipping at norm 1** (upstream default of both fine-tuners). TabPFN LGD gradients
  (norm 8-35 in the first study) are clipped at every update, so with AdamW each update's
  gradient is normalised: the update size is unchanged, but every table visit weighs equally.
  Experiment 0's `noclip` pilot measures whether this matters.

## Experiments

One arm = 4 models × 4 partitions × the listed tasks. E1 is the shared control of E2, E3 and
E5–E10 and is never re-run. 760 standard runs plus 100 short single-source runs.

| RQ | Exp | Question | Change vs the reference | Tasks | Runs |
|---|---|---|---|---|---|
| 1 | E1 | Transfer and its development | none | PD, LGD | 32 |
| 1 | E2 | Training randomness | seeds 43 and 44 (with E1: three seeds) | PD, LGD | 64 |
| 1 | E3 | Domain specificity | CPT on the matched non-credit twins of the partition's credit training tables | PD, LGD | 32 |
| 1 | E4 | CPT vs fine-tuning on the target table | per fold: fine-tune the original and the E1 model on the fold's training rows (upstream fine-tuners; lr chosen from 1e-6, 1e-5, 6.5e-5 on the validation split, which also drives early stopping) | PD, LGD | fine-tunes only |
| 2 | E5 | Update size × anchoring | lr 1e-7, 1e-6, 3e-6, 1e-5 at λ 0.003; λ 0 and 0.03 at lr 3e-6 and 1e-5; λ 0 at the reference rate | PD, LGD | 288 |
| 2 | E6 | Which weights change | frozen backbone (3e-7, 3e-6); DiagScale (1e-4, 1e-3); LoRA r16 query/value (3e-6, 1e-5); λ 0.003 throughout | PD, LGD | 192 |
| 3 | E7 | Class-balanced episodes | balanced PD sampling | PD | 16 |
| 3 | E8 | Seeing every row | full pass, one update per chunk | PD, LGD | 32 |
| 3 | E9 | Small vs large tables | small tables only (≤10,000 rows) vs the same number of large tables | PD | 32 |
| 3 | E10 | Episode size | 2,048 / 4,096 / 8,192 / 16,384 (v3, TabICL) rows and the reference size on the large-table corpus; equal updates, rows and compute; matched-context scoring | PD | 72 |
| 4 | E11 | Context scarcity | evaluation only: original vs E1 with 8 … 50,000 labelled context rows and the full pool, five folds; the fold's validation rows stay labelled for calibration and thresholds, so this is context scarcity rather than total label scarcity | PD, LGD | 0 |
| 4 | E12 | Which sources help which targets | CPT on one table (2,500 updates, lr 1e-6), scored on every other table | PD, LGD | 100 short |

### Design choices and their evidence

* **3e-7 stays the reference** — the best of the first study's rates: final five-fold PD AUC
  +0.0009 on held-out credit tables (v2 +0.005), LGD least harmed; 1e-5 and 3e-5 hurt (mostly
  v3). 1e-7 tests below it; λ is varied only where the first study showed it matters (≥3e-6).
* **No longer-training arm.** First-study held-out curves were flat by 2,500 updates at 3e-7 and
  falling at larger rates; dense monitoring shows every shorter horizon.
* **Frozen backbone at the same rates as full updates** so that scope is the only difference;
  per-dataset tuning as in Rubachev et al. 2025 is impossible without target labels.
* **E4 fine-tuning rates 1e-6, 1e-5 and 6.5e-5, chosen on the validation split** (12-11-2026).
  Rubachev et al. tune the rate per dataset over ten log-spaced values from 5e-6 to 5e-4; their
  two most frequent winners are 2.3e-5 and 1.8e-4, with geometric midpoint 6.5e-5. Ning et al.
  2026 fine-tune TabPFN-3 at 1e-6 and Tanna et al. 2026 report TabICL collapsing at 1e-5 and
  above, so one fixed rate would favour some models. Each cell records the validation score and
  the test score of every candidate, so the fixed-rate comparison remains available.
* **Dropped arms.** Muon, the 80/20 context/query split and λ 0.3 were removed on 05-11-2026 to
  spend the budget on a third seed and the LGD domain control.
* **E9 count-matched**: small-only (4–5 tables per partition) versus the same number of large
  tables. The two corpora still differ in which tables they contain (features, targets, sources), so
  E9 compares these selected corpora rather than isolating table size; LGD has only two tables
  above 10,000 rows.
* **Monitoring with the training views**: in the first study, PD AUC on the model's *own training
  tables* fell at 1e-5 while training loss fell, which points to a mismatch between training
  episodes (10,000–26,000 rows, 2 views) and the monitor (≤2,000 rows, 4 views). The monitor now
  uses 2 views and larger contexts; Experiment 0 includes a single-table sanity check (the
  trained table's own score must improve).

## RQ4 analysis (after training, from saved predictions)

For every experiment and every credit and retention table: the paired gain (CPT minus original,
averaged over folds) related to table size, features, share of categorical features and missing
values, default rate or LGD zero/one mass, the original model's score (headroom) and similarity
to the training tables (dataset properties and the model's own representation); credit segment;
discrimination vs calibration; and whether a table's gain exceeds the seed noise of the three
reference seeds (E1, E2). E12 gives the source→target transfer matrix from the training monitor (one fixed split, up to
10,000 rows per target table; not a five-fold evaluation); E11 the gain as a function
of labelled rows. Heavy
computation (similarities, the transfer matrix) runs in `src` on VSC; notebooks only call `src`.

## Experiment 0

Debugging only (`config/design.yaml` `experiment0`, partition 0, 250 updates): zero learning rate
(state and monitor parity), reference, monitor sizes 2,000/10,000/50,000, the single-table sanity
check, learning rates 0, 1e-8, 3e-7 and 1e-5 (dose response), no gradient clipping, bar NLL vs
window NLL per TabPFN model, frozen, DiagScale and LoRA each with and without L2-SP, full pass,
balanced, small-only, large-matched, 2,048 and 16,384-row episodes, single-source, domain control
(PD and LGD); exact resume for AdamW, DiagScale and LoRA on every model; update-zero training
loss in FP32 versus the training precision per model; FP32
inference capacity at each model's largest benchmark context; a mini context curve and E4
fine-tuning per model; end-to-end five-fold benchmark of the reference pilots including the
classical baselines; LGD density roughness per table and model; the container test suite with
upstream-parity inputs required.

## Limitations

Twenty-five tables give limited power for table-level explanations; effects in the first study
were small (≤0.005 AUC), so results are reported with per-table paired differences and seed noise
rather than single means. Partitions were drawn once; same-source tables must sit in one
partition (audited before launch). DiagScale's L2-SP is an exact weight-change penalty only for
v2/v2.6. The window objective deviates slightly from Real-TabPFN's bar NLL.
