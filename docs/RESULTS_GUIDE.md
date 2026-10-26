# Results organized by research question

This index follows the manuscript's theory-informed exploratory questions. The framing was
refined after outcomes were available; it is not a preregistration. Figures summarize completed
campaign measurements, not new model fits. The [research brief](RESEARCH_BRIEF.md) owns the method.

## Q1. When does specialization help, and what does it preserve?

| Contrast | Notebook and exact saved figure | Unit and control | Boundary |
|---|---|---|---|
| Matched CPT choices | [PD results, section 2](../notebooks/experiment1/03_results_pd.ipynb); [AUC/Brier/log-loss choice summary](../output%20CreditPFN/figures/experiment1/experiment1__03_results_pd__01_pd_choice_contrasts_credit.pdf); [LGD R²/RMSE summary](../output%20CreditPFN/figures/experiment1/experiment1__04_results_lgd__01_lgd_choice_contrasts_credit.pdf) | Pair all other settings, average differences within table, then weight tables equally. Each base keeps a complete cohort across displayed metrics. | Positive choice contrasts can mean less damage, not CPT beating original weights. Neither main-grid update mode uses LoRA. |
| Recipe combinations | [AUC](../output%20CreditPFN/figures/experiment1/experiment1__03_results_pd__02_pd_recipe_grid_credit_roc_auc.pdf), [Brier](../output%20CreditPFN/figures/experiment1/experiment1__03_results_pd__03_pd_recipe_grid_credit_brier_score.pdf), [log loss](../output%20CreditPFN/figures/experiment1/experiment1__03_results_pd__04_pd_recipe_grid_credit_log_loss.pdf); [R²](../output%20CreditPFN/figures/experiment1/experiment1__04_results_lgd__02_lgd_recipe_grid_credit_r2.pdf), [RMSE](../output%20CreditPFN/figures/experiment1/experiment1__04_results_lgd__03_lgd_recipe_grid_credit_rmse.pdf) | 17 PD / 8 LGD credit tables; all 16 rate × anchoring × scope recipes within base. Each cell compares CPT with its own original model. | Metrics remain separate. Printed values retain small effects; table counts and spread remain in the summary. Maxima are exploratory, not selected deployment recipes. |
| Transfer and retention | [PD recipe effects](../output%20CreditPFN/figures/experiment1/experiment1__03_results_pd__05_pd_benchmark_transfer_retention.pdf); [LGD recipe effects](../output%20CreditPFN/figures/experiment1/experiment1__04_results_lgd__04_lgd_benchmark_transfer_retention.pdf) | Same recipe paired across credit and non-credit domains; retention first requires all four CPT partitions per table. | Four same-task retention tables per track are a finite panel, not general retention. Panel-specific axes reveal small effects. |
| Conservative reference | [PD results, section 4](../notebooks/experiment1/03_results_pd.ipynb); [reference effects](../output%20CreditPFN/figures/experiment1/experiment1__03_results_pd__06_pd_reference_uncertainty.pdf) | 17 PD tables; CPT minus original AUC, paired over five outer folds. Full updates, LR 3e-7, L2-SP 0.003, seed 42. | Table-bootstrap intervals condition on fitted checkpoints and an unverified exchangeability assumption. Per-table effects for all reference metrics remain printed. |
| Probability and error functionals | [PD calibration](../output%20CreditPFN/figures/experiment1/experiment1__03_results_pd__07_pd_calibration_overview.pdf); [LGD distribution diagnostics](../output%20CreditPFN/figures/experiment1/experiment1__04_results_lgd__07_lgd_v26_distribution_tradeoff.pdf) | Raw/Platt/isotonic changes use the same 17 tables within base. Calibrators fit inner validation rows. LGD compares the fixed v2.6 reference on eight tables. | ECE is bin-dependent; the historical clipped log score is not uncapped likelihood. Detailed repeated reliability and threshold plots are omitted from the default notebook. |
| Fixed-context monitors | [PD change versus learning rate](../output%20CreditPFN/figures/experiment1/experiment1__03_results_pd__09_pd_monitor_recipe_changes.pdf); [LGD](../output%20CreditPFN/figures/experiment1/experiment1__04_results_lgd__08_lgd_monitor_recipe_changes.pdf) | One equal-table mean per rate × anchoring × scope recipe at 10,000 updates, paired with update zero. | These are training monitors, not final five-fold performance. Separate legend boxes explain rate, L2-SP and scope; constant initial-score x-axes are retired. |
| Training randomness | [Seed results, section 2](../notebooks/experiment2/02_results.ipynb); [PD seed contrast](../output%20CreditPFN/figures/experiment2/experiment2__02_results__01_pd_seed_benchmark_credit.pdf), [LGD seed contrast](../output%20CreditPFN/figures/experiment2/experiment2__02_results__03_lgd_seed_benchmark_credit.pdf) | Seed 43 minus 42 at the fixed reference; match partition, table and inference protocol before aggregation. | Two seeds at one recipe do not identify grid-wide variance. |
| Numerical-policy qualification | [Auxiliary checks, section 4](../notebooks/experiment0/04_auxiliary_experiments.ipynb); [TabICLv2 precision effects](../output%20CreditPFN/figures/experiment0/experiment0__04_auxiliary_experiments__01_pd_tabicl_numerical_policy.pdf) | Three post-hoc diagnostic PD tables × five paired outer folds. Original, unchanged-save and CPT; automatic versus FP32/no-FA3. | Selected-table policy dependence, not its prevalence, LGD sensitivity or an all-table FP32 replacement. Boundary probes remain tables. |

Positive benchmark effects favor CPT. AUC percentage-point changes, Brier reductions multiplied
by 100, log-loss reductions, R² changes and relative RMSE gains are never pooled. Relative RMSE
contrasts use the original model's error as denominator, including comparisons between CPT choices.

## Q2. Does additional evidence justify its cost?

| Contrast | Notebook and exact saved figure | Unit and control | Boundary |
|---|---|---|---|
| Training exposure versus quality | [Sampling results, section 4](../notebooks/experiment3/02_results.ipynb); [PD quality/cost](../output%20CreditPFN/figures/experiment3/experiment3__02_results__07_pd_sampling_benchmark_quality_cost.pdf) | Final paired table effects; recorded training hours sum four CPT partitions. Three protocols use their own one-sample control. | Equal successful updates do not match compute. Classical controls and main-grid balanced-PD CPT are not replacement controls. |
| Matched time and exposure ratios | [PD training-cost ratios](../output%20CreditPFN/figures/experiment3/experiment3__01_training__18_pd_sampling_costs.pdf), [LGD training-cost ratios](../output%20CreditPFN/figures/experiment3/experiment3__01_training__21_lgd_sampling_costs.pdf); complete base/partition ratios in results section 4's text summary | Full pass or accumulation divided by one sample within the same base and original training partition. Means/IQRs summarize partition ratios. | Time excludes evaluation/queueing; exposure counts repeated processed rows. Averaged chunk gradients do not form one all-row attention context. |
| Target labeled context | [Context curves, section 3](../notebooks/experiment4/01_context_curves.ipynb); [PD through 50,000](../output%20CreditPFN/figures/experiment4/experiment4__01_context_curves__04_context_pd_through_50000.pdf), [LGD through 2,048](../output%20CreditPFN/figures/experiment4/experiment4__01_context_curves__06_context_lgd_through_2048.pdf) | Original/CPT five-fold table means. PD fixed cohorts contain 17/14/9 tables through 128/2,048/50,000 rows; LGD has 8/4 through 128/2,048. Both families share each cohort. | Test rows are unchanged; validation labels are an additional fixed budget. Thin curves are individual table effects, not repeated-context variability. |

Main and seed-check PD CPT use balanced context/query draws. **All three experiment-3 PD
protocols use proportional draws.** Thus experiment 3's one-sample checkpoint is its own control;
its differences from experiment 1 also change prevalence policy. Follow-ups 5–7 use proportional PD draws.

## Q3. Do capacity, corpus policy or episode budget explain the response?

| Contrast | Notebook and exact saved figure | Unit and control | Boundary |
|---|---|---|---|
| Restricted query scaling | [Adapter results, section 2](../notebooks/experiment5/02_results.ipynb); [PD](../output%20CreditPFN/figures/experiment5/experiment5__02_results__01_adapter_benchmark_pd_roc_auc.pdf), [LGD](../output%20CreditPFN/figures/experiment5/experiment5__02_results__02_adapter_benchmark_lgd_r2.pdf) | All three DiagScale rates against their matched original; 17 PD / 8 LGD tables, with four-table same-task retention panels. | TabICL original/adapted inference is matched FP32/no-FA3 here. Different precision, optimizer and episode settings prevent an isolated parameter-count comparison with experiment 1. |
| Corpus eligibility | [Corpus results, section 2](../notebooks/experiment6/02_results.ipynb); [all versus large tables](../output%20CreditPFN/figures/experiment6/experiment6__02_results__01_size_benchmark_roc_auc.pdf) | The same 17 held-out PD tables and original partitions; 8,192-row episode cap in both arms. | Filtering changes corpus identities, diversity and revisit frequency; it is not a pure table-size treatment. |
| Episode budget | [Episode results, section 2](../notebooks/experiment7/02_results.ipynb); [episode-size contrasts](../output%20CreditPFN/figures/experiment7/experiment7__02_results__01_size_benchmark_roc_auc.pdf) | The same large-table training pool and 17 excluded PD tables; 2,048/8,192/26,000-row episodes. | The 8,192-row arm is exactly eight experiment-6 large-corpus checkpoints, reused once. Larger episodes also change query count, exposure and compute. |

TabICLv2's automatic-policy context/corpus/episode effects retain the selected-table numerical
qualification in Q1. Cross-study original scores under different policies are not interchangeable.
No unavailable measurement is treated as zero, and no unrelated task metrics are averaged together.

## Reading and reproducing the evidence

Experiment 8 adds a prospective restricted-capacity check in
[LoRA training](../notebooks/experiment8/01_training.ipynb) and
[LoRA results](../notebooks/experiment8/02_results.ipynb). Rank-16 Q/V updates are compared
with their own original model at three adapter rates. Its plots require observed evidence
and complete paired folds; pending campaigns produce coverage tables rather than empty
figures. Training shows native losses, effective Q/V displacement, credit/retention monitors
and cost. Results show AUC/R², Brier/log loss or RMSE and retention on common table cohorts.
TabICL uses matched FP32/no-FA3 inference; experiment-1 automatic-policy results are not
pooled into this comparison. This tests a specified LoRA recipe, not all adapter designs.

Training notebooks contain trajectories and native losses, with compact shared choice, movement and cost summaries. Detailed optimizer/device diagnostics remain in tables.
Results notebooks distinguish fixed-context milestone monitors from completed five-fold final
evaluation. Monitor/final variants retain separate filenames, explicit captions and structured
`measurement` metadata. Their numbers are not interchangeable.

Every displayed plot/table has its full numerical text in the notebook's final code cell and in
[All_Results.md](../output%20CreditPFN/All_Results.md). Captions appear inline and in
[CAPTIONS.md](../output%20CreditPFN/figures/CAPTIONS.md); PDF and metadata files are generated locally
and remain gitignored. This index points to current notebook-generated sources; manuscript-local
figure copies are independent version assets. Refresh selected copies deliberately when revising a manuscript.

From the repository root in PowerShell, regenerate the complete collection in two parallel kernels:

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 2
```

The notebook collection is the full analysis record. A manuscript should select a few panels that
answer these questions, retaining the remaining diagnostics in appendices or the repository.
