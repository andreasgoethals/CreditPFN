# Results by research question

Start with the [PD overview](../notebooks/general/03_overview_pd.ipynb) and [LGD overview](../notebooks/general/04_overview_lgd.ipynb). Their figures show original/CPT paired effects, named tables, primary and secondary metrics, complete factorial contrasts, recipe differences, retention, context and recorded cost. Their design and unique-trial inventories prevent reused evidence from being counted twice.

| Question | Detailed notebooks | Interpretation |
|---|---|---|
| RQ1: benefits and stability | [E1 PD results](../notebooks/experiment1/03_results_pd.ipynb), [E1 LGD results](../notebooks/experiment1/04_results_lgd.ipynb), [E2 seeds](../notebooks/experiment2/02_results.ipynb) | Full updates with anchoring across learning rates; conservative reference and one additional seed. Individual tables accompany means. |
| RQ2: constrained updates | [E3 anchoring](../notebooks/experiment3/02_results.ipynb), [E4 freezing](../notebooks/experiment4/02_results.ipynb), [E5 DiagScale](../notebooks/experiment5/02_results.ipynb), [E6 LoRA](../notebooks/experiment6/02_results.ipynb) | Keep all four full/frozen × unanchored/anchored cells and their interaction. The adapter studies evaluate specified recipes, not isolated adapter mechanisms. |
| RQ3: data and exposure | [E7 sampling](../notebooks/experiment7/02_results.ipynb), [E8 corpus](../notebooks/experiment8/02_results.ipynb), [E9 episode size](../notebooks/experiment9/02_results.ipynb) | Separate corpus composition from episode length. Reuse the exact 8,192-row large-corpus control. Equal updates are not equal compute or row exposure. |
| RQ4: target context | [E10 context curves](../notebooks/experiment10/01_context_curves.ipynb) | Absolute original/CPT performance and paired differences; AUC/Brier or R²/RMSE; fixed table cohorts and complete test rows; five folds. Named-table pages expose heterogeneity. |

Experiment 0 notebooks report correctness evidence. Training notebooks retain the corresponding trajectories and cost diagnostics. Historical run/config IDs differ from the new public experiment numbers; [README.md](../README.md#stable-campaign-identities-and-storage) gives the exact mapping. The shared 512-run factorial contributes once to unique-run counts.

AUC/R² changes use CPT minus original; Brier/RMSE gains use the opposite sign, so positive favors CPT. Benchmark RMSE gains are relative percentages where stated; context panels retain absolute RMSE and its absolute difference. Five folds are averaged within table before equal-table averages. Fold agreement and table heterogeneity are not independent replication or confidence intervals.

The main reference is fixed independently of held-out outcomes. Highest observed recipes and leave-one-table-out recipe selection are retrospective sensitivity summaries, not new confirmatory evaluations. The later v2/v2.6 extension is exploratory. Read the actual inference-policy table: cross-study TabICL claims require the published FP32 refresh, not merely a pending job or new configuration.

Each notebook's last code cell prints its complete section-ordered record. The same summaries are in [All_Results.md](../output%20CreditPFN/All_Results.md); captions are inline and in [CAPTIONS.md](../output%20CreditPFN/figures/CAPTIONS.md). PDFs are generated under `output CreditPFN/figures/<public experiment>/`. Choose manuscript figures by research question; the full notebooks remain the detailed evidence record.

Regenerate all notebooks, figures, captions and summaries in parallel:

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 2
```
