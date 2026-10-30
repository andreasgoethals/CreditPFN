# Results by research question

Start with the [PD overview](../notebooks/general/03_overview_pd.ipynb) and [LGD overview](../notebooks/general/04_overview_lgd.ipynb). They show the cross-study outcome pattern, the reference on two metrics, retention and recorded cost. Detailed comparisons live once with their research question. Design and unique-trial inventories prevent reused evidence from being counted twice.

| Question | Detailed notebooks | Interpretation |
|---|---|---|
| RQ1: benefits and stability | [Training](../notebooks/rq1/01_training.ipynb), [results](../notebooks/rq1/02_results.ipynb) | E1 learning rates and E2 seeds together. Individual tables accompany averages; seed contrasts use the predefined reference. |
| RQ2: constrained updates | [Training](../notebooks/rq2/01_training.ipynb), [results](../notebooks/rq2/02_results.ipynb) | E3/E4 share one complete full/frozen × unanchored/anchored view with the interaction retained. E5/E6 share recipe dashboards, without claiming isolated adapter effects. |
| RQ3: data and exposure | [Training](../notebooks/rq3/01_training.ipynb), [results](../notebooks/rq3/02_results.ipynb) | E7 sampling and E8/E9 corpus/episode effects. Paired contrasts retain the exact shared 8,192-row control. Equal updates are not equal compute or row exposure. |
| RQ4: target context | [Context curves](../notebooks/rq4/01_context_curves.ipynb) | E10 absolute original/CPT performance and paired differences; AUC/Brier or R²/RMSE; fixed table cohorts and complete test rows; five folds. Named-table pages expose heterogeneity. |

There are 15 notebooks: four in `experiment0/`, four in `general/`, two each in `rq1/`–`rq3/`, and one in `rq4/`. Experiment 0 reports correctness evidence; training notebooks retain trajectories, optimizer health, parameter movement and costs. Historical run/config IDs differ from the public experiment numbers; [README.md](../README.md#stable-campaign-identities-and-storage) gives the exact mapping. The shared 512-run factorial contributes once to unique-run counts.

AUC/R² changes use CPT minus original; Brier/RMSE gains use the opposite sign, so positive favors CPT. Benchmark RMSE gains are relative percentages where stated; context panels retain absolute RMSE and its absolute difference. Five folds are averaged within table before equal-table averages. Fold agreement and table heterogeneity are not independent replication or confidence intervals.

The main reference is fixed independently of held-out outcomes. Highest observed recipes and leave-one-table-out recipe selection are retrospective sensitivity summaries, not new confirmatory evaluations. The later v2/v2.6 extension is exploratory. Read the actual inference-policy table: cross-study TabICL claims require the published FP32 refresh, not merely a pending job or new configuration.

Each notebook's last code cell prints its complete section-ordered record. The same summaries are in [All_Results.md](../output%20CreditPFN/All_Results.md); captions are inline and in [CAPTIONS.md](../output%20CreditPFN/figures/CAPTIONS.md). PDFs use the same six groups under `output CreditPFN/figures/`. Each notebook owns its filename prefix; full reruns retire the former experiment-notebook figures without touching raw scientific evidence. This grouping deliberately differs from the template's experiment-per-notebook layout.

Regenerate all notebooks, figures, captions and summaries in parallel:

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_notebooks --workers 2
```
