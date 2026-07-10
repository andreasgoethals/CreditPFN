# 2026-07-09/10 supercomputer run: forensic analysis

## Scope and status

This report analyses the log snapshot copied on 2026-07-10: **183 logs**
(64 training, 118 PD evaluation, and one evaluation-gate log; about 84 MB).
It is a snapshot of an active pipeline, not a completed experiment.

| Stage | Observed state | Interpretation |
|---|---:|---|
| LGD training | 32/32 tasks completed | Training grid complete; downstream evaluation is still required. |
| PD training | 27/32 tasks completed | Tasks 0–4 ended abruptly together without traceback, final summary, or checkpoint. This pattern is consistent with external cancellation/termination, not a caught Python exception. |
| PD evaluation logs | 102 complete, 13 still partial, 3 surplus/out-of-bounds | Current metrics are provisional. Forty-five valid task indices had no log in the snapshot. |
| LGD evaluation logs | 0 | No downstream LGD conclusion can yet be drawn. |

The five missing PD trials are all v3: LR `3e-7` full FT/LoRA with both
pass modes (tasks 0–3), plus LR `1e-6` full FT in `one_sample` mode (task
4). These conservative full-FT trials are scientifically important and must
not be treated as failed-performing models; they were simply not evaluated.

## What the completed PD results say

The table below averages the available per-dataset mean metrics. Untuned v3
and the classical baselines have all five held-out datasets; untuned v2.6 has
only four completed datasets in this snapshot.

| Model | Datasets | ROC-AUC | F1 | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| XGBoost | 5 | 0.7494 | 0.6037 | 0.0137 | 0.1548 |
| CatBoost | 5 | 0.7490 | 0.6049 | 0.0123 | 0.1545 |
| Logistic regression | 5 | 0.7018 | 0.5439 | 0.0223 | 0.1717 |
| Untuned TabPFN v3 | 5 | **0.7622** | **0.6138** | 0.0196 | **0.1526** |
| Untuned TabPFN v2.6 | 4 | 0.7751 | — | 0.0176 | — |

Per-dataset AUC leaders in the completed snapshot:

| Dataset | Best observed AUC | Reading |
|---|---:|---|
| Taiwan credit card | v3 LoRA LR `1e-5`: 0.7894 | Untuned v3 is 0.7893; effectively no change. |
| myHome | Untuned/v3 LoRA: about 0.5974 | TabPFN leads XGBoost (0.5812), but fine-tuning adds nothing visible. |
| bank status | Untuned v3: 0.8130 | v3 clearly beats the completed v2.6 results here. |
| AlgorithmWatch | XGBoost: 0.6620 | CatBoost 0.6617 and untuned v3 0.6605 are effectively tied; v2.6 evaluations were still incomplete. |
| credit risk | v3 LoRA LR `1e-5`: 0.9510 | Untuned v3 is 0.9506 and CatBoost 0.9475; the fine-tuning difference is negligible. |

### Fine-tuned versus matching untuned checkpoint

The robust pattern is negative or neutral, not positive:

- v3 `one_sample` LoRA at `1e-6`/`1e-5` is numerically indistinguishable
  from untuned v3 (mean AUC change about 0.0000). LoRA at `3e-5` is slightly
  worse and slightly less calibrated.
- v3 full FT is unstable at the evaluated learning rates/exposures. Examples:
  LR `1e-5` `one_sample` has mean AUC change about -0.0285 and ECE +0.0806;
  LR `1e-6` `full_pass` about -0.0506 / +0.1473; LR `3e-5` variants degrade
  more strongly. This is catastrophic forgetting/overconfidence, not useful
  domain adaptation.
- v2.6 LoRA is also almost unchanged from untuned. Full-pass full FT becomes
  worse as LR rises (approximately -0.012 mean AUC at `1e-5` and -0.050 at
  `3e-5` on the datasets completed for both models).

Therefore this snapshot provides **no convincing evidence that continued
pretraining improves held-out PD performance**. It does provide evidence that
high-exposure full FT can damage both discrimination and calibration, while
the current LoRA setup often moves the predictor too little to matter. The
missing low-LR v3 full-FT trials prevent a final statement about the most
conservative adaptation regime.

## Why the training curves appeared more optimistic

The epoch monitor did not compare like with like. The pre-finetuning baseline
used one random seed, while each later epoch added an epoch-dependent seed and
therefore evaluated different sampled rows and context/query splits. On only
2 000 monitor rows per dataset, sampling noise created apparent AUC/RMSE lift
that disappeared in full held-out cross-validation. The monitor is diagnostic,
not a substitute for the benchmark, and its historical baseline-to-final
differences are invalid. The code now reuses one fixed train sample and one
fixed test sample throughout every learning curve.

LGD training summaries similarly appear to improve sampled RMSE, but no LGD
evaluation logs were supplied and the same monitor-seed confound applies.
There is currently **no defensible downstream LGD result**.

## Operational findings from the logs

- The sklearn `ColumnTransformer` FutureWarning occurred **97 877 times**.
  The old regex did not match its leading newline/multiline text. The filter is
  now exact, multiline-aware, and leaves unrelated warnings visible.
- PyTorch emitted 32 `searchsorted` warnings for non-contiguous LGD targets.
  The target views are now made contiguous before bar-distribution NLL.
- AMP discarded **302 optimizer steps** because of inf/NaN gradients, mainly
  in LGD/v3. Historical logs misleadingly reported zero skipped steps. B200
  runs now prefer BF16 (FP32-like exponent range), count AMP skips separately,
  and can abort a sustained skip storm.
- All 64 training jobs could read but not write project staging and correctly
  fell back to `$VSC_DATA`. The gate could not archive the fallback copy back
  to staging, so durable-storage permissions still require an infrastructure
  fix; code cannot grant that permission.
- A single shared success sentinel meant one successful array task released
  evaluation for an incomplete 32-task track. Success is now per task and the
  gate requires the complete planned grid.
- The old submitter sized evaluation from the planned training grid before
  training finished. With five absent checkpoints it launched surplus tasks.
  It now computes the exact post-training roster immediately before `sbatch`.
- The gate printed 2 183 identical polling lines. It now logs state changes
  plus a ten-minute heartbeat and treats timeout as failure.
- Concurrent Slurm tasks could both create the shared manifest with `"w"` and
  silently truncate a completed row. Manifest creation/appends are now locked
  across processes and flushed durably.

## Scientific limitations that remain

Even after the pipeline completes, claims must be qualified: only five PD and
two LGD datasets are held out; there is one random seed (`42`); random CV does
not test temporal/portfolio shift; and selecting the best of 32 variants on
the same held-out datasets incurs winner's-curse/test-selection bias. Report
all variants or use a nested model-selection design, add replicate seeds, and
include temporal/out-of-domain splits before making a production or thesis
claim about improvement.

## Rerun recommendation

For a scientifically clean comparison, rerun the **entire 64-trial training
grid and both evaluation tracks** after these fixes. The new run uses BF16,
fixed monitor samples, correct manifest/sentinel semantics, and exact eval
arrays; mixing new tasks 0–4 with the old FP16 sweep would complete the table
but not create one homogeneous experiment. If compute is temporarily limited,
rerunning missing PD tasks 0–4 and resuming unscored PD pairs can recover the
old sweep operationally, but it should be labelled a mixed-code diagnostic
run rather than the definitive experiment.
