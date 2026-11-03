# Next steps for the second campaign (written 03/04-11-2026)

The authoritative grid is [`config/design.yaml`](../config/design.yaml). This page lists what is
done, what still needs a decision, and the order of work. Delete it once the campaign launches.

## Done (03-11-2026, night)

- First study archived locally in `archive/study-2026-11/` (results, figures, executed
  notebooks). Tracked notebooks are output-free.
- VSC: all CreditPFN outputs and trained checkpoints moved to quarantine folders
  (`/lustre1/project/stg_00211/CreditPFN/_to_delete_20261103/`,
  `$VSC_DATA/CreditPFN_to_delete_20261103/`). Base checkpoints, canonical data, runtime
  `env20261029f`, the old Conda environment and the Git checkout are kept.
- Local `.venv` = Python 3.13 + the accepted container's exact pins (`config/runtime/`).
- v2.6 LGD anomaly diagnosed (comb-shaped bucket density of the released regressor; not a
  pipeline bug). Upstream-parity tests and a roughness guard added.
- Implemented and tested, not yet decided: Muon arm, bucket-CRPS LGD objective, L2-SP on
  DiagScale corrections, FP32 DiagScale multiply under BF16, TabPFN FP32 evaluation switch.
- New defaults: 10,000 updates, query fraction 0.4, v2.6 cap 10,000, proportional PD sampling.

## Decisions needed from Andreas

1. **v2.6 LGD objective.** (a) Keep bar NLL for all models and report v2.6 LGD as "density
   reshaping" (current behaviour, known to cost ~0.05-0.08 R2); (b) use bucket CRPS for all
   TabPFN LGD runs (changes the reference objective; matches evaluation better); (c) bucket
   CRPS only for v2.6 (a declared per-model deviation); (d) drop v2.6 from LGD. Experiment 0
   can run a short v2.6 LGD pilot with both objectives to inform this.
2. **Retention panel** (`config/retention_panel_proposed.yaml`). TabArena-v0.1 has only 10
   regression tables below 25,000 rows and 17 binary classification tables after credit
   exclusions. Options: add OpenML-CTR23 regression tables; raise the row limit; accept 10 + 20.
   Confirm the borderline exclusions (two bankruptcy tables, "Is-this-a-good-customer").
3. **FP32 evaluation for all models** (`CREDITPFN_TABPFN_PRECISION=float32`): confirm, subject
   to the Experiment 0 capacity check at 50,000-row contexts.
4. **Optional arms:** Muon (32 runs, `design.yaml` `optional.muon`), top-quarter layers scope.
5. **LoRA L2-SP:** keep "not applicable" (current) or implement an effective-weight penalty.
6. **External credit test set:** any credit tables outside the 25-table corpus?
7. **Permanent deletion** of the VSC quarantine folders (one command per tier, after review):
   `rm -rf /lustre1/project/stg_00211/CreditPFN/_to_delete_20261103`
   `rm -rf "$VSC_DATA/CreditPFN_to_delete_20261103"`
   and locally `archive/_to_delete/` (old venv, review images). The local transfer backup
   `.codex/collection-final20261103a` could not be moved (permission denied).

## Order of work once decided

1. Freeze the code for Experiment 0. Runtime acceptance is keyed to the hash of all `src/`
   code, so every code change needs a fresh acceptance run before container jobs can start.
2. VSC (needs a valid certificate): `git pull --ff-only` in the DATA checkout; runtime
   acceptance for the new code identity (container vs preserved Conda environment).
3. Experiment 0 on VSC:
   - CPU: full test suite in the container, including `tests/test_upstream_parity.py`
     (container = local numbers), design/deviation tests, retention-panel overlap audit.
   - GPU: zero-learning-rate null controls and recovery/resume for all 4 models x 2 tasks;
     250-update pilots of the reference with dense monitoring (time and memory per point);
     capacity at the final caps (v2.6 at 10,000 rows) and FP32 evaluation at 50,000-row
     contexts; short pilots of DiagScale (BF16 + L2-SP), LoRA, Muon, E8 full pass and the
     v2.6 LGD objective comparison; record density roughness for every LGD table and model.
4. Launch the campaign by RQ (E1/E2/E4/E5 share the reference, then E3, E6-E10, then E11).
5. Rewrite the notebooks for the new experiment structure, adding classical baselines,
   absolute scores, seed agreement, training-table curves and retention correlation.

## Still to rewrite in the code (not done tonight)

- Slurm launchers and the benchmark/context dispatch for the slug-named experiment groups
  (`scripts/slurm/run_*.sh`, `_benchmark_dispatch.sh`, `_study_map.sh`). Output routing
  already accepts the slugs (`src/utils/paths.py` `CAMPAIGN2_GROUPS`).
- Retention-panel data staging for the new TabArena tables (`src/utils/prepare_retention`).
- Removal of first-study machinery (`legacy_*`, migration/study-layout code) once the
  archive is final.
