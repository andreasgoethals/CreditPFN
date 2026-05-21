"""Continued-pretraining pipeline for TabPFN (v2.6/v3) on the credit-risk corpus.

Sub-modules
-----------
* :mod:`src.train.corpus`     — dataset-level train/test split.
* :mod:`src.train.dataloader` — torch ``Dataset`` that reads sanitized
  CSVs directly and draws a fresh random subsample every epoch.
* :mod:`src.train.model`      — load a TabPFN checkpoint into a
  ``PerFeatureTransformer`` and save it back after finetuning.
* :mod:`src.train.metrics`    — ROC-AUC / log_loss / neg-NLL / RMSE
  averaged across train + test datasets at end of each epoch.
* :mod:`src.train.loop`       — one-config training + per-epoch monitor
  + final-checkpoint save (no validation set, no early stopping).

CLI entry point
---------------
``python scripts/train_pipeline.py``  (see that file).
"""
