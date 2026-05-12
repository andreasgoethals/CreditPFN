"""Continued-pretraining pipeline for TabPFN (v2.5/v2.6/v3) on the credit-risk corpus.

Sub-modules
-----------
* :mod:`src.train.corpus`     — dataset-level train/val/test split.
* :mod:`src.train.dataloader` — torch ``Dataset`` wrapping the cached
  ``.npz`` chunks; per-step ctx/query resplit + subsampling.
* :mod:`src.train.model`      — load a TabPFN checkpoint into a
  ``PerFeatureTransformer`` and save it back after finetuning.
* :mod:`src.train.metrics`    — ROC-AUC / log_loss / neg-NLL / RMSE
  averaged across val datasets.
* :mod:`src.train.loop`       — one-config training + validation + early
  stopping + best-checkpoint save.
CLI entry point
---------------
``python scripts/train_pipeline.py``  (see that file).
"""
