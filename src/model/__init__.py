"""Baseline + TabPFN models for the eval benchmark.

Every model in this package implements a small, common interface
(see :class:`src.model.base.BaselineModel`) so the eval pipeline
can iterate them uniformly:

    model.fit(X_context, y_context, categorical_idx)
    proba = model.predict_proba(X_query)        # classification
    pred  = model.predict(X_query)              # regression

The current roster:

* :mod:`src.model.tabpfn_models`     — TabPFN-untuned (the base
  checkpoint, no continued pretraining) and TabPFN-trained (any
  checkpoint produced by ``scripts/train_pipeline.py``).
* :mod:`src.model.boosting`          — XGBoost, CatBoost.
* :mod:`src.model.linear`            — LogReg (PD), LinReg (LGD).

Public registry — :func:`build_baselines` — yields, for a given
track, every classical baseline ready to be benchmarked. The eval
pipeline pairs that with the TabPFN variants from the training
manifest CSV.
"""

from src.model.base import BaselineModel, ModelHandle  # noqa: F401
from src.model.tabpfn_models import (  # noqa: F401
    TabPFNUntuned, TabPFNTrained,
)
from src.model.boosting import XGBoostModel, CatBoostModel  # noqa: F401
from src.model.linear import LogRegModel, LinRegModel  # noqa: F401
from src.model.registry import build_baselines  # noqa: F401
