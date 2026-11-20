"""An interrupted E5 fit must return the same held-out predictions as a fresh fit."""
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.eval.finetune_recovery import InterruptedEpoch
from src.eval.target_finetune import finetune_cell
from src.utils.paths import checkpoints_dir


@pytest.mark.slow
@pytest.mark.parametrize("family,name", [
    ("tabpfn", "tabpfn-v2-classifier-v2_default.ckpt"),
    ("tabicl", "tabicl-classifier-v2-20260212.ckpt"),
])
def test_resume_inside_real_candidate(family, name, tmp_path):
    path = checkpoints_dir(name)
    if not path.is_file():
        pytest.skip("base checkpoint unavailable")
    X = np.random.default_rng(31).normal(size=(192, 5))
    data = SimpleNamespace(X=pd.DataFrame(X), y=(X[:, 0] > 0).astype(int),
                           task_type="classification", categorical_columns=[])
    def fit(**extra):
        torch.manual_seed(17)
        np.random.seed(17)
        random.seed(17)
        return finetune_cell(data, np.arange(128), np.arange(128, 160), np.arange(160, 192),
                             path, family, recipe={"epochs": 3}, device="cpu", views=1, **extra)
    original, native_arrays = fit()
    reference = tmp_path / "reference.pt"
    expected, arrays = fit(recovery_path=reference)
    latest = tmp_path / "epoch.pt"
    with pytest.raises(InterruptedEpoch, match="Saved epoch 1"):
        fit(recovery_path=latest, stop_after_epoch=1)
    actual, resumed = fit(recovery_path=latest)
    assert actual["resumed_at_epoch"] == 1
    assert actual["optimizer_steps"] == expected["optimizer_steps"] == original["optimizer_steps"] > 0
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], native_arrays[key])
        np.testing.assert_array_equal(resumed[key], arrays[key])
    fresh = torch.load(reference, weights_only=False)
    restored = torch.load(latest, weights_only=False)
    for key in ("model", "optimizer", "scheduler", "best", "patience", "global_step", "torch_rng"):
        torch.testing.assert_close(restored[key], fresh[key], rtol=0, atol=0)
