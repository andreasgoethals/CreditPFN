"""Shared optimizer anchoring and outer-fold isolation invariants."""
import numpy as np
import pytest
import torch


def test_accumulated_anchor_is_added_once_and_clipped_after_mean():
    from src.train.optimization import step_mean_gradient
    # Different chunk objectives, same parameter: averaging must keep lambda fixed.
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad(): model.weight.fill_(2.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    for target in (0., 1., 3.):
        loss = (model.weight-target).square().mean() + .003*(model.weight-1.).square().sum()
        loss.backward()
    expected_grad = np.mean([2*(2-target) for target in (0,1,3)]) + .006
    norm, ok = step_mean_gradient(model, optimizer, scaler, microbatches=3, max_norm=None)
    assert ok and norm == pytest.approx(expected_grad)
    assert model.weight.item() == pytest.approx(2-.1*expected_grad)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_outer_test_rows_never_enter_inner_fit_or_validation(task):
    from src.eval.benchmark import _inner_split, _make_outer_folds
    y = np.tile([0,1],100) if task == "classification" else np.linspace(0,1,200)
    observed = []
    for fold, (train, test) in enumerate(_make_outer_folds(y, task_type=task, n_folds=5, seed=99)):
        fit, validation = _inner_split(train, y[train], task_type=task, val_fraction=.2, seed=99+fold)
        assert set(fit).isdisjoint(validation) and set(fit).isdisjoint(test) and set(validation).isdisjoint(test)
        assert set(fit) | set(validation) == set(train)
        observed.extend(test)
    assert sorted(observed) == list(range(200))
