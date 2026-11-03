"""Bucket CRPS and the density-roughness guard (v2.6 regressor diagnosis, 03-11-2026)."""
import pytest
import torch

from src.train.density_checks import ROUGHNESS_LIMIT, bucket_roughness, check_regression_objective
from src.train.regression_objectives import bucket_crps, regression_loss

BORDERS = torch.linspace(-4, 4, 401)          # 400 buckets of width 0.02


def point_mass(index, k=400):
    logits = torch.full((k,), -1e4)
    logits[index] = 0.0
    return logits


def test_crps_of_a_point_mass_is_the_distance_to_the_target():
    for index, y in ((200, 0.5), (100, -1.3), (350, 3.0)):
        centre_right = BORDERS[index + 1]
        loss = bucket_crps(point_mass(index), BORDERS, torch.tensor(y))
        assert float(loss) == pytest.approx(abs(float(centre_right) - y), abs=0.03)


def test_crps_is_minimised_by_mass_on_the_target_bucket():
    y = torch.tensor(0.31)
    target = int(torch.searchsorted(BORDERS, y)) - 1
    losses = torch.stack([bucket_crps(point_mass(i), BORDERS, y) for i in range(400)])
    assert int(losses.argmin()) in (target, target + 1)


def test_crps_barely_reacts_to_a_comb_that_nll_punishes():
    torch.manual_seed(0)
    smooth = -((BORDERS[:-1] + 0.01) ** 2) / 2
    comb = smooth.clone()
    comb[1::2] -= 8.0                         # every other bucket nearly empty
    y = torch.tensor(0.035)                   # bucket 201: an emptied (odd) bucket
    nll = lambda logits: -(logits.log_softmax(-1)[int(torch.searchsorted(BORDERS, y)) - 1])
    assert float(nll(comb) - nll(smooth)) > 5.0
    assert abs(float(bucket_crps(comb, BORDERS, y) - bucket_crps(smooth, BORDERS, y))) < 0.02


def test_targets_outside_the_borders_are_penalised_linearly():
    logits = point_mass(399)
    assert float(bucket_crps(logits, BORDERS, torch.tensor(6.0))) == pytest.approx(2.0, abs=0.05)


def test_regression_loss_rejects_unknown_objectives():
    with pytest.raises(ValueError, match="Unknown regression objective"):
        regression_loss(torch.zeros(1, 1, 400), torch.zeros(1, 1), criterion=None, objective="mse")


def test_roughness_separates_a_comb_from_a_smooth_density():
    smooth = (-((BORDERS[:-1]) ** 2) / 2).softmax(-1)
    comb_logits = -((BORDERS[:-1]) ** 2) / 2
    comb_logits[1::2] -= 6.0
    assert float(bucket_roughness(smooth)) < 0.1
    assert float(bucket_roughness(comb_logits.softmax(-1))) > ROUGHNESS_LIMIT


def test_bar_nll_is_refused_for_a_comb_density_unless_declared():
    comb_logits = -((BORDERS[:-1]) ** 2) / 2
    comb_logits[1::2] -= 6.0
    logits = comb_logits.expand(8, -1)
    with pytest.raises(ValueError, match="roughness"):
        check_regression_objective(logits, objective="bar_nll", base="v2.6")
    assert check_regression_objective(logits, objective="bucket_crps", base="v2.6") > ROUGHNESS_LIMIT
    assert check_regression_objective(logits, objective="bar_nll", base="v2.6", allow_rough_density=True)
