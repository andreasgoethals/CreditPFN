"""Regression objectives (window NLL, bucket CRPS) and the density-roughness guard."""
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


# ---- window_nll (reference LGD objective) ----------------------------------------------
def _edges(k=50):
    return torch.linspace(-2.5, 2.5, k + 1)


def test_window_nll_of_uniform_density_equals_bar_density():
    from src.train.regression_objectives import window_nll
    edges = _edges()
    logits = torch.zeros(7, 50)
    y = torch.linspace(-2.4, 2.4, 7)
    expected = torch.log(torch.tensor(5.0))  # total width 5 -> density 1/5 everywhere
    assert torch.allclose(window_nll(logits, edges, y), expected.expand(7), atol=1e-5)


def test_window_nll_ignores_a_comb_inside_the_window():
    """Alternate empty buckets (a comb) cost bar NLL a lot but the window barely notices."""
    from src.train.regression_objectives import window_nll
    edges = _edges()
    smooth = torch.zeros(1, 50)
    comb = torch.zeros(1, 50)
    comb[0, 1::2] = -9.0                                 # odd buckets ~1e-4 of the even ones
    y = torch.tensor([float(edges[25] + edges[26]) / 2])  # target in odd bucket 25
    smooth_loss, comb_loss = window_nll(smooth, edges, y), window_nll(comb, edges, y)
    assert float(comb_loss - smooth_loss) < 0.75       # density shrinks at most ~2x at this resolution
    bucket = torch.log_softmax(comb, -1)[0, 25] - torch.log(edges[26] - edges[25])
    assert float(-bucket - comb_loss) > 5                # single-bucket NLL is far worse


def test_window_nll_is_finite_and_clamped_at_the_edges():
    from src.train.regression_objectives import window_nll
    edges = _edges()
    logits = torch.randn(4, 50)
    y = torch.tensor([-10.0, -2.5, 2.5, 10.0])
    out = window_nll(logits, edges, y)
    assert torch.isfinite(out).all()


def test_window_nll_rejects_even_windows_and_bad_borders():
    from src.train.regression_objectives import window_nll
    with pytest.raises(ValueError):
        window_nll(torch.zeros(1, 50), _edges(), torch.zeros(1), window=4)
    with pytest.raises(ValueError):
        window_nll(torch.zeros(1, 50), _edges(49), torch.zeros(1))


def test_window_nll_close_to_bar_nll_for_smooth_density():
    from src.train.regression_objectives import window_nll
    edges = _edges(500)
    centres = (edges[1:] + edges[:-1]) / 2
    logits = -(centres ** 2).unsqueeze(0) / (2 * 0.5 ** 2)  # smooth Gaussian over buckets
    y = torch.tensor([0.3])
    k = int(torch.searchsorted(edges, y, right=True)) - 1
    bar = -(torch.log_softmax(logits, -1)[0, k] - torch.log(edges[k + 1] - edges[k]))
    assert abs(float(window_nll(logits, edges, y)) - float(bar)) < 0.01


def test_window_nll_ignores_missing_targets_and_rejects_oversized_windows():
    from types import SimpleNamespace

    from src.train.regression_objectives import regression_loss, window_nll
    edges = _edges()
    logits = torch.zeros(3, 1, 50)
    y = torch.tensor([[0.1], [float("nan")], [-0.4]])
    loss = regression_loss(logits, y, criterion=SimpleNamespace(borders=edges), objective="window_nll")
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="exceeds"):
        window_nll(torch.zeros(1, 3), torch.linspace(0, 1, 4), torch.zeros(1))
