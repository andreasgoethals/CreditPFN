import torch

from src.model.tabpfn_density import translate_bar_mass


def test_preserves_small_tail_probabilities_and_mass_on_shifted_grids():
    from tabpfn.utils import translate_probs_across_borders
    source = torch.tensor([0., 1., 2., 3.])
    target = torch.tensor([-.3, 1.5, 2.5, 3.2])
    logits = torch.tensor([[0., -25., -50.], [-.1, .7, .6]])
    old = translate_probs_across_borders(logits, frm=source, to=target)
    assert old[0, 2] == 0
    stable = translate_bar_mass(logits, frm=source, to=target, chunk_budget_elements=1)
    probabilities = logits.double().softmax(-1)
    expected = probabilities @ torch.tensor([[1., 0., 0.], [.5, .5, 0.], [0., .5, .5]], dtype=torch.float64)
    torch.testing.assert_close(stable, expected, rtol=1e-13, atol=0)
    torch.testing.assert_close(stable.sum(-1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(stable[1], old[1].double(), rtol=1e-6, atol=1e-7)


def test_preserves_collapsed_bins_and_folds_outside_mass_into_edge_bins():
    source = torch.tensor([-1., 0., 0., 1., 2.])
    target = torch.tensor([-.5, 0., .5])
    mass = translate_bar_mass(torch.zeros(4), frm=source, to=target)
    torch.testing.assert_close(mass, torch.tensor([.25, .75], dtype=torch.float64))
