"""Regressions for the production sampler and optimizer, without foundation-model weights."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.train.corpus import DatasetRef
from src.train.dataloader import ProcessedDatasetLoader, identity_collate
from src.train.sampling import EpochSampler, partition_rows
from src.train.optimization import step_mean_gradient


@pytest.mark.parametrize("members, queries, classes", [(1, 3, 2), (2, 17, 2), (4, 23, 5)])
@pytest.mark.parametrize("weighted, smoothing", [(False, 0.), (True, .1)])
def test_ensemble_classification_loss_preserves_objective_and_gradients(members, queries, classes, weighted, smoothing):
    from src.train.loop import _classification_loss_BE_LQ

    # Non-contiguous, class-first ensemble logits, including unused head columns.
    generator = torch.Generator().manual_seed(54)
    logits = torch.randn(members, queries, 10, generator=generator).transpose(1, 2).requires_grad_()
    targets = torch.randint(classes, (members, queries), generator=generator)
    weight = torch.arange(1., 11.) if weighted else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)
    expected = criterion(logits, targets)
    expected_grad, = torch.autograd.grad(expected, logits)

    # The spatial NLL reduction is forbidden by strict CUDA determinism. A CPU
    # fixture can enforce the same input contract without claiming GPU coverage.
    def nonspatial_loss(values, labels):
        assert values.ndim == 2 and labels.ndim == 1, "spatial NLL kernel would be selected"
        return criterion(values, labels)

    actual = _classification_loss_BE_LQ(logits, targets, n_classes=classes, criterion=nonspatial_loss)
    actual_grad, = torch.autograd.grad(actual, logits)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert (actual_grad[:, classes:] != 0).any(), "Keep all ten output columns in the loss"


def make_dataset(tmp_path):
    refs = []
    for i, n in enumerate((240, 80, 160)):
        path = tmp_path / f"synthetic{i}.csv"
        pd.DataFrame({"x": np.arange(n), "y": np.arange(n) % 2}).to_csv(path, index=False)
        refs.append(DatasetRef(f"synthetic{i}", "pd", "classification", "y", (), path, n))
    return ProcessedDatasetLoader(refs, max_rows_per_epoch=80, query_fraction=.4,
                                  seed=11, pass_mode="accumulate")


def test_persistent_workers_get_new_epoch_and_match_serial(tmp_path):
    ds = make_dataset(tmp_path)
    sampler = EpochSampler(ds, seed=11, accumulate=True)
    loader = DataLoader(ds, sampler=sampler, batch_size=1, collate_fn=identity_collate,
                        num_workers=1, multiprocessing_context="spawn", persistent_workers=True)
    first = None
    try:
        for epoch in (0, 1, 2):
            ds.set_epoch(epoch)
            sampler.set_epoch(epoch)
            parallel = list(loader)
            serial = [ds[index] for index in sampler]
            for p, s in zip(parallel, serial):
                assert p.dataset_id == s.dataset_id
                assert torch.equal(p.X_context, s.X_context)
                assert torch.equal(p.y_query, s.y_query)
            sample = next(p.X_context for p in parallel if p.dataset_id == "synthetic0")
            if first is None:
                first = sample
            else:
                assert not torch.equal(sample, first)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_accumulation_updates_never_mix_datasets(tmp_path):
    ds = make_dataset(tmp_path)
    sampler = EpochSampler(ds, seed=4, accumulate=True)
    for epoch in range(4):
        sampler.set_epoch(epoch)
        group = set()
        updates = []
        for (_, idx), end in zip(sampler, sampler.dataset_end_flags):
            group.add(ds._plan[idx][0])
            if end:
                assert len(group) == 1
                updates.extend(group)
                group.clear()
        assert sorted(updates) == [0, 1, 2]
        assert sorted(sampler.order) == list(range(len(ds)))


@pytest.mark.parametrize("microbatches", [1, 2, 7])
def test_microbatch_mean_matches_one_large_batch_before_clipping(microbatches):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for _ in range(microbatches):
        (model(torch.ones(1, 1)).square().mean()).backward()
    norm, stepped = step_mean_gradient(model, optimizer, scaler,
                                       microbatches=microbatches, max_norm=1.)
    assert norm == pytest.approx(2.)
    assert stepped
    assert model.weight.item() == pytest.approx(.9)


def test_nonfinite_bf16_style_gradient_does_not_corrupt_weights():
    model = torch.nn.Linear(1, 1, bias=False)
    original = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    _, stepped = step_mean_gradient(model, optimizer, scaler, microbatches=1, max_norm=1.)
    assert not stepped
    assert torch.equal(original, model.weight)


@pytest.mark.parametrize("classification", [False, True])
def test_row_partition_coverage_cap_balance_and_fresh_epochs(classification):
    # A naive cap-sized slice would leave a singleton. Include a rare class.
    y = np.array([0] * 75 + [1] * 25 + [2]) if classification else np.arange(101) / 10
    chunks = partition_rows(y, row_cap=20, rng=np.random.default_rng(7), classification=classification)
    repeated = partition_rows(y, row_cap=20, rng=np.random.default_rng(7), classification=classification)
    changed = partition_rows(y, row_cap=20, rng=np.random.default_rng(8), classification=classification)
    assert len(chunks) == 6
    assert np.array_equal(np.sort(np.concatenate(chunks)), np.arange(len(y)))
    assert 2 <= min(map(len, chunks)) <= max(map(len, chunks)) <= 20
    assert max(map(len, chunks)) - min(map(len, chunks)) <= 1
    assert all(np.array_equal(a, b) for a, b in zip(chunks, repeated))
    assert any(set(a) != set(b) for a, b in zip(chunks, changed))
    if classification:
        for label in np.unique(y):
            counts = [np.count_nonzero(y[chunk] == label) for chunk in chunks]
            assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("n, cap", [(1, 20), (5, 1), (3, 2)])
def test_partition_refuses_impossible_context_query_chunks(n, cap):
    with pytest.raises(ValueError):
        partition_rows(np.arange(n), row_cap=cap, rng=np.random.default_rng(0), classification=False)


@pytest.mark.parametrize("family", ["legacy", "tabpfn", "tabicl"])
def test_all_builders_honor_disjoint_partitions_and_same_modes(tmp_path, monkeypatch, family):
    from src.train import tabpfn_preprocessing, tabicl_compat
    n = 101
    path = tmp_path / 'rows.csv'
    pd.DataFrame({'row_id': np.arange(n), 'y': np.arange(n) % 5 == 0}).to_csv(path, index=False)
    ref = DatasetRef('synthetic', 'pd', 'classification', 'y', (), path, n)

    def fake_clean(**kwargs):
        return SimpleNamespace(X_clean=kwargs['X_full_df'].to_numpy(), y=kwargs['y_full'],
                               feature_schema=None, ensemble_configs=None, outlier_removal_std=None)

    def fake_ensemble(**kwargs):
        return np.concatenate([kwargs['X_ctx'][:, 0], kwargs['X_qry'][:, 0]])

    def fake_meta(X, y, **kwargs):
        split = len(X) - kwargs['query_size']
        return SimpleNamespace(X=torch.tensor(X).unsqueeze(0), y_train=torch.tensor(y[:split]),
                               y_query=torch.tensor(y[split:]), train_size=split,
                               y_scaler_mean=None, y_scaler_std=None)

    monkeypatch.setattr(tabpfn_preprocessing, 'clean_loaded_dataset', fake_clean)
    monkeypatch.setattr(tabpfn_preprocessing, 'build_ensemble_members', fake_ensemble)
    monkeypatch.setattr(tabicl_compat, 'import_tabicl_finetune_data', lambda: (None, fake_meta))
    loaders = [ProcessedDatasetLoader([ref], max_rows_per_epoch=20, query_fraction=.4, seed=9,
                pass_mode=mode, model_family='tabicl' if family == 'tabicl' else 'tabpfn',
                inference_config=object() if family == 'tabpfn' else None)
               for mode in ('full_pass', 'accumulate')]
    epochs = []
    for epoch in (0, 1, 0):  # out-of-order requests must reproduce earlier partitions too
        modes = []
        for loader in loaders:
            chunks = []
            for idx in range(len(loader)):
                batch = loader[(epoch, idx)]
                if family == 'legacy':
                    rows = torch.cat([batch.X_context, batch.X_query])[:, 0, 0].numpy()
                elif family == 'tabicl':
                    rows = batch.X[0, :, 0].numpy()
                else:
                    rows = batch
                chunks.append(rows)
            assert np.array_equal(np.sort(np.concatenate(chunks)), np.arange(n))
            assert max(map(len, chunks)) <= 20
            assert len(loader._row_partitions) == 1
            modes.append(chunks)
        assert all(np.array_equal(a, b) for a, b in zip(*modes))
        epochs.append(modes[0])
    assert all(np.array_equal(a, b) for a, b in zip(epochs[0], epochs[2]))
    assert any(set(a) != set(b) for a, b in zip(epochs[0], epochs[1]))


def test_full_pass_rejects_balancing_and_unknown_modes(tmp_path):
    refs = make_dataset(tmp_path).refs
    for kwargs in ({'pass_mode': 'full_pass', 'context_sampling': 'balanced'},
                   {'pass_mode': 'typo'}, {'context_sampling': 'typo'}):
        with pytest.raises(ValueError):
            ProcessedDatasetLoader(refs, max_rows_per_epoch=80, query_fraction=.4, **kwargs)
