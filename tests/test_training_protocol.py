"""Regressions for the production sampler and optimizer, without foundation-model weights."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.train.corpus import DatasetRef
from src.train.dataloader import ProcessedDatasetLoader, identity_collate
from src.train.sampling import EpochSampler
from src.train.optimization import step_mean_gradient


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
