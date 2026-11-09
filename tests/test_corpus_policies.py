"""Second-campaign corpus policies: table size (E9/E10) and single-source transfer (E12)."""
from types import SimpleNamespace

import pytest

from src.train import corpus as c


def _ref(name, rows):
    return SimpleNamespace(dataset_id=name, n_rows=rows)


def _split(train, test=("held",)):
    return c.CorpusSplit(train=list(train), test=[_ref(t, 1) for t in test])


TRAIN = [_ref("a", 500), _ref("b", 9_000), _ref("c", 10_000), _ref("d", 10_001), _ref("e", 50_000),
         _ref("f", 200_000), _ref("g", 12_000)]


def test_small_only_keeps_tables_at_or_below_the_threshold():
    out = c.apply_size_policy(_split(TRAIN), "small_only")
    assert [r.dataset_id for r in out.train] == ["a", "b", "c"]
    assert [r.dataset_id for r in out.test] == ["held"]


def test_large_matched_draws_as_many_large_tables_as_there_are_small_ones():
    out = c.apply_size_policy(_split(TRAIN), "large_matched", seed=7)
    assert len(out.train) == 3 and all(r.n_rows > 10_000 for r in out.train)
    again = c.apply_size_policy(_split(TRAIN), "large_matched", seed=7)
    assert [r.dataset_id for r in out.train] == [r.dataset_id for r in again.train]   # deterministic
    other = {tuple(r.dataset_id for r in c.apply_size_policy(_split(TRAIN), "large_matched", seed=s).train)
             for s in range(20)}
    assert len(other) > 1                                                               # seed matters


def test_policies_never_touch_held_out_tables_and_reject_unknown_values():
    assert c.apply_size_policy(_split(TRAIN), None).train == TRAIN
    with pytest.raises(ValueError):
        c.apply_size_policy(_split(TRAIN), "medium_only")
    with pytest.raises(ValueError, match="small training table"):
        c.apply_size_policy(_split([_ref("x", 50_000)]), "small_only")
    with pytest.raises(ValueError, match="Too few large"):
        c.apply_size_policy(_split([_ref("x", 5), _ref("y", 6), _ref("z", 50_000)]), "large_matched")


def test_single_source_trains_on_one_table_and_tests_on_all_others(monkeypatch):
    pool = [_ref(n, 1000) for n in ("t2", "t0", "t1")]
    monkeypatch.setattr(c, "build_dataset_pool", lambda track: pool)
    split = c.single_source_split("pd", 1)
    assert [r.dataset_id for r in split.train] == ["t1"]
    assert sorted(r.dataset_id for r in split.test) == ["t0", "t2"]
    with pytest.raises(ValueError):
        c.single_source_split("pd", 3)


@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_real_partitions_support_the_policies(track):
    """On the real registry: PD can be split by size in every partition; held-out tables unchanged."""
    from pathlib import Path

    from src.train.config import load_train_config
    from src.utils.experiment import apply_split_index
    path = Path("config/experiments/E00_pilots") / f"reference_{track}.yaml"
    base = load_train_config(config_path=str(path))
    base.corpus.n_splits = 4
    if not c.build_dataset_pool(track):
        pytest.skip("processed data unavailable")
    for k in range(4):
        cfg = apply_split_index(base.copy(), k)
        full = c.split_from_cfg(cfg)
        if track == "lgd":
            continue
        for policy in c.SIZE_POLICIES:
            cfg.corpus.train_size_policy = policy
            out = c.split_from_cfg(cfg)
            assert [r.dataset_id for r in out.test] == [r.dataset_id for r in full.test]
            assert set(r.dataset_id for r in out.train) <= set(r.dataset_id for r in full.train)
            assert len(out.train) == sum(r.n_rows <= c.SIZE_THRESHOLD for r in full.train)
        cfg.corpus.train_size_policy = None
