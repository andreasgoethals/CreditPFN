"""Numerical preparation must preserve finite data and fit clipping on context only."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.sanitize import _cast_numericals_to
from src.train.tabpfn_preprocessing import apply_outlier_clip


def test_float32_preparation_preserves_extreme_finite_values():
    source = pd.DataFrame({"large": [-9e41, -2e37, 0., 4e28, 7e41, np.nan],
                           "ordinary": [1., 2., 3., 4., 5., 6.], "target": [0, 1, 0, 1, 0, 1]})
    result = _cast_numericals_to(source.copy(), "target", ["large", "ordinary"], "float32")
    assert np.array_equal(result.large.notna(), source.large.notna())
    assert np.all(np.diff(result.large.dropna()) > 0)
    assert np.max(np.abs(result.large.dropna())) <= 1
    np.testing.assert_array_equal(result.ordinary, source.ordinary)
    pd.testing.assert_series_equal(result.target, source.target)


def test_clipping_preserves_values_inside_the_fitted_bounds():
    values = torch.tensor([-2., -1., 0., 1., 2.]).reshape(-1, 1, 1)
    assert torch.equal(apply_outlier_clip(values, n_sigma=12.), values)


def test_query_values_do_not_change_context_clipping():
    context = torch.tensor([-2., -1., 0., 1., 2.]).reshape(-1, 1, 1)
    outputs = [apply_outlier_clip(torch.cat([context, torch.tensor([[[query]]])]),
                                 n_sigma=2., context_rows=len(context)) for query in (10., 1e6)]
    assert torch.equal(outputs[0][:-1], outputs[1][:-1])
    assert outputs[1][-1].item() < 30
    assert torch.equal(outputs[0][:-1], context)


def test_unit_changes_are_recorded_and_preserve_missingness_categories_and_target():
    shifts = {}
    source = pd.DataFrame({"large": [1e41, -1e41, np.nan, np.inf, -np.inf],
                           "category": [1, 2, 1, 2, 1], "target": [1e41] * 5})
    result = _cast_numericals_to(source.copy(), "target", ["large", "target"], "float32", unit_shifts=shifts)
    assert shifts == {"large": int(np.frexp(1e41)[1])}
    np.testing.assert_allclose(np.ldexp(result.large[:2].astype(float), shifts["large"]), source.large[:2], rtol=1e-7)
    assert result.large[2:].isna().all()
    pd.testing.assert_series_equal(result.category, source.category)
    pd.testing.assert_series_equal(result.target, source.target)


def test_unit_change_rejects_unrepresentable_dynamic_range():
    with pytest.raises(ValueError, match="too wide"):
        _cast_numericals_to(pd.DataFrame({"x": [1e300, 1e-30]}), "y", ["x"], "float32")


def test_data_preparation_persists_unit_shifts_in_the_existing_manifest(tmp_path, monkeypatch):
    import json
    from omegaconf import OmegaConf
    from src.data import register, sanitize

    cfg = OmegaConf.load("config/data.yaml")
    cfg.paths.raw = str(tmp_path / "raw")
    cfg.paths.processed = str(tmp_path / "processed")
    cfg.paths.manifest_pd = str(tmp_path / "manifest_pd.csv")
    cfg.paths.manifest_lgd = str(tmp_path / "manifest_lgd.csv")
    raw = tmp_path / "raw" / "pd"
    raw.mkdir(parents=True)
    frame = pd.DataFrame({"x": [1e41, 2e41, 3e41, np.nan], "y": [0, 1, 0, 1]})
    frame.to_csv(raw / "synthetic.csv", index=False)
    metadata = {"synthetic": {"track": "pd", "task_type": "classification", "target_column": "y",
                              "categorical_columns": [], "source": "synthetic", "source_url": None}}
    for module in (register, sanitize):
        monkeypatch.setattr(module, "DATASET_METADATA", metadata)
        monkeypatch.setattr(module, "apply_dataset_specific_fixes", lambda df, dataset_id: df)
    assert register.main(cfg) == 0
    assert sanitize.main(cfg) == 0
    manifest = pd.read_csv(cfg.paths.manifest_pd)
    shifts = json.loads(manifest.loc[0, "numeric_unit_shifts"])
    assert shifts["x"] == int(np.frexp(3e41)[1])
    processed = pd.read_csv(tmp_path / "processed" / "pd" / "synthetic.sanitized.csv")
    assert processed.x.notna().tolist() == frame.x.notna().tolist()
    np.testing.assert_allclose(np.ldexp(processed.x[:3], shifts["x"]), frame.x[:3], rtol=1e-7)
    assert processed.y.tolist() == frame.y.tolist()


def test_clipping_uses_second_pass_sample_variance_and_preserves_categoricals():
    # The initial 1-sigma filter removes +/-100. The second fit sees [-1,0,1],
    # with mean 0 and sample std 1, so the final bounds are exactly [-1,1].
    x = torch.tensor([-100., -1., 0., 1., 100., 10.]).reshape(-1, 1, 1).repeat(1, 1, 2)
    original = x.clone()
    result = apply_outlier_clip(x, n_sigma=1., context_rows=5, categorical_idx=[1])
    assert result[-1, 0, 0].item() == pytest.approx(1 + np.log(11), rel=1e-6)
    assert result[0, 0, 0].item() == pytest.approx(-1 - np.log(101), rel=1e-6)
    assert torch.equal(result[:, :, 1], original[:, :, 1])
    assert torch.equal(x, original)


def test_clipping_preserves_all_nan_columns_without_creating_new_missing_values():
    x = torch.tensor([[float("nan"), 1.], [float("nan"), 2.], [float("nan"), 3.]]).unsqueeze(1)
    result = apply_outlier_clip(x, n_sigma=12., context_rows=2)
    assert torch.equal(torch.isnan(result), torch.isnan(x))
    torch.testing.assert_close(result, x, equal_nan=True)


def test_clipping_refuses_overflow_instead_of_erasing_finite_inputs():
    x = torch.full((16, 1, 1), 3e38)
    with pytest.raises(ValueError, match="rebuild processed data"):
        apply_outlier_clip(x, n_sigma=12., context_rows=10)


@pytest.mark.parametrize("values", [[-1], [0.5], [float("nan")], [float("inf")], []])
def test_debugging_audit_rejects_invalid_skip_measurements(values):
    from src.utils.audit_experiment import numerical_skip_counts

    with pytest.raises(ValueError, match="measurements"):
        numerical_skip_counts(pd.DataFrame({"data_skipped_steps": values, "amp_skipped_steps": values}))
