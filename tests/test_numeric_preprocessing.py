"""Numerical preparation must preserve finite data and fit clipping on context only."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.sanitize import _cast_numericals_to
from src.train.tabpfn_preprocessing import apply_outlier_clip


def test_regression_query_uses_fitted_member_transform_without_refitting(monkeypatch):
    import copy
    import sys
    from types import ModuleType, SimpleNamespace
    from sklearn.preprocessing import PowerTransformer
    from src.train.tabpfn_preprocessing import build_ensemble_members

    # Exercise our builder against the upstream member contract. The fitted
    # member holds a separate config: the cached template is not fitted.
    schema = SimpleNamespace(indices_for=lambda modality: [])
    configs = [SimpleNamespace(target_transform=None),
               SimpleNamespace(target_transform=PowerTransformer())]

    class Preprocessor:
        def __init__(self, **kwargs):
            self.configs = copy.deepcopy(kwargs["configs"])

        def fit_transform_ensemble_members(self, *, X_train, y_train):
            result = []
            for config in self.configs:
                y = y_train if config.target_transform is None else config.target_transform.fit_transform(y_train[:, None]).ravel()
                result.append(SimpleNamespace(config=config, X_train=X_train, y_train=y,
                              feature_schema=schema, transform_X_test=lambda X: X))
            return result

    preprocessing = ModuleType("tabpfn.preprocessing")
    preprocessing.FeatureSubsamplingMethod = lambda value: value
    ensemble = ModuleType("tabpfn.preprocessing.ensemble")
    ensemble.TabPFNEnsemblePreprocessor = Preprocessor
    datamodel = ModuleType("tabpfn.preprocessing.datamodel")
    datamodel.FeatureModality = SimpleNamespace(CATEGORICAL="categorical")
    for module in (preprocessing, ensemble, datamodel):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    inference = SimpleNamespace(FEATURE_SUBSAMPLING_METHOD="random",
        FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT=50, SUBSAMPLE_SAMPLES=None,
        FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT="auto")
    context = np.array([0., .2, .3, .9, 1.6, 3.])
    query = np.array([.1, 5.])
    batch = build_ensemble_members(X_ctx=np.ones((6, 2)), y_ctx_raw=context,
        X_qry=np.ones((2, 2)), y_qry_raw=query, feature_schema=schema,
        ensemble_configs=configs, outlier_removal_std=None, task_type="regression",
        n_classes=None, inference_config=inference, n_estimators=2, rng_seed=42,
        dataset_id="synthetic_regression")
    context_z = ((context - context.mean()) / context.std()).astype(np.float32)
    query_z = ((query - context.mean()) / context.std()).astype(np.float32)
    transform = PowerTransformer().fit(context_z[:, None])
    expected = transform.transform(query_z[:, None]).ravel()
    np.testing.assert_allclose(batch.members[0].y_query.flatten(), query_z)
    np.testing.assert_allclose(batch.members[1].y_query.flatten(), expected, rtol=1e-6)
    np.testing.assert_allclose(batch.y_query.flatten(), query_z)
    for original, moved in zip(batch.members, batch.to("cpu").members):
        torch.testing.assert_close(original.y_query, moved.y_query)
    assert not hasattr(configs[1].target_transform, "lambdas_"), "Fit only the member's copy"


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
