"""Conservative rebinning for the pinned TabPFN regressor's predictive distribution."""
from __future__ import annotations

from functools import lru_cache

import torch


def translate_bar_mass(logits, *, frm, to, chunk_budget_elements=1_000_000):
    """Redistribute probability by interval overlap, including both open-ended tails.

    The upstream CDF subtraction drops rare bins near cumulative probability one.
    Here every contribution is nonnegative. The two edge bins collect all exterior
    mass, matching the endpoint convention in the pinned regressor.
    """
    source = frm.double()
    target = to.double()
    if torch.equal(source, target):
        flat = logits.reshape(-1, logits.shape[-1])
        result = torch.empty_like(flat, dtype=torch.float64)
        chunk = max(1, chunk_budget_elements // flat.shape[-1])
        for start in range(0, len(flat), chunk):
            result[start:start + chunk] = flat[start:start + chunk].double().softmax(-1)
        return result.reshape(logits.shape)
    if not (torch.isfinite(source).all() and torch.isfinite(target).all()
            and (source.diff() >= 0).all() and (target.diff() > 0).all()):
        raise ValueError("TabPFN prediction requires finite ordered source and target borders")
    # Split the source bars at every destination boundary. Each resulting interval
    # contributes to exactly one output bar, including intervals outside the new support.
    cuts = torch.cat((source, target[1:-1].clamp(source[0], source[-1]))).sort().values
    widths = cuts.diff()
    left = cuts[:-1][widths > 0]
    source_index = torch.searchsorted(source, left, right=True) - 1
    dest_index = torch.searchsorted(target[1:-1].contiguous(), left, right=True)
    fractions = widths[widths > 0] / source.diff()[source_index]
    # A target transform can collapse a source bin to a point without cancelling
    # its logit. Retain that atom on the right of a coincident internal boundary.
    collapsed = torch.where(source.diff() == 0)[0]
    source_index = torch.cat((source_index, collapsed))
    dest_index = torch.cat((dest_index, torch.searchsorted(target[1:-1].contiguous(), source[collapsed], right=True)))
    fractions = torch.cat((fractions, torch.ones_like(collapsed, dtype=torch.float64)))
    flat = logits.reshape(-1, logits.shape[-1])
    result = torch.empty((len(flat), len(target) - 1), dtype=torch.float64, device=flat.device)
    chunk = max(1, chunk_budget_elements // max(len(source_index), logits.shape[-1]))
    for start in range(0, len(flat), chunk):
        probabilities = flat[start:start + chunk].double().softmax(-1)
        out = result[start:start + chunk].zero_()
        out.scatter_add_(1, dest_index.expand(len(out), -1), probabilities[:, source_index] * fractions)
    return result.reshape(*logits.shape[:-1], len(target) - 1)


@lru_cache(maxsize=1)
def _stable_predictor():
    """Keep sklearn's pinned predict workflow and replace its local rebinner only.

    A private function namespace leaves every other estimator and concurrent
    prediction untouched. Reapply the public method's sklearn config context.
    """
    from inspect import unwrap
    from types import FunctionType

    from sklearn import config_context
    from tabpfn import TabPFNRegressor

    public = unwrap(TabPFNRegressor.predict)
    if "translate_probs_across_borders" not in public.__code__.co_names:
        raise RuntimeError("Review the density adapter for this version of TabPFN")
    namespace = dict(public.__globals__, translate_probs_across_borders=translate_bar_mass)
    predict = FunctionType(public.__code__, namespace, public.__name__, public.__defaults__, public.__closure__)
    predict.__kwdefaults__ = public.__kwdefaults__
    return config_context(transform_output="default")(predict)


def predict_full(model, X, **kwargs):
    from tabpfn import TabPFNRegressor

    if isinstance(model, TabPFNRegressor) and type(model).predict is TabPFNRegressor.predict:
        return _stable_predictor()(model, X, output_type="full", **kwargs)
    return model.predict(X, output_type="full", **kwargs)


def predict_mean(model, X):
    """Use the same distribution for benchmark, monitoring and context/fine-tuning tasks."""
    from tabpfn import TabPFNRegressor

    estimator = getattr(model, "finetuned_inference_regressor_", model)
    if isinstance(estimator, TabPFNRegressor) and type(estimator).predict is TabPFNRegressor.predict:
        return _stable_predictor()(estimator, X, output_type="mean")
    return model.predict(X)
