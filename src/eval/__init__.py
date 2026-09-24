"""Held-out benchmarking and shared metrics, imported independently of training."""
from importlib import import_module

_BENCHMARK = {"EvalRow", "run_benchmark", "load_trained_handles", "resolve_test_datasets",
              "find_existing_results", "_method_dirname", "_output_path_for"}
_DATASET = {"ProcessedDataset", "encode_for_model", "load_processed_dataset", "subsample"}
__all__ = sorted(_BENCHMARK | _DATASET)


def __getattr__(name):
    if name in _BENCHMARK | _DATASET:
        module = import_module("src.eval.benchmark" if name in _BENCHMARK else "src.eval.dataset_loader")
        return getattr(module, name)
    raise AttributeError(name)
