"""Final-benchmark evaluation: TabPFN variants vs classical baselines.

Sub-modules
-----------
* :mod:`src.eval.benchmark` — load the training-manifest CSV, build
  the baseline + TabPFN-untuned set, then for every test chunk run
  every model with ``fit(X_context, y_context); predict(X_query)``
  and write a long-format comparison CSV.

CLI entry point
---------------
``python scripts/eval_pipeline.py``  (see that file).
"""

from src.eval.benchmark import (  # noqa: F401
    EvalRow, run_benchmark, load_trained_handles, resolve_test_datasets,
    find_existing_results,
    _method_dirname, _output_path_for,
)
from src.eval.dataset_loader import (  # noqa: F401
    ProcessedDataset, encode_for_model, load_processed_dataset, subsample,
)
