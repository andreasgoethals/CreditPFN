"""CreditPFN data pipeline.

Four-stage sequence run from the repository root:

    python -m src.data.dedup --pass pre   # raw-corpus duplicate sweep
    python -m src.data.register           # build per-track manifests
    python -m src.data.sanitize           # surgical fixes + agnostic clean
    python -m src.data.dedup --pass post  # post-clean duplicate sweep

The pipeline ends at sanitized CSVs under ``data/processed/{track}/``.
There is no `.npz` chunking step — the training loop reads the
sanitized CSVs directly via ``src.train.dataloader``.

All scripts share ``config/data.yaml``.
"""
