"""CreditPFN data pipeline.

Two-stage sequence run from the repository root:

    python -m src.data.register           # build per-track manifests
    python -m src.data.sanitize           # surgical fixes + agnostic clean

The pipeline ends at sanitized CSVs under ``data/processed/{track}/``.
There is no `.npz` chunking step — the training loop reads the
sanitized CSVs directly via ``src.train.dataloader``.

All scripts share ``config/data.yaml``.
"""
