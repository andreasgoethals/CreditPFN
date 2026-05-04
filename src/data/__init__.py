"""CreditPFN data pipeline.

Five-stage sequence run from the repository root:

    python -m src.data.dedup --pass pre   # raw-corpus duplicate sweep
    python -m src.data.register           # build per-track manifests
    python -m src.data.sanitize           # surgical fixes + agnostic clean
    python -m src.data.dedup --pass post  # post-clean duplicate sweep
    python -m src.data.dataset            # chunk and cache to .npz

All scripts share ``config/data.yaml``.
"""
