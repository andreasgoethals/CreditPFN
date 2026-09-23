"""Version-tolerant TabICLv2 imports and checkpoint-family detection.

Training uses private upstream finetuning APIs, isolated here and checked by
smoke_test. The tabicl[finetune] dependency supplies their required extras; the
prepared plan records the actual installed package version.

Filenames containing 'tabicl' select this family. Frozen-backbone adaptation
uses src.train.freeze, which freezes the largest repeated transformer stack
and leaves all other modules trainable. It is not the upstream stage-3 freeze
or head-only training. Old _iclhead filenames remain readable as historical data.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

#: HF Hub filenames of the v2 checkpoints (repo `jingang/TabICL`). Also used
#: as the expected local basenames under staging `checkpoints/`.
TABICL_V2_CLASSIFIER = "tabicl-classifier-v2-20260212.ckpt"
TABICL_V2_REGRESSOR = "tabicl-regressor-v2-20260212.ckpt"


def model_family(base_path: str | Path) -> str:
    """``"tabicl"`` or ``"tabpfn"`` from a base-checkpoint path/name."""
    return "tabicl" if "tabicl" in Path(str(base_path)).name.lower() else "tabpfn"


def import_tabicl_core():
    """Return the bare ``TabICLv2`` nn.Module class (training forward)."""
    try:
        from tabicl._model.tabicl import TabICL
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "The `tabicl` package (>=2.1.1) is required for the TabICLv2 model "
            "family. Install with `pip install 'tabicl[finetune]>=2.1.1,<3'`. "
            f"Underlying error: {exc}"
        ) from exc
    return TabICL


def import_tabicl_finetune_data():
    """Return ``(MetaBatch, _build_meta_batch)`` from tabicl's official
    finetuning internals — the exact per-step preprocessing (context/query
    split, EnsembleGenerator variants, class-shuffle remap, regression
    z-norm) their ``FinetunedTabICL*`` wrappers train with.

    Requires the ``tabicl[finetune]`` extra: importing this module executes
    ``tabicl._finetune/__init__.py`` → ``base`` → ``tabicl.train._optim`` →
    ``transformers``, and tabicl declares ``transformers`` ONLY under its
    finetune/pretrain/all extras. Inference works without it (the sklearn
    wrappers are lazily imported), so a plain ``pip install tabicl`` fails
    here and nowhere else.
    """
    try:
        from tabicl._finetune.data import MetaBatch, _build_meta_batch
    except ModuleNotFoundError as exc:                          # pragma: no cover
        # Distinguish "optional extra not installed" (by far the likeliest
        # cause, and the one a version check will NOT reveal) from "upstream
        # moved the symbol". The first message we shipped blamed the version
        # and sent a real debugging session down the wrong path (2026-08-05).
        missing = getattr(exc, "name", "") or str(exc)
        if "tabicl" not in missing:
            raise ImportError(
                f"tabicl's finetuning internals need the optional dependency "
                f"'{missing}', which is NOT installed. This is a missing "
                f"EXTRA, not a version problem: tabicl declares it under its "
                f"`finetune` extra, so plain `pip install tabicl` gives you a "
                f"working INFERENCE install that fails only here.\n"
                f"  Fix: pip install 'tabicl[finetune]>=2.1.1,<3'\n"
                f"  Install it into the environment the SLURM jobs activate "
                f"(each job log prints 'Active conda env: ...' near the top) "
                f"— an interactive venv is not necessarily that environment.\n"
                f"  Underlying error: {exc}"
            ) from exc
        raise ImportError(
            "tabicl._finetune.data moved or is unavailable — the pinned "
            "tabicl version (2.1.x) ships it. Check the installed version "
            f"before adjusting this shim. Underlying error: {exc}"
        ) from exc
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "tabicl._finetune.data moved or is unavailable — the pinned "
            "tabicl version (2.1.x) ships it. Check the installed version "
            f"before adjusting this shim. Underlying error: {exc}"
        ) from exc
    return MetaBatch, _build_meta_batch


def import_tabicl_sklearn():
    """Return ``(TabICLClassifier, TabICLRegressor)`` — the sklearn-style
    inference wrappers (used by the monitor eval and the benchmark)."""
    try:
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "Could not import TabICLClassifier/TabICLRegressor from `tabicl`. "
            f"Underlying error: {exc}"
        ) from exc
    return TabICLClassifier, TabICLRegressor


def smoke_test(track: str) -> None:
    """Fail-fast preflight (SLURM prolog): verify every tabicl import the
    given track's training will need, in seconds, before GPU time is spent.
    Mirrors ``tabpfn_compat.smoke_test``. Safe to call when the grid contains
    no tabicl base — it only checks imports, not checkpoints.

    Checks are ordered cheapest-and-most-fundamental first and each is
    reported, so a failure says WHICH capability is missing rather than just
    "tabicl is broken". Inference and training have different dependency
    sets (see :func:`import_tabicl_finetune_data`), and only the training
    one needs the ``[finetune]`` extra.
    """
    from importlib.metadata import version
    print(f"tabicl_compat smoke_test (track={track}, tabicl={version('tabicl')})")

    TabICL = import_tabicl_core()                               # noqa: N806
    assert TabICL is not None
    print("  [ok] core model class      (tabicl._model.tabicl.TabICL)")

    clf, reg = import_tabicl_sklearn()
    assert clf is not None and reg is not None
    print("  [ok] inference wrappers    (TabICLClassifier / TabICLRegressor)")

    import_tabicl_finetune_data()
    # ASCII only: this runs in a SLURM prolog, and a UnicodeEncodeError in a
    # preflight check would abort the job it exists to protect.
    print("  [ok] finetuning internals  (tabicl._finetune.data) "
          "- requires the [finetune] extra")
    print("tabicl_compat smoke_test PASSED")
