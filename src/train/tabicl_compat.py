"""Version-tolerant imports + family detection for TabICLv2 (v2).

TabICLv2 is the second model family CreditPFN continued-pretrains (2026-08-04),
next to TabPFN. It is fully open (code BSD-3, weights on HF `jingang/TabICL`)
and ships official finetuning internals under ``tabicl._finetune`` — private
modules, so every import is funneled through this file (mirroring
``tabpfn_compat.py``) and pinned via ``pyproject.toml`` (``tabicl[finetune]>=2.1.1,<3``).
If an upstream release moves these symbols, this is the ONE file to fix.

Family detection
----------------
A base checkpoint belongs to the ``"tabicl"`` family iff its filename contains
``tabicl`` (e.g. ``checkpoints/tabicl-classifier-v2-20260212.ckpt``); anything
else is ``"tabpfn"``. The checkpoint files are downloaded once from
https://huggingface.co/jingang/TabICL into the staging ``checkpoints/`` dir —
same pattern as the TabPFN bases.

Adaptation-mode caveat (from the literature, 2026-08-04)
--------------------------------------------------------
Two independent reports show TabICLv2 is fragile under aggressive full SFT
(Tanna 2026: TabZilla accuracy 0.873→0.567; Kolberg 2026: their CPT recipe
"failed to train TabICLv2"), while TabICLv2's own pretraining stage 3 freezes
everything except the ICL module. CreditPFN therefore maps the grid's
``use_lora`` axis, for this family, to **freeze-backbone / train-ICL-head-only**
(the upstream-sanctioned adaptation) instead of LoRA — see
``load_tabicl_for_training(freeze_backbone=...)`` and the ``_iclhead`` tag in
checkpoint names.
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
