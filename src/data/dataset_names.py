"""Reader-facing dataset names — the ONLY place display names are resolved.

Maps a dataset slug (or its bare stem) to the name a reader/figure should show. The real
slug ↔ name mapping lives in the GITIGNORED ``src/data/_private_names.py``; this tracked
module imports it when present and DEGRADES to the raw slug when it is absent. A public
clone has neither the mapping nor the private data, so it never sees a private slug to
leak; a checkout WITH the private data also has the mapping, so private datasets render as
``Prop*`` in every figure and summary.

Never hard-code a display name anywhere else — always call :func:`display_name`.
"""
from __future__ import annotations

import re

_INDEX_PREFIX = re.compile(r"^\d+\.")


def _stem(slug: object) -> str:
    """The join-key stem: the slug with any leading ``NNNN.`` index stripped."""
    return _INDEX_PREFIX.sub("", str(slug)).strip()


try:  # the gitignored mapping is present only where the private data is
    from src.data._private_names import DISPLAY as _DISPLAY, PROPRIETARY as _PROPRIETARY
except Exception:  # public clone / mapping absent -> degrade to raw slugs
    _DISPLAY, _PROPRIETARY = {}, set()


def display_name(slug: object) -> str:
    """Reader-facing name for one dataset id; the raw slug if unknown or mapping absent."""
    return _DISPLAY.get(_stem(slug), str(slug))


def is_proprietary(slug: object) -> bool:
    """True iff this dataset must never be named in anything published."""
    return _stem(slug) in _PROPRIETARY


def display_id_list(ids: object, sep: str = ";") -> object:
    """Map a ``sep``-joined slug list (manifest ``train/test_dataset_ids``) to display names."""
    if ids is None or (isinstance(ids, float)):  # NaN / missing -> leave as-is
        return ids
    parts = [display_name(p) for p in str(ids).split(sep) if p != ""]
    return sep.join(parts)


def sort_key(slug: object) -> tuple[int, str]:
    """(is_proprietary, display_name) — public datasets first, then alphabetical."""
    return (int(is_proprietary(slug)), display_name(slug))


def redact_private_names(value: object) -> object:
    """Remove private identifiers embedded in reader-facing errors and filenames."""
    if not isinstance(value, str):
        return value
    for slug in sorted(_PROPRIETARY, key=len, reverse=True):
        # Underscores are filename separators, not identifier boundaries here.
        pattern = rf"(?<![A-Za-z0-9])(?:\d+\.)?{re.escape(slug)}(?![A-Za-z0-9])"
        value = re.sub(pattern, lambda _: _DISPLAY.get(slug, "[private dataset]"), value)
    return value


def display_frame(frame):
    """Sanitize free text only at presentation boundaries, after internal joins."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]):
        result[column] = result[column].map(redact_private_names)
    return result
