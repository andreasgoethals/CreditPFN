"""Focused tests for structured logging and warning hygiene."""

from __future__ import annotations

import logging
import warnings

from src.utils.run_log import _StructuredFormatter, configure_warning_filters


def test_multiline_column_transformer_futurewarning_is_filtered() -> None:
    configure_warning_filters()
    message = (
        "\nThe format of the columns of the 'remainder' transformer in "
        "ColumnTransformer.transformers_ will change in version 1.7 to match "
        "the format of the other transformers.\n"
        "To use the new behavior now and suppress this warning, use "
        "force_int_remainder_cols=False."
    )
    with warnings.catch_warnings(record=True) as caught:
        # ``catch_warnings`` resets the filter list, so configure inside it.
        configure_warning_filters()
        warnings.warn(message, FutureWarning)
        warnings.warn("an unrelated future change", FutureWarning)

    assert [str(item.message) for item in caught] == [
        "an unrelated future change",
    ]


def test_warning_level_label_is_readable() -> None:
    formatter = _StructuredFormatter(use_color=False)
    record = logging.LogRecord(
        name="test", level=logging.WARNING, pathname=__file__, lineno=1,
        msg="example", args=(), exc_info=None,
    )
    rendered = formatter.format(record)
    assert "[WARN ]" in rendered
    assert "[WARNI]" not in rendered
