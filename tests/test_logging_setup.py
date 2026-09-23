"""Focused tests for structured logging and warning hygiene."""

from __future__ import annotations

import logging
import warnings

from src.utils.logging_setup import RepeatedWarnings, _StructuredFormatter, configure_warning_filters


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


def test_repeated_numerical_warnings_are_bounded_but_exact_totals_and_errors_remain(caplog):
    logger = logging.getLogger("numerical-warning-test")
    counter = RepeatedWarnings(logger)
    with caplog.at_level(logging.WARNING):
        for step in range(10_000):
            counter.warning("nonfinite_loss/table-a", "step=%d: non-finite loss", step)
        counter.warning("missing_context_class/table-b", "a different condition")
        counter.summary()
        try:
            raise RuntimeError("a genuine failure")
        except RuntimeError:
            logger.exception("training failed")
    assert len(caplog.records) < 20
    assert caplog.records[0].getMessage() == "step=0: non-finite loss"
    assert "nonfinite_loss/table-a=10000" in caplog.text
    assert "missing_context_class/table-b=1" in caplog.text
    assert "a genuine failure" in caplog.text
    assert caplog.records[-1].exc_info is not None
    # A new trial gets its first diagnostic again.
    RepeatedWarnings(logger).warning("nonfinite_loss/table-a", "first warning in next trial")
    assert "first warning in next trial" in caplog.text
