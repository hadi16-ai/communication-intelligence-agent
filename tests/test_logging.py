"""Tests for app.utils.logging: the configured-logger factory.

Does not assert on actual emitted output (that's covered end-to-end by
tests/test_pipeline.py's TestObservabilityLogging) — this file only checks
get_logger()'s own contract: naming, idempotent handler setup, and level.
"""

from __future__ import annotations

import logging

from app.utils.logging import get_logger


def test_get_logger_returns_a_standard_library_logger():
    logger = get_logger("some.module")
    assert isinstance(logger, logging.Logger)


def test_logger_name_is_namespaced_under_the_app():
    logger = get_logger("app.core.pipeline")
    assert logger.name == "communication_intelligence_agent.app.core.pipeline"


def test_repeated_calls_do_not_attach_duplicate_handlers():
    """Under Streamlit's rerun-the-whole-script-on-every-interaction
    model, module-level `logger = get_logger(__name__)` executes on every
    rerun — handler setup must be idempotent or output would multiply.
    """
    before = len(logging.getLogger("communication_intelligence_agent").handlers)

    get_logger("module_a")
    get_logger("module_b")
    get_logger("module_a")

    after = len(logging.getLogger("communication_intelligence_agent").handlers)
    assert after == before or after == 1  # either already configured, or exactly one handler added


def test_root_namespace_logger_does_not_propagate_to_the_true_root():
    """Deliberate: avoids double-logging through pytest's/Streamlit's own
    root logger handler.
    """
    get_logger("anything")
    assert logging.getLogger("communication_intelligence_agent").propagate is False
