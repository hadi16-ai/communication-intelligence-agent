"""Structured logging setup.

Provides one configured logger per module via `get_logger(__name__)`.
Handler/level setup happens exactly once per process (idempotent — safe
to call repeatedly, which matters under Streamlit's rerun-the-whole-
script-on-every-interaction model).

Privacy discipline: this module logs whatever its callers pass it — it
does not itself inspect or redact anything. The actual guarantee ("never
log subject, sender, or email body — only IDs, categories, and numeric
metrics") is enforced by the call sites that use this logger
(app/ai/gemini.py, app/core/pipeline.py), not by code here. Keep it that
way: never pass raw email content to any logger obtained from this module.
"""

from __future__ import annotations

import logging
import sys

from app.core.config import Settings

_ROOT_LOGGER_NAME = "communication_intelligence_agent"
_configured = False


def _configure_once() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    try:
        level_name = Settings().log_level.upper()
    except Exception:
        level_name = "INFO"
    root.setLevel(getattr(logging, level_name, logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.propagate = False  # don't double-log through pytest's/Streamlit's own root logger

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """A configured logger under this app's own namespace.

    Usage: `logger = get_logger(__name__)` at module level, then
    `logger.info(...)` / `.warning(...)` / `.error(...)` with only IDs,
    enum values, and numeric metrics as arguments — never email content.
    """
    _configure_once()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
