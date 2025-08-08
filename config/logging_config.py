"""
Logging configuration for the AI News Summarizer.

This module avoids configuring logging at import time. Use get_logger()
from your application entry point to initialize the logger on first use.
"""
import logging
import sys
import threading
from typing import Optional

_LOCK = threading.Lock()
_LOGGER: Optional[logging.Logger] = None


def _default_level() -> str:
    """Resolve default level from settings lazily; fallback to INFO if unavailable."""
    try:
        # Lazy import to avoid triggering settings at module import time
        from config.settings import settings  # type: ignore

        return getattr(settings, "LOG_LEVEL", "INFO")
    except Exception:
        return "INFO"


def _resolve_level(level: Optional[str]) -> int:
    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    effective = (level or _default_level()).upper()
    return levels.get(effective, logging.INFO)


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure and return the application logger. Safe to call multiple times.

    Args:
        level: Logging level name (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). If not
               provided, uses settings.LOG_LEVEL when available, else INFO.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger("news_summarizer")
    lvl = _resolve_level(level)

    with _LOCK:
        # Always set the logger's level
        logger.setLevel(lvl)

        # Create handler only once
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Keep handler levels aligned with logger level
        for h in logger.handlers:
            h.setLevel(lvl)

    return logger


def get_logger(level: Optional[str] = None) -> logging.Logger:
    """Return the shared logger, initializing it on first use.

    If level is provided on first call, it will be used to configure the logger; on
    subsequent calls, the existing logger is returned unchanged.
    """
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = setup_logging(level)
    return _LOGGER
