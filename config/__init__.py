"""Configuration package for the AI News Summarizer.

Exports are lazy to avoid import-time side effects. Accessing attributes will
import and resolve them on first use.
"""
from typing import Any
import importlib
from .settings import settings  # ensure instance is available as package attribute

__all__ = ("settings", "logger", "setup_logging", "get_logger")


def __getattr__(name: str) -> Any:
	if name == "logger":
		# Backward-compat: expose a logger instance, initialized on first access
		module = importlib.import_module(f"{__name__}.logging_config")
		return module.get_logger()
	if name == "setup_logging":
		module = importlib.import_module(f"{__name__}.logging_config")
		return module.setup_logging
	if name == "get_logger":
		module = importlib.import_module(f"{__name__}.logging_config")
		return module.get_logger
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
	return sorted(list(globals().keys()) + list(__all__))
