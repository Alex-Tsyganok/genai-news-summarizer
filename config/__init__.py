"""Configuration package for the AI News Summarizer."""

from .settings import settings
from .logging_config import logger, setup_logging

__all__ = ['settings', 'logger', 'setup_logging']
