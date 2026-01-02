"""Structured logging helpers."""
from __future__ import annotations

import logging
from logging.config import dictConfig

from .config import get_settings


def configure_logging() -> None:
    """Apply a sane default logging configuration."""

    settings = get_settings()
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": settings.log_level,
                }
            },
            "root": {"handlers": ["console"], "level": settings.log_level},
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Convenience factory that ensures logging is configured exactly once."""

    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
