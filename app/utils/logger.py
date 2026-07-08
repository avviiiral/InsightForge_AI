"""Centralized logging configuration for InsightForge-AI.

Every agent and service imports `get_logger(__name__)` from this module so
that log formatting, levels, and (optional) file sinks stay consistent
across the whole platform.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from app.config import BASE_DIR, get_settings

_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "insightforge.log",
        level="DEBUG",
        rotation="5 MB",
        retention=5,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[component]} - {message}",
    )
    _CONFIGURED = True


def get_logger(component: str):
    """Return a loguru logger bound to `component` (usually `__name__`)."""
    _configure_once()
    return logger.bind(component=component)
