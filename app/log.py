"""Application logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


NOISY_EXTERNAL_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.http2",
    "httpcore.connection",
    "httpcore.proxy",
    "hpack",
    "hpack.hpack",
    "hpack.table",
    "h2",
    "h2.connection",
    "h2.stream",
    "uvicorn.access",
)


def configure_logging(*, level: str, console: bool, file_enabled: bool, file_path: str, external_level: str = "warning") -> None:
    root = logging.getLogger()
    root.handlers.clear()

    app_level = _level(level, logging.INFO)
    third_party_level = _level(external_level, logging.WARNING)

    root.setLevel(app_level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s")

    if console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(app_level)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if file_enabled:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(app_level)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    _silence_external_loggers(third_party_level)


def logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _level(value: str, default: int) -> int:
    return getattr(logging, str(value).upper(), default)


def _silence_external_loggers(level: int) -> None:
    for name in NOISY_EXTERNAL_LOGGERS:
        logging.getLogger(name).setLevel(level)
