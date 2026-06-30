"""Logging setup and structured debug helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


_JSON_INDENT = 4
_COMPONENT_WIDTH = 9
_LOGGER_ALIASES = {
    "adapter.anthropic": "anthropic",
    "adapter.openai": "openai",
    "gateway.headers": "headers",
    "gateway.session_cancel": "session",
}

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
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
)


def log_debug_json(self: logging.Logger, key: str = "", data: Any = None) -> None:
    """Log structured data as pretty JSON when DEBUG is enabled.

    This method is attached to ``logging.Logger`` so every project logger can do::

        log.debug_json("provider.describe", provider.describe())

    Behavior follows the project's debugging style:
    - DEBUG enabled: pretty JSON for dict/list, fallback to ``str`` for objects.
    - DEBUG disabled: no output. Call ``info_json`` for structured data that
      should be visible during normal operation.
    """
    if not self.isEnabledFor(logging.DEBUG):
        return
    if isinstance(data, (dict, list, tuple)):
        try:
            payload = json.dumps(data, indent=_JSON_INDENT, ensure_ascii=False, default=str)
        except TypeError:
            payload = str(data)
        self.debug("%s\n%s", key, payload)
    else:
        self.debug("%s %s", key, data)


def log_info_json(self: logging.Logger, key: str = "", data: Any = None) -> None:
    """Always log structured data at INFO level.

    Keep this for startup/config/provider summaries. Runtime payloads should
    generally use ``debug_json`` to avoid noisy normal logs.
    """
    if isinstance(data, (dict, list, tuple)):
        try:
            payload = json.dumps(data, indent=_JSON_INDENT, ensure_ascii=False, default=str)
        except TypeError:
            payload = str(data)
        self.info("%s\n%s", key, payload)
    else:
        self.info("%s %s", key, data)


def install_logger_methods() -> None:
    """Install project helper methods on standard Logger instances."""
    if not hasattr(logging.Logger, "debug_json"):
        setattr(logging.Logger, "debug_json", log_debug_json)
    if not hasattr(logging.Logger, "info_json"):
        setattr(logging.Logger, "info_json", log_info_json)


# Make the helpers available even if code obtains a logger before
# ``configure_logging`` is called.
install_logger_methods()


def configure_logging(
    *,
    level: str = "info",
    external_level: str = "warning",
    console: bool = True,
    file_enabled: bool = False,
    file_path: str = "logs/server.log",
) -> None:
    install_logger_methods()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(_level(level))

    formatter = AlignedFormatter(datefmt="%Y-%m-%d %H:%M:%S")

    if console:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if file_enabled:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        root.addHandler(handler)

    _silence_external_loggers(_level(external_level))


def logger(name: str) -> logging.Logger:
    install_logger_methods()
    return logging.getLogger(f"llm_adapter.{name}")


class AlignedFormatter(logging.Formatter):
    """Keep log prefixes aligned so message text starts in one column."""

    def __init__(self, *, datefmt: str):
        super().__init__(fmt="%(asctime)s %(levelname)-5s %(component)s | %(message)s", datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        component = record.name
        if component.startswith("llm_adapter."):
            component = component[len("llm_adapter."):]
        component = _LOGGER_ALIASES.get(component, component.rsplit(".", 1)[-1])
        if len(component) > _COMPONENT_WIDTH:
            component = component[:_COMPONENT_WIDTH]
        record.component = component.ljust(_COMPONENT_WIDTH)
        return super().format(record)


def _level(value: str) -> int:
    return getattr(logging, str(value).upper(), logging.INFO)


def _silence_external_loggers(level: int) -> None:
    for name in NOISY_EXTERNAL_LOGGERS:
        logging.getLogger(name).setLevel(level)
