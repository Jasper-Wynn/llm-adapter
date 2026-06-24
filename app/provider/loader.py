"""Dynamic provider loading."""

from __future__ import annotations

import importlib
import logging
from typing import Any


def load_provider(cfg: Any, *, logger: logging.Logger | None = None):
    class_path = cfg.PROVIDER_CLASS
    module_name, _, class_name = class_path.rpartition(".")

    if not module_name or not class_name:
        raise RuntimeError(
            "PROVIDER_CLASS must be a full import path, "
            "for example app.provider.openai_compatible.OpenAICompatibleProvider"
        )

    module = importlib.import_module(module_name)
    provider_cls = getattr(module, class_name)

    if hasattr(provider_cls, "from_config"):
        return provider_cls.from_config(cfg, logger=logger)

    return provider_cls()
