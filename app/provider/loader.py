"""Dynamic provider loader."""
from __future__ import annotations

import importlib

from app import config as cfg


def load_provider(*, logger=None):
    module_name, class_name = cfg.PROVIDER_CLASS.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    provider = cls(logger=logger)
    if logger:
        logger.info("Loaded provider class=%s", cfg.PROVIDER_CLASS)
    return provider
