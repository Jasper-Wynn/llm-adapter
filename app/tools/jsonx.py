"""Project JSON helpers."""

from __future__ import annotations

import json
from typing import Any


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads(value: str) -> Any:
    return json.loads(value)


def as_json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return dumps(value)
