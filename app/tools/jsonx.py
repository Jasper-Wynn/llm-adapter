"""JSON helpers."""
from __future__ import annotations

import json
from typing import Any


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def loads(value: str) -> Any:
    return json.loads(value)


def loads_or_none(value: str) -> Any | None:
    try:
        return json.loads(value)
    except Exception:
        return None


def parse_json_object(value: str | None, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if default is None:
        default = {}
    if not value:
        return dict(default)
    try:
        parsed = json.loads(value)
    except Exception:
        return dict(default)
    return parsed if isinstance(parsed, dict) else dict(default)


def parse_string_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    value = value.strip()
    if not value or value == "{}":
        return {}
    if value.startswith("{"):
        parsed = parse_json_object(value)
        return {str(k): str(v) for k, v in parsed.items()}
    result: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip() or "=" not in item:
            continue
        key, val = item.split("=", 1)
        result[key.strip()] = val.strip()
    return result


def as_json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return dumps(value)
