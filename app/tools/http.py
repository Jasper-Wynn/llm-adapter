"""Project HTTP helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from fastapi.responses import JSONResponse


STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream; charset=utf-8",
}
SENSITIVE_HEADERS = {"authorization", "x-api-key", "x-auth-token", "api-key"}


def bool_value(value: Any) -> bool:
    if value is True:
        return True

    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}

    return False


def safe_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    result = {}

    for key, value in headers.items():
        text = str(value)
        lower = key.lower()

        if lower in SENSITIVE_HEADERS:
            result[str(key)] = "Bearer ***" if text.lower().startswith("bearer ") else "***"
        else:
            result[str(key)] = text

    return result


def json_error(status_code: int, message: str, error_type: str = "invalid_request_error", *, code: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"type": error_type, "message": message, "code": code}})


def parse_size(value: str | int) -> int:
    if isinstance(value, int):
        return value

    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)\s*(b|kb|mb|gb)?", text)

    if not match:
        raise ValueError(f"Invalid size: {value}")

    number = int(match.group(1))
    unit = match.group(2) or "b"

    return number * {
        "b": 1,
        "kb": 1024,
        "mb": 1024 * 1024,
        "gb": 1024 * 1024 * 1024,
    }[unit]


def with_request_id(headers: dict[str, str], request_id: str) -> dict[str, str]:
    return {
        **headers,
        "X-Request-ID": request_id,
    }
