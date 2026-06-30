"""HTTP helper functions."""
from __future__ import annotations

from typing import Mapping

from fastapi.responses import JSONResponse

from app import config as cfg

STREAM_HEADERS = cfg.STREAM_HEADERS
SENSITIVE_HEADERS = cfg.SENSITIVE_HEADERS


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_size(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    multipliers = {
        "b": 1,
        "kb": 1024,
        "k": 1024,
        "mb": 1024 ** 2,
        "m": 1024 ** 2,
        "gb": 1024 ** 3,
        "g": 1024 ** 3,
    }
    for suffix, multiplier in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(text)


def json_error(status_code: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"type": error_type, "message": message}},
    )


def with_request_id(headers: Mapping[str, str], request_id: str) -> dict[str, str]:
    result = dict(headers)
    result[cfg.HEADER_REQUEST_ID] = request_id
    return result


def safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return cfg.safe_headers(headers)
