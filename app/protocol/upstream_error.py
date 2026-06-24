"""Upstream error normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class UpstreamError:
    message: str
    code: str | None = None
    type: str = "upstream_error"
    status_code: int = 502
    raw: dict[str, Any] | None = None

    def to_openai(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "code": self.code,
            }
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "type": "error",
            "error": {
                "type": self.type,
                "message": self.to_anthropic_message(),
            },
        }

    def to_anthropic_message(self) -> str:
        if self.code:
            return f"{self.message} ({self.code})"
        return self.message


class UpstreamErrorCodec:
    @staticmethod
    def from_payload(payload: Any, *, status_code: int = 200) -> UpstreamError | None:
        if not isinstance(payload, dict):
            return None

        if isinstance(payload.get("choices"), list):
            return None

        if _has_openai_error(payload):
            error = payload.get("error") or {}
            code = error.get("code")
            return UpstreamError(
                message=str(error.get("message") or "Upstream error"),
                code=code,
                type=str(error.get("type") or "upstream_error"),
                status_code=_status_from_code(code, status_code),
                raw=payload,
            )

        if not _looks_like_error(payload):
            return None

        code = _first_text(
            payload.get("error_code"),
            _nested(payload, "error", "error_code"),
            _nested(payload, "error", "code"),
            payload.get("code"),
        )

        raw_message = _first_text(
            _nested(payload, "error", "message"),
            _nested(payload, "error", "error_msg"),
            payload.get("message"),
            payload.get("error_msg"),
            payload.get("text"),
        )

        parsed_inner = _parse_json_object(raw_message)

        if parsed_inner:
            inner_type = _first_text(parsed_inner.get("type"))
            message = _first_text(
                parsed_inner.get("message"),
                parsed_inner.get("error_msg"),
                raw_message,
            )
        else:
            inner_type = None
            message = raw_message

        message = message or "Upstream error"

        return UpstreamError(
            message=message,
            code=code,
            type=_map_type(code=code, inner_type=inner_type, message=message),
            status_code=_status_from_code(code, status_code),
            raw=payload,
        )


def _looks_like_error(payload: dict[str, Any]) -> bool:
    if payload.get("error") is not None:
        return True

    if payload.get("error_code") or payload.get("error_msg"):
        return True

    if payload.get("text") == "[DONE]" and payload.get("error"):
        return True

    return False


def _has_openai_error(payload: dict[str, Any]) -> bool:
    error = payload.get("error")
    return isinstance(error, dict) and isinstance(error.get("message"), str)


def _nested(data: dict[str, Any], *path: str) -> Any:
    cur: Any = data

    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)

    return cur


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue

        if isinstance(value, str) and value.strip():
            return value.strip()

        if not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text

    return None


def _parse_json_object(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _map_type(*, code: str | None, inner_type: str | None, message: str) -> str:
    joined = " ".join(
        item.lower()
        for item in (code, inner_type, message)
        if item
    )

    if "429" in joined or "rpm" in joined or "quota" in joined or "rate" in joined:
        return "rate_limit_error"

    if "auth" in joined or "permission" in joined or "token" in joined:
        return "authentication_error"

    if "timeout" in joined:
        return "timeout_error"

    return "upstream_error"


def _status_from_code(code: str | None, fallback: int) -> int:
    text = str(code or "")

    if "429" in text:
        return 429
    if "401" in text:
        return 401
    if "403" in text:
        return 403
    if "404" in text:
        return 404
    if "408" in text:
        return 408
    if "500" in text:
        return 502

    if fallback and fallback >= 400:
        return fallback

    return 502
