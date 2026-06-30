"""Normalize upstream error payloads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class UpstreamError:
    type: str
    message: str
    code: str | None = None
    status_code: int = 502

    def to_openai(self) -> dict[str, Any]:
        return {"error": {"type": self.type, "message": self.message, "code": self.code}}

    def to_anthropic(self) -> dict[str, Any]:
        return {"type": "error", "error": {"type": self.type, "message": self.message}}

    def to_anthropic_message(self) -> str:
        return self.message


class UpstreamErrorCodec:
    @staticmethod
    def from_payload(payload: Any) -> UpstreamError | None:
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("msg") or str(error)
            error_type = error.get("type") or error.get("code") or "upstream_error"
            code = error.get("code")
            status = _status_from_payload(payload, code)
            return UpstreamError(str(error_type), str(message), None if code is None else str(code), status)
        if isinstance(error, str):
            return UpstreamError("upstream_error", error)
        # Common internal gateway shape: {code: ..., message: ...}
        if ("message" in payload or "msg" in payload) and ("code" in payload or "error_code" in payload):
            msg = payload.get("message") or payload.get("msg")
            code = payload.get("code") or payload.get("error_code")
            return UpstreamError("upstream_error", str(msg), str(code), _status_from_payload(payload, code))
        return None


def _status_from_payload(payload: dict[str, Any], code: Any) -> int:
    for key in ("status", "status_code", "http_status"):
        try:
            value = int(payload.get(key))
            if 400 <= value <= 599:
                return value
        except Exception:
            pass
    try:
        num = int(code)
        if 400 <= num <= 599:
            return num
    except Exception:
        pass
    return 502
