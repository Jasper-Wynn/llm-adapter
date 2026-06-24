"""Protocol exceptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProtocolError(ValueError):
    message: str
    status_code: int = 400
    error_type: str = "invalid_request_error"
    code: str | None = None

    def __post_init__(self) -> None:
        ValueError.__init__(self, self.message)

    def to_openai(self) -> dict:
        return {
            "error": {
                "type": self.error_type,
                "message": self.message,
                "code": self.code,
            }
        }

    def to_anthropic(self) -> dict:
        return {
            "type": "error",
            "error": {
                "type": self.error_type,
                "message": self.message,
            },
        }
