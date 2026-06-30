"""Protocol exceptions."""
from __future__ import annotations


class ProtocolError(Exception):
    def __init__(self, message: str, *, status_code: int = 400, error_type: str = "invalid_request_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type

    def to_anthropic(self) -> dict:
        return {"type": "error", "error": {"type": self.error_type, "message": self.message}}
