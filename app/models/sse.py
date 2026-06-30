"""Internal SSE event model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SSEEvent:
    data: str = ""
    event: str | None = None
    status_code: int = 200
