"""SSE data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SSEChunk:
    status_code: int
    data: str
    event: str | None = None
