"""Request context model."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class RequestContext:
    request_id: str
    path: str
    model: str | None
    stream: bool
    client: str | None = None
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000
