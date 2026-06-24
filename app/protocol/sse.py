"""SSE protocol helpers."""

from __future__ import annotations

from typing import Any, AsyncIterator

from app.models.sse import SSEChunk
from app.tools.jsonx import dumps


DONE = "[DONE]"


class SSE:
    @staticmethod
    def encode(event: str | None, data: dict[str, Any] | str) -> str:
        payload = dumps(data) if isinstance(data, dict) else data
        return f"event: {event}\ndata: {payload}\n\n" if event else f"data: {payload}\n\n"

    @staticmethod
    async def parse(status_code: int, lines: AsyncIterator[str]) -> AsyncIterator[SSEChunk]:
        event: str | None = None
        data: list[str] = []

        async for raw in lines:
            line = raw.rstrip("\r\n")

            if not line:
                if data:
                    yield SSEChunk(status_code=status_code, event=event, data="\n".join(data))
                event, data = None, []
                continue

            if line.startswith(":"):
                continue

            if line.startswith("event:"):
                event = line[6:].strip()
                continue

            if line.startswith("data:"):
                data.append(line[5:].strip())
                continue

            if line == DONE or line.startswith("{"):
                data.append(line)

        if data:
            yield SSEChunk(status_code=status_code, event=event, data="\n".join(data))
