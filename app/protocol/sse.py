"""SSE codec helpers."""
from __future__ import annotations

from typing import Any

from app.tools.jsonx import dumps

DONE = "[DONE]"


class SSE:
    @staticmethod
    def encode(event: str | None, data: Any) -> str:
        parts: list[str] = []
        if event:
            parts.append(f"event: {event}")
        if isinstance(data, str):
            data_text = data
        else:
            data_text = dumps(data)
        for line in data_text.splitlines() or [""]:
            parts.append(f"data: {line}")
        return "\n".join(parts) + "\n\n"


def iter_sse_payloads_from_lines(lines: list[str]):
    event: str | None = None
    data: list[str] = []
    for line in lines:
        if line == "":
            if data or event:
                yield event, "\n".join(data)
            event = None
            data = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        else:
            yield None, line
    if data or event:
        yield event, "\n".join(data)
