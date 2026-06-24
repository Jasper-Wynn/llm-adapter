"""Project id helpers."""

from __future__ import annotations

import re
import uuid


def request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def tool_use_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


def anthropic_tool_id(value: str | None) -> str:
    if not value:
        return tool_use_id()

    if value.startswith("toolu_"):
        return value

    safe = re.sub(r"[^a-zA-Z0-9_]", "_", value)

    if safe.startswith("call_"):
        safe = safe[5:]

    return f"toolu_{safe[:24]}" if safe else tool_use_id()
