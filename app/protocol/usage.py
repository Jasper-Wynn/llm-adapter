"""Usage protocol conversion."""

from __future__ import annotations

from typing import Any

from app.models.usage import Usage


class UsageCodec:
    @staticmethod
    def from_openai(usage: dict[str, Any] | None) -> Usage:
        if not usage:
            return Usage()

        prompt = usage.get("prompt_tokens", 0) or 0
        output = usage.get("completion_tokens", 0) or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0

        return Usage(
            input_tokens=max(prompt - cached, 0),
            output_tokens=output,
            cache_read_input_tokens=cached,
        )
