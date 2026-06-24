"""Reasoning/thinking protocol conversion."""

from __future__ import annotations

from typing import Any


class ReasoningCodec:
    @staticmethod
    def anthropic_to_upstream(thinking: dict[str, Any] | None, *, output_config: dict[str, Any] | None = None, max_tokens: int | None = None) -> dict[str, Any] | None:
        if not thinking or not isinstance(thinking, dict):
            return None

        kind = thinking.get("type")

        if kind == "disabled":
            return None

        if kind == "adaptive":
            effort = _effort(thinking.get("effort") or (output_config or {}).get("effort"))

        elif kind == "enabled":
            budget = _int(thinking.get("budget_tokens"), 5000)

            if max_tokens and budget >= max_tokens:
                budget = max(1024, max_tokens - 1)

            effort = "low" if budget <= 2000 else "high" if budget >= 10000 else "medium"

        else:
            return None

        return {"reasoning": {"effort": effort}}

    @staticmethod
    def extract_from_message(message: dict[str, Any]) -> dict[str, str] | None:
        for key in ("thinking", "reasoning"):
            value = message.get(key)

            if isinstance(value, dict):
                text = value.get("content") or value.get("thinking") or value.get("reasoning")
                if text:
                    return {
                        "content": str(text),
                        "signature": str(value.get("signature", "")),
                    }

            elif value:
                return {
                    "content": str(value),
                    "signature": "",
                }

        if message.get("reasoning_content"):
            return {
                "content": str(message["reasoning_content"]),
                "signature": "",
            }

        return None

    @staticmethod
    def to_anthropic_block(value: dict[str, str] | None) -> dict[str, str] | None:
        if not value or not value.get("content"):
            return None

        return {
            "type": "thinking",
            "thinking": value["content"],
            "signature": value.get("signature", ""),
        }


def _effort(value: Any) -> str:
    normalized = str(value or "medium").lower().strip()
    return normalized if normalized in {"low", "medium", "high", "xhigh", "max"} else "medium"


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
