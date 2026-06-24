"""Anthropic SSE event builders."""

from __future__ import annotations

from typing import Any

from app.models.usage import Usage
from app.protocol.sse import SSE


class AnthropicEvents:
    @staticmethod
    def message_start(message_id: str, model: str, usage: Usage) -> str:
        return SSE.encode(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": 0,
                    },
                },
            },
        )

    @staticmethod
    def block_start(index: int, block: dict[str, Any]) -> str:
        return SSE.encode(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": block,
            },
        )

    @staticmethod
    def block_delta(index: int, delta: dict[str, Any]) -> str:
        return SSE.encode(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            },
        )

    @staticmethod
    def block_stop(index: int) -> str:
        return SSE.encode(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": index,
            },
        )

    @staticmethod
    def message_delta(stop_reason: str, usage: Usage) -> str:
        return SSE.encode(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                },
                "usage": usage.as_dict(),
            },
        )

    @staticmethod
    def message_stop() -> str:
        return SSE.encode("message_stop", {"type": "message_stop"})

    @staticmethod
    def error(message: str, error_type: str = "api_error") -> str:
        return SSE.encode(
            "error",
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
        )
