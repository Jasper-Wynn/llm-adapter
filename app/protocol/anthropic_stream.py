"""Anthropic stream lifecycle state."""

from __future__ import annotations

from app.models.tool_call import ToolCallState
from app.models.usage import Usage
from app.protocol.anthropic_events import AnthropicEvents
from app.protocol.usage import UsageCodec
from app.tools.ids import message_id


class AnthropicStream:
    def __init__(self, model: str):
        self.model = model
        self.message_id = message_id()
        self.usage = Usage()

        self.started = False
        self.closed = False

        self.next_block_index = 0
        self.current_index = -1
        self.current_type: str | None = None
        self.signature_sent = False

    def next_index(self) -> int:
        value = self.next_block_index
        self.next_block_index += 1
        return value

    def update_usage(self, usage: dict | None) -> None:
        if usage:
            self.usage = UsageCodec.from_openai(usage)

    def start(self) -> list[str]:
        if self.started:
            return []

        self.started = True
        return [AnthropicEvents.message_start(self.message_id, self.model, self.usage)]

    def close_block(self) -> list[str]:
        if self.current_index < 0:
            return []

        index = self.current_index

        self.current_index = -1
        self.current_type = None
        self.signature_sent = False

        return [AnthropicEvents.block_stop(index)]

    def thinking(self, text: str) -> list[str]:
        if not text:
            return []

        out = self._ensure_block(
            "thinking",
            {
                "type": "thinking",
                "thinking": "",
                "signature": "",
            },
        )
        out.append(
            AnthropicEvents.block_delta(
                self.current_index,
                {
                    "type": "thinking_delta",
                    "thinking": text,
                },
            )
        )
        return out

    def signature(self, value: str = "") -> list[str]:
        if self.current_type != "thinking":
            return []

        self.signature_sent = True
        return [
            AnthropicEvents.block_delta(
                self.current_index,
                {
                    "type": "signature_delta",
                    "signature": value,
                },
            )
        ]

    def text(self, value: str) -> list[str]:
        if not value:
            return []

        out: list[str] = []

        if self.current_type == "thinking" and not self.signature_sent:
            out.extend(self.signature(""))

        out.extend(
            self._ensure_block(
                "text",
                {
                    "type": "text",
                    "text": "",
                },
            )
        )
        out.append(
            AnthropicEvents.block_delta(
                self.current_index,
                {
                    "type": "text_delta",
                    "text": value,
                },
            )
        )
        return out

    def tool_start(self, state: ToolCallState) -> list[str]:
        out = self.close_block()

        self.current_index = state.block_index
        self.current_type = "tool_use"

        out.append(
            AnthropicEvents.block_start(
                state.block_index,
                {
                    "type": "tool_use",
                    "id": state.tool_id,
                    "name": state.name,
                    "input": {},
                },
            )
        )
        return out

    def tool_delta(self, state: ToolCallState, partial_json: str) -> list[str]:
        if not partial_json:
            return []

        return [
            AnthropicEvents.block_delta(
                state.block_index,
                {
                    "type": "input_json_delta",
                    "partial_json": partial_json,
                },
            )
        ]

    def finish(self, stop_reason: str = "end_turn") -> list[str]:
        if self.closed:
            return []

        out = self.start()

        if self.current_type == "thinking" and not self.signature_sent:
            out.extend(self.signature(""))

        out.extend(self.close_block())
        out.append(AnthropicEvents.message_delta(stop_reason, self.usage))
        out.append(AnthropicEvents.message_stop())

        self.closed = True
        return out

    def error(self, message: str, error_type: str = "api_error") -> str:
        return AnthropicEvents.error(message, error_type)

    def _ensure_block(self, kind: str, block: dict) -> list[str]:
        if self.current_type == kind:
            return []

        out = self.close_block()

        self.current_index = self.next_index()
        self.current_type = kind
        self.signature_sent = False

        out.append(AnthropicEvents.block_start(self.current_index, block))
        return out
