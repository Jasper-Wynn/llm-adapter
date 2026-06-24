"""Tool-call streaming state models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ToolCallState:
    openai_index: int
    block_index: int
    tool_id: str
    name: str
    arguments: str = ""
    started: bool = False
    emitted_arguments_len: int = 0

    def append_arguments(self, value: str) -> None:
        if value:
            self.arguments += value

    def pending_arguments(self) -> str:
        return self.arguments[self.emitted_arguments_len :]

    def mark_arguments_emitted(self) -> None:
        self.emitted_arguments_len = len(self.arguments)
