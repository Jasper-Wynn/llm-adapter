"""Model tool-calling protocol conversion."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.models.tool_call import ToolCallState
from app.protocol.exceptions import ProtocolError
from app.tools.ids import anthropic_tool_id, tool_use_id
from app.tools.jsonx import as_json_text


TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
SUPPORTED_ANTHROPIC_TOOL_TYPES = {None, "custom"}
SUPPORTED_OPENAI_TOOL_TYPES = {"function"}


class ToolCallingCodec:
    @staticmethod
    def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for item in tools or []:
            if not isinstance(item, dict):
                raise ProtocolError("Each tool must be an object.")

            # Already OpenAI style; keep it compatible but validate.
            if item.get("type") in SUPPORTED_OPENAI_TOOL_TYPES and isinstance(item.get("function"), dict):
                function = item["function"]
                _validate_tool_name(function.get("name"))
                result.append(item)
                continue

            tool_type = item.get("type")
            if tool_type not in SUPPORTED_ANTHROPIC_TOOL_TYPES:
                raise ProtocolError(
                    f"Anthropic server tool is not supported by this OpenAI-compatible upstream: {tool_type}",
                    code="unsupported_server_tool",
                )

            name = item.get("name")
            _validate_tool_name(name)

            input_schema = item.get("input_schema") or item.get("parameters")
            if input_schema is None:
                input_schema = {"type": "object", "properties": {}}

            if not isinstance(input_schema, dict):
                raise ProtocolError(
                    f"Tool {name!r} input_schema must be an object.",
                    code="invalid_tool_schema",
                )

            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": item.get("description", ""),
                        "parameters": input_schema,
                    },
                }
            )

        return result

    @staticmethod
    def anthropic_choice_to_openai(choice: dict[str, Any] | str | None) -> dict[str, Any]:
        if not choice:
            return {}

        if isinstance(choice, str):
            return {
                "tool_choice": choice
                if choice in {"auto", "none", "required"}
                else "auto"
            }

        kind = choice.get("type")

        if kind == "auto":
            out = {"tool_choice": "auto"}
        elif kind == "none":
            out = {"tool_choice": "none"}
        elif kind == "any":
            out = {"tool_choice": "required"}
        elif kind == "tool" and choice.get("name"):
            _validate_tool_name(choice.get("name"))
            out = {
                "tool_choice": {
                    "type": "function",
                    "function": {
                        "name": choice["name"],
                    },
                }
            }
        else:
            out = {"tool_choice": "auto"}

        if choice.get("disable_parallel_tool_use") is True:
            out["parallel_tool_calls"] = False

        return out

    @staticmethod
    def anthropic_result_to_openai(block: dict[str, Any]) -> dict[str, Any]:
        tool_use_id_value = block.get("tool_use_id")
        if not tool_use_id_value:
            raise ProtocolError("tool_result requires tool_use_id.", code="missing_tool_use_id")

        content = block.get("content", "")

        if isinstance(content, list):
            content = "\n".join(
                _content_block_text(item)
                for item in content
                if item is not None
            )
        else:
            content = as_json_text(content)

        return {
            "role": "tool",
            "tool_call_id": tool_use_id_value,
            "content": content,
        }

    @staticmethod
    def anthropic_use_to_openai_call(block: dict[str, Any]) -> dict[str, Any]:
        name = block.get("name") or "tool"
        _validate_tool_name(name)

        return {
            "id": block.get("id") or tool_use_id(),
            "type": "function",
            "function": {
                "name": name,
                "arguments": as_json_text(block.get("input", {})),
            },
        }

    @staticmethod
    def openai_call_to_anthropic_use(call: dict[str, Any]) -> dict[str, Any]:
        fn = call.get("function") or {}
        name = fn.get("name") or "tool"
        _validate_tool_name(name)

        raw_args = fn.get("arguments", "{}")

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {"_raw": raw_args}

        return {
            "type": "tool_use",
            "id": anthropic_tool_id(call.get("id")),
            "name": name,
            "input": args,
        }


@dataclass(slots=True)
class ToolCallMeta:
    raw_id: str | None = None
    seen_name: bool = False


@dataclass
class ToolCallStream:
    states: dict[int, ToolCallState] = field(default_factory=dict)
    meta: dict[int, ToolCallMeta] = field(default_factory=dict)

    def apply(self, delta: list[dict[str, Any]], next_block_index: Callable[[], int]) -> None:
        for item in delta:
            if not isinstance(item, dict):
                continue

            openai_index = int(item.get("index", 0))
            function = item.get("function") or {}

            state = self._ensure_state(
                openai_index=openai_index,
                item=item,
                function=function,
                next_block_index=next_block_index,
            )

            self._update_identity(
                openai_index=openai_index,
                state=state,
                item=item,
                function=function,
            )

            arguments_delta = function.get("arguments")
            if arguments_delta:
                state.append_arguments(arguments_delta)

    def has_calls(self) -> bool:
        return bool(self.states)

    def ordered(self) -> list[ToolCallState]:
        return [self.states[index] for index in sorted(self.states)]

    def _ensure_state(self, *, openai_index: int, item: dict[str, Any], function: dict[str, Any], next_block_index: Callable[[], int]) -> ToolCallState:
        state = self.states.get(openai_index)

        if state:
            return state

        raw_id = item.get("id")
        name = function.get("name") or f"tool_{openai_index}"
        _validate_tool_name(name)

        state = ToolCallState(
            openai_index=openai_index,
            block_index=next_block_index(),
            tool_id=anthropic_tool_id(raw_id),
            name=name,
        )

        self.states[openai_index] = state
        self.meta[openai_index] = ToolCallMeta(
            raw_id=raw_id,
            seen_name=bool(function.get("name")),
        )

        return state

    def _update_identity(self, *, openai_index: int, state: ToolCallState, item: dict[str, Any], function: dict[str, Any]) -> None:
        meta = self.meta[openai_index]

        raw_id = item.get("id")
        if raw_id and raw_id != meta.raw_id:
            meta.raw_id = raw_id
            state.tool_id = anthropic_tool_id(raw_id)

        name = function.get("name")
        if name and (not meta.seen_name or state.name.startswith("tool_")):
            _validate_tool_name(name)
            meta.seen_name = True
            state.name = name


def _validate_tool_name(name: Any) -> str:
    if not isinstance(name, str) or not TOOL_NAME_RE.fullmatch(name):
        raise ProtocolError(
            f"Invalid tool name: {name!r}. Expected pattern: ^[a-zA-Z0-9_-]{{1,64}}$",
            code="invalid_tool_name",
        )
    return name


def _content_block_text(item: Any) -> str:
    if isinstance(item, str):
        return item

    if not isinstance(item, dict):
        return as_json_text(item)

    if item.get("type") == "text":
        return item.get("text", "")

    return as_json_text(item)
