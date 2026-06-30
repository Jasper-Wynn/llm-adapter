"""Anthropic <-> OpenAI protocol conversion."""

from __future__ import annotations

from typing import Any

from app.protocol.reasoning import ReasoningCodec
from app.protocol.tool_calling import ToolCallingCodec
from app.protocol.usage import UsageCodec
from app.tools.http import bool_value
from app.tools.ids import message_id


STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
    None: "end_turn",
}


class AnthropicCodec:
    @staticmethod
    def request_to_openai(req: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": req.get("model"),
            "messages": [],
        }

        _copy(out, req, "max_tokens")
        _copy(out, req, "temperature")
        _copy(out, req, "top_p")

        if req.get("stream") is not None:
            out["stream"] = bool_value(req["stream"])

        if req.get("stop_sequences"):
            out["stop"] = req["stop_sequences"]

        if req.get("system"):
            out["messages"].extend(_system(req["system"]))

        for msg in req.get("messages", []):
            role = msg.get("role")
            if role == "system":
                out["messages"].extend(_system(msg.get("content", "")))
            elif role == "user":
                out["messages"].extend(_user(msg))
            elif role == "assistant":
                out["messages"].append(_assistant(msg))

        if req.get("tools"):
            out["tools"] = ToolCallingCodec.anthropic_tools_to_openai(req["tools"])

        if req.get("tool_choice"):
            out.update(ToolCallingCodec.anthropic_choice_to_openai(req["tool_choice"]))

        reasoning = ReasoningCodec.anthropic_to_upstream(
            req.get("thinking"),
            output_config=req.get("output_config"),
            max_tokens=req.get("max_tokens"),
        )
        if reasoning:
            out.update(reasoning)

        return out

    @staticmethod
    def response_from_openai(resp: dict[str, Any], *, model: str, thinking: bool = False) -> dict[str, Any]:
        choice = (resp.get("choices") or [{}])[0] or {}
        msg = choice.get("message") or {}

        content: list[dict[str, Any]] = []

        if thinking:
            block = ReasoningCodec.to_anthropic_block(
                ReasoningCodec.extract_from_message(msg)
            )
            if block:
                content.append(block)

        if msg.get("content"):
            content.append({"type": "text", "text": msg["content"]})

        for call in msg.get("tool_calls") or []:
            content.append(ToolCallingCodec.openai_call_to_anthropic_use(call))

        return {
            "id": message_id(),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": STOP_REASON.get(choice.get("finish_reason"), "end_turn"),
            "stop_sequence": None,
            "usage": UsageCodec.from_openai(resp.get("usage")).as_dict(),
        }


def _copy(out: dict[str, Any], src: dict[str, Any], key: str) -> None:
    if src.get(key) is not None:
        out[key] = src[key]


def _system(system: str | list[dict[str, Any]]) -> list[dict[str, str]]:
    if isinstance(system, str):
        return [{"role": "system", "content": system}]

    text = "\n".join(
        item.get("text", "")
        for item in system or []
        if isinstance(item, dict) and item.get("type") == "text"
    )

    return [{"role": "system", "content": text}] if text else []


def _user(msg: dict[str, Any]) -> list[dict[str, Any]]:
    content = msg.get("content", "")

    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    result: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal blocks
        if blocks:
            result.append({"role": "user", "content": _collapse(blocks)})
            blocks = []

    for item in content or []:
        if not isinstance(item, dict):
            blocks.append({"type": "text", "text": str(item)})
            continue

        kind = item.get("type")

        if kind == "tool_result":
            flush()
            result.append(ToolCallingCodec.anthropic_result_to_openai(item))
        elif kind == "text":
            blocks.append({"type": "text", "text": item.get("text", "")})
        elif kind == "image":
            image = _image(item)
            if image:
                blocks.append(image)

    flush()
    return result


def _assistant(msg: dict[str, Any]) -> dict[str, Any]:
    content = msg.get("content", "")

    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    text: list[str] = []
    calls: list[dict[str, Any]] = []

    for item in content or []:
        if not isinstance(item, dict):
            text.append(str(item))
            continue

        if item.get("type") == "text":
            text.append(item.get("text", ""))
        elif item.get("type") == "tool_use":
            calls.append(ToolCallingCodec.anthropic_use_to_openai_call(item))

    out: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text),
    }

    if calls:
        out["tool_calls"] = calls

    return out


def _collapse(blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if all(item.get("type") == "text" for item in blocks):
        return "".join(item.get("text", "") for item in blocks)
    return blocks


def _image(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") or {}

    if source.get("type") == "base64":
        media = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media};base64,{data}",
            },
        }

    if source.get("type") == "url":
        return {
            "type": "image_url",
            "image_url": {
                "url": source.get("url", ""),
            },
        }

    return None
