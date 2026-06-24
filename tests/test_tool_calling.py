from __future__ import annotations

import pytest

from app.protocol.exceptions import ProtocolError
from app.protocol.tool_calling import ToolCallingCodec, ToolCallStream


def test_anthropic_tool_to_openai():
    tools = [
        {
            "name": "read_file",
            "description": "Read file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]

    converted = ToolCallingCodec.anthropic_tools_to_openai(tools)

    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "read_file"
    assert converted[0]["function"]["parameters"]["required"] == ["path"]


def test_invalid_tool_name_rejected():
    with pytest.raises(ProtocolError):
        ToolCallingCodec.anthropic_tools_to_openai(
            [{"name": "bad name", "input_schema": {"type": "object"}}]
        )


def test_server_tool_rejected():
    with pytest.raises(ProtocolError):
        ToolCallingCodec.anthropic_tools_to_openai(
            [{"type": "web_search_20260209", "name": "web_search"}]
        )


def test_tool_choice_to_openai():
    out = ToolCallingCodec.anthropic_choice_to_openai(
        {"type": "tool", "name": "read_file", "disable_parallel_tool_use": True}
    )

    assert out["tool_choice"]["function"]["name"] == "read_file"
    assert out["parallel_tool_calls"] is False


def test_tool_call_stream_accumulates_id_name_arguments():
    stream = ToolCallStream()
    next_index = iter([0, 1]).__next__

    stream.apply(
        [
            {
                "index": 0,
                "function": {"arguments": "{\"path\""},
            }
        ],
        next_index,
    )
    stream.apply(
        [
            {
                "index": 0,
                "id": "call_abc",
                "function": {"name": "read_file", "arguments": ":\"a.py\"}"},
            }
        ],
        next_index,
    )

    calls = stream.ordered()
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].tool_id == "toolu_abc"
    assert calls[0].arguments == '{"path":"a.py"}'
