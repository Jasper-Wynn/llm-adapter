import asyncio
import json

from app.adapters.anthropic import AnthropicAdapter
from app.models.request import RequestContext
from app.protocol.anthropic_codec import AnthropicCodec


def test_openai_response_to_anthropic_tool_use_and_reasoning_restored():
    resp = {
        "id": "chatcmpl_1",
        "choices": [
            {
                "message": {
                    "content": "I will call a tool.",
                    "reasoning_content": "thinking...",
                    "tool_calls": [
                        {
                            "id": "call_search_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": json.dumps({"q": "abc"})},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    }

    out = AnthropicCodec.response_from_openai(resp, model="claude-test", thinking=True)

    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0]["type"] == "thinking"
    assert out["content"][1] == {"type": "text", "text": "I will call a tool."}
    assert out["content"][2]["type"] == "tool_use"
    assert out["content"][2]["name"] == "search"
    assert out["content"][2]["input"] == {"q": "abc"}
    assert out["usage"]["input_tokens"] == 7
    assert out["usage"]["cache_read_input_tokens"] == 3


class FakeProvider:
    async def stream_sse(self, request, *, request_id=None, client_headers=None):
        del request, request_id, client_headers
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "hello"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "search", "arguments": '{"q"'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ':"abc"}'}}
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        for chunk in chunks:
            yield type("Event", (), {"status_code": 200, "data": json.dumps(chunk), "event": None})()
        yield type("Event", (), {"status_code": 200, "data": "[DONE]", "event": None})()


def _parse_sse_events(raw: str):
    events = []
    for item in raw.strip().split("\n\n"):
        if not item:
            continue
        event_name = None
        data_lines = []
        for line in item.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def test_anthropic_stream_converts_openai_sse_to_anthropic_events():
    adapter = AnthropicAdapter(provider=FakeProvider(), drop_thinking_block=False, throttle_text_delta_ms=0)
    ctx = RequestContext(request_id="req_test", path="/v1/messages", model="claude-test", stream=True, client="test")

    async def run():
        parts = []
        async for part in adapter._stream({"model": "claude-test", "messages": [], "stream": True}, model="claude-test", ctx=ctx):
            parts.append(part)
        return "".join(parts)

    raw = asyncio.run(run())
    events = _parse_sse_events(raw)
    names = [name for name, _ in events]

    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert "content_block_delta" in names
    assert "content_block_stop" in names
    assert names[-2] == "message_delta"
    assert names[-1] == "message_stop"

    deltas = [payload["delta"] for name, payload in events if name == "content_block_delta"]
    assert {"type": "thinking_delta", "thinking": "think"} in deltas
    assert {"type": "text_delta", "text": "hello"} in deltas
    assert any(delta.get("type") == "input_json_delta" for delta in deltas)
