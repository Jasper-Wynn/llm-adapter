from __future__ import annotations

import asyncio
import json

from app.adapters.anthropic import AnthropicAdapter
from app.models.request import RequestContext
from app.models.sse import SSEChunk


class ToolCallProvider:
    async def stream_sse(self, payload, request_id=None):
        yield SSEChunk(
            status_code=200,
            event=None,
            data=json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_abc",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path"',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
        )
        await asyncio.sleep(0.2)
        yield SSEChunk(
            status_code=200,
            event=None,
            data=json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": ':"a.py"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
        )


def test_anthropic_tool_call_stream_emits_block_before_finish():
    async def run():
        adapter = AnthropicAdapter(provider=ToolCallProvider(), drop_thinking_block=False, throttle_text_delta_ms=0)
        path = "/v1/messages"
        ctx = RequestContext(request_id="test-request", path=path, model="test-model", stream=True, client=None)
        response = await adapter.messages(
            {
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "read"}],
            },
            ctx,
        )
        chunks = response.body_iterator.__aiter__()

        first = await asyncio.wait_for(anext(chunks), timeout=0.1)
        second = await asyncio.wait_for(anext(chunks), timeout=0.1)

        assert "event: message_start" in first
        assert "event: content_block_start" in second
        assert '"type":"tool_use"' in second
        assert '"name":"read_file"' in second

    asyncio.run(run())
