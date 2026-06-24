"""Anthropic Messages adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.request import RequestContext
from app.protocol.anthropic_codec import AnthropicCodec, STOP_REASON
from app.protocol.anthropic_stream import AnthropicStream
from app.protocol.exceptions import ProtocolError
from app.protocol.tool_calling import ToolCallStream
from app.protocol.upstream_error import UpstreamErrorCodec
from app.tools.http import STREAM_HEADERS, bool_value, with_request_id

Logger = logging.Logger | None


class AnthropicAdapter:
    def __init__(self, *, provider, drop_thinking_block: bool, throttle_text_delta_ms: int, logger: Logger = None):
        self.provider = provider
        self.drop_thinking_block = drop_thinking_block
        self.throttle_text_delta_ms = throttle_text_delta_ms
        self.log = logger or logging.getLogger(__name__)

    async def messages(self, body: dict[str, Any], ctx: RequestContext):
        stream = bool_value(body.get("stream"))

        try:
            request = AnthropicCodec.request_to_openai(body)
        except ProtocolError as exc:
            response = JSONResponse(status_code=exc.status_code, content=exc.to_anthropic())
            response.headers["X-Request-ID"] = ctx.request_id
            return response

        request["stream"] = stream

        self.log.info(
            "[%s] anthropic request model=%s stream=%s tools=%s messages=%s",
            ctx.request_id,
            body.get("model"),
            stream,
            bool(body.get("tools")),
            len(body.get("messages", [])),
        )

        if not stream:
            return await self._non_stream(request, body, ctx)

        return StreamingResponse(
            self._stream(request, model=body.get("model", ""), ctx=ctx),
            media_type="text/event-stream",
            headers=with_request_id(STREAM_HEADERS, ctx.request_id),
        )

    async def _non_stream(self, request: dict[str, Any], body: dict[str, Any], ctx: RequestContext) -> JSONResponse:
        try:
            completion = await self.provider.create_completion(request, request_id=ctx.request_id)

            upstream_error = UpstreamErrorCodec.from_payload(completion)

            if upstream_error:
                response = JSONResponse(
                    status_code=upstream_error.status_code,
                    content=upstream_error.to_anthropic(),
                )
            else:
                response = JSONResponse(
                    content=AnthropicCodec.response_from_openai(
                        completion,
                        model=body.get("model", ""),
                        thinking=_thinking_enabled(body),
                    ),
                    headers={"anthropic-version": "2023-06-01"},
                )

            response.headers["X-Request-ID"] = ctx.request_id
            return response

        except ProtocolError as exc:
            response = JSONResponse(status_code=exc.status_code, content=exc.to_anthropic())
            response.headers["X-Request-ID"] = ctx.request_id
            return response

        except httpx.HTTPStatusError as exc:
            response = JSONResponse(
                status_code=exc.response.status_code,
                content={
                    "type": "error",
                    "error": {
                        "type": "upstream_http_error",
                        "message": exc.response.text,
                    },
                },
            )
            response.headers["X-Request-ID"] = ctx.request_id
            return response

    async def _stream(self, request: dict[str, Any], *, model: str, ctx: RequestContext):
        writer = AnthropicStream(model)
        tool_calls = ToolCallStream()

        try:
            async for event in self.provider.stream_sse(request, request_id=ctx.request_id):
                if writer.closed:
                    break

                if event.status_code >= 400:
                    yield writer.error(f"Upstream returned {event.status_code}: {event.data}")
                    return

                if not event.data or event.data == "[DONE]":
                    continue

                try:
                    chunk = json.loads(event.data)
                except json.JSONDecodeError:
                    continue

                upstream_error = UpstreamErrorCodec.from_payload(chunk)

                if upstream_error:
                    self.log.warning(
                        "[%s] upstream stream error type=%s code=%s message=%s",
                        ctx.request_id,
                        upstream_error.type,
                        upstream_error.code,
                        upstream_error.message,
                    )
                    yield writer.error(upstream_error.to_anthropic_message())
                    return

                writer.update_usage(chunk.get("usage"))

                for item in writer.start():
                    yield item

                choice = (chunk.get("choices") or [{}])[0] or {}
                delta = choice.get("delta") or {}

                try:
                    for item in self._emit_delta(writer, tool_calls, delta):
                        yield item
                    for item in self._emit_tool_call_progress(writer, tool_calls):
                        yield item
                except ProtocolError as exc:
                    yield writer.error(exc.message, exc.error_type)
                    return

                if delta.get("content") and self.throttle_text_delta_ms > 0:
                    await asyncio.sleep(self.throttle_text_delta_ms / 1000)

                finish_reason = choice.get("finish_reason")

                if finish_reason:
                    for item in self._flush_tool_calls(writer, tool_calls):
                        yield item

                    for item in writer.finish(STOP_REASON.get(finish_reason, "end_turn")):
                        yield item

                    self.log.info(
                        "[%s] anthropic stream finished reason=%s elapsed=%.1fms",
                        ctx.request_id,
                        finish_reason,
                        ctx.elapsed_ms(),
                    )
                    break

        except asyncio.CancelledError:
            self.log.info(
                "[%s] anthropic stream cancelled elapsed=%.1fms",
                ctx.request_id,
                ctx.elapsed_ms(),
            )
            raise

        except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            yield writer.error(f"Upstream timeout: {exc}", "timeout_error")

        except httpx.HTTPError as exc:
            yield writer.error(f"Upstream error: {exc}")

        if not writer.closed:
            for item in self._flush_tool_calls(writer, tool_calls):
                yield item

            for item in writer.finish("end_turn"):
                yield item

    def _emit_delta(self, writer: AnthropicStream, tool_calls: ToolCallStream, delta: dict[str, Any]):
        if not self.drop_thinking_block:
            thinking = delta.get("thinking")

            if isinstance(thinking, dict):
                if thinking.get("content"):
                    yield from writer.thinking(thinking["content"])

                if thinking.get("signature"):
                    yield from writer.signature(thinking["signature"])

            elif isinstance(thinking, str):
                yield from writer.thinking(thinking)

            if delta.get("reasoning_content"):
                yield from writer.thinking(delta["reasoning_content"])

        if delta.get("content"):
            yield from writer.text(delta["content"])

        if delta.get("tool_calls"):
            tool_calls.apply(delta["tool_calls"], writer.next_index)

    def _flush_tool_calls(self, writer: AnthropicStream, tool_calls: ToolCallStream):
        if not tool_calls.has_calls():
            return

        for state in tool_calls.ordered():
            if not state.started:
                yield from writer.tool_start(state)
                state.started = True

            pending = state.pending_arguments()
            if pending:
                yield from writer.tool_delta(state, pending)
                state.mark_arguments_emitted()

            yield from writer.close_block()

    def _emit_tool_call_progress(self, writer: AnthropicStream, tool_calls: ToolCallStream):
        if not tool_calls.has_calls():
            return

        for state in tool_calls.ordered():
            meta = tool_calls.meta[state.openai_index]

            if not meta.seen_name:
                continue

            if not state.started:
                yield from writer.tool_start(state)
                state.started = True

            pending = state.pending_arguments()
            if pending:
                yield from writer.tool_delta(state, pending)
                state.mark_arguments_emitted()


def _thinking_enabled(body: dict[str, Any]) -> bool:
    thinking = body.get("thinking")

    if not thinking:
        return False

    if isinstance(thinking, dict):
        return thinking.get("type") != "disabled"

    return bool(thinking)
