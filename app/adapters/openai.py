"""OpenAI Chat Completions adapter."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing
from typing import Any, AsyncIterator

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.request import RequestContext
from app.protocol.sse import DONE, SSE
from app.protocol.upstream_error import UpstreamErrorCodec
from app.tools.http import STREAM_HEADERS, bool_value, with_request_id


class OpenAIAdapter:
    def __init__(self, *, provider, stream_registry=None, logger: logging.Logger | None = None):
        self.provider = provider
        self.stream_registry = stream_registry
        self.log = logger or logging.getLogger(__name__)

    async def chat_completions(self, body: dict[str, Any], ctx: RequestContext):
        stream = bool_value(body.get("stream"))
        payload = {**body, "stream": stream}
        self.log.info(
            "[%s] openai request model=%s stream=%s tools=%s messages=%s",
            ctx.request_id,
            body.get("model"),
            stream,
            bool(body.get("tools")),
            len(body.get("messages", [])),
        )
        if not stream:
            return await self._non_stream(payload, ctx)
        return StreamingResponse(
            self._stream(payload, ctx),
            media_type="text/event-stream",
            headers=with_request_id(STREAM_HEADERS, ctx.request_id),
        )

    async def _non_stream(self, payload: dict[str, Any], ctx: RequestContext) -> JSONResponse:
        try:
            data = await self.provider.create_completion(payload, request_id=ctx.request_id, client_headers=ctx.client_headers)
            upstream_error = UpstreamErrorCodec.from_payload(data)
            if upstream_error:
                response = JSONResponse(status_code=upstream_error.status_code, content=upstream_error.to_openai())
            else:
                response = JSONResponse(content=data)
            response.headers["X-Request-ID"] = ctx.request_id
            return response
        except httpx.HTTPStatusError as exc:
            response = _openai_http_status_response(exc, ctx.request_id)
            return response
        except httpx.TimeoutException as exc:
            response = JSONResponse(status_code=504, content={"error": {"type": "upstream_timeout", "message": str(exc), "code": "timeout"}})
            response.headers["X-Request-ID"] = ctx.request_id
            return response
        except httpx.HTTPError as exc:
            response = JSONResponse(status_code=502, content={"error": {"type": "upstream_error", "message": str(exc), "code": "http_error"}})
            response.headers["X-Request-ID"] = ctx.request_id
            return response
        except Exception as exc:
            response = JSONResponse(status_code=500, content={"error": {"type": "adapter_error", "message": str(exc), "code": "adapter_error"}})
            response.headers["X-Request-ID"] = ctx.request_id
            return response

    async def _stream(self, payload: dict[str, Any], ctx: RequestContext) -> AsyncIterator[str]:
        done_sent = False
        close_reason = "unknown"
        if self.stream_registry:
            self.stream_registry.register(ctx)
        self.log.info("[%s] stream.start protocol=openai %s", ctx.request_id, ctx.stream_label())
        try:
            async with aclosing(self.provider.stream_sse(payload, request_id=ctx.request_id, client_headers=ctx.client_headers)) as stream_iter:
                async for event in stream_iter:
                    if event.status_code >= 400:
                        close_reason = "upstream_error"
                        yield _openai_error("upstream_error", f"Upstream returned {event.status_code}: {event.data}", code=str(event.status_code))
                        yield _done()
                        done_sent = True
                        return
                    if not event.data:
                        continue
                    if event.data == DONE:
                        close_reason = "completed"
                        yield _done()
                        done_sent = True
                        return
                    parsed = _try_json(event.data)
                    upstream_error = UpstreamErrorCodec.from_payload(parsed)
                    if upstream_error:
                        close_reason = "upstream_error"
                        self.log.warning(
                            "[%s] upstream stream error type=%s code=%s message=%s",
                            ctx.request_id,
                            upstream_error.type,
                            upstream_error.code,
                            upstream_error.message,
                        )
                        yield SSE.encode(None, upstream_error.to_openai())
                        yield _done()
                        done_sent = True
                        return
                    yield f"data: {event.data}\n\n"
            if not done_sent:
                close_reason = "provider_exhausted"
                yield _done()
                done_sent = True
            elif close_reason == "unknown":
                close_reason = "completed"
        except asyncio.CancelledError:
            close_reason = "client_cancelled"
            self.log.info("[%s] stream.cancelled protocol=openai source=client %s elapsed=%.1fms", ctx.request_id, ctx.stream_label(), ctx.elapsed_ms())
            raise
        except httpx.TimeoutException as exc:
            close_reason = "timeout"
            yield _openai_error("upstream_timeout", str(exc), code="timeout")
            if not done_sent:
                yield _done()
                done_sent = True
        except httpx.HTTPError as exc:
            close_reason = "http_error"
            yield _openai_error("upstream_error", str(exc), code="http_error")
            if not done_sent:
                yield _done()
                done_sent = True
        except Exception as exc:
            close_reason = "adapter_error"
            yield _openai_error("adapter_error", str(exc), code="adapter_error")
            if not done_sent:
                yield _done()
                done_sent = True
        finally:
            if close_reason == "unknown":
                close_reason = "consumer_closed"
            cancelled_children = 0
            if self.stream_registry:
                if close_reason == "client_cancelled":
                    cancelled_children = self.stream_registry.cancel_descendants(ctx)
                self.stream_registry.unregister(ctx)
            self.log.info(
                "[%s] stream.closed protocol=openai reason=%s done_sent=%s cancelled_children=%s %s elapsed=%.1fms",
                ctx.request_id,
                close_reason,
                done_sent,
                cancelled_children,
                ctx.stream_label(),
                ctx.elapsed_ms(),
            )


def _try_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _openai_error(error_type: str, message: str, *, code: str | None = None) -> str:
    return SSE.encode(None, {"error": {"type": error_type, "message": message, "code": code}})


def _done() -> str:
    return f"data: {DONE}\n\n"


def _openai_http_status_response(exc: httpx.HTTPStatusError, request_id: str) -> JSONResponse:
    payload = _response_json(exc.response)
    upstream_error = UpstreamErrorCodec.from_payload(payload)
    status_code = exc.response.status_code
    if upstream_error:
        content = upstream_error.to_openai()
        content["error"]["code"] = content["error"].get("code") or str(status_code)
    else:
        content = {
            "error": {
                "type": "upstream_http_error",
                "message": exc.response.text[:1000],
                "code": str(status_code),
            }
        }
    response = JSONResponse(status_code=status_code, content=content)
    response.headers["X-Request-ID"] = request_id
    return response


def _response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return None
