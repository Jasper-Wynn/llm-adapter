"""OpenAI Chat Completions adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.request import RequestContext
from app.protocol.sse import DONE, SSE
from app.protocol.upstream_error import UpstreamErrorCodec
from app.tools.http import STREAM_HEADERS, bool_value, with_request_id


class OpenAIAdapter:
    def __init__(self, *, provider, logger: logging.Logger | None = None):
        self.provider = provider
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
            data = await self.provider.create_completion(
                payload,
                request_id=ctx.request_id,
            )

            upstream_error = UpstreamErrorCodec.from_payload(data)

            if upstream_error:
                response = JSONResponse(
                    status_code=upstream_error.status_code,
                    content=upstream_error.to_openai(),
                )
            else:
                response = JSONResponse(content=data)

            response.headers["X-Request-ID"] = ctx.request_id
            return response

        except httpx.HTTPStatusError as exc:
            response = JSONResponse(
                status_code=exc.response.status_code,
                content={
                    "error": {
                        "type": "upstream_http_error",
                        "message": exc.response.text,
                        "code": str(exc.response.status_code),
                    }
                },
            )
            response.headers["X-Request-ID"] = ctx.request_id
            return response

        except httpx.TimeoutException as exc:
            response = JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "type": "upstream_timeout",
                        "message": str(exc),
                        "code": "timeout",
                    }
                },
            )
            response.headers["X-Request-ID"] = ctx.request_id
            return response

        except httpx.HTTPError as exc:
            response = JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "type": "upstream_error",
                        "message": str(exc),
                        "code": "http_error",
                    }
                },
            )
            response.headers["X-Request-ID"] = ctx.request_id
            return response

    async def _stream(self, payload: dict[str, Any], ctx: RequestContext) -> AsyncIterator[str]:
        done_sent = False

        try:
            async for event in self.provider.stream_sse(
                payload,
                request_id=ctx.request_id,
            ):
                if event.status_code >= 400:
                    yield _openai_error(
                        "upstream_error",
                        f"Upstream returned {event.status_code}: {event.data}",
                        code=str(event.status_code),
                    )
                    yield _done()
                    done_sent = True
                    return

                if not event.data:
                    continue

                if event.data == DONE:
                    yield _done()
                    done_sent = True
                    return

                parsed = _try_json(event.data)
                upstream_error = UpstreamErrorCodec.from_payload(parsed)

                if upstream_error:
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

        except asyncio.CancelledError:
            self.log.info(
                "[%s] openai stream cancelled elapsed=%.1fms",
                ctx.request_id,
                ctx.elapsed_ms(),
            )
            raise

        except httpx.TimeoutException as exc:
            yield _openai_error("upstream_timeout", str(exc), code="timeout")

        except httpx.HTTPError as exc:
            yield _openai_error("upstream_error", str(exc), code="http_error")

        except Exception as exc:
            yield _openai_error("adapter_error", str(exc), code="adapter_error")

        finally:
            if not done_sent:
                yield _done()

            self.log.info(
                "[%s] openai stream closed elapsed=%.1fms",
                ctx.request_id,
                ctx.elapsed_ms(),
            )


def _try_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _openai_error(error_type: str, message: str, *, code: str | None = None) -> str:
    return SSE.encode(
        None,
        {
            "error": {
                "type": error_type,
                "message": message,
                "code": code,
            }
        },
    )


def _done() -> str:
    return f"data: {DONE}\n\n"
