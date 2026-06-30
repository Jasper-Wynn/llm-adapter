"""Gateway middlewares."""
from __future__ import annotations

from typing import Callable, Awaitable

from app.tools.jsonx import dumps


class BodyLimitMiddleware:
    """ASGI request body limiter.

    Unlike the previous Content-Length-only check, this also counts chunked bodies.
    The body is buffered and replayed to downstream FastAPI. That is fine here because
    chat completion requests are JSON bodies, not huge file uploads.
    """

    def __init__(self, app, *, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
        content_length = headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
            except ValueError:
                size = 0
            if size > self.max_bytes:
                await _send_json_error(send, 413, f"Request body too large: {size} bytes > {self.max_bytes} bytes")
                return

        messages = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") == "http.disconnect":
                break
            if message.get("type") == "http.request":
                body = message.get("body", b"") or b""
                total += len(body)
                if total > self.max_bytes:
                    await _send_json_error(send, 413, f"Request body too large: {total} bytes > {self.max_bytes} bytes")
                    return
                if not message.get("more_body", False):
                    break

        index = 0

        async def replay_receive():
            nonlocal index
            if index < len(messages):
                msg = messages[index]
                index += 1
                return msg

            # Important for StreamingResponse:
            # after replaying the buffered request body, delegate back to the
            # original ASGI receive so Starlette can block while waiting for
            # a real http.disconnect event. Returning a synthetic empty
            # http.request forever creates a tight loop in Starlette's
            # disconnect listener and can prevent the stream generator from
            # running, so the upstream request appears to never be sent.
            return await receive()

        await self.app(scope, replay_receive, send)


async def _send_json_error(send, status_code: int, message: str) -> None:
    body = dumps({"error": {"type": "request_entity_too_large", "message": message}}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"application/json; charset=utf-8"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
