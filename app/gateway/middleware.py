"""Gateway middlewares."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.tools.http import json_error


class BodyLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                size = int(content_length)
            except ValueError:
                size = 0

            if size > self.max_bytes:
                return json_error(
                    413,
                    f"Request body too large: {size} bytes > {self.max_bytes} bytes",
                    "request_entity_too_large",
                )

        return await call_next(request)
