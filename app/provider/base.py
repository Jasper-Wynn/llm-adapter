"""Provider interface used by adapters."""

from __future__ import annotations

from typing import Any, AsyncIterator

from app.models.sse import SSEChunk


class BaseProvider:
    BASE_URL = ""
    CHAT_ENDPOINT = "/v1/chat/completions"
    MODELS_ENDPOINT = "/v1/models"
    TIMEOUT = 300000
    VERIFY_SSL = True
    DEBUG_STREAM = False

    AUTH_TOKEN = ""
    AUTH_HEADER = "Authorization"
    AUTH_SCHEME = "Bearer"
    HEADERS: dict[str, str] = {}
    MODELS_QUERY: dict[str, Any] = {}
    STREAM_OPTIONS: dict[str, Any] | None = {"include_usage": True}

    def headers(self, *, request_id: str | None = None, stream: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json", "Accept-Encoding": "identity", "Cache-Control": "no-cache"}
        headers.update({str(key): str(value) for key, value in self.HEADERS.items()})

        if self.AUTH_TOKEN:
            value = f"{self.AUTH_SCHEME} {self.AUTH_TOKEN}" if self.AUTH_SCHEME else self.AUTH_TOKEN
            headers[self.AUTH_HEADER] = value

        return headers

    async def get_models(self, *, check_permission: bool = True) -> list[dict[str, Any]]:
        """Return upstream model metadata."""
        raise NotImplementedError

    async def create_completion(self, payload: dict[str, Any], *, request_id: str | None = None) -> dict[str, Any]:
        """Create a non-streaming chat completion."""
        raise NotImplementedError

    async def stream_sse(self, payload: dict[str, Any], *, request_id: str | None = None) -> AsyncIterator[SSEChunk]:
        """Stream upstream SSE chunks."""
        raise NotImplementedError
        yield

    async def aclose(self) -> None:
        """Close provider resources."""
        raise NotImplementedError
