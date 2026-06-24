"""Configurable OpenAI-compatible upstream provider."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

from app.models.sse import SSEChunk
from app.provider.base import BaseProvider
from app.protocol.sse import SSE
from app.tools.http import safe_headers


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, *, base_url: str, chat_endpoint: str, models_endpoint: str, timeout: int, verify_ssl: bool, debug_stream: bool, models_query: dict[str, Any], stream_options: dict[str, Any] | None, logger: logging.Logger | None = None):
        if not base_url:
            raise RuntimeError("BASE_URL must be configured on the provider class.")

        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = chat_endpoint
        self.models_endpoint = models_endpoint
        self.models_query = models_query
        self.stream_options = stream_options
        self.debug_stream = debug_stream
        self.log = logger or logging.getLogger(__name__)

        timeout_seconds = timeout / 1000 if timeout > 1000 else timeout
        timeout_config = httpx.Timeout(connect=30, read=timeout_seconds, write=30, pool=30)

        client_options = {"http2": True, "verify": verify_ssl, "timeout": timeout_config, "trust_env": False}
        self.client = httpx.AsyncClient(**client_options)
        self.stream_client = httpx.AsyncClient(**client_options)

    @classmethod
    def from_config(cls, cfg, *, logger: logging.Logger | None = None):
        timeout = getattr(cfg, "PROVIDER_TIMEOUT", cls.TIMEOUT)
        debug_stream = getattr(cfg, "DEBUG_STREAM", cls.DEBUG_STREAM)
        return cls(base_url=cls.BASE_URL, chat_endpoint=cls.CHAT_ENDPOINT, models_endpoint=cls.MODELS_ENDPOINT, timeout=timeout, verify_ssl=cls.VERIFY_SSL, debug_stream=debug_stream, models_query=cls.MODELS_QUERY, stream_options=cls.STREAM_OPTIONS, logger=logger)

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.stream_client.aclose()

    async def get_models(self, *, check_permission: bool = True) -> list[dict[str, Any]]:
        headers = self.headers()
        self._log_outgoing_headers("models", None, headers)
        response = await self.client.get(f"{self.base_url}{self.models_endpoint}", params=self.models_query, headers=headers)
        response.raise_for_status()

        data = response.json()

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                return data["data"]

            if isinstance(data.get("models"), list):
                return data["models"]

        return []

    async def create_completion(self, payload: dict[str, Any], *, request_id: str | None = None) -> dict[str, Any]:
        body = dict(payload)
        body["stream"] = False

        headers = self.headers(request_id=request_id)
        self._log_outgoing_headers("chat", request_id, headers)
        response = await self.client.post(f"{self.base_url}{self.chat_endpoint}", json=body, headers=headers)

        if self.debug_stream:
            self.log.info("[%s] upstream non-stream status=%s http=%s content-type=%s", request_id or "-", response.status_code, response.http_version, response.headers.get("content-type"))

        response.raise_for_status()
        return response.json()

    async def stream_sse(self, payload: dict[str, Any], *, request_id: str | None = None) -> AsyncIterator[SSEChunk]:
        body = dict(payload)
        body["stream"] = True

        if self.stream_options is not None:
            body.setdefault("stream_options", self.stream_options)

        headers = self.headers(request_id=request_id, stream=True)
        self._log_outgoing_headers("stream", request_id, headers)

        async with self.stream_client.stream("POST", f"{self.base_url}{self.chat_endpoint}", json=body, headers=headers) as response:
            if self.debug_stream:
                self.log.info("[%s] upstream stream status=%s http=%s content-type=%s encoding=%s", request_id or "-", response.status_code, response.http_version, response.headers.get("content-type"), response.headers.get("content-encoding"))

            if response.status_code >= 400:
                data = await response.aread()
                yield SSEChunk(status_code=response.status_code, data=data.decode("utf-8", errors="replace"))
                return

            async for chunk in SSE.parse(response.status_code, response.aiter_lines()):
                yield chunk

    def _log_outgoing_headers(self, kind: str, request_id: str | None, headers: dict[str, str]) -> None:
        self.log.debug("[%s] outgoing provider headers kind=%s headers=%s", request_id or "-", kind, safe_headers(headers))
