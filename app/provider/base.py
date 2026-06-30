"""Provider base class.

A provider is the only place that describes how to call one upstream.
The base class supplies small HTTP/header helpers and default values only.
It never reads global upstream settings and never overwrites subclass class attrs.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Mapping

import httpx

from app import config as cfg
from app.models.sse import SSEEvent


class BaseProvider:
    # Upstream location. Concrete providers should normally override these.
    BASE_URL = ""
    CHAT_ENDPOINT = "/v1/chat/completions"
    MODELS_ENDPOINT = "/v1/models"

    # Provider HTTP defaults. Subclasses can override any of these.
    TIMEOUT = cfg.HTTP_TIMEOUT
    CONNECT_TIMEOUT = cfg.HTTP_CONNECT_TIMEOUT
    VERIFY_SSL = cfg.HTTP_VERIFY_SSL
    TRUST_ENV = cfg.HTTP_TRUST_ENV
    HTTP2 = cfg.HTTP2
    MAX_CONNECTIONS = cfg.HTTP_MAX_CONNECTIONS
    MAX_KEEPALIVE_CONNECTIONS = cfg.HTTP_MAX_KEEPALIVE_CONNECTIONS
    KEEPALIVE_EXPIRY = cfg.HTTP_KEEPALIVE_EXPIRY

    # Provider-owned static headers. Applied after client header passthrough.
    HEADERS: dict[str, str] = {}

    # Client headers are forwarded by default. Only hop-by-hop / transport
    # headers are blocked globally. Subclasses may append to this tuple.
    CLIENT_HEADER_BLACKLIST: tuple[str, ...] = cfg.CLIENT_HEADER_BLACKLIST

    # Optional provider-owned model mapping/static model list.
    MODEL_MAP: dict[str, str] = {}
    STATIC_MODELS: tuple[str, ...] = ()

    def __init__(self, *, logger: logging.Logger | None = None):
        self.log = logger or logging.getLogger(__name__)

    def describe(self) -> dict[str, Any]:
        provider_headers = self.provider_headers({})
        return {
            "provider": type(self).__name__,
            "base_url": self.BASE_URL,
            "chat_endpoint": self.CHAT_ENDPOINT,
            "models_endpoint": self.MODELS_ENDPOINT,
            "timeout": self.TIMEOUT,
            "connect_timeout": self.CONNECT_TIMEOUT,
            "verify_ssl": self.VERIFY_SSL,
            "trust_env": self.TRUST_ENV,
            "http2": self.HTTP2,
            "max_connections": self.MAX_CONNECTIONS,
            "max_keepalive_connections": self.MAX_KEEPALIVE_CONNECTIONS,
            "keepalive_expiry": self.KEEPALIVE_EXPIRY,
            "client_header_blacklist": self.CLIENT_HEADER_BLACKLIST,
            "provider_header_names": cfg.safe_header_names(provider_headers),
            "provider_headers": cfg.safe_headers(provider_headers),
        }

    def build_http_client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(
            float(self.TIMEOUT),
            connect=min(float(self.CONNECT_TIMEOUT), float(self.TIMEOUT)),
            read=float(self.TIMEOUT),
            write=float(self.TIMEOUT),
        )
        limits = httpx.Limits(
            max_connections=int(self.MAX_CONNECTIONS),
            max_keepalive_connections=int(self.MAX_KEEPALIVE_CONNECTIONS),
            keepalive_expiry=float(self.KEEPALIVE_EXPIRY),
        )
        return httpx.AsyncClient(
            http2=bool(self.HTTP2),
            verify=bool(self.VERIFY_SSL),
            trust_env=bool(self.TRUST_ENV),
            timeout=timeout,
            limits=limits,
        )

    def headers(
        self,
        *,
        request_id: str | None = None,
        stream: bool = False,
        client_headers: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        incoming = client_headers or {}
        headers: dict[str, str] = {}

        # 1) Forward client headers by default, except blacklist.
        self._update_headers(headers, self.forward_client_headers(incoming))

        # 2) Provider headers add/override normal HTTP headers.
        self._update_headers(headers, self.provider_headers(incoming))

        # 3) Adapter-owned headers are set last. They must not be inherited
        # from clients because OpenAI and Anthropic clients send different
        # Accept values. For upstream OpenAI-compatible streaming we need SSE.
        self._update_headers(headers, {cfg.HEADER_CONTENT_TYPE: cfg.MEDIA_TYPE_JSON})
        self._update_headers(headers, {"Accept": cfg.MEDIA_TYPE_SSE if stream else cfg.MEDIA_TYPE_JSON})

        # 4) Adapter request id wins over client X-Request-ID.
        if request_id:
            self._update_headers(headers, {cfg.HEADER_REQUEST_ID: request_id})
        return headers

    def provider_headers(self, client_headers: Mapping[str, str]) -> dict[str, str]:
        """Return upstream headers required by this provider."""
        return dict(self.HEADERS)

    @staticmethod
    def _update_headers(headers: dict[str, str], updates: Mapping[str, str]) -> None:
        """Case-insensitive dict update for HTTP headers."""
        for key, value in updates.items():
            if value is None:
                continue
            lower = str(key).lower()
            for existing in list(headers.keys()):
                if existing.lower() == lower and existing != key:
                    headers.pop(existing, None)
            headers[str(key)] = str(value)

    def forward_client_headers(self, client_headers: Mapping[str, str]) -> dict[str, str]:
        if not client_headers:
            return {}
        blocked = {item.lower() for item in self.CLIENT_HEADER_BLACKLIST}
        return {
            str(key): str(value)
            for key, value in client_headers.items()
            if value is not None and str(key).lower() not in blocked
        }

    @staticmethod
    def client_header(client_headers: Mapping[str, str], name: str) -> str | None:
        target = name.lower()
        for key, value in client_headers.items():
            if key.lower() == target:
                return str(value)
        return None

    def map_model(self, model: str | None) -> str | None:
        if not model:
            return model
        return self.MODEL_MAP.get(model, model)

    def transform_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        mapped = dict(payload)
        if "model" in mapped:
            mapped["model"] = self.map_model(mapped.get("model"))
        return mapped

    async def create_completion(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        client_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def stream_sse(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        client_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[SSEEvent]:
        raise NotImplementedError

    async def get_models(self, *, check_permission: bool = True) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None
