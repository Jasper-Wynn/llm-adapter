"""OpenAI-compatible upstream provider."""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Mapping
from urllib.parse import urljoin

from app import config as cfg
from app.models.sse import SSEEvent
from app.provider.base import BaseProvider
from app.protocol.sse import DONE


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, *, logger=None):
        super().__init__(logger=logger)
        self.client = self.build_http_client()
        self.stream_client = self.build_http_client()

    async def create_completion(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        client_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        url = self._url(self.CHAT_ENDPOINT)
        body = self.transform_payload(payload)
        headers = self.headers(request_id=request_id, client_headers=client_headers)
        started = time.perf_counter()
        self._log_upstream_request(request_id, "POST", url, body, headers)
        response = await self.client.post(url, json=body, headers=headers)
        elapsed = (time.perf_counter() - started) * 1000
        self.log.info("[%s] upstream response status=%s elapsed=%.1fms", request_id, response.status_code, elapsed)
        if response.status_code >= 400:
            self.log.warning("[%s] upstream error status=%s body=%s", request_id, response.status_code, response.text[:1000])
            response.raise_for_status()
        return response.json()

    async def stream_sse(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
        client_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[SSEEvent]:
        url = self._url(self.CHAT_ENDPOINT)
        body = self.transform_payload({**payload, "stream": True})
        headers = self.headers(request_id=request_id, stream=True, client_headers=client_headers)
        started = time.perf_counter()
        first_chunk = False
        chunks = 0
        close_reason = "unknown"
        response_status: int | None = None

        self._log_upstream_request(request_id, "POST", url, body, headers, stream=True)
        try:
            async with self.stream_client.stream("POST", url, json=body, headers=headers) as response:
                response_status = response.status_code
                elapsed = (time.perf_counter() - started) * 1000
                self.log.info("[%s] upstream.stream.start status=%s elapsed=%.1fms", request_id, response.status_code, elapsed)
                if response.status_code >= 400:
                    close_reason = "upstream_error"
                    text = await response.aread()
                    preview = text.decode("utf-8", errors="replace")[:1000]
                    self.log.warning("[%s] upstream stream error status=%s body=%s", request_id, response.status_code, preview)
                    yield SSEEvent(data=preview, status_code=response.status_code)
                    return

                event_name: str | None = None
                data_lines: list[str] = []

                async for line in response.aiter_lines():
                    if line == DONE:
                        close_reason = "completed"
                    if not first_chunk:
                        first_chunk = True
                        self.log.info(
                            "[%s] upstream.stream.first_chunk elapsed=%.1fms",
                            request_id,
                            (time.perf_counter() - started) * 1000,
                        )
                    if line == "":
                        if data_lines or event_name:
                            data = "\n".join(data_lines)
                            if data == DONE:
                                close_reason = "completed"
                            chunks += 1
                            yield SSEEvent(data=data, event=event_name, status_code=response.status_code)
                        event_name = None
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                        continue
                    # Some gateways emit raw JSONL instead of SSE.
                    if line == DONE or line.startswith("{"):
                        if line == DONE:
                            close_reason = "completed"
                        chunks += 1
                        yield SSEEvent(data=line, status_code=response.status_code)

                if data_lines or event_name:
                    data = "\n".join(data_lines)
                    if data == DONE:
                        close_reason = "completed"
                    chunks += 1
                    yield SSEEvent(data=data, event=event_name, status_code=response.status_code)

                close_reason = "completed"

        except asyncio.CancelledError:
            close_reason = "client_cancelled"
            self.log.info(
                "[%s] upstream.stream.cancelled source=downstream_request action=closing_response elapsed=%.1fms",
                request_id,
                (time.perf_counter() - started) * 1000,
            )
            raise
        except GeneratorExit:
            if close_reason == "unknown":
                close_reason = "consumer_closed"
            log_method = self.log.debug if close_reason == "completed" else self.log.info
            log_method("[%s] upstream.stream.generator_closing reason=%s elapsed=%.1fms", request_id, close_reason, (time.perf_counter() - started) * 1000)
            raise
        finally:
            self.log.info(
                "[%s] upstream.stream.closed reason=%s status=%s chunks=%s first_chunk=%s elapsed=%.1fms",
                request_id,
                close_reason,
                response_status,
                chunks,
                first_chunk,
                (time.perf_counter() - started) * 1000,
            )

    async def get_models(self, *, check_permission: bool = True) -> list[dict[str, Any]]:
        static = self._static_models()
        if static:
            self.log.info("[models] static models count=%s", len(static))
            self.log.debug_json("[models] static models sample", _model_sample(static))
            return static
        if not self.BASE_URL:
            self.log.warning("[models] provider BASE_URL is empty, returning empty model list")
            return []
        params = {"checkUserPermission": "true" if check_permission else "false"}
        url = self._url(self.MODELS_ENDPOINT)
        self.log.info("[models] upstream request method=GET url=%s", url)
        response = await self.client.get(url, params=params, headers=self.headers())
        self.log.info("[models] upstream response status=%s", response.status_code)
        if response.status_code >= 400:
            self.log.warning("[models] upstream error status=%s body=%s", response.status_code, response.text[:1000])
            response.raise_for_status()
        payload = response.json()
        return _normalize_models(payload, logger=self.log)

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.stream_client.aclose()

    def _url(self, endpoint: str) -> str:
        if not self.BASE_URL:
            raise RuntimeError("Provider BASE_URL is empty. Create/select a concrete provider with BASE_URL set.")
        return urljoin(self.BASE_URL.rstrip("/") + "/", endpoint.lstrip("/"))

    def _static_models(self) -> list[dict[str, Any]]:
        return [{"id": model_id, "object": "model", "created": 0, "owned_by": "upstream"} for model_id in self.STATIC_MODELS]

    def _log_upstream_request(self, request_id: str | None, method: str, url: str, body: dict[str, Any], headers: Mapping[str, str], *, stream: bool = False) -> None:
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        self.log.info(
            "[%s] upstream request method=%s url=%s model=%s stream=%s messages=%s tools=%s header_names=%s",
            request_id,
            method,
            url,
            body.get("model"),
            stream or body.get("stream"),
            len(messages),
            len(tools),
            cfg.safe_header_names(headers),
        )
        self.log.debug_json(f"[{request_id}] upstream headers", cfg.safe_headers(headers))
        summary = {
            "model": body.get("model"),
            "stream": stream or body.get("stream"),
            "messages_count": len(messages),
            "tools_count": len(tools),
            "max_tokens": body.get("max_tokens"),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
        }
        self.log.debug_json(f"[{request_id}] upstream payload_summary", summary)
        if cfg.DEBUG_STREAM:
            self.log.debug_json(f"[{request_id}] upstream payload", body)


def _normalize_models(payload: Any, *, logger=None) -> list[dict[str, Any]]:
    if logger:
        logger.debug_json("[models] _normalize_models input", _models_payload_summary(payload))

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            source_key = "data"
            items = payload["data"]
        elif isinstance(payload.get("models"), list):
            source_key = "models"
            items = payload["models"]
        elif isinstance(payload.get("result"), list):
            source_key = "result"
            items = payload["result"]
        else:
            source_key = "none"
            items = []
    elif isinstance(payload, list):
        source_key = "list"
        items = payload
    else:
        source_key = "none"
        items = []

    result: list[dict[str, Any]] = []
    skipped = 0
    for item in items:
        if isinstance(item, str):
            result.append({"id": item, "object": "model", "created": 0, "owned_by": "upstream"})
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("model") or item.get("modelId") or item.get("name")
            if model_id:
                normalized = dict(item)
                normalized.setdefault("id", str(model_id))
                normalized.setdefault("object", "model")
                normalized.setdefault("created", 0)
                normalized.setdefault("owned_by", "upstream")
                result.append(normalized)
            else:
                skipped += 1
        else:
            skipped += 1

    if logger:
        logger.debug_json(
            "[models] _normalize_models output",
            {
                "source_key": source_key,
                "input_count": len(items),
                "output_count": len(result),
                "skipped_count": skipped,
                "sample": _model_sample(result),
            },
        )
    return result


def _models_payload_summary(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        summary: dict[str, Any] = {
            "type": "dict",
            "keys": sorted(str(key) for key in payload.keys()),
        }
        for key in ("data", "models", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                summary[f"{key}_count"] = len(value)
                summary[f"{key}_sample_type"] = type(value[0]).__name__ if value else None
        return summary
    if isinstance(payload, list):
        return {
            "type": "list",
            "count": len(payload),
            "sample_type": type(payload[0]).__name__ if payload else None,
        }
    return {"type": type(payload).__name__, "value_preview": str(payload)[:200]}


def _model_sample(models: list[dict[str, Any]], limit: int = 10) -> list[str]:
    return [str(item.get("id")) for item in models[:limit] if item.get("id")]
