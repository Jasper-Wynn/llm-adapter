"""HTTP gateway routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.gateway.auth import auth_dependency
from app.log import logger
from app.models.request import RequestContext
from app.tools.http import bool_value, json_error, safe_headers
from app.tools.ids import request_id as new_request_id


router = APIRouter()
header_log = logger("gateway.headers")


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/v1/models")
async def list_models(request: Request):
    models = await request.app.state.provider.get_models(check_permission=True)
    return {"object": "list", "data": models}


@router.get("/chat/modles")
async def upstream_models(request: Request):
    check_permission = request.query_params.get("checkUserPermission") == "true"
    return await request.app.state.provider.get_models(check_permission=check_permission)


@router.post("/v1/messages")
async def anthropic_messages(request: Request, _=Depends(auth_dependency)):
    body, error = await _read_body(request)

    if error:
        return error

    ctx = _context(request, body)
    _log_request_headers(request, ctx)
    return await request.app.state.anthropic_adapter.messages(body, ctx)


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: Request, _=Depends(auth_dependency)):
    body, error = await _read_body(request)

    if error:
        return error

    ctx = _context(request, body)
    _log_request_headers(request, ctx)
    return await request.app.state.openai_adapter.chat_completions(body, ctx)


async def _read_body(request: Request) -> tuple[dict[str, Any] | None, Any | None]:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        return None, json_error(400, f"Invalid JSON body: {exc}")
    except Exception as exc:
        return None, json_error(400, f"Invalid request body: {exc}")

    if not isinstance(body, dict):
        return None, json_error(400, "JSON body must be an object.")

    return body, None


def _context(request: Request, body: dict[str, Any]) -> RequestContext:
    rid = request.headers.get("x-request-id") or new_request_id()
    stream = bool_value(body.get("stream"))
    client = request.headers.get("user-agent")

    return RequestContext(request_id=rid, path=request.url.path, model=body.get("model"), stream=stream, client=client)


def _log_request_headers(request: Request, ctx: RequestContext) -> None:
    header_log.debug("[%s] incoming headers path=%s headers=%s", ctx.request_id, ctx.path, safe_headers(request.headers))
