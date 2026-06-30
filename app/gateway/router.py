"""HTTP gateway routes."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request

from app import config as cfg
from app.gateway.auth import auth_dependency
from app.log import logger
from app.models.request import RequestContext
from app.tools.http import bool_value, json_error, safe_headers
from app.tools.ids import request_id as new_request_id

router = APIRouter()
header_log = logger("gateway.headers")


@router.get(cfg.HEALTH_ROUTE)
async def health():
    return {"status": "ok"}


@router.get(cfg.OPENAI_MODELS_ROUTE)
async def list_models(request: Request, _=Depends(auth_dependency)):
    models = await request.app.state.provider.get_models(check_permission=True)
    return {"object": "list", "data": models}


@router.get(cfg.COMPAT_MODELS_ROUTE)
@router.get(cfg.COMPAT_MODELS_ROUTE_LEGACY)
async def upstream_models(request: Request, _=Depends(auth_dependency)):
    check_permission = request.query_params.get("checkUserPermission") == "true"
    return await request.app.state.provider.get_models(check_permission=check_permission)


@router.post(cfg.ANTHROPIC_MESSAGES_ROUTE)
async def anthropic_messages(request: Request, _=Depends(auth_dependency)):
    body, error = await _read_body(request)
    if error:
        return error
    ctx = _context(request, body)
    _log_request_headers(request, ctx)
    return await request.app.state.anthropic_adapter.messages(body, ctx)


@router.post(cfg.OPENAI_CHAT_COMPLETIONS_ROUTE)
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
    client_headers = dict(request.headers.items())
    return RequestContext(
        request_id=rid,
        path=request.url.path,
        model=body.get("model"),
        stream=stream,
        client=client,
        client_headers=client_headers,
        session_id=_first_header(client_headers, "x-claude-code-session-id", "x-session-id"),
        parent_session_id=_first_header(client_headers, "x-claude-code-parent-session-id", "x-parent-session-id"),
        agent_id=_header(client_headers, "x-claude-code-agent-id"),
        parent_agent_id=_header(client_headers, "x-claude-code-parent-agent-id"),
    )


def _log_request_headers(request: Request, ctx: RequestContext) -> None:
    header_log.debug("[%s] incoming headers path=%s", ctx.request_id, ctx.path)
    header_log.debug_json(f"[{ctx.request_id}] incoming headers", safe_headers(request.headers))


def _header(headers: dict[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _first_header(headers: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = _header(headers, name)
        if value:
            return value
    return None
