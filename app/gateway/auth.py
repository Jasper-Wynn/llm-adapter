"""Gateway authentication."""
from __future__ import annotations

from fastapi import Header, HTTPException

from app.config import get_config


async def auth_dependency(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_auth_token: str | None = Header(default=None),
):
    cfg = get_config()
    expected = (cfg.SERVER_API_KEY or "").strip()
    if not expected:
        return True

    candidates: list[str] = []
    if authorization:
        auth = authorization.strip()
        candidates.append(auth[7:].strip() if auth.lower().startswith("bearer ") else auth)
    if x_api_key:
        candidates.append(x_api_key.strip())
    if x_auth_token:
        candidates.append(x_auth_token.strip())

    if expected in candidates:
        return True

    raise HTTPException(
        status_code=401,
        detail={"error": {"type": "authentication_error", "message": "Invalid or missing API key."}},
    )
