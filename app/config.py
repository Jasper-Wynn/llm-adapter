"""Minimal runtime config.

Only two kinds of values live here:
1. Local adapter runtime settings.
2. Default HTTP settings shared by providers.

Upstream details do not belong here.  A concrete provider owns its BASE_URL,
endpoints, headers, model mapping, and any special authentication.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=False)


# ---------------------------------------------------------------------------
# Small env helpers. No schema framework, no duplicated settings class.
# ---------------------------------------------------------------------------
def env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def env_int(name: str, default: int) -> int:
    value = env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Local adapter runtime settings.
# ---------------------------------------------------------------------------
APP_TITLE = "Anthropic/OpenAI Format Adapter"
APP_VERSION = "4.1.0-provider-core"

HOST = env("HOST", env("UVICORN_HOST", "127.0.0.1"))
PORT = env_int("PORT", 18002)
SERVER_PORT = PORT
SERVER_API_KEY = env("SERVER_API_KEY", "sk-llm-adapter")

PROVIDER_CLASS = env("PROVIDER_CLASS", "app.provider.openai_compatible.OpenAICompatibleProvider")

REQUEST_TIMEOUT = env_int("REQUEST_TIMEOUT", 120000)
BODY_LIMIT = env("BODY_LIMIT", "20mb")

LOG_LEVEL = env("LOG_LEVEL", "info")
LOG_EXTERNAL_LEVEL = env("LOG_EXTERNAL_LEVEL", "warning")
LOG_CONSOLE = env_bool("LOG_CONSOLE", True)
LOG_FILE = env_bool("LOG_FILE", False)
LOG_FILE_PATH = env("LOG_FILE_PATH", "logs/server.log")

DEBUG_STREAM = env_bool("DEBUG_STREAM", False)
DROP_THINKING_BLOCK = env_bool("DROP_THINKING_BLOCK", False)
DEBUG_THROTTLE_TEXT_DELTA_MS = env_int("DEBUG_THROTTLE_TEXT_DELTA_MS", 0)

UVICORN_RELOAD = env_bool("UVICORN_RELOAD", False)
UVICORN_ACCESS_LOG = env_bool("UVICORN_ACCESS_LOG", True)
UVICORN_TIMEOUT_KEEP_ALIVE = env_int("UVICORN_TIMEOUT_KEEP_ALIVE", 5)
UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN = env_int("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", 5)


# ---------------------------------------------------------------------------
# Provider HTTP defaults. Providers inherit these class defaults and may
# override them directly in their provider class.
# ---------------------------------------------------------------------------
HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 300.0)
HTTP_CONNECT_TIMEOUT = env_float("HTTP_CONNECT_TIMEOUT", 30.0)
HTTP_VERIFY_SSL = env_bool("HTTP_VERIFY_SSL", True)
HTTP_TRUST_ENV = env_bool("HTTP_TRUST_ENV", False)
HTTP2 = env_bool("HTTP2", True)
HTTP_MAX_CONNECTIONS = env_int("HTTP_MAX_CONNECTIONS", 200)
HTTP_MAX_KEEPALIVE_CONNECTIONS = env_int("HTTP_MAX_KEEPALIVE_CONNECTIONS", 40)
HTTP_KEEPALIVE_EXPIRY = env_float("HTTP_KEEPALIVE_EXPIRY", 30.0)


# ---------------------------------------------------------------------------
# Shared route / HTTP constants. These are adapter constants, not upstream
# config. Provider-specific routes still belong to provider classes.
# ---------------------------------------------------------------------------
ROOT_ROUTE = "/"
HEALTH_ROUTE = "/health"
OPENAI_MODELS_ROUTE = "/v1/models"
OPENAI_CHAT_COMPLETIONS_ROUTE = "/v1/chat/completions"
ANTHROPIC_MESSAGES_ROUTE = "/v1/messages"
COMPAT_MODELS_ROUTE = "/chat/models"
COMPAT_MODELS_ROUTE_LEGACY = "/chat/modles"
UVICORN_APP = "app.main:app"

HEADER_REQUEST_ID = "X-Request-ID"
HEADER_CONTENT_TYPE = "Content-Type"
HEADER_CONTENT_LENGTH = "content-length"
HEADER_AUTHORIZATION = "authorization"
HEADER_X_API_KEY = "x-api-key"
HEADER_X_AUTH_TOKEN = "x-auth-token"
HEADER_USER_AGENT = "user-agent"
HEADER_ANTHROPIC_VERSION = "anthropic-version"
HEADER_ANTHROPIC_BETA = "anthropic-beta"

MEDIA_TYPE_JSON = "application/json"
MEDIA_TYPE_JSON_UTF8 = "application/json; charset=utf-8"
MEDIA_TYPE_SSE = "text/event-stream"

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

SENSITIVE_HEADERS = {
    HEADER_AUTHORIZATION,
    HEADER_X_API_KEY,
    HEADER_X_AUTH_TOKEN,
    "api-key",
    "cookie",
    "set-cookie",
}

CLIENT_HEADER_BLACKLIST = (
    "connection",
    "accept",
    "content-type",
    HEADER_CONTENT_LENGTH,
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
)

CORS_ALLOW_ORIGINS = ["*"]
CORS_ALLOW_METHODS = ["GET", "POST", "OPTIONS", "HEAD"]
CORS_ALLOW_HEADERS = [
    "Accept",
    HEADER_CONTENT_TYPE,
    "Authorization",
    "X-Auth-Token",
    "x-api-key",
    "x-request-id",
    HEADER_ANTHROPIC_VERSION,
    HEADER_ANTHROPIC_BETA,
]
CORS_EXPOSE_HEADERS = [
    HEADER_CONTENT_TYPE,
    "Cache-Control",
    "X-Accel-Buffering",
    HEADER_REQUEST_ID,
]


def safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in headers.items():
        result[key] = "***" if key.lower() in SENSITIVE_HEADERS else str(value)
    return result


def safe_header_names(headers: Mapping[str, str]) -> list[str]:
    return sorted(str(key).lower() for key in headers.keys())


def as_dict() -> dict[str, object]:
    return {
        "HOST": HOST,
        "PORT": PORT,
        "SERVER_API_KEY": "***" if SERVER_API_KEY else "",
        "PROVIDER_CLASS": PROVIDER_CLASS,
        "REQUEST_TIMEOUT": REQUEST_TIMEOUT,
        "BODY_LIMIT": BODY_LIMIT,
        "LOG_LEVEL": LOG_LEVEL,
        "DEBUG_STREAM": DEBUG_STREAM,
        "DROP_THINKING_BLOCK": DROP_THINKING_BLOCK,
        "HTTP_TIMEOUT": HTTP_TIMEOUT,
        "HTTP_CONNECT_TIMEOUT": HTTP_CONNECT_TIMEOUT,
        "HTTP_VERIFY_SSL": HTTP_VERIFY_SSL,
        "HTTP_TRUST_ENV": HTTP_TRUST_ENV,
        "HTTP2": HTTP2,
    }


def get_config():
    """Compatibility helper for older call sites that expect cfg.X."""
    return sys.modules[__name__]
