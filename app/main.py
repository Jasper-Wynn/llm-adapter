"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.anthropic import AnthropicAdapter
from app.adapters.openai import OpenAIAdapter
from app.config import get_config
from app.gateway.middleware import BodyLimitMiddleware
from app.gateway.router import router
from app.log import configure_logging, logger
from app.provider.loader import load_provider
from app.tools.http import parse_size


APP_TITLE = "Anthropic/OpenAI Format Adapter"
APP_VERSION = "3.4.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()

    configure_logging(level=cfg.LOG_LEVEL, external_level=cfg.LOG_EXTERNAL_LEVEL, console=cfg.LOG_CONSOLE, file_enabled=cfg.LOG_FILE, file_path=cfg.LOG_FILE_PATH)

    log = logger("app")

    provider = load_provider(cfg, logger=logger("provider"))

    app.state.provider = provider
    app.state.openai_adapter = OpenAIAdapter(provider=provider, logger=logger("adapter.openai"))
    app.state.anthropic_adapter = AnthropicAdapter(provider=provider, drop_thinking_block=cfg.DROP_THINKING_BLOCK, throttle_text_delta_ms=cfg.DEBUG_THROTTLE_TEXT_DELTA_MS, logger=logger("adapter.anthropic"))
    app.state.is_shutting_down = False

    _print_startup_banner(cfg)
    print(f"  Provider: {cfg.PROVIDER_CLASS}")
    print("=" * 60)

    try:
        _print_models(await provider.get_models(check_permission=True))
    except Exception as exc:
        log.warning("Models fetch failed: %s", exc)

    try:
        yield

    finally:
        app.state.is_shutting_down = True

        try:
            await asyncio.wait_for(provider.aclose(), timeout=3.0)
            log.info("Provider closed.")
        except asyncio.TimeoutError:
            log.warning("Provider close timeout.")
        except Exception as exc:
            log.warning("Provider close failed: %s", exc)

        log.info("Server shutdown complete.")


def create_app() -> FastAPI:
    cfg = get_config()

    app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)

    _install_middlewares(app, cfg)
    _install_root_routes(app)
    app.include_router(router)

    return app


def _install_middlewares(app: FastAPI, cfg) -> None:
    app.add_middleware(BodyLimitMiddleware, max_bytes=parse_size(cfg.BODY_LIMIT))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
        allow_headers=[
            "Accept",
            "Content-Type",
            "Authorization",
            "X-Auth-Token",
            "x-api-key",
            "x-request-id",
            "anthropic-version",
            "anthropic-beta",
        ],
        expose_headers=[
            "Content-Type",
            "Cache-Control",
            "X-Accel-Buffering",
            "X-Request-ID",
        ],
    )


def _install_root_routes(app: FastAPI) -> None:
    @app.get("/")
    async def root():
        return {
            "service": APP_TITLE,
            "version": APP_VERSION,
            "status": "running",
            "endpoints": {
                "anthropic": "POST /v1/messages",
                "openai": "POST /v1/chat/completions",
                "models": "GET /v1/models",
                "health": "GET /health",
            },
        }

    @app.head("/")
    async def root_head():
        return Response(status_code=200)


def _print_startup_banner(cfg) -> None:
    print("=" * 60)
    print(f"  {APP_TITLE} Server")
    print("=" * 60)
    print(f"  Server: http://{cfg.UVICORN_HOST}:{cfg.PORT}")
    print(f"  Body Limit: {cfg.BODY_LIMIT}")
    print(f"  Debug Stream: {cfg.DEBUG_STREAM}")
    print(f"  Drop Thinking Block: {cfg.DROP_THINKING_BLOCK}")
    print(f"  Text Delta Throttle: {cfg.DEBUG_THROTTLE_TEXT_DELTA_MS}ms")
    print("=" * 60)


def _print_models(models: list[dict[str, Any]]) -> None:
    print(f"\n  Models ({len(models)})")
    print("  " + "-" * 60)

    for model in models[:20]:
        model_id = str(model.get("modelId") or model.get("id") or "")[:18].ljust(18)
        name = str(model.get("name") or "")[:19].ljust(19)
        context = str(model.get("context") or 0).rjust(8)
        input_tokens = str(model.get("input") or 0).rjust(8)
        output_tokens = str(model.get("output") or 0).rjust(8)
        print(f"  {model_id} {name} {context} {input_tokens} {output_tokens}")

    print("  " + "-" * 60)


app = create_app()


def main() -> None:
    import uvicorn

    cfg = get_config()

    uvicorn.run("app.main:app", host=cfg.UVICORN_HOST, port=cfg.PORT, log_level=cfg.LOG_LEVEL.lower(), access_log=cfg.UVICORN_ACCESS_LOG, timeout_keep_alive=cfg.UVICORN_TIMEOUT_KEEP_ALIVE, timeout_graceful_shutdown=cfg.UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN, reload=cfg.UVICORN_RELOAD)


if __name__ == "__main__":
    main()
