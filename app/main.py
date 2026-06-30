"""FastAPI application entry point."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app import config as cfg
from app.adapters.anthropic import AnthropicAdapter
from app.adapters.openai import OpenAIAdapter
from app.gateway.middleware import BodyLimitMiddleware
from app.gateway.router import router
from app.gateway.session_cancel import StreamCancellationRegistry
from app.log import configure_logging, logger
from app.provider.loader import load_provider
from app.tools.http import parse_size


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(
        level=cfg.LOG_LEVEL,
        external_level=cfg.LOG_EXTERNAL_LEVEL,
        console=cfg.LOG_CONSOLE,
        file_enabled=cfg.LOG_FILE,
        file_path=cfg.LOG_FILE_PATH,
    )
    log = logger("app")
    provider = load_provider(logger=logger("provider"))
    stream_registry = StreamCancellationRegistry(logger=logger("gateway.session_cancel"))
    app.state.provider = provider
    app.state.stream_registry = stream_registry
    app.state.openai_adapter = OpenAIAdapter(provider=provider, stream_registry=stream_registry, logger=logger("adapter.openai"))
    app.state.anthropic_adapter = AnthropicAdapter(
        provider=provider,
        drop_thinking_block=cfg.DROP_THINKING_BLOCK,
        throttle_text_delta_ms=cfg.DEBUG_THROTTLE_TEXT_DELTA_MS,
        stream_registry=stream_registry,
        logger=logger("adapter.anthropic"),
    )
    app.state.is_shutting_down = False

    _log_startup(log, provider)
    try:
        models = await provider.get_models(check_permission=True)
        _log_models(log, models)
    except Exception as exc:
        log.warning("[startup] models fetch failed: %s", exc)

    try:
        yield
    finally:
        app.state.is_shutting_down = True
        try:
            await asyncio.wait_for(provider.aclose(), timeout=3.0)
            log.info("[shutdown] provider closed")
        except asyncio.TimeoutError:
            log.warning("[shutdown] provider close timeout")
        except Exception as exc:
            log.warning("[shutdown] provider close failed: %s", exc)
        log.info("[shutdown] server shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(title=cfg.APP_TITLE, version=cfg.APP_VERSION, lifespan=lifespan)
    _install_middlewares(app)
    _install_root_routes(app)
    app.include_router(router)
    return app


def _install_middlewares(app: FastAPI) -> None:
    app.add_middleware(BodyLimitMiddleware, max_bytes=parse_size(cfg.BODY_LIMIT))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.CORS_ALLOW_ORIGINS,
        allow_methods=cfg.CORS_ALLOW_METHODS,
        allow_headers=cfg.CORS_ALLOW_HEADERS,
        expose_headers=cfg.CORS_EXPOSE_HEADERS,
    )


def _install_root_routes(app: FastAPI) -> None:
    @app.get(cfg.ROOT_ROUTE)
    async def root():
        return {
            "service": cfg.APP_TITLE,
            "version": cfg.APP_VERSION,
            "status": "running",
            "endpoints": {
                "anthropic": f"POST {cfg.ANTHROPIC_MESSAGES_ROUTE}",
                "openai": f"POST {cfg.OPENAI_CHAT_COMPLETIONS_ROUTE}",
                "models": f"GET {cfg.OPENAI_MODELS_ROUTE}",
                "upstream_models": f"GET {cfg.COMPAT_MODELS_ROUTE}",
                "health": f"GET {cfg.HEALTH_ROUTE}",
            },
        }

    @app.head(cfg.ROOT_ROUTE)
    async def root_head():
        return Response(status_code=200)


def _log_startup(log, provider) -> None:
    description = provider.describe()
    log.info("[startup] app=%s version=%s", cfg.APP_TITLE, cfg.APP_VERSION)
    log.info("[startup] server=http://%s:%s body_limit=%s", cfg.HOST, cfg.PORT, cfg.BODY_LIMIT)
    log.info(
        "[startup] provider=%s base_url=%s chat=%s models=%s verify_ssl=%s trust_env=%s http2=%s headers=%s",
        description.get("provider"),
        description.get("base_url"),
        description.get("chat_endpoint"),
        description.get("models_endpoint"),
        description.get("verify_ssl"),
        description.get("trust_env"),
        description.get("http2"),
        description.get("provider_header_names"),
    )
    log.info("[startup] debug_stream=%s drop_thinking_block=%s throttle_ms=%s", cfg.DEBUG_STREAM, cfg.DROP_THINKING_BLOCK, cfg.DEBUG_THROTTLE_TEXT_DELTA_MS)
    log.debug_json("[startup] provider", description)
    log.debug_json("[startup] runtime_config", cfg.as_dict())


def _log_models(log, models: list[dict[str, Any]]) -> None:
    model_ids = [str(model.get("id") or model.get("modelId") or model.get("name") or "") for model in models[:20]]
    log.info("[startup] models count=%s sample=%s", len(models), model_ids)


app = create_app()


def main() -> None:
    uvicorn.run(
        cfg.UVICORN_APP,
        host=cfg.HOST,
        port=cfg.PORT,
        log_level=cfg.LOG_LEVEL.lower(),
        access_log=cfg.UVICORN_ACCESS_LOG,
        timeout_keep_alive=cfg.UVICORN_TIMEOUT_KEEP_ALIVE,
        timeout_graceful_shutdown=cfg.UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN,
        reload=cfg.UVICORN_RELOAD,
    )


if __name__ == "__main__":
    main()
