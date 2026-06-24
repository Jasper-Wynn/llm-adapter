"""Application configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.tools.env import ENV_FILE


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    # =========================
    # Server
    # =========================
    PORT: int = 18002
    SERVER_API_KEY: str = "sk-w3-adapter"
    REQUEST_TIMEOUT: int = 120000
    BODY_LIMIT: str = "20mb"

    # =========================
    # Provider
    # =========================
    PROVIDER_CLASS: str = "app.provider.openai_compatible.OpenAICompatibleProvider"
    PROVIDER_TIMEOUT: int = 300000

    # =========================
    # Logging
    # =========================
    LOG_LEVEL: str = "info"
    LOG_EXTERNAL_LEVEL: str = "warning"
    LOG_CONSOLE: bool = True
    LOG_FILE: bool = False
    LOG_FILE_PATH: str = "logs/server.log"

    # =========================
    # Debug
    # =========================
    DEBUG_STREAM: bool = False
    DROP_THINKING_BLOCK: bool = False
    DEBUG_THROTTLE_TEXT_DELTA_MS: int = 0

    # =========================
    # Uvicorn
    # =========================
    UVICORN_HOST: str = "0.0.0.0"
    UVICORN_RELOAD: bool = False
    UVICORN_ACCESS_LOG: bool = True
    UVICORN_TIMEOUT_KEEP_ALIVE: int = 2
    UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN: int = 5

    @property
    def SERVER_PORT(self) -> int:
        return self.PORT


@lru_cache
def get_config() -> Config:
    return Config()
