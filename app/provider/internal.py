"""Internal OpenAI-compatible provider example.

This file shows the intended customization style:
- Upstream location lives in the provider.
- Special header/token logic lives in the provider.
- The global config never knows about this upstream.
"""
from __future__ import annotations

from typing import Mapping

from app.config import env
from app.provider.openai_compatible import OpenAICompatibleProvider


class InternalOpenAIProvider(OpenAICompatibleProvider):
    BASE_URL = "http://127.0.0.1:8000"
    CHAT_ENDPOINT = "/v1/chat/completions"
    MODELS_ENDPOINT = "/v1/models"

    # Keep transport blacklist; append provider-specific blocked headers only if needed.
    CLIENT_HEADER_BLACKLIST = OpenAICompatibleProvider.CLIENT_HEADER_BLACKLIST + (
        # "cookie",
        # "authorization",
    )

    HEADERS = {
        # "X-Provider": "internal",
    }

    def provider_headers(self, client_headers: Mapping[str, str]) -> dict[str, str]:
        headers = dict(self.HEADERS)
        token = env("INTERNAL_AUTH_TOKEN")
        if token:
            headers["X-Auth-Token"] = token
        return headers
