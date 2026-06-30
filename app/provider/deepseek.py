"""DeepSeek provider example."""
from __future__ import annotations

from typing import Mapping

from app.config import env
from app.provider.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    BASE_URL = "https://api.deepseek.com"
    CHAT_ENDPOINT = "/chat/completions"
    MODELS_ENDPOINT = "/models"

    def provider_headers(self, client_headers: Mapping[str, str]) -> dict[str, str]:
        headers = dict(self.HEADERS)
        token = env("DEEPSEEK_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
