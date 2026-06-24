"""DeepSeek provider."""

from __future__ import annotations

from app.provider.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    BASE_URL = "https://api.deepseek.com"
    CHAT_ENDPOINT = "/chat/completions"
    MODELS_ENDPOINT = "/models"
    AUTH_TOKEN = ""
