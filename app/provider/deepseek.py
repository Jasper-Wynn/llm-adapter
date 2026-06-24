"""DeepSeek provider."""

from __future__ import annotations

from app.provider.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    BASE_URL = "https://api.deepseek.com"
    CHAT_ENDPOINT = "/chat/completions"
    MODELS_ENDPOINT = "/models"
    AUTH_TOKEN = "sk-701e54a6954849999ee3799c29fe972a"
