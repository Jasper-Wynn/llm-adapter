"""GLM provider."""

from __future__ import annotations

from app.provider.openai_compatible import OpenAICompatibleProvider


class GLMProvider(OpenAICompatibleProvider):
    BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
    CHAT_ENDPOINT = "/chat/completions"
    MODELS_ENDPOINT = "/models"
    AUTH_TOKEN = "ad0129c21bf3416eb1928a28efae0dd3.Lt2gHJ1nGHpnbNR6"
