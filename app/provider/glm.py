"""GLM provider example."""
from __future__ import annotations

from typing import Mapping

from app.config import env
from app.provider.openai_compatible import OpenAICompatibleProvider


class GLMProvider(OpenAICompatibleProvider):
    BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
    CHAT_ENDPOINT = "/chat/completions"
    MODELS_ENDPOINT = "/models"

    def provider_headers(self, client_headers: Mapping[str, str]) -> dict[str, str]:
        headers = dict(self.HEADERS)
        token = env("GLM_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
