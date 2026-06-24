from __future__ import annotations

from app.config import Config
from app.provider.base import BaseProvider
from app.provider.deepseek import DeepSeekProvider
from app.provider.glm import GLMProvider
from app.provider.loader import load_provider
from app.provider.openai_compatible import OpenAICompatibleProvider


class FakeProvider:
    @classmethod
    def from_config(cls, cfg, *, logger=None):
        return cls()


def test_config_uses_provider_class_without_w3_defaults():
    cfg = Config(_env_file=None)

    assert cfg.PROVIDER_CLASS == "app.provider.openai_compatible.OpenAICompatibleProvider"
    assert cfg.PROVIDER_TIMEOUT == 300000
    assert not hasattr(cfg, "W3_BASE_URL")
    assert not hasattr(cfg, "W3_CHAT_ENDPOINT")
    assert not hasattr(cfg, "W3_MODELS_ENDPOINT")
    assert not hasattr(cfg, "W3_AUTH_TOKEN")
    assert not hasattr(cfg, "PROVIDER_BASE_URL")
    assert not hasattr(cfg, "PROVIDER_CHAT_ENDPOINT")
    assert not hasattr(cfg, "PROVIDER_MODELS_ENDPOINT")
    assert not hasattr(cfg, "PROVIDER_AUTH_TOKEN")
    assert not hasattr(cfg, "PROVIDER_HEADERS_JSON")


def test_load_provider_from_config_class_name():
    class DummyConfig:
        PROVIDER_CLASS = "tests.test_provider_loading.FakeProvider"

    provider = load_provider(DummyConfig())

    assert isinstance(provider, FakeProvider)


class HardcodedProvider(OpenAICompatibleProvider):
    BASE_URL = "https://provider.example.invalid"
    CHAT_ENDPOINT = "/custom/chat"
    MODELS_ENDPOINT = "/custom/models"
    AUTH_HEADER = "X-Test-Token"
    AUTH_SCHEME = ""
    AUTH_TOKEN = "secret-token"
    HEADERS = {"X-Custom": "yes"}
    MODELS_QUERY = {"permission": "true"}
    STREAM_OPTIONS = {"include_usage": False}


class MinimalProvider(OpenAICompatibleProvider):
    BASE_URL = "https://minimal.example.invalid"
    AUTH_TOKEN = "minimal-token"


class RequestHeaderProvider(OpenAICompatibleProvider):
    BASE_URL = "https://request-header.example.invalid"

    def headers(self, *, request_id: str | None = None, stream: bool = False) -> dict[str, str]:
        headers = super().headers(request_id=request_id, stream=stream)
        headers["X-Request-ID"] = request_id or "-"
        headers["X-Stream"] = "yes" if stream else "no"
        return headers


def test_openai_compatible_provider_can_be_configured_by_provider_class():
    class DummyConfig:
        PROVIDER_CLASS = "tests.test_provider_loading.HardcodedProvider"
        PROVIDER_TIMEOUT = 123
        DEBUG_STREAM = True

    provider = load_provider(DummyConfig())

    assert provider.base_url == "https://provider.example.invalid"
    assert provider.chat_endpoint == "/custom/chat"
    assert provider.models_endpoint == "/custom/models"
    assert provider.models_query == {"permission": "true"}
    assert provider.stream_options == {"include_usage": False}
    assert provider.debug_stream is True
    assert provider.headers()["X-Test-Token"] == "secret-token"
    assert provider.headers()["X-Custom"] == "yes"


def test_openai_compatible_provider_inherits_base_defaults():
    class DummyConfig:
        PROVIDER_CLASS = "tests.test_provider_loading.MinimalProvider"
        PROVIDER_TIMEOUT = 123
        DEBUG_STREAM = False

    provider = load_provider(DummyConfig())

    assert isinstance(provider, BaseProvider)
    assert provider.chat_endpoint == "/v1/chat/completions"
    assert provider.models_endpoint == "/v1/models"
    assert provider.stream_options == {"include_usage": True}
    assert provider.headers()["Authorization"] == "Bearer minimal-token"


def test_provider_can_inject_request_headers_per_request():
    provider = RequestHeaderProvider.from_config(type("DummyConfig", (), {"PROVIDER_TIMEOUT": 123, "DEBUG_STREAM": False})())

    assert provider.headers(request_id="rid-1")["X-Request-ID"] == "rid-1"
    assert provider.headers(request_id="rid-1", stream=False)["X-Stream"] == "no"
    assert provider.headers(request_id="rid-1", stream=True)["X-Stream"] == "yes"


def test_deepseek_provider_defaults():
    provider = DeepSeekProvider.from_config(type("DummyConfig", (), {"PROVIDER_TIMEOUT": 123, "DEBUG_STREAM": False})())

    assert provider.base_url == "https://api.deepseek.com"
    assert provider.chat_endpoint == "/chat/completions"
    assert provider.models_endpoint == "/models"
    headers = provider.headers()

    if DeepSeekProvider.AUTH_TOKEN:
        assert headers["Authorization"].startswith("Bearer ")
    else:
        assert "Authorization" not in headers


def test_glm_provider_defaults():
    provider = GLMProvider.from_config(type("DummyConfig", (), {"PROVIDER_TIMEOUT": 123, "DEBUG_STREAM": False})())

    assert provider.base_url == GLMProvider.BASE_URL.rstrip("/")
    assert provider.base_url.startswith("https://open.bigmodel.cn")
    assert provider.chat_endpoint == "/chat/completions"
    assert provider.models_endpoint == "/models"
    headers = provider.headers()

    if GLMProvider.AUTH_TOKEN:
        assert headers["Authorization"].startswith("Bearer ")
    else:
        assert "Authorization" not in headers
