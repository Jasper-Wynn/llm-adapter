from app.provider.internal import InternalOpenAIProvider
from app.provider.openai_compatible import OpenAICompatibleProvider
from app.provider.base import BaseProvider


def test_provider_subclass_location_is_not_overwritten_by_config():
    class DemoProvider(OpenAICompatibleProvider):
        BASE_URL = "https://demo.example/api"
        CHAT_ENDPOINT = "/chat/completions"
        MODELS_ENDPOINT = "/models"
        VERIFY_SSL = False
        TRUST_ENV = True
        HTTP2 = False

    provider = DemoProvider()
    assert provider.BASE_URL == "https://demo.example/api"
    assert provider.CHAT_ENDPOINT == "/chat/completions"
    assert provider.MODELS_ENDPOINT == "/models"
    assert provider.VERIFY_SSL is False
    assert provider.TRUST_ENV is True
    assert provider.HTTP2 is False


def test_base_provider_has_no_auth_abstraction():
    provider = OpenAICompatibleProvider()
    assert hasattr(provider, "HEADERS")
    assert hasattr(provider, "CLIENT_HEADER_BLACKLIST")
    assert not hasattr(provider, "AUTH_HEADER")
    assert not hasattr(provider, "AUTH_SCHEME")
    assert not hasattr(provider, "AUTH_TOKEN_ENV")
    assert not hasattr(provider, "AUTH_TOKEN_CONFIG_FIELD")


def test_provider_forwards_client_headers_by_default_and_blocks_transport_headers():
    provider = OpenAICompatibleProvider()
    headers = provider.headers(
        request_id="server-rid",
        client_headers={
            "x-tenant-id": "tenant-a",
            "authorization": "Bearer client-token",
            "cookie": "sid=abc",
            "host": "client.example",
            "content-length": "999",
            "connection": "keep-alive",
        },
    )
    lower = {key.lower(): value for key, value in headers.items()}
    assert lower["x-tenant-id"] == "tenant-a"
    assert lower["authorization"] == "Bearer client-token"
    assert lower["cookie"] == "sid=abc"
    assert "host" not in lower
    assert "content-length" not in lower
    assert "connection" not in lower
    assert headers["X-Request-ID"] == "server-rid"


def test_static_provider_headers_override_passthrough_case_insensitively():
    class DemoProvider(OpenAICompatibleProvider):
        HEADERS = {"X-Demo": "provider", "X-Tenant-ID": "provider-tenant"}

    provider = DemoProvider()
    headers = provider.headers(
        client_headers={"x-tenant-id": "client-tenant", "x-demo": "client-demo"},
    )
    assert headers["X-Demo"] == "provider"
    assert headers["X-Tenant-ID"] == "provider-tenant"
    assert "x-tenant-id" not in headers


def test_dynamic_provider_headers_can_use_client_headers():
    class DemoProvider(OpenAICompatibleProvider):
        def provider_headers(self, client_headers):
            headers = super().provider_headers(client_headers)
            tenant = self.client_header(client_headers, "x-tenant-id")
            if tenant:
                headers["X-Upstream-Tenant"] = tenant
            return headers

    provider = DemoProvider()
    headers = provider.headers(client_headers={"x-tenant-id": "tenant-a"})
    assert headers["x-tenant-id"] == "tenant-a"
    assert headers["X-Upstream-Tenant"] == "tenant-a"


def test_internal_provider_adds_x_auth_token_directly(monkeypatch):
    monkeypatch.setenv("INTERNAL_AUTH_TOKEN", "token456")
    provider = InternalOpenAIProvider()
    headers = provider.headers(request_id="r2")
    assert headers["X-Auth-Token"] == "token456"
    assert "Authorization" not in headers
    assert headers["X-Request-ID"] == "r2"


def test_base_provider_defaults_come_from_config_by_class_inheritance():
    assert BaseProvider.TIMEOUT == OpenAICompatibleProvider.TIMEOUT
    assert BaseProvider.CLIENT_HEADER_BLACKLIST == OpenAICompatibleProvider.CLIENT_HEADER_BLACKLIST


def test_provider_controls_accept_and_content_type_case_insensitively():
    provider = OpenAICompatibleProvider()
    stream_headers = provider.headers(
        stream=True,
        client_headers={
            "accept": "application/json",
            "content-type": "text/plain",
            "x-client": "yes",
        },
    )
    lower = {key.lower(): value for key, value in stream_headers.items()}
    assert lower["accept"] == "text/event-stream"
    assert lower["content-type"] == "application/json"
    assert lower["x-client"] == "yes"
    assert list(key.lower() for key in stream_headers).count("content-type") == 1
    assert list(key.lower() for key in stream_headers).count("accept") == 1

    json_headers = provider.headers(
        stream=False,
        client_headers={"Accept": "text/event-stream", "Content-Type": "text/plain"},
    )
    lower = {key.lower(): value for key, value in json_headers.items()}
    assert lower["accept"] == "application/json"
    assert lower["content-type"] == "application/json"
