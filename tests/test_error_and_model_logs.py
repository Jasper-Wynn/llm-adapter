import httpx

from app.adapters.anthropic import _anthropic_http_status_response
from app.adapters.openai import _openai_http_status_response
from app.provider.openai_compatible import _normalize_models


def _http_status_error(status_code: int, payload):
    request = httpx.Request("POST", "https://upstream.example/chat/completions")
    response = httpx.Response(status_code, json=payload, request=request)
    return httpx.HTTPStatusError("bad response", request=request, response=response)


def test_openai_http_status_error_preserves_upstream_error_payload():
    exc = _http_status_error(429, {"error": {"type": "rate_limit_error", "message": "slow down", "code": "429"}})
    response = _openai_http_status_response(exc, "req_test")
    assert response.status_code == 429
    assert response.headers["X-Request-ID"] == "req_test"
    assert b"rate_limit_error" in response.body
    assert b"slow down" in response.body


def test_anthropic_http_status_error_preserves_upstream_error_payload():
    exc = _http_status_error(401, {"error": {"type": "authentication_error", "message": "bad token", "code": "401"}})
    response = _anthropic_http_status_response(exc, "req_test")
    assert response.status_code == 401
    assert response.headers["X-Request-ID"] == "req_test"
    assert b"authentication_error" in response.body
    assert b"bad token" in response.body


def test_normalize_models_logs_input_and_output():
    seen = []

    class DummyLogger:
        def debug_json(self, key, data):
            seen.append((key, data))

    result = _normalize_models({"models": ["m1", {"name": "m2"}, {"no_id": True}]}, logger=DummyLogger())
    assert [item["id"] for item in result] == ["m1", "m2"]
    keys = [item[0] for item in seen]
    assert "[models] _normalize_models input" in keys
    assert "[models] _normalize_models output" in keys
