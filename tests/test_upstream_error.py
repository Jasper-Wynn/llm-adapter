from __future__ import annotations

from app.protocol.upstream_error import UpstreamErrorCodec


def test_quota_error_normalized():
    payload = {
        "text": "[DONE]",
        "error": {
            "error_msg": '{"type":"RPM", "message":"There is no request model quota", "identity":"userId", "quota":0, "used":0}',
            "error_code": "InferHub.002002010.429",
        },
        "error_code": "InferHub.002002010.429",
        "error_msg": '{"type":"RPM", "message":"There is no request model quota", "identity":"userId", "quota":0, "used":0}',
    }

    error = UpstreamErrorCodec.from_payload(payload)

    assert error is not None
    assert error.status_code == 429
    assert error.type == "rate_limit_error"
    assert error.code == "InferHub.002002010.429"
    assert error.message == "There is no request model quota"


def test_valid_openai_chunk_is_not_error():
    payload = {"choices": [{"delta": {"content": "hello"}}]}
    assert UpstreamErrorCodec.from_payload(payload) is None
