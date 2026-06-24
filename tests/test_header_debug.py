from __future__ import annotations

from app.tools.http import safe_headers


def test_safe_headers_masks_sensitive_values():
    headers = safe_headers({"Authorization": "Bearer secret", "x-api-key": "abc", "X-Normal": "ok"})

    assert headers == {"Authorization": "Bearer ***", "x-api-key": "***", "X-Normal": "ok"}
