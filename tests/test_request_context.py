from app.gateway.router import _context


class FakeURL:
    path = "/v1/chat/completions"


class FakeRequest:
    def __init__(self, headers):
        self.headers = headers
        self.url = FakeURL()


def test_context_reads_opencode_session_headers():
    ctx = _context(
        FakeRequest(
            {
                "x-request-id": "req_test",
                "x-session-id": "session_child",
                "x-parent-session-id": "session_parent",
                "user-agent": "opencode/test",
            }
        ),
        {"model": "glm-4.7", "stream": True},
    )

    assert ctx.request_id == "req_test"
    assert ctx.session_id == "session_child"
    assert ctx.parent_session_id == "session_parent"
    assert ctx.client == "opencode/test"
