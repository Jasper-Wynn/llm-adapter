from app.protocol.anthropic_codec import AnthropicCodec


def test_anthropic_request_to_openai_text():
    payload = AnthropicCodec.request_to_openai({
        "model": "claude-test",
        "max_tokens": 100,
        "system": "be nice",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert payload["model"] == "claude-test"
    assert payload["messages"][0] == {"role": "system", "content": "be nice"}
    assert payload["messages"][1] == {"role": "user", "content": "hello"}


def test_openai_response_to_anthropic_text():
    response = AnthropicCodec.response_from_openai({
        "id": "chatcmpl_1",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }, model="m")
    assert response["content"] == [{"type": "text", "text": "ok"}]
    assert response["stop_reason"] == "end_turn"
    assert response["usage"]["input_tokens"] == 2


def test_anthropic_message_system_role_is_tolerated():
    payload = AnthropicCodec.request_to_openai({
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": "system inside messages"},
            {"role": "user", "content": "hello"},
        ],
    })
    assert payload["messages"][0] == {"role": "system", "content": "system inside messages"}
    assert payload["messages"][1] == {"role": "user", "content": "hello"}


def test_anthropic_message_system_role_blocks_are_tolerated():
    payload = AnthropicCodec.request_to_openai({
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "user", "content": "hello"},
        ],
    })
    assert payload["messages"][0] == {"role": "system", "content": "a\nb"}
