from app.protocol.tool_calling import ToolCallStream


def test_tool_call_stream_accumulates_arguments():
    s = ToolCallStream()
    s.apply([{"index": 0, "id": "call_1", "function": {"name": "search", "arguments": "{\"q\""}}], 0)
    s.apply([{"index": 0, "function": {"arguments": ":\"hi\"}"}}], 0)
    state = s.ordered()[0]
    assert state.name == "search"
    assert state.arguments == '{"q":"hi"}'
    assert s.meta[0].seen_name is True
