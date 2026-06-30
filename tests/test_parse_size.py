from app.tools.http import parse_size


def test_parse_size():
    assert parse_size("1kb") == 1024
    assert parse_size("2mb") == 2 * 1024 * 1024
