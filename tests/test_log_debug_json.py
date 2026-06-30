import logging

from app.log import AlignedFormatter, configure_logging, install_logger_methods, logger


def test_logger_has_debug_json_method():
    install_logger_methods()
    log = logger("test.debug_json")
    assert hasattr(log, "debug_json")
    assert hasattr(log, "info_json")


def test_debug_json_pretty_prints_json_when_debug_enabled(caplog):
    install_logger_methods()
    log = logger("test.debug_json.pretty")
    log.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        log.debug_json("payload", {"hello": "世界", "items": [1, 2]})
    text = caplog.text
    assert "payload" in text
    assert '"hello": "世界"' in text
    assert '"items": [' in text


def test_debug_json_is_quiet_when_debug_disabled(caplog):
    install_logger_methods()
    log = logger("test.debug_json.info")
    log.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO):
        log.debug_json("payload", {"a": 1})
    assert "payload" not in caplog.text
    assert "{'a': 1}" not in caplog.text


def test_configure_logging_silences_noisy_external_loggers():
    configure_logging(level="debug", external_level="error", console=False, file_enabled=False)
    assert logging.getLogger("httpcore.http2").level == logging.ERROR
    assert logging.getLogger("hpack.hpack").level == logging.ERROR
    assert logging.getLogger("h2.connection").level == logging.ERROR


def test_aligned_formatter_keeps_message_column_stable():
    formatter = AlignedFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    short = logging.LogRecord("llm_adapter.app", logging.INFO, __file__, 1, "short message", (), None)
    long = logging.LogRecord("llm_adapter.gateway.session_cancel", logging.INFO, __file__, 1, "long message", (), None)

    short_line = formatter.format(short)
    long_line = formatter.format(long)

    assert short_line.index("short message") == long_line.index("long message")
    assert "INFO  app" in short_line
    assert "INFO  session" in long_line
