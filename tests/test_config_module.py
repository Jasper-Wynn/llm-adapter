import types

import app.config as config


def test_get_config_returns_config_module_not_settings_class():
    cfg = config.get_config()
    assert isinstance(cfg, types.ModuleType)
    assert cfg.PORT == config.PORT
    assert cfg.SERVER_PORT == cfg.PORT


def test_config_is_minimal_and_has_no_upstream_globals():
    assert hasattr(config, "PROVIDER_CLASS")
    assert hasattr(config, "HTTP_TIMEOUT")
    assert not hasattr(config, "UPSTREAM_BASE_URL")
    assert not hasattr(config, "UPSTREAM_CHAT_ENDPOINT")
    assert not hasattr(config, "UPSTREAM_AUTH_HEADER")
