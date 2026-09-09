"""Tests for app.core.config.Settings.

Every Settings() construction below passes `_env_file=None` so tests never
depend on (or accidentally read secrets from) a real local .env file on the
developer's machine — only explicit monkeypatched env vars are exercised.
"""

from pydantic import SecretStr, ValidationError

from app.core.config import DEFAULT_GEMINI_MODEL, Settings

ENV_VARS = [
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "DATABASE_PATH",
    "LOG_LEVEL",
    "GMAIL_CLIENT_SECRET_PATH",
    "GMAIL_TOKEN_PATH",
]


def _clear_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_default_gemini_model_constant():
    assert DEFAULT_GEMINI_MODEL == "gemini-3-flash-preview"


def test_settings_defaults(monkeypatch):
    _clear_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.gemini_api_key is None
    assert settings.database_path == "data/app.db"
    assert settings.log_level == "INFO"
    assert settings.gmail_client_secret_path == "credentials/client_secret.json"
    assert settings.gmail_token_path == "credentials/token.json"


def test_settings_environment_overrides(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key-123")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-model")
    monkeypatch.setenv("DATABASE_PATH", "custom/path.db")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("GMAIL_CLIENT_SECRET_PATH", "custom/client_secret.json")
    monkeypatch.setenv("GMAIL_TOKEN_PATH", "custom/token.json")

    settings = Settings(_env_file=None)

    assert settings.gemini_model == "gemini-custom-model"
    assert settings.database_path == "custom/path.db"
    assert settings.log_level == "DEBUG"
    assert settings.gmail_client_secret_path == "custom/client_secret.json"
    assert settings.gmail_token_path == "custom/token.json"
    assert isinstance(settings.gemini_api_key, SecretStr)
    assert settings.gemini_api_key.get_secret_value() == "test-secret-key-123"


def test_settings_gemini_api_key_not_exposed_in_repr_or_dump(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value-do-not-leak")

    settings = Settings(_env_file=None)

    assert "super-secret-value-do-not-leak" not in repr(settings)
    assert "super-secret-value-do-not-leak" not in str(settings)
    assert "super-secret-value-do-not-leak" not in str(settings.model_dump())
    assert "super-secret-value-do-not-leak" not in settings.model_dump_json()


def test_settings_missing_api_key_is_none_not_error(monkeypatch):
    _clear_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key is None


def test_settings_rejects_unrelated_kwargs_without_error_due_to_extra_ignore(
    monkeypatch,
):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SOME_UNRELATED_ENV_VAR", "irrelevant")

    # extra="ignore" means unrelated environment variables must not break
    # construction.
    settings = Settings(_env_file=None)

    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert not hasattr(settings, "some_unrelated_env_var")
