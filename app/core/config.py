"""Application settings, loaded from environment variables / a local .env
file via pydantic-settings. Never hard-code secrets here — see
.env.example for the documented variable names, and pass `_env_file=None`
when constructing `Settings` in tests to avoid depending on a developer's
real local .env file.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


class Settings(BaseSettings):
    """Typed application configuration.

    Field names are matched case-insensitively against environment
    variables of the same name (e.g. `gemini_api_key` <- `GEMINI_API_KEY`),
    so no explicit aliases are needed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini
    gemini_api_key: SecretStr | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL

    # Storage
    database_path: str = "data/app.db"

    # Logging
    log_level: str = "INFO"

    # Gmail OAuth (paths only — the files themselves are never committed)
    gmail_client_secret_path: str = "credentials/client_secret.json"
    gmail_token_path: str = "credentials/token.json"
