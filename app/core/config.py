"""Application settings, loaded from environment variables / a local .env
file via pydantic-settings. Never hard-code secrets here — see
.env.example for the documented variable names, and pass `_env_file=None`
when constructing `Settings` in tests to avoid depending on a developer's
real local .env file.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_AI_PROVIDER = "anthropic"


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

    # Which classifier backend app.ai.classifier.get_classifier() builds —
    # "gemini" or "anthropic". Both implementations always exist in the
    # codebase; this only selects which one is active. See
    # app/ai/classifier.py for the routing logic.
    ai_provider: str = DEFAULT_AI_PROVIDER

    # Gemini
    gemini_api_key: SecretStr | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL

    # Anthropic (Claude)
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL

    # Storage
    database_path: str = "data/app.db"

    # Logging
    log_level: str = "INFO"

    # Gmail OAuth (paths only — the files themselves are never committed)
    gmail_client_secret_path: str = "credentials/client_secret.json"
    gmail_token_path: str = "credentials/token.json"
    gmail_fetch_max_results: int = 10
