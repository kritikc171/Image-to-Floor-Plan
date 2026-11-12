"""Application core: configuration and shared constants."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import field_validator

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class Settings(BaseSettings):
    """Backend settings loaded from environment / backend/.env.

    Secret values (API keys) are read from the environment only and are never
    included in any API response.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # Groq reasoning provider (optional)
    groq_api_key: str = ""
    groq_model: str = DEFAULT_GROQ_MODEL
    groq_timeout_seconds: int = 45
    groq_max_retries: int = 2

    # Optional novel-view generation (disabled by default)
    novel_view_enabled: bool = False
    novel_view_provider: str = "disabled"
    novel_view_max_iterations: int = 4

    # Image intake limits
    max_upload_bytes: int = 15 * 1024 * 1024  # 15 MiB
    max_image_pixels: int = 4096  # per-dimension cap

    # CORS
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Artifacts
    artifact_ttl_seconds: int = 3600
    artifact_root: str = "artifacts"

    # Concurrency guard
    max_concurrent_jobs: int = 4

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached application settings (safe to call repeatedly)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests)."""
    get_settings.cache_clear()
