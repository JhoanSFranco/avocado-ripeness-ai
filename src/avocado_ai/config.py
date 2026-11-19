"""Runtime configuration.

Every setting can be overridden with an environment variable of the same name
(case-insensitive) or through a `.env` file in the working directory.
See `.env.example` at the repository root for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Server -----------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    # Comma-separated list of allowed browser origins. Not needed when the
    # frontend is served through the bundled nginx proxy (same origin).
    cors_origins: str = "http://localhost:3000"

    # --- Model ------------------------------------------------------------------
    model_path: Path = Path("models/avocado_ripeness_resnet50.pth")
    # "auto" picks CUDA when available, otherwise CPU.
    device: str = "auto"
    # Number of Monte Carlo Dropout passes per image (>= 2 to estimate variance).
    mc_samples: int = Field(default=8, ge=2, le=200)

    # --- Confidence thresholds (top-1 probability) -------------------------------
    confidence_high: float = Field(default=0.70, gt=0, le=1)
    confidence_medium: float = Field(default=0.50, gt=0, le=1)

    # --- Natural-language insights (optional) ------------------------------------
    # Without a key the API falls back to rule-based local insights.
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_seconds: float = 20.0

    # --- Sessions (multi-image fusion) -------------------------------------------
    session_ttl_seconds: int = Field(default=3600, ge=60)
    max_sessions: int = Field(default=1000, ge=1)
    max_images_per_session: int = Field(default=10, ge=1)

    # --- Uploads & storage -------------------------------------------------------
    max_upload_mb: float = Field(default=10.0, gt=0)
    feedback_path: Path = Path("data/feedback/feedback.csv")

    @model_validator(mode="after")
    def _check_thresholds(self) -> Settings:
        if self.confidence_medium > self.confidence_high:
            raise ValueError("CONFIDENCE_MEDIUM must be <= CONFIDENCE_HIGH")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    return Settings()
