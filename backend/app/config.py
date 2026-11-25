from functools import lru_cache
from pathlib import Path
import re

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


from .providers.openrouter_routing import (
    FAST_MODEL, GENERATION_MODEL, RECOMMENDATION_MODEL, RERANK_MODEL, VERIFICATION_MODEL, VISION_MODEL,
    OutputLimits,
)


class Settings(BaseSettings):
    answer_quality_pass_threshold: float = Field(default=1.0, gt=0, le=1)
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_client_id: str = ""
    mongodb_uri: str = ""
    mongodb_database: str = "citewise"
    jwt_secret: str = Field(default="", min_length=32, validate_default=False)
    citewise_demo_mode: bool = False
    citewise_demo_mongodb_uri: str = "mongodb://127.0.0.1:27017"
    citewise_demo_database: str = "citewise_demo_mentor"
    citewise_demo_jwt_secret: SecretStr | None = None
    allowed_origins: str = "http://localhost:5173"
    session_cookie_secure: bool = False
    session_cookie_name: str = "citewise_session"
    session_duration_minutes: int = Field(default=60, ge=5, le=1440)
    storage_directory: Path = Path(__file__).resolve().parents[1] / "storage"
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: SecretStr | None = None
    max_pdf_size_mb: int = Field(default=25, ge=1, le=25)
    max_pdf_pages: int = Field(default=300, ge=1, le=1000)
    max_pdf_characters: int = Field(default=2_000_000, ge=1000)
    embedding_batch_size: int = Field(default=32, ge=1, le=64)
    openrouter_api_key: SecretStr | None = None
    openrouter_generation_model: str = GENERATION_MODEL
    openrouter_external_generation_model: str = ""
    openrouter_verification_model: str = VERIFICATION_MODEL
    openrouter_structured_fallback_model: str = FAST_MODEL
    openrouter_recommendation_model: str = RECOMMENDATION_MODEL
    openrouter_generation_fallback: str | None = None
    openrouter_rerank_model: str = RERANK_MODEL
    openrouter_rerank_enabled: bool = False
    openrouter_vision_model: str = VISION_MODEL
    openrouter_vision_enabled: bool = False
    openrouter_require_zdr: bool = False
    openrouter_simple_max_tokens: int = Field(default=OutputLimits.simple, ge=600, le=1200)
    openrouter_research_max_tokens: int = Field(default=OutputLimits.research, ge=1500, le=8192)
    openrouter_external_max_tokens: int = Field(default=OutputLimits.external, ge=1500, le=8192)
    openrouter_assessment_max_tokens: int = Field(default=OutputLimits.assessment, ge=2048, le=8192)
    openrouter_verification_max_tokens: int = Field(default=OutputLimits.verification, ge=4096, le=8192)
    openrouter_recommendation_max_tokens: int = Field(default=OutputLimits.recommendations, ge=768, le=3072)
    openrouter_daily_limit: int = Field(default=50, ge=1, le=50)
    openrouter_minute_limit: int = Field(default=20, ge=1, le=20)
    answer_timeout_seconds: int = Field(default=120, ge=10, le=300)
    tavily_api_key: SecretStr | None = None
    openalex_api_key: SecretStr | None = None
    ncbi_api_key: SecretStr | None = None
    crossref_contact_email: str = ""

    @model_validator(mode="after")
    def validate_pdf_storage(self) -> "Settings":
        if self.citewise_demo_mode:
            if (not re.fullmatch(r"citewise_demo_[a-z0-9_-]{1,40}", self.citewise_demo_database)
                    or self.citewise_demo_database == self.mongodb_database):
                raise ValueError("Demo mode requires a separate citewise_demo_* database.")
            secret = self.citewise_demo_jwt_secret.get_secret_value() if self.citewise_demo_jwt_secret else ""
            if len(secret) < 32 or secret == self.jwt_secret:
                raise ValueError("Demo mode requires a separate session secret of at least 32 characters.")
            self.session_cookie_name = "citewise_demo_session"
            return self
        if not (self.mongodb_uri and self.jwt_secret and self.google_client_id):
            raise ValueError("The MongoDB URI, Google client ID, and session secret are required.")
        if not (
            self.cloudinary_cloud_name and self.cloudinary_api_key
            and self.cloudinary_api_secret and self.cloudinary_api_secret.get_secret_value()
        ):
            raise ValueError("Cloudinary storage requires a cloud name, API key, and API secret.")
        return self

    @property
    def session_signing_secret(self) -> str:
        if self.citewise_demo_mode and self.citewise_demo_jwt_secret is not None:
            return self.citewise_demo_jwt_secret.get_secret_value()
        return self.jwt_secret

    @property
    def data_directory(self) -> Path:
        if self.citewise_demo_mode:
            return self.storage_directory / "citewise-demo" / self.citewise_demo_database
        return self.storage_directory

    @property
    def openrouter_output_limits(self) -> OutputLimits:
        return OutputLimits(self.openrouter_simple_max_tokens, self.openrouter_research_max_tokens,
                            self.openrouter_external_max_tokens, self.openrouter_assessment_max_tokens,
                            self.openrouter_verification_max_tokens, self.openrouter_recommendation_max_tokens)

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
