from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_client_id: str
    mongodb_uri: str
    mongodb_database: str = "citewise"
    jwt_secret: str = Field(min_length=32)
    allowed_origins: str = "http://localhost:5173"
    session_cookie_secure: bool = False
    session_cookie_name: str = "citewise_session"
    session_duration_minutes: int = Field(default=60, ge=5, le=1440)
    storage_directory: Path = Path(__file__).resolve().parents[1] / "storage"
    max_pdf_size_mb: int = Field(default=25, ge=1, le=25)
    max_pdf_pages: int = Field(default=300, ge=1, le=1000)
    max_pdf_characters: int = Field(default=2_000_000, ge=1000)
    embedding_batch_size: int = Field(default=32, ge=1, le=64)

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
