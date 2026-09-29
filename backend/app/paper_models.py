from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class PaperResponse(BaseModel):
    id: str
    filename: str
    status: Literal["processing", "ready", "failed"]
    page_count: int = 0
    chunk_count: int = 0
    embedding_model: str
    embedding_dimensions: int
    created_at: datetime
    error: str | None = None


class ChunkResponse(BaseModel):
    chunk_index: int
    page_number: int
    text: str
    content_hash: str


class PaperDetail(BaseModel):
    paper: PaperResponse
    chunks: list[ChunkResponse]


class PaperSummary(BaseModel):
    total: int
    ready: int
    processing: int
    failed: int
