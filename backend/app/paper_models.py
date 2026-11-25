from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class PaperResponse(BaseModel):
    id: str
    filename: str
    status: Literal["processing", "ready", "failed", "not_research", "needs_review"]
    document_state: Literal["READY_FOR_RESEARCH", "NEEDS_REVIEW", "REJECTED"] | None = None
    page_count: int = 0
    chunk_count: int = 0
    embedding_model: str
    embedding_dimensions: int
    created_at: datetime
    error: str | None = None
    document_category: str | None = None
    publication_identity: dict | None = None


class ChunkResponse(BaseModel):
    chunk_index: int
    page_number: int
    text: str
    content_hash: str
    section: str | None = None


class ExtractedPaperMetadata(BaseModel):
    title: str | None = None
    authors: list[str] | None = None
    abstract: str | None = None
    version: str | None = None
    year: int | None = None
    venue: str | None = None
    publisher: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None


class PaperDetail(BaseModel):
    paper: PaperResponse
    chunks: list[ChunkResponse]
    extracted_metadata: ExtractedPaperMetadata | None = None


class PaperSummary(BaseModel):
    total: int
    ready: int
    processing: int
    failed: int
