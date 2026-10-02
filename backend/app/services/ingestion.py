from functools import cached_property
from pathlib import Path

from ..config import Settings
from ..providers.embeddings import LocalEmbeddings
from ..paper_models import ChunkResponse
from ..repositories.vectors import VectorRepository
from .pdf_processing import create_chunks, extract_pages
from .pdf_storage import CloudinaryPdfStorage, LocalPdfStorage


class IngestionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = LocalEmbeddings(settings.storage_directory / "models")

    @cached_property
    def vectors(self) -> VectorRepository:
        return VectorRepository(self.settings.storage_directory / "chroma")

    @cached_property
    def local_storage(self) -> LocalPdfStorage:
        return LocalPdfStorage(self.settings.storage_directory / "uploads")

    @cached_property
    def cloudinary_storage(self) -> CloudinaryPdfStorage:
        return CloudinaryPdfStorage(self.settings)

    def storage_for(self, provider: str):
        if provider == "local":
            return self.local_storage
        if provider == "cloudinary":
            return self.cloudinary_storage
        raise ValueError("Unknown PDF storage provider.")

    def file_path(self, paper_id: str) -> Path:
        # Existing local papers still use their server-generated file path.
        return self.local_storage.path(paper_id)

    def process(self, owner_id: str, paper_id: str, content: bytes) -> dict[str, int]:
        page_count, pages = extract_pages(
            content, self.settings.max_pdf_pages, self.settings.max_pdf_characters,
        )
        chunks = create_chunks(pages, self.embeddings.tokenizer())

        batch_size = self.settings.embedding_batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            embeddings = self.embeddings.embed([chunk.text for chunk in batch])
            self.vectors.save(owner_id, paper_id, batch, embeddings)

        self.cloudinary_storage.save(paper_id, content)
        return {"page_count": page_count, "chunk_count": len(chunks)}

    def cleanup(self, owner_id: str, paper_id: str, provider: str = "local") -> None:
        self.vectors.remove(owner_id, paper_id)
        self.storage_for(provider).delete(paper_id)

    def preview(self, owner_id: str, paper_id: str, offset: int = 0, limit: int = 10) -> list[ChunkResponse]:
        return self.vectors.preview(owner_id, paper_id, offset, limit)
