from pathlib import Path

import chromadb
from chromadb.config import Settings

from ..paper_models import ChunkResponse
from ..services.pdf_processing import TextChunk


class VectorRepository:
    def __init__(self, directory: Path):
        client = chromadb.PersistentClient(path=str(directory), settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_or_create_collection("paper_chunks_minilm_v1", embedding_function=None)

    @staticmethod
    def paper_filter(owner_id: str, paper_id: str) -> dict:
        return {"$and": [{"owner_id": owner_id}, {"paper_id": paper_id}]}

    def save(self, owner_id: str, paper_id: str, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        self.collection.upsert(
            ids=[f"{paper_id}:{chunk.chunk_index}" for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[{
                "owner_id": owner_id, "paper_id": paper_id,
                "page_number": chunk.page_number, "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
            } for chunk in chunks],
        )

    def preview(self, owner_id: str, paper_id: str, offset: int = 0, limit: int = 10) -> list[ChunkResponse]:
        result = self.collection.get(
            ids=[f"{paper_id}:{index}" for index in range(offset, offset + limit)],
            where=self.paper_filter(owner_id, paper_id), include=["documents", "metadatas"],
        )
        chunks = [ChunkResponse(text=text, **metadata)
                  for text, metadata in zip(result["documents"], result["metadatas"])]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def remove(self, owner_id: str, paper_id: str) -> None:
        self.collection.delete(where=self.paper_filter(owner_id, paper_id))
