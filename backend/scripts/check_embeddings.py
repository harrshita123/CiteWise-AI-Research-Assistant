"""Run a real local embedding check without connecting to MongoDB."""
import tempfile
from pathlib import Path

import numpy as np
import pymupdf

from app.config import Settings
from app.providers.embeddings import LocalEmbeddings
from app.repositories.vectors import VectorRepository
from app.services.ingestion import IngestionService
from app.services.pdf_processing import PageText, create_chunks


def main() -> None:
    model_directory = Path(__file__).resolve().parents[1] / "storage" / "models"
    with tempfile.TemporaryDirectory() as directory:
        settings = Settings(
            _env_file=None, google_client_id="local-check", mongodb_uri="mongodb://localhost",
            jwt_secret="local-check-only-" * 3, storage_directory=Path(directory),
        )
        service = IngestionService(settings)
        service.embeddings = LocalEmbeddings(model_directory)
        with pymupdf.open() as document:
            document.new_page().insert_text((50, 60), "Research methods use experiments and evidence.")
            document.new_page().insert_text((50, 60), "The results support verifiable research citations.")
            content = document.tobytes()
        counts = service.process("demo-owner", "demo-paper", content)
        assert counts == {"page_count": 2, "chunk_count": 2}, counts
        reopened = VectorRepository(Path(directory) / "chroma")
        stored = reopened.collection.get(
            where=reopened.paper_filter("demo-owner", "demo-paper"), include=["embeddings"],
        )
        vectors = stored["embeddings"]
        assert vectors.shape == (2, 384), vectors.shape
        assert np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        assert not np.allclose(vectors[0], vectors[1])
        assert reopened.preview("another-owner", "demo-paper") == []
        tokenizer = service.embeddings.tokenizer()
        text = "Evidence-based retrieval, tokenization and reproducibility. " * 200
        chunks = create_chunks([PageText(1, text)], tokenizer)
        assert len(chunks) > 1
        assert all(len(tokenizer.encode(chunk.text).ids) <= 254 for chunk in chunks)
        assert chunks[-1].text.endswith("reproducibility.")
        print("PASS: real PDF -> page-aware chunks -> MiniLM -> persistent ChromaDB")
        print("PASS: 2 distinct, normalized 384-dimensional vectors; owner isolation; long-text token limits")


if __name__ == "__main__":
    main()
