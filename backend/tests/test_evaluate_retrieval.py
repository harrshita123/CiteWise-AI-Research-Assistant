import tempfile
import unittest
from pathlib import Path

from app.repositories.vectors import VectorRepository
from app.services.pdf_processing import TextChunk
from scripts.evaluate_retrieval import evaluate


class Embeddings:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_page_hit_at_k_uses_owner_and_paper_filters(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        vectors.save("alice", "paper-1", [TextChunk(0, 2, "Chinese libraries worked with OCLC.", "h1")],
                     [[1.0, 0.0]])
        vectors.save("bob", "paper-1", [TextChunk(1, 9, "Another user's matching text.", "h2")],
                     [[1.0, 0.0]])
        report = evaluate([
            {"owner_id": "alice", "paper_id": "paper-1", "question": "Which libraries?", "expected_pages": [2]},
            {"owner_id": "alice", "paper_id": "paper-1", "question": "Which page?", "expected_pages": [9]},
        ], Embeddings(), vectors)
        self.assertEqual(report["score"], 0.5)
        self.assertEqual(report["results"][0]["returned_pages"], [2])
        self.assertEqual(report["results"][1]["returned_pages"], [2])
