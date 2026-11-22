import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import get_current_user
from app.models import UserResponse
from app.repositories.vectors import VectorRepository
from app.routers.research import router
from app.services.pdf_processing import TextChunk, extract_pages, create_chunks, PdfValidationError
from tests.test_papers import make_pdf, test_tokenizer
from app.services.question_recommendations import QuestionRecommendationService


class Embeddings:
    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "method" in lowered or "design" in lowered:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "finding" in lowered or "result" in lowered:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            elif "limit" in lowered or "caveat" in lowered:
                vectors.append([0.0, 0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 1.0])
        return vectors


class Provider:
    def __init__(self):
        self.calls = 0

    def available(self):
        return True

    def recommend_questions(self, mode, context, evidence, limit):
        self.calls += 1
        return [
            "What objective does the paper describe?",
            "What methodology or study design is described in the paper?",
            "What findings does the paper report?",
            "What limitations or caveats does the paper identify?",
            "What national education policy changed?",
        ]


class Papers:
    paper = {"status": "ready", "file_hash": "file-a", "chunk_count": 4, "filename": "study.pdf"}

    async def find(self, owner_id, paper_id):
        return self.paper if owner_id == "alice" and paper_id == "paper-1" else None


class Conversations:
    def __init__(self):
        self.items = {
            "conversation-1": {"_id": "conversation-1", "owner_id": "alice", "paper_id": "paper-1",
                               "messages": [{"id": "message-1", "question": "What objective does the paper describe?",
                                             "answer": "It studies monitoring.", "status": "answered", "sources": []}]}
        }

    async def find(self, owner_id, conversation_id):
        item = self.items.get(conversation_id)
        return item if item and item["owner_id"] == owner_id else None


class Ingestion:
    def __init__(self, vectors):
        self.vectors = vectors
        self.embeddings = Embeddings()


class QuestionRecommendationTests(unittest.TestCase):
    def setUp(self):
        with QuestionRecommendationService._cache_lock:
            QuestionRecommendationService._cache.clear()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.vectors = VectorRepository(Path(directory.name))
        self.vectors.save("alice", "paper-1", [
            TextChunk(0, 1, "The objective of this paper is to study online examination monitoring.", "objective"),
            TextChunk(1, 2, "Methodology: the study used a survey design with students.", "method"),
            TextChunk(2, 3, "Results and findings showed students reported privacy concerns.", "findings"),
            TextChunk(3, 4, "A limitation was the small sample and short observation period.", "limits"),
        ], Embeddings().embed([
            "objective", "methodology", "findings", "limitations",
        ]))
        self.provider = Provider()
        self.service = QuestionRecommendationService(self.vectors, Embeddings().embed, self.provider)
        self.paper = {"file_hash": "file-a", "chunk_count": 4}

    def test_starter_questions_are_grounded_diverse_and_cached(self):
        first = self.service.starter("alice", "paper-1", self.paper)
        second = self.service.starter("alice", "paper-1", self.paper)
        self.assertEqual([item.text for item in first], [item.text for item in second])
        self.assertEqual(len(first), 4)
        self.assertEqual(len({item.text.casefold() for item in first}), 4)
        self.assertEqual(self.provider.calls, 1)
        self.assertNotIn("What national education policy changed?", [item.text for item in first])

    def test_previous_or_other_users_questions_are_not_recommended(self):
        result = self.service.alternatives("alice", "paper-1", "What national education policy changed?",
                                           self.paper, ["What objective does the paper describe?"])
        self.assertNotIn("What objective does the paper describe?", [item.text for item in result])
        self.assertEqual(self.service.starter("bob", "paper-1", self.paper), [])

    def test_missing_provider_reports_error_instead_of_caching_empty_results(self):
        fallback = QuestionRecommendationService(self.vectors, Embeddings().embed, None)
        with self.assertRaises(RuntimeError):
            fallback.starter("alice", "paper-1", {**self.paper, "file_hash": "file-without-provider"})
        self.assertEqual(len(QuestionRecommendationService._cache), 0)

    def test_distinct_extracted_pdfs_produce_scoped_questions(self):
        class TopicProvider:
            def recommend_questions(self, mode, context, evidence, limit):
                text = " ".join(item.text.lower() for item in evidence)
                if "volcanoes" in text:
                    return ["What does the study report about volcanoes?"]
                if "pollinators" in text:
                    return ["What does the study report about pollinators?"]
                return []
        service = QuestionRecommendationService(self.vectors, Embeddings().embed, TopicProvider())
        for paper_id, text in [("volcano", "This study reports that volcanoes produce ash during eruptions."),
                               ("bees", "This study reports that pollinators help flowering plants reproduce.")]:
            _, pages = extract_pages(make_pdf(text), 10, 10000)
            chunks = create_chunks(pages, test_tokenizer())
            self.vectors.save("alice", paper_id, chunks, Embeddings().embed([chunk.text for chunk in chunks]))
            result = service.starter("alice", paper_id, {"file_hash": paper_id, "chunk_count": len(chunks)})
            self.assertEqual(len(result), 1)
            self.assertIn("volcanoes" if paper_id == "volcano" else "pollinators", result[0].text)
            self.assertIn("volcanoes" if paper_id == "volcano" else "pollinators", result[0].source.text)

    def test_empty_pdf_is_rejected_safely(self):
        with self.assertRaises(PdfValidationError):
            extract_pages(make_pdf(""), 10, 10000)

    def test_followups_use_answer_context_and_exclude_previous_question(self):
        class ContextProvider(Provider):
            def recommend_questions(self, mode, context, evidence, limit):
                self.mode, self.context = mode, context
                return ["What objective does the paper describe?", "What findings does the paper report?"]
        provider = ContextProvider()
        service = QuestionRecommendationService(self.vectors, Embeddings().embed, provider)
        sources = [chunk.model_dump() for chunk in self.vectors.sample("alice", "paper-1", 4)]
        items = service.follow_up("alice", "paper-1", "What objective does the paper describe?",
                                  "The paper studies online examination monitoring.", sources,
                                  ["What objective does the paper describe?"])
        self.assertEqual(provider.mode, "follow_up")
        self.assertIn("online examination monitoring", provider.context)
        self.assertNotIn("What objective does the paper describe?", [item.text for item in items])
        self.assertEqual([item.text for item in items], ["What findings does the paper report?"])

    def test_distinct_questions_in_same_category_are_not_discarded(self):
        class ThresholdProvider:
            def recommend_questions(self, mode, context, evidence, limit):
                return ["What is the citation precision target?", "What is the citation coverage target?",
                        "What is the retrieval latency target?"]
        text = ("Citation precision target is 90 percent. Citation coverage target is 90 percent. "
                "Retrieval latency target is 8 seconds.")
        self.vectors.save("alice", "thresholds", [TextChunk(0, 1, text, "thresholds-hash")], [[0, 0, 0, 1]])
        service = QuestionRecommendationService(self.vectors, Embeddings().embed, ThresholdProvider())
        items = service.starter("alice", "thresholds", {"file_hash": "thresholds", "chunk_count": 1})
        self.assertEqual(len(items), 3)
        self.assertTrue(all(item.recommendation_type == "paper" for item in items))

    def test_endpoints_require_owner_and_handle_missing_conversations(self):
        app = FastAPI()
        app.include_router(router)
        app.state.paper_repository = Papers()
        app.state.ingestion_service = Ingestion(self.vectors)
        app.state.answer_provider = self.provider
        app.state.conversation_repository = Conversations()
        identity = ["alice"]
        app.dependency_overrides[get_current_user] = lambda: UserResponse(
            id=identity[0], email=f"{identity[0]}@example.com", role="user")
        client = TestClient(app)

        response = client.get("/api/v1/research/papers/paper-1/recommendations")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["recommendations"]), 4)
        self.assertEqual(client.post("/api/v1/research/conversations/missing/recommendations",
                                     json={"message_id": "message-1"}).status_code, 404)
        identity[0] = "bob"
        self.assertEqual(client.get("/api/v1/research/papers/paper-1/recommendations").status_code, 404)

    def test_partial_answer_with_verified_pdf_claim_allows_followups(self):
        app = FastAPI()
        app.include_router(router)
        app.state.paper_repository = Papers()
        app.state.ingestion_service = Ingestion(self.vectors)
        app.state.answer_provider = self.provider
        app.state.conversation_repository = Conversations()
        message = app.state.conversation_repository.items["conversation-1"]["messages"][0]
        message.update(status="partially_supported", sources=[chunk.model_dump() for chunk in
                       self.vectors.sample("alice", "paper-1", 4)],
                       claims=[{"section": "pdf", "status": "supported", "supporting": [{"source_id": "c1", "quote": "survey design"}]}])
        app.dependency_overrides[get_current_user] = lambda: UserResponse(id="alice", email="alice@example.com", role="user")
        client = TestClient(app)
        response = client.post("/api/v1/research/conversations/conversation-1/recommendations", json={"message_id": "message-1"})
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json()["recommendations"]), 0)
        self.assertEqual(self.provider.calls, 1)


if __name__ == "__main__":
    unittest.main()
