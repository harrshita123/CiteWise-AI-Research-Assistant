import os
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient
from pymongo import AsyncMongoClient

from app.dependencies import get_current_user
from app.models import UserResponse
from app.providers.answers import GeneratedAnswer
from app.repositories.conversations import ConversationRepository
from app.routers.evaluation import router as evaluation_router
from test_research import Answers
import test_research


@unittest.skipUnless(os.getenv("CITEWISE_TEST_MONGO_URI"), "test MongoDB not configured")
class AnswerQualityMongoTests(unittest.TestCase):
    def test_automatic_scores_filtering_reload_and_admin_owner_scope(self):
        fixture = test_research.ResearchTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        app = fixture.client.app
        app.include_router(evaluation_router)
        database = "citewise_quality_test_" + uuid4().hex
        actor = ["alice", "user"]
        app.dependency_overrides[get_current_user] = lambda: UserResponse(
            id=actor[0], email=f"{actor[0]}@example.com", role=actor[1])
        prefix = "/api/v1/evaluation/answer-quality"
        with TestClient(app) as client:
            connection = AsyncMongoClient(os.environ["CITEWISE_TEST_MONGO_URI"], serverSelectionTimeoutMS=2000)
            repository = ConversationRepository(connection[database].conversations)
            app.state.conversation_repository = repository
            try:
                created = []
                for text in ("Alpha result [c0-hash-a].", "Alpha result."):
                    app.state.answer_provider = Answers(GeneratedAnswer(text, ["c0-hash-a"]))
                    response = client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "Alpha?"})
                    self.assertEqual(response.status_code, 200, response.text)
                    created.append(response.json())
                detail = f"{prefix}/{created[0]['conversation_id']}/{created[0]['message_id']}/1"
                self.assertEqual(client.get(prefix).status_code, 403)
                self.assertEqual(client.get(detail).status_code, 403)
                actor[1] = "admin"
                refreshed = client.get(prefix, params={"paper_id": "paper-1", "evidence": "pdf"})
                self.assertEqual(refreshed.status_code, 200, refreshed.text)
                result = refreshed.json()
                self.assertEqual((result["evaluated_answers"], result["total_versions"]), (2, 2))
                self.assertEqual(result["metrics"]["citation_coverage"],
                                 {"average": .5, "measured": 2, "numerator": 1, "denominator": 2})
                self.assertEqual(result["metrics"]["citation_precision"],
                                 {"average": 1, "measured": 1, "numerator": 1, "denominator": 1})
                self.assertEqual(client.get(prefix, params={"date_from": "2000-01-01", "date_to": "2000-01-01"}).json()["rows"], [])
                self.assertEqual(client.get(detail).json()["metrics"]["citation_coverage"]["score"], 1)
                actor[0] = "bob"
                self.assertEqual(client.get(prefix).json()["rows"], [])
                self.assertEqual(client.get(prefix, params={"paper_id": "paper-1"}).status_code, 404)
                self.assertEqual(client.get(detail).status_code, 404)
            finally:
                client.portal.call(connection.drop_database, database)
                client.portal.call(connection.close)
