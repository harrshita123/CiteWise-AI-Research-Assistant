import unittest
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_admin
from app.models import UserResponse
from app.routers.evaluation import router


class Papers:
    async def find(self, owner_id, paper_id):
        if owner_id == "alice" and paper_id == "owned":
            return {"status": "ready", "page_count": 8}
        if owner_id == "alice" and paper_id == "pending":
            return {"status": "processing", "page_count": 8}
        return None


class Embeddings:
    def embed(self, questions):
        return [[1.0] for _ in questions]


class Passage:
    def __init__(self, page_number):
        self.page_number = page_number


class Vectors:
    def search(self, owner_id, paper_id, embedding, limit, question):
        assert (owner_id, paper_id, limit) == ("alice", "owned", 5)
        return [Passage(2), Passage(5)]


class Ingestion:
    embeddings = Embeddings()
    vectors = Vectors()


class Evaluations:
    def __init__(self):
        self.items = []

    async def create(self, owner_id, report):
        item = {"_id": "run-1", "owner_id": owner_id,
                "created_at": datetime.now(timezone.utc), **report}
        self.items.append(item)
        return item

    async def list_for_user(self, owner_id):
        return [item for item in self.items if item["owner_id"] == owner_id]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        app.state.paper_repository = Papers()
        app.state.ingestion_service = Ingestion()
        app.state.evaluation_repository = Evaluations()
        app.dependency_overrides[require_admin] = lambda: UserResponse(
            id="alice", email="alice@example.com", role="admin")
        self.client = TestClient(app)

    def test_run_reports_hit_at_5_and_saves_it(self):
        response = self.client.post("/api/v1/evaluation/runs", json={"cases": [
            {"paper_id": "owned", "question": "First?", "expected_page": 2},
            {"paper_id": "owned", "question": "Second?", "expected_page": 5},
            {"paper_id": "owned", "question": "Third?", "expected_page": 7},
        ]})
        self.assertEqual(response.status_code, 201)
        report = response.json()
        self.assertEqual(report["case_count"], 3)
        self.assertNotIn("hit_at_1", report)
        self.assertAlmostEqual(report["hit_at_5"], 2 / 3)
        self.assertEqual([item["returned_pages"] for item in report["results"]], [[2, 5]] * 3)
        self.assertEqual(self.client.get("/api/v1/evaluation/runs").json()[0]["id"], report["id"])

    def test_old_run_with_hit_at_1_still_loads(self):
        self.client.app.state.evaluation_repository.items.append({
            "_id": "old-run", "owner_id": "alice", "created_at": datetime.now(timezone.utc),
            "case_count": 1, "hit_at_1": 0.0, "hit_at_5": 1.0,
            "results": [{"paper_id": "owned", "question": "What?", "expected_page": 5,
                         "returned_pages": [2, 5], "hit_at_1": False, "hit_at_5": True}],
        })
        response = self.client.get("/api/v1/evaluation/runs")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("hit_at_1", response.json()[0])
        self.assertNotIn("hit_at_1", response.json()[0]["results"][0])

    def test_rejects_foreign_unready_and_invalid_pages(self):
        for paper_id, page, status in (("foreign", 1, 404), ("pending", 1, 409), ("owned", 9, 422)):
            response = self.client.post("/api/v1/evaluation/runs", json={"cases": [
                {"paper_id": paper_id, "question": "What?", "expected_page": page},
            ]})
            self.assertEqual(response.status_code, status)
        self.assertEqual(self.client.app.state.evaluation_repository.items, [])
