import copy
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_admin
from app.models import UserResponse
from app.paper_models import ChunkResponse
from app.providers.answers import AnswerRateLimitError
from app.routers.evaluation import router


class Papers:
    async def find(self, owner, paper_id):
        if owner == "alice" and paper_id in ("p1", "p2"):
            return {"status": "ready", "file_hash": "hash-" + paper_id, "chunk_count": 1}
        return None


class Vectors:
    def sample(self, owner, paper_id, count, limit):
        return [ChunkResponse(chunk_index=0, page_number=1, content_hash="chunk-" + paper_id,
                              text="This paper reports a sample of 20 participants.")]


class Ingestion:
    vectors = Vectors()
    def preview(self, owner, paper_id, offset, limit):
        return self.vectors.sample(owner, paper_id, 1, 1)


class Provider:
    def propose_evaluation_cases(self, evidence):
        source = evidence[0].id
        return [
            {"question": "How many participants?", "answer": "20", "behavior": "answer", "axis": "approach",
             "label": "Quantitative", "source_id": source, "quote": "20 participants"},
            {"question": "HOW MANY PARTICIPANTS?", "answer": "20", "behavior": "answer", "axis": "", "label": "",
             "source_id": source, "quote": "20 participants"},
            {"question": "Invalid citation?", "answer": "wrong", "behavior": "answer", "axis": "", "label": "",
             "source_id": source, "quote": "30 participants"},
            {"question": "What was not measured?", "answer": "Not reported", "behavior": "abstain",
             "axis": "", "label": "", "source_id": "", "quote": ""},
        ]


class Repository:
    def __init__(self):
        self.documents = {}
        self.versions = {}
    async def create_proposals(self, owner, paper_id, cases):
        doc = {"_id": str(len(self.documents)), "owner_id": owner, "paper_id": paper_id, "cases": cases,
               "revision": 0, "created_at": datetime.now(timezone.utc)}
        self.documents[doc["_id"]] = copy.deepcopy(doc)
        return doc
    async def find_proposals(self, owner, proposal_id):
        doc = self.documents.get(proposal_id)
        return copy.deepcopy(doc) if doc and doc["owner_id"] == owner else None
    async def replace_proposals(self, owner, proposal_id, revision, cases):
        doc = self.documents[proposal_id]
        if doc["owner_id"] != owner or doc["revision"] != revision:
            return False
        doc.update(revision=revision + 1, cases=copy.deepcopy(cases))
        return True
    async def list_proposals(self, owner, paper_id):
        return [copy.deepcopy(doc) for doc in self.documents.values() if doc["owner_id"] == owner and doc["paper_id"] == paper_id]
    async def create_benchmark_version(self, owner, benchmark, fingerprint):
        self.versions[(owner, benchmark["version"])] = {"benchmark": benchmark, "fingerprint": fingerprint}


class ProposalTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        app.state.paper_repository = Papers()
        app.state.ingestion_service = Ingestion()
        app.state.evaluation_repository = Repository()
        app.state.answer_provider = Provider()
        app.dependency_overrides[require_admin] = lambda: UserResponse(id="alice", email="a@example.com", role="admin")
        self.client = TestClient(app)

    def generate(self, paper="p1"):
        response = self.client.post(f"/api/v1/evaluation/papers/{paper}/proposals")
        self.assertEqual(response.status_code, 201)
        return response.json()

    def test_pending_validation_review_and_freeze(self):
        draft = self.generate()
        self.assertEqual(len(draft["cases"]), 2)
        self.assertTrue(all(item["status"] == "pending" for item in draft["cases"]))
        case = draft["cases"][0]["case"]
        case["expected_answer"] = "Twenty participants"
        url = f"/api/v1/evaluation/proposals/{draft['id']}/cases/{case['id']}"
        reviewed = self.client.put(url, json={"revision": 0, "status": "approved", "case": case})
        self.assertEqual(reviewed.status_code, 200)
        self.assertEqual(reviewed.json()["cases"][0]["case"]["reviewed_by"], "alice")
        self.assertEqual(self.client.put(url, json={"revision": 0, "status": "approved", "case": case}).status_code, 409)
        missing = reviewed.json()["cases"][1]["case"]
        url2 = f"/api/v1/evaluation/proposals/{draft['id']}/cases/{missing['id']}"
        self.assertEqual(self.client.put(url2, json={"revision": 1, "status": "rejected", "case": missing}).status_code, 200)
        frozen = self.client.post("/api/v1/evaluation/proposals/freeze", json={"version": "v1", "proposal_ids": [draft["id"]]})
        self.assertEqual(frozen.status_code, 201)
        snapshot = self.client.app.state.evaluation_repository.versions[("alice", "v1")]["benchmark"]
        self.assertEqual(len(snapshot["cases"]), 1)
        self.assertEqual(snapshot["cases"][0]["expected_answer"], "Twenty participants")

    def test_foreign_and_cross_paper_evidence_are_rejected(self):
        self.assertEqual(self.client.post("/api/v1/evaluation/papers/foreign/proposals").status_code, 404)
        draft = self.generate("p1")
        case = draft["cases"][0]["case"]
        case["evidence"][0]["content_hash"] = "chunk-p2"
        url = f"/api/v1/evaluation/proposals/{draft['id']}/cases/{case['id']}"
        self.assertEqual(self.client.put(url, json={"revision": 0, "status": "approved", "case": case}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/evaluation/papers/p2/proposals").json(), [])

    def test_rate_limit_preserved(self):
        def limited(evidence):
            raise AnswerRateLimitError("60")
        self.client.app.state.answer_provider = SimpleNamespace(propose_evaluation_cases=limited)
        response = self.client.post("/api/v1/evaluation/papers/p1/proposals")
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "60")
        self.assertEqual(self.client.app.state.evaluation_repository.documents, {})
