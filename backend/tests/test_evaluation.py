import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

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
        self.assertAlmostEqual(report["mean_reciprocal_rank"], 0.5)
        self.assertEqual([item["reciprocal_rank"] for item in report["results"]], [1.0, 0.5, 0.0])
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
        self.assertIsNone(response.json()[0]["mean_reciprocal_rank"])
        self.assertIsNone(response.json()[0]["results"][0]["reciprocal_rank"])

    def test_rejects_foreign_unready_and_invalid_pages(self):
        for paper_id, page, status in (("foreign", 1, 404), ("pending", 1, 409), ("owned", 9, 422)):
            response = self.client.post("/api/v1/evaluation/runs", json={"cases": [
                {"paper_id": paper_id, "question": "What?", "expected_page": page},
            ]})
            self.assertEqual(response.status_code, status)
        self.assertEqual(self.client.app.state.evaluation_repository.items, [])

    def test_benchmark_score_checks_owned_paper_hash_and_exact_evidence(self):
        from app.paper_models import ChunkResponse

        class BenchmarkPapers:
            async def find(self, owner_id, paper_id):
                return {"status": "ready", "file_hash": "pdf-hash"} if (owner_id, paper_id) == ("alice", "owned") else None

        class BenchmarkIngestion:
            def preview(self, owner_id, paper_id, offset, limit):
                self.last_query = (owner_id, paper_id, offset, limit)
                return [ChunkResponse(chunk_index=3, page_number=2, content_hash="chunk-hash", text="The result was observed.")]

        self.client.app.state.paper_repository = BenchmarkPapers()
        ingestion = BenchmarkIngestion()
        self.client.app.state.ingestion_service = ingestion
        case = {"id": "case", "paper_id": "owned", "paper_hash": "pdf-hash", "question": "What was observed?",
                "expected_behavior": "answer", "expected_answer": "The result", "review_status": "approved",
                "reviewed_by": "Reviewer", "evidence": [{"page": 2, "chunk_index": 3,
                "content_hash": "chunk-hash", "quote": "result was observed"}]}
        payload = {"benchmark": {"version": "1.0", "cases": [case]},
                   "observations": [{"case_id": "case", "returned_pages": [2], "answer_correct": True}]}
        result = self.client.post("/api/v1/evaluation/benchmark/score", json=payload)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["overall"]["hit_at_5"], 1.0)
        self.assertEqual(ingestion.last_query, ("alice", "owned", 3, 1))
        case["evidence"][0]["quote"] = "invented evidence"
        self.assertEqual(self.client.post("/api/v1/evaluation/benchmark/score", json=payload).status_code, 422)
        case["paper_id"] = "foreign"
        self.assertEqual(self.client.post("/api/v1/evaluation/benchmark/score", json=payload).status_code, 404)

    def test_run_validates_ownership_and_saves_report(self):
        from app.paper_models import ChunkResponse
        class OwnedPaper:
            async def find(self, owner_id, paper_id):
                return {"status": "ready", "file_hash": "file"} if (owner_id, paper_id) == ("alice", "owned") else None
        class Evidence:
            def preview(self, owner_id, paper_id, offset, limit):
                return [ChunkResponse(chunk_index=0, page_number=1, content_hash="chunk", text="Evidence here")]
        self.client.app.state.paper_repository = OwnedPaper()
        self.client.app.state.ingestion_service = Evidence()
        self.client.app.state.evaluation_repository.create_benchmark_report = AsyncMock(return_value={
            "_id": "report", "created_at": datetime.now(timezone.utc)})
        case = {"id": "1", "paper_id": "owned", "paper_hash": "file", "question": "What?",
                "expected_behavior": "answer", "expected_answer": "Evidence", "review_status": "approved",
                "reviewed_by": "alice", "evidence": [{"page": 1, "chunk_index": 0,
                    "content_hash": "chunk", "quote": "Evidence"}]}
        payload = {"version": "v1", "cases": [case]}
        with patch("app.routers.evaluation.run_benchmark", new_callable=AsyncMock,
                   return_value={"metrics": {}, "results": []}) as runner:
            self.assertEqual(self.client.post("/api/v1/evaluation/benchmark/run", json=payload).status_code, 201)
            case["paper_id"] = "foreign"
            self.assertEqual(self.client.post("/api/v1/evaluation/benchmark/run", json=payload).status_code, 404)
            self.assertEqual(runner.await_count, 1)
        self.client.app.state.evaluation_repository.create_benchmark_report.assert_awaited_once()

    def test_human_judgments_do_not_override_retrieval_or_score_incomplete(self):
        from app.services.evaluation_benchmark import BenchmarkCase, EvidenceRef, FrozenBenchmark
        case = BenchmarkCase(id="one", paper_id="owned", paper_hash="hash", question="What?",
            expected_behavior="answer", expected_answer="gold", evidence=(EvidenceRef(page=1, chunk_index=2,
            content_hash="chunk", quote="gold"),), review_status="approved", reviewed_by="alice")
        benchmark = FrozenBenchmark(version="v1", cases=(case,))
        report = {"metrics": {"benchmark_version": "v1", "benchmark_hash": benchmark.fingerprint()},
            "results": [{"case_id": "one", "execution_status": "completed", "latency_ms": 100,
                "retrieved_chunks": [{"page": 1, "chunk_index": 2, "content_hash": "chunk"}]}]}
        repo = self.client.app.state.evaluation_repository
        repo.find_benchmark_report = AsyncMock(return_value=report)
        repo.find_benchmark_version = AsyncMock(return_value={"benchmark": benchmark.model_dump(mode="json"),
                                                               "fingerprint": benchmark.fingerprint()})
        repo.update_benchmark_report = AsyncMock(return_value=True)
        response = self.client.put("/api/v1/evaluation/benchmark/reports/report/judgments", json={
            "judgments": [{"case_id": "one", "answer_correct": True, "citations_supported": 1,
                           "citations_total": 2, "claims_cited": 1, "claims_requiring_citation": 1}]})
        self.assertEqual(response.status_code, 200)
        scores = response.json()["metrics"]["overall"]
        self.assertEqual((scores["hit_at_5"], scores["answer_correctness"], scores["citation_precision"]), (1, 1, 0.5))
        invalid = self.client.put("/api/v1/evaluation/benchmark/reports/report/judgments", json={
            "judgments": [{"case_id": "one", "citations_supported": 2, "citations_total": 1}]})
        self.assertEqual(invalid.status_code, 422)
        report["results"][0]["execution_status"] = "rate_limited"
        self.assertEqual(self.client.put("/api/v1/evaluation/benchmark/reports/report/judgments", json={
            "judgments": [{"case_id": "one", "answer_correct": True}]}).status_code, 422)

    def test_partial_judgment_save_keeps_other_cases_and_incomplete_cases(self):
        from app.services.evaluation_benchmark import BenchmarkCase, EvidenceRef, FrozenBenchmark
        cases = tuple(BenchmarkCase(id=str(i), paper_id="owned", paper_hash="hash", question="What?",
            expected_behavior="answer", expected_answer="gold", evidence=(EvidenceRef(page=1, chunk_index=2,
            content_hash="chunk", quote="gold"),), review_status="approved", reviewed_by="alice") for i in range(3))
        benchmark = FrozenBenchmark(version="v1", cases=cases)
        report = {"metrics": {"benchmark_version": "v1", "benchmark_hash": benchmark.fingerprint()},
            "judgments": [{"case_id": "0", "answer_correct": True}],
            "results": [{"case_id": str(i), "execution_status": "not_run" if i == 2 else "completed",
                         "retrieved_chunks": [{"page": 1, "chunk_index": 2, "content_hash": "chunk"}]} for i in range(3)]}
        repo = self.client.app.state.evaluation_repository
        repo.find_benchmark_report = AsyncMock(return_value=report)
        repo.find_benchmark_version = AsyncMock(return_value={"benchmark": benchmark.model_dump(mode="json"),
                                                               "fingerprint": benchmark.fingerprint()})
        repo.update_benchmark_report = AsyncMock(return_value=True)
        response = self.client.put("/api/v1/evaluation/benchmark/reports/report/judgments", json={
            "judgments": [{"case_id": "1", "answer_correct": False}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["judgments"]), 2)
        self.assertEqual(response.json()["metrics"]["overall"]["answer_correctness"], 0.5)
        self.assertEqual(response.json()["metrics"]["overall"]["completed_count"], 2)
        response = self.client.put("/api/v1/evaluation/benchmark/reports/report/judgments", json={
            "judgments": [{"case_id": "1", "citations_supported": 0}]})
        self.assertEqual(response.status_code, 422)
        self.assertIn("both be entered", response.json()["detail"])

    def test_resume_only_incomplete_case_and_preserves_review(self):
        from app.services.evaluation_benchmark import BenchmarkCase, EvidenceRef, FrozenBenchmark
        cases = tuple(BenchmarkCase(id=str(i), paper_id=f"p{i}", paper_hash=f"hash{i}", question="What?",
            expected_behavior="answer", expected_answer="gold", evidence=(EvidenceRef(page=1, chunk_index=0,
            content_hash="chunk", quote="gold"),), review_status="approved", reviewed_by="alice") for i in range(2))
        benchmark = FrozenBenchmark(version="v1", cases=cases)
        old = {"case_id": "0", "paper_id": "p0", "execution_status": "completed", "answer": "original",
               "citations": [{"id": "old"}], "latency_ms": 5,
               "retrieved_chunks": [{"page": 1, "chunk_index": 0, "content_hash": "chunk"}]}
        report = {"_id": "r", "created_at": datetime.now(timezone.utc),
            "metrics": {"benchmark_version": "v1", "benchmark_hash": benchmark.fingerprint()},
            "results": [old, {"case_id": "1", "execution_status": "not_run"}],
            "judgments": [{"case_id": "0", "answer_correct": False}]}
        repo = self.client.app.state.evaluation_repository
        repo.find_benchmark_report = AsyncMock(return_value=report)
        repo.acquire_resume = AsyncMock(return_value=report)
        repo.release_resume = AsyncMock()
        repo.save_resume = AsyncMock(return_value=True)
        repo.find_benchmark_version = AsyncMock(return_value={"benchmark": benchmark.model_dump(mode="json"),
                                                               "fingerprint": benchmark.fingerprint()})
        new = {"case_id": "1", "paper_id": "p1", "execution_status": "completed", "answer": "new",
               "retrieved_chunks": [{"page": 1, "chunk_index": 0, "content_hash": "chunk"}]}
        with patch("app.routers.evaluation._validate_benchmark", new_callable=AsyncMock), \
             patch("app.routers.evaluation.run_benchmark", new_callable=AsyncMock, return_value={"results": [new]}) as runner:
            response = self.client.post("/api/v1/evaluation/benchmark/reports/r/resume")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["results"][0], old)
        self.assertEqual(result["results"][1]["answer"], "new")
        self.assertFalse(result["judgments"][0]["answer_correct"])
        self.assertEqual(result["metrics"]["overall"]["answer_correctness"], 0)
        self.assertEqual(runner.await_args.args[0].cases[0].paper_id, "p1")
        repo.save_resume.assert_awaited_once()

    def test_resume_obeys_cooldown_and_owner(self):
        from app.services.evaluation_benchmark import BenchmarkCase, EvidenceRef, FrozenBenchmark
        case = BenchmarkCase(id="one", paper_id="p", paper_hash="hash", question="What?", expected_behavior="answer",
            expected_answer="gold", evidence=(EvidenceRef(page=1, chunk_index=0, content_hash="chunk", quote="gold"),),
            review_status="approved", reviewed_by="alice")
        benchmark = FrozenBenchmark(version="v1", cases=(case,))
        report = {"_id": "r", "created_at": datetime.now(timezone.utc),
                  "metrics": {"benchmark_version": "v1", "benchmark_hash": benchmark.fingerprint()},
                  "results": [{"case_id": "one", "execution_status": "rate_limited", "retry_after": "90"}]}
        repo = self.client.app.state.evaluation_repository
        repo.find_benchmark_report = AsyncMock(return_value=report)
        repo.acquire_resume = AsyncMock(return_value=report)
        repo.release_resume = AsyncMock()
        repo.find_benchmark_version = AsyncMock(return_value={"benchmark": benchmark.model_dump(mode="json"),
                                                               "fingerprint": benchmark.fingerprint()})
        with patch("app.routers.evaluation.run_benchmark", new_callable=AsyncMock) as runner:
            response = self.client.post("/api/v1/evaluation/benchmark/reports/r/resume")
        self.assertEqual(response.status_code, 429)
        self.assertGreater(int(response.headers["Retry-After"]), 0)
        runner.assert_not_awaited()
        self.assertEqual(repo.find_benchmark_report.await_args.args, ("alice", "r"))

    def test_resume_blocks_changed_paper_and_rejects_concurrent_request(self):
        from fastapi import HTTPException
        from app.services.evaluation_benchmark import BenchmarkCase, EvidenceRef, FrozenBenchmark
        case = BenchmarkCase(id="one", paper_id="p", paper_hash="hash", question="What?", expected_behavior="answer",
            expected_answer="gold", evidence=(EvidenceRef(page=1, chunk_index=0, content_hash="chunk", quote="gold"),),
            review_status="approved", reviewed_by="alice")
        benchmark = FrozenBenchmark(version="v1", cases=(case,))
        report = {"_id": "r", "created_at": datetime.now(timezone.utc),
            "metrics": {"benchmark_version": "v1", "benchmark_hash": benchmark.fingerprint()},
            "results": [{"case_id": "one", "execution_status": "not_run"}], "judgments": []}
        repo = self.client.app.state.evaluation_repository
        repo.find_benchmark_report = AsyncMock(return_value=report)
        repo.acquire_resume = AsyncMock(return_value=report)
        repo.release_resume = AsyncMock()
        repo.save_resume = AsyncMock(return_value=True)
        repo.find_benchmark_version = AsyncMock(return_value={"benchmark": benchmark.model_dump(mode="json"),
                                                               "fingerprint": benchmark.fingerprint()})
        with patch("app.routers.evaluation._validate_benchmark", new_callable=AsyncMock, side_effect=HTTPException(409)), \
             patch("app.routers.evaluation.run_benchmark", new_callable=AsyncMock) as runner:
            response = self.client.post("/api/v1/evaluation/benchmark/reports/r/resume")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["execution_status"], "blocked")
        runner.assert_not_awaited()
        repo.acquire_resume.return_value = None
        self.assertEqual(self.client.post("/api/v1/evaluation/benchmark/reports/r/resume").status_code, 409)
