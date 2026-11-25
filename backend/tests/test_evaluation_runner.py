import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, Request

from app.services.evaluation_benchmark import BenchmarkCase, EvidenceRef, FrozenBenchmark
from app.services.evaluation_runner import run_benchmark


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        evidence = (EvidenceRef(page=1, chunk_index=0, content_hash="hash", quote="finding"),)
        cases = tuple(BenchmarkCase(id=str(i), paper_id=f"paper{i}", paper_hash=f"hash{i}",
            question="What is the finding?", expected_behavior="answer", expected_answer="finding",
            evidence=evidence, review_status="approved", reviewed_by="human") for i in range(2))
        self.benchmark = FrozenBenchmark(version="v1", cases=cases)
        state = SimpleNamespace(_state={"conversation_repository": object()})
        self.request = Request({"type": "http", "method": "POST", "path": "/", "headers": [],
                                "app": SimpleNamespace(state=state)})
        self.user = SimpleNamespace(id="owner")

    async def test_uses_actual_pipeline_and_isolates_conversations(self):
        original = self.request.app.state._state["conversation_repository"]
        async def answer(payload, request, response, user):
            self.assertIsNot(request.app.state.conversation_repository, original)
            self.assertEqual(user.id, "owner")
            request.state.evaluation_passages = [SimpleNamespace(page_number=1, chunk_index=0, content_hash="hash")]
            return SimpleNamespace(answer="finding", status="supported", sources=[], verification_status="verified")
        with patch("app.services.evaluation_runner.ask_paper", side_effect=answer) as call:
            report = await run_benchmark(self.benchmark, self.request, self.user)
        self.assertEqual(call.await_count, 2)
        self.assertEqual(report["metrics"]["overall"]["hit_at_5"], 1)
        self.assertIsNone(report["metrics"]["overall"]["answer_correctness"])
        self.assertEqual(self.request.app.state._state["conversation_repository"], original)

    async def test_rate_limit_stops_further_provider_calls(self):
        with patch("app.services.evaluation_runner.ask_paper", new_callable=AsyncMock,
                   side_effect=HTTPException(429, headers={"Retry-After": "60"})) as call:
            report = await run_benchmark(self.benchmark, self.request, self.user)
        self.assertEqual(call.await_count, 1)
        self.assertEqual([r["execution_status"] for r in report["results"]], ["rate_limited", "not_run"])
        self.assertEqual(report["results"][0]["retry_after"], "60")
        self.assertIsNone(report["metrics"]["overall"]["hit_at_5"])
