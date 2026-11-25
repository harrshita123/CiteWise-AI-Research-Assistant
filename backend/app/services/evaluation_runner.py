"""Run reviewed cases through the ordinary answer endpoint without chat writes."""

import logging
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4

from fastapi import HTTPException, Request, Response

from ..routers.research import AskRequest, ask_paper
from .evaluation_benchmark import CaseObservation, FrozenBenchmark, RetrievedChunk, evaluate_benchmark

logger = logging.getLogger(__name__)


class _EphemeralConversations:
    async def create(self, owner_id, paper_id, message, filename=None):
        return {"_id": uuid4().hex}


async def run_benchmark(benchmark: FrozenBenchmark, request: Request, user) -> dict:
    """Run cases independently. Stop calling the provider after a rate limit."""
    state = dict(request.app.state._state)
    state["conversation_repository"] = _EphemeralConversations()
    observations = []
    results = []
    rate_limited = False
    for case in benchmark.cases:
        if rate_limited:
            observations.append(CaseObservation(case_id=case.id, execution_status="not_run"))
            results.append({"case_id": case.id, "execution_status": "not_run"})
            continue
        scope = dict(request.scope)
        scope.update(app=SimpleNamespace(state=SimpleNamespace(**state)),
                     citewise_evaluation=True, state={})
        evaluation_request = Request(scope)
        start = monotonic()
        try:
            answer = await ask_paper(AskRequest(paper_id=case.paper_id, question=case.question),
                                     evaluation_request, Response(), user)
            # Verification can report provider throttling in a successful HTTP response.
            rate_limited = bool(getattr(answer, "retry_after_seconds", 0))
            elapsed = (monotonic() - start) * 1000
            passages = getattr(evaluation_request.state, "evaluation_passages", [])
            observations.append(CaseObservation(case_id=case.id,
                returned_pages=tuple(p.page_number for p in passages),
                returned_chunks=tuple(RetrievedChunk(page=p.page_number, chunk_index=p.chunk_index,
                                                     content_hash=p.content_hash) for p in passages),
                latency_ms=elapsed))
            results.append({"case_id": case.id, "paper_id": case.paper_id,
                "paper_hash": case.paper_hash, "execution_status": "completed",
                "question": case.question, "reference_answer": case.expected_answer,
                "expected_behavior": case.expected_behavior,
                "retrieved_chunks": [{"page": p.page_number, "chunk_index": p.chunk_index,
                                      "content_hash": p.content_hash} for p in passages],
                "answer": answer.answer, "status": answer.status,
                "citations": [s.model_dump(mode="json") for s in answer.sources],
                "verification_status": answer.verification_status,
                "retry_after_seconds": getattr(answer, "retry_after_seconds", 0), "latency_ms": elapsed})
        except HTTPException as exc:
            status = "rate_limited" if exc.status_code == 429 else (
                "provider_error" if exc.status_code in (502, 503, 504) else "internal_error")
            rate_limited = status == "rate_limited"
            observations.append(CaseObservation(case_id=case.id, execution_status=status))
            results.append({"case_id": case.id, "execution_status": status,
                            "http_status": exc.status_code,
                            "retry_after": (exc.headers or {}).get("Retry-After")})
        except Exception as exc:
            logger.error("Benchmark case failed: case_id=%s error_type=%s", case.id, type(exc).__name__)
            observations.append(CaseObservation(case_id=case.id, execution_status="internal_error"))
            results.append({"case_id": case.id, "execution_status": "internal_error"})
    return {"metrics": evaluate_benchmark(benchmark, observations), "results": results}
