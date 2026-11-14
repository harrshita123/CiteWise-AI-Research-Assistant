from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..dependencies import require_admin
from ..models import UserResponse
from ..repositories.evaluations import EvaluationRepository
from ..repositories.papers import PaperRepository
from ..services.ingestion import IngestionService

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


class EvaluationCase(BaseModel):
    paper_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1000)
    expected_page: int = Field(ge=1)


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1, max_length=50)


class CaseResult(EvaluationCase):
    returned_pages: list[int]
    hit_at_5: bool


class EvaluationRun(BaseModel):
    id: str
    created_at: datetime
    case_count: int
    hit_at_5: float
    results: list[CaseResult]


def _run_response(document: dict) -> EvaluationRun:
    return EvaluationRun(id=document["_id"], created_at=document["created_at"],
                         case_count=document["case_count"],
                         hit_at_5=document["hit_at_5"], results=document["results"])


@router.get("/runs", response_model=list[EvaluationRun])
async def list_runs(request: Request, user: UserResponse = Depends(require_admin)):
    repository: EvaluationRepository = request.app.state.evaluation_repository
    return [_run_response(item) for item in await repository.list_for_user(user.id)]


@router.post("/runs", response_model=EvaluationRun, status_code=201)
async def create_run(payload: EvaluationRequest, request: Request,
                     user: UserResponse = Depends(require_admin)):
    papers: PaperRepository = request.app.state.paper_repository
    page_counts: dict[str, int] = {}
    for case in payload.cases:
        if not case.question.strip():
            raise HTTPException(422, "Each case needs a question.")
        if case.paper_id not in page_counts:
            paper = await papers.find(user.id, case.paper_id)
            if paper is None:
                raise HTTPException(404, "Paper not found.")
            if paper["status"] != "ready":
                raise HTTPException(409, "All evaluation papers must be ready.")
            page_counts[case.paper_id] = paper["page_count"]
        if case.expected_page > page_counts[case.paper_id]:
            raise HTTPException(422, "Expected page exceeds the paper's page count.")

    ingestion: IngestionService = request.app.state.ingestion_service

    def evaluate() -> list[dict]:
        questions = [case.question.strip() for case in payload.cases]
        embeddings = []
        for start in range(0, len(questions), 32):
            embeddings.extend(ingestion.embeddings.embed(questions[start:start + 32]))
        results = []
        for case, question, embedding in zip(payload.cases, questions, embeddings):
            passages = ingestion.vectors.search(user.id, case.paper_id, embedding, 5, question=question)
            pages = [passage.page_number for passage in passages]
            results.append({
                "paper_id": case.paper_id,
                "question": question,
                "expected_page": case.expected_page,
                "returned_pages": pages,
                "hit_at_5": case.expected_page in pages,
            })
        return results

    results = await run_in_threadpool(evaluate)
    count = len(results)
    report = {
        "case_count": count,
        "hit_at_5": sum(item["hit_at_5"] for item in results) / count,
        "results": results,
    }
    repository: EvaluationRepository = request.app.state.evaluation_repository
    return _run_response(await repository.create(user.id, report))
