from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from uuid import uuid4
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError
from starlette.concurrency import run_in_threadpool

from ..dependencies import require_admin
from ..models import UserResponse
from ..repositories.evaluations import EvaluationRepository
from ..repositories.papers import PaperRepository
from ..services.ingestion import IngestionService
from ..services.evaluation_benchmark import BenchmarkCase, FrozenBenchmark, CaseObservation, RetrievedChunk, TAXONOMY, SKILLS, evaluate_benchmark
from ..services.evaluation_runner import run_benchmark
from ..services.evaluation_proposals import generate_proposals
from ..providers.answers import AnswerRateLimitError

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
    reciprocal_rank: float | None = None


class EvaluationRun(BaseModel):
    id: str
    created_at: datetime
    case_count: int
    hit_at_5: float
    mean_reciprocal_rank: float | None = None
    results: list[CaseResult]


class BenchmarkScoreRequest(BaseModel):
    benchmark: FrozenBenchmark
    observations: list[CaseObservation] = Field(max_length=50)


class CaseReview(BaseModel):
    revision: int = Field(ge=0)
    status: str
    case: BenchmarkCase


class ProposalFreeze(BaseModel):
    version: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._-]+$")
    proposal_ids: list[str] = Field(min_length=1, max_length=50)


class CaseJudgment(BaseModel):
    case_id: str
    answer_correct: bool | None = None
    citations_supported: int | None = Field(default=None, ge=0)
    citations_total: int | None = Field(default=None, ge=0)
    claims_cited: int | None = Field(default=None, ge=0)
    claims_requiring_citation: int | None = Field(default=None, ge=0)
    refused: bool | None = None


class ReportJudgments(BaseModel):
    judgments: list[CaseJudgment] = Field(max_length=50)


@router.get("/taxonomy")
async def research_taxonomy(user: UserResponse = Depends(require_admin)):
    return {"axes": TAXONOMY, "skills": SKILLS}


@router.get("/answer-quality")
async def list_answer_quality(request: Request, response: Response, user: UserResponse = Depends(require_admin),
                              paper_id: str | None = Query(None, max_length=128),
                              date_from: date | None = None, date_to: date | None = None,
                              evidence: Literal["pdf", "external", "mixed", "none"] | None = None,
                              state: Literal["COMPLETE", "PENDING", "FAILED", "UNAVAILABLE"] | None = None,
                              priority: Literal["HIGH", "MEDIUM"] | None = None,
                              limit: int = Query(25, ge=1, le=50), offset: int = Query(0, ge=0)):
    papers = request.app.state.paper_repository
    if paper_id and not await papers.find(user.id, paper_id):
        raise HTTPException(404, "Paper not found.")
    if date_from and date_to and date_from > date_to or date_to == date.max:
        raise HTTPException(422, "Choose a valid date range.")
    result = await request.app.state.conversation_repository.quality_monitor(user.id, paper_id=paper_id,
        start=datetime.combine(date_from, time.min, timezone.utc) if date_from else None,
        end=datetime.combine(date_to + timedelta(days=1), time.min, timezone.utc) if date_to else None,
        evidence=evidence, state=state, priority=priority, limit=limit, offset=offset)
    for row in result["rows"]:
        row["evaluated_at"] = _utc(row["evaluated_at"])
    response.headers["Cache-Control"] = "private, no-store"
    return result


@router.get("/answer-quality/{conversation_id}/{message_id}/{revision}")
async def get_answer_quality(conversation_id: str, message_id: str, revision: int, request: Request, response: Response,
                             user: UserResponse = Depends(require_admin)):
    item = await request.app.state.conversation_repository.quality_detail(user.id, conversation_id, message_id, revision)
    if item is None:
        raise HTTPException(404, "Answer evaluation not found.")
    item["evaluated_at"] = _utc(item["evaluated_at"])
    for metric in item["metrics"].values():
        metric["evaluated_at"] = _utc(metric["evaluated_at"])
    response.headers["Cache-Control"] = "private, no-store"
    return item


def _proposal_response(document):
    return {"id": document["_id"], "paper_id": document["paper_id"],
            "revision": document["revision"], "cases": document["cases"],
            "created_at": document["created_at"]}


@router.get("/papers/{paper_id}/proposals")
async def list_proposals(paper_id: str, request: Request, user: UserResponse = Depends(require_admin)):
    if await request.app.state.paper_repository.find(user.id, paper_id) is None:
        raise HTTPException(404, "Paper not found.")
    rows = await request.app.state.evaluation_repository.list_proposals(user.id, paper_id)
    return [_proposal_response(row) for row in rows]


@router.post("/papers/{paper_id}/proposals", status_code=201)
async def propose_cases(paper_id: str, request: Request, user: UserResponse = Depends(require_admin)):
    paper = await request.app.state.paper_repository.find(user.id, paper_id)
    if paper is None:
        raise HTTPException(404, "Paper not found.")
    if paper.get("status") != "ready":
        raise HTTPException(409, "Wait until this paper is ready.")
    provider = getattr(request.app.state, "answer_provider", None)
    if not provider or not callable(getattr(provider, "propose_evaluation_cases", None)):
        raise HTTPException(503, "Evaluation generation is unavailable.")
    ingestion = request.app.state.ingestion_service
    chunks = await run_in_threadpool(ingestion.vectors.sample, user.id, paper_id, paper.get("chunk_count", 0), 16)
    if not chunks or not paper.get("file_hash"):
        raise HTTPException(409, "This paper has no indexed content or content hash.")
    try:
        cases = await run_in_threadpool(generate_proposals, paper_id, paper["file_hash"], chunks, provider)
    except AnswerRateLimitError as exc:
        raise HTTPException(429, "Evaluation generation is rate limited.",
                            headers={"Retry-After": exc.retry_after} if exc.retry_after else None) from None
    except Exception:
        raise HTTPException(503, "Evaluation generation is temporarily unavailable.") from None
    document = await request.app.state.evaluation_repository.create_proposals(user.id, paper_id, cases)
    return _proposal_response(document)


@router.put("/proposals/{proposal_id}/cases/{case_id}")
async def review_case(proposal_id: str, case_id: str, payload: CaseReview, request: Request,
                      user: UserResponse = Depends(require_admin)):
    repository = request.app.state.evaluation_repository
    document = await repository.find_proposals(user.id, proposal_id)
    if document is None:
        raise HTTPException(404, "Proposal not found.")
    if payload.revision != document["revision"]:
        raise HTTPException(409, "The proposals changed; reload before reviewing.")
    existing = next((item for item in document["cases"] if item["case"]["id"] == case_id), None)
    if existing is None:
        raise HTTPException(404, "Case not found.")
    if payload.status not in ("approved", "rejected", "pending"):
        raise HTTPException(422, "Invalid review status.")
    case = payload.case.model_dump(mode="json")
    if (case["id"] != case_id or case["paper_id"] != document["paper_id"]
            or case["paper_hash"] != existing["case"]["paper_hash"]):
        raise HTTPException(422, "The case paper identity cannot be changed.")
    case["review_status"] = "approved" if payload.status == "approved" else "pending"
    case["reviewed_by"] = user.id if payload.status == "approved" else None
    if payload.status == "approved":
        for label in case["labels"]:
            label["needs_review"] = False
        try:
            reviewed = BenchmarkCase.model_validate(case)
            await _validate_benchmark(FrozenBenchmark(version="review", cases=(reviewed,)), request, user)
        except ValueError:
            raise HTTPException(422, "The reviewed case or its evidence is invalid.") from None
    updated = [{"status": payload.status, "case": case} if item["case"]["id"] == case_id else item
               for item in document["cases"]]
    if not await repository.replace_proposals(user.id, proposal_id, payload.revision, updated):
        raise HTTPException(409, "The proposals changed; reload before reviewing.")
    document.update(cases=updated, revision=payload.revision + 1)
    return _proposal_response(document)


@router.post("/proposals/freeze", status_code=201)
async def freeze_proposals(payload: ProposalFreeze, request: Request, user: UserResponse = Depends(require_admin)):
    cases = []
    for proposal_id in dict.fromkeys(payload.proposal_ids):
        document = await request.app.state.evaluation_repository.find_proposals(user.id, proposal_id)
        if document is None:
            raise HTTPException(404, "Proposal not found.")
        cases.extend(BenchmarkCase.model_validate(item["case"]) for item in document["cases"]
                     if item["status"] == "approved" and item["case"].get("reviewed_by") == user.id)
    if not cases or len(cases) > 50:
        raise HTTPException(422, "Choose between 1 and 50 approved cases.")
    return await freeze_benchmark(FrozenBenchmark(version=payload.version, cases=tuple(cases)), request, user)


@router.post("/benchmark/score")
async def score_reviewed_benchmark(payload: BenchmarkScoreRequest, request: Request,
                                   user: UserResponse = Depends(require_admin)):
    """Score a frozen, human-labelled snapshot without adding gold data to Chroma."""
    if len(payload.benchmark.cases) > 50:
        raise HTTPException(422, "A benchmark can contain at most 50 cases.")
    if {case.id for case in payload.benchmark.cases} != {row.case_id for row in payload.observations}:
        raise HTTPException(422, "Provide exactly one observation for every benchmark case.")
    await _validate_benchmark(payload.benchmark, request, user)
    try:
        return evaluate_benchmark(payload.benchmark, payload.observations)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


async def _validate_benchmark(benchmark: FrozenBenchmark, request: Request, user: UserResponse):
    if len(benchmark.cases) > 50:
        raise HTTPException(422, "A benchmark can contain at most 50 cases.")
    papers: PaperRepository = request.app.state.paper_repository
    ingestion: IngestionService = request.app.state.ingestion_service
    checked = {}
    for case in benchmark.cases:
        if case.paper_id not in checked:
            paper = await papers.find(user.id, case.paper_id)
            if paper is None:
                raise HTTPException(404, "Paper not found.")
            if paper.get("status") != "ready":
                raise HTTPException(409, "All benchmark papers must be ready.")
            checked[case.paper_id] = paper
        if checked[case.paper_id].get("file_hash") != case.paper_hash:
            raise HTTPException(409, "A benchmark paper has changed; review the benchmark again.")
        for reference in (*case.evidence, *(ref for label in case.labels for ref in label.evidence)):
            chunks = await run_in_threadpool(ingestion.preview, user.id, case.paper_id, reference.chunk_index, 1)
            if (not chunks or chunks[0].chunk_index != reference.chunk_index
                    or chunks[0].page_number != reference.page or chunks[0].content_hash != reference.content_hash
                    or reference.quote not in chunks[0].text):
                raise HTTPException(422, "Benchmark evidence does not match the indexed paper.")


@router.post("/benchmark/run", status_code=201)
async def execute_reviewed_benchmark(benchmark: FrozenBenchmark, request: Request,
                                     user: UserResponse = Depends(require_admin)):
    await _validate_benchmark(benchmark, request, user)
    report = await run_benchmark(benchmark, request, user)
    saved = await request.app.state.evaluation_repository.create_benchmark_report(user.id, report)
    return {"id": saved["_id"], "created_at": saved["created_at"], **report}


@router.post("/benchmark/versions", status_code=201)
async def freeze_benchmark(benchmark: FrozenBenchmark, request: Request,
                           user: UserResponse = Depends(require_admin)):
    """Freeze only cases explicitly approved by the authenticated reviewer."""
    if any(case.reviewed_by != user.id for case in benchmark.cases):
        raise HTTPException(422, "All cases must be approved by the current reviewer.")
    await _validate_benchmark(benchmark, request, user)
    repository: EvaluationRepository = request.app.state.evaluation_repository
    try:
        await repository.create_benchmark_version(user.id, benchmark.model_dump(mode="json"), benchmark.fingerprint())
    except DuplicateKeyError:
        raise HTTPException(409, "This benchmark version is already frozen.") from None
    return {"version": benchmark.version, "fingerprint": benchmark.fingerprint()}


@router.post("/benchmark/versions/{version}/run", status_code=201)
async def execute_frozen_benchmark(version: str, request: Request,
                                   user: UserResponse = Depends(require_admin)):
    repository: EvaluationRepository = request.app.state.evaluation_repository
    document = await repository.find_benchmark_version(user.id, version)
    if document is None:
        raise HTTPException(404, "Benchmark version not found.")
    benchmark = FrozenBenchmark.model_validate(document["benchmark"])
    if benchmark.fingerprint() != document["fingerprint"]:
        raise HTTPException(409, "The frozen benchmark has changed.")
    return await execute_reviewed_benchmark(benchmark, request, user)


@router.get("/benchmark/reports")
async def list_benchmark_reports(request: Request, user: UserResponse = Depends(require_admin)):
    documents = await request.app.state.evaluation_repository.list_benchmark_reports(user.id)
    return [{"id": doc["_id"], "created_at": doc["created_at"],
             "version": doc["metrics"]["benchmark_version"],
             "completed": sum(item["execution_status"] == "completed" for item in doc["results"]),
             "total": len(doc["results"])} for doc in documents]


@router.get("/benchmark/reports/{report_id}")
async def get_benchmark_report(report_id: str, request: Request,
                               user: UserResponse = Depends(require_admin)):
    report = await request.app.state.evaluation_repository.find_benchmark_report(user.id, report_id)
    if report is None:
        raise HTTPException(404, "Benchmark report not found.")
    return {"id": report["_id"], "created_at": report["created_at"],
            "metrics": report["metrics"], "results": report["results"], "judgments": report.get("judgments", []),
            "cooldown_until": report.get("cooldown_until")}


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _cooldown_deadline(result, timestamp):
    raw = result.get("retry_after_seconds") or result.get("retry_after")
    if not raw and result.get("execution_status") != "rate_limited":
        return None
    try:
        return _utc(timestamp) + timedelta(seconds=max(0, int(raw or 60)))
    except (ValueError, TypeError, OverflowError):
        try:
            return _utc(parsedate_to_datetime(raw))
        except (ValueError, TypeError, OverflowError):
            return _utc(timestamp) + timedelta(seconds=60)


def _score_results(benchmark, results, judgments):
    rows = {item["case_id"]: item for item in judgments}
    observations = []
    for result in results:
        fields = {k: v for k, v in rows.get(result["case_id"], {}).items() if k != "case_id"}
        observations.append(CaseObservation(case_id=result["case_id"], execution_status=result["execution_status"],
            returned_chunks=tuple(RetrievedChunk.model_validate(item) for item in result.get("retrieved_chunks", [])),
            latency_ms=result.get("latency_ms"), **fields))
    return evaluate_benchmark(benchmark, observations)


@router.post("/benchmark/reports/{report_id}/resume")
async def resume_benchmark_report(report_id: str, request: Request, user: UserResponse = Depends(require_admin)):
    """Resume at most one case; a caller may make another request after it is saved."""
    repository = request.app.state.evaluation_repository
    existing = await repository.find_benchmark_report(user.id, report_id)
    if existing is None:
        raise HTTPException(404, "Benchmark report not found.")
    token = uuid4().hex
    now = datetime.now(timezone.utc)
    report = await repository.acquire_resume(user.id, report_id, token, now, now + timedelta(minutes=15))
    if report is None:
        raise HTTPException(409, "This evaluation is already being resumed.")
    try:
        snapshot = await repository.find_benchmark_version(user.id, report["metrics"]["benchmark_version"])
        if snapshot is None:
            raise HTTPException(409, "The frozen benchmark is unavailable.")
        benchmark = FrozenBenchmark.model_validate(snapshot["benchmark"])
        if benchmark.fingerprint() != snapshot["fingerprint"] or benchmark.fingerprint() != report["metrics"]["benchmark_hash"]:
            raise HTTPException(409, "Benchmark identity does not match this report.")
        deadline = report.get("cooldown_until")
        if deadline is None:
            deadlines = [_cooldown_deadline(result, report["created_at"]) for result in report["results"]]
            deadline = max((d for d in deadlines if d is not None), default=None)
        if deadline and _utc(deadline) > now:
            seconds = max(1, int((_utc(deadline) - now).total_seconds() + 0.999))
            raise HTTPException(429, "The provider cooldown is still active.", headers={"Retry-After": str(seconds)})
        cases = {case.id: case for case in benchmark.cases}
        eligible = next((item for item in report["results"] if item["execution_status"] in ("rate_limited", "not_run")), None)
        if eligible is None:
            return {"id": report_id, "metrics": report["metrics"], "results": report["results"],
                    "judgments": report.get("judgments", []), "resumed": False}
        case = cases.get(eligible["case_id"])
        if case is None:
            raise HTTPException(409, "Report contains a case outside the frozen benchmark.")
        try:
            await _validate_benchmark(FrozenBenchmark(version=benchmark.version, cases=(case,)), request, user)
        except HTTPException as exc:
            if exc.status_code not in (404, 409, 422):
                raise
            replacement = {"case_id": case.id, "paper_id": case.paper_id, "execution_status": "blocked",
                           "reason": "The paper or its reviewed evidence is unavailable or changed."}
        else:
            partial = await run_benchmark(FrozenBenchmark(version=benchmark.version, cases=(case,)), request, user)
            replacement = partial["results"][0]
        results = [replacement if item["case_id"] == case.id else item for item in report["results"]]
        # Fetch judgments again: a review can have been saved during the provider call.
        latest = await repository.find_benchmark_report(user.id, report_id)
        judgments = latest.get("judgments", [])
        metrics = _score_results(benchmark, results, judgments)
        cooldown = _cooldown_deadline(replacement, datetime.now(timezone.utc))
        if not await repository.save_resume(user.id, report_id, token, results, metrics, cooldown):
            raise HTTPException(409, "Resume lease expired; reload the report.")
        return {"id": report_id, "metrics": metrics, "results": results, "judgments": judgments,
                "cooldown_until": cooldown, "resumed": True, "last_status": replacement["execution_status"]}
    finally:
        await repository.release_resume(user.id, report_id, token)


@router.put("/benchmark/reports/{report_id}/judgments")
async def judge_benchmark_report(report_id: str, payload: ReportJudgments, request: Request,
                                 user: UserResponse = Depends(require_admin)):
    repository = request.app.state.evaluation_repository
    report = await repository.find_benchmark_report(user.id, report_id)
    if report is None:
        raise HTTPException(404, "Benchmark report not found.")
    version = report["metrics"]["benchmark_version"]
    snapshot = await repository.find_benchmark_version(user.id, version)
    if snapshot is None:
        raise HTTPException(409, "A frozen benchmark is required for review.")
    benchmark = FrozenBenchmark.model_validate(snapshot["benchmark"])
    if benchmark.fingerprint() != report["metrics"]["benchmark_hash"] or benchmark.fingerprint() != snapshot["fingerprint"]:
        raise HTTPException(409, "Benchmark identity does not match this report.")
    submitted = {item.case_id: item for item in payload.judgments}
    if len(submitted) != len(payload.judgments):
        raise HTTPException(422, "Duplicate judgment.")
    results = {item["case_id"]: item for item in report["results"]}
    if any(case_id not in results or results[case_id]["execution_status"] != "completed" for case_id in submitted):
        raise HTTPException(422, "Only completed cases can be reviewed.")
    judgments = {item["case_id"]: CaseJudgment.model_validate(item) for item in report.get("judgments", [])}
    judgments.update(submitted)
    for item in submitted.values():
        for numerator, denominator, label in ((item.citations_supported, item.citations_total, "Supported citations and total citations"),
                                              (item.claims_cited, item.claims_requiring_citation, "Claims with citations and claims needing citations")):
            if (numerator is None) != (denominator is None):
                raise HTTPException(422, f"{label} must both be entered.")
            if numerator is not None and numerator > denominator:
                raise HTTPException(422, f"{label}: the first count cannot exceed the second.")
    observations = []
    try:
        for result in report["results"]:
            judgment = judgments.get(result["case_id"])
            fields = judgment.model_dump(exclude={"case_id"}) if judgment else {}
            observations.append(CaseObservation(case_id=result["case_id"],
                execution_status=result["execution_status"],
                returned_chunks=tuple(RetrievedChunk.model_validate(chunk) for chunk in result.get("retrieved_chunks", [])),
                latency_ms=result.get("latency_ms"), **fields))
        metrics = evaluate_benchmark(benchmark, observations)
    except ValueError:
        raise HTTPException(422, "Invalid judgment counts.") from None
    saved = [item.model_dump(mode="json") for item in judgments.values()]
    if not await repository.update_benchmark_report(user.id, report_id, metrics, saved):
        raise HTTPException(404, "Benchmark report not found.")
    return {"id": report_id, "metrics": metrics, "results": report["results"], "judgments": saved}


def _run_response(document: dict) -> EvaluationRun:
    return EvaluationRun(id=document["_id"], created_at=document["created_at"],
                         case_count=document["case_count"],
                         hit_at_5=document["hit_at_5"],
                         mean_reciprocal_rank=document.get("mean_reciprocal_rank"),
                         results=document["results"])


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
            rank = next((index for index, page in enumerate(pages, 1)
                         if page == case.expected_page), None)
            results.append({
                "paper_id": case.paper_id,
                "question": question,
                "expected_page": case.expected_page,
                "returned_pages": pages,
                "hit_at_5": case.expected_page in pages,
                "reciprocal_rank": 1 / rank if rank else 0.0,
            })
        return results

    results = await run_in_threadpool(evaluate)
    count = len(results)
    report = {
        "case_count": count,
        "hit_at_5": sum(item["hit_at_5"] for item in results) / count,
        "mean_reciprocal_rank": sum(item["reciprocal_rank"] for item in results) / count,
        "results": results,
    }
    repository: EvaluationRepository = request.app.state.evaluation_repository
    return _run_response(await repository.create(user.id, report))
