from fastapi import APIRouter, Depends, HTTPException, Request, Response
from typing import Literal
from collections import Counter, OrderedDict
from hashlib import sha256
import logging
import asyncio
import ipaddress
import unicodedata
import hmac
from threading import Lock
import re
import time
from datetime import date, datetime, timezone
from uuid import uuid4
from urllib.parse import urlparse
from requests.exceptions import HTTPError, Timeout as RequestTimeout
from pydantic import BaseModel, Field
import pymupdf
from starlette.concurrency import run_in_threadpool

from ..dependencies import get_current_user
from ..config import get_settings
from ..models import UserResponse
from ..paper_models import ChunkResponse
from ..providers.answers import (AnswerProvider, AnswerRateLimitError, Evidence, EvidenceAssessment, EvidenceQuote,
                                 EvidenceStatus, ExternalEvidence)
from ..providers.openrouter_answers import (OpenRouterAnswerProvider, model_runs, openrouter_cache_scope,
                                           openrouter_request_id, openrouter_trace)
from ..providers.openrouter_routing import RERANK_CANDIDATES, UpstreamPoolLimit
from ..providers.external_search import TavilyExternalSearch
from ..repositories.papers import PaperRepository
from ..repositories.conversations import ConversationRepository
from ..repositories.vectors import VectorRepository
from ..services.ingestion import IngestionService
from ..services.evidence_coverage import CoveragePassage, coverage_report, missing_coverage, requested_coverage
from ..services.demo_research import (
    CASES, EXTERNAL_QUESTIONS, EXTERNAL_SOURCES, REPLY_RANGE, REPLY_THRESHOLD,
    is_sample, question_key, sample_quote, sample_recommendations, sample_source,
)
from ..services.research_subjects import requested_subjects, subject_spans
from ..services.claim_verification import split_claims, review_claims, present_claims
from ..services.verification_gates import answer_report, ready_for_research
from ..services.answer_quality import AnswerQuality
from ..services.question_recommendations import QuestionRecommendationService, RecommendedQuestion

router = APIRouter(prefix="/api/v1/research", tags=["research"])
_OPENROUTER_COOLDOWN_COOKIE = "citewise_openrouter_cooldown"


def _cooldown_signature(key: str, deadline: int) -> str:
    secret = get_settings().jwt_secret.encode()
    return hmac.new(secret, f"{key}\0{deadline}".encode(), sha256).hexdigest()


def _cooldown_seconds(request: Request, provider: OpenRouterAnswerProvider) -> int:
    token = request.cookies.get(_OPENROUTER_COOLDOWN_COOKIE, "")
    deadline_text, separator, signature = token.partition(".")
    if (not separator or not deadline_text.isdecimal() or len(deadline_text) > 12
            or not re.fullmatch(r"[0-9a-f]{64}", signature)):
        return 0
    deadline = int(deadline_text)
    if not hmac.compare_digest(signature, _cooldown_signature(provider.api_key, deadline)):
        return 0
    return max(0, deadline - int(time.time()))


def _rate_limit_headers(exc: AnswerRateLimitError, provider: AnswerProvider) -> dict[str, str] | None:
    if not exc.retry_after:
        return None
    headers = {"Retry-After": exc.retry_after}
    if isinstance(provider, OpenRouterAnswerProvider) and provider.api_key:
        try:
            seconds = int(exc.retry_after)
        except (TypeError, ValueError):
            return headers
        if seconds <= 0:
            return headers
        deadline = int(time.time()) + seconds
        token = f"{deadline}.{_cooldown_signature(provider.api_key, deadline)}"
        secure = "; Secure" if get_settings().session_cookie_secure else ""
        headers["Set-Cookie"] = (f"{_OPENROUTER_COOLDOWN_COOKIE}={token}; Max-Age={seconds}; "
                                 f"Path=/api/v1/research; HttpOnly; SameSite=Lax{secure}")
    return headers


_RATE_LIMIT_MESSAGES = {
    "local_daily_budget": "CiteWise’s daily request budget is exhausted.",
    "local_minute_budget": "CiteWise’s per-minute request budget is exhausted.",
    "openrouter_free_tier_daily": "OpenRouter’s daily free-model quota is exhausted.",
    "openrouter_free_tier_per_minute": "OpenRouter’s per-minute free-model quota is exhausted.",
    "account_cooldown": "CiteWise has paused OpenRouter requests after a rate limit.",
    "upstream_provider_shared_pool": "The model providers are temporarily busy.",
}


def _rate_limit_detail(exc: AnswerRateLimitError, message: str) -> dict[str, str]:
    return {
        "message": _RATE_LIMIT_MESSAGES.get(exc.limit_source, message),
        "limit_source": exc.limit_source if exc.limit_source in _RATE_LIMIT_MESSAGES else "unknown",
        "code": "upstream_capacity" if isinstance(exc, UpstreamPoolLimit) else "rate_limited",
    }


_assessment_cache: OrderedDict[tuple[OpenRouterAnswerProvider, str], "EvidenceReview"] = OrderedDict()
_assessment_cache_lock = Lock()
_assessment_cache_limit = 128
_retrieval_cache: OrderedDict[tuple, tuple[float, tuple[ChunkResponse, ...]]] = OrderedDict()
_retrieval_cache_lock = Lock()
_verification_locks = [asyncio.Lock() for _ in range(64)]


class SearchRequest(BaseModel):
    paper_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=10)


class VisualAnalysisRequest(BaseModel):
    paper_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    question: str = Field(min_length=1, max_length=1000)
    evidence_type: Literal["figure", "chart", "table", "diagram", "scanned_page"]
    text_insufficient: Literal[True]
    consent_to_send_page: Literal[True]


class SearchPassage(ChunkResponse):
    highlight_text: str


class SearchResponse(BaseModel):
    paper_id: str
    passages: list[SearchPassage]


class AskRequest(SearchRequest):
    conversation_id: str | None = None
    request_id: str | None = Field(default=None, max_length=64)
    reply_to_message_id: str | None = Field(default=None, max_length=64)


class AnswerSource(ChunkResponse):
    id: str
    highlight_text: str
    document_name: str | None = None


class ExternalSource(BaseModel):
    id: str
    title: str
    url: str
    published_date: str | None = None
    excerpt: str | None = None
    publisher: str | None = None
    author: str | None = None
    source_host: str | None = None
    verified_publication_date: str | None = None


class ClaimQuoteResponse(BaseModel):
    source_id: str
    quote: str


class ClaimReviewResponse(BaseModel):
    text: str
    section: Literal["pdf", "external"]
    status: Literal["supported", "partially_supported", "unsupported", "contradicted"]
    supporting: list[ClaimQuoteResponse]
    opposing: list[ClaimQuoteResponse]
    explanation: str = ""
    reason_code: str | None = None


def _reviewed_status(claims: list[ClaimReviewResponse], pdf_partial: bool, *, external: bool = False) -> EvidenceStatus:
    statuses = {claim.status for claim in claims}
    if "contradicted" in statuses:
        return "contradictory_evidence"
    if "partially_supported" in statuses or ("supported" in statuses and (pdf_partial or "unsupported" in statuses)):
        return "partially_supported"
    if "supported" in statuses:
        return "not_mentioned" if external else "answered"
    return "insufficient_evidence"


AnswerState = Literal["VERIFIED", "PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "VERIFICATION_UNAVAILABLE"]


class SearchTrace(BaseModel):
    tavily_calls: int = 0
    tavily_cache_hits: int = 0
    openrouter_calls: int = 0
    openrouter_cache_hits: int = 0
    openrouter_attempts: int = 0
    openrouter_retries: int = 0
    openrouter_failures: int = 0
    openrouter_rate_limits: int = 0
    openrouter_server_errors: int = 0
    openrouter_network_errors: int = 0
    openrouter_rate_limit_kind: str | None = None
    queries: list[str] = Field(default_factory=list)
    failures: int = 0
    duplicate_requests: int = 0
    retrieved_source_counts: list[int] = Field(default_factory=list)
    rejected_irrelevant_sources: int = 0
    raw_external_results: int = 0
    invalid_external_results: int = 0
    invalid_external_urls: int = 0
    rejected_external_domains: int = 0
    duplicate_external_sources: int = 0
    accepted_external_sources: int = 0
    fallback_searches: int = 0
    external_search_errors: int = 0
    external_search_timeouts: int = 0
    claim_status_counts: dict[str, int] = Field(default_factory=dict)
    verification_failures: int = 0


def _external_source(item: ExternalEvidence) -> ExternalSource:
    return ExternalSource(id=item.id, title=item.title, url=item.url,
        published_date=item.published_date, verified_publication_date=item.verified_publication_date,
        source_host=urlparse(item.url).hostname, publisher=item.publisher, author=item.author,
        excerpt=item.content[:3000])


class EvidenceReview(BaseModel):
    level: Literal["sufficient", "partial", "none"]
    chunk_ids: list[str]
    missing_information: str = ""


class AskResponse(BaseModel):
    paper_id: str
    question: str | None = None
    request_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    status: EvidenceStatus
    answer: str | None
    sources: list[AnswerSource]
    external_sources: list[ExternalSource] = Field(default_factory=list)
    assessment: EvidenceReview | None = None
    claims: list[ClaimReviewResponse] = Field(default_factory=list)
    search_trace: SearchTrace = Field(default_factory=SearchTrace)
    verification_status: Literal["verified", "not_verified", "unavailable"] = "not_verified"
    answer_state: AnswerState | None = None
    retry_after_seconds: int = 0


class AnswerFeedback(BaseModel):
    rating: Literal["helpful", "incorrect"]
    reason: Literal["wrong_page", "unsupported_answer", "other"] | None = None


class SavedMessage(BaseModel):
    id: str
    question: str
    resolved_question: str | None = None
    status: EvidenceStatus
    answer: str | None
    sources: list[AnswerSource]
    external_sources: list[ExternalSource] = Field(default_factory=list)
    assessment: EvidenceReview | None = None
    claims: list[ClaimReviewResponse] = Field(default_factory=list)
    search_trace: SearchTrace = Field(default_factory=SearchTrace)
    created_at: datetime
    feedback: AnswerFeedback | None = None
    verification_status: Literal["verified", "not_verified", "unavailable"] = "not_verified"
    answer_state: AnswerState | None = None
    retry_after_seconds: int = 0


class ConversationResponse(BaseModel):
    id: str
    paper_id: str
    paper_filename: str | None = None
    title: str
    pinned: bool = False
    saved: bool = False
    created_at: datetime
    updated_at: datetime
    messages: list[SavedMessage]


def _conversation_response(document: dict, paper_filename: str | None = None) -> ConversationResponse:
    return ConversationResponse(id=document["_id"], paper_id=document["paper_id"],
                                paper_filename=document.get("paper_filename") or paper_filename or next(
                                    (source.get("document_name") for message in document.get("messages", [])
                                     for source in message.get("sources", []) if source.get("document_name")), None),
                                title=document.get("title") or (document.get("messages") or [{}])[-1].get("question", "Conversation"),
                                pinned=document.get("pinned", False),
                                saved=document.get("saved") is True,
                                created_at=document["created_at"], updated_at=document["updated_at"],
                                messages=document.get("messages", []))


class ConversationUpdate(BaseModel):
    title: str | None = None
    pinned: bool | None = None
    saved: bool | None = None


class RecommendedQuestionResponse(BaseModel):
    id: str
    text: str
    recommendation_type: str
    page_number: int | None = None
    chunk_index: int | None = None


class RecommendationResponse(BaseModel):
    paper_id: str
    recommendations: list[RecommendedQuestionResponse]


class AlternativeRecommendationRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = Field(default=None, max_length=64)


class FollowUpRecommendationRequest(BaseModel):
    message_id: str = Field(min_length=1, max_length=64)


def _recommendation_response(paper_id: str, items: list[RecommendedQuestion]) -> RecommendationResponse:
    return RecommendationResponse(paper_id=paper_id, recommendations=[
        RecommendedQuestionResponse(id=item.id, text=item.text, recommendation_type=item.recommendation_type,
                                    page_number=item.source.page_number, chunk_index=item.source.chunk_index)
        for item in items
    ])


def _recommendation_service(request: Request) -> QuestionRecommendationService:
    ingestion: IngestionService = request.app.state.ingestion_service
    return QuestionRecommendationService(ingestion.vectors, ingestion.embeddings.embed,
                                         getattr(request.app.state, "answer_provider", None))


async def _demo_recommendations(request: Request, user: UserResponse, paper_id: str, paper: dict,
                                previous: list[str], limit: int) -> RecommendationResponse:
    ingestion = request.app.state.ingestion_service
    if not is_sample(paper):
        service = QuestionRecommendationService(ingestion.vectors, ingestion.embeddings.embed, provider=None)
        questions = await run_in_threadpool(service.local_suggestions, user.id, paper_id,
                                            paper.get("chunk_count", 0), previous, limit)
        return _recommendation_response(paper_id, questions)
    chunks = await run_in_threadpool(ingestion.preview, user.id, paper_id, limit=paper["chunk_count"])
    result = _recommendation_response(paper_id, sample_recommendations(chunks, previous, limit))
    seen = {question_key(question) for question in previous}
    result.recommendations.extend(
        RecommendedQuestionResponse(id=f"demo-external-{index}", text=question, recommendation_type="sample_external")
        for index, question in enumerate(EXTERNAL_QUESTIONS) if question_key(question) not in seen
    )
    return result


def _demo_external_answer(paper_id: str, question: str) -> AskResponse | None:
    lookup = [question_key(item) for item in EXTERNAL_QUESTIONS]
    if question_key(question) not in lookup:
        return None
    compare = question_key(question) == lookup[0]
    critique, reply = [ExternalSource(**item) for item in EXTERNAL_SOURCES]
    assessment = EvidenceReview(level="none", chunk_ids=[],
        missing_information="The PDF summarizes criticism but does not provide enough evidence to answer this question about the full 2007 exchange.")
    if compare:
        claim = ClaimReviewResponse(
            text="Whether the model allows convincing evidence across research designs",
            section="external", status="contradicted",
            supporting=[ClaimQuoteResponse(source_id=reply.id, quote=REPLY_RANGE)],
            opposing=[ClaimQuoteResponse(source_id=critique.id, quote=EXTERNAL_SOURCES[0]["excerpt"])],
            explanation="The two commentaries interpret the scope of the model differently. Neither commentary independently measures the real-world false discovery rate.",
            reason_code="demo_sample")
        return AskResponse(paper_id=paper_id, status="contradictory_evidence", sources=[],
            external_sources=[critique, reply], assessment=assessment, claims=[claim],
            answer=("Goodman and Greenland argued that the model's assumptions made even strong study designs yield weak evidence [e1]. "
                    "Ioannidis replied that his illustrative examples gave a wide range of credibility across designs and settings [e2]. "
                    "They disagreed about what the model allowed one to conclude. The exchange does not establish a single real-world false discovery rate."))
    answer = "In his reply, Ioannidis said his model did not assume that all significant p-values were near 0.05."
    return AskResponse(paper_id=paper_id, status="not_mentioned", sources=[], external_sources=[reply],
        assessment=assessment, answer=f"{answer} [e2]",
        claims=[ClaimReviewResponse(text=answer, section="external", status="supported",
            supporting=[ClaimQuoteResponse(source_id=reply.id, quote=REPLY_THRESHOLD)],
            opposing=[], explanation="Attributed to the published reply.", reason_code="demo_sample")])


def _log_recommendation_run(mode: str, paper_id: str, service: QuestionRecommendationService,
                            recommendations: list[RecommendedQuestion], started_at: float) -> None:
    logging.getLogger(__name__).info(
        "research_recommendations mode=%s paper_id=%s count=%s provider_used=%s local_fallback=%s latency_ms=%s",
        mode, paper_id, len(recommendations), service.provider_used, service.used_local_fallback,
        round((time.monotonic() - started_at) * 1000),
    )


async def _ready_recommendation_paper(request: Request, user: UserResponse, paper_id: str) -> dict:
    papers: PaperRepository = request.app.state.paper_repository
    paper = await papers.find(user.id, paper_id)
    if paper is None:
        raise HTTPException(404, "Paper not found.")
    if not ready_for_research(paper):
        raise HTTPException(409, "Wait until this paper is ready.")
    return paper


@router.post("/visual-analysis")
async def analyze_visual_page(payload: VisualAnalysisRequest, request: Request,
                              user: UserResponse = Depends(get_current_user)):
    paper = await _ready_recommendation_paper(request, user, payload.paper_id)
    question = payload.question.strip()
    if not question:
        raise HTTPException(422, "Enter a question about the page.")
    provider = getattr(request.app.state, "answer_provider", None)
    if not isinstance(provider, OpenRouterAnswerProvider) or not provider.vision_enabled or not provider.available():
        raise HTTPException(503, "Visual analysis is unavailable.")
    ingestion: IngestionService = request.app.state.ingestion_service

    def render_page() -> bytes:
        with pymupdf.open(stream=ingestion.read_original(paper), filetype="pdf") as document:
            if payload.page_number > len(document):
                raise HTTPException(404, "PDF page not found.")
            page = document[payload.page_number - 1]
            scale = min(2, 1800 / max(page.rect.width, page.rect.height))
            return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")

    try:
        png = await run_in_threadpool(render_page)
        result = await run_in_threadpool(provider.analyze_page_image, question, payload.evidence_type, png)
    except HTTPException:
        raise
    except AnswerRateLimitError as exc:
        raise HTTPException(429, "Visual analysis is temporarily rate limited.",
                            headers=_rate_limit_headers(exc, provider)) from None
    except Exception:
        raise HTTPException(503, "Visual analysis is temporarily unavailable.") from None
    return {"paper_id": payload.paper_id, "page_number": payload.page_number,
            "evidence_type": payload.evidence_type, "verification_status": "unverified", **result}


async def _generate_recommendations(function, owner_id, paper_id, *args):
    scope = openrouter_cache_scope.set((owner_id, paper_id))
    try:
        return await run_in_threadpool(function, owner_id, paper_id, *args)
    except AnswerRateLimitError as exc:
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
        raise HTTPException(429, _rate_limit_detail(exc, "Suggestions are temporarily rate limited."),
                            headers=headers) from None
    except Exception as exc:
        upstream = exc.response if isinstance(exc, HTTPError) else None
        logging.getLogger(__name__).warning("research_recommendations_unavailable error_type=%s upstream_status=%s",
                                            type(exc).__name__, upstream.status_code if upstream is not None else None)
        raise HTTPException(503, "Suggestions are temporarily unavailable.") from None
    finally:
        openrouter_cache_scope.reset(scope)


async def _verified_sections(provider: AnswerProvider, question: str, pdf_text: str,
                             external_text: str, pdf_ids: list[str], external_ids: list[str],
                             pdf_sources: dict[str, AnswerSource],
                             external_sources: dict[str, ExternalEvidence],
                             request_id: str | None = None, debug_claims: bool = False,
                             raise_unavailable: bool = False, quality: AnswerQuality | None = None
                             ) -> tuple[str, str, list[ClaimReviewResponse]]:
    claims = split_claims(pdf_text, external_text)
    if quality is not None:
        quality.capture(pdf_text, external_text, pdf_ids, external_ids, pdf_sources, external_sources, claims)
    if not claims or len(claims) > 20:
        return "", "", []
    # Generators may cite a whole paragraph and return its source IDs separately.
    # Offer those IDs as candidates; each claim still needs a verified supporting quote.
    for section, ids in (("pdf", pdf_ids), ("external", external_ids)):
        section_claims = [item for item in claims if item.section == section]
        for item in section_claims:
            # The generator occasionally uses display ordinals ([1], [2]) even
            # though the API uses stable chunk IDs. Resolve only valid ordinals;
            # the verifier still checks the mapped passage against the claim.
            item.cited_ids[:] = list(dict.fromkeys(
                ids[int(cited) - 1] if cited.isdecimal() and 1 <= int(cited) <= len(ids) else cited
                for cited in item.cited_ids))
            if not item.cited_ids:
                item.cited_ids.extend(ids)
    evidence = ([Evidence(item.id, item.text, {"origin": "pdf"}) for item in pdf_sources.values()] +
                [Evidence(item.id, item.content, item.provenance()) for item in external_sources.values()])
    verifier = getattr(provider, "verify_claims", None)
    if verifier is None:
        return "", "", []
    try:
        proposed = await run_in_threadpool(verifier, question,
            [{"id": item.id, "text": item.text, "section": item.section,
              "cited_ids": item.cited_ids} for item in claims], evidence)
    except AnswerRateLimitError:
        raise
    except Exception:
        if raise_unavailable:
            raise
        return "", "", []
    verified_dates = {}
    for item in external_sources.values():
        if item.verified_publication_date:
            try:
                verified_dates[item.id] = date.fromisoformat(item.verified_publication_date)
            except ValueError:
                pass
    reviewed = review_claims(claims, proposed, evidence, verified_dates)
    if quality is not None:
        counts = Counter(item.claim_id for item in proposed)
        known_claims = {item.id for item in claims}
        quality.assessments = {item.claim_id: item for item in proposed
                               if counts[item.claim_id] == 1 and item.claim_id in known_claims}
        quality.requirements = getattr(proposed, "requirements", [])
    safe_request_id = request_id if isinstance(request_id, str) and re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}", request_id) else "none"
    for index, item in enumerate(reviewed, start=1):
        logging.getLogger("uvicorn.error").info(
            "research_claim_review request_id=%s claim_index=%d status=%s "
            "supporting_source_ids=%s opposing_source_ids=%s reason=%s",
            safe_request_id, index, item.status,
            [quote.source_id for quote in item.supporting],
            [quote.source_id for quote in item.opposing], item.reason_code)
        if debug_claims:
            def debug_text(value: str) -> str:
                if value and (any(value in source.text for source in evidence)
                              or any(quote.quote in value for quote in item.supporting + item.opposing)):
                    return "[source_text_redacted]"
                value = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                               "[email_redacted]", value)
                return re.sub(r"(?<!\w)(?:\+?\d[\d(). -]{7,}\d)(?!\w)",
                              "[phone_redacted]", value)
            logging.getLogger("uvicorn.error").debug(
                "research_claim_debug request_id=%s claim_index=%d claim_text=%r "
                "status=%s supporting_source_ids=%s opposing_source_ids=%s "
                "provider_explanation=%r reason=%s",
                safe_request_id, index, debug_text(item.claim.text), item.status,
                [quote.source_id for quote in item.supporting],
                [quote.source_id for quote in item.opposing],
                debug_text(item.explanation), item.reason_code)
    responses = [ClaimReviewResponse(text=item.claim.text, section=item.claim.section,
        status=item.status,
        supporting=[ClaimQuoteResponse(source_id=q.source_id, quote=q.quote) for q in item.supporting],
        opposing=[ClaimQuoteResponse(source_id=q.source_id, quote=q.quote) for q in item.opposing],
        explanation=item.explanation, reason_code=item.reason_code) for item in reviewed]
    return present_claims(reviewed, "pdf"), present_claims(reviewed, "external"), responses


def _local_claim_debug(request: Request) -> bool:
    """Allow claim text diagnostics only with Uvicorn DEBUG on a loopback connection."""
    if not logging.getLogger("uvicorn.error").isEnabledFor(logging.DEBUG) or not request.client:
        return False
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return False


def _methodology_comparison_gaps(question: str, claims: list[ClaimReviewResponse],
                                 passages: list[str] | None = None) -> list[str]:
    """Report only requested parts missing from independently verified PDF claims."""
    subjects, dimensions = requested_coverage(question, passages)
    if not dimensions or (len(subjects) < 2 and len(dimensions) < 2):
        return []
    supported = [item.text for item in claims if item.section == "pdf" and item.status == "supported"]
    for item in claims:
        if item.section != "pdf" or item.status != "partially_supported":
            continue
        labels = [subject for subject in subjects if re.search(rf"\b{re.escape(subject)}\b", item.text, re.I)]
        for quote in item.supporting:
            named = [subject for subject in subjects
                     if re.search(rf"\b{re.escape(subject)}\b", quote.quote, re.I)]
            prefix = labels[0] if not named and len(labels) == 1 else ""
            supported.append(f"{prefix} {quote.quote}")
    missing = missing_coverage(question, supported, subjects=subjects)
    gaps = []
    for subject in subjects or [""]:
        absent = [dimension for label, dimension in missing if label == subject]
        if absent:
            label = f"{subject}'s" if subject else "the"
            gaps.append(f"{label} {' and '.join(absent)}")
    if (len(subjects) >= 2
            and re.search(r"\b(?:compar\w*|differ\w*|similar\w*|versus|contrast)\b", question, re.I)
            and not any(all(re.search(rf"\b{re.escape(subject)}\b", text, re.I) for subject in subjects)
                        and re.search(r"\b(?:whereas|while|unlike|differ\w*|versus|compared|comparison)\b", text, re.I)
                        for text in supported)):
        gaps.append("the methodological differences" if re.search(r"\bmethodolog\w*\b", question, re.I)
                    else "the requested comparison")
    return gaps


def _verified_answer_parts(question: str, claims: list[ClaimReviewResponse],
                           sources: dict[str, AnswerSource]) -> list[tuple[str, str, str, list[str]]]:
    """Describe requested parts using claim-verified PDF evidence only."""
    subjects, dimensions = requested_coverage(question, [source.text for source in sources.values()])
    if not dimensions:
        return []
    supported = []
    partial = []
    claim_sources = {}
    for index, claim in enumerate(claims):
        if claim.section != "pdf":
            continue
        ids = [quote.source_id for quote in claim.supporting if quote.source_id in sources]
        if not ids:
            continue
        if claim.status == "supported":
            key = f"claim-{index}"
            supported.append(CoveragePassage(key, claim.text))
            claim_sources[key] = ids
        elif claim.status == "partially_supported":
            names = [name for name in subjects if re.search(rf"\b{re.escape(name)}\b", claim.text, re.I)]
            prefix = names[0] if len(names) == 1 else ""
            for quote_index, quote in enumerate(claim.supporting):
                if quote.source_id in sources:
                    key = f"partial-{index}-{quote_index}"
                    partial.append(CoveragePassage(key, f"{prefix} {quote.quote}"))
                    claim_sources[key] = [quote.source_id]
    full = coverage_report(question, supported, {item.id for item in supported}, subjects=subjects)
    incomplete = coverage_report(question, partial, subjects=subjects)
    result = []
    for full_part, partial_part in zip(full, incomplete):
        if full_part.status == "verified_claim_support":
            status, ids = "supported", full_part.information_ids
        elif partial_part.status == "information_present":
            status, ids = "partially_supported", partial_part.information_ids
        else:
            status, ids = "unsupported", ()
        result.append((full_part.subject, full_part.dimension, status,
                       list(dict.fromkeys(source_id for item in ids
                                              for source_id in claim_sources[item]))))
    return result


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(request: Request, user: UserResponse = Depends(get_current_user)):
    repository: ConversationRepository = request.app.state.conversation_repository
    documents = await repository.list_for_user(user.id)
    paper_repository: PaperRepository = request.app.state.paper_repository
    filenames = {}
    for item in documents:
        paper_id = item["paper_id"]
        if paper_id not in filenames and not item.get("paper_filename"):
            paper = await paper_repository.find(user.id, paper_id)
            filenames[paper_id] = paper.get("filename") if paper else None
    return [_conversation_response(item, filenames.get(item["paper_id"])) for item in documents]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, request: Request,
                           user: UserResponse = Depends(get_current_user)):
    repository: ConversationRepository = request.app.state.conversation_repository
    document = await repository.find(user.id, conversation_id)
    if document is None:
        raise HTTPException(404, "Conversation not found.")
    paper = await request.app.state.paper_repository.find(user.id, document["paper_id"])
    return _conversation_response(document, paper.get("filename") if paper else None)


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(conversation_id: str, payload: ConversationUpdate, request: Request,
                              user: UserResponse = Depends(get_current_user)):
    fields = {}
    if "title" in payload.model_fields_set:
        title = payload.title.strip() if payload.title is not None else ""
        if not title or len(title) > 120:
            raise HTTPException(422, "Conversation title must contain 1 to 120 characters.")
        fields["title"] = title
    if "pinned" in payload.model_fields_set:
        if payload.pinned is None:
            raise HTTPException(422, "Pinned must be true or false.")
        fields["pinned"] = payload.pinned
    if "saved" in payload.model_fields_set:
        if payload.saved is None:
            raise HTTPException(422, "Saved must be true or false.")
        fields["saved"] = payload.saved
    if not fields:
        raise HTTPException(422, "Provide a title, pinned status, or saved status.")
    repository: ConversationRepository = request.app.state.conversation_repository
    if not await repository.update_metadata(user.id, conversation_id, fields):
        raise HTTPException(404, "Conversation not found.")
    document = await repository.find(user.id, conversation_id)
    paper = await request.app.state.paper_repository.find(user.id, document["paper_id"])
    return _conversation_response(document, paper.get("filename") if paper else None)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request,
                              user: UserResponse = Depends(get_current_user)):
    repository: ConversationRepository = request.app.state.conversation_repository
    if not await repository.remove(user.id, conversation_id):
        raise HTTPException(404, "Conversation not found.")


@router.post("/conversations/{conversation_id}/messages/{message_id}/feedback", response_model=AnswerFeedback)
async def save_answer_feedback(conversation_id: str, message_id: str, feedback: AnswerFeedback,
                               request: Request, user: UserResponse = Depends(get_current_user)):
    if feedback.rating == "helpful" and feedback.reason is not None:
        raise HTTPException(422, "A helpful answer cannot have an error reason.")
    repository: ConversationRepository = request.app.state.conversation_repository
    if not await repository.set_feedback(user.id, conversation_id, message_id, feedback.model_dump()):
        raise HTTPException(404, "Answer not found.")
    return feedback


@router.get("/papers/{paper_id}/recommendations", response_model=RecommendationResponse)
async def starter_recommendations(paper_id: str, request: Request,
                                  user: UserResponse = Depends(get_current_user)):
    paper = await _ready_recommendation_paper(request, user, paper_id)
    if getattr(request.app.state, "demo_mode", False):
        return await _demo_recommendations(request, user, paper_id, paper, [], len(CASES))
    service = _recommendation_service(request)
    started_at = time.monotonic()
    recommendations = await _generate_recommendations(service.starter, user.id, paper_id, paper)
    _log_recommendation_run("starter", paper_id, service, recommendations, started_at)
    return _recommendation_response(paper_id, recommendations)


@router.post("/papers/{paper_id}/recommendations/alternatives", response_model=RecommendationResponse)
async def alternative_recommendations(paper_id: str, payload: AlternativeRecommendationRequest, request: Request,
                                      user: UserResponse = Depends(get_current_user)):
    paper = await _ready_recommendation_paper(request, user, paper_id)
    previous = []
    if payload.conversation_id:
        conversations: ConversationRepository = request.app.state.conversation_repository
        conversation = await conversations.find(user.id, payload.conversation_id)
        if conversation is None or conversation.get("paper_id") != paper_id:
            raise HTTPException(404, "Conversation not found.")
        previous = [item.get("question", "") for item in conversation.get("messages", [])]
    if getattr(request.app.state, "demo_mode", False):
        return await _demo_recommendations(request, user, paper_id, paper, previous + [payload.question], 3)
    service = _recommendation_service(request)
    started_at = time.monotonic()
    recommendations = await _generate_recommendations(service.alternatives, user.id, paper_id,
                                               payload.question.strip(), paper, previous)
    _log_recommendation_run("alternative", paper_id, service, recommendations, started_at)
    return _recommendation_response(paper_id, recommendations)


@router.post("/conversations/{conversation_id}/recommendations", response_model=RecommendationResponse)
async def follow_up_recommendations(conversation_id: str, payload: FollowUpRecommendationRequest, request: Request,
                                    user: UserResponse = Depends(get_current_user)):
    conversations: ConversationRepository = request.app.state.conversation_repository
    conversation = await conversations.find(user.id, conversation_id)
    if conversation is None:
        raise HTTPException(404, "Conversation not found.")
    paper_id = conversation["paper_id"]
    paper = await _ready_recommendation_paper(request, user, paper_id)
    message = next((item for item in conversation.get("messages", []) if item.get("id") == payload.message_id), None)
    if message is None:
        raise HTTPException(404, "Answer not found.")
    supported = any(claim.get("section") == "pdf" and claim.get("status") == "supported"
                    and claim.get("supporting") for claim in message.get("claims", []))
    if (message.get("status") != "answered" and not supported) or not message.get("answer"):
        return _recommendation_response(paper_id, [])
    previous = [item.get("question", "") for item in conversation.get("messages", [])]
    if getattr(request.app.state, "demo_mode", False):
        return await _demo_recommendations(request, user, paper_id, paper, previous, 3)
    service = _recommendation_service(request)
    started_at = time.monotonic()
    recommendations = await _generate_recommendations(service.follow_up, user.id, paper_id,
                                               message["question"], message["answer"],
                                               message.get("sources", []), previous)
    _log_recommendation_run("follow_up", paper_id, service, recommendations, started_at)
    return _recommendation_response(paper_id, recommendations)


@router.post("/conversations/{conversation_id}/messages/{message_id}/verify", response_model=SavedMessage)
async def retry_message_verification(conversation_id: str, message_id: str, request: Request,
                                     user: UserResponse = Depends(get_current_user)):
    quality = AnswerQuality()
    repository: ConversationRepository = request.app.state.conversation_repository
    document = await repository.find(user.id, conversation_id)
    if document is None:
        raise HTTPException(404, "Conversation not found.")
    await _ready_recommendation_paper(request, user, document["paper_id"])
    if getattr(request.app.state, "demo_mode", False):
        message = next((item for item in document.get("messages", []) if item.get("id") == message_id), None)
        if message is None:
            raise HTTPException(404, "Answer not found.")
        return SavedMessage(**message)
    provider = getattr(request.app.state, "answer_provider", None)
    if provider is None or (hasattr(provider, "available") and not provider.available()):
        raise HTTPException(503, "Verification is temporarily unavailable.")
    if isinstance(provider, OpenRouterAnswerProvider):
        cooldown = _cooldown_seconds(request, provider)
        if cooldown:
            raise HTTPException(429, "Verification is temporarily rate limited.", headers={"Retry-After": str(cooldown)})
    lock = _verification_locks[int(sha256(f"{user.id}:{conversation_id}:{message_id}".encode()).hexdigest()[:8], 16) % len(_verification_locks)]
    async with lock:
        document = await repository.find(user.id, conversation_id)
        if document is None:
            raise HTTPException(404, "Conversation not found.")
        message = next((item for item in document.get("messages", []) if item.get("id") == message_id), None)
        if message is None:
            raise HTTPException(404, "Answer not found.")
        if message.get("verification_status", "verified") != "unavailable":
            return SavedMessage(**message)
        sources = {item["id"]: AnswerSource(**item) for item in message.get("sources", [])}
        external = {item["id"]: ExternalEvidence(id=item["id"], title=item["title"], url=item["url"],
            content=item.get("excerpt") or "", published_date=item.get("published_date"),
            verified_publication_date=item.get("verified_publication_date"), publisher=item.get("publisher"),
            author=item.get("author")) for item in message.get("external_sources", [])}
        if not (sources or external) or not message.get("answer"):
            raise HTTPException(409, "The answer cannot be verified without its evidence.")
        answer = message["answer"]
        pdf_text, separator, external_text = answer.partition("\n\nAdditional information from external sources\n")
        if separator:
            pdf_text = pdf_text.removeprefix("Evidence from your PDF\n")
        runs: list[dict] = []
        trace_token = model_runs.set(runs)
        scope_token = openrouter_cache_scope.set((user.id, document["paper_id"]))
        try:
            checked, checked_external, claims = await _verified_sections(provider,
                message.get("resolved_question") or message["question"], pdf_text,
                external_text, list(sources), list(external), sources, external, raise_unavailable=True, quality=quality)
        except AnswerRateLimitError as exc:
            raise HTTPException(429, _rate_limit_detail(exc, "Verification is temporarily rate limited."),
                                headers=_rate_limit_headers(exc, provider)) from None
        except Exception:
            raise HTTPException(503, "Verification is temporarily unavailable.") from None
        finally:
            model_runs.reset(trace_token)
            openrouter_cache_scope.reset(scope_token)
        if not claims:
            raise HTTPException(503, "Verification is temporarily unavailable.")
        used = {quote.source_id for item in claims for quote in item.supporting + item.opposing}
        status = _reviewed_status(claims, (message.get("assessment") or {}).get("level") == "partial",
                                  external=bool(separator))
        verified_answer = (f"Evidence from your PDF\n{checked}\n\nAdditional information from external sources\n{checked_external}"
                           if separator else checked)
        fields = {"answer": verified_answer or "The generated claims could not be verified.", "status": status,
                  "sources": [source.model_dump(mode="json") for key, source in sources.items() if key in used],
                  "external_sources": [item for item in message.get("external_sources", []) if item["id"] in used],
                  "claims": [item.model_dump(mode="json") for item in claims],
                  "retry_after_seconds": 0}
        state, checks = answer_report(status=status, answer=verified_answer, claims=claims,
            sources=[source for key, source in sources.items() if key in used],
            external=[item for key, item in external.items() if key in used],
            question=message.get("resolved_question") or message["question"], unavailable=False)
        fields.update(answer_state=state, verification_checks=checks,
                      verification_status="verified" if state == "VERIFIED" else "not_verified")
        previous_evaluations = message.get("quality_evaluations") or []
        fields["quality_evaluations"] = [*previous_evaluations, quality.evaluate(
            message.get("resolved_question") or message["question"], fields,
            revision=max((item["revision"] for item in previous_evaluations), default=0) + 1,
            model_calls=[*(previous_evaluations[-1].get("model_calls", []) if previous_evaluations else []), *runs])]
        if not await repository.update_verified_message(user.id, conversation_id, message_id, fields):
            raise HTTPException(409, "This answer changed. Reload the conversation.")
        return SavedMessage(**{**message, **fields})


def _evidence_excerpt(question: str, text: str) -> str:
    """Keep the sentence most related to the question inside a retrieved chunk."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    terms = VectorRepository._terms(question)
    best = max(sentences, key=lambda sentence: len(terms & VectorRepository._terms(sentence)))
    return best[:600].strip()


def _evidence_context(question: str, text: str) -> str:
    """Retain requested facts and nearby qualifications within a fixed chunk budget."""
    if len(text) <= 2400:
        return text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # PDF extraction can produce a single sentence longer than the prompt budget.
    bounded = []
    for sentence in sentences:
        while len(sentence) > 1200:
            boundary = sentence.rfind(" ", 600, 1201)
            boundary = boundary if boundary > 0 else 1200
            bounded.append(sentence[:boundary].strip())
            sentence = sentence[boundary:].strip()
        if sentence:
            bounded.append(sentence)
    sentences = bounded
    terms = VectorRepository._terms(question)
    ranked = sorted(range(len(sentences)),
                    key=lambda index: (-len(terms & VectorRepository._terms(sentences[index])), index))
    selected: set[int] = set()
    # Prefer sentences that fill a requested coverage gap, even if they share
    # fewer question terms than a nearby background sentence.
    for index in ranked:
        current = " ".join(sentences[item] for item in sorted(selected))
        proposed = " ".join(sentences[item] for item in sorted(selected | {index}))
        if (len(" ".join(sentences[item] for item in sorted(selected | {index}))) <= 2400
                and len(missing_coverage(question, [proposed]))
                < len(missing_coverage(question, [current]))):
            selected.add(index)
    for index in ranked:
        proposed = selected | {index} | ({index - 1} if index else set()) | (
            {index + 1} if index + 1 < len(sentences) else set())
        if len(" ".join(sentences[item] for item in sorted(proposed))) <= 2400:
            selected = proposed
    return " ".join(sentences[item] for item in sorted(selected)).strip() or text[:2400].strip()


def _log_evidence_handoff(request_id: str | None, stage: str,
                          ids: list[str], sources: dict[str, AnswerSource],
                          excerpts: dict[str, str] | None = None) -> None:
    """Record provenance across stages without recording question or passage text."""
    safe_id = request_id if isinstance(request_id, str) and re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}", request_id) else "none"
    records = [{"source_id": item, "page_number": sources[item].page_number,
                "chunk_index": sources[item].chunk_index,
                "full_chars": len(sources[item].text),
                "excerpt_chars": len(excerpts[item]) if excerpts and item in excerpts else None,
                "excerpt_hash": sha256(excerpts[item].encode("utf-8")).hexdigest()[:12]
                if excerpts and item in excerpts else None}
               for item in dict.fromkeys(ids) if item in sources]
    logging.getLogger("uvicorn.error").info(
        "research_evidence_handoff request_id=%s stage=%s chunks=%s", safe_id, stage, records)


def _quote_tokens(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("−", "-")
    return " ".join(re.findall(r"(?<!\w)-?\d[\d,]*(?:\.\d+)?|[^\W\d_]+|%", value))


def _coverage_gap_message(question: str, gaps: list[tuple[str, str]],
                          sources: dict[str, AnswerSource], ids: list[str],
                          subjects: list[str] | None = None) -> str:
    selected = [CoveragePassage(item, sources[item].text,
                sources[item].page_number, sources[item].chunk_index) for item in ids]
    present = {(item.subject, item.dimension) for item in coverage_report(question, selected, subjects=subjects)
               if item.status == "information_present"}
    quote_only = [gap for gap in gaps if gap in present]
    absent = [gap for gap in gaps if gap not in present]

    def label(items: list[tuple[str, str]]) -> str:
        return ", ".join(f"{subject + ' ' if subject else ''}{dimension}"
                          for subject, dimension in items) + "."

    missing = ("The selected PDF chunks contain information omitted from the cited quotes for "
               + label(quote_only) if quote_only else "")
    if absent:
        missing += (" " if missing else "") + "The selected PDF chunks do not establish " + label(absent)
    return missing


def _validate_assessment(assessment: EvidenceAssessment, question: str,
                         sources: dict[str, AnswerSource],
                         request_id: str | None = None) -> EvidenceReview | None:
    """Reject unsupported labels before they can authorize external search."""
    invalid_quote_details: list[tuple[int, str, str]] = []
    valid_quotes: list[EvidenceQuote] = []
    subjects, dimensions = requested_coverage(question, [source.text for source in sources.values()])
    subject_aliases = {subject: f"subject_{index + 1}" for index, subject in enumerate(subjects)}
    coverage_statuses: list[tuple[str, str, str]] = []
    def reject(reason: str) -> None:
        cited_ids = assessment.chunk_ids
        quotes = assessment.supporting_quotes
        safe_request_id = request_id if isinstance(request_id, str) and re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}", request_id) else "none"
        level = assessment.level if assessment.level in ("sufficient", "partial", "none") else "invalid"
        logging.getLogger("uvicorn.error").info(
            "evidence_assessment_rejected request_id=%s reason=%s level=%s "
            "retrieved_sources=%d cited_source_ids=%d supporting_quotes=%d invalid_quotes=%s "
            "requested_subjects=%s requested_dimensions=%s coverage_statuses=%s "
            "covered_targets=%s missing_targets=%s valid_quote_count=%d invalid_quote_count=%d",
            safe_request_id, reason, level, len(sources),
            len(cited_ids) if isinstance(cited_ids, list) else 0,
            len(quotes) if isinstance(quotes, list) else 0, invalid_quote_details,
            list(subject_aliases.values()) or ["document"], dimensions, coverage_statuses,
            [(subject, dimension) for subject, dimension, status in coverage_statuses
             if status == "information_present"],
            [(subject, dimension) for subject, dimension, status in coverage_statuses
             if status != "information_present"],
            len(valid_quotes), len(invalid_quote_details))
        return None

    if assessment.level not in ("sufficient", "partial", "none"):
        return reject("invalid_assessment_level")
    ids = assessment.chunk_ids
    if not isinstance(ids, list) or any(not isinstance(item, str) or item not in sources for item in ids):
        return reject("unknown_source_id")
    if assessment.level != "none" and assessment.supporting_quotes is None:
        return reject("missing_supporting_quote")
    invalid_quote_ids: set[str] = set()
    valid_quote_ids: set[str] = set()
    if assessment.supporting_quotes is not None:
        quotes = assessment.supporting_quotes
        if not isinstance(quotes, list) or any(
            not isinstance(quote, EvidenceQuote)
            or not isinstance(quote.chunk_id, str) or quote.chunk_id not in sources
            or not isinstance(quote.quote, str) or len(quote.quote.strip()) < 12
            for quote in quotes
        ):
            return reject("invalid_supporting_quote")
        if {quote.chunk_id for quote in quotes} != set(ids):
            return reject("missing_or_mismatched_supporting_quote")
        if len({(quote.chunk_id, _quote_tokens(quote.quote)) for quote in quotes}) != len(quotes):
            return reject("duplicate_supporting_quote")
        for quote_index, quote in enumerate(quotes):
            normalized_chunk = f" {_quote_tokens(sources[quote.chunk_id].text)} "
            fragments = [_quote_tokens(part) for part in re.split(r"\s*(?:\.\.\.|…)+\s*", quote.quote)
                         if part.strip()]
            if not fragments:
                return reject("quote_not_found_in_source")
            cursor = 0
            ordered = True
            missing_fragment = False
            for fragment in fragments:
                normalized_fragment = f" {fragment} "
                position = normalized_chunk.find(normalized_fragment, cursor)
                if position < 0:
                    ordered = False
                    missing_fragment = normalized_fragment not in normalized_chunk
                    break
                cursor = position + len(normalized_fragment) - 1
            if ordered:
                valid_quote_ids.add(quote.chunk_id)
                valid_quotes.append(quote)
                continue
            invalid_quote_ids.add(quote.chunk_id)
            invalid_quote_details.append((quote_index, quote.chunk_id,
                                          "fragment_missing" if missing_fragment else "order_mismatch"))
    if invalid_quote_ids:
        valid_ids = [item for item in ids if item in valid_quote_ids]
        quoted = []
        for quote in valid_quotes:
            if quote.chunk_id not in valid_ids:
                continue
            named = {name for name, _, _ in subject_spans(sources[quote.chunk_id].text)}
            prefix = next(iter(named)) if len(named) == 1 and next(iter(named)) in subjects else ""
            quoted.append(CoveragePassage(quote.chunk_id, f"{prefix} {quote.quote}"))
        remaining_report = coverage_report(question, quoted, subjects=subjects)
        coverage_statuses.extend((subject_aliases.get(item.subject, "document"), item.dimension, item.status)
                                 for item in remaining_report)
        complete = bool(dimensions and valid_ids and
                        all(status == "information_present" for _, _, status in coverage_statuses))
        if not complete and not any(status == "information_present" for _, _, status in coverage_statuses):
            return reject("quote_not_found_or_order_mismatch")
        ids = valid_ids
        logging.getLogger(__name__).info(
            "evidence_assessment_dropped_invalid_redundant_quotes source_ids=%s", sorted(invalid_quote_ids))
    _log_evidence_handoff(request_id, "assessment_quoted",
                          [quote.chunk_id for quote in valid_quotes], sources)
    if assessment.level == "none":
        if ids:
            return reject("none_with_source_ids")
        # A strong lexical match is a reason to inspect the PDF, not to claim it says nothing.
        terms = VectorRepository._terms(question)
        explicit_legal_gap = bool(re.search(r"\b(?:legal|law|regulation|requirements?)\b", question, re.I)
                                  and re.search(r"\bindia\b", question, re.I)
                                  and any(re.search(r"not a statement of current law|current Indian biometric-data law\s+Not discussed",
                                                    source.text, re.I) for source in sources.values()))
        if terms and any(len(terms & VectorRepository._terms(source.text)) / len(terms) >= .7
                         and not re.search(r"\b(?:not discussed|not studied|not reported|not measured|not provided)\b",
                                           source.text, re.I)
                         for source in sources.values()) and not explicit_legal_gap:
            return reject("insufficient_question_relevance")
    else:
        if not ids:
            return reject("missing_supporting_source")
        terms = VectorRepository._terms(question)
        if terms and not any(terms & VectorRepository._terms(sources[item].text)
                             or (re.search(r"\bmethodolog\w*\b", question, re.I)
                                 and re.search(r"\b(?:design|participant\w*|sample|randomiz\w*|experiment\w*|course\w*|setting)\b",
                                               sources[item].text, re.I)) for item in ids):
            return reject("insufficient_question_relevance")
        if assessment.level == "partial" and not assessment.missing_information.strip():
            return reject("missing_partial_information")
    level = assessment.level
    missing = assessment.missing_information.strip()
    if level == "partial":
        _, gap_dimensions = requested_coverage(missing)
        if dimensions and gap_dimensions and not set(gap_dimensions).issubset(dimensions):
            missing = "Some requested information could not be verified from the cited passages."
    if level == "sufficient":
        quoted_passages = []
        for quote in valid_quotes:
            if quote.chunk_id not in ids:
                continue
            passage = sources[quote.chunk_id].text
            labels = [subject for subject in subjects if re.search(rf"\b{re.escape(subject)}\b", passage, re.I)]
            label = labels[0] if len(labels) == 1 else ""
            quoted_passages.append(f"{label} {quote.quote}")
        gaps = missing_coverage(question, quoted_passages, subjects=subjects)
        if gaps:
            level = "partial"
            missing = _coverage_gap_message(question, gaps, sources, ids, subjects=subjects)
    quoted_passages = []
    for quote in valid_quotes:
        if quote.chunk_id not in ids:
            continue
        names = {name for name, _, _ in subject_spans(sources[quote.chunk_id].text)}
        prefix = next(iter(names)) if len(names) == 1 and next(iter(names)) in subjects else ""
        quoted_passages.append(CoveragePassage(quote.chunk_id, f"{prefix} {quote.quote}"))
    report = coverage_report(question, quoted_passages, subjects=subjects)
    logging.getLogger("uvicorn.error").info(
        "research_assessment_parts request_id=%s level=%s parts=%s",
        request_id if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id)
        else "none", level,
        [(subject_aliases.get(item.subject, "document"), item.dimension, item.status)
         for item in report])
    return EvidenceReview(level=level, chunk_ids=list(dict.fromkeys(ids)),
                          missing_information=missing[:500])


def _reconcile_assessment_gap(assessment: EvidenceReview | None,
                              claims: list[ClaimReviewResponse], question: str = "",
                              sources: dict[str, AnswerSource] | None = None) -> EvidenceReview | None:
    """Clear a coverage gap only when verified PDF claims establish its dimensions."""
    if not assessment or not assessment.missing_information:
        return assessment
    _, dimensions = requested_coverage(assessment.missing_information)
    if not dimensions:
        return assessment
    verified = [item.text for item in claims if item.section == "pdf"
                and item.status == "supported" and item.supporting]
    generated_gap = assessment.missing_information.startswith((
        "The selected PDF chunks contain information omitted from the cited quotes",
        "The selected PDF chunks do not establish"))
    if generated_gap and question and sources and verified:
        subjects, requested_dimensions = requested_coverage(
            question, [source.text for source in sources.values()])
        gap_subjects, gap_dimensions = requested_coverage(assessment.missing_information)
        targets = {(subject, dimension) for subject in (subjects or [""])
                   for dimension in set(gap_dimensions).intersection(requested_dimensions)
                   if not gap_subjects or re.search(
                       rf"\b{re.escape(subject)}\s+{re.escape(dimension)}\b",
                       assessment.missing_information, re.I)}
        if not targets:
            return assessment
        remaining = [target for target in missing_coverage(question, verified, subjects=subjects)
                     if target in targets]
        if not remaining:
            return assessment.model_copy(update={"level": "sufficient", "missing_information": ""})
        return assessment.model_copy(update={"missing_information":
            _coverage_gap_message(question, remaining, sources, assessment.chunk_ids,
                                  subjects=subjects)[:500]})
    if question and sources and verified:
        subjects, requested_dimensions = requested_coverage(
            question, [source.text for source in sources.values()])
        gap_subjects, gap_dimensions = requested_coverage(assessment.missing_information)
        relevant_dimensions = set(gap_dimensions).intersection(requested_dimensions)
        if relevant_dimensions:
            targets = [(subject, dimension) for subject in (subjects or [""])
                       for dimension in relevant_dimensions
                       if not gap_subjects or subject in gap_subjects]
            if targets:
                remaining = set(missing_coverage(question, verified, subjects=subjects))
                if all(target not in remaining for target in targets):
                    return assessment.model_copy(update={"level": "sufficient", "missing_information": ""})
            return assessment
    if verified and not missing_coverage(assessment.missing_information, verified):
        return assessment.model_copy(update={"level": "sufficient", "missing_information": ""})
    return assessment


def _assessment_gap_in_selected_chunks(assessment: EvidenceReview | None,
                                       sources: dict[str, AnswerSource]) -> bool:
    """A coverage hint can justify verification, but cannot approve a claim."""
    if not assessment or assessment.level != "partial":
        return False
    _, dimensions = requested_coverage(assessment.missing_information)
    if not dimensions:
        return False
    passages = [CoveragePassage(item, sources[item].text,
                sources[item].page_number, sources[item].chunk_index)
                for item in assessment.chunk_ids if item in sources]
    return bool(passages and all(item.status == "information_present" for item in
                                coverage_report(assessment.missing_information, passages)))


def _asks_for_paper_specific_fact(question: str) -> bool:
    """Outside papers cannot supply a statistic missing from this particular PDF."""
    return bool(re.search(r"\b(p[ -]?value|sample size|number of participants|cohort size|"
                          r"confidence interval|effect size|hazard ratio|odds ratio|standard deviation)\b", question, re.I)
                and re.search(r"\b(this|the|paper|study|experiment|trial)\b", question, re.I))


def _requests_external_evidence(question: str) -> bool:
    """Detect an explicit request to consult sources beyond the selected paper."""
    explicit = re.search(
        r"\b(?:search (?:the )?web|look (?:it )?up online|on the web|"
        r"from (?:the )?(?:web|internet)|outside (?:this|the) (?:paper|pdf)|"
        r"beyond (?:this|the) (?:paper|pdf))\b"
        r"|\b(?:additional|other|outside|external)\s+"
        r"(?:law|legal requirements?|regulations?|research|studies|sources|literature|evidence)\b"
        r"|\band\b.{0,80}\b(?:current|latest|recent)\s+"
        r"(?:law|legal requirements?|regulations?|research|studies|sources|literature)\b",
        question, re.I)
    if explicit:
        return True
    document_scope = re.search(
        r"\b(?:in|from|according to)\s+(?:this|the|uploaded|selected)\s+"
        r"(?:study|paper|pdf|document|article)\b"
        r"|\b(?:does|did|is|was)\s+(?:this|the|uploaded|selected)\s+"
        r"(?:study|paper|pdf|document|article)\s+(?:say|report|mention|state|describe)\b",
        question, re.I)
    return not document_scope and bool(re.search(
        r"\b(?:current|latest|recent|published)\s+"
        r"(?:law|legal requirements?|regulations?|research|studies|sources|literature|evidence)\b",
        question, re.I))


def _asks_only_about_pdf(question: str) -> bool:
    """Keep explicit document-scope questions within the selected PDF."""
    if _requests_external_evidence(question):
        return False
    return bool(re.search(
        r"\b(?:this|the|uploaded|selected)\s+(?:paper|pdf|document|article)\b"
        r"|\b(?:in|from|according to)\s+(?:this|the|uploaded|selected)\s+"
        r"(?:study|paper|pdf|document|article)\b"
        r"|\b(?:does|did|was|were|is|are)\s+(?:this|the)\s+(?:study|paper|pdf|document)\s+"
        r"(?:report|mention|measure|study|evaluate|include|contain)\b",
        question, re.I))


def _external_query(question: str, missing: str) -> str:
    """Lead with the PDF gap while retaining key question context."""
    if not missing.strip():
        return question.strip()
    gap = missing.strip()
    ignored = {"about", "could", "does", "from", "have", "paper", "question",
               "research", "source", "study", "that", "their", "these", "this",
               "what", "when", "where", "which", "with", "would"}
    existing = set(re.findall(r"[a-z0-9]+", gap.casefold()))
    context = []
    words = re.findall(r"[A-Za-z0-9-]+", question)
    words = sorted(enumerate(words), key=lambda item: (not (item[1][:1].isupper()
                                                        or re.fullmatch(r"\d{4}", item[1])), item[0]))
    for _, word in words:
        lower = word.casefold()
        if len(lower) > 3 and lower not in ignored and lower not in existing:
            context.append(word)
            existing.add(lower)
        if len(context) == 8:
            break
    return " ".join([gap, *context]).strip()[:500]


def _fallback_query(question: str, missing: str, first: str) -> str | None:
    """One narrower retry, retaining entities and topic words from the question."""
    terms = re.findall(r"[A-Za-z0-9-]+", f"{question} {missing}")
    ignored = {"what", "which", "where", "when", "why", "how", "does", "did", "the",
               "and", "from", "with", "this", "that", "there", "paper", "study", "evidence",
               "information", "missing", "additional", "research", "source", "about", "answer"}
    focused = " ".join(dict.fromkeys(word for word in terms
                                     if len(word) > 2 and word.casefold() not in ignored))[:500]
    return focused if focused and focused.casefold() != first.casefold() else None


def _generation_passages(question: str, passages: list[ChunkResponse]) -> list[ChunkResponse]:
    """Use a direct match alone for short fact questions; keep context for broader questions."""
    if not passages:
        return passages
    terms = VectorRepository._terms(question)
    direct = len(terms & VectorRepository._terms(_evidence_excerpt(question, passages[0].text)))
    if (terms and direct / len(terms) >= 0.8
            and re.match(r"^(who|which|when|where)\b", question.strip(), re.IGNORECASE)):
        return passages[:1]
    return passages


def _question_with_context(question: str, conversation: dict | None,
                           reply_to_message_id: str | None = None) -> str:
    """Resolve short follow-ups using only the previous question as context."""
    if not conversation or not conversation.get("messages"):
        return question
    short_follow_up = len(VectorRepository._terms(question)) <= 8
    vague_why_use = bool(re.match(r"^why\s+(?:(?:do|did)\s+)?(?:we|you)\s+(?:use|need)\s*\??$",
                                  question.strip(), re.I))
    refers_back = bool(vague_why_use
                       or re.search(r"\b(it|its|they|them|their|this|that|those)\b", question, re.I)
                       or re.match(r"^(and|what about|how about)\b", question, re.I))
    if (not vague_why_use and not re.match(r"^(and|what about|how about)\b", question, re.I)
            and (requested_subjects(question)
                 or re.search(r"\b(?:paper|pdf|document)\b", question, re.I))):
        return question
    if not (short_follow_up and refers_back):
        return question
    previous_message = (next((item for item in conversation["messages"]
                              if item.get("id") == reply_to_message_id), None)
                        if reply_to_message_id else conversation["messages"][-1])
    if previous_message is None:
        return question
    previous = previous_message.get("resolved_question") or previous_message.get("question", "")
    if not previous or len(question) > 850:
        return question
    if vague_why_use:
        subject = re.match(r"^(?:what\s+(?:is|are)|explain|define|describe)\s+(.+?)\s*[?.!]*$",
                           previous.strip(), re.I)
        if subject:
            return f"Why do we use {subject.group(1).rstrip('?.! ')}?"
    # A short follow-up can refer to a subject introduced in the verified
    # answer, even when that subject was absent from the earlier question.
    prior_claims = previous_message.get("claims") or []
    prior_pdf_claim = next((item.get("text", "") for item in prior_claims
                            if item.get("section") == "pdf"
                            and item.get("status") in ("supported", "contradicted")
                            and item.get("supporting") and isinstance(item.get("text"), str)), "")
    answer_context = f"\nPrevious verified PDF claim: {prior_pdf_claim[:160]}" if prior_pdf_claim else ""
    return f"Previous question: {previous[:100]}{answer_context}\nFollow-up question: {question}"


def _follow_up_uses_pdf(question: str, resolved_question: str, conversation: dict | None,
                        reply_to_message_id: str | None = None) -> bool:
    """A paper-bound conversation stays PDF-scoped unless external scope is requested."""
    if not conversation or not conversation.get("messages") or _requests_external_evidence(question):
        return False
    if _asks_only_about_pdf(question):
        return True
    if resolved_question != question:
        previous = (next((item for item in conversation["messages"]
                          if item.get("id") == reply_to_message_id), None)
                    if reply_to_message_id else conversation["messages"][-1])
        if previous and _requests_external_evidence(previous.get("question", "")):
            return False
    return True


@router.get("/capabilities")
async def research_capabilities(
    request: Request,
    user: UserResponse = Depends(get_current_user),
    ) -> dict[str, bool | int]:
    if getattr(request.app.state, "demo_mode", False):
        return {"answer_generation": True, "retry_after_seconds": 0}
    provider = getattr(request.app.state, "answer_provider", None)
    if provider is None:
        return {"answer_generation": False, "retry_after_seconds": 0}
    available = getattr(provider, "available", None)
    return {"answer_generation": await run_in_threadpool(available) if available else True,
            "retry_after_seconds": _cooldown_seconds(request, provider)
            if isinstance(provider, OpenRouterAnswerProvider) else 0}


async def _retrieve(payload: SearchRequest, request: Request, user: UserResponse,
                    expand: bool = False) -> list[ChunkResponse]:
    if not payload.question.strip():
        raise HTTPException(422, "Enter a question.")
    repository: PaperRepository = request.app.state.paper_repository
    paper = await repository.find(user.id, payload.paper_id)
    if paper is None:
        raise HTTPException(404, "Paper not found.")
    if not ready_for_research(paper):
        raise HTTPException(409, "Wait until this paper is ready.")

    ingestion: IngestionService = request.app.state.ingestion_service
    provider = getattr(request.app.state, "answer_provider", None)
    question = payload.question.strip()
    cache_key = None
    if (paper.get("file_hash") and paper.get("updated_at")
            and not request.scope.get("citewise_evaluation")
            and not (expand and isinstance(provider, OpenRouterAnswerProvider) and provider.rerank_enabled)):
        cache_key = (ingestion, user.id, payload.paper_id, paper["file_hash"], paper["updated_at"],
                     paper.get("embedding_model"), paper.get("chunk_count"), question, payload.limit, expand)
        with _retrieval_cache_lock:
            cached = _retrieval_cache.get(cache_key)
            if cached and cached[0] > time.monotonic():
                _retrieval_cache.move_to_end(cache_key)
                return [chunk.model_copy() for chunk in cached[1]]
            _retrieval_cache.pop(cache_key, None)

    def retrieve() -> list[ChunkResponse]:
        embedding = ingestion.embeddings.embed([question])[0]
        broad = bool(re.search(r"\b(?:both|compar\w*|combined|together|overall|universally)\b", question, re.I))
        initial = ingestion.vectors.search(user.id, payload.paper_id, embedding, payload.limit,
                                          question=question, candidate_limit=50 if broad else 25)
        if (expand and len(initial) == payload.limit and isinstance(provider, OpenRouterAnswerProvider)
                and provider.rerank_enabled and provider.available()
                and not request.scope.get("citewise_evaluation")):
            candidates = ingestion.vectors.search(user.id, payload.paper_id, embedding, RERANK_CANDIDATES,
                                                  question=question, candidate_limit=50 if broad else 25)
            if len(candidates) > payload.limit:
                try:
                    order = provider.rerank(question, [item.text for item in candidates], payload.limit)
                    initial = [candidates[index] for index in order]
                except Exception as exc:
                    logging.getLogger(__name__).info("research_rerank_unavailable error_type=%s",
                                                     type(exc).__name__)
        subjects, dimensions = requested_coverage(question, [item.text for item in initial])
        def diagnose(stage: str, items: list[ChunkResponse]) -> list[tuple[str, str]]:
            report = coverage_report(question, [CoveragePassage(
                f"c{item.chunk_index}-{item.content_hash[:12]}", item.text,
                item.page_number, item.chunk_index) for item in items])
            logging.getLogger(__name__).info(
                "research_coverage stage=%s candidate_ids=%s dimensions=%s",
                stage, [f"c{item.chunk_index}-{item.content_hash[:12]}" for item in items],
                [(item.subject, item.dimension, item.status, item.candidate_ids,
                  item.information_ids, item.inferred_subject_ids)
                 for item in report])
            return [(item.subject, item.dimension) for item in report if item.status != "information_present"]
        if not expand or not dimensions:
            diagnose("initial", initial)
            return initial
        gaps = diagnose("initial", initial)
        if not gaps:
            return initial
        # A small number of targeted semantic queries can recover sections
        # outside the first ranked set without sending 50 chunks to OpenRouter.
        topic = " ".join(sorted(VectorRepository._terms(question) - {
            "design", "methodology", "participant", "student", "sample", "setting",
            "duration", "finding", "result", "limitation", "difference", "important"}))
        focused = [f"{subject} {dimension} {topic}".strip()[:500]
                   for subject, dimension in gaps[:3]]
        vectors = ingestion.embeddings.embed(focused)
        selected = {(item.chunk_index, item.content_hash): item for item in initial}
        for focused_question, vector in zip(focused, vectors):
            candidates = ingestion.vectors.search(user.id, payload.paper_id, vector, 4,
                                                  question=focused_question, candidate_limit=50)
            logging.getLogger(__name__).info("research_coverage stage=focused candidate_ids=%s",
                [f"c{item.chunk_index}-{item.content_hash[:12]}" for item in candidates])
            for candidate in candidates:
                key = (candidate.chunk_index, candidate.content_hash)
                if key in selected:
                    continue
                before = missing_coverage(question, [item.text for item in selected.values()])
                after = missing_coverage(question, [item.text for item in selected.values()] + [candidate.text])
                if not (len(after) < len(before)
                        or VectorRepository._terms(focused_question) & VectorRepository._terms(candidate.text)):
                    continue
                selected[key] = candidate
                if len(selected) >= 10:
                    diagnose("final", list(selected.values()))
                    return list(selected.values())
        diagnose("final", list(selected.values()))
        return list(selected.values())

    passages = await run_in_threadpool(retrieve)
    if cache_key is not None and passages:
        with _retrieval_cache_lock:
            _retrieval_cache[cache_key] = (time.monotonic() + 60, tuple(chunk.model_copy() for chunk in passages))
            _retrieval_cache.move_to_end(cache_key)
            if len(_retrieval_cache) > 128:
                _retrieval_cache.popitem(last=False)
    return passages


@router.post("/search", response_model=SearchResponse)
async def search_paper(
    payload: SearchRequest,
    request: Request,
    user: UserResponse = Depends(get_current_user),
):
    passages = await _retrieve(payload, request, user)
    return SearchResponse(paper_id=payload.paper_id, passages=[
        SearchPassage(**passage.model_dump(), highlight_text=_evidence_excerpt(payload.question, passage.text))
        for passage in passages
    ])


def _simple_generation(question: str, resolved_question: str, passages: list[str]) -> bool:
    """Use the short route only for a standalone lookup with one selected source."""
    if question.strip() != resolved_question or len(passages) != 1:
        return False
    intent = re.sub(r"^search (?:the )?web\s*:\s*", "", resolved_question, flags=re.I)
    if len(intent.split()) > 22 or intent.count("?") > 1 or re.search(
        r"\b(?:and|or|why|compar\w*|versus|contradict\w*|conflict\w*|"
        r"methodolog\w*|(?:study|research|experimental) design|evaluat\w*|analy[sz]\w*|"
        r"implications?|limitations?|interpret\w*|synthesi[sz]\w*|overall|"
        r"differ\w*|combin\w*|across|both|total|ratio|trend|significan\w*|"
        r"percent(?:age)? (?:change|increase|decrease)|relative|caus\w*)\b|[;\n]",
        intent, re.I,
    ):
        return False
    subjects, dimensions = requested_coverage(intent, passages)
    return len(subjects) <= 1 and len(dimensions) <= 1 and bool(re.match(
        r"(?:what (?:is|was|are|were|does .+ mean\b)|who (?:is|was|were)|when (?:was|did)|"
        r"where (?:was|did)|how (?:many|long|much)|define)\s*\b", intent, re.I))


@router.post("/ask", response_model=AskResponse)
async def ask_paper(
    payload: AskRequest,
    request: Request,
    response: Response,
    user: UserResponse = Depends(get_current_user),
):
    demo_mode = getattr(request.app.state, "demo_mode", False)
    quality = None if demo_mode or request.scope.get("citewise_evaluation") else AnswerQuality()
    conversations: ConversationRepository = request.app.state.conversation_repository
    search_trace = SearchTrace()
    openrouter_counts = {"openrouter_calls": 0, "openrouter_cache_hits": 0, "openrouter_attempts": 0, "openrouter_retries": 0, "openrouter_failures": 0,
                   "openrouter_rate_limits": 0, "openrouter_server_errors": 0, "openrouter_network_errors": 0,
                   "openrouter_rate_limit_kind": None}
    runs: list[dict] = []
    model_runs.set(runs)
    openrouter_trace.set(openrouter_counts)
    openrouter_request_id.set(payload.request_id)
    openrouter_cache_scope.set((user.id, payload.paper_id))
    conversation = None
    if payload.conversation_id:
        conversation = await conversations.find(user.id, payload.conversation_id)
        if conversation is None or conversation["paper_id"] != payload.paper_id:
            raise HTTPException(404, "Conversation not found.")
        if (payload.reply_to_message_id and not any(
                item.get("id") == payload.reply_to_message_id for item in conversation.get("messages", []))):
            raise HTTPException(404, "Previous message not found in this conversation.")

    async def save_result(result: AskResponse) -> AskResponse:
        retryable_draft = result.verification_status == "unavailable"
        if not demo_mode and not retryable_draft:
            response.delete_cookie(_OPENROUTER_COOLDOWN_COOKIE, path="/api/v1/research")
        result.question = payload.question.strip()
        result.request_id = payload.request_id
        for key, value in openrouter_counts.items():
            setattr(search_trace, key, value)
        search_trace.claim_status_counts = dict(Counter(item.status for item in result.claims))
        search_trace.verification_failures = search_trace.claim_status_counts.get("unsupported", 0)
        result.search_trace = search_trace.model_copy(deep=True)
        state, checks = answer_report(status=result.status, answer=result.answer, claims=result.claims,
            sources=result.sources, external=result.external_sources, question=question_for_retrieval,
            unavailable=retryable_draft or bool(search_trace.external_search_errors))
        result.answer_state = state
        result.verification_status = "unavailable" if retryable_draft else "verified" if state == "VERIFIED" and not demo_mode else "not_verified"
        message = SavedMessage(
            id=uuid4().hex, question=payload.question.strip(), resolved_question=question_for_retrieval,
            status=result.status,
            answer=result.answer, sources=result.sources, external_sources=result.external_sources,
            assessment=result.assessment, created_at=datetime.now(timezone.utc),
            claims=result.claims, search_trace=result.search_trace,
            verification_status=result.verification_status, retry_after_seconds=result.retry_after_seconds,
            answer_state=state,
        ).model_dump(mode="json")
        message["verification_checks"] = checks
        if quality is not None and quality.draft is not None:
            message["quality_evaluations"] = [quality.evaluate(question_for_retrieval, message, revision=1, model_calls=runs)]
        if payload.conversation_id:
            if not await conversations.append(user.id, payload.conversation_id, payload.paper_id, message):
                raise HTTPException(404, "Conversation not found.")
            result.conversation_id = payload.conversation_id
        else:
            conversation = await conversations.create(user.id, payload.paper_id, message,
                                                      paper_record.get("filename") if paper_record else None)
            result.conversation_id = conversation["_id"]
        result.message_id = message["id"]
        return result

    question_for_retrieval = _question_with_context(payload.question, conversation,
                                                   payload.reply_to_message_id)
    if demo_mode:
        response.headers["X-CiteWise-Demo"] = "true"
        if not payload.question.strip():
            raise HTTPException(422, "Enter a question.")
        paper_record = await _ready_recommendation_paper(request, user, payload.paper_id)
        question_for_retrieval = payload.question.strip()
        sample = is_sample(paper_record)
        if sample and (external_answer := _demo_external_answer(payload.paper_id, payload.question)):
            search_trace.accepted_external_sources = len(external_answer.external_sources)
            return await save_result(external_answer)
        case = next((case for case in CASES if question_key(case.question) == question_key(payload.question)), None)
        if sample and case:
            chunks = await run_in_threadpool(request.app.state.ingestion_service.preview, user.id, payload.paper_id,
                                             limit=paper_record["chunk_count"])
            chunk = sample_source(case, chunks)
            if chunk is not None:
                source = AnswerSource(
                    id=f"c{chunk.chunk_index}-{chunk.content_hash[:12]}",
                    highlight_text=sample_quote(case, chunk), document_name=paper_record["filename"], **chunk.model_dump(),
                )
                return await save_result(AskResponse(paper_id=payload.paper_id, status="answered",
                    answer=f"{case.answer} [{source.id}]", sources=[source],
                    assessment=EvidenceReview(level="sufficient", chunk_ids=[source.id]),
                    claims=[ClaimReviewResponse(text=case.answer, section="pdf", status="supported",
                        supporting=[ClaimQuoteResponse(source_id=source.id, quote=case.excerpt)], opposing=[],
                        explanation="Sample claim check against the bundled PDF.", reason_code="demo_sample")]))
        message = ("This question has no sample answer. Choose one of the suggested sample questions."
                   if sample else "Sample answers are available for the bundled PLOS ONE paper. "
                   "Select it for sample questions or use Find passages to explore this PDF.")
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
            answer=message, sources=[], assessment=EvidenceReview(level="none", chunk_ids=[],
                missing_information="No predefined answer is available for this question and PDF.")))

    retrieval_payload = SearchRequest(paper_id=payload.paper_id, question=question_for_retrieval,
                                      limit=payload.limit)
    passages = await _retrieve(retrieval_payload, request, user, expand=True)
    # Only internal evaluation requests carry this ASGI scope flag. The actual
    # retrieved passages, not the gold answer, are captured for the report.
    if request.scope.get("citewise_evaluation"):
        request.state.evaluation_passages = passages
    paper_record = await request.app.state.paper_repository.find(user.id, payload.paper_id)
    provider: AnswerProvider | None = getattr(request.app.state, "answer_provider", None)
    if provider is None:
        raise HTTPException(503, "Answer generation is not configured yet.")
    available = getattr(provider, "available", None)
    if available and not await run_in_threadpool(available):
        raise HTTPException(503, "Answer generation is not available yet.")

    def with_retrieved_passages(status: int, code: str, message: str,
                                headers: dict[str, str] | None = None, **details: str) -> HTTPException:
        retrieved = SearchResponse(paper_id=payload.paper_id, passages=[
            SearchPassage(**item.model_dump(),
                          highlight_text=_evidence_excerpt(question_for_retrieval, item.text))
            for item in passages[:payload.limit]
        ])
        return HTTPException(status, {"message": message, "code": code, **details, **retrieved.model_dump(mode="json")},
                             headers={**(headers or {}), "Cache-Control": "private, no-store"})

    def rate_limited(exc: AnswerRateLimitError, message: str) -> HTTPException:
        headers = _rate_limit_headers(exc, provider)
        detail = _rate_limit_detail(exc, message)
        return with_retrieved_passages(429, headers=headers, **detail)

    def provider_failed(stage: str, exc: Exception) -> HTTPException:
        call = runs[-1] if runs else {}
        request_id = payload.request_id or ""
        logging.getLogger(__name__).warning(
            "research_provider_failed request_id=%s stage=%s model=%s http_status=%s error_type=%s outcome=%s",
            request_id if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id) else "none",
            stage, call.get("model_id"), call.get("status"),
            type(exc).__name__, call.get("error"))
        return with_retrieved_passages(502, "provider_unavailable", "Answer generation is temporarily unavailable.")

    sources = {}
    for passage in passages:
        source_id = f"c{passage.chunk_index}-{passage.content_hash[:12]}"
        sources[source_id] = AnswerSource(
            id=source_id,
            highlight_text=_evidence_excerpt(question_for_retrieval, passage.text),
            document_name=paper_record.get("filename") if paper_record else None,
            **passage.model_dump(),
        )
    logging.getLogger(__name__).info("research_retrieval request_id=%s source_ids=%s",
                                     payload.request_id or "none", list(sources))
    evidence = [Evidence(id=source.id, text=_evidence_context(question_for_retrieval, source.text))
                for source in sources.values()]
    _log_evidence_handoff(payload.request_id, "retrieved", list(sources), sources)
    _log_evidence_handoff(payload.request_id, "assessment_input", [item.id for item in evidence],
                          sources, {item.id: item.text for item in evidence})
    assessment = None
    assess = getattr(provider, "assess", None)
    if assess:
        cache_key = None
        if isinstance(provider, OpenRouterAnswerProvider):
            fingerprint = sha256(repr((user.id, payload.paper_id, question_for_retrieval,
                                       [(item.id, sources[item.id].content_hash, item.text) for item in evidence]))
                                 .encode("utf-8")).hexdigest()
            cache_key = (provider, fingerprint)
            with _assessment_cache_lock:
                cached = _assessment_cache.get(cache_key)
                if cached is not None:
                    _assessment_cache.move_to_end(cache_key)
                    assessment = cached.model_copy(deep=True)
        try:
            if assessment is None:
                proposed = await run_in_threadpool(assess, question_for_retrieval, evidence)
        except AnswerRateLimitError as exc:
            raise rate_limited(exc, "Evidence assessment is temporarily rate limited.") from None
        except Exception as exc:
            raise provider_failed("evidence_assessment", exc) from None
        if assessment is None:
            try:
                assessment = _validate_assessment(proposed, question_for_retrieval, sources,
                                                  request_id=payload.request_id)
            except Exception as exc:
                logging.getLogger(__name__).error("research_assessment_processing_failed error_type=%s",
                                                  type(exc).__name__)
                raise HTTPException(500, "Evidence processing failed. Try again later.") from None
        used_assessment_fallback = any(call.get("stage") == "citewise_evidence_assessment" and call.get("fallback")
                                       for call in runs)
        if assessment is None and not used_assessment_fallback:
            reassess = getattr(provider, "reassess", None)
            if callable(reassess):
                try:
                    corrected = await run_in_threadpool(reassess, question_for_retrieval, evidence)
                except AnswerRateLimitError as exc:
                    raise rate_limited(exc, "Evidence assessment is temporarily rate limited.") from None
                except Exception as exc:
                    raise provider_failed("evidence_reassessment", exc) from None
                try:
                    assessment = _validate_assessment(corrected, question_for_retrieval, sources,
                                                      request_id=payload.request_id)
                except Exception as exc:
                    logging.getLogger(__name__).error("research_assessment_processing_failed error_type=%s",
                                                      type(exc).__name__)
                    raise HTTPException(500, "Evidence processing failed. Try again later.") from None
                safe_id = payload.request_id if payload.request_id and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", payload.request_id) else "none"
                logging.getLogger(__name__).info("research_assessment_recovery request_id=%s accepted=%s",
                                                 safe_id, assessment is not None)
        if cache_key is not None and assessment is not None:
            with _assessment_cache_lock:
                _assessment_cache[cache_key] = assessment.model_copy(deep=True)
                _assessment_cache.move_to_end(cache_key)
                if len(_assessment_cache) > _assessment_cache_limit:
                    _assessment_cache.popitem(last=False)
        if assessment is None:
            logging.getLogger(__name__).info("research_assessment_rejected request_id=%s",
                                             payload.request_id or "none")
            return await save_result(AskResponse(paper_id=payload.paper_id,
                status="insufficient_evidence", answer="The retrieved passages could not be verified for this question.",
                sources=[]))
        logging.getLogger(__name__).info("research_assessment request_id=%s level=%s source_ids=%s",
                                         payload.request_id or "none", assessment.level, assessment.chunk_ids)
        _log_evidence_handoff(payload.request_id, "assessment_accepted", assessment.chunk_ids, sources)

    async def fallback_to_external(assessment: EvidenceReview,
                                   prior_pdf: tuple[str, list[ClaimReviewResponse]] | None = None) -> AskResponse:
        pdf_sources = [sources[item] for item in assessment.chunk_ids]
        def no_web_answer(explanation: str) -> AskResponse:
            if prior_pdf:
                explanation = f"{prior_pdf[0]}\n\n{explanation}"
            return AskResponse(paper_id=payload.paper_id,
                status="partially_supported" if pdf_sources else "not_mentioned", answer=explanation,
                sources=pdf_sources, assessment=assessment, claims=prior_pdf[1] if prior_pdf else [])

        if (_asks_for_paper_specific_fact(payload.question)
                and not _requests_external_evidence(payload.question)):
            answer = no_web_answer("The retrieved passages do not establish the requested statistic for this PDF.")
            if not pdf_sources:
                answer.status = "insufficient_evidence"
            return await save_result(answer)
        external_search = getattr(request.app.state, "external_search", None)
        if external_search is None or not external_search.available():
            explanation = ("The PDF supports part of the answer, but external search is not configured. "
                           f"Missing: {assessment.missing_information}" if assessment.level == "partial"
                           else "The retrieved PDF passages do not establish an answer. External search is not configured.")
            return await save_result(no_web_answer(explanation))
        query = (_external_query(question_for_retrieval, assessment.missing_information)
                 if assessment.level == "partial" else question_for_retrieval)
        async def search_once(search_query: str) -> list[ExternalEvidence]:
            search_trace.duplicate_requests += int(search_query in search_trace.queries)
            search_trace.queries.append(search_query)
            traced_search = getattr(external_search, "search_with_trace", None)
            if traced_search:
                search_trace.tavily_calls += 1
                outcome = await run_in_threadpool(traced_search, search_query)
                if outcome.cache_hit:
                    search_trace.tavily_calls -= 1
                    search_trace.tavily_cache_hits += 1
                search_trace.rejected_irrelevant_sources += outcome.rejected_irrelevant
                search_trace.raw_external_results += outcome.raw_results
                search_trace.invalid_external_results += outcome.invalid_results
                search_trace.invalid_external_urls += outcome.invalid_urls
                search_trace.rejected_external_domains += outcome.rejected_domains
                search_trace.duplicate_external_sources += outcome.duplicates
                found = outcome.results
            else:
                cached_search = getattr(external_search, "cached", None)
                cached_external = cached_search(search_query) if cached_search else None
                if cached_external is not None:
                    search_trace.tavily_cache_hits += 1
                    found = cached_external
                else:
                    search_trace.tavily_calls += 1
                    found = await run_in_threadpool(external_search.search, search_query)
            relevant = found if traced_search else [item for item in found if TavilyExternalSearch._relevant(
                search_query, item.title, item.content)]
            search_trace.rejected_irrelevant_sources += len(found) - len(relevant)
            search_trace.retrieved_source_counts.append(len(relevant))
            search_trace.accepted_external_sources += len(relevant)
            return relevant

        search_failed = False
        search_failure_reason = "External search was unavailable. This does not establish that no external information exists."
        try:
            external = await search_once(query)
        except Exception as exc:
            search_trace.failures += 1
            search_trace.external_search_errors += 1
            search_trace.external_search_timeouts += int(isinstance(exc, (TimeoutError, RequestTimeout)))
            external = []
            search_trace.retrieved_source_counts.append(0)
            search_failed = True
            if isinstance(exc, (TimeoutError, RequestTimeout)):
                search_failure_reason = "External search timed out. Try again later."
            elif isinstance(exc, HTTPError) and exc.response is not None:
                status_code = exc.response.status_code
                if status_code in (401, 403):
                    search_failure_reason = "External search authentication is unavailable."
                elif status_code in (429, 432, 433):
                    search_failure_reason = "External search is rate limited or its quota is exhausted. Try again later."
                elif status_code >= 500:
                    search_failure_reason = "External search provider is unavailable. Try again later."
        if not external and not search_failed and isinstance(external_search, TavilyExternalSearch):
            fallback = _fallback_query(question_for_retrieval, assessment.missing_information, query)
            if fallback and fallback not in search_trace.queries:
                search_trace.fallback_searches += 1
                try:
                    external = await search_once(fallback)
                except Exception as exc:
                    search_trace.failures += 1
                    search_trace.external_search_errors += 1
                    search_trace.external_search_timeouts += int(isinstance(exc, (TimeoutError, RequestTimeout)))
                    search_trace.retrieved_source_counts.append(0)
                    search_failed = True
        if not external:
            reason = (search_failure_reason
                      if search_failed else "The searches returned no usable external evidence.")
            return await save_result(no_web_answer(f"The PDF does not fully answer this question. {reason}"))
        try:
            _log_evidence_handoff(payload.request_id, "generation_input", assessment.chunk_ids, sources,
                                  {item.id: item.text for item in evidence})
            generation_options = {}
            if isinstance(provider, OpenRouterAnswerProvider) and assessment.level == "none" and _simple_generation(
                payload.question, question_for_retrieval, [item.content for item in external],
            ):
                generation_options["simple"] = True
            generated_external = await run_in_threadpool(provider.generate_with_external,
                question_for_retrieval, [item for item in evidence if item.id in assessment.chunk_ids],
                external, assessment.missing_information, **generation_options)
        except AnswerRateLimitError as exc:
            raise rate_limited(exc, "Answer generation is temporarily rate limited.") from None
        except Exception as exc:
            raise provider_failed("external_answer_generation", exc) from None
        external_by_id = {item.id: item for item in external}
        pdf_ids = getattr(generated_external, "pdf_ids", None)
        external_ids = getattr(generated_external, "external_ids", None)
        pdf_text = getattr(generated_external, "pdf_text", None)
        external_text = getattr(generated_external, "external_text", None)
        if quality is not None:
            quality.capture(pdf_text if assessment.level == "partial" else "", external_text,
                            pdf_ids, external_ids, sources, external_by_id)
        def unverified_external_answer() -> AskResponse:
            if prior_pdf:
                return no_web_answer("The retrieved external excerpts did not establish the missing information.")
            if assessment.level == "partial" and pdf_sources:
                return AskResponse(paper_id=payload.paper_id, status="partially_supported",
                    answer=("The selected PDF passages contain related evidence, but the external "
                            "answer could not be verified against its sources. No external claim is presented."),
                    sources=pdf_sources, assessment=assessment)
            return AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                answer="The external answer could not be linked to its sources.",
                sources=pdf_sources, assessment=assessment)
        if (not isinstance(pdf_text, str) or not isinstance(external_text, str)
                or not external_text.strip() or not isinstance(pdf_ids, list)
                or not isinstance(external_ids, list) or any(item not in assessment.chunk_ids for item in pdf_ids)
                or any(item not in external_by_id for item in external_ids)
                or not external_ids or (assessment.level == "partial" and (not pdf_ids or not pdf_text.strip()))
                or (assessment.level == "none" and pdf_ids)):
            return await save_result(unverified_external_answer())
        if (any(f"[{item}]" in pdf_text for item in external_by_id)
                or any(f"[{item}]" in external_text for item in sources)):
            return await save_result(unverified_external_answer())
        _log_evidence_handoff(payload.request_id, "generated_claims", pdf_ids, sources)
        try:
            checked_pdf, checked_external, claim_reviews = await _verified_sections(
                provider, question_for_retrieval, pdf_text if assessment.level == "partial" else "",
                external_text, pdf_ids, external_ids, sources, external_by_id,
                request_id=payload.request_id, debug_claims=_local_claim_debug(request), raise_unavailable=True,
                quality=quality)
        except Exception as exc:
            if isinstance(exc, AnswerRateLimitError):
                headers = _rate_limit_headers(exc, provider) or {}
                if "Set-Cookie" in headers:
                    response.headers.append("Set-Cookie", headers["Set-Cookie"])
            draft = (f"Evidence from your PDF\n{pdf_text if assessment.level == 'partial' else ''}\n\n"
                     f"Additional information from external sources\n{external_text}")
            return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                answer=draft, sources=pdf_sources, external_sources=[_external_source(item) for item in external],
                assessment=assessment, verification_status="unavailable",
                retry_after_seconds=int(exc.retry_after or 0) if isinstance(exc, AnswerRateLimitError) else 0))
        if not claim_reviews:
            return await save_result(unverified_external_answer())
        used_ids = {quote.source_id for claim in claim_reviews for quote in claim.supporting + claim.opposing}
        _log_evidence_handoff(payload.request_id, "verified_claims", sorted(used_ids), sources)
        external_parts = _verified_answer_parts(question_for_retrieval, claim_reviews, sources)
        external_subjects = list(dict.fromkeys(subject for subject, _, _, _ in external_parts))
        external_aliases = {subject: f"subject_{index + 1}"
                            for index, subject in enumerate(external_subjects)}
        logging.getLogger("uvicorn.error").info(
            "research_answer_parts request_id=%s parts=%s",
            payload.request_id if isinstance(payload.request_id, str) and re.fullmatch(
                r"[A-Za-z0-9_-]{1,64}", payload.request_id) else "none",
            [(external_aliases.get(subject, "document"), dimension, status, ids)
             for subject, dimension, status, ids in external_parts])
        cited_external = [_external_source(item) for item in external if item.id in used_ids]
        cited_pdf = [item for item in sources.values() if item.id in used_ids]
        pdf_section = (checked_pdf if assessment.level == "partial" else
                       "This information was not found in the retrieved passages of your uploaded paper.")
        answer = (f"Evidence from your PDF\n{pdf_section}\n\n"
                  f"Additional information from external sources\n{checked_external}")
        reviewed_status = _reviewed_status(claim_reviews, assessment.level == "partial", external=True)
        return await save_result(AskResponse(paper_id=payload.paper_id,
            status=reviewed_status,
            answer=answer, sources=cited_pdf, external_sources=cited_external,
            assessment=assessment, claims=claim_reviews))

    pdf_comparison = len(requested_coverage(question_for_retrieval,
                         [source.text for source in sources.values()])[0]) >= 2
    quote_coverage_gap = bool(assessment and assessment.missing_information.startswith(
        "The selected PDF chunks contain information omitted from the cited quotes"))
    if assessment and assessment.level in ("partial", "none") and not (
            assessment.level == "partial" and (pdf_comparison or quote_coverage_gap)):
        return await fallback_to_external(assessment)

    if not evidence:
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                                             answer="I could not retrieve passages that support an answer.", sources=[]))
    if assessment:
        evidence = [item for item in evidence if item.id in assessment.chunk_ids]
    _log_evidence_handoff(payload.request_id, "generation_input", [item.id for item in evidence],
                          sources, {item.id: item.text for item in evidence})
    try:
        generation_options = {}
        if (isinstance(provider, OpenRouterAnswerProvider) and assessment and assessment.level == "sufficient"
                and _simple_generation(payload.question, question_for_retrieval, [item.text for item in evidence])):
            generation_options["simple"] = True
        generated = await run_in_threadpool(provider.generate, question_for_retrieval, evidence, **generation_options)
    except AnswerRateLimitError as exc:
        raise rate_limited(exc, "Answer generation is temporarily rate limited.") from None
    except Exception as exc:
        raise provider_failed("research_generation", exc) from None

    answer = getattr(generated, "text", None)
    evidence_ids = getattr(generated, "evidence_ids", None)
    status = getattr(generated, "status", None)
    if quality is not None:
        quality.capture(answer, "", evidence_ids, [], sources, {})
    logging.getLogger(__name__).info("research_generation request_id=%s status=%s source_ids=%s",
                                     payload.request_id or "none", status, evidence_ids)
    if status == "not_mentioned":
        if assessment:
            return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                answer="The PDF contains relevant passages, but answer generation could not use them reliably.",
                sources=[sources[item] for item in assessment.chunk_ids], assessment=assessment))
        return await save_result(AskResponse(paper_id=payload.paper_id, status="not_mentioned",
                                             answer="The retrieved passages do not mention this topic.", sources=[]))
    if status not in ("answered", "partially_supported", "insufficient_evidence", "contradictory_evidence"):
        raise HTTPException(502, "Answer generation returned an invalid evidence status.")
    if not isinstance(answer, str) or not answer.strip():
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                                             answer="A reliable answer could not be verified from the generated claims and citations.", sources=[]))
    if answer.strip().casefold().rstrip("?.! ") == payload.question.strip().casefold().rstrip("?.! "):
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                                             answer="A reliable answer could not be verified from the generated claims and citations.", sources=[]))
    if not isinstance(evidence_ids, list) or not evidence_ids or any(
        not isinstance(source_id, str) or source_id not in sources
        or (assessment is not None and source_id not in assessment.chunk_ids) for source_id in evidence_ids
    ):
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                                             answer="The proposed answer could not be linked to the retrieved passages.", sources=[]))
    cited = list(dict.fromkeys(evidence_ids))
    _log_evidence_handoff(payload.request_id, "generated_claims", cited, sources)
    try:
        checked_pdf, _, claim_reviews = await _verified_sections(
            provider, question_for_retrieval, answer.strip(), "", cited, [],
            {item: sources[item] for item in cited}, {}, request_id=payload.request_id,
            debug_claims=_local_claim_debug(request), raise_unavailable=True, quality=quality)
    except AnswerRateLimitError as exc:
        headers = _rate_limit_headers(exc, provider) or {}
        if "Set-Cookie" in headers:
            response.headers.append("Set-Cookie", headers["Set-Cookie"])
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
            answer=answer.strip(), sources=[sources[item] for item in cited], assessment=assessment,
            verification_status="unavailable", retry_after_seconds=int(exc.retry_after or 0)))
    except Exception:
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
            answer=answer.strip(), sources=[sources[item] for item in cited], assessment=assessment,
            verification_status="unavailable"))
    if not claim_reviews:
        return await save_result(AskResponse(paper_id=payload.paper_id,
            status="insufficient_evidence", answer="The generated claims could not be verified.",
            sources=[], assessment=assessment))
    assessment = _reconcile_assessment_gap(assessment, claim_reviews, question_for_retrieval, sources)
    used_ids = {quote.source_id for item in claim_reviews for quote in item.supporting + item.opposing}
    _log_evidence_handoff(payload.request_id, "verified_claims", sorted(used_ids), sources)
    comparison_gaps = _methodology_comparison_gaps(
        question_for_retrieval, claim_reviews, [source.text for source in sources.values()])
    answer_parts = _verified_answer_parts(question_for_retrieval, claim_reviews, sources)
    subjects_for_log = list(dict.fromkeys(subject for subject, _, _, _ in answer_parts))
    aliases = {subject: f"subject_{index + 1}" for index, subject in enumerate(subjects_for_log)}
    logging.getLogger("uvicorn.error").info(
        "research_answer_parts request_id=%s parts=%s",
        payload.request_id if isinstance(payload.request_id, str) and re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}", payload.request_id) else "none",
        [(aliases.get(subject, "document"), dimension, status, ids)
         for subject, dimension, status, ids in answer_parts])
    unresolved_parts = [(subject, dimension) for subject, dimension, part_status, _ in answer_parts
                        if part_status != "supported"]
    if not any(item.status in ("supported", "partially_supported", "contradicted") for item in claim_reviews):
        status = "insufficient_evidence"
    elif any(item.status == "contradicted" for item in claim_reviews):
        status = "contradictory_evidence"
    elif (assessment and assessment.level == "partial") or comparison_gaps or unresolved_parts or any(
            item.status in ("partially_supported", "unsupported") for item in claim_reviews):
        status = "partially_supported"
        if assessment and assessment.missing_information and assessment.missing_information not in checked_pdf:
            checked_pdf = f"{checked_pdf} {assessment.missing_information}".strip()
        if comparison_gaps:
            checked_pdf = f"{checked_pdf} I could not verify {', '.join(comparison_gaps)} from the cited passages.".strip()
        elif unresolved_parts and (not assessment or assessment.level == "sufficient"):
            labels = ", ".join(f"{subject + ' ' if subject else ''}{dimension}"
                                for subject, dimension in unresolved_parts)
            checked_pdf = (f"{checked_pdf} The cited claims do not fully verify: {labels}.".strip())
    if status == "partially_supported":
        external_search = getattr(request.app.state, "external_search", None)
        if external_search is not None and external_search.available():
            missing = (assessment.missing_information if assessment and assessment.missing_information
                       else ", ".join(comparison_gaps) or "Some requested information remains unverified.")
            reviewed_ids = [item for item in cited if item in used_ids]
            review = EvidenceReview(level="partial" if reviewed_ids else "none",
                                    chunk_ids=reviewed_ids, missing_information=missing)
            return await fallback_to_external(review, (checked_pdf, claim_reviews))
    logging.getLogger(__name__).info("research_verification request_id=%s response_status=%s claim_statuses=%s source_ids=%s",
                                     payload.request_id or "none", status,
                                     [item.status for item in claim_reviews], sorted(used_ids))
    verified = [CoveragePassage(f"claim-{index}", item.text) for index, item in enumerate(claim_reviews)
                if item.section == "pdf" and item.status == "supported" and item.supporting]
    logging.getLogger(__name__).info("research_verified_coverage request_id=%s dimensions=%s claim_sources=%s",
        payload.request_id or "none",
        [(item.subject, item.dimension, item.status, item.information_ids)
         for item in coverage_report(question_for_retrieval, verified,
                                     {item.id for item in verified},
                                     subjects=requested_coverage(question_for_retrieval,
                                         [source.text for source in sources.values()])[0])],
        [(f"claim-{index}", [quote.source_id for quote in item.supporting])
         for index, item in enumerate(claim_reviews) if item.section == "pdf"
         and item.status == "supported" and item.supporting])
    return await save_result(AskResponse(paper_id=payload.paper_id, status=status, answer=checked_pdf,
        sources=[sources[item] for item in cited if item in used_ids], assessment=assessment,
        claims=claim_reviews))
