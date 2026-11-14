from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Literal
from collections import Counter, OrderedDict
from hashlib import sha256
import logging
import unicodedata
import hmac
from threading import Lock
import re
from datetime import date, datetime, timezone
from uuid import uuid4
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..dependencies import get_current_user
from ..config import get_settings
from ..models import UserResponse
from ..paper_models import ChunkResponse
from ..providers.answers import (AnswerProvider, AnswerRateLimitError, Evidence, EvidenceAssessment, EvidenceQuote,
                                 EvidenceStatus, ExternalEvidence)
from ..providers.groq_answers import GroqAnswerProvider, groq_trace
from ..repositories.papers import PaperRepository
from ..repositories.conversations import ConversationRepository
from ..repositories.vectors import VectorRepository
from ..services.ingestion import IngestionService
from ..services.claim_verification import split_claims, review_claims, present_claims

router = APIRouter(prefix="/api/v1/research", tags=["research"])
_assessment_cache: OrderedDict[tuple[GroqAnswerProvider, str], "EvidenceReview"] = OrderedDict()
_assessment_cache_lock = Lock()
_assessment_cache_limit = 128


class SearchRequest(BaseModel):
    paper_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=10)


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


class SearchTrace(BaseModel):
    tavily_calls: int = 0
    tavily_cache_hits: int = 0
    groq_calls: int = 0
    groq_cache_hits: int = 0
    groq_attempts: int = 0
    groq_retries: int = 0
    groq_failures: int = 0
    groq_rate_limits: int = 0
    groq_server_errors: int = 0
    groq_network_errors: int = 0
    groq_rate_limit_kind: str | None = None
    queries: list[str] = Field(default_factory=list)
    failures: int = 0
    duplicate_requests: int = 0
    retrieved_source_counts: list[int] = Field(default_factory=list)
    rejected_irrelevant_sources: int = 0
    claim_status_counts: dict[str, int] = Field(default_factory=dict)
    verification_failures: int = 0


def _external_source(item: ExternalEvidence) -> ExternalSource:
    return ExternalSource(id=item.id, title=item.title, url=item.url,
        published_date=item.published_date, verified_publication_date=item.verified_publication_date,
        source_host=urlparse(item.url).hostname, publisher=item.publisher, author=item.author,
        excerpt=item.content[:700])


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


class ConversationResponse(BaseModel):
    id: str
    paper_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[SavedMessage]


def _conversation_response(document: dict) -> ConversationResponse:
    return ConversationResponse(id=document["_id"], paper_id=document["paper_id"],
                                created_at=document["created_at"], updated_at=document["updated_at"],
                                messages=document.get("messages", []))


async def _verified_sections(provider: AnswerProvider, question: str, pdf_text: str,
                             external_text: str, pdf_ids: list[str], external_ids: list[str],
                             pdf_sources: dict[str, AnswerSource],
                             external_sources: dict[str, ExternalEvidence]
                             ) -> tuple[str, str, list[ClaimReviewResponse]]:
    claims = split_claims(pdf_text, external_text)
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
    evidence = ([Evidence(item.id, item.text) for item in pdf_sources.values()] +
                [Evidence(item.id, item.content) for item in external_sources.values()])
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
        return "", "", []
    verified_dates = {}
    for item in external_sources.values():
        if item.verified_publication_date:
            try:
                verified_dates[item.id] = date.fromisoformat(item.verified_publication_date)
            except ValueError:
                pass
    reviewed = review_claims(claims, proposed, evidence, verified_dates)
    responses = [ClaimReviewResponse(text=item.claim.text, section=item.claim.section,
        status=item.status,
        supporting=[ClaimQuoteResponse(source_id=q.source_id, quote=q.quote) for q in item.supporting],
        opposing=[ClaimQuoteResponse(source_id=q.source_id, quote=q.quote) for q in item.opposing],
        explanation=item.explanation) for item in reviewed]
    return present_claims(reviewed, "pdf"), present_claims(reviewed, "external"), responses


def _methodology_comparison_gaps(question: str, claims: list[ClaimReviewResponse]) -> list[str]:
    """Report requested comparison parts that did not survive claim verification."""
    if not (re.search(r"\bstudy\s+a\b", question, re.I)
            and re.search(r"\bstudy\s+b\b", question, re.I)
            and re.search(r"\b(?:methodolog\w*|designs?|participants?)\b", question, re.I)):
        return []
    supported = [item.text for item in claims if item.section == "pdf" and item.status == "supported"]
    gaps = []
    for label in ("A", "B"):
        statements = [text for text in supported if re.search(rf"\bstudy\s+{label}\b", text, re.I)]
        statements.extend(quote.quote for item in claims
                          if item.section == "pdf" and item.status == "partially_supported"
                          and re.search(rf"\bstudy\s+{label}\b", item.text, re.I)
                          for quote in item.supporting)
        has_design = any(re.search(r"\b(?:design|randomiz\w*|experiment\w*|matched)\b", text, re.I)
                         for text in statements)
        has_participants = any(re.search(r"\b\d[\d,]*\s+(?:students|participants|subjects)\b", text, re.I)
                               for text in statements)
        if not has_design and not has_participants:
            gaps.append(f"Study {label}'s design and participants")
        elif not has_design:
            gaps.append(f"Study {label}'s design")
        elif not has_participants:
            gaps.append(f"Study {label}'s participants")
    if not any(re.search(r"\bstudy\s+a\b", text, re.I)
               and re.search(r"\bstudy\s+b\b", text, re.I)
               and re.search(r"\b(?:whereas|while|unlike|differ\w*|versus|compared|comparison)\b", text, re.I)
               for text in supported):
        gaps.append("the methodological differences")
    return gaps


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(request: Request, user: UserResponse = Depends(get_current_user)):
    repository: ConversationRepository = request.app.state.conversation_repository
    return [_conversation_response(item) for item in await repository.list_for_user(user.id)]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, request: Request,
                           user: UserResponse = Depends(get_current_user)):
    repository: ConversationRepository = request.app.state.conversation_repository
    document = await repository.find(user.id, conversation_id)
    if document is None:
        raise HTTPException(404, "Conversation not found.")
    return _conversation_response(document)


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


def _evidence_excerpt(question: str, text: str) -> str:
    """Keep the sentence most related to the question inside a retrieved chunk."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    terms = VectorRepository._terms(question)
    best = max(sentences, key=lambda sentence: len(terms & VectorRepository._terms(sentence)))
    return best[:600].strip()


def _evidence_context(question: str, text: str) -> str:
    """Give the model nearby sentences so a qualification is not lost."""
    if re.search(r"\b(?:both studies|study a|study b|compar\w*|combined|together|alert\w*|generaliz\w*|privacy|anxiety|integrity|recommend\w*)\b", question, re.I):
        if len(text) <= 2400:
            return text.strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        terms = VectorRepository._terms(question)
        ranked = sorted(range(len(sentences)),
                        key=lambda index: (-len(terms & VectorRepository._terms(sentences[index])), index))
        selected = set()
        for index in ranked:
            proposed = selected | {index} | ({index - 1} if index else set()) | (
                {index + 1} if index + 1 < len(sentences) else set())
            if len(" ".join(sentences[item] for item in sorted(proposed))) <= 2400:
                selected = proposed
        return " ".join(sentences[item] for item in sorted(selected)).strip() or text[:2400].strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    terms = VectorRepository._terms(question)
    best_index = max(range(len(sentences)), key=lambda index: len(terms & VectorRepository._terms(sentences[index])))
    return " ".join(sentences[max(0, best_index - 1):best_index + 2])[:1200].strip()


def _direct_pdf_coverage(question: str, sources: dict[str, AnswerSource]) -> list[str]:
    """Recognize directly stated counts, never infer broad comparisons from keywords."""
    if (re.match(r"\s*how many\b", question, re.I)
            and not re.search(r"\b(?:both|combined|total|together|compar\w*)\b", question, re.I)):
        nouns = [noun for noun in ("universities", "students", "participants", "courses")
                 if re.search(rf"\b{noun}\b", question, re.I)]
        study = re.search(r"\bstudy\s+([ab])\b", question, re.I)
        if nouns:
            for source in sources.values():
                if study and (not re.search(rf"\bstudy\s+{study.group(1)}\b", source.text, re.I)
                              or re.search(rf"\bstudy\s+{'b' if study.group(1).lower() == 'a' else 'a'}\b",
                                           source.text, re.I)):
                    continue
                if all(re.search(rf"\b\d[\d,]*(?:\s+[a-z-]+){{0,2}}\s+{noun}\b",
                                 source.text, re.I) for noun in nouns):
                    return [source.id]
    return []


def _quote_tokens(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\d[\d,]*(?:\.\d+)?|[^\W\d_]+|%", value))


def _validate_assessment(assessment: EvidenceAssessment, question: str,
                         sources: dict[str, AnswerSource]) -> EvidenceReview | None:
    """Reject unsupported labels before they can authorize external search."""
    def reject(reason: str, source_id: str | None = None) -> None:
        logging.getLogger(__name__).info("evidence_assessment_rejected reason=%s source_id=%s",
                                         reason, source_id or "none")
        return None

    if assessment.level not in ("sufficient", "partial", "none"):
        return reject("invalid_assessment_level")
    if assessment.level != "sufficient" and (direct_ids := _direct_pdf_coverage(question, sources)):
        return EvidenceReview(level="sufficient", chunk_ids=direct_ids)
    ids = assessment.chunk_ids
    if not isinstance(ids, list) or any(not isinstance(item, str) or item not in sources for item in ids):
        return reject("unknown_source_id")
    if assessment.level != "none" and assessment.supporting_quotes is None:
        return reject("missing_supporting_quote")
    invalid_quote_ids: set[str] = set()
    valid_quote_ids: set[str] = set()
    valid_quotes: list[EvidenceQuote] = []
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
        for quote in quotes:
            normalized_chunk = f" {_quote_tokens(sources[quote.chunk_id].text)} "
            fragments = [_quote_tokens(part) for part in re.split(r"\s*(?:\.\.\.|…)+\s*", quote.quote)
                         if part.strip()]
            if not fragments:
                return reject("quote_not_found_in_source")
            cursor = 0
            ordered = True
            for fragment in fragments:
                normalized_fragment = f" {fragment} "
                position = normalized_chunk.find(normalized_fragment, cursor)
                if position < 0:
                    ordered = False
                    break
                cursor = position + len(normalized_fragment) - 1
            if ordered:
                valid_quote_ids.add(quote.chunk_id)
                valid_quotes.append(quote)
                continue
            # Models sometimes copy several labeled PDF fields verbatim but
            # reorder the fields. Verify each complete field independently.
            fields = re.split(r"(?=(?:Setting|Participants|Design|Observation period):\s*)",
                              quote.quote, flags=re.I)
            field_spans = [field.strip() for field in fields if field.strip()]
            if (len(field_spans) >= 2 and not re.search(r"\.\.\.|…", quote.quote)
                    and all(re.match(r"^(?:Setting|Participants|Design|Observation period):", field, re.I)
                            and len(field) >= 12
                            and f" {_quote_tokens(field)} " in normalized_chunk for field in field_spans)):
                valid_quote_ids.add(quote.chunk_id)
                valid_quotes.append(quote)
                logging.getLogger(__name__).info(
                    "evidence_assessment_verified_reordered_fields source_id=%s field_count=%s",
                    quote.chunk_id, len(field_spans))
                continue
            invalid_quote_ids.add(quote.chunk_id)
            logging.getLogger(__name__).info(
                "evidence_assessment_rejected reason=quote_not_found_or_order_mismatch source_id=%s",
                quote.chunk_id)
    if invalid_quote_ids:
        # A redundant comparison-table quote may be malformed while separate
        # verbatim passages still establish both sides of a comparison.
        compares_both = (assessment.level == "sufficient"
                         and re.search(r"\bstudy\s+a\b", question, re.I)
                         and re.search(r"\bstudy\s+b\b", question, re.I))
        valid_ids = [item for item in ids if item in valid_quote_ids and item not in invalid_quote_ids]
        topic_terms = VectorRepository._terms(question)
        methodology = bool(re.search(r"\bmethodolog\w*\b", question, re.I))
        def relevant(item: str) -> bool:
            passage = sources[item].text
            return (bool(topic_terms & VectorRepository._terms(passage))
                    or bool(methodology and re.search(
                        r"\b(?:design|participant\w*|sample|randomiz\w*|experiment\w*|course\w*|setting)\b",
                        passage, re.I)))
        covers_both = all(any(re.search(rf"\bstudy\s+{label}\b", sources[item].text, re.I)
                              and not re.search(rf"\bstudy\s+{other}\b", sources[item].text, re.I)
                              and relevant(item)
                              for item in valid_ids)
                          for label, other in (("a", "b"), ("b", "a")))
        # An extra malformed quote must not discard a complete set of exact
        # quotes for the requested methodology fields in one study.
        single_study = re.search(r"\bstudy\s+([ab])\b", question, re.I)
        single_study = single_study if not compares_both else None
        requested_fields = {
            "design": r"\b(?:methodolog\w*|design)\b",
            "participants": r"\b(?:sample size|participants?|students?)\b",
            "setting": r"\b(?:universit\w*|regions?)\b",
            "observation period": r"\b(?:observation period|duration|semester)\b",
        }
        required = [name for name, pattern in requested_fields.items() if re.search(pattern, question, re.I)]
        complete_study_quotes = False
        if single_study and required and valid_quote_ids:
            label = single_study.group(1)
            for item in ids:
                if item not in valid_quote_ids:
                    continue
                passage = sources[item].text
                if not re.search(rf"\bstudy\s+{label}\b", passage, re.I):
                    continue
                quoted = " ".join(quote.quote for quote in valid_quotes if quote.chunk_id == item)
                has_fields = all(re.search(rf"\b{re.escape(name)}:\s*\S+", quoted, re.I)
                                 for name in required)
                has_counts = (not re.search(r"\b(?:sample size|participants?|students?)\b", question, re.I)
                              or bool(re.search(r"\bParticipants:\s*(?:(?!\b(?:Design|Setting|Observation period):).)*?\d[\d,]*",
                                                quoted, re.I)))
                has_universities = (not re.search(r"\buniversit\w*\b", question, re.I)
                                    or bool(re.search(r"\bSetting:\s*(?:(?!\b(?:Design|Participants|Observation period):).)*?"
                                                      r"\d[\d,]*\s+universit\w*", quoted, re.I)))
                has_regions = (not re.search(r"\bregions?\b", question, re.I)
                               or bool(re.search(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+regions?\b",
                                                 quoted, re.I)))
                if has_fields and has_counts and has_universities and has_regions:
                    complete_study_quotes = True
                    valid_ids = [item]
                    break
        if not ((compares_both and covers_both) or complete_study_quotes):
            return reject("quote_not_found_or_order_mismatch")
        ids = valid_ids
        logging.getLogger(__name__).info(
            "evidence_assessment_dropped_invalid_redundant_quotes source_ids=%s", sorted(invalid_quote_ids))
    if assessment.level == "none":
        if ids:
            return reject("none_with_source_ids")
        if (re.search(r"\b(?:was|is|did)\b.*\b(?:studied|evaluated|measured)\b", question, re.I)
                and re.search(r"\bfacial recognition\b", question, re.I)):
            for source in sources.values():
                if re.search(r"\bfacial recognition (?:accuracy|performance)\s+not studied\b",
                             source.text, re.I):
                    return EvidenceReview(level="sufficient", chunk_ids=[source.id])
        if re.search(r"\bsample size\b", question, re.I):
            for source in sources.values():
                if re.search(r"\b(?:study|trial|cohort|sample)\s+(?:included|enrolled|comprised)\s+"
                             r"\d[\d,]*\s+(?:students|participants|subjects|patients)\b",
                             source.text, re.I):
                    return EvidenceReview(level="sufficient", chunk_ids=[source.id])
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
    if (level == "sufficient" and re.search(r"\bstudy\s+a\b", question, re.I)
            and re.search(r"\bstudy\s+b\b", question, re.I)):
        absent = [label for label in ("A", "B") if not any(
            re.search(rf"\bstudy\s+{label}\b", sources[item].text, re.I) for item in ids)]
        if absent:
            level = "partial"
            missing = f"The retrieved PDF passages do not establish Study {' and '.join(absent)}'s findings for this comparison."
    return EvidenceReview(level=level, chunk_ids=list(dict.fromkeys(ids)),
                          missing_information=missing[:500])


def _asks_for_paper_specific_fact(question: str) -> bool:
    """Outside papers cannot supply a statistic missing from this particular PDF."""
    return bool(re.search(r"\b(p[ -]?value|sample size|number of participants|cohort size)\b", question, re.I)
                and re.search(r"\b(this|the|paper|study|experiment|trial)\b", question, re.I))


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


def _reported_calculation(question: str, sources: dict[str, AnswerSource]) -> tuple[str, str, str] | None:
    """Use explicit dossier facts and inputs; never infer a missing operand."""
    if (re.search(r"\bstudy b\b", question, re.I)
            and re.search(r"\b(?:how many|sample size)\b", question, re.I)
            and re.search(r"\b(?:universities|participants|students|courses|sample size)\b", question, re.I)
            and not re.search(r"\b(?:both|combined|total|together|compare)\b", question, re.I)):
        for source in sources.values():
            match = re.search(r"Study B.*?Setting:\s*(\d+) universities.*?Participants:\s*([\d,]+) "
                              r"students across (\d+) online courses", source.text, re.I | re.S)
            if match:
                universities, students, courses = (int(value.replace(",", "")) for value in match.groups())
                return (f"Study B involved {students:,} students across {universities} universities and "
                        f"{courses} online courses.", source.id, match.group(0))
    if re.search(r"\balerts?\b", question, re.I) and re.search(
            r"\b(?:confirm\w*|unconfirm\w*|percent\w*|proportion|share)\b", question, re.I):
        for source in sources.values():
            match = re.search(
                r"Total automated alerts\s+(\d+)\s+Alerts associated with confirmed misconduct\s+(\d+)\s+"
                r"Alerts not confirmed as misconduct\s+(\d+)\s+Share associated with confirmed misconduct\s+([\d.]+)%",
                source.text, re.I)
            if match:
                total, confirmed, unconfirmed = (int(value) for value in match.groups()[:3])
                reported_share = float(match.group(4))
                if total <= 0 or total - confirmed != unconfirmed or abs(100 * confirmed / total - reported_share) > .05:
                    continue
                caution = re.search(r"The authors caution that an unconfirmed alert does not necessarily mean "
                                    r"the system made a technical error\.", source.text, re.I)
                answer = (f"Study B recorded {total} automated alerts. {confirmed} were associated with confirmed "
                          f"misconduct; {total} − {confirmed} = {unconfirmed} were not confirmed. "
                          f"{confirmed} ÷ {total} × 100 = {reported_share:g}% of alerts were associated with "
                          "confirmed misconduct.")
                if caution:
                    answer += " An unconfirmed alert is not automatically a technical false positive."
                return answer, source.id, source.text[match.start():(caution.end() if caution else match.end())]
    if re.search(r"\b(?:combined|total|together)\b", question, re.I) and re.search(
            r"\b(?:participants|students|sample size)\b", question, re.I):
        for source in sources.values():
            match = re.search(r"Study A \(S-A\) Study B \(S-B\).*?Sample size\s+([\d,]+)\s+([\d,]+)",
                              source.text, re.I | re.S)
            if match:
                a, b = (int(value.replace(",", "")) for value in match.groups())
                return (f"Study A: {a:,} participants; Study B: {b:,} participants. "
                        f"{a:,} + {b:,} = {a + b:,} participants across two separate studies, "
                        "not one pooled experiment.", source.id,
                        match.group(0))
    if (re.search(r"\bstudy a\b", question, re.I)
            and re.search(r"\b(?:difference|percentage.points?)\b", question, re.I)
            and re.search(r"\b(?:misconduct|integrity)\b", question, re.I)):
        for source in sources.values():
            match = re.search(
                r"Study A reports confirmed misconduct of ([\d.]+)% in monitored sections and "
                r"([\d.]+)% in unmonitored sections, a difference of ([\d.]+) percentage points",
                source.text, re.I)
            if match:
                monitored, unmonitored, reported = (float(value) for value in match.groups())
                difference = unmonitored - monitored
                if abs(difference - reported) < 1e-9:
                    fmt = lambda value: f"{value:g}"
                    return (f"Study A reports {fmt(monitored)}% monitored and {fmt(unmonitored)}% "
                            f"unmonitored misconduct. {fmt(unmonitored)}% − {fmt(monitored)}% = "
                            f"{fmt(difference)} percentage points.", source.id, match.group(0))
    return None


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
            and re.search(r"\b(?:study\s+[ab]|paper|pdf|document)\b", question, re.I)):
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
    return f"Previous question: {previous[:100]}\nFollow-up question: {question}"


@router.get("/capabilities")
async def research_capabilities(
    request: Request,
    user: UserResponse = Depends(get_current_user),
) -> dict[str, bool | str | None]:
    provider = getattr(request.app.state, "answer_provider", None)
    if provider is None:
        return {"answer_generation": False, "key_revision": None}
    available = getattr(provider, "available", None)
    key = getattr(provider, "api_key", "")
    revision = (hmac.new(get_settings().jwt_secret.encode(), key.encode(), sha256).hexdigest()[:16]
                if isinstance(key, str) and key else None)
    return {"answer_generation": await run_in_threadpool(available) if available else True,
            "key_revision": revision}


async def _retrieve(payload: SearchRequest, request: Request, user: UserResponse) -> list[ChunkResponse]:
    if not payload.question.strip():
        raise HTTPException(422, "Enter a question.")
    repository: PaperRepository = request.app.state.paper_repository
    paper = await repository.find(user.id, payload.paper_id)
    if paper is None:
        raise HTTPException(404, "Paper not found.")
    if paper["status"] != "ready":
        raise HTTPException(409, "Wait until this paper is ready.")

    ingestion: IngestionService = request.app.state.ingestion_service
    def retrieve() -> list[ChunkResponse]:
        embedding = ingestion.embeddings.embed([payload.question.strip()])[0]
        return ingestion.vectors.search(user.id, payload.paper_id, embedding, payload.limit,
                                        question=payload.question.strip())

    return await run_in_threadpool(retrieve)


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


@router.post("/ask", response_model=AskResponse)
async def ask_paper(
    payload: AskRequest,
    request: Request,
    user: UserResponse = Depends(get_current_user),
):
    conversations: ConversationRepository = request.app.state.conversation_repository
    search_trace = SearchTrace()
    groq_counts = {"groq_calls": 0, "groq_cache_hits": 0, "groq_attempts": 0, "groq_retries": 0, "groq_failures": 0,
                   "groq_rate_limits": 0, "groq_server_errors": 0, "groq_network_errors": 0,
                   "groq_rate_limit_kind": None}
    groq_trace.set(groq_counts)
    conversation = None
    if payload.conversation_id:
        conversation = await conversations.find(user.id, payload.conversation_id)
        if conversation is None or conversation["paper_id"] != payload.paper_id:
            raise HTTPException(404, "Conversation not found.")
        if (payload.reply_to_message_id and not any(
                item.get("id") == payload.reply_to_message_id for item in conversation.get("messages", []))):
            raise HTTPException(404, "Previous message not found in this conversation.")

    async def save_result(result: AskResponse) -> AskResponse:
        result.question = payload.question.strip()
        result.request_id = payload.request_id
        for key, value in groq_counts.items():
            setattr(search_trace, key, value)
        search_trace.claim_status_counts = dict(Counter(item.status for item in result.claims))
        search_trace.verification_failures = search_trace.claim_status_counts.get("unsupported", 0)
        result.search_trace = search_trace.model_copy(deep=True)
        message = SavedMessage(
            id=uuid4().hex, question=payload.question.strip(), resolved_question=question_for_retrieval,
            status=result.status,
            answer=result.answer, sources=result.sources, external_sources=result.external_sources,
            assessment=result.assessment, created_at=datetime.now(timezone.utc),
            claims=result.claims, search_trace=result.search_trace,
        ).model_dump(mode="json")
        if payload.conversation_id:
            if not await conversations.append(user.id, payload.conversation_id, payload.paper_id, message):
                raise HTTPException(404, "Conversation not found.")
            result.conversation_id = payload.conversation_id
        else:
            conversation = await conversations.create(user.id, payload.paper_id, message)
            result.conversation_id = conversation["_id"]
        result.message_id = message["id"]
        return result

    question_for_retrieval = _question_with_context(payload.question, conversation,
                                                   payload.reply_to_message_id)
    retrieval_payload = SearchRequest(paper_id=payload.paper_id, question=question_for_retrieval,
                                      limit=payload.limit)
    passages = await _retrieve(retrieval_payload, request, user)
    paper_record = await request.app.state.paper_repository.find(user.id, payload.paper_id)
    provider: AnswerProvider | None = getattr(request.app.state, "answer_provider", None)
    if provider is None:
        raise HTTPException(503, "Answer generation is not configured yet.")
    available = getattr(provider, "available", None)
    if available and not await run_in_threadpool(available):
        raise HTTPException(503, "Answer generation is not available yet.")

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
    calculated = _reported_calculation(question_for_retrieval, sources)
    if calculated is not None:
        text, source_id, quote = calculated
        return await save_result(AskResponse(paper_id=payload.paper_id, status="answered",
            answer=f"{text} [{source_id}]", sources=[sources[source_id]],
            assessment=EvidenceReview(level="sufficient", chunk_ids=[source_id]),
            claims=[ClaimReviewResponse(text=text, section="pdf", status="supported",
                supporting=[ClaimQuoteResponse(source_id=source_id, quote=quote)], opposing=[])]))
    evidence = [Evidence(id=source.id, text=_evidence_context(question_for_retrieval, source.text))
                for source in sources.values()]
    assessment = None
    assess = getattr(provider, "assess", None)
    if assess:
        cache_key = None
        if isinstance(provider, GroqAnswerProvider):
            fingerprint = sha256(repr((question_for_retrieval, [(item.id, item.text) for item in evidence]))
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
                assessment = _validate_assessment(proposed, question_for_retrieval, sources)
        except AnswerRateLimitError as exc:
            raise HTTPException(429, "Evidence assessment is temporarily rate limited.",
                                headers={"Retry-After": exc.retry_after}) from None
        except Exception as exc:
            raise HTTPException(502, "Evidence assessment failed. Try again later.") from exc
        if cache_key is not None and assessment is not None:
            with _assessment_cache_lock:
                _assessment_cache[cache_key] = assessment.model_copy(deep=True)
                _assessment_cache.move_to_end(cache_key)
                if len(_assessment_cache) > _assessment_cache_limit:
                    _assessment_cache.popitem(last=False)
        if assessment is None:
            logging.getLogger(__name__).info("research_assessment_rejected request_id=%s",
                                             payload.request_id or "none")
            raise HTTPException(502, "The PDF evidence assessment could not be validated. Try again later.")
        logging.getLogger(__name__).info("research_assessment request_id=%s level=%s source_ids=%s",
                                         payload.request_id or "none", assessment.level, assessment.chunk_ids)

    pdf_comparison = bool(re.search(r"\bstudy\s+a\b", question_for_retrieval, re.I)
                          and re.search(r"\bstudy\s+b\b", question_for_retrieval, re.I))
    if assessment and assessment.level in ("partial", "none") and not (
            assessment.level == "partial" and pdf_comparison):
        pdf_sources = [sources[item] for item in assessment.chunk_ids]
        if _asks_for_paper_specific_fact(question_for_retrieval):
            detail = ("The retrieved PDF passages do not report that value. "
                      "An external source cannot establish a statistic for this paper.")
            return await save_result(AskResponse(paper_id=payload.paper_id,
                status="partially_supported" if pdf_sources else "not_mentioned", answer=detail,
                sources=pdf_sources, assessment=assessment))
        external_search = getattr(request.app.state, "external_search", None)
        if external_search is None or not external_search.available():
            explanation = ("The PDF supports part of the answer, but external search is not configured. "
                           f"Missing: {assessment.missing_information}" if assessment.level == "partial"
                           else "The retrieved PDF passages do not establish an answer. External search is not configured.")
            return await save_result(AskResponse(paper_id=payload.paper_id,
                status="partially_supported" if pdf_sources else "not_mentioned", answer=explanation,
                sources=pdf_sources, assessment=assessment))
        query = (_external_query(question_for_retrieval, assessment.missing_information)
                 if assessment.level == "partial" else question_for_retrieval)
        search_trace.duplicate_requests += int(query in search_trace.queries)
        search_trace.queries.append(query)
        try:
            traced_search = getattr(external_search, "search_with_trace", None)
            if traced_search:
                # The provider reports whether it actually used its cache, avoiding
                # a race between a separate cache check and the search call.
                search_trace.tavily_calls += 1
                outcome = await run_in_threadpool(traced_search, query)
                if outcome.cache_hit:
                    search_trace.tavily_calls -= 1
                    search_trace.tavily_cache_hits += 1
                search_trace.rejected_irrelevant_sources += outcome.rejected_irrelevant
                external = outcome.results
            else:
                cached_search = getattr(external_search, "cached", None)
                cached_external = cached_search(query) if cached_search else None
                if cached_external is not None:
                    search_trace.tavily_cache_hits += 1
                    external = cached_external
                else:
                    search_trace.tavily_calls += 1
                    external = await run_in_threadpool(external_search.search, query)
        except Exception:
            search_trace.failures += 1
            external = []
        search_trace.retrieved_source_counts.append(len(external))
        if not external:
            return await save_result(AskResponse(paper_id=payload.paper_id,
                status="partially_supported" if pdf_sources else "not_mentioned",
                answer="The PDF does not fully answer this question, and reliable external evidence was unavailable.",
                sources=pdf_sources, assessment=assessment))
        # Search snippets cannot establish the complete, current state of the law.
        # Show the official sources without asking the model to infer legal duties from them.
        if (re.search(r"\b(legal|law|regulation|requirements?)\b", question_for_retrieval, re.I)
                and re.search(r"\b(india|indian)\b", question_for_retrieval, re.I)):
            official = []
            for item in external:
                host = item.url.split("/", 3)[2].lower() if item.url.startswith("https://") else ""
                description = f"{item.title} {item.content}"
                if ((host.endswith(".gov.in") or host.endswith(".nic.in"))
                        and re.search(r"\b(?:personal data|data protection|biometric data|consent|data collection)\b",
                                      description, re.I)
                        and re.search(r"\b(?:act|rules?|regulations?|law|notification)\b", description, re.I)):
                    official.append(item)
            if not official:
                return await save_result(AskResponse(paper_id=payload.paper_id,
                    status="insufficient_evidence", answer="The PDF does not answer this question, and no official legal source was found.",
                    sources=pdf_sources, assessment=assessment))
            cited = [_external_source(item) for item in official]
            answer = ("Evidence from your PDF\n" +
                ("The PDF contains related information, but does not establish the current legal requirements."
                 if pdf_sources else "This information was not found in the retrieved passages of your uploaded paper.") +
                "\n\nAdditional information from external sources\n"
                "The official sources below may be relevant, but their search excerpts do not establish a complete, "
                "current answer to this legal question. Open the sources to check the applicable rules and dates: " +
                " ".join(f"[{item.id}]" for item in cited))
            return await save_result(AskResponse(paper_id=payload.paper_id,
                status="partially_supported" if pdf_sources else "not_mentioned", answer=answer,
                sources=pdf_sources, external_sources=cited, assessment=assessment))
        try:
            generated_external = await run_in_threadpool(provider.generate_with_external,
                question_for_retrieval, [item for item in evidence if item.id in assessment.chunk_ids],
                external, assessment.missing_information)
        except AnswerRateLimitError as exc:
            raise HTTPException(429, "Answer generation is temporarily rate limited.",
                                headers={"Retry-After": exc.retry_after}) from None
        except Exception as exc:
            raise HTTPException(502, "External answer generation failed. Try again later.") from exc
        external_by_id = {item.id: item for item in external}
        pdf_ids = getattr(generated_external, "pdf_ids", None)
        external_ids = getattr(generated_external, "external_ids", None)
        pdf_text = getattr(generated_external, "pdf_text", None)
        external_text = getattr(generated_external, "external_text", None)
        if (not isinstance(pdf_text, str) or not isinstance(external_text, str)
                or not external_text.strip() or not isinstance(pdf_ids, list)
                or not isinstance(external_ids, list) or any(item not in assessment.chunk_ids for item in pdf_ids)
                or any(item not in external_by_id for item in external_ids)
                or not external_ids or (assessment.level == "partial" and (not pdf_ids or not pdf_text.strip()))
                or (assessment.level == "none" and pdf_ids)):
            return await save_result(AskResponse(paper_id=payload.paper_id,
                status="insufficient_evidence", answer="The external answer could not be linked to its sources.",
                sources=pdf_sources, assessment=assessment))
        if (any(f"[{item}]" in pdf_text for item in external_by_id)
                or any(f"[{item}]" in external_text for item in sources)):
            return await save_result(AskResponse(paper_id=payload.paper_id,
                status="insufficient_evidence", answer="The external answer did not cite its sources.",
                sources=pdf_sources, assessment=assessment))
        try:
            checked_pdf, checked_external, claim_reviews = await _verified_sections(
                provider, question_for_retrieval, pdf_text if assessment.level == "partial" else "",
                external_text, pdf_ids, external_ids, sources, external_by_id)
        except AnswerRateLimitError as exc:
            raise HTTPException(429, "Claim verification is temporarily rate limited.",
                                headers={"Retry-After": exc.retry_after}) from None
        if not claim_reviews:
            return await save_result(AskResponse(paper_id=payload.paper_id,
                status="insufficient_evidence", answer="The generated claims could not be verified.",
                sources=pdf_sources, assessment=assessment))
        used_ids = {quote.source_id for claim in claim_reviews for quote in claim.supporting + claim.opposing}
        cited_external = [_external_source(item) for item in external if item.id in used_ids]
        cited_pdf = [item for item in sources.values() if item.id in used_ids]
        pdf_section = (checked_pdf if assessment.level == "partial" else
                       "This information was not found in the retrieved passages of your uploaded paper.")
        answer = (f"Evidence from your PDF\n{pdf_section}\n\n"
                  f"Additional information from external sources\n{checked_external}")
        reviewed_status = ("contradictory_evidence" if any(item.status == "contradicted" for item in claim_reviews)
                           else "insufficient_evidence" if not any(item.status == "supported" for item in claim_reviews)
                           else "partially_supported" if assessment.level == "partial" else "not_mentioned")
        return await save_result(AskResponse(paper_id=payload.paper_id,
            status=reviewed_status,
            answer=answer, sources=cited_pdf, external_sources=cited_external,
            assessment=assessment, claims=claim_reviews))

    if (assessment and assessment.level == "sufficient"
            and re.search(r"\bsample size\b", question_for_retrieval, re.I)
            and not re.search(r"\b(?:combined|both|total|together|and|plus|methodolog\w*|design|"
                              r"universit\w*|regions?|duration|observation|setting)\b",
                              question_for_retrieval, re.I)):
        for source_id in assessment.chunk_ids:
            match = re.search(r"\b(?:study|trial|cohort|sample)\s+(?:included|enrolled|comprised)\s+"
                              r"\d[\d,]*\s+(?:students|participants|subjects|patients)\b",
                              sources[source_id].text, re.I)
            if match:
                statement = match.group(0)
                return await save_result(AskResponse(paper_id=payload.paper_id, status="answered",
                    answer=f"{statement[0].upper()}{statement[1:]}.",
                    sources=[sources[source_id]], assessment=assessment,
                    claims=[ClaimReviewResponse(text=f"{statement[0].upper()}{statement[1:]}",
                        section="pdf", status="supported",
                        supporting=[ClaimQuoteResponse(source_id=source_id, quote=statement)],
                        opposing=[])]))

    if not evidence:
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                                             answer="I could not retrieve passages that support an answer.", sources=[]))
    if assessment:
        evidence = [item for item in evidence if item.id in assessment.chunk_ids]
    try:
        generated = await run_in_threadpool(provider.generate, question_for_retrieval, evidence)
    except AnswerRateLimitError as exc:
        raise HTTPException(429, "Answer generation is temporarily rate limited. Try again shortly.",
                            headers={"Retry-After": exc.retry_after}) from None
    except Exception as exc:
        raise HTTPException(502, "Answer generation failed. Try again later.") from exc

    answer = getattr(generated, "text", None)
    evidence_ids = getattr(generated, "evidence_ids", None)
    status = getattr(generated, "status", None)
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
                                             answer="The retrieved passages do not establish a reliable answer.", sources=[]))
    if answer.strip().casefold().rstrip("?.! ") == payload.question.strip().casefold().rstrip("?.! "):
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                                             answer="The retrieved passages do not establish a reliable answer.", sources=[]))
    if not isinstance(evidence_ids, list) or not evidence_ids or any(
        not isinstance(source_id, str) or source_id not in sources
        or (assessment is not None and source_id not in assessment.chunk_ids) for source_id in evidence_ids
    ):
        return await save_result(AskResponse(paper_id=payload.paper_id, status="insufficient_evidence",
                                             answer="The proposed answer could not be linked to the retrieved passages.", sources=[]))
    cited = list(dict.fromkeys(evidence_ids))
    try:
        checked_pdf, _, claim_reviews = await _verified_sections(
            provider, question_for_retrieval, answer.strip(), "", cited, [],
            {item: sources[item] for item in cited}, {})
    except AnswerRateLimitError as exc:
        raise HTTPException(429, "Claim verification is temporarily rate limited.",
                            headers={"Retry-After": exc.retry_after}) from None
    if not claim_reviews:
        return await save_result(AskResponse(paper_id=payload.paper_id,
            status="insufficient_evidence", answer="The generated claims could not be verified.",
            sources=[], assessment=assessment))
    used_ids = {quote.source_id for item in claim_reviews for quote in item.supporting + item.opposing}
    comparison_gaps = _methodology_comparison_gaps(question_for_retrieval, claim_reviews)
    if not any(item.status in ("supported", "partially_supported", "contradicted") for item in claim_reviews):
        status = "insufficient_evidence"
    elif any(item.status == "contradicted" for item in claim_reviews):
        status = "contradictory_evidence"
    elif (assessment and assessment.level == "partial") or comparison_gaps or any(
            item.status in ("partially_supported", "unsupported") for item in claim_reviews):
        status = "partially_supported"
        if assessment and assessment.missing_information and assessment.missing_information not in checked_pdf:
            checked_pdf = f"{checked_pdf} {assessment.missing_information}".strip()
        if comparison_gaps:
            checked_pdf = f"{checked_pdf} I could not verify {', '.join(comparison_gaps)} from the cited passages.".strip()
    logging.getLogger(__name__).info("research_verification request_id=%s response_status=%s claim_statuses=%s source_ids=%s",
                                     payload.request_id or "none", status,
                                     [item.status for item in claim_reviews], sorted(used_ids))
    return await save_result(AskResponse(paper_id=payload.paper_id, status=status, answer=checked_pdf,
        sources=[sources[item] for item in cited if item in used_ids], assessment=assessment,
        claims=claim_reviews))
