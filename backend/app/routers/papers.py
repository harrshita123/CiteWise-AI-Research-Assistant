import asyncio
import logging
from datetime import datetime, timezone
from hashlib import sha256
from math import ceil
from uuid import uuid4
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from pymongo.errors import DuplicateKeyError
from starlette.responses import FileResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from ..config import get_settings
from ..dependencies import get_current_user
from ..models import UserResponse
from ..paper_models import PaperDetail, PaperResponse, PaperSummary
from ..repositories.papers import PaperRepository, paper_response
from ..services.ingestion import IngestionService
from ..services.demo_research import SAMPLE_IDENTITY, is_sample
from ..services.pdf_processing import extract_pages, PdfTextUnavailable, PdfValidationError
from ..services.document_classification import classify_document, DocumentType
from ..services.publication_identity import extract_identity_input, extracted_paper_metadata
from ..services.verification_gates import VerificationReport, document_quality, ready_for_research

router = APIRouter(prefix="/api/v1/papers", tags=["papers"])

REVIEW_MESSAGE = "We couldn't confidently determine whether this is a research paper. Please check the document or request review."
REJECTION_MESSAGES = {
    "a certificate": "This document appears to be a certificate, not an academic research paper. CiteWise currently accepts research papers only. Please upload a valid research article.",
    "a résumé": "This document appears to be a resume or CV, not a research paper.",
}

def get_paper_repository(request: Request) -> PaperRepository:
    return request.app.state.paper_repository


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_upload_lock(request: Request) -> asyncio.Lock:
    return request.app.state.upload_lock


def _public_paper(request: Request, paper: dict) -> PaperResponse:
    if getattr(request.app.state, "demo_mode", False) and is_sample(paper) and ready_for_research(paper):
        paper = {**paper, "publication_identity": SAMPLE_IDENTITY}
    return paper_response(paper)


@router.get("", response_model=list[PaperResponse])
async def list_papers(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Literal["ready", "processing", "failed"] | None = None,
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
):
    return [_public_paper(request, paper) for paper in await repository.list_for_user(user.id, limit, offset, status)]


@router.get("/summary", response_model=PaperSummary)
async def get_paper_summary(
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
):
    return await repository.summary_for_user(user.id)


@router.get("/{paper_id}", response_model=PaperDetail)
async def get_paper(
    paper_id: str,
    request: Request,
    chunk_offset: int = Query(default=0, ge=0),
    chunk_limit: int = Query(default=10, ge=1, le=50),
    include_chunks: bool = True,
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    paper = await repository.find(user.id, paper_id)
    if paper is None:
        raise HTTPException(404, "Paper not found.")
    identity_input = paper.get("identity_input") or {}
    if paper["status"] == "ready" and identity_input and not paper.get("document_type"):
        outcome, category = await run_in_threadpool(classify_document, identity_input)
        if outcome == DocumentType.NOT_RESEARCH_PAPER:
            paper = await repository.mark_nonresearch(paper, category) or await repository.find(user.id, paper_id)
            if paper is None:
                raise HTTPException(404, "Paper not found.")
    chunks = []
    if include_chunks and ready_for_research(paper):
        chunks = await run_in_threadpool(ingestion.preview, user.id, paper_id, chunk_offset, chunk_limit)
    return PaperDetail(paper=_public_paper(request, paper), chunks=chunks,
                       extracted_metadata=(extracted_paper_metadata(identity_input) or None)
                       if paper["status"] in {"ready", "needs_review"} else None)


@router.post("/{paper_id}/verification/retry", status_code=202)
async def retry_publication_verification(paper_id: str, request: Request,
        user: UserResponse = Depends(get_current_user), repository: PaperRepository = Depends(get_paper_repository)):
    origin = request.headers.get("origin")
    if origin and origin not in get_settings().cors_origins:
        raise HTTPException(403, "This request origin is not allowed.")
    paper = await repository.find(user.id, paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found.")
    if getattr(request.app.state, "demo_mode", False):
        raise HTTPException(409, "Publication verification is unavailable in the demo.")
    if paper["status"] == "not_research":
        raise HTTPException(409, "This document is not a research paper.")
    retry_after = (paper.get("publication_identity") or {}).get("retry_after")
    if isinstance(retry_after, datetime):
        seconds = ceil((retry_after.replace(tzinfo=retry_after.tzinfo or timezone.utc) - datetime.now(timezone.utc)).total_seconds())
        if seconds > 0:
            raise HTTPException(429, f"Verification can be retried in {seconds} seconds.", headers={"Retry-After": str(seconds)})
    if "identity_input" not in paper and await repository.backfill_identity_job(user.id, paper_id, paper["file_hash"]):
        return {"status": "pending"}
    if not await repository.retry_identity_job(user.id, paper_id):
        raise HTTPException(409, "Verification cannot be retried for this paper right now.")
    return {"status": "pending"}


@router.get("/{paper_id}/file")
async def view_paper_file(
    paper_id: str,
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    paper = await repository.find(user.id, paper_id)
    if paper is None or paper["status"] not in {"ready", "not_research", "needs_review"}:
        raise HTTPException(404, "Paper not found.")
    if paper.get("storage_provider", "local") == "cloudinary":
        try:
            url = ingestion.cloudinary_storage.view_url(paper_id)
        except Exception:
            raise HTTPException(503, "The original PDF is temporarily unavailable.") from None
        return RedirectResponse(url, headers={"Cache-Control": "private, no-store"})
    path = ingestion.file_path(paper_id)
    if not path.is_file():
        raise HTTPException(404, "The original PDF is unavailable.")
    return FileResponse(
        path,
        media_type="application/pdf",
        content_disposition_type="inline",
        filename=paper["filename"],
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.delete("/{paper_id}", status_code=204)
async def delete_paper(
    paper_id: str,
    request: Request,
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    origin = request.headers.get("origin")
    if origin and origin not in get_settings().cors_origins:
        raise HTTPException(403, "This request origin is not allowed.")

    paper = await repository.find(user.id, paper_id)
    if paper is None:
        raise HTTPException(404, "Paper not found.")
    if paper["status"] == "processing":
        raise HTTPException(409, "Wait until this paper finishes processing before deleting it.")

    try:
        await run_in_threadpool(ingestion.cleanup, user.id, paper_id, paper.get("storage_provider", "local"))
        await repository.remove(user.id, paper_id)
    except Exception:
        raise HTTPException(503, "The paper could not be deleted. Please try again.") from None


@router.post("", response_model=PaperDetail, status_code=201)
async def upload_paper(
    request: Request,
    file: UploadFile,
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
    ingestion: IngestionService = Depends(get_ingestion_service),
    upload_lock: asyncio.Lock = Depends(get_upload_lock),
):
    settings = get_settings()
    report = VerificationReport("document")

    async def reject(check, reason, status_code, message):
        report.record(check, "FAIL", reason)
        for pending in ("H1", "H2", "H3", "H4", "H5", *(f"M{i}" for i in range(1, 11))):
            if pending not in report.checks:
                report.record(pending, "NOT_APPLICABLE", "upload_blocked_before_check")
        try:
            await repository.record_rejection(user.id, report.dump())
        except Exception as exc:
            logging.getLogger(__name__).warning("upload audit unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(status_code, message)

    origin = request.headers.get("origin")
    if origin and origin not in settings.cors_origins:
        raise HTTPException(403, "This upload origin is not allowed.")
    report.record("H1", "PASS", "authenticated_owner_assigned")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        await reject("H2", "filename_not_pdf", 422, "Choose a PDF file.")
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        await reject("H2", "unaccepted_media_type", 422, "Choose a PDF file.")
    if upload_lock.locked():
        raise HTTPException(429, "A paper is already processing. Please try again shortly.", headers={"Retry-After": "5"})

    # One ingestion at a time keeps the local model and PDF parser within a predictable budget.
    async with upload_lock:
        try:
            content = await file.read(settings.max_pdf_size_mb * 1024 * 1024 + 1)
        finally:
            await file.close()
        if len(content) > settings.max_pdf_size_mb * 1024 * 1024:
            await reject("H2", "size_limit_exceeded", 413, f"PDFs must be {settings.max_pdf_size_mb} MB or smaller.")
        if not content.startswith(b"%PDF-"):
            await reject("H2", "invalid_pdf_header", 422, "This file does not contain a valid PDF header.")

        prepared, review_reason, identity_input = None, None, {}
        try:
            prepared = await run_in_threadpool(extract_pages, content, settings.max_pdf_pages, settings.max_pdf_characters)
        except PdfTextUnavailable as exc:
            review_reason = str(exc)
        except PdfValidationError as exc:
            await reject("H2", "pdf_invalid_or_over_limit", 422, str(exc))
        report.record("H2", "PASS", "pdf_integrity_and_limits_checked")
        report.record("H4", "PASS" if prepared else "UNAVAILABLE",
                      "usable_text_mapped_to_pages" if prepared else "no_extractable_text")
        if prepared:
            try:
                identity_input = await run_in_threadpool(extract_identity_input, content, include_tail=True)
            except (RuntimeError, ValueError):
                identity_input = {}
        outcome, category = await run_in_threadpool(classify_document, identity_input)
        if outcome == DocumentType.NOT_RESEARCH_PAPER:
            await reject("H3", "nonresearch_document", 422,
                         REJECTION_MESSAGES.get(category) or
                         f"This document appears to be {category}, not a research paper. Please upload a research paper PDF.")
        report.record("H3", "PASS" if outcome == DocumentType.RESEARCH_PAPER else "UNAVAILABLE",
                      "scholarly_document" if outcome == DocumentType.RESEARCH_PAPER else "classification_inconclusive")
        ready = report.passes("H1", "H2", "H3", "H4")
        if not ready:
            review_reason = f"{REVIEW_MESSAGE} {review_reason}" if review_reason else REVIEW_MESSAGE

        identity_input.pop("tail", None)
        paper_id = uuid4().hex
        storage_provider = "local" if settings.citewise_demo_mode else "cloudinary"
        identity = ({"status": "not_verified", "reason": "Publication verification is unavailable in the demo."}
                    if settings.citewise_demo_mode else {"status": "pending", "reason": "Publication identity check pending."})
        paper = {
            "_id": paper_id, "owner_id": user.id,
            "filename": file.filename.replace("\\", "/").split("/")[-1][:255],
            "file_hash": sha256(content).hexdigest(), "status": "processing",
            "document_type": outcome.value,
            "storage_provider": storage_provider,
            "storage_id": paper_id if settings.citewise_demo_mode else ingestion.cloudinary_storage.public_id(paper_id),
            "embedding_model": ingestion.embeddings.model_name,
            "embedding_dimensions": ingestion.embeddings.dimensions,
            "created_at": datetime.now(timezone.utc),
            "publication_identity": identity if ready else None,
        }
        try:
            await repository.create(paper)
        except DuplicateKeyError:
            raise HTTPException(409, "This PDF is already in your library.") from None

        try:
            if ready:
                counts = await run_in_threadpool(ingestion.process, user.id, paper_id, content, prepared)
                chunks = await run_in_threadpool(ingestion.preview, user.id, paper_id)
                if not chunks or not all(1 <= chunk.page_number <= counts["page_count"] for chunk in chunks):
                    raise RuntimeError("Indexed passages are unavailable or have invalid page mappings.")
                report.record("H4", "PASS", "usable_text_mapped_to_pages",
                              (f"page:{chunk.page_number}" for chunk in chunks))
            else:
                storage = ingestion.local_storage if settings.citewise_demo_mode else ingestion.cloudinary_storage
                await run_in_threadpool(storage.save, paper_id, content)
                counts, chunks = {"page_count": prepared[0] if prepared else 0, "chunk_count": 0}, []
            metadata = extracted_paper_metadata(identity_input)
            document_quality(report, metadata, paper["publication_identity"] or {}, usable_text=ready)
            fields = dict(status="ready" if ready else "needs_review", error=review_reason,
                          document_state="READY_FOR_RESEARCH" if ready else "NEEDS_REVIEW",
                          identity_input=identity_input, verification_checks=report.dump(), **counts)
            await repository.update(user.id, paper_id, **fields)
        except Exception as error:
            message = str(error) if isinstance(error, PdfValidationError) else (
                "Processing could not finish. Check the embedding model and PDF storage, then try again."
            )
            try:
                await run_in_threadpool(ingestion.cleanup, user.id, paper_id, storage_provider)
                await repository.remove(user.id, paper_id)
            except Exception:
                await repository.update(user.id, paper_id, status="failed", error="Processing failed; storage cleanup needs attention.")
            if isinstance(error, PdfValidationError):
                await reject("H4", "usable_index_text_missing", 422, message)
            raise HTTPException(422 if isinstance(error, PdfValidationError) else 503, message) from None

        paper.update(fields)
        return PaperDetail(paper=_public_paper(request, paper), chunks=chunks,
                           extracted_metadata=metadata or None)
