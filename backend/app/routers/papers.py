import asyncio
from datetime import datetime, timezone
from hashlib import sha256
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
from ..services.pdf_processing import PdfValidationError

router = APIRouter(prefix="/api/v1/papers", tags=["papers"])


def get_paper_repository(request: Request) -> PaperRepository:
    return request.app.state.paper_repository


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_upload_lock(request: Request) -> asyncio.Lock:
    return request.app.state.upload_lock


@router.get("", response_model=list[PaperResponse])
async def list_papers(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Literal["ready", "processing", "failed"] | None = None,
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
):
    return [paper_response(paper) for paper in await repository.list_for_user(user.id, limit, offset, status)]


@router.get("/summary", response_model=PaperSummary)
async def get_paper_summary(
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
):
    return await repository.summary_for_user(user.id)


@router.get("/{paper_id}", response_model=PaperDetail)
async def get_paper(
    paper_id: str,
    chunk_offset: int = Query(default=0, ge=0),
    chunk_limit: int = Query(default=10, ge=1, le=50),
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    paper = await repository.find(user.id, paper_id)
    if paper is None:
        raise HTTPException(404, "Paper not found.")
    chunks = []
    if paper["status"] == "ready":
        chunks = await run_in_threadpool(ingestion.preview, user.id, paper_id, chunk_offset, chunk_limit)
    return PaperDetail(paper=paper_response(paper), chunks=chunks)


@router.get("/{paper_id}/file")
async def view_paper_file(
    paper_id: str,
    user: UserResponse = Depends(get_current_user),
    repository: PaperRepository = Depends(get_paper_repository),
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    paper = await repository.find(user.id, paper_id)
    if paper is None or paper["status"] != "ready":
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
    origin = request.headers.get("origin")
    if origin and origin not in settings.cors_origins:
        raise HTTPException(403, "This upload origin is not allowed.")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(422, "Choose a PDF file.")
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(422, "Choose a PDF file.")
    if upload_lock.locked():
        raise HTTPException(429, "A paper is already processing. Please try again shortly.", headers={"Retry-After": "5"})

    # One ingestion at a time keeps the local model and PDF parser within a predictable budget.
    async with upload_lock:
        try:
            content = await file.read(settings.max_pdf_size_mb * 1024 * 1024 + 1)
        finally:
            await file.close()
        if len(content) > settings.max_pdf_size_mb * 1024 * 1024:
            raise HTTPException(413, f"PDFs must be {settings.max_pdf_size_mb} MB or smaller.")
        if not content.startswith(b"%PDF-"):
            raise HTTPException(422, "This file does not contain a valid PDF header.")

        paper_id = uuid4().hex
        paper = {
            "_id": paper_id, "owner_id": user.id,
            "filename": file.filename.replace("\\", "/").split("/")[-1][:255],
            "file_hash": sha256(content).hexdigest(), "status": "processing",
            "storage_provider": "cloudinary",
            "storage_id": ingestion.cloudinary_storage.public_id(paper_id),
            "embedding_model": ingestion.embeddings.model_name,
            "embedding_dimensions": ingestion.embeddings.dimensions,
            "created_at": datetime.now(timezone.utc),
        }
        try:
            await repository.create(paper)
        except DuplicateKeyError:
            raise HTTPException(409, "This PDF is already in your library.") from None

        try:
            counts = await run_in_threadpool(ingestion.process, user.id, paper_id, content)
            chunks = await run_in_threadpool(ingestion.preview, user.id, paper_id)
            await repository.update(user.id, paper_id, status="ready", **counts)
        except Exception as error:
            message = str(error) if isinstance(error, PdfValidationError) else (
                "Processing could not finish. Check the embedding model and PDF storage, then try again."
            )
            try:
                await run_in_threadpool(ingestion.cleanup, user.id, paper_id, "cloudinary")
                await repository.remove(user.id, paper_id)
            except Exception:
                await repository.update(user.id, paper_id, status="failed", error="Processing failed; storage cleanup needs attention.")
            raise HTTPException(422 if isinstance(error, PdfValidationError) else 503, message) from None

        paper.update(status="ready", **counts)
        return PaperDetail(paper=paper_response(paper), chunks=chunks)
