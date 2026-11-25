import asyncio
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from starlette.concurrency import run_in_threadpool

from ..config import Settings, get_settings
from ..dependencies import get_current_user, get_user_repository, to_user_response
from ..models import AuthenticationResponse, GoogleCredentialRequest, LoginRequest, SignupRequest, UserResponse
from ..repositories.users import UserRepository
from ..services.demo_research import CASES, SAMPLE_FILENAME, SAMPLE_IDENTITY, SAMPLE_PDF, SAMPLE_SHA256, sample_source
from ..services.document_classification import DocumentType, classify_document
from ..services.publication_identity import extract_identity_input, extracted_paper_metadata
from ..services.verification_gates import VerificationReport, document_quality
from ..security import (
    create_session_token,
    dummy_password_hash,
    hash_password,
    verify_google_credential,
    verify_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


def require_live_auth(settings: Settings = Depends(get_settings)) -> None:
    if settings.citewise_demo_mode:
        raise HTTPException(404, "Use the demo workspace to sign in.")


@router.get("/demo")
async def demo_status(settings: Settings = Depends(get_settings)):
    return {"demo_mode": settings.citewise_demo_mode}


@router.post("/demo", status_code=201)
async def start_demo(request: Request, response: Response, settings: Settings = Depends(get_settings),
                     users: UserRepository = Depends(get_user_repository)):
    if not settings.citewise_demo_mode:
        raise HTTPException(404, "Demo mode is unavailable.")
    origin = request.headers.get("origin")
    if origin and origin not in settings.cors_origins:
        raise HTTPException(403, "This request origin is not allowed.")
    lock = request.app.state.upload_lock
    if lock.locked():
        raise HTTPException(503, "The demo is preparing a workspace. Try again shortly.")
    async with lock:
        user = await users.create_demo_guest()
        owner = str(user["_id"])
        paper_id = uuid4().hex
        ingestion = request.app.state.ingestion_service
        papers = request.app.state.paper_repository
        try:
            content = SAMPLE_PDF.read_bytes()
            if sha256(content).hexdigest() != SAMPLE_SHA256:
                raise ValueError("Bundled PDF does not match the checked edition.")
            identity = await run_in_threadpool(extract_identity_input, content, include_tail=True)
            outcome, _ = await run_in_threadpool(classify_document, identity)
            if outcome != DocumentType.RESEARCH_PAPER:
                raise ValueError("Bundled PDF failed research classification.")
            identity.pop("tail", None)
            counts = await run_in_threadpool(ingestion.process, owner, paper_id, content)
            chunks = await run_in_threadpool(ingestion.preview, owner, paper_id, limit=counts["chunk_count"])
            if any(sample_source(case, chunks) is None for case in CASES):
                raise ValueError("Bundled sample evidence is missing.")
            report = VerificationReport("document")
            for check, reason in (("H1", "authenticated_owner_assigned"), ("H2", "checked_publisher_pdf"),
                                  ("H3", "scholarly_document"), ("H4", "usable_text_mapped_to_pages")):
                report.record(check, "PASS", reason)
            document_quality(report, extracted_paper_metadata(identity), SAMPLE_IDENTITY, usable_text=True)
            await papers.create({"_id": paper_id, "owner_id": owner, "filename": SAMPLE_FILENAME,
                "file_hash": SAMPLE_SHA256, "demo_sample": True, "status": "ready",
                "document_type": DocumentType.RESEARCH_PAPER.value, "document_state": "READY_FOR_RESEARCH",
                "verification_checks": report.dump(),
                "storage_provider": "local", "storage_id": paper_id, "identity_input": identity,
                "publication_identity": SAMPLE_IDENTITY,
                "embedding_model": ingestion.embeddings.model_name,
                "embedding_dimensions": ingestion.embeddings.dimensions,
                "created_at": datetime.now(timezone.utc), **counts})
        except Exception:
            await run_in_threadpool(ingestion.cleanup, owner, paper_id, "local")
            await users.collection.delete_one({"_id": user["_id"]})
            raise HTTPException(503, "The sample paper could not be prepared. Try again shortly.") from None
        set_session_cookie(response, owner, settings)
        return {"user": to_user_response(user), "paper_id": paper_id}


def set_session_cookie(response: Response, user_id: str, settings: Settings) -> None:
    response.set_cookie(
        key=settings.session_cookie_name,
        value=create_session_token(user_id, settings),
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=settings.session_duration_minutes * 60,
        path="/",
    )


@router.post("/signup", response_model=AuthenticationResponse, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_live_auth)])
async def signup(
    payload: SignupRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
    user_repository: UserRepository = Depends(get_user_repository),
) -> AuthenticationResponse:
    encoded_password = await asyncio.to_thread(hash_password, payload.password.get_secret_value())
    user = await user_repository.create_password_user(payload.name, str(payload.email), encoded_password)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    set_session_cookie(response, str(user["_id"]), settings)
    return AuthenticationResponse(user=to_user_response(user))


@router.post("/login", response_model=AuthenticationResponse, dependencies=[Depends(require_live_auth)])
async def login(
    payload: LoginRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
    user_repository: UserRepository = Depends(get_user_repository),
) -> AuthenticationResponse:
    user = await user_repository.find_by_email(str(payload.email))
    stored_hash = user.get("password_hash") if user else dummy_password_hash
    password_matches = await asyncio.to_thread(
        verify_password,
        payload.password.get_secret_value(),
        stored_hash,
    )

    if user is None or not user.get("password_hash") or not password_matches:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password is incorrect.",
        )

    set_session_cookie(response, str(user["_id"]), settings)
    return AuthenticationResponse(user=to_user_response(user))


@router.post("/google", response_model=AuthenticationResponse, dependencies=[Depends(require_live_auth)])
async def sign_in_with_google(
    payload: GoogleCredentialRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
    user_repository: UserRepository = Depends(get_user_repository),
) -> AuthenticationResponse:
    profile = await verify_google_credential(payload.credential)
    user = await user_repository.upsert_google_user(profile)
    set_session_cookie(response, str(user["_id"]), settings)

    return AuthenticationResponse(user=to_user_response(user))


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    return current_user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(settings: Settings = Depends(get_settings)) -> Response:
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(
        key=settings.session_cookie_name,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        path="/",
    )
    return response
