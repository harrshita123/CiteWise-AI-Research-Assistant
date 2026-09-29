import asyncio

from fastapi import APIRouter, Depends, HTTPException, Response, status

from ..config import Settings, get_settings
from ..dependencies import get_current_user, get_user_repository, to_user_response
from ..models import AuthenticationResponse, GoogleCredentialRequest, LoginRequest, SignupRequest, UserResponse
from ..repositories.users import UserRepository
from ..security import (
    create_session_token,
    dummy_password_hash,
    hash_password,
    verify_google_credential,
    verify_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


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


@router.post("/signup", response_model=AuthenticationResponse, status_code=status.HTTP_201_CREATED)
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


@router.post("/login", response_model=AuthenticationResponse)
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


@router.post("/google", response_model=AuthenticationResponse)
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
