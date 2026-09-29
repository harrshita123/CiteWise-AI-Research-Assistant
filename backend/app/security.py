import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import HTTPException, status
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from .config import Settings, get_settings

password_hash = PasswordHash.recommended()
dummy_password_hash = password_hash.hash("citewise-dummy-password")


def _verify_google_credential(credential: str, client_id: str) -> dict[str, Any]:
    return id_token.verify_oauth2_token(credential, GoogleRequest(), client_id)


async def verify_google_credential(credential: str) -> dict[str, Any]:
    settings = get_settings()

    try:
        profile = await asyncio.to_thread(
            _verify_google_credential,
            credential,
            settings.google_client_id,
        )
    except (ValueError, google_auth_exceptions.GoogleAuthError) as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google sign-in could not be verified.",
        ) from error

    if profile.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google sign-in came from an unexpected issuer.",
        )

    if profile.get("email_verified") is not True:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google account email must be verified.",
        )

    if not profile.get("sub") or not profile.get("email"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google sign-in did not include the required account details.",
        )

    return profile


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    return password_hash.verify(password, stored_hash)


def create_session_token(user_id: str, settings: Settings) -> str:
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(minutes=settings.session_duration_minutes)

    return jwt.encode(
        {
            "sub": user_id,
            "type": "session",
            "iat": issued_at,
            "exp": expires_at,
        },
        settings.jwt_secret,
        algorithm="HS256",
    )


def read_session_subject(token: str, settings: Settings) -> str:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            options={"require": ["sub", "exp", "iat"]},
        )
    except InvalidTokenError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session is invalid or has expired.",
        ) from error

    if payload.get("type") != "session":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session is invalid.",
        )

    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session is invalid.",
        )

    return subject
