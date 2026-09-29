from typing import Any

from fastapi import Depends, HTTPException, Request, status

from .config import get_settings
from .models import UserResponse
from .repositories.users import UserRepository
from .security import read_session_subject


def get_user_repository(request: Request) -> UserRepository:
    return request.app.state.user_repository


def to_user_response(user: dict[str, Any]) -> UserResponse:
    return UserResponse(
        id=str(user["_id"]),
        email=user["email"],
        name=user.get("name"),
        role=user["role"],
    )


async def get_current_user(
    request: Request,
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    settings = get_settings()
    token = request.cookies.get(settings.session_cookie_name)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in is required.",
        )

    user_id = read_session_subject(token, settings)
    user = await user_repository.find_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your account is no longer available.",
        )

    return to_user_response(user)


async def require_admin(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access is required.",
        )

    return current_user
