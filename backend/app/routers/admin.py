from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_user_repository, require_admin, to_user_response
from ..models import UserResponse
from ..repositories.users import UserRepository


router = APIRouter(prefix="/api/v1/admin", tags=["administration"])


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    _current_admin: Annotated[UserResponse, Depends(require_admin)],
    user_repository: Annotated[UserRepository, Depends(get_user_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[UserResponse]:
    users = await user_repository.list_users(limit=limit, offset=offset)
    return [to_user_response(user) for user in users]
