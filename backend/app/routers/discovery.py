"""Authenticated academic discovery, isolated from user paper identifiers."""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from datetime import datetime, timezone

from ..dependencies import get_current_user, require_admin
from ..models import UserResponse

router = APIRouter(prefix="/api/v1/discovery", tags=["discovery"])


@router.get("/health")
async def discovery_health(request: Request, user: UserResponse = Depends(require_admin)):
    """Report only observed status; this endpoint does not probe providers."""
    return request.app.state.discovery_service.client.health()


@router.get("/diagnostics")
async def discovery_diagnostics(request: Request, user: UserResponse = Depends(require_admin)):
    cursor = request.app.state.discovery_audit.find({"owner_id": user.id}).sort("created_at", -1).limit(20)
    return [{"created_at": item["created_at"], "diagnostics": item["diagnostics"]}
            for item in await cursor.to_list(length=20)]


@router.get("")
async def discover(request: Request, query: str = Query(min_length=2, max_length=200),
                   offset: int = Query(default=0, ge=0, le=1000),
                   limit: int = Query(default=10, ge=1, le=20), user: UserResponse = Depends(get_current_user)):
    if getattr(request.app.state, "demo_mode", False):
        raise HTTPException(503, "Live academic discovery is unavailable in the demo. Open the sample from your Library.")
    try:
        result = await run_in_threadpool(request.app.state.discovery_service.search, query, offset, limit)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    diagnostics = result.pop("_diagnostics", [])
    if diagnostics:
        await request.app.state.discovery_audit.insert_one({"owner_id": user.id, "created_at": datetime.now(timezone.utc),
                                                           "diagnostics": diagnostics})
    collection = request.app.state.discovery_records
    for record in result["results"]:
        await collection.update_one({"_id": record["id"]}, {"$set": {"record": record}}, upsert=True)
    return result


@router.get("/record")
async def discovery_record(request: Request, identifier: str = Query(min_length=3, max_length=260),
                           user: UserResponse = Depends(get_current_user)):
    if not identifier.startswith(("doi:", "arxiv:", "pmid:")):
        raise HTTPException(422, "Invalid paper identifier.")
    result = await discover(request, identifier, 0, 1, user)
    if not result["results"]:
        raise HTTPException(404, "A verified record is unavailable.")
    return result["results"][0]
