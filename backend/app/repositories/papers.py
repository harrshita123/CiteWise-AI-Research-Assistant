from datetime import datetime, timezone
from typing import Any

from pymongo.asynchronous.collection import AsyncCollection

from ..paper_models import PaperResponse


class PaperRepository:
    def __init__(self, collection: AsyncCollection[dict[str, Any]]):
        self.collection = collection

    async def ensure_indexes(self) -> None:
        await self.collection.create_index([("owner_id", 1), ("file_hash", 1)], unique=True)
        await self.collection.create_index([("owner_id", 1), ("created_at", -1)])

    async def create(self, paper: dict[str, Any]) -> None:
        await self.collection.insert_one(paper)

    async def find(self, owner_id: str, paper_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"_id": paper_id, "owner_id": owner_id})

    async def list_for_user(self, owner_id: str, limit: int, offset: int, status: str | None = None) -> list[dict[str, Any]]:
        query = {"owner_id": owner_id}
        if status is not None:
            query["status"] = status
        cursor = self.collection.find(query).sort("created_at", -1).skip(offset).limit(limit)
        return await cursor.to_list(length=limit)

    async def summary_for_user(self, owner_id: str) -> dict[str, int]:
        counts = {"total": await self.collection.count_documents({"owner_id": owner_id})}
        for status in ("ready", "processing", "failed"):
            counts[status] = await self.collection.count_documents({"owner_id": owner_id, "status": status})
        return counts

    async def update(self, owner_id: str, paper_id: str, **fields: Any) -> None:
        await self.collection.update_one(
            {"_id": paper_id, "owner_id": owner_id},
            {"$set": {**fields, "updated_at": datetime.now(timezone.utc)}},
        )

    async def remove(self, owner_id: str, paper_id: str) -> None:
        await self.collection.delete_one({"_id": paper_id, "owner_id": owner_id})


def paper_response(paper: dict[str, Any]) -> PaperResponse:
    return PaperResponse(id=paper["_id"], **{key: value for key, value in paper.items() if key != "_id"})
