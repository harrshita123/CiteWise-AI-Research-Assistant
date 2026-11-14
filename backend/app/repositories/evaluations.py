from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo.asynchronous.collection import AsyncCollection


class EvaluationRepository:
    def __init__(self, collection: AsyncCollection[dict[str, Any]]):
        self.collection = collection

    async def ensure_indexes(self) -> None:
        await self.collection.create_index([("owner_id", 1), ("created_at", -1)])

    async def create(self, owner_id: str, report: dict[str, Any]) -> dict[str, Any]:
        document = {
            "_id": uuid4().hex,
            "owner_id": owner_id,
            "created_at": datetime.now(timezone.utc),
            **report,
        }
        await self.collection.insert_one(document)
        return document

    async def list_for_user(self, owner_id: str, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.collection.find({"owner_id": owner_id}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
