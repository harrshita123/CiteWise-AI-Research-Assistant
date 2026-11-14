from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo.asynchronous.collection import AsyncCollection


class ConversationRepository:
    def __init__(self, collection: AsyncCollection[dict[str, Any]]):
        self.collection = collection

    async def ensure_indexes(self) -> None:
        await self.collection.create_index([("owner_id", 1), ("updated_at", -1)])
        await self.collection.create_index([("owner_id", 1), ("paper_id", 1), ("updated_at", -1)])

    async def create(self, owner_id: str, paper_id: str, message: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        document = {
            "_id": uuid4().hex,
            "owner_id": owner_id,
            "paper_id": paper_id,
            "created_at": now,
            "updated_at": now,
            "messages": [message],
        }
        await self.collection.insert_one(document)
        return document

    async def find(self, owner_id: str, conversation_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"_id": conversation_id, "owner_id": owner_id})

    async def append(self, owner_id: str, conversation_id: str, paper_id: str,
                     message: dict[str, Any]) -> bool:
        result = await self.collection.update_one(
            {"_id": conversation_id, "owner_id": owner_id, "paper_id": paper_id},
            {"$push": {"messages": {"$each": [message], "$slice": -50}},
             "$set": {"updated_at": datetime.now(timezone.utc)}},
        )
        return result.matched_count == 1

    async def set_feedback(self, owner_id: str, conversation_id: str, message_id: str,
                           feedback: dict[str, str | None]) -> bool:
        result = await self.collection.update_one(
            {"_id": conversation_id, "owner_id": owner_id, "messages.id": message_id},
            {"$set": {"messages.$.feedback": feedback}},
        )
        return result.matched_count == 1

    async def list_for_user(self, owner_id: str, limit: int = 30) -> list[dict[str, Any]]:
        cursor = self.collection.find(
            {"owner_id": owner_id}, {"messages": {"$slice": -1}},
        ).sort("updated_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def remove(self, owner_id: str, conversation_id: str) -> bool:
        result = await self.collection.delete_one({"_id": conversation_id, "owner_id": owner_id})
        return result.deleted_count == 1
