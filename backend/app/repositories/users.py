from datetime import datetime, timezone
from typing import Any

from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from bson.errors import InvalidId


class UserRepository:
    def __init__(self, collection: AsyncCollection[dict[str, Any]]):
        self.collection = collection

    async def ensure_indexes(self) -> None:
        indexes = await self.collection.index_information()
        google_index = indexes.get("google_sub_1")

        if google_index and not google_index.get("sparse"):
            await self.collection.drop_index("google_sub_1")

        await self.collection.create_index("google_sub", unique=True, sparse=True)
        await self.collection.create_index("email", unique=True)

    async def find_by_id(self, user_id: str) -> dict[str, Any] | None:
        try:
            object_id = ObjectId(user_id)
        except InvalidId:
            return None

        return await self.collection.find_one({"_id": object_id})

    async def find_by_email(self, email: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"email": email.lower()})

    async def list_users(self, limit: int, offset: int) -> list[dict[str, Any]]:
        cursor = (
            self.collection.find({})
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def create_password_user(self, name: str, email: str, password_hash: str) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)

        try:
            result = await self.collection.insert_one(
                {
                    "name": name,
                    "email": email.lower(),
                    "password_hash": password_hash,
                    "role": "user",
                    "created_at": now,
                    "updated_at": now,
                }
            )
        except DuplicateKeyError:
            return None

        return await self.collection.find_one({"_id": result.inserted_id})

    async def upsert_google_user(self, profile: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc)

        email = profile["email"].lower()
        user = await self.collection.find_one_and_update(
            {"email": email},
            {
                "$set": {
                    "google_sub": profile["sub"],
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "email": email,
                    "name": profile.get("name"),
                    "role": "user",
                    "created_at": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        if user is None:
            raise RuntimeError("Google user could not be created.")

        return user
