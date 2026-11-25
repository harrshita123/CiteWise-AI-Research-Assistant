from datetime import datetime, timezone
from typing import Any
from datetime import timedelta
from uuid import uuid4
from pymongo import ReturnDocument

from pymongo.asynchronous.collection import AsyncCollection

from ..paper_models import PaperResponse
from ..services.publication_identity import extracted_paper_metadata
from ..services.verification_gates import VerificationReport, document_quality


class PaperRepository:
    def __init__(self, collection: AsyncCollection[dict[str, Any]], lease_seconds: int = 900):
        self.collection = collection
        self.lease_seconds = lease_seconds

    async def ensure_indexes(self) -> None:
        await self.collection.create_index([("owner_id", 1), ("file_hash", 1)], unique=True)
        await self.collection.create_index([("owner_id", 1), ("created_at", -1)])
        await self.collection.database["paper_verification_rejections"].create_index(
            "created_at", expireAfterSeconds=30 * 86400)

    async def record_rejection(self, owner_id, checks):
        await self.collection.database["paper_verification_rejections"].insert_one({
            "owner_id": owner_id, "state": "REJECTED", "verification_checks": checks,
            "created_at": datetime.now(timezone.utc),
        })

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

    async def mark_nonresearch(self, paper, category):
        return await self.collection.find_one_and_update(
            {"_id": paper["_id"], "owner_id": paper["owner_id"], "file_hash": paper["file_hash"],
             "status": "ready", "document_type": {"$exists": False}},
            {"$set": {"status": "not_research", "document_type": "not_research_paper",
                      "document_category": category, "updated_at": datetime.now(timezone.utc)},
             "$unset": {"publication_identity": ""}},
            return_document=ReturnDocument.AFTER)

    async def claim_identity_job(self):
        now = datetime.now(timezone.utc)
        token = uuid4().hex
        return await self.collection.find_one_and_update(
            {"status": "ready", "identity_input": {"$exists": True},
             "$or": [{"publication_identity.status": "pending", "$or": [
                         {"publication_identity.next_retry_at": {"$exists": False}},
                         {"publication_identity.next_retry_at": {"$lte": now}}]},
                     {"publication_identity.status": "checking", "publication_identity.lease_until": {"$lte": now}}]},
            {"$set": {"publication_identity.status": "checking", "publication_identity.lease_token": token,
                      "publication_identity.lease_until": now + timedelta(seconds=self.lease_seconds)},
             "$inc": {"publication_identity.attempts": 1}},
            return_document=ReturnDocument.AFTER)

    async def finish_identity_job(self, paper, identity, *, identity_input=None):
        fields = {"publication_identity": identity, "updated_at": datetime.now(timezone.utc)}
        if identity_input is not None:
            fields["identity_input"] = identity_input
        report = VerificationReport("document", paper.get("verification_checks"))
        document_quality(report, extracted_paper_metadata(identity_input if identity_input is not None else
                         paper.get("identity_input") or {}), identity, usable_text=paper.get("chunk_count", 0) > 0)
        fields["verification_checks"] = report.dump()
        result = await self.collection.update_one(
            {"_id": paper["_id"], "owner_id": paper["owner_id"], "file_hash": paper["file_hash"],
             "status": "ready", "publication_identity.status": "checking",
             "publication_identity.lease_token": paper["publication_identity"]["lease_token"]},
            {"$set": fields})
        return result.modified_count == 1

    async def retry_identity_job(self, owner_id, paper_id):
        result = await self.collection.update_one(
            {"_id": paper_id, "owner_id": owner_id, "status": "ready", "identity_input": {"$exists": True},
             "publication_identity.status": {"$in": ["verification_unavailable", "not_verified", "conflicting", "registry_record"]},
             "$or": [{"publication_identity.retry_after": {"$exists": False}},
                     {"publication_identity.retry_after": {"$lte": datetime.now(timezone.utc)}}]},
            {"$set": {"publication_identity": {"status": "pending", "refresh_extraction": True, "reason": "Publication identity check pending."},
                      "updated_at": datetime.now(timezone.utc)}})
        return result.modified_count == 1

    async def backfill_identity_job(self, owner_id, paper_id, file_hash):
        result = await self.collection.update_one(
            {"_id": paper_id, "owner_id": owner_id, "file_hash": file_hash, "status": "ready",
             "identity_input": {"$exists": False},
             "$and": [
                 {"$or": [{"publication_identity": {"$exists": False}},
                          {"publication_identity.status": {"$in": ["verification_unavailable", "not_verified", "conflicting", "registry_record"]}}]},
                 {"$or": [{"publication_identity.retry_after": {"$exists": False}},
                          {"publication_identity.retry_after": {"$lte": datetime.now(timezone.utc)}}]},
             ]},
            {"$set": {"identity_input": {},
                      "publication_identity": {"status": "pending", "refresh_extraction": True, "reason": "Publication identity check pending."},
                      "updated_at": datetime.now(timezone.utc)}})
        return result.modified_count == 1


def paper_response(paper: dict[str, Any]) -> PaperResponse:
    fields = {key: value for key, value in paper.items() if key != "_id"}
    if identity := fields.get("publication_identity"):
        # PyMongo decodes BSON timestamps as naive UTC by default. Include the
        # offset so browsers compute retry deadlines in the correct time zone.
        fields["publication_identity"] = {**identity, **{
            key: value.replace(tzinfo=timezone.utc) for key in ("retry_after", "next_retry_at")
            if isinstance(value := identity.get(key), datetime) and value.tzinfo is None
        }}
    return PaperResponse(id=paper["_id"], **fields)
