from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo.asynchronous.collection import AsyncCollection


class EvaluationRepository:
    def __init__(self, collection: AsyncCollection[dict[str, Any]]):
        self.collection = collection

    async def ensure_indexes(self) -> None:
        await self.collection.create_index([("owner_id", 1), ("created_at", -1)])
        await self.collection.create_index([("owner_id", 1), ("version", 1)], unique=True,
            partialFilterExpression={"kind": "benchmark_version"})

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
        cursor = self.collection.find({"owner_id": owner_id, "kind": {"$exists": False}}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def create_benchmark_report(self, owner_id: str, report: dict[str, Any]) -> dict[str, Any]:
        return await self.create(owner_id, {"kind": "benchmark_report", **report})

    async def create_benchmark_version(self, owner_id: str, benchmark: dict[str, Any], fingerprint: str):
        return await self.create(owner_id, {"kind": "benchmark_version", "version": benchmark["version"],
                                            "fingerprint": fingerprint, "benchmark": benchmark})

    async def find_benchmark_version(self, owner_id: str, version: str):
        return await self.collection.find_one({"owner_id": owner_id, "kind": "benchmark_version", "version": version})

    async def find_benchmark_report(self, owner_id: str, report_id: str):
        return await self.collection.find_one({"_id": report_id, "owner_id": owner_id, "kind": "benchmark_report"})

    async def list_benchmark_reports(self, owner_id: str, limit: int = 20):
        cursor = self.collection.find({"owner_id": owner_id, "kind": "benchmark_report"}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def acquire_resume(self, owner_id: str, report_id: str, token: str, now: datetime, expires: datetime):
        from pymongo import ReturnDocument
        return await self.collection.find_one_and_update(
            {"_id": report_id, "owner_id": owner_id, "kind": "benchmark_report",
             "$or": [{"resume_expires": {"$exists": False}}, {"resume_expires": {"$lte": now}}]},
            {"$set": {"resume_token": token, "resume_expires": expires}}, return_document=ReturnDocument.AFTER)

    async def save_resume(self, owner_id: str, report_id: str, token: str, results: list[dict], metrics: dict, cooldown_until):
        result = await self.collection.update_one({"_id": report_id, "owner_id": owner_id,
            "kind": "benchmark_report", "resume_token": token}, {"$set": {"results": results, "metrics": metrics,
            "cooldown_until": cooldown_until}})
        return result.matched_count == 1

    async def release_resume(self, owner_id: str, report_id: str, token: str):
        await self.collection.update_one({"_id": report_id, "owner_id": owner_id, "resume_token": token},
                                         {"$unset": {"resume_token": "", "resume_expires": ""}})

    async def update_benchmark_report(self, owner_id: str, report_id: str, metrics: dict, judgments: list[dict]):
        result = await self.collection.update_one({"_id": report_id, "owner_id": owner_id, "kind": "benchmark_report"},
                                                  {"$set": {"metrics": metrics, "judgments": judgments}})
        return result.matched_count == 1

    async def create_proposals(self, owner_id: str, paper_id: str, cases: list[dict]):
        return await self.create(owner_id, {"kind": "evaluation_proposals", "paper_id": paper_id,
                                            "revision": 0, "cases": cases})

    async def find_proposals(self, owner_id: str, proposal_id: str):
        return await self.collection.find_one({"_id": proposal_id, "owner_id": owner_id,
                                               "kind": "evaluation_proposals"})

    async def list_proposals(self, owner_id: str, paper_id: str):
        cursor = self.collection.find({"owner_id": owner_id, "paper_id": paper_id,
                                       "kind": "evaluation_proposals"}).sort("created_at", -1).limit(20)
        return await cursor.to_list(length=20)

    async def replace_proposals(self, owner_id: str, proposal_id: str, revision: int, cases: list[dict]):
        result = await self.collection.update_one({"_id": proposal_id, "owner_id": owner_id,
            "kind": "evaluation_proposals", "revision": revision},
            {"$set": {"cases": cases}, "$inc": {"revision": 1}})
        return result.modified_count == 1
