from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo.asynchronous.collection import AsyncCollection
from ..services.answer_quality import SUMMARY_METRICS


class ConversationRepository:
    def __init__(self, collection: AsyncCollection[dict[str, Any]]):
        self.collection = collection

    async def ensure_indexes(self) -> None:
        await self.collection.create_index([("owner_id", 1), ("updated_at", -1)])
        await self.collection.create_index([("owner_id", 1), ("paper_id", 1), ("updated_at", -1)])

    async def create(self, owner_id: str, paper_id: str, message: dict[str, Any],
                     paper_filename: str | None = None) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        document = {
            "_id": uuid4().hex,
            "owner_id": owner_id,
            "paper_id": paper_id,
            "paper_filename": paper_filename,
            "title": message["question"],
            "pinned": False,
            "saved": True,
            "created_at": now,
            "updated_at": now,
            "messages": [message],
        }
        await self.collection.insert_one(document)
        return document

    async def find(self, owner_id: str, conversation_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"_id": conversation_id, "owner_id": owner_id})

    async def quality_monitor(self, owner_id, *, paper_id=None, start=None, end=None,
                              evidence=None, state=None, priority=None, limit=25, offset=0):
        owned = {"owner_id": owner_id, "messages.quality_evaluations.0": {"$exists": True}}
        if paper_id:
            owned["paper_id"] = paper_id
        filters = {}
        if start or end:
            filters["evaluation.evaluated_at"] = {**({"$gte": start} if start else {}), **({"$lt": end} if end else {})}
        if evidence:
            filters["evaluation.evidence_type"] = evidence
        if priority:
            filters[f"evaluation.{priority.lower()}_status"] = {"$ne": "NOT_APPLICABLE"}
        if state:
            filters["evaluation.state"] = state
        metric_fields = {}
        for name in SUMMARY_METRICS:
            root = f"$evaluation.metrics.{name}"
            metric_fields.update({f"{name}_average": {"$avg": f"{root}.score"},
                f"{name}_measured": {"$sum": {"$cond": [{"$isNumber": f"{root}.score"}, 1, 0]}},
                f"{name}_numerator": {"$sum": f"{root}.numerator"},
                f"{name}_denominator": {"$sum": f"{root}.denominator"}})
        state_fields = {key.lower(): {"$sum": {"$cond": [{"$eq": ["$evaluation.state", key]}, 1, 0]}}
                        for key in ("COMPLETE", "PENDING", "FAILED", "UNAVAILABLE")}
        pipeline = [{"$match": owned}, {"$unwind": "$messages"},
                    {"$project": {"paper_id": 1, "paper_filename": 1,
                                  "message_id": "$messages.id", "question": "$messages.question",
                                  "evaluation": "$messages.quality_evaluations"}},
                    {"$unwind": "$evaluation"}, {"$match": filters},
                    {"$facet": {
                        "summary": [{"$group": {"_id": None, "total_versions": {"$sum": 1}, **metric_fields, **state_fields}}],
                        "answers": [{"$match": {"evaluation.state": "COMPLETE"}},
                                    {"$group": {"_id": {"conversation": "$_id", "message": "$message_id"}}},
                                    {"$count": "total"}],
                        "rows": [{"$sort": {"evaluation.evaluated_at": -1, "_id": 1, "message_id": 1, "evaluation.revision": -1}},
                                 {"$skip": offset}, {"$limit": limit}, {"$project": {
                                     "_id": 0, "conversation_id": "$_id", "message_id": 1, "paper_id": 1, "paper_filename": 1,
                                     "question": {"$substrCP": ["$question", 0, 160]}, "revision": "$evaluation.revision",
                                     "evaluated_at": "$evaluation.evaluated_at", "evidence_type": "$evaluation.evidence_type",
                                     "state": "$evaluation.state", "high_status": "$evaluation.high_status",
                                     "medium_status": "$evaluation.medium_status", "metrics": "$evaluation.metrics",
                                 }}],
                    }}]
        cursor = await self.collection.aggregate(pipeline)
        result = await cursor.next()
        totals = next(iter(result["summary"]), {})
        return {"rows": result["rows"], "total_versions": totals.get("total_versions", 0),
                "evaluated_answers": next(iter(result["answers"]), {}).get("total", 0),
                "counts": {key: totals.get(key, 0) for key in state_fields},
                "metrics": {name: {key: totals.get(f"{name}_{key}", None if key == "average" else 0)
                                   for key in ("average", "measured", "numerator", "denominator")}
                            for name in SUMMARY_METRICS}}

    async def quality_detail(self, owner_id, conversation_id, message_id, revision):
        record = await self.collection.find_one(
            {"owner_id": owner_id, "_id": conversation_id, "messages.id": message_id},
            {"paper_id": 1, "paper_filename": 1, "messages": {"$elemMatch": {"id": message_id}}})
        if record and record.get("messages"):
            message = record["messages"][0]
            evaluation = next((item for item in message.get("quality_evaluations", []) if item["revision"] == revision), None)
            if evaluation is not None:
                return {**evaluation, "question": message["question"], "paper_id": record["paper_id"],
                        "paper_filename": record.get("paper_filename")}
        return None

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

    async def update_verified_message(self, owner_id: str, conversation_id: str, message_id: str,
                                      fields: dict[str, Any]) -> bool:
        result = await self.collection.update_one(
            {"_id": conversation_id, "owner_id": owner_id,
             "messages": {"$elemMatch": {"id": message_id, "verification_status": "unavailable"}}},
            {"$set": {f"messages.$.{key}": value for key, value in fields.items()}},
        )
        return result.matched_count == 1

    async def list_for_user(self, owner_id: str) -> list[dict[str, Any]]:
        cursor = self.collection.find(
            {"owner_id": owner_id}, {"messages": {"$slice": -1}},
        ).sort("updated_at", -1)
        return await cursor.to_list(length=None)

    async def update_metadata(self, owner_id: str, conversation_id: str,
                              fields: dict[str, str | bool]) -> bool:
        result = await self.collection.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {**fields, "updated_at": datetime.now(timezone.utc)}},
        )
        return result.matched_count == 1

    async def remove(self, owner_id: str, conversation_id: str) -> bool:
        result = await self.collection.delete_one({"_id": conversation_id, "owner_id": owner_id})
        return result.deleted_count == 1
