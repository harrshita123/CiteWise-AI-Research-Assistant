"""Free-only routing policies and an account budget shared by API workers."""

import math
import time
from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError
from pymongo.write_concern import WriteConcern

from .answers import AnswerRateLimitError

GENERATION_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
VERIFICATION_MODEL = "inclusionai/ling-3.2-flash:free"
RECOMMENDATION_MODEL = VERIFICATION_MODEL
FAST_MODEL = "baidu/ernie-5.5-flash-preview:free"
PDF_FALLBACK_MODEL = "google/gemma-4-31b-it:free"
VISION_MODEL = "nvidia/nemotron-3.5-super-vl:free"
STRUCTURED_MODELS = frozenset({
    VERIFICATION_MODEL, "liquid/lfm-3-1.7b:free", FAST_MODEL, VISION_MODEL,
})
RERANK_MODEL = "nvidia/nemotron-3.5-rerank-8b:free"
RERANK_CANDIDATES = 15


@dataclass(frozen=True)
class ModelRoute:
    role: str
    model: str
    max_tokens: int
    timeout: int
    fallback: str = ""
    structured_recovery: bool = False


@dataclass(frozen=True)
class OutputLimits:
    simple: int = 1200
    research: int = 3500
    external: int = 3500
    assessment: int = 4096
    verification: int = 6144
    recommendations: int = 1536


class ModelRouter:
    def __init__(self, generation=GENERATION_MODEL, timeout=120, *,
                 verification=VERIFICATION_MODEL, recommendation=RECOMMENDATION_MODEL, fallback=None,
                 external_generation="", rerank=RERANK_MODEL, vision=VISION_MODEL,
                 structured_fallback=FAST_MODEL, output_limits=OutputLimits()):
        external_generation = external_generation or generation
        if not {generation, external_generation}.issubset(STRUCTURED_MODELS | {GENERATION_MODEL}):
            raise ValueError("Choose a catalog-verified free OpenRouter generation model.")
        if not {verification, recommendation}.issubset(STRUCTURED_MODELS):
            raise ValueError("Choose a catalog-verified free OpenRouter structured chat model.")
        if fallback and (fallback not in STRUCTURED_MODELS | {PDF_FALLBACK_MODEL} or fallback == generation):
            raise ValueError("Choose a distinct catalog-verified free OpenRouter fallback.")
        if rerank != RERANK_MODEL or vision != VISION_MODEL:
            raise ValueError("Choose a catalog-verified free OpenRouter rerank and vision model.")
        if structured_fallback not in ("", FAST_MODEL):
            raise ValueError("Choose the validated free structured recovery model or disable it.")
        def generation_fallback(primary, default):
            if fallback is None:
                return default if primary == GENERATION_MODEL else ""
            return fallback if fallback != primary else ""

        self.generation = ModelRoute("research_generation", generation, output_limits.research, timeout,
                                     generation_fallback(generation, PDF_FALLBACK_MODEL))
        self.external = ModelRoute("external_answer_generation", external_generation, output_limits.external, timeout,
                                   generation_fallback(external_generation, FAST_MODEL))
        self.simple_generation = ModelRoute(self.generation.role, FAST_MODEL, output_limits.simple,
                                            min(timeout, 45),
                                            self.generation.fallback if self.generation.fallback != FAST_MODEL else "")
        self.simple_external = ModelRoute(self.external.role, FAST_MODEL, output_limits.simple,
                                          min(timeout, 45),
                                          external_generation if fallback != "" and external_generation != FAST_MODEL else "")
        def structured(role, primary, tokens, seconds):
            alternate = structured_fallback if primary != structured_fallback else ""
            return ModelRoute(role, primary, tokens, seconds, alternate, structured_recovery=True)

        self.evidence = structured("evidence_assessment", verification, output_limits.assessment, timeout)
        self.verifier = structured("claim_verification", verification, output_limits.verification, timeout)
        self.suggestions = structured("question_recommendation", recommendation, output_limits.recommendations, min(timeout, 30))
        proposal_model = generation if generation in STRUCTURED_MODELS else VERIFICATION_MODEL
        self.evaluation = structured("evaluation_proposals", proposal_model, 8192, timeout)
        self.rerank = ModelRoute("retrieval_reranking", rerank, 0, 15)
        self.vision = ModelRoute("visual_document_analysis", vision, 4096, min(timeout, 45))

    def for_stage(self, stage, *, simple=False):
        return {
            "citewise_answer": self.simple_generation if simple else self.generation,
            "citewise_external_answer": self.simple_external if simple else self.external,
            "citewise_evidence_assessment": self.evidence,
            "citewise_evidence_reassessment": self.evidence,
            "citewise_claim_verification": self.verifier,
            "citewise_question_recommendations": self.suggestions,
            "citewise_evaluation_proposals": self.evaluation,
            "citewise_visual_analysis": self.vision,
        }[stage]


class UpstreamPoolLimit(AnswerRateLimitError):
    """Only the selected model's upstream pool is unavailable."""

    def __init__(self, retry_after: str | None = None):
        super().__init__(retry_after, limit_source="upstream_provider_shared_pool")


class SharedRequestBudget:
    """Atomic admission against two rolling windows and a shared upstream cooldown."""

    reservation_seconds = 30

    def __init__(self, collection, account_scope, daily_limit=50, minute_limit=20):
        # A worker must not send a request against a reservation that can roll back on failover.
        self.collection = collection.with_options(write_concern=WriteConcern("majority"))
        self.scope = {"_id": account_scope}
        self.daily_limit = daily_limit
        self.minute_limit = minute_limit

    @staticmethod
    def _model_cooldown_field(model: str) -> str:
        return "model_cooldowns." + sha256(model.encode()).hexdigest()

    @staticmethod
    def _release_reservations(condition, now, outcome):
        history = {"$ifNull": ["$attempts", []]}
        eligible = {"$and": [
            {"$eq": ["$$item.state", "reserved"]},
            {"$eq": [{"$type": "$$item.dispatched_at"}, "missing"]},
            condition,
        ]}
        released = {"$filter": {"input": history, "as": "item", "cond": eligible}}
        return {"$set": {
            "attempts": {"$filter": {"input": history, "as": "item", "cond": {"$not": [eligible]}}},
            "released_reservations": {"$slice": [{"$concatArrays": [
                {"$ifNull": ["$released_reservations", []]},
                {"$map": {"input": released, "as": "item", "in": {"$mergeObjects": [
                    "$$item", {"state": "failed_before_dispatch", "outcome": outcome, "released_at": now},
                ]}}},
            ]}, -128]},
        }}

    def reserve(self, model: str | None = None, **details) -> str:
        now = time.time()
        ticket = uuid4().hex
        entry = {**details, "at": now, "reserved_at": now, "expires_at": now + self.reservation_seconds,
                 "ticket": ticket, "state": "reserved", "model": model, "account_scope": self.scope["_id"]}
        expired = {"$and": [
            {"$isNumber": "$$item.expires_at"}, {"$lte": ["$$item.expires_at", now]},
        ]}
        daily = {"$filter": {"input": "$attempts", "as": "item", "cond": {"$gt": ["$$item.at", now - 86400]}}}
        minute = {"$filter": {"input": "$attempts", "as": "item",
                              "cond": {"$gt": ["$$item.at", now - 60]}}}
        admit = {"$and": [
            {"$lte": [{"$ifNull": ["$cooldown_until", 0]}, now]},
            {"$lt": [{"$size": "$attempts"}, self.daily_limit]},
            {"$lt": [{"$size": minute}, self.minute_limit]},
        ]}
        model_field = self._model_cooldown_field(model) if model else None
        if model_field:
            admit["$and"].append({"$lte": [{"$ifNull": [f"${model_field}", 0]}, now]})
        pipeline = [
            self._release_reservations(expired, now, "reservation_expired"),
            {"$set": {"attempts": daily}},
            {"$set": {"attempts": {"$cond": [admit,
                {"$concatArrays": ["$attempts", {"$literal": [entry]}]}, "$attempts"]}}},
        ]
        try:
            record = self.collection.find_one_and_update(
                self.scope, pipeline, upsert=True, return_document=ReturnDocument.AFTER)
        except DuplicateKeyError:
            record = self.collection.find_one_and_update(
                self.scope, pipeline, return_document=ReturnDocument.AFTER)
        attempts = record["attempts"]
        if any(item["ticket"] == ticket for item in attempts):
            return ticket
        recent = [item for item in attempts if item["at"] > now - 60]

        def window_reset(items, limit, seconds):
            ends = []
            for item in items:
                end = item["at"] + seconds
                if (item.get("state") == "reserved" and "dispatched_at" not in item
                        and isinstance(item.get("expires_at"), (int, float))):
                    end = min(end, item["expires_at"])
                ends.append(end)
            return sorted(ends)[-limit]

        deadlines = [(record.get("cooldown_until", 0), record.get("cooldown_source", "account_cooldown"))]
        if len(attempts) >= self.daily_limit:
            deadlines.append((window_reset(attempts, self.daily_limit, 86400), "local_daily_budget"))
        if len(recent) >= self.minute_limit:
            deadlines.append((window_reset(recent, self.minute_limit, 60), "local_minute_budget"))
        deadline, source = max(deadlines, key=lambda item: item[0])
        if deadline > now:
            raise AnswerRateLimitError(str(math.ceil(deadline - now)), limit_source=source)
        if model_field:
            model_deadline = record.get("model_cooldowns", {}).get(model_field.split(".", 1)[1], 0)
            if model_deadline > now:
                raise UpstreamPoolLimit(str(math.ceil(model_deadline - now)))
        raise AnswerRateLimitError()

    def dispatch(self, ticket: str) -> bool:
        now = time.time()
        result = self.collection.update_one(
            {**self.scope, "attempts": {"$elemMatch": {
                "ticket": ticket, "state": "reserved", "expires_at": {"$gt": now, "$type": "number"},
                "dispatched_at": {"$exists": False},
            }}},
            {"$set": {"attempts.$.state": "dispatched", "attempts.$.at": now,
                      "attempts.$.dispatched_at": now}},
        )
        return result.modified_count == 1

    def fail_before_dispatch(self, ticket: str) -> bool:
        result = self.collection.update_one(
            {**self.scope, "attempts": {"$elemMatch": {
                "ticket": ticket, "state": "reserved", "dispatched_at": {"$exists": False},
            }}},
            [self._release_reservations({"$eq": ["$$item.ticket", ticket]}, time.time(), "dispatch_not_started")],
        )
        return result.modified_count == 1

    def finish(self, ticket: str, outcome: str, **details) -> None:
        changes = {**details, "state": "unknown" if outcome == "transport_failed" else "completed",
                   "outcome": outcome, "finished_at": time.time()}
        self.collection.update_one(
            {**self.scope, "attempts": {"$elemMatch": {"ticket": ticket, "state": "dispatched"}}},
            {"$set": {f"attempts.$.{key}": value for key, value in changes.items()}},
        )

    def cool_down(self, seconds, *, model: str | None = None, limit_source: str = "account_cooldown"):
        deadline = time.time() + seconds
        if model:
            self.collection.update_one(self.scope, {"$max": {self._model_cooldown_field(model): deadline}})
        else:
            self.collection.update_one({**self.scope, "cooldown_until": {"$not": {"$gte": deadline}}},
                                       {"$set": {"cooldown_until": deadline, "cooldown_source": limit_source}})
