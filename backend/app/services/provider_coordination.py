"""MongoDB-backed academic-provider leases and cooldowns."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from threading import Event, Thread
from contextlib import contextmanager

from pymongo import ReturnDocument


class ProviderCoordinator:
    def __init__(self, collection, lease_seconds=300):
        self.collection = collection
        self.lease_seconds = lease_seconds

    def acquire(self, provider):
        now = datetime.now(timezone.utc)
        self.collection.update_one({"_id": provider}, {"$setOnInsert": {"next_allowed": now, "lease_until": now}}, upsert=True)
        token = uuid4().hex
        item = self.collection.find_one_and_update(
            {"_id": provider, "next_allowed": {"$lte": now}, "lease_until": {"$lte": now}},
            {"$set": {"lease_token": token, "lease_until": now + timedelta(seconds=self.lease_seconds)}},
            return_document=ReturnDocument.AFTER)
        return token if item else None

    def renew(self, provider, token):
        now = datetime.now(timezone.utc)
        result = self.collection.update_one(
            {"_id": provider, "lease_token": token, "$expr": {"$gt": ["$lease_until", "$$NOW"]}},
            {"$set": {"lease_until": now + timedelta(seconds=self.lease_seconds)}})
        return result.matched_count == 1

    def cached(self, provider, key):
        now = datetime.now(timezone.utc)
        match = {"key": key, "expires_at": {"$gt": now}}
        item = self.collection.find_one({"_id": provider}, {"entries": {"$elemMatch": match}})
        entries = (item or {}).get("entries", [])
        return bytes(entries[0]["data"]) if entries else None

    def publish(self, provider, token, key, data):
        # Six entries at the transport's 2 MB limit fit inside one BSON document.
        # Server time fences a delayed write even if the client clock is behind.
        now = datetime.now(timezone.utc)
        entry = {"key": key, "data": data, "provider": provider, "retrieved_at": now,
                 "expires_at": now + timedelta(minutes=5)}
        retained = {"$filter": {"input": {"$ifNull": ["$entries", []]}, "as": "e",
                    "cond": {"$and": [{"$ne": ["$$e.key", key]}, {"$gt": ["$$e.expires_at", "$$NOW"]}]}}}
        result = self.collection.update_one(
            {"_id": provider, "lease_token": token, "$expr": {"$gt": ["$lease_until", "$$NOW"]}},
            [{"$set": {"entries": {"$slice": [{"$concatArrays": [retained, {"$literal": [entry]}]}, -6]}}},
             {"$unset": "cache"}])
        return result.matched_count == 1

    @contextmanager
    def keep_alive(self, provider, token):
        stop, lost = Event(), Event()
        def heartbeat():
            while not stop.wait(max(0.1, self.lease_seconds / 3)):
                try:
                    if not self.renew(provider, token):
                        lost.set()
                        return
                except Exception:
                    lost.set()
                    return
        thread = Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield lost
        finally:
            stop.set()
            thread.join()

    def release(self, provider, token, interval):
        now = datetime.now(timezone.utc)
        self.collection.update_one({"_id": provider, "lease_token": token},
            {"$set": {"lease_until": now}, "$max": {"next_allowed": now + timedelta(seconds=interval)},
             "$unset": {"lease_token": ""}})

    def defer(self, provider, seconds, token=None):
        query = {"_id": provider}
        if token:
            query.update({"lease_token": token, "lease_until": {"$gt": datetime.now(timezone.utc)}})
        return self.collection.update_one(query,
            {"$max": {"next_allowed": datetime.now(timezone.utc) + timedelta(seconds=seconds)}})

    def retry_delay(self):
        now = datetime.now(timezone.utc)
        values = []
        for item in self.collection.find({}, {"next_allowed": 1}):
            deadline = item.get("next_allowed")
            if deadline:
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                values.append((deadline - now).total_seconds())
        return max([0.0, *values])
