"""Optional integration test: point CITEWISE_IDENTITY_TEST_MONGO_URI at a test MongoDB."""
import asyncio
import os
import unittest
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pymongo import AsyncMongoClient
from app.repositories.papers import PaperRepository
from app.services.provider_coordination import ProviderCoordinator
from pymongo import MongoClient


@unittest.skipUnless(os.getenv("CITEWISE_IDENTITY_TEST_MONGO_URI"), "test MongoDB not configured")
class IdentityMongoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncMongoClient(os.environ["CITEWISE_IDENTITY_TEST_MONGO_URI"], serverSelectionTimeoutMS=2000)
        self.database_name = "citewise_identity_test_" + uuid4().hex
        self.collection = self.client[self.database_name]["papers"]
        self.repo = PaperRepository(self.collection)
        await self.collection.insert_one({"_id": "one", "owner_id": "alice", "file_hash": "hash1",
            "status": "ready", "identity_input": {"first": "test"}, "publication_identity": {"status": "pending"}})

    async def asyncTearDown(self):
        await self.client.drop_database(self.database_name)
        await self.client.close()

    async def test_concurrent_claim_restart_recovery_and_stale_result(self):
        first, other = await asyncio.gather(self.repo.claim_identity_job(), self.repo.claim_identity_job())
        job = first or other
        self.assertEqual(sum(item is not None for item in (first, other)), 1)
        await self.collection.update_one({"_id": "one"}, {"$set": {"publication_identity.lease_until": datetime.now(timezone.utc) - timedelta(seconds=1)}})
        restarted = PaperRepository(self.collection)
        recovered = await restarted.claim_identity_job()
        self.assertIsNotNone(recovered)
        self.assertFalse(await self.repo.finish_identity_job(job, {"status": "verified_published"}))
        self.assertTrue(await restarted.finish_identity_job(recovered, {"status": "verification_unavailable"}))
        self.assertFalse(await restarted.retry_identity_job("bob", "one"))
        self.assertTrue(await restarted.retry_identity_job("alice", "one"))
        self.assertFalse(await restarted.retry_identity_job("alice", "one"))

    async def test_real_worker_process_restart(self):
        env = {**os.environ, "IDENTITY_TEST_DB": self.database_name, "IDENTITY_TEST_BLOCK": "1"}
        script = str(Path(__file__).with_name("identity_worker_process.py"))
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        first = await asyncio.create_subprocess_exec(sys.executable, script, env=env)
        try:
            for _ in range(100):
                job = await self.collection.find_one({"_id": "one"})
                if job["publication_identity"]["status"] == "checking":
                    break
                await asyncio.sleep(.05)
            self.assertEqual(job["publication_identity"]["status"], "checking")
            first.kill()
            await first.wait()
            env.pop("IDENTITY_TEST_BLOCK")
            second = await asyncio.create_subprocess_exec(sys.executable, script, env=env)
            try:
                self.assertEqual(await asyncio.wait_for(second.wait(), 10), 0)
            finally:
                if second.returncode is None:
                    second.kill()
                    await second.wait()
            completed = await self.collection.find_one({"_id": "one"})
            self.assertEqual(completed["publication_identity"]["status"], "verification_unavailable")
            self.assertEqual((completed["_id"], completed["owner_id"], completed["file_hash"]), ("one", "alice", "hash1"))
        finally:
            if first.returncode is None:
                first.kill()
                await first.wait()

    async def test_shared_provider_leases_cooldowns_and_stale_release(self):
        sync = MongoClient(os.environ["CITEWISE_IDENTITY_TEST_MONGO_URI"])
        try:
            one = ProviderCoordinator(sync[self.database_name]["providers"])
            two = ProviderCoordinator(sync[self.database_name]["providers"])
            token = one.acquire("arxiv")
            self.assertIsNotNone(token)
            self.assertIsNone(two.acquire("arxiv"))
            one.defer("arxiv", 30)
            one.release("arxiv", token, 3)
            self.assertIsNone(two.acquire("arxiv"))
            self.assertGreater(two.retry_delay(), 20)
            first_token = two.acquire("crossref")
            self.assertIsNotNone(first_token)
            sync[self.database_name]["providers"].update_one({"_id": "crossref"},
                {"$set": {"lease_until": datetime.now(timezone.utc) - timedelta(seconds=1)}})
            second_token = one.acquire("crossref")
            self.assertIsNotNone(second_token)
            two.release("crossref", first_token, 0)
            self.assertIsNone(two.acquire("crossref"))
            one.release("crossref", second_token, 0)
        finally:
            sync.close()

    async def test_two_real_workers_renew_and_serialize_long_requests(self):
        await self.collection.update_one({"_id": "one"}, {"$set": {"identity_input.id": "first"}})
        await self.collection.insert_one({"_id": "two", "owner_id": "bob", "file_hash": "hash2", "status": "ready",
            "identity_input": {"id": "second"}, "publication_identity": {"status": "pending"}})
        env = {**os.environ, "IDENTITY_TEST_DB": self.database_name,
               "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
        script = str(Path(__file__).with_name("provider_worker_process.py"))
        workers = [await asyncio.create_subprocess_exec(sys.executable, script, env=env) for _ in range(2)]
        try:
            self.assertEqual(await asyncio.wait_for(asyncio.gather(*(p.wait() for p in workers)), 12), [0, 0])
            events = await self.client[self.database_name]["events"].find({}).sort("time", 1).to_list(length=10)
            self.assertEqual([e["event"] for e in events], ["start", "end", "start", "end"])
            self.assertNotEqual(events[0]["worker"], events[2]["worker"])
            self.assertGreaterEqual(events[2]["time"] - events[1]["time"], .2)
            papers = await self.collection.find({}).to_list(length=2)
            self.assertEqual({(p["_id"], p["owner_id"], p["file_hash"], p["publication_identity"]["status"]) for p in papers},
                {("one", "alice", "hash1", "verification_unavailable"), ("two", "bob", "hash2", "verification_unavailable")})
        finally:
            for process in workers:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    async def test_two_processes_share_same_provider_response(self):
        env = {**os.environ, "IDENTITY_TEST_DB": self.database_name,
               "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
        script = str(Path(__file__).with_name("provider_worker_process.py"))
        for provider in ("crossref", "arxiv"):
            await self.collection.delete_many({})
            await self.client[self.database_name]["events"].delete_many({})
            await self.client[self.database_name]["providers"].delete_many({})
            for identifier, owner in (("one", "alice"), ("two", "bob")):
                await self.collection.insert_one({"_id": identifier, "owner_id": owner, "file_hash": identifier,
                    "status": "ready", "identity_input": {"id": "same"}, "publication_identity": {"status": "pending"}})
            env["TEST_PROVIDER"] = provider
            workers = [await asyncio.create_subprocess_exec(sys.executable, script, env=env) for _ in range(2)]
            try:
                self.assertEqual(await asyncio.wait_for(asyncio.gather(*(p.wait() for p in workers)), 12), [0, 0])
                events = await self.client[self.database_name]["events"].find({}).to_list(length=10)
                self.assertEqual([e["event"] for e in events], ["start", "end"])
                self.assertEqual(await self.collection.count_documents({"publication_identity.status": "verification_unavailable"}), 2)
            finally:
                for process in workers:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()

    async def test_expired_and_stale_owner_cannot_publish(self):
        sync = MongoClient(os.environ["CITEWISE_IDENTITY_TEST_MONGO_URI"])
        try:
            collection = sync[self.database_name]["providers"]
            one, two = ProviderCoordinator(collection), ProviderCoordinator(collection)
            old = one.acquire("crossref")
            collection.update_one({"_id": "crossref"}, {"$set": {"lease_until": datetime.now(timezone.utc) - timedelta(seconds=1)}})
            new = two.acquire("crossref")
            self.assertTrue(two.publish("crossref", new, "key", b"new"))
            self.assertTrue(two.publish("crossref", new, "other", b"another paper"))
            self.assertEqual(two.cached("crossref", "other"), b"another paper")
            self.assertFalse(one.publish("crossref", old, "key", b"stale"))
            self.assertEqual(one.cached("crossref", "key"), b"new")
            collection.update_one({"_id": "crossref"}, {"$set": {"entries.0.expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}})
            self.assertIsNone(two.cached("crossref", "key"))
        finally:
            sync.close()

    async def test_similar_titles_and_pagination_have_independent_entries(self):
        from unittest.mock import patch
        from app.services.paper_discovery import AcademicClient

        with MongoClient(os.environ["CITEWISE_IDENTITY_TEST_MONGO_URI"]) as mongo:
            coordinator = ProviderCoordinator(mongo[self.database_name]["providers"])
            client = AcademicClient(coordinator=coordinator)
            with patch.object(client, "_get_local", return_value=b'{"results": []}') as fetch:
                def lookup(title, page=1):
                    client.get("openalex", "/works", {"page": page, "search": title})

                lookup("Ice loss")
                lookup("Ice loss?")
                lookup("Ice loss", 2)
                lookup(" Ice loss ")
                self.assertEqual(fetch.call_count, 3)
                for i in range(6):
                    lookup("unrelated", i)
                self.assertEqual(len(coordinator.collection.find_one()["entries"]), 6)
                lookup("Ice loss")
                self.assertEqual(fetch.call_count, 10)
