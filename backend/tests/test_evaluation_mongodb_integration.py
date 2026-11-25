"""Optional isolated MongoDB integration: set CITEWISE_TEST_MONGO_URI to run."""
import os
import unittest
from uuid import uuid4

from pymongo import AsyncMongoClient
from pymongo.errors import DuplicateKeyError

from app.repositories.evaluations import EvaluationRepository


@unittest.skipUnless(os.environ.get("CITEWISE_TEST_MONGO_URI"), "isolated MongoDB URI not configured")
class MongoEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncMongoClient(os.environ["CITEWISE_TEST_MONGO_URI"], serverSelectionTimeoutMS=3000)
        self.database = "citewise_evaluation_test_" + uuid4().hex
        self.repo = EvaluationRepository(self.client[self.database]["evaluation_runs"])
        await self.repo.ensure_indexes()

    async def asyncTearDown(self):
        await self.client.drop_database(self.database)
        await self.client.close()

    async def test_owner_revision_and_version_uniqueness(self):
        draft = await self.repo.create_proposals("alice", "p1", [{"status": "pending"}])
        self.assertIsNone(await self.repo.find_proposals("bob", draft["_id"]))
        self.assertTrue(await self.repo.replace_proposals("alice", draft["_id"], 0, [{"status": "approved"}]))
        self.assertFalse(await self.repo.replace_proposals("alice", draft["_id"], 0, []))
        benchmark = {"version": "v1", "cases": []}
        await self.repo.create_benchmark_version("alice", benchmark, "fingerprint")
        with self.assertRaises(DuplicateKeyError):
            await self.repo.create_benchmark_version("alice", benchmark, "other")
        self.assertIsNone(await self.repo.find_benchmark_version("bob", "v1"))
        self.assertEqual((await self.repo.find_benchmark_version("alice", "v1"))["fingerprint"], "fingerprint")
        self.assertEqual(await self.repo.list_for_user("alice"), [])
