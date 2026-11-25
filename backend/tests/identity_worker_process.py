"""Test-only standalone worker with deterministic, offline verification."""
import asyncio
import os
import time

from pymongo import AsyncMongoClient
from app.repositories.papers import PaperRepository
from app.services import identity_worker


def fake_verify(_input, _discovery):
    if os.environ.get("IDENTITY_TEST_BLOCK"):
        time.sleep(60)
    return {"status": "verification_unavailable", "reason": "No reliable match."}


async def main():
    client = AsyncMongoClient(os.environ["CITEWISE_IDENTITY_TEST_MONGO_URI"])
    repo = PaperRepository(client[os.environ["IDENTITY_TEST_DB"]]["papers"], lease_seconds=2)
    identity_worker.verify_identity_input = fake_verify
    while not await identity_worker.process_one(repo, object()):
        await asyncio.sleep(.1)
    await client.close()


asyncio.run(main())
