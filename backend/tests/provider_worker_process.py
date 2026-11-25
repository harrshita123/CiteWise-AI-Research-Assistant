"""Test-only worker: real job claim and provider lease, offline fake HTTP."""
import asyncio
import os
import time
from pymongo import AsyncMongoClient, MongoClient
from app.repositories.papers import PaperRepository
from app.services import identity_worker, paper_discovery
from app.services.provider_coordination import ProviderCoordinator

uri, db = os.environ["CITEWISE_IDENTITY_TEST_MONGO_URI"], os.environ["IDENTITY_TEST_DB"]
sync = MongoClient(uri)
coordinator = ProviderCoordinator(sync[db]["providers"], lease_seconds=1)
academic = paper_discovery.AcademicClient(coordinator=coordinator)


class Response:
    status_code, headers = 200, {}
    def __enter__(self): return self
    def __exit__(self, *_): return None
    def iter_content(self, _size):
        sync[db]["events"].insert_one({"worker": os.getpid(), "event": "start", "time": time.time()})
        time.sleep(1.5)
        sync[db]["events"].insert_one({"worker": os.getpid(), "event": "end", "time": time.time()})
        yield (b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
               if os.environ.get("TEST_PROVIDER") == "arxiv" else b'{"message": {}}')


paper_discovery.requests.get = lambda *args, **kwargs: Response()
def fake_verify(identity_input, _discovery):
    provider = os.environ.get("TEST_PROVIDER", "crossref")
    if provider == "arxiv":
        academic.get(provider, "/api/query", {"id_list": identity_input["id"]})
    else:
        academic.get(provider, "/works/" + identity_input["id"])
    return {"status": "verification_unavailable", "reason": "No reliable match."}
identity_worker.verify_identity_input = fake_verify


async def main():
    client = AsyncMongoClient(uri)
    await identity_worker.process_one(PaperRepository(client[db]["papers"]), object())
    await client.close()
    sync.close()

asyncio.run(main())
