import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from pymongo.errors import DuplicateKeyError
from tokenizers import Tokenizer, models, pre_tokenizers

from app.config import Settings
from app.dependencies import get_current_user
from app.models import UserResponse
from app.repositories.papers import PaperRepository
from app.repositories.vectors import VectorRepository
from app.routers.papers import router
from app.services.ingestion import IngestionService
from app.services.pdf_storage import CloudinaryPdfStorage
from app.services.pdf_processing import PageText, PdfValidationError, create_chunks, extract_pages


def make_pdf(*pages: str) -> bytes:
    with pymupdf.open() as document:
        for text in pages:
            document.new_page().insert_textbox(pymupdf.Rect(40, 40, 550, 800), text)
        return document.tobytes()


def test_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


class FakeEmbeddings:
    model_name = "test-only"
    dimensions = 384

    def tokenizer(self):
        return test_tokenizer()

    def embed(self, texts):
        return [[1.0] + [0.0] * 383 for _ in texts]


class MemoryPapers:
    def __init__(self):
        self.papers = {}

    async def create(self, paper):
        if any(saved["owner_id"] == paper["owner_id"] and saved["file_hash"] == paper["file_hash"]
               for saved in self.papers.values()):
            raise DuplicateKeyError("duplicate")
        self.papers[paper["_id"]] = paper.copy()

    async def find(self, owner_id, paper_id):
        paper = self.papers.get(paper_id)
        return paper if paper and paper["owner_id"] == owner_id else None

    async def list_for_user(self, owner_id, limit, offset, status=None):
        return [paper for paper in self.papers.values()
                if paper["owner_id"] == owner_id and (status is None or paper["status"] == status)][offset:offset + limit]

    async def summary_for_user(self, owner_id):
        owned = [paper for paper in self.papers.values() if paper["owner_id"] == owner_id]
        return {"total": len(owned), **{status: sum(paper["status"] == status for paper in owned)
                                       for status in ("ready", "processing", "failed")}}

    async def update(self, owner_id, paper_id, **fields):
        paper = await self.find(owner_id, paper_id)
        if paper:
            paper.update(fields)

    async def remove(self, owner_id, paper_id):
        if await self.find(owner_id, paper_id):
            del self.papers[paper_id]


class PdfProcessingTests(unittest.TestCase):
    def test_extracts_real_pdf_with_original_page_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            path.write_bytes(make_pdf("Evidence on page one.", "", "Results on page three."))
            count, pages = extract_pages(path, 300, 10000)
        self.assertEqual(count, 3)
        self.assertEqual([page.page_number for page in pages], [1, 3])
        self.assertIn("Results", pages[1].text)

    def test_removes_repeated_running_margins_without_removing_body_text(self):
        with pymupdf.open() as document:
            for number in range(1, 4):
                page = document.new_page(width=600, height=800)
                page.insert_text((40, 40), "Research conference running header")
                page.insert_text((40, 120), f"Evidence on page {number}.")
                if number == 1:
                    page.insert_text((40, 150), "Resource Discovery in a Network Environment is the paper title.")
                if number == 2:
                    page.insert_text((40, 700), "Unique footnote with useful evidence.")
                page.insert_text((40, 745),
                                 f"Page | {number} Lorcan Dempsey. Resource Discovery in a Network Environment. Sept 2010")
            content = document.tobytes()

        count, pages = extract_pages(content, 300, 10000)
        chunks = create_chunks(pages, test_tokenizer())

        self.assertEqual(count, 3)
        self.assertEqual([page.page_number for page in pages], [1, 2, 3])
        self.assertTrue(all(f"Evidence on page {number}." in pages[number - 1].text
                            for number in range(1, 4)))
        self.assertIn("paper title", pages[0].text)
        self.assertIn("Unique footnote with useful evidence.", pages[1].text)
        self.assertTrue(all("running header" not in chunk.text and "Page |" not in chunk.text
                            for chunk in chunks))

    def test_chunks_overlap_without_crossing_pages_or_losing_tail(self):
        pages = [PageText(2, " ".join(f"word{i}" for i in range(500))), PageText(4, "Final evidence.")]
        tokenizer = test_tokenizer()
        chunks = create_chunks(pages, tokenizer)
        self.assertEqual(chunks[0].text.split()[-30:], chunks[1].text.split()[:30])
        self.assertTrue(chunks[-2].text.endswith("word499"))
        self.assertEqual(chunks[-1].page_number, 4)
        self.assertTrue(all(len(tokenizer.encode(chunk.text).ids) <= 220 for chunk in chunks))
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(len(chunks))))

    def test_rejects_scanned_encrypted_and_over_limit_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            cases = [(make_pdf(""), 300, 10000), (make_pdf("one", "two"), 1, 10000),
                     (make_pdf("too much text"), 300, 3)]
            with pymupdf.open(stream=make_pdf("Secret"), filetype="pdf") as document:
                cases.append((document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                                              owner_pw="owner", user_pw="reader"), 300, 10000))
            for content, max_pages, max_chars in cases:
                with self.subTest(max_pages=max_pages, max_chars=max_chars):
                    path.write_bytes(content)
                    with self.assertRaises(PdfValidationError):
                        extract_pages(path, max_pages, max_chars)


class PaperApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = Settings(_env_file=None, google_client_id="test", mongodb_uri="mongodb://localhost",
                                 jwt_secret="test-secret-" * 4, storage_directory=Path(self.directory.name),
                                 cloudinary_cloud_name="test-cloud", cloudinary_api_key="test-key",
                                 cloudinary_api_secret=SecretStr("test-secret"))
        upload_patch = patch("app.services.pdf_storage.cloudinary.uploader.upload",
                             side_effect=lambda _source, **options: {"public_id": options["public_id"]})
        destroy_patch = patch("app.services.pdf_storage.cloudinary.uploader.destroy", return_value={"result": "ok"})
        download_patch = patch("app.services.pdf_storage.cloudinary.utils.private_download_url",
                               return_value="https://api.cloudinary.com/signed-pdf")
        self.cloudinary_upload = upload_patch.start()
        self.cloudinary_destroy = destroy_patch.start()
        self.cloudinary_download = download_patch.start()
        self.addCleanup(upload_patch.stop)
        self.addCleanup(destroy_patch.stop)
        self.addCleanup(download_patch.stop)
        self.repository = MemoryPapers()
        self.ingestion = IngestionService(self.settings)
        self.ingestion.embeddings = FakeEmbeddings()
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.state.paper_repository = self.repository
        self.app.state.user_repository = None
        self.app.state.ingestion_service = self.ingestion
        self.app.state.upload_lock = asyncio.Lock()
        self.user_id = "alice"
        self.app.dependency_overrides[get_current_user] = lambda: UserResponse(
            id=self.user_id, email=f"{self.user_id}@example.com", role="user",
        )
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        settings_patch = patch("app.routers.papers.get_settings", return_value=self.settings)
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.content = make_pdf("Research evidence on page one.", "Methods and results on page two.")

    def upload(self, content=None, filename="paper.pdf"):
        return self.client.post("/api/v1/papers", files={"file": (filename, content or self.content, "application/pdf")})

    def test_upload_saves_pdf_chunks_and_vectors_and_survives_reopen(self):
        response = self.upload(filename="../../paper.pdf")
        self.assertEqual(response.status_code, 201, response.text)
        detail = response.json()
        paper = detail["paper"]
        self.assertEqual(paper["filename"], "paper.pdf")
        self.assertEqual(paper["status"], "ready")
        self.assertEqual(paper["page_count"], 2)
        self.assertEqual(paper["chunk_count"], 2)
        self.assertEqual([chunk["page_number"] for chunk in detail["chunks"]], [1, 2])
        self.assertEqual(self.repository.papers[paper["id"]]["storage_provider"], "cloudinary")
        self.assertEqual(self.cloudinary_upload.call_args.args[0].getvalue(), self.content)
        reopened = VectorRepository(self.settings.storage_directory / "chroma")
        self.assertEqual(len(reopened.preview("alice", paper["id"])), 2)
        vectors = reopened.collection.get(where=reopened.paper_filter("alice", paper["id"]), include=["embeddings"])
        self.assertEqual(vectors["embeddings"].shape, (2, 384))
        self.assertEqual(self.client.get(f'/api/v1/papers/{paper["id"]}').status_code, 200)
        second_chunk = self.client.get(f'/api/v1/papers/{paper["id"]}?chunk_offset=1&chunk_limit=1')
        self.assertEqual([chunk["chunk_index"] for chunk in second_chunk.json()["chunks"]], [1])
        self.assertEqual(self.client.get(f'/api/v1/papers/{paper["id"]}?chunk_offset=2').json()["chunks"], [])
        file_response = self.client.get(f'/api/v1/papers/{paper["id"]}/file', follow_redirects=False)
        self.assertEqual(file_response.status_code, 307)
        self.assertEqual(file_response.headers["location"], "https://api.cloudinary.com/signed-pdf")
        self.assertEqual(file_response.headers["cache-control"], "private, no-store")

    def test_duplicate_upload_does_not_create_more_vectors(self):
        self.assertEqual(self.upload().status_code, 201)
        duplicate = self.upload()
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(duplicate.json()["detail"], "This PDF is already in your library.")
        self.assertEqual(self.ingestion.vectors.collection.count(), 2)

    def test_delete_removes_owned_paper_pdf_and_vectors_and_allows_reupload(self):
        paper_id = self.upload().json()["paper"]["id"]
        self.assertEqual(self.cloudinary_upload.call_count, 1)

        # Recreate the service as it would be after an API restart.
        restarted_ingestion = IngestionService(self.settings)
        restarted_ingestion.embeddings = FakeEmbeddings()
        self.app.state.ingestion_service = restarted_ingestion

        response = self.client.delete(f"/api/v1/papers/{paper_id}")
        self.assertEqual(response.status_code, 204, response.text)
        self.cloudinary_destroy.assert_called_once()
        self.assertEqual(self.ingestion.vectors.collection.count(), 0)
        self.assertEqual(self.client.get("/api/v1/papers").json(), [])
        self.assertEqual(self.client.get(f"/api/v1/papers/{paper_id}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/v1/papers/{paper_id}/file").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/v1/papers/{paper_id}").status_code, 404)
        self.assertEqual(self.upload().status_code, 201)

    def test_delete_is_owner_only_and_blocks_processing_papers(self):
        paper_id = self.upload().json()["paper"]["id"]
        self.user_id = "bob"
        self.assertEqual(self.client.delete(f"/api/v1/papers/{paper_id}").status_code, 404)
        self.assertEqual(self.cloudinary_destroy.call_count, 0)

        self.user_id = "alice"
        self.repository.papers[paper_id]["status"] = "processing"
        self.assertEqual(self.client.delete(f"/api/v1/papers/{paper_id}").status_code, 409)
        self.assertEqual(self.cloudinary_destroy.call_count, 0)

    def test_delete_failure_preserves_record_for_retry(self):
        paper_id = self.upload().json()["paper"]["id"]
        with patch.object(self.ingestion, "cleanup", side_effect=OSError("private details")):
            response = self.client.delete(f"/api/v1/papers/{paper_id}")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private details", response.text)
        self.assertEqual(self.client.get(f"/api/v1/papers/{paper_id}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/v1/papers/{paper_id}").status_code, 204)

    def test_owner_isolation_and_same_file_allowed_for_another_user(self):
        paper_id = self.upload().json()["paper"]["id"]
        self.assertEqual(self.client.get("/api/v1/papers/summary").json(),
                         {"total": 1, "ready": 1, "processing": 0, "failed": 0})
        self.assertEqual(len(self.client.get("/api/v1/papers?status=ready").json()), 1)
        self.assertEqual(self.client.get("/api/v1/papers?status=processing").json(), [])
        self.user_id = "bob"
        self.assertEqual(self.client.get("/api/v1/papers").json(), [])
        self.assertEqual(self.client.get("/api/v1/papers/summary").json(),
                         {"total": 0, "ready": 0, "processing": 0, "failed": 0})
        self.assertEqual(self.client.get(f"/api/v1/papers/{paper_id}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/v1/papers/{paper_id}/file").status_code, 404)
        self.assertEqual(self.ingestion.vectors.preview("bob", paper_id), [])
        self.assertEqual(self.upload().status_code, 201)

    def test_unauthenticated_upload_is_rejected(self):
        paper_id = self.upload().json()["paper"]["id"]
        self.app.dependency_overrides.clear()
        self.assertEqual(self.upload().status_code, 401)
        self.assertEqual(self.client.get(f"/api/v1/papers/{paper_id}/file").status_code, 401)
        self.assertEqual(self.client.delete(f"/api/v1/papers/{paper_id}").status_code, 401)

    def test_rejects_foreign_origin_and_invalid_pdf(self):
        response = self.client.post("/api/v1/papers", headers={"Origin": "https://untrusted.example"},
    files={"file": ("paper.pdf", self.content, "application/pdf")})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.upload(b"not a PDF").status_code, 422)
        self.assertEqual(self.upload(b"%PDF- broken document").status_code, 422)
        self.assertEqual(self.upload(filename="paper.txt").status_code, 422)
        self.assertEqual(self.repository.papers, {})

    def test_delete_rejects_foreign_origin(self):
        paper_id = self.upload().json()["paper"]["id"]
        response = self.client.delete(f"/api/v1/papers/{paper_id}", headers={"Origin": "https://untrusted.example"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.cloudinary_destroy.call_count, 0)

    def test_oversized_upload_is_rejected_before_storage(self):
        self.settings.max_pdf_size_mb = 1
        self.assertEqual(self.upload(b"%PDF-" + b"x" * (1024 * 1024)).status_code, 413)
        self.assertEqual(self.repository.papers, {})

    def test_busy_processor_returns_retry_after(self):
        asyncio.run(self.app.state.upload_lock.acquire())
        response = self.upload()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "5")

    def test_embedding_failure_cleans_partial_vectors_and_allows_retry(self):
        self.settings.embedding_batch_size = 1
        with patch.object(self.ingestion.embeddings, "embed", side_effect=[[[1.0] * 384], RuntimeError("private details")]):
            response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private details", response.text)
        self.assertEqual(self.ingestion.vectors.collection.count(), 0)
        self.assertEqual(self.repository.papers, {})
        self.assertEqual(self.cloudinary_upload.call_count, 0)
        self.assertEqual(self.upload().status_code, 201)

    def test_cloudinary_upload_view_and_delete_keep_owner_checks(self):
        with patch("app.services.pdf_storage.cloudinary.uploader.upload") as upload, \
             patch("app.services.pdf_storage.cloudinary.uploader.destroy", return_value={"result": "ok"}) as destroy, \
             patch("app.services.pdf_storage.cloudinary.utils.private_download_url", return_value="https://api.cloudinary.com/signed-pdf") as download_url:
            upload.side_effect = lambda _source, **options: {"public_id": options["public_id"]}
            response = self.upload()
            self.assertEqual(response.status_code, 201, response.text)
            paper_id = response.json()["paper"]["id"]
            self.assertEqual(self.repository.papers[paper_id]["storage_provider"], "cloudinary")
            self.assertEqual(self.repository.papers[paper_id]["storage_id"], CloudinaryPdfStorage.public_id(paper_id))
            self.assertFalse(self.ingestion.file_path(paper_id).exists())
            self.assertEqual(upload.call_args.kwargs["type"], "authenticated")
            self.assertEqual(upload.call_args.kwargs["resource_type"], "image")
            self.assertEqual(upload.call_args.args[0].getvalue(), self.content)

            self.user_id = "bob"
            self.assertEqual(self.client.get(f"/api/v1/papers/{paper_id}/file").status_code, 404)
            self.assertEqual(self.client.delete(f"/api/v1/papers/{paper_id}").status_code, 404)
            download_url.assert_not_called()
            self.user_id = "alice"

            file_response = self.client.get(f"/api/v1/papers/{paper_id}/file", follow_redirects=False)
            self.assertEqual(file_response.status_code, 307)
            self.assertEqual(file_response.headers["location"], "https://api.cloudinary.com/signed-pdf")
            self.assertEqual(file_response.headers["cache-control"], "private, no-store")
            self.assertEqual(download_url.call_args.kwargs["type"], "authenticated")
            self.assertEqual(self.client.delete(f"/api/v1/papers/{paper_id}").status_code, 204)
            destroy.assert_called_once_with(
                CloudinaryPdfStorage.public_id(paper_id), resource_type="image", type="authenticated", invalidate=True,
            )
            self.assertEqual(self.ingestion.vectors.collection.count(), 0)

    def test_cloudinary_failure_removes_partial_vectors(self):
        with patch("app.services.pdf_storage.cloudinary.uploader.upload", side_effect=RuntimeError("private details")), \
             patch("app.services.pdf_storage.cloudinary.uploader.destroy", return_value={"result": "not found"}):
            response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private details", response.text)
        self.assertEqual(self.ingestion.vectors.collection.count(), 0)
        self.assertEqual(self.repository.papers, {})

    def test_existing_local_paper_survives_switch_to_cloudinary(self):
        paper_id = "legacy-paper"
        path = self.ingestion.file_path(paper_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content)
        self.repository.papers[paper_id] = {
            "_id": paper_id, "owner_id": "alice", "filename": "legacy.pdf", "status": "ready",
            "file_hash": "legacy", "created_at": datetime.now(timezone.utc),
        }
        self.app.state.ingestion_service = IngestionService(self.settings)
        self.assertEqual(self.client.get(f"/api/v1/papers/{paper_id}/file").content, self.content)
        self.assertEqual(self.client.delete(f"/api/v1/papers/{paper_id}").status_code, 204)
        self.assertFalse(path.exists())


class StorageSettingsTests(unittest.TestCase):
    def test_cloudinary_requires_credentials(self):
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, google_client_id="test", mongodb_uri="mongodb://localhost",
                     jwt_secret="test-secret-" * 4)


class PaperRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_database_reads_and_writes_include_owner(self):
        from unittest.mock import AsyncMock
        collection = AsyncMock()
        repository = PaperRepository(collection)
        await repository.find("alice", "paper")
        collection.find_one.assert_awaited_once_with({"_id": "paper", "owner_id": "alice"})
        await repository.update("alice", "paper", status="ready")
        self.assertEqual(collection.update_one.call_args.args[0], {"_id": "paper", "owner_id": "alice"})
        await repository.remove("alice", "paper")
        collection.delete_one.assert_awaited_once_with({"_id": "paper", "owner_id": "alice"})

    async def test_summary_counts_only_owned_papers(self):
        from unittest.mock import AsyncMock
        collection = AsyncMock()
        collection.count_documents.side_effect = [5, 3, 1, 1]
        result = await PaperRepository(collection).summary_for_user("alice")
        self.assertEqual(result, {"total": 5, "ready": 3, "processing": 1, "failed": 1})
        self.assertEqual([call.args[0] for call in collection.count_documents.await_args_list], [
            {"owner_id": "alice"},
            {"owner_id": "alice", "status": "ready"},
            {"owner_id": "alice", "status": "processing"},
            {"owner_id": "alice", "status": "failed"},
        ])

    async def test_filtered_list_keeps_owner_constraint(self):
        from unittest.mock import MagicMock
        collection = MagicMock()
        collection.find.return_value.sort.return_value.skip.return_value.limit.return_value.to_list = unittest.mock.AsyncMock(return_value=[])
        await PaperRepository(collection).list_for_user("alice", 10, 0, "ready")
        collection.find.assert_called_once_with({"owner_id": "alice", "status": "ready"})
