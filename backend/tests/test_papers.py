import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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
        self.rejections = []

    async def record_rejection(self, owner_id, checks):
        self.rejections.append((owner_id, checks))

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

    async def mark_nonresearch(self, paper, category):
        saved = await self.find(paper["owner_id"], paper["_id"])
        if not saved or saved["status"] != "ready" or saved.get("document_type"):
            return None
        saved.update(status="not_research", document_type="not_research_paper", document_category=category)
        saved.pop("publication_identity", None)
        return saved

    async def claim_identity_job(self):
        for paper in self.papers.values():
            if paper["status"] == "ready" and paper.get("publication_identity", {}).get("status") == "pending":
                attempts = paper["publication_identity"].get("attempts", 0) + 1
                paper["publication_identity"] = {**paper["publication_identity"], "status": "checking", "lease_token": "test", "attempts": attempts}
                return paper.copy()
        return None

    async def finish_identity_job(self, paper, identity, *, identity_input=None):
        saved = await self.find(paper["owner_id"], paper["_id"])
        if saved and saved.get("file_hash") == paper["file_hash"] and saved["publication_identity"].get("lease_token") == paper["publication_identity"]["lease_token"]:
            saved["publication_identity"] = identity
            if identity_input is not None:
                saved["identity_input"] = identity_input
            return True
        return False

    async def retry_identity_job(self, owner_id, paper_id):
        paper = await self.find(owner_id, paper_id)
        if paper and "identity_input" in paper and paper["publication_identity"]["status"] in {"verification_unavailable", "not_verified", "conflicting", "registry_record"}:
            paper["publication_identity"] = {"status": "pending", "refresh_extraction": True}
            return True
        return False

    async def backfill_identity_job(self, owner_id, paper_id, file_hash):
        paper = await self.find(owner_id, paper_id)
        if paper and paper.get("file_hash") == file_hash and "identity_input" not in paper and not paper.get("publication_identity"):
            paper["identity_input"] = {}
            paper["publication_identity"] = {"status": "pending", "refresh_extraction": True}
            return True
        return False


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

    def test_layout_headings_split_sections_without_fixed_heading_names(self):
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=800)
            page.insert_text((40, 120), "1 Unfamiliar topic", fontsize=16)
            page.insert_text((40, 145), "Evidence includes a value of 42 and a useful qualification.", fontsize=10)
            page.insert_text((40, 200), "2 Another unusual heading", fontsize=16)
            page.insert_text((40, 225), "Independent evidence contains a different value of 15.", fontsize=10)
            content = document.tobytes()
        _, pages = extract_pages(content, 300, 10000)
        chunks = create_chunks(pages, test_tokenizer())
        self.assertEqual([chunk.section for chunk in chunks], ["1 Unfamiliar topic", "2 Another unusual heading"])
        self.assertIn("42", chunks[0].text)
        self.assertIn("15", chunks[1].text)
        self.assertTrue(all(chunk.page_number == 1 for chunk in chunks))

    def test_chunking_without_headings_keeps_fallback_and_section_metadata_is_optional(self):
        _, pages = extract_pages(make_pdf("Simple plain evidence without special typography."), 300, 10000)
        chunks = create_chunks(pages, test_tokenizer())
        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].section)
        self.assertIn("plain evidence", chunks[0].text)

    def test_section_metadata_survives_index_and_old_chunks_still_load(self):
        with tempfile.TemporaryDirectory() as directory:
            vectors = VectorRepository(Path(directory))
            from app.services.pdf_processing import TextChunk
            chunks = [TextChunk(0, 1, "Evidence under heading.", "first", "Unfamiliar heading"),
                      TextChunk(1, 2, "Legacy-style evidence.", "second")]
            vectors.save("alice", "paper", chunks, [[1.0, 0.0], [0.0, 1.0]])
            loaded = vectors.preview("alice", "paper", 0, 2)
            self.assertEqual([item.section for item in loaded], ["Unfamiliar heading", None])
            self.assertEqual(vectors.preview("bob", "paper", 0, 2), [])

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
        self.content = make_pdf("Abstract\nResearch evidence on page one.\nIntroduction\nContext.", "Methods\nMethods and results on page two.")

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

    def test_identity_check_stores_results_per_owned_paper(self):
        from unittest.mock import Mock
        discovery = Mock()
        discovery.search.return_value = {"results": [{"title": "A Study of Unusual Materials",
            "authors": ["Ada Miller"], "year": 2024, "venue": "Research Conference",
            "status": "verified_published", "doi": "10.1234/example",
            "sources": ["Crossref"], "landing_url": "https://doi.org/10.1234/example"}], "provider_errors": {}}
        self.app.state.discovery_service = discovery
        first = self.upload(make_pdf("A Study of Unusual Materials\nAda Miller\n10.1234/example\nAbstract\nA study.\nIntroduction\nMethods overview.\nMethods\nMeasurements."))
        self.assertEqual(first.status_code, 201)
        first_id = first.json()["paper"]["id"]
        second = self.upload(make_pdf("Abstract\nDifferent study of atmospheric pressure during seasonal storms.\nIntroduction\nContext for the study.\nMethods\nMeasurements."))
        self.assertEqual(second.status_code, 201)
        second_id = second.json()["paper"]["id"]
        from app.services.identity_worker import process_one
        asyncio.run(process_one(self.repository, discovery))
        asyncio.run(process_one(self.repository, discovery))
        identity = self.client.get(f"/api/v1/papers/{first_id}").json()["paper"]["publication_identity"]
        self.assertEqual(identity["status"], "verified_published")
        self.assertEqual((identity["authors"], identity["year"], identity["venue"]),
                         (["Ada Miller"], 2024, "Research Conference"))
        self.assertEqual(self.client.get(f"/api/v1/papers/{second_id}").json()["paper"]["publication_identity"]["status"], "not_verified")
        self.user_id = "bob"
        self.assertEqual(self.client.get(f"/api/v1/papers/{first_id}").status_code, 404)

    def test_upload_returns_extracted_metadata_before_provider_lookup(self):
        title = "Measuring Coastal Sound Propagation"
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 550, 750), "\n".join([
                title, "Ana Rivera", "DOI: 10.1234/coast", "Abstract",
                "We measured acoustic propagation along the coast.", "1 Introduction",
                "This later section should be excluded from the abstract.", "References", "DOI: 10.1234/other",
            ]))
            document.set_metadata({"title": title, "author": "Ana Rivera; Unrelated Author"})
            response = self.upload(document.tobytes(), "coast.pdf")
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["paper"]["publication_identity"]["status"], "pending")
        metadata = payload["extracted_metadata"]
        self.assertEqual(metadata["title"], title)
        self.assertEqual(metadata["authors"], ["Ana Rivera"])
        self.assertEqual(metadata["doi"], "10.1234/coast")
        self.assertEqual(metadata["abstract"], "We measured acoustic propagation along the coast.")
        url = f"/api/v1/papers/{payload['paper']['id']}?include_chunks=false"
        self.assertEqual(self.client.get(url).json()["extracted_metadata"], metadata)
        self.assertEqual(self.client.get(url).json()["chunks"], [])
        self.user_id = "bob"
        self.assertEqual(self.client.get(url).status_code, 404)

    def test_outage_preserves_metadata_and_old_papers_can_omit_it(self):
        from app.services.identity_worker import process_one
        from unittest.mock import Mock

        response = self.upload(make_pdf(
            "A Study of Coastal Sounds\nAvery Miller\nDOI: 10.1234/coast\nAbstract: Measured sounds. A sample of patterns was collected.\nMethods\nWe collected observations at three coastal stations."))
        paper_id = response.json()["paper"]["id"]
        paper = self.repository.papers[paper_id]
        paper["publication_identity"]["attempts"] = 2
        offline = Mock(client=None)
        offline.search.return_value = {"results": [], "provider_errors": {"crossref": "timeout"}}
        asyncio.run(process_one(self.repository, offline))
        url = f"/api/v1/papers/{paper_id}?include_chunks=false"
        detail = self.client.get(url).json()
        self.assertEqual(detail["paper"]["publication_identity"]["status"], "verification_unavailable")
        self.assertTrue(detail["extracted_metadata"]["abstract"].startswith("Measured sounds."))
        paper.pop("identity_input")
        self.assertIsNone(self.client.get(url).json()["extracted_metadata"])

    def test_certificate_is_rejected_and_unreadable_pdf_needs_review(self):
        certificate = make_pdf("Certificate of Completion\nAwarded to an attendee who has successfully completed all the lessons and projects.")
        rejection = self.upload(certificate)
        self.assertEqual(rejection.status_code, 422)
        self.assertIn("appears to be a certificate", rejection.json()["detail"])
        self.assertFalse(self.repository.papers)
        self.assertEqual(self.repository.rejections[0][0], "alice")
        checks = {item["check_id"]: item for item in self.repository.rejections[0][1]}
        self.assertEqual(checks["H3"]["status"], "FAIL")
        unknown = self.upload(make_pdf(""))
        self.assertEqual(unknown.status_code, 201)
        public = unknown.json()["paper"]
        self.assertEqual(public["document_state"], "NEEDS_REVIEW")
        self.assertIn("No readable text", public["error"])
        self.assertNotIn("verification_checks", public)
        saved = self.repository.papers[public["id"]]
        checks = {item["check_id"]: item for item in saved["verification_checks"]}
        self.assertEqual(checks["H4"]["status"], "UNAVAILABLE")
        self.assertEqual(checks["M9"]["status"], "UNAVAILABLE")
        self.cloudinary_upload.assert_called_once()
        self.assertEqual(self.ingestion.vectors.collection.count(), 0)

    def test_corrupt_pdf_has_a_rejection_audit_and_is_never_indexed(self):
        response = self.upload(b"%PDF-1.7\nMissing all objects and cross references")
        self.assertEqual(response.status_code, 422)
        checks = {item["check_id"]: item for item in self.repository.rejections[0][1]}
        self.assertEqual(checks["H2"]["status"], "FAIL")
        self.assertFalse(self.repository.papers)
        self.cloudinary_upload.assert_not_called()

    def test_renamed_nonresearch_uploads_are_rejected_before_storage(self):
        for text, label in (
            ("Curriculum Vitae\nJamie Rivera\nSkills\nSoftware development and testing.\nEducation\nBachelor of Science.",
             "resume or CV"),
            ("Invoice 302\nBill to the company listed below.\nAmount due equals the subtotal plus tax.",
             "invoice or receipt"),
            ("Homework 3\nDue date: Monday. Submit your report to your instructor for grading.",
             "academic assignment"),
        ):
            with self.subTest(label=label):
                response = self.upload(make_pdf(text), "Peer-reviewed Journal Article.pdf")
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn(label, response.json()["detail"])
        self.assertEqual(self.upload(b"Not a PDF", "Conference Research Paper.pdf").status_code, 422)
        self.assertFalse(self.repository.papers)
        self.cloudinary_upload.assert_not_called()
        self.assertEqual(self.ingestion.vectors.collection.count(), 0)

    def test_readable_ambiguous_pdf_is_review_only(self):
        response = self.upload(make_pdf("An undated note about a new idea for collecting observations during the spring."))
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["paper"]["status"], "needs_review")
        self.assertIn("couldn't confidently determine", response.json()["paper"]["error"])
        self.assertEqual(response.json()["chunks"], [])
        self.assertEqual(self.ingestion.vectors.collection.count(), 0)

    def test_arxiv_preprint_is_accepted_and_matched_as_a_preprint(self):
        from unittest.mock import Mock
        from app.services.identity_worker import process_one

        title = "Bounds on Repeated Sampling"
        content = make_pdf(f"{title}\nAda Miller\narXiv:2101.12345v1 [stat.ME]\nAbstract\n"
                           "We analyze sampling bounds for a class of estimators.\nIntroduction\n"
                           "This article investigates repeated sampling.\nMethods\n"
                           "We derive the bounds in finite samples.")
        response = self.upload(content)
        self.assertEqual(response.status_code, 201, response.text)
        paper = response.json()["paper"]
        self.assertEqual(paper["status"], "ready")
        self.assertEqual(response.json()["extracted_metadata"]["arxiv_id"], "2101.12345v1")
        registry = Mock(client=None)
        registry.search.return_value = {"results": [{
            "title": title, "authors": ["Ada Miller"], "status": "verified_preprint",
            "arxiv_id": "2101.12345v1", "sources": ["arXiv"], "landing_url": "https://arxiv.org/abs/2101.12345v1",
        }], "provider_errors": {}}
        asyncio.run(process_one(self.repository, registry))
        result = self.client.get(f'/api/v1/papers/{paper["id"]}').json()["paper"]["publication_identity"]
        self.assertEqual(result["status"], "verified_preprint")

    def test_optional_metadata_and_pending_identity_do_not_block_readiness(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201)
        public = response.json()["paper"]
        self.assertEqual(public["document_state"], "READY_FOR_RESEARCH")
        saved = self.repository.papers[public["id"]]
        checks = {item["check_id"]: item for item in saved["verification_checks"]}
        self.assertTrue(all(checks[f"H{i}"]["status"] == "PASS" for i in range(1, 5)))
        self.assertEqual(checks["H5"]["status"], "UNAVAILABLE")
        self.assertEqual(checks["M6"]["status"], "NOT_APPLICABLE")
        self.assertEqual(checks["M9"]["status"], "PASS")

    def test_legacy_certificate_is_reclassified_and_original_remains_owned(self):
        paper_id = self.upload().json()["paper"]["id"]
        paper = self.repository.papers[paper_id]
        paper.pop("document_type")
        certificate = "Certificate of Completion\nAwarded to an attendee who has successfully completed all the lessons and projects."
        paper["identity_input"] = {"first": certificate, "text": certificate}
        paper["publication_identity"] = {"status": "verified_published"}
        url = f"/api/v1/papers/{paper_id}"
        self.user_id = "bob"
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(paper["status"], "ready")
        self.user_id = "alice"
        detail = self.client.get(url).json()
        self.assertEqual(detail["paper"]["status"], "not_research")
        self.assertIsNone(detail["paper"]["publication_identity"])
        self.assertEqual(detail["chunks"], [])
        self.assertEqual(self.client.get(url + "/file", follow_redirects=False).status_code, 307)
        self.assertEqual(self.client.get("/api/v1/papers?status=ready").json(), [])
        self.assertEqual(self.client.post(url + "/verification/retry").status_code, 409)
        self.cloudinary_upload.assert_called_once()

    def test_retry_is_owner_scoped_and_does_not_reupload(self):
        from app.services.identity_worker import process_one
        from unittest.mock import Mock
        response = self.upload(make_pdf("Abstract\nAnalysing spring air pressure patterns across a range of altitudes.\nIntroduction\nScope.\nMethods\nMeasurements."))
        paper_id = response.json()["paper"]["id"]
        asyncio.run(process_one(self.repository, Mock()))
        self.user_id = "bob"
        self.assertEqual(self.client.post(f"/api/v1/papers/{paper_id}/verification/retry").status_code, 404)
        self.user_id = "alice"
        self.assertEqual(self.client.post(f"/api/v1/papers/{paper_id}/verification/retry").status_code, 202)
        self.assertEqual(self.client.post(f"/api/v1/papers/{paper_id}/verification/retry").status_code, 409)
        self.assertEqual(self.cloudinary_upload.call_count, 1)

    def test_old_paper_schedules_metadata_recovery_without_reindex(self):
        paper_id = self.upload(make_pdf("Abstract\nAn older study with enough readable first page text.\nIntroduction\nScope.\nMethods\nMeasurements.")).json()["paper"]["id"]
        self.repository.papers[paper_id].pop("identity_input")
        self.repository.papers[paper_id].pop("publication_identity")
        count = self.ingestion.vectors.collection.count()
        self.user_id = "bob"
        self.assertEqual(self.client.post(f"/api/v1/papers/{paper_id}/verification/retry").status_code, 404)
        self.user_id = "alice"
        self.assertEqual(self.client.post(f"/api/v1/papers/{paper_id}/verification/retry").status_code, 202)
        self.assertEqual(self.client.post(f"/api/v1/papers/{paper_id}/verification/retry").status_code, 409)
        self.assertEqual(self.ingestion.vectors.collection.count(), count)
        self.assertEqual(self.cloudinary_upload.call_count, 1)

    def test_retry_reextracts_the_stored_pdf_and_persists_new_identity(self):
        from app.services.identity_worker import process_one
        from unittest.mock import Mock

        title = "Measuring Coastal Sound Propagation"
        content = make_pdf(f"{title}\nAna Rivera\nDOI: 10.1234/coast\nAbstract\nWe measured coastal sounds.\nMethods\nWe measured three field stations.")
        response = self.upload(content)
        self.assertEqual(response.status_code, 201)
        paper_id = response.json()["paper"]["id"]
        paper = self.repository.papers[paper_id]
        paper["identity_input"] = {"first": "", "text": ""}
        discovery = Mock(client=None)
        asyncio.run(process_one(self.repository, discovery, self.ingestion.read_original))
        self.assertEqual(paper["publication_identity"]["reason_code"], "insufficient_metadata")
        discovery.search.assert_not_called()
        url = f"/api/v1/papers/{paper_id}"
        self.assertEqual(self.client.post(url + "/verification/retry").status_code, 202)
        self.assertEqual(self.client.get(url).json()["paper"]["publication_identity"]["status"], "pending")
        discovery.search.return_value = {"results": [{"title": title, "authors": ["Ana Rivera"],
            "doi": "10.1234/coast", "status": "verified_published", "year": 2023, "sources": ["Crossref"]}]}
        download = Mock(status_code=200)
        download.iter_content.return_value = [content]
        with patch("app.services.pdf_storage.requests.get") as request:
            request.return_value.__enter__.return_value = download
            asyncio.run(process_one(self.repository, discovery, self.ingestion.read_original))
        discovery.search.assert_called_once_with("10.1234/coast", 0, 1)
        reopened = self.client.get(url).json()
        self.assertEqual(reopened["paper"]["publication_identity"]["status"], "verified_published")
        self.assertEqual(reopened["extracted_metadata"]["doi"], "10.1234/coast")
        self.assertEqual(self.cloudinary_upload.call_count, 1)
        self.assertEqual(self.ingestion.vectors.collection.count(), response.json()["paper"]["chunk_count"])

    def test_retry_reports_missing_original_and_enforces_cooldown(self):
        from app.services.identity_worker import process_one
        from unittest.mock import Mock

        paper_id = self.upload().json()["paper"]["id"]
        paper = self.repository.papers[paper_id]
        paper.update(identity_input={}, publication_identity={"status": "not_verified"})
        url = f"/api/v1/papers/{paper_id}/verification/retry"
        self.assertEqual(self.client.post(url).status_code, 202)
        asyncio.run(process_one(self.repository, Mock(), Mock(side_effect=OSError)))
        self.assertEqual(paper["publication_identity"]["reason_code"], "original_pdf_unavailable")
        self.assertIn("original PDF could not be reopened", paper["publication_identity"]["reason"])
        deadline = datetime.now(timezone.utc) + timedelta(seconds=30)
        paper["publication_identity"]["retry_after"] = deadline.replace(tzinfo=None)
        limited = self.client.post(url)
        self.assertEqual(limited.status_code, 429)
        self.assertGreater(int(limited.headers["Retry-After"]), 0)
        saved_deadline = self.client.get(url.removesuffix("/verification/retry")).json()["paper"]["publication_identity"]["retry_after"]
        self.assertEqual(datetime.fromisoformat(saved_deadline), deadline)

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
