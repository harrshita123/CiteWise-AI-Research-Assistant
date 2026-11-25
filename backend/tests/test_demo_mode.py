import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import pymupdf
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pymongo import MongoClient

from app.config import Settings, get_settings
from app.providers.embeddings import LocalEmbeddings
from app.services.demo_research import CASES, EXTERNAL_QUESTIONS, SAMPLE_IDENTITY, SAMPLE_PDF, SAMPLE_SHA256, SAMPLE_TITLE, is_sample
from app.services.document_classification import DocumentType
from app.services.publication_identity import verify_pdf_identity
from tests.test_papers import make_pdf, test_tokenizer as make_tokenizer


class DemoConfigurationTests(unittest.TestCase):
    def test_demo_requires_separate_database_and_session_secret(self):
        config = {"_env_file": None, "citewise_demo_mode": True, "citewise_demo_jwt_secret": "d" * 40,
                  "mongodb_uri": "mongodb://production.example", "mongodb_database": "citewise"}
        settings = Settings(**config)
        self.assertEqual(settings.citewise_demo_mongodb_uri, "mongodb://127.0.0.1:27017")
        self.assertEqual(settings.session_cookie_name, "citewise_demo_session")
        self.assertNotEqual(settings.data_directory, settings.storage_directory)
        with self.assertRaises(ValidationError):
            Settings(**config, citewise_demo_database="citewise")
        with self.assertRaises(ValidationError):
            Settings(**config, jwt_secret="d" * 40)

    def test_demo_defaults_to_single_chunk_embeddings_and_allows_an_override(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = {"_env_file": None, "citewise_demo_mode": True, "citewise_demo_jwt_secret": "d" * 40}
            self.assertEqual(Settings(**settings).embedding_batch_size, 1)
            self.assertEqual(Settings(**settings, embedding_batch_size=4).embedding_batch_size, 4)
            self.assertEqual(Settings.model_fields["embedding_batch_size"].default, 32)

    def test_every_sample_answer_is_supported_on_the_cited_pdf_page(self):
        with pymupdf.open(SAMPLE_PDF) as pdf:
            self.assertEqual(pdf.metadata["title"], SAMPLE_TITLE)
            for case in CASES:
                self.assertIn(case.excerpt, " ".join(pdf[case.page - 1].get_text().split()))

    def test_bundled_pdf_matches_corroborated_publication_metadata(self):
        record = {**SAMPLE_IDENTITY, "sources": ["Crossref"], "landing_url": SAMPLE_IDENTITY["source_url"]}
        discovery = Mock()
        discovery.search.return_value = {"results": [record], "provider_errors": {}}
        result = verify_pdf_identity(SAMPLE_PDF.read_bytes(), discovery)
        for field in ("status", "title", "authors", "doi", "venue", "year"):
            self.assertEqual(result[field], SAMPLE_IDENTITY[field])

    def test_sample_eligibility_uses_uploaded_file_fingerprint_only(self):
        self.assertTrue(is_sample({"filename": "demo.pdf", "file_hash": SAMPLE_SHA256}))
        self.assertFalse(is_sample({"filename": "demo.pdf", "file_hash": "different",
                                    "demo_sample": True, "publication_identity": SAMPLE_IDENTITY}))


@unittest.skipUnless(os.getenv("CITEWISE_TEST_MONGO_URI"), "isolated MongoDB URI not configured")
class DemoWorkflowTests(unittest.TestCase):
    def test_guest_workspaces_sample_answers_and_uploaded_pdf_retrieval_are_isolated(self):
        from app import main

        uri = os.environ["CITEWISE_TEST_MONGO_URI"]
        tag = uuid4().hex
        demo_db, prod_db = f"citewise_demo_{tag}", f"citewise_fixture_{tag}"
        with tempfile.TemporaryDirectory() as storage, MongoClient(uri) as mongo:
            self.addCleanup(get_settings.cache_clear)
            self.addCleanup(main.app.dependency_overrides.pop, get_settings, None)
            marker = {"_id": "openrouter", "attempts": [{"ticket": "preserved", "at": 1}]}
            mongo[prod_db].llm_request_budget.insert_one(marker)
            mongo[prod_db].papers.insert_one({"_id": "private-paper", "owner_id": "real-user"})
            settings = Settings(_env_file=None, citewise_demo_mode=True,
                citewise_demo_mongodb_uri=uri, citewise_demo_database=demo_db, citewise_demo_jwt_secret="x" * 40,
                mongodb_uri="mongodb://invalid-production-host.example", mongodb_database=prod_db,
                storage_directory=storage, session_cookie_secure=True, openrouter_api_key="should-never-be-used")
            main.app.dependency_overrides[get_settings] = lambda: settings
            try:
                with patch.object(main, "settings", settings), \
                        patch.object(main, "OpenRouterAnswerProvider") as live, \
                        patch.object(main, "TavilyExternalSearch") as tavily, \
                        patch.object(main, "SharedRequestBudget") as reservations, \
                        patch.object(main, "run_worker") as worker, \
                        patch("app.dependencies.get_settings", return_value=settings), \
                        patch("app.routers.papers.get_settings", return_value=settings), \
                        patch.object(LocalEmbeddings, "tokenizer", return_value=make_tokenizer()), \
                        patch.object(LocalEmbeddings, "embed", side_effect=lambda texts: [[1.0] + [0.0] * 383 for _ in texts]), \
                        TestClient(main.app, base_url="https://testserver") as client:
                    self.assertEqual(client.get("/api/v1/auth/demo").json(), {"demo_mode": True})
                    self.assertEqual(client.post("/api/v1/auth/login", json={
                        "email":"demo@example.com", "password":"not-a-real-password"}).status_code, 404)
                    start = client.post("/api/v1/auth/demo")
                    self.assertEqual(start.status_code, 201, start.text)
                    self.assertIn("citewise_demo_session", start.headers["set-cookie"])
                    self.assertIn("Secure", start.headers["set-cookie"])
                    paper_id = start.json()["paper_id"]
                    paper_url = f"/api/v1/papers/{paper_id}"
                    research = "/api/v1/research"
                    self.assertEqual(client.get(paper_url + "/file").content, SAMPLE_PDF.read_bytes())
                    self.assertEqual(client.get(paper_url).json()["paper"]["publication_identity"], SAMPLE_IDENTITY)
                    self.assertEqual(client.get("/api/v1/papers/private-paper").status_code, 404)
                    hints = client.get(f"{research}/papers/{paper_id}/recommendations").json()["recommendations"]
                    self.assertEqual([item["text"] for item in hints[:len(CASES)]], [case.question for case in CASES])
                    self.assertEqual([item["text"] for item in hints if item["recommendation_type"] == "sample_external"],
                                     list(EXTERNAL_QUESTIONS))
                    for case in CASES:
                        answer = client.post(research + "/ask", json={"paper_id": paper_id, "question": case.question})
                        self.assertEqual(answer.status_code, 200, answer.text)
                        self.assertEqual(answer.headers["x-citewise-demo"], "true")
                        result = answer.json()
                        self.assertEqual(result["status"], "answered")
                        self.assertIn(case.answer, result["answer"])
                        self.assertEqual(result["sources"][0]["page_number"], case.page)
                        self.assertEqual(result["assessment"]["level"], "sufficient")
                        self.assertEqual(result["claims"][0]["supporting"][0]["quote"], case.excerpt)
                        self.assertEqual(result["search_trace"]["openrouter_calls"], 0)
                        self.assertEqual(result["verification_status"], "not_verified")
                    follow_ups = client.post(f"{research}/conversations/{result['conversation_id']}/recommendations",
                        json={"message_id": result["message_id"]}).json()["recommendations"]
                    self.assertEqual(len(follow_ups), 5)
                    self.assertNotIn(CASES[-1].question, [suggestion["text"] for suggestion in follow_ups])
                    for question, expected in zip(EXTERNAL_QUESTIONS, ("contradictory_evidence", "not_mentioned")):
                        answer = client.post(research + "/ask", json={"paper_id": paper_id, "question": question})
                        self.assertEqual(answer.status_code, 200, answer.text)
                        external = answer.json()
                        self.assertEqual(external["status"], expected)
                        self.assertEqual(external["assessment"]["level"], "none")
                        self.assertEqual(external["sources"], [])
                        self.assertEqual(external["search_trace"]["tavily_calls"], 0)
                        self.assertEqual(external["search_trace"]["openrouter_calls"], 0)
                        citations = {source["id"]: source for source in external["external_sources"]}
                        claim = external["claims"][0]
                        self.assertEqual(bool(claim["opposing"]), expected == "contradictory_evidence")
                        for quote in claim["supporting"] + claim["opposing"]:
                            source = citations[quote["source_id"]]
                            self.assertIn(quote["quote"], source["excerpt"])
                            self.assertIn(f'[{source["id"]}]', external["answer"])
                            self.assertTrue(source["url"].startswith("https://doi.org/10.1371/"))
                    review = client.post("/api/v1/papers", files={"file": ("field-notes.pdf", make_pdf(
                        "An undated note about a new idea for collecting observations during the spring."
                    ), "application/pdf")}).json()
                    self.assertEqual(review["paper"]["status"], "needs_review")
                    for action in ("ask", "search"):
                        blocked = client.post(f"{research}/{action}", json={
                            "paper_id": review["paper"]["id"], "question": CASES[0].question,
                        })
                        self.assertEqual(blocked.status_code, 409)
                    missing = client.post(research + "/ask", json={"paper_id": paper_id, "question": "Who won the World Cup?"}).json()
                    self.assertIn("Choose one of the suggested sample questions", missing["answer"])
                    self.assertEqual(missing["sources"], [])
                    related = client.post(f"{research}/papers/{paper_id}/recommendations/alternatives",
                        json={"question": "Who won the World Cup?", "conversation_id": missing["conversation_id"]})
                    self.assertEqual(len(related.json()["recommendations"]), 5)

                    content = make_pdf("A controlled study of soil pH. We measured magnesium concentrations in soil samples.")
                    with patch("app.routers.papers.classify_document",
                               return_value=(DocumentType.RESEARCH_PAPER, None)):
                        uploaded = client.post("/api/v1/papers", files={"file":("observations.pdf", content, "application/pdf")})
                    self.assertEqual(uploaded.status_code, 201, uploaded.text)
                    uploaded_id = uploaded.json()["paper"]["id"]
                    local = client.get(f"{research}/papers/{uploaded_id}/recommendations").json()
                    self.assertEqual(local["paper_id"], uploaded_id)
                    self.assertEqual([(item["text"], item["page_number"]) for item in local["recommendations"]],
                                     [("What methods did the authors use?", 1)])
                    self.assertEqual(client.get(f"/api/v1/papers/{uploaded_id}/file").content, content)
                    search = client.post(research + "/search", json={"paper_id": uploaded_id, "question": "magnesium"})
                    self.assertIn("magnesium", search.json()["passages"][0]["text"])
                    different = client.post(research + "/ask", json={"paper_id": uploaded_id, "question": CASES[0].question}).json()
                    self.assertIn("bundled PLOS ONE paper", different["answer"])
                    self.assertEqual(different["sources"], [])
                    client.post("/api/v1/auth/logout")
                    second = client.post("/api/v1/auth/demo").json()
                    self.assertNotEqual(second["paper_id"], paper_id)
                    self.assertEqual(client.get(paper_url).status_code, 404)
                    self.assertEqual(client.get(f"{research}/conversations/{result['conversation_id']}").status_code, 404)

                    sample_upload = {"file": ("demo.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")}
                    self.assertEqual(client.post("/api/v1/papers", files=sample_upload).status_code, 409)
                    self.assertEqual(client.delete(f"/api/v1/papers/{second['paper_id']}").status_code, 204)
                    copy_response = client.post("/api/v1/papers", files=sample_upload)
                    self.assertEqual(copy_response.status_code, 201, copy_response.text)
                    copy_id = copy_response.json()["paper"]["id"]
                    copy_hints = client.get(f"{research}/papers/{copy_id}/recommendations").json()
                    self.assertEqual(copy_hints["paper_id"], copy_id)
                    self.assertEqual(copy_hints["recommendations"], hints)
                    listed = client.get("/api/v1/papers?status=ready").json()
                    self.assertEqual(listed[0]["publication_identity"], SAMPLE_IDENTITY)
                    case = CASES[1]
                    clicked = client.post(research + "/ask", json={"paper_id": copy_id, "question": case.question}).json()
                    self.assertIn(case.answer, clicked["answer"])
                    self.assertEqual(clicked["sources"][0]["page_number"], case.page)
                    live.assert_not_called()
                    tavily.assert_not_called()
                    reservations.assert_not_called()
                    worker.assert_not_called()
                self.assertEqual(mongo[prod_db].llm_request_budget.find_one({"_id":"openrouter"}), marker)
                self.assertEqual(mongo[prod_db].papers.count_documents({}), 1)
                self.assertNotIn("llm_request_budget", mongo[demo_db].list_collection_names())
            finally:
                mongo.drop_database(demo_db)
                mongo.drop_database(prod_db)
