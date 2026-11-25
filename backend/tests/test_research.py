import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from pathlib import Path
from requests.exceptions import HTTPError, Timeout
import pymupdf

from fastapi import FastAPI
from starlette.requests import Request
from fastapi.testclient import TestClient

from app.dependencies import get_current_user
from app.models import UserResponse
from app.paper_models import ChunkResponse
from app.repositories.vectors import VectorRepository
from app.providers.answers import (AnswerRateLimitError, ClaimAssessment, ClaimQuote, EvidenceAssessment, EvidenceQuote, ExternalAnswer,
                                   ExternalEvidence, GeneratedAnswer)
from app.providers.openrouter_answers import OpenRouterAnswerProvider
from app.providers.openrouter_routing import FAST_MODEL, GENERATION_MODEL, PDF_FALLBACK_MODEL, VERIFICATION_MODEL, ModelRouter, UpstreamPoolLimit
from app.providers.external_search import TavilyExternalSearch
from app.routers.research import (_evidence_context, _evidence_excerpt, _generation_passages,
                                  _asks_only_about_pdf, _follow_up_uses_pdf,
                                  _methodology_comparison_gaps, _question_with_context,
                                  _validate_assessment, _reconcile_assessment_gap, _coverage_gap_message,
                                  _verified_answer_parts,
                                  _assessment_gap_in_selected_chunks, _verified_sections,
                                  _local_claim_debug, _retrieve, _simple_generation, AnswerSource, ClaimQuoteResponse, ClaimReviewResponse,
                                  EvidenceReview, SearchRequest, router)
from app.services.evidence_coverage import missing_coverage, requested_coverage
from app.services.claim_verification import split_claims
from app.services.pdf_processing import TextChunk


class Papers:
    async def find(self, owner_id, paper_id):
        if owner_id == "alice" and paper_id == "paper-1":
            return {"status": "ready", "filename": "paper.pdf"}
        if owner_id == "alice" and paper_id == "pending":
            return {"status": "processing"}
        return None


class Embeddings:
    def embed(self, texts):
        return [[1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0] for text in texts]


class Ingestion:
    def __init__(self, vectors):
        self.vectors = vectors
        self.embeddings = Embeddings()


class Answers:
    def __init__(self, result):
        self.result = result
        self.received = None

    def generate(self, question, evidence):
        self.received = (question, evidence)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def verify_claims(self, question, claims, evidence):
        answers = []
        for item in claims:
            first = next(src for src in evidence if src.id == item["cited_ids"][0])
            second = next((src for src in evidence if src.id != first.id), None)
            opposite = self.result.status == "contradictory_evidence" and second is not None
            answers.append(ClaimAssessment(item["id"], "contradicted" if opposite else "supported",
                [ClaimQuote(first.id, first.text)],
                [ClaimQuote(second.id, second.text)] if opposite else []))
        return answers


class UnavailableAnswers(Answers):
    def available(self):
        return False


class AssessedAnswers(Answers):
    def __init__(self, assessment):
        super().__init__(GeneratedAnswer("Alpha result is reported.", ["c0-hash-a"]))
        self.assessment = assessment
        self.external_calls = 0

    def assess(self, question, evidence):
        return self.assessment

    def generate_with_external(self, question, pdf_evidence, external_evidence, missing_information):
        self.external_calls += 1
        pdf_ids = [item.id for item in pdf_evidence]
        pdf_text = f"PDF result [{pdf_ids[0]}]." if pdf_ids else "No PDF evidence."
        return ExternalAnswer(pdf_text, "External result [ext1].", pdf_ids, ["ext1"])

    def verify_claims(self, question, claims, evidence):
        return [ClaimAssessment(item["id"], "supported",
            [ClaimQuote(item["cited_ids"][0],
                        next(src.text for src in evidence if src.id == item["cited_ids"][0]))], [])
            for item in claims]


class ExternalSearch:
    def __init__(self):
        self.queries = []

    def available(self):
        return True

    def search(self, query):
        self.queries.append(query)
        if "India biometric law" in query:
            return [ExternalEvidence("ext1", "Digital Personal Data Protection Act",
                                     "https://www.meity.gov.in/notice", "Official India biometric data law excerpt")]
        return [ExternalEvidence("ext1", query, "https://example.org/study", f"External result about {query}")]


class Conversations:
    def __init__(self):
        self.items = {}

    async def create(self, owner_id, paper_id, message, paper_filename=None):
        now = datetime.now(timezone.utc)
        item = {"_id": f"conversation-{len(self.items) + 1}", "owner_id": owner_id,
                "paper_id": paper_id, "paper_filename": paper_filename, "title": message["question"],
                "pinned": False, "saved": True, "created_at": now, "updated_at": now, "messages": [message]}
        self.items[item["_id"]] = item
        return item

    async def find(self, owner_id, conversation_id):
        item = self.items.get(conversation_id)
        return item if item and item["owner_id"] == owner_id else None

    async def append(self, owner_id, conversation_id, paper_id, message):
        item = await self.find(owner_id, conversation_id)
        if item is None or item["paper_id"] != paper_id:
            return False
        item["messages"].append(message)
        return True

    async def list_for_user(self, owner_id):
        return sorted((item for item in self.items.values() if item["owner_id"] == owner_id),
                      key=lambda item: item["updated_at"], reverse=True)

    async def update_metadata(self, owner_id, conversation_id, fields):
        item = await self.find(owner_id, conversation_id)
        if item is None:
            return False
        item.update(fields)
        item["updated_at"] = datetime.now(timezone.utc)
        return True

    async def remove(self, owner_id, conversation_id):
        item = await self.find(owner_id, conversation_id)
        if item is None:
            return False
        del self.items[conversation_id]
        return True

    async def set_feedback(self, owner_id, conversation_id, message_id, feedback):
        item = await self.find(owner_id, conversation_id)
        if item is None:
            return False
        for message in item["messages"]:
            if message["id"] == message_id:
                message["feedback"] = feedback
                return True
        return False

    async def update_verified_message(self, owner_id, conversation_id, message_id, fields):
        item = await self.find(owner_id, conversation_id)
        if not item:
            return False
        for message in item["messages"]:
            if message["id"] == message_id and message.get("verification_status") == "unavailable":
                message.update(fields)
                return True
        return False


class ResearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        vectors.save("alice", "paper-1", [
            TextChunk(0, 1, "Alpha result", "hash-a"),
            TextChunk(1, 2, "Beta result", "hash-b"),
        ], [[1.0, 0.0], [0.0, 1.0]])
        vectors.save("bob", "paper-1", [TextChunk(2, 3, "Bob's alpha secret", "hash-c")], [[1.0, 0.0]])

        app = FastAPI()
        app.include_router(router)
        app.state.paper_repository = Papers()
        app.state.ingestion_service = Ingestion(vectors)
        app.state.conversation_repository = Conversations()
        app.dependency_overrides[get_current_user] = lambda: UserResponse(
            id="alice", email="alice@example.com", role="user")
        self.client = TestClient(app)

    def test_simple_routing_requires_one_independent_lookup_and_one_passage(self):
        simple = "How many participants enrolled?"
        solo = ["The study enrolled 200 participants."]
        self.assertTrue(_simple_generation(simple, simple, solo))
        self.assertTrue(_simple_generation("What does attrition mean?", "What does attrition mean?", solo))
        self.assertFalse(_simple_generation(simple, simple, solo * 2))
        self.assertFalse(_simple_generation(simple, f"Earlier study. Follow-up question: {simple}", solo))
        for complex_question in ("What was the study design?", "What is the percentage change since baseline?",
                                 "What was the sample size and follow-up duration?", "Compare the findings.",
                                 "Alpha?"):
            with self.subTest(question=complex_question):
                self.assertFalse(_simple_generation(complex_question, complex_question, solo))

    def test_simple_pdf_answer_runs_one_generator_and_preserves_verification_on_retry(self):
        self.client.app.state.answer_provider = OpenRouterAnswerProvider("key", ModelRouter())
        passage = "The Alpha measurement was eight units."
        self.client.app.state.ingestion_service.vectors.save("alice", "paper-1",
            [TextChunk(0, 1, passage, "hash-measurement")], [[1.0, 0.0]])
        source_id = "c0-hash-measure"
        values = [
            {"level": "sufficient", "chunk_ids": [source_id], "missing_information": "",
             "supporting_quotes": [{"chunk_id": source_id, "quote": passage}]},
            {"status": "answered", "text": f"The Alpha measurement was eight units [{source_id}].",
             "evidence_ids": [source_id]},
            {"claims": [{"claim_id": "claim-1", "status": "supported",
                         "supporting": [{"source_id": source_id, "quote": passage}],
                         "opposing": [], "explanation": "Direct observation.", "question_fit": "direct",
                         "evidence_use": "establishes_claim", "unsupported_source_ids": [], "scope_preserved": True}],
             "requirements": [{"requirement": "Identify the result", "addressed_by": ["claim-1"]}]},
        ]
        responses = []
        for value in [*values, values[-1]]:
            reply = Mock(status_code=200)
            reply.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
            responses.append(reply)
        with patch("app.providers.openrouter_answers.requests.post", side_effect=responses) as transport:
            for expected_calls in (3, 4):
                response = self.client.post("/api/v1/research/ask",
                    json={"paper_id": "paper-1", "question": "What was the Alpha measurement?"})
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(data["verification_status"], "verified", data)
                self.assertEqual([source["page_number"] for source in data["sources"]], [1])
                self.assertEqual(transport.call_count, expected_calls)
        self.assertEqual([call.kwargs["json"]["model"] for call in transport.call_args_list],
                         [VERIFICATION_MODEL, FAST_MODEL, VERIFICATION_MODEL, VERIFICATION_MODEL])

    def test_retrieval_cache_requires_same_owner_and_current_paper_version(self):
        default = {"status": "ready", "file_hash": "digest", "updated_at": "v1", "chunk_count": 2}
        repository = self.client.app.state.paper_repository
        ingestion = self.client.app.state.ingestion_service
        payload = {"paper_id": "paper-1", "question": "Alpha?", "limit": 1}
        with patch.object(repository, "find", return_value=default.copy()) as find, \
                patch.object(ingestion.embeddings, "embed", wraps=ingestion.embeddings.embed) as embed:
            first = self.client.post("/api/v1/research/search", json=payload)
            second = self.client.post("/api/v1/research/search", json=payload)
            self.assertEqual(first.json(), second.json())
            self.assertEqual(embed.call_count, 1)
            self.client.app.dependency_overrides[get_current_user] = lambda: UserResponse(
                id="bob", email="bob@example.com", role="user")
            other = self.client.post("/api/v1/research/search", json=payload)
            self.assertEqual(embed.call_count, 2)
            self.assertNotEqual(first.json(), other.json())
            find.return_value = {**default, "updated_at": "v2"}
            self.assertEqual(self.client.post("/api/v1/research/search", json=payload).status_code, 200)
            self.assertEqual(embed.call_count, 3)
            find.return_value = None
            self.assertEqual(self.client.post("/api/v1/research/search", json=payload).status_code, 404)

    def test_simple_external_route_keeps_tavily_provenance_and_verifies_once(self):
        self.client.app.state.answer_provider = OpenRouterAnswerProvider("key", ModelRouter())
        self.client.app.state.external_search = TavilyExternalSearch("test-key")
        passage = "The Alpine station opened in 2012."
        url = "https://example.edu/alpine-station"
        question = {"paper_id": "paper-1", "question": "When did the Alpine station open?"}
        assessment = {"level": "none", "chunk_ids": [], "missing_information": "", "supporting_quotes": []}
        attempts = []
        def reply_for(endpoint, **kwargs):
            response = Mock(status_code=200)
            if "tavily.com" in endpoint:
                response.json.return_value = {"results": [{"title": "Alpine station opening",
                    "url": url, "content": passage, "published_date": "2022-01-01"}]}
                return response
            body = kwargs["json"]
            stage = body.get("response_format", {}).get("json_schema", {}).get("name")
            attempts.append(body["model"])
            if stage == "citewise_evidence_assessment":
                value = assessment
            else:
                source_id = re.search(r"(?:\[|Source ID: )(ext[\w-]+)", body["messages"][1]["content"]).group(1)
                if stage == "citewise_external_answer":
                    value = {"pdf_text": "", "pdf_ids": [], "external_text": f"{passage[:-1]} [{source_id}].",
                             "external_ids": [source_id]}
                else:
                    value = {"claims": [{"claim_id": "claim-1", "status": "supported",
                        "supporting": [{"source_id": source_id, "quote": passage}], "opposing": [],
                        "explanation": "Direct opening year.", "question_fit": "direct",
                        "evidence_use": "establishes_claim", "unsupported_source_ids": [], "scope_preserved": True}],
                        "requirements": [{"requirement": "Identify the opening year", "addressed_by": ["claim-1"]}]}
            response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(value)}}]}
            return response

        with patch("app.providers.openrouter_answers.requests.post", side_effect=reply_for) as post:
            result = self.client.post("/api/v1/research/ask", json=question)
        self.assertEqual(result.status_code, 200, result.text)
        data = result.json()
        self.assertEqual(data["verification_status"], "verified", data)
        self.assertEqual(data["sources"], [])
        self.assertEqual([source["url"] for source in data["external_sources"]], [url])
        self.assertEqual(data["search_trace"]["tavily_calls"], 1)
        self.assertEqual(attempts, [VERIFICATION_MODEL, FAST_MODEL, VERIFICATION_MODEL])
        self.assertEqual(post.call_count, 4)

    def test_evidence_gap_without_external_sources_skips_generation_and_verification(self):
        self.client.app.state.answer_provider = OpenRouterAnswerProvider("key", ModelRouter())
        assessment = {"level": "none", "chunk_ids": [], "missing_information": "", "supporting_quotes": []}
        reply = Mock(status_code=200)
        reply.json.return_value = {"choices": [{"finish_reason": "stop",
            "message": {"content": json.dumps(assessment)}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=reply) as post:
            response = self.client.post("/api/v1/research/ask",
                json={"paper_id": "paper-1", "question": "What was the antenna length?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["status"], ("not_mentioned", "insufficient_evidence"))
        self.assertEqual(response.json()["sources"], [])
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["model"], VERIFICATION_MODEL)

    def test_visual_analysis_requires_consent_insufficient_text_and_owned_page(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(), vision_enabled=True)
        self.client.app.state.answer_provider = provider
        with pymupdf.open() as pdf:
            pdf.new_page(width=120, height=140)
            pdf.new_page(width=160, height=180)
            self.client.app.state.ingestion_service.read_original = Mock(return_value=pdf.tobytes())
        original = self.client.app.state.ingestion_service.read_original
        route = "/api/v1/research/visual-analysis"
        payload = {"paper_id": "paper-1", "page_number": 2, "question": "What trend does the chart show?",
                   "evidence_type": "chart", "text_insufficient": True, "consent_to_send_page": True}
        good = Mock(status_code=200)
        good.json.return_value = {"choices": [{"message": {"content": '{"observation":"Trend unclear."}'}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=good) as post:
            for field in ("text_insufficient", "consent_to_send_page"):
                self.assertEqual(self.client.post(route, json={**payload, field: False}).status_code, 422)
            self.assertEqual(self.client.post(route, json={**payload, "paper_id": "other"}).status_code, 404)
            provider.vision_enabled = False
            self.assertEqual(self.client.post(route, json=payload).status_code, 503)
            original.assert_not_called()
            provider.vision_enabled = True
            self.assertEqual(self.client.post(route, json={**payload, "page_number": 3}).status_code, 404)
            post.assert_not_called()
            response = self.client.post(route, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["page_number"], 2)
        self.assertEqual(response.json()["verification_status"], "unverified")
        self.assertEqual(response.json()["model_id"], "nvidia/nemotron-3.5-super-vl:free")
        post.assert_called_once()
        body = post.call_args.kwargs["json"]
        self.assertTrue(body["provider"]["zdr"])
        image = body["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(image.startswith("data:image/png;base64,"))

    def test_rerank_opt_in_uses_authorized_candidates_and_falls_back_on_failure(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter())
        self.client.app.state.answer_provider = provider
        query = SearchRequest(paper_id="paper-1", question="Alpha?", limit=1)
        request = Request({"type": "http", "app": self.client.app})
        user = UserResponse(id="alice", email="alice@example.com", role="user")
        with patch.object(provider, "rerank", side_effect=[[1], AnswerRateLimitError("60")]) as rerank:
            original = asyncio.run(_retrieve(query, request, user, expand=True))
            rerank.assert_not_called()
            provider.rerank_enabled = True
            reranked = asyncio.run(_retrieve(query, request, user, expand=True))
            fallback = asyncio.run(_retrieve(query, request, user, expand=True))
        self.assertEqual(rerank.call_args.args, ("Alpha?", ["Alpha result", "Beta result"], 1))
        self.assertEqual([(item.chunk_index, item.page_number) for item in reranked], [(1, 2)])
        self.assertEqual(fallback, original)

    def test_ready_marker_cannot_bypass_failed_document_gate(self):
        repository = self.client.app.state.paper_repository
        from unittest.mock import AsyncMock
        with patch.object(repository, "find", new=AsyncMock(return_value={
            "status": "ready", "document_state": "READY_FOR_RESEARCH", "verification_checks": [
                {"check_id": "H4", "status": "FAIL"},
            ],
        })):
            result = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "What was measured?"})
        self.assertEqual(result.status_code, 409)
        self.assertFalse(self.client.app.state.conversation_repository.items)

    def test_missing_grounding_cannot_be_saved_as_verified(self):
        self.client.app.state.answer_provider = Answers(GeneratedAnswer("A reported outcome.", ["invented"]))
        result = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "What was measured?"})
        self.assertEqual(result.status_code, 200, result.text)
        public = result.json()
        self.assertEqual(public["answer_state"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(public["verification_status"], "not_verified")
        message = self.client.app.state.conversation_repository.items[public["conversation_id"]]["messages"][0]
        checks = {item["check_id"]: item for item in message["verification_checks"]}
        self.assertEqual(checks["AH4"]["status"], "FAIL")
        self.assertEqual(len(checks), 17)
        reopened = self.client.get(f"/api/v1/research/conversations/{public['conversation_id']}").json()["messages"][0]
        self.assertNotIn("verification_checks", reopened)
        self.assertEqual(reopened["answer_state"], public["answer_state"])

    def test_methodology_comparison_reports_unverified_parts(self):
        question = ("Compare the methodologies used in Study A and Study B. "
                    "Explain their study designs, participants, and important methodological differences.")
        def claim(text, status="supported"):
            return ClaimReviewResponse(text=text, section="pdf", status=status,
                                       supporting=[], opposing=[])
        study_a = claim("Study A used a randomized design with 2,400 students.")
        study_b = claim("Study B used a matched-course design with 780 students.")
        difference = claim("Study A randomized sections whereas Study B matched courses.")
        self.assertEqual(_methodology_comparison_gaps(question, [study_a, study_b, difference]), [])
        self.assertEqual(_methodology_comparison_gaps(question, [
            claim(study_a.text, "unsupported"), claim(study_b.text, "partially_supported"),
            claim(difference.text, "unsupported")]),
            ["Study A's design and participants", "Study B's design and participants",
             "the methodological differences"])
        self.assertEqual(_methodology_comparison_gaps(question, [study_a, study_b]),
                         ["the methodological differences"])
        partly_verified_b = ClaimReviewResponse(
            text="Study B used a matched-course design with 780 students and an unverified qualifier.",
            section="pdf", status="partially_supported",
            supporting=[ClaimQuoteResponse(source_id="cB",
                quote="Design: matched-course comparison. Participants: 780 students")], opposing=[])
        self.assertEqual(_methodology_comparison_gaps(question, [study_a, partly_verified_b, difference]), [])

    def test_coverage_checks_only_requested_dimensions(self):
        question = "Compare Study A and Study B designs and limitations."
        self.assertEqual(requested_coverage(question)[1], ["design", "limitations"])
        passages = ["Study A used a randomized design; a short period limits generalizability.",
                    "Study B used a matched design; volunteers may bias the result."]
        self.assertEqual(missing_coverage(question, passages), [])
        self.assertEqual(missing_coverage(question, passages[:1]),
                         [("Study B", "design"), ("Study B", "limitations")])
        self.assertEqual(requested_coverage("Compare Study A and Study B designs.")[1], ["design"])
        claims = [ClaimReviewResponse(text="Study C used a randomized design.", section="pdf",
                  status="supported", supporting=[], opposing=[]),
                  ClaimReviewResponse(text="Study D used a matched design.", section="pdf",
                  status="supported", supporting=[], opposing=[]),
                  ClaimReviewResponse(text="Study C randomized while Study D matched courses.", section="pdf",
                  status="supported", supporting=[], opposing=[])]
        self.assertEqual(_methodology_comparison_gaps("Compare Study C and Study D designs.", claims), [])
        self.assertEqual(_methodology_comparison_gaps(
            "Compare Study C and Study D designs and participants.", claims),
            ["Study C's participants", "Study D's participants"])
        self.assertEqual(_methodology_comparison_gaps(
            "What were Study C's design and participants?", claims[:1]),
            ["Study C's participants"])

    def test_pdf_scoped_gap_triggers_external_search(self):
        search = ExternalSearch()
        self.client.app.state.external_search = search
        provider = AssessedAnswers(EvidenceAssessment("none", []))
        self.client.app.state.answer_provider = provider
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Does this PDF report long-term outcomes?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "not_mentioned")
        self.assertEqual(response.json()["search_trace"]["tavily_calls"], 1)
        self.assertEqual(len(search.queries), 1)
        self.assertEqual(provider.external_calls, 1)
        provider.assessment = EvidenceAssessment("partial", ["c0-hash-a"],
            "Beta findings are not established.", [EvidenceQuote("c0-hash-a", "Alpha result")])
        partial = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "What does this paper say about Alpha and Beta?"})
        self.assertEqual(partial.status_code, 200)
        self.assertEqual(partial.json()["assessment"]["level"], "partial")
        self.assertEqual(partial.json()["sources"][0]["id"], "c0-hash-a")
        self.assertIn("external sources", partial.json()["answer"])
        self.assertEqual(len(search.queries), 2)
        self.assertTrue(_asks_only_about_pdf("What does this paper say about privacy?"))
        self.assertFalse(_asks_only_about_pdf("What are the current legal requirements in India?"))
        self.assertFalse(_asks_only_about_pdf("What does this paper say and what do external sources say?"))

    def test_contextual_pdf_follow_up_searches_external_evidence(self):
        search = ExternalSearch()
        self.client.app.state.external_search = search
        provider = AssessedAnswers(EvidenceAssessment("sufficient", ["c0-hash-a"],
            supporting_quotes=[EvidenceQuote("c0-hash-a", "Alpha result")]))
        self.client.app.state.answer_provider = provider
        first = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Alpha?"})
        self.assertEqual(first.status_code, 200)
        provider.assessment = EvidenceAssessment("partial", ["c0-hash-a"],
            "The result's statistical uncertainty was not verified.",
            [EvidenceQuote("c0-hash-a", "Alpha result")])
        follow_up = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Was that result statistically established?",
            "conversation_id": first.json()["conversation_id"]})
        self.assertEqual(follow_up.status_code, 200)
        body = follow_up.json()
        self.assertEqual(body["status"], "partially_supported")
        self.assertEqual(body["search_trace"]["tavily_calls"], 1)
        self.assertTrue(body["external_sources"])
        self.assertEqual([item["id"] for item in body["sources"]], ["c0-hash-a"])
        self.assertEqual(len(search.queries), 1)
        self.assertEqual(provider.external_calls, 1)
        self.assertIn("external sources", body["answer"].lower())

    def test_verified_pdf_partial_result_continues_to_external_search(self):
        class PartiallyVerified(AssessedAnswers):
            def generate(self, question, evidence):
                return GeneratedAnswer("Alpha result [c0-hash-a].", ["c0-hash-a"])

        search = ExternalSearch()
        self.client.app.state.external_search = search
        provider = PartiallyVerified(EvidenceAssessment("partial", ["c0-hash-a"],
            "The selected PDF chunks contain information omitted from the cited quotes.",
            [EvidenceQuote("c0-hash-a", "Alpha result")]))
        self.client.app.state.answer_provider = provider
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "What do the Alpha and Beta results establish?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["search_trace"]["tavily_calls"], 1)
        self.assertEqual(provider.external_calls, 1)
        self.assertTrue(response.json()["external_sources"])

    def test_follow_up_scope_uses_current_request_and_paper_bound_parent(self):
        conversation = {"messages": [{"id": "first", "question":
            "What did Trial Maple report about recovery?"}]}
        self.assertTrue(_follow_up_uses_pdf("Was that difference reliable?",
            "Previous question: What did Trial Maple report about recovery?\n"
            "Follow-up question: Was that difference reliable?", conversation, "first"))
        self.assertTrue(_follow_up_uses_pdf("How were participants selected?",
            "How were participants selected?", conversation, "first"))
        self.assertFalse(_follow_up_uses_pdf("Compare that with recent literature.",
            "Previous question: What did Trial Maple report about recovery?\n"
            "Follow-up question: Compare that with recent literature.", conversation, "first"))
        self.assertTrue(_follow_up_uses_pdf("Did this PDF discuss online learning?",
            "Did this PDF discuss online learning?", conversation, "first"))
        external_parent = {"messages": [{"id": "first", "question":
            "What do external sources report about recovery?"}]}
        self.assertFalse(_follow_up_uses_pdf("What about that?",
            "Previous question: What do external sources report about recovery?\n"
            "Follow-up question: What about that?", external_parent, "first"))
        self.assertTrue(_follow_up_uses_pdf("What does this PDF say about recovery?",
            "What does this PDF say about recovery?", external_parent, "first"))
        self.assertTrue(_asks_only_about_pdf("What does this PDF report about current law?"))
        self.assertFalse(_asks_only_about_pdf(
            "What does this PDF report, and what does current law require?"))

    def test_short_follow_up_can_retrieve_subject_from_verified_prior_answer(self):
        conversation = {"messages": [{"id": "first", "question": "What was reported?",
            "claims": [{"text": "Trial Maple reported reduced recovery time.",
                        "section": "pdf", "status": "supported",
                        "supporting": [{"source_id": "c1", "quote": "reduced recovery time"}]},
                       {"text": "Trial Cedar reported a larger effect.",
                        "section": "pdf", "status": "unsupported", "supporting": []}]}]}
        resolved = _question_with_context("What were its limitations?", conversation, "first")
        self.assertIn("Trial Maple reported reduced recovery time", resolved)
        self.assertNotIn("Trial Cedar", resolved)
        self.assertTrue(resolved.endswith("Follow-up question: What were its limitations?"))

    def test_contextual_follow_up_explicitly_requesting_external_evidence_still_searches(self):
        search = ExternalSearch()
        self.client.app.state.external_search = search
        provider = AssessedAnswers(EvidenceAssessment("sufficient", ["c0-hash-a"],
            supporting_quotes=[EvidenceQuote("c0-hash-a", "Alpha result")]))
        self.client.app.state.answer_provider = provider
        first = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Alpha?"})
        self.assertEqual(first.status_code, 200)
        provider.assessment = EvidenceAssessment("partial", ["c0-hash-a"],
            "External context is missing.", [EvidenceQuote("c0-hash-a", "Alpha result")])
        follow_up = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "How does that Alpha result compare with recent research outside this PDF?",
            "conversation_id": first.json()["conversation_id"]})
        self.assertEqual(follow_up.status_code, 200)
        self.assertEqual(follow_up.json()["search_trace"]["tavily_calls"], 1)
        self.assertEqual(follow_up.json()["external_sources"][0]["id"], "ext1")
        self.assertEqual(follow_up.json()["external_sources"][0]["url"],
                         "https://example.org/study")
        self.assertEqual([item["id"] for item in follow_up.json()["sources"]], ["c0-hash-a"])
        self.assertTrue(any(item["section"] == "external" and item["status"] == "supported"
                            for item in follow_up.json()["claims"]))
        self.assertEqual(provider.external_calls, 1)

    def test_invalid_external_generation_retains_pdf_evidence_without_external_claim(self):
        class InvalidExternal(AssessedAnswers):
            def generate_with_external(self, question, pdf_evidence, external_evidence, missing_information):
                self.external_calls += 1
                return ExternalAnswer("PDF result [c0-hash-a].", "Uncited external assertion.",
                                      ["c0-hash-a"], [])

        self.client.app.state.external_search = ExternalSearch()
        self.client.app.state.answer_provider = InvalidExternal(EvidenceAssessment(
            "partial", ["c0-hash-a"], "External context is missing.",
            [EvidenceQuote("c0-hash-a", "Alpha result")]))
        result = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Compare this PDF with external research on Alpha."})
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(body["status"], "partially_supported")
        self.assertEqual([item["id"] for item in body["sources"]], ["c0-hash-a"])
        self.assertEqual(body["external_sources"], [])
        self.assertNotIn("Uncited external assertion", body["answer"])
        self.assertFalse(_asks_only_about_pdf("What does this paper say, and what is the current law?"))

    def test_retrieval_recovers_second_study_outside_dense_candidates(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        chunks = [TextChunk(index, index + 1,
                    f"Study A used a randomized design with {200 + index} participants.", f"a{index}")
                  for index in range(7)]
        chunks.append(TextChunk(7, 8, "Study B used a matched-course comparison with 780 students.", "b7"))
        chunks.append(TextChunk(8, 9, "Study B library opening hours and book lending.", "b8"))
        vectors.save("alice", "paper-1", chunks, [[1.0, 0.0]] * 7 + [[0.0, 1.0]] * 2)
        class FocusedEmbeddings:
            def embed(self, questions):
                return [[0.0, 1.0] if text.startswith("Study B") else [1.0, 0.0]
                        for text in questions]
        ingestion = Ingestion(vectors)
        ingestion.embeddings = FocusedEmbeddings()
        self.client.app.state.ingestion_service = ingestion
        request = Request({"type": "http", "app": self.client.app})
        user = UserResponse(id="alice", email="alice@example.com", role="user")
        payload = SearchRequest(paper_id="paper-1",
            question="Compare Study A and Study B designs and participants.")
        initial = asyncio.run(_retrieve(payload, request, user))
        with self.assertLogs("app.routers.research", level="INFO") as logs:
            expanded = asyncio.run(_retrieve(payload, request, user, expand=True))
        self.assertTrue(any("stage=initial" in line and "c7-b7" in line for line in logs.output))
        self.assertFalse(any("Bob's alpha secret" in line for line in logs.output))
        self.assertTrue(any(item.chunk_index == 7 for item in initial))
        self.assertTrue(any(item.chunk_index == 7 for item in expanded))
        self.assertFalse(any(item.chunk_index == 8 for item in expanded))
        self.assertLessEqual(len(expanded), 10)
        self.assertEqual(next(item.page_number for item in expanded if item.chunk_index == 7), 8)
        self.assertTrue(all("Bob's" not in item.text for item in expanded))

    def test_single_dimension_gap_expands_retrieval_without_changing_paper_scope(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        chunks = [TextChunk(index, index + 1, f"Background note {index} on the trial.",
                            f"background-{index}") for index in range(6)]
        chunks.append(TextChunk(6, 9, "The intervention lasted 12 weeks.", "duration"))
        vectors.save("alice", "paper-1", chunks, [[1.0, 0.0]] * 6 + [[0.0, 1.0]])
        vectors.save("bob", "paper-1", [TextChunk(7, 10, "The intervention lasted 20 weeks.",
                                                    "other-owner")], [[0.0, 1.0]])

        class FocusedEmbeddings:
            def embed(self, questions):
                return [[0.0, 1.0] if text.startswith("duration") else [1.0, 0.0]
                        for text in questions]

        ingestion = Ingestion(vectors)
        ingestion.embeddings = FocusedEmbeddings()
        self.client.app.state.ingestion_service = ingestion
        request = Request({"type": "http", "app": self.client.app})
        user = UserResponse(id="alice", email="alice@example.com", role="user")
        payload = SearchRequest(paper_id="paper-1", question="What was the duration?", limit=5)
        initial = asyncio.run(_retrieve(payload, request, user))
        expanded = asyncio.run(_retrieve(payload, request, user, expand=True))
        self.assertFalse(any(item.chunk_index == 6 for item in initial))
        self.assertTrue(any(item.chunk_index == 6 and item.page_number == 9 for item in expanded))
        self.assertFalse(any(item.chunk_index == 7 for item in expanded))
        self.assertLessEqual(len(expanded), 10)

    def test_sufficient_assessment_is_downgraded_when_quote_misses_requested_part(self):
        question = "Compare Study A and Study B designs and participants."
        a = AnswerSource(id="cA", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text="Study A used a randomized design with 200 participants.")
        b = AnswerSource(id="cB", chunk_index=1, page_number=2, content_hash="b",
            highlight_text="", text="Study B used a matched design with 780 students.")
        assessment = EvidenceAssessment("sufficient", ["cA", "cB"], supporting_quotes=[
            EvidenceQuote("cA", "Study A used a randomized design with 200 participants."),
            EvidenceQuote("cB", "Study B used a matched design")])
        validated = _validate_assessment(assessment, question, {"cA": a, "cB": b})
        self.assertEqual(validated.level, "partial")
        self.assertIn("Study B participants", validated.missing_information)
        self.assertIn("omitted from the cited quotes", validated.missing_information)
        self.assertNotIn("do not establish Study B participants", validated.missing_information)

    def test_assessment_context_retains_fact_outside_best_sentence_window(self):
        text = ("The survey examined remote work and attendance. "
                "The survey examined remote work and attendance again. "
                "Participants: 48 nurses. The observation lasted six months.")
        context = _evidence_context("How many participants were in the survey?", text)
        self.assertIn("48 nurses", context)
        self.assertIn("six months", context)
        self.assertLessEqual(len(context), 2400)

    def test_evidence_handoff_logs_chunk_provenance_without_private_text(self):
        passage = ChunkResponse(chunk_index=4, page_number=7, content_hash="tracehash",
            text="Trial Juniper measured recovery in 54 patients.")
        source_id = "c4-tracehash"

        class Provider:
            def assess(self, question, evidence):
                return EvidenceAssessment("sufficient", [source_id], supporting_quotes=[
                    EvidenceQuote(source_id, "Trial Juniper measured recovery in 54 patients")])

            def generate(self, question, evidence):
                return GeneratedAnswer(f"Trial Juniper included 54 patients [{source_id}].", [source_id])

            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(item["id"], "supported", [ClaimQuote(
                    source_id, "Trial Juniper measured recovery in 54 patients")], []) for item in claims]

        self.client.app.state.answer_provider = Provider()
        with patch("app.routers.research._retrieve", return_value=[passage]), \
                self.assertLogs("uvicorn.error", level="INFO") as logs:
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "How many patients were included?",
                "request_id": "trace-123"})
        self.assertEqual(response.status_code, 200)
        events = "\n".join(logs.output)
        for stage in ("retrieved", "assessment_input", "assessment_quoted",
                      "assessment_accepted", "generation_input", "generated_claims", "verified_claims"):
            self.assertIn(f"stage={stage}", events)
        self.assertIn("request_id=trace-123", events)
        self.assertIn("c4-tracehash", events)
        self.assertIn("'page_number': 7", events)
        self.assertNotIn("Trial Juniper measured", events)
        self.assertNotIn("How many patients", events)

    def test_verified_answer_parts_use_claim_status_and_cited_source(self):
        sources = {
            "c1": AnswerSource(id="c1", chunk_index=0, page_number=1, content_hash="one",
                highlight_text="", text="Trial Juniper enrolled 54 patients for six months. Recovery improved."),
            "c2": AnswerSource(id="c2", chunk_index=1, page_number=2, content_hash="two",
                highlight_text="", text="Trial Willow enrolled 60 patients. Recovery was unchanged."),
        }
        claims = [
            ClaimReviewResponse(text="Trial Juniper enrolled 54 patients.", section="pdf",
                status="supported", supporting=[ClaimQuoteResponse(source_id="c1", quote="54 patients")], opposing=[]),
            ClaimReviewResponse(text="Trial Juniper lasted six months.", section="pdf",
                status="supported", supporting=[ClaimQuoteResponse(source_id="c1", quote="six months")], opposing=[]),
            ClaimReviewResponse(text="Trial Willow enrolled 60 patients.", section="pdf",
                status="partially_supported", supporting=[ClaimQuoteResponse(source_id="c2",
                    quote="Trial Willow enrolled 60 patients")], opposing=[]),
            ClaimReviewResponse(text="Trial Willow improved recovery.", section="pdf",
                status="unsupported", supporting=[], opposing=[]),
        ]
        parts = _verified_answer_parts(
            "Compare both studies' participants, duration and findings.", claims, sources)
        self.assertIn(("Trial Juniper", "participants", "supported", ["c1"]), parts)
        self.assertIn(("Trial Juniper", "duration", "supported", ["c1"]), parts)
        self.assertIn(("Trial Willow", "participants", "partially_supported", ["c2"]), parts)
        self.assertIn(("Trial Willow", "findings", "unsupported", []), parts)
        self.assertIn(("Trial Willow", "duration", "unsupported", []), parts)

    def test_implicit_comparison_final_gap_keeps_each_study_separate(self):
        claims = [ClaimReviewResponse(text="Trial Juniper enrolled 54 patients.", section="pdf",
            status="supported", supporting=[ClaimQuoteResponse(source_id="c1", quote="54 patients")],
            opposing=[])]
        gaps = _methodology_comparison_gaps("Compare both studies' participants.", claims,
            ["Trial Juniper enrolled 54 patients.", "Trial Willow enrolled 60 patients."])
        self.assertIn("Trial Willow's participants", gaps)

    def test_answer_part_tracking_across_factual_numerical_and_synthesis_requests(self):
        source = AnswerSource(id="c7", chunk_index=7, page_number=4, content_hash="seven",
            highlight_text="", text="Cohort Ember used interviews with 31 participants. Retention improved.")
        cases = [
            ("What was Cohort Ember's design?", "Cohort Ember used interviews.", "design"),
            ("How many participants were in Cohort Ember?", "Cohort Ember included 31 participants.",
             "participants"),
            ("What were Cohort Ember's findings?", "Cohort Ember reported improved retention.",
             "findings"),
        ]
        for question, statement, dimension in cases:
            with self.subTest(dimension=dimension):
                claim = ClaimReviewResponse(text=statement, section="pdf", status="supported",
                    supporting=[ClaimQuoteResponse(source_id="c7", quote=source.text)], opposing=[])
                parts = _verified_answer_parts(question, [claim], {"c7": source})
                self.assertIn(("Cohort Ember", dimension, "supported", ["c7"]), parts)

    def test_verified_answer_does_not_claim_complete_when_requested_part_is_omitted(self):
        passage = ChunkResponse(chunk_index=5, page_number=3, content_hash="juniper",
            text="Trial Juniper used interviews and enrolled 54 participants.")
        source_id = "c5-juniper"

        class Provider:
            def assess(self, question, evidence):
                return EvidenceAssessment("sufficient", [source_id], supporting_quotes=[
                    EvidenceQuote(source_id, passage.text)])

            def generate(self, question, evidence):
                return GeneratedAnswer(f"Trial Juniper used interviews [{source_id}].", [source_id])

            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(item["id"], "supported", [ClaimQuote(source_id,
                    "Trial Juniper used interviews")], []) for item in claims]

        self.client.app.state.answer_provider = Provider()
        with patch("app.routers.research._retrieve", return_value=[passage]):
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "How many participants were in Trial Juniper?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "partially_supported")
        self.assertIn("cited claims", response.json()["answer"])
        self.assertNotIn("entire PDF", response.json()["answer"])

    def test_long_assessment_context_keeps_requested_fact_within_budget(self):
        text = ("The survey examined remote work and attendance. " * 65
                + "Participants: 48 patients. The observation lasted six months.")
        context = _evidence_context("How many participants were in the survey?", text)
        self.assertIn("48 patients", context)
        self.assertLessEqual(len(context), 2400)

    def test_single_overlong_sentence_keeps_late_requested_fact(self):
        text = "Background context " * 170 + "participants numbered 54 patients"
        context = _evidence_context("How many participants were there?", text)
        self.assertIn("54 patients", context)
        self.assertLessEqual(len(context), 2400)

    def test_assessment_gap_distinguishes_absent_fact_from_omitted_quote(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text="Trial Meridian used a cohort design. Participants: 48 patients.")
        assessment = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", "Trial Meridian used a cohort design")])
        result = _validate_assessment(assessment,
            "What were Trial Meridian's design, participants and duration?", {"c0": source})
        self.assertEqual(result.level, "partial")
        self.assertIn("omitted from the cited quotes for Trial Meridian participants", result.missing_information)
        self.assertIn("do not establish Trial Meridian duration", result.missing_information)

    def test_implicit_subject_gap_uses_original_subject_identity_after_filtering(self):
        source = AnswerSource(id="c1", chunk_index=3, page_number=2, content_hash="one",
            highlight_text="", text="Trial Willow enrolled 60 participants.")
        result = _coverage_gap_message("Compare both studies' participant counts.",
            [("Trial Willow", "participants")], {"c1": source}, ["c1"],
            subjects=["Trial Juniper", "Trial Willow"])
        self.assertIn("omitted from the cited quotes for Trial Willow participants", result)
        self.assertNotIn("do not establish Trial Willow participants", result)

    def test_partial_assessment_does_not_report_unrequested_dimension(self):
        source = AnswerSource(id="c1", chunk_index=1, page_number=1, content_hash="one",
            highlight_text="", text="Trial Willow reported improved recovery.")
        reviewed = _validate_assessment(EvidenceAssessment("partial", ["c1"],
            "duration was not established", [EvidenceQuote("c1", "Trial Willow reported improved recovery")]),
            "What findings did Trial Willow report?", {"c1": source})
        self.assertIsNotNone(reviewed)
        self.assertEqual(reviewed.level, "partial")
        self.assertNotIn("duration", reviewed.missing_information)
        self.assertIn("requested", reviewed.missing_information)

    def test_partial_assessment_does_not_mix_requested_and_unrequested_gaps(self):
        source = AnswerSource(id="c1", chunk_index=1, page_number=1, content_hash="one",
            highlight_text="", text="Trial Willow reported improved recovery.")
        reviewed = _validate_assessment(EvidenceAssessment("partial", ["c1"],
            "findings and duration could not be verified",
            [EvidenceQuote("c1", "Trial Willow reported improved recovery")]),
            "What findings did Trial Willow report?", {"c1": source})
        self.assertIsNotNone(reviewed)
        self.assertNotIn("duration", reviewed.missing_information)
        self.assertIn("requested", reviewed.missing_information)

    def test_unnamed_comparison_gap_needs_both_studies_verified(self):
        sources = {
            "c1": AnswerSource(id="c1", chunk_index=1, page_number=1, content_hash="one",
                highlight_text="", text="Trial Juniper enrolled 54 participants."),
            "c2": AnswerSource(id="c2", chunk_index=2, page_number=2, content_hash="two",
                highlight_text="", text="Trial Willow enrolled 60 participants."),
        }
        assessment = EvidenceReview(level="partial", chunk_ids=list(sources),
            missing_information="participant counts remain unverified")
        one_claim = ClaimReviewResponse(text="Trial Juniper enrolled 54 participants.", section="pdf",
            status="supported", supporting=[ClaimQuoteResponse(source_id="c1", quote="54 participants")],
            opposing=[])
        question = "Compare both studies' participant counts."
        unresolved = _reconcile_assessment_gap(assessment, [one_claim], question, sources)
        self.assertEqual(unresolved.level, "partial")
        both_claims = [one_claim, ClaimReviewResponse(text="Trial Willow enrolled 60 participants.",
            section="pdf", status="supported", supporting=[ClaimQuoteResponse(source_id="c2",
                quote="60 participants")], opposing=[])]
        resolved = _reconcile_assessment_gap(assessment, both_claims, question, sources)
        self.assertEqual(resolved.level, "sufficient")

    def test_verified_claim_resolves_matching_assessment_gap(self):
        assessment = EvidenceReview(level="partial", chunk_ids=["c0"],
            missing_information="Trial Meridian participants")
        claim = ClaimReviewResponse(text="Trial Meridian included 48 patients.", section="pdf",
            status="supported", supporting=[ClaimQuoteResponse(source_id="c0", quote="48 patients")],
            opposing=[])
        resolved = _reconcile_assessment_gap(assessment, [claim])
        self.assertEqual(resolved.level, "sufficient")
        self.assertEqual(resolved.missing_information, "")
        self.assertEqual(assessment.level, "partial")
        absent = _reconcile_assessment_gap(EvidenceReview(level="partial", chunk_ids=["c0"],
            missing_information="Trial Meridian duration"), [claim])
        self.assertEqual(absent.level, "partial")
        self.assertIn("duration", absent.missing_information)

    def test_partly_resolved_assessment_keeps_only_unverified_gap(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text="Trial Meridian used a cohort design. Participants: 48 patients.")
        question = "What were Trial Meridian's design, participants and duration?"
        assessment = EvidenceReview(level="partial", chunk_ids=["c0"],
            missing_information=("The selected PDF chunks contain information omitted from the cited "
                                 "quotes for Trial Meridian participants. The selected PDF chunks do not "
                                 "establish Trial Meridian duration."))
        claim = ClaimReviewResponse(text="Trial Meridian included 48 patients.", section="pdf",
            status="supported", supporting=[ClaimQuoteResponse(source_id="c0", quote="48 patients")],
            opposing=[])
        remaining = _reconcile_assessment_gap(assessment, [claim], question, {"c0": source})
        self.assertEqual(remaining.level, "partial")
        self.assertIn("Trial Meridian duration", remaining.missing_information)
        self.assertNotIn("Trial Meridian participants", remaining.missing_information)

    def test_resolved_gap_does_not_reopen_other_assessed_parts(self):
        sources = {"c0": AnswerSource(id="c0", chunk_index=0, page_number=2,
            content_hash="a", highlight_text="",
            text="Trial Meridian used a cohort design and enrolled 48 patients.")}
        assessment = EvidenceReview(level="partial", chunk_ids=["c0"],
            missing_information="The selected PDF chunks contain information omitted from the cited quotes for Trial Meridian participants.")
        claim = ClaimReviewResponse(text="Trial Meridian enrolled 48 patients.", section="pdf",
            status="supported", supporting=[ClaimQuoteResponse(source_id="c0",
                quote="enrolled 48 patients")], opposing=[])
        resolved = _reconcile_assessment_gap(assessment, [claim],
            "What were Trial Meridian's design and participant count?", sources)
        self.assertEqual(resolved.level, "sufficient")
        self.assertEqual(resolved.missing_information, "")

    def test_reconciliation_keeps_subject_dimension_pairs_separate(self):
        sources = {
            "c0": AnswerSource(id="c0", chunk_index=0, page_number=1, content_hash="a",
                highlight_text="", text="Trial Cedar enrolled 42 patients."),
            "c1": AnswerSource(id="c1", chunk_index=1, page_number=2, content_hash="b",
                highlight_text="", text="Trial Maple lasted six weeks."),
        }
        assessment = EvidenceReview(level="partial", chunk_ids=list(sources),
            missing_information=("The selected PDF chunks do not establish Trial Cedar participants, "
                                 "Trial Maple duration."))
        claims = [ClaimReviewResponse(text="Trial Cedar enrolled 42 patients.", section="pdf",
            status="supported", supporting=[ClaimQuoteResponse(source_id="c0",
                quote="enrolled 42 patients")], opposing=[])]
        result = _reconcile_assessment_gap(assessment, claims,
            "Compare Trial Cedar and Trial Maple participant counts and duration.", sources)
        self.assertIn("Trial Maple duration", result.missing_information)
        self.assertNotIn("Trial Cedar participants", result.missing_information)
        self.assertNotIn("Trial Cedar duration", result.missing_information)

    def test_implicit_comparison_gap_is_not_cleared_by_one_study_claim(self):
        sources = {
            "c1": AnswerSource(id="c1", chunk_index=0, page_number=1, content_hash="a",
                highlight_text="", text="Trial Solstice used a cohort design."),
            "c2": AnswerSource(id="c2", chunk_index=1, page_number=2, content_hash="b",
                highlight_text="", text="Trial Polaris used a randomized design."),
        }
        assessment = EvidenceReview(level="partial", chunk_ids=list(sources),
            missing_information="The selected PDF chunks do not establish Trial Polaris design.")
        claim = ClaimReviewResponse(text="Trial Solstice used a cohort design.", section="pdf",
            status="supported", supporting=[ClaimQuoteResponse(source_id="c1",
                quote="Trial Solstice used a cohort design")], opposing=[])
        remaining = _reconcile_assessment_gap(assessment, [claim],
            "Compare both studies' designs.", sources)
        self.assertEqual(remaining.level, "partial")
        self.assertIn("Trial Polaris design", remaining.missing_information)

    def test_selected_chunk_gap_is_sent_to_normal_verification(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text="Trial Meridian used a cohort design. Participants: 48 patients.")
        assessment = EvidenceReview(level="partial", chunk_ids=["c0"],
            missing_information="Trial Meridian participants")
        self.assertTrue(_assessment_gap_in_selected_chunks(assessment, {"c0": source}))
        self.assertFalse(_assessment_gap_in_selected_chunks(EvidenceReview(
            level="partial", chunk_ids=["c0"], missing_information="Trial Meridian duration"),
            {"c0": source}))

    def test_verified_answer_clears_assessment_gap_in_final_response(self):
        passage = ChunkResponse(chunk_index=3, page_number=2, content_hash="meridian",
            text="Trial Meridian used a cohort design. Participants: 48 patients.")
        source_id = "c3-meridian"

        class Provider:
            def assess(self, question, evidence):
                return EvidenceAssessment("sufficient", [source_id], supporting_quotes=[
                    EvidenceQuote(source_id, "Trial Meridian used a cohort design")])

            def generate(self, question, evidence):
                return GeneratedAnswer(
                    f"Trial Meridian used a cohort design and included 48 patients [{source_id}].",
                    [source_id])

            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(item["id"], "supported", [ClaimQuote(
                    source_id, passage.text)], []) for item in claims]

        self.client.app.state.answer_provider = Provider()
        with patch("app.routers.research._retrieve", return_value=[passage]):
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "What does this PDF report about Trial Meridian design and participants?"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["assessment"]["level"], "sufficient")
        self.assertEqual(result["assessment"]["missing_information"], "")
        self.assertEqual(result["status"], "answered")
        self.assertIn("48 patients", result["answer"])
        self.assertNotIn("could not", result["answer"])

    def test_invalid_or_unrelated_assessment_quote_still_rejected(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text="Trial Meridian included 48 patients.")
        invalid = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", "Trial Meridian included 80 patients")])
        self.assertIsNone(_validate_assessment(invalid, "How many patients were in Trial Meridian?",
                                              {"c0": source}))
        unrelated = AnswerSource(id="c1", chunk_index=1, page_number=2, content_hash="b",
            highlight_text="", text="The archives catalogued medieval maps and manuscripts.")
        irrelevant = EvidenceAssessment("sufficient", ["c1"], supporting_quotes=[
            EvidenceQuote("c1", "The archives catalogued medieval maps and manuscripts")])
        self.assertIsNone(_validate_assessment(irrelevant,
                                              "How many patients were in Trial Meridian?", {"c1": unrelated}))

    def test_partial_comparison_is_not_reported_as_a_complete_answer(self):
        question = ("Compare the methodologies used in Study A and Study B. "
                    "Explain their study designs, participants, and important methodological differences.")
        study_a = ChunkResponse(chunk_index=3, page_number=3, content_hash="aaaaaaaaaaaa",
            text="Study A used a randomized design with 2,400 students.")
        study_b = ChunkResponse(chunk_index=4, page_number=4, content_hash="bbbbbbbbbbbb",
            text="Study B used a matched-course design with 780 students.")

        class PartialComparisonProvider:
            def assess(self, question, evidence):
                return EvidenceAssessment("sufficient", ["c3-aaaaaaaaaaaa", "c4-bbbbbbbbbbbb"],
                    supporting_quotes=[EvidenceQuote("c3-aaaaaaaaaaaa", study_a.text),
                                       EvidenceQuote("c4-bbbbbbbbbbbb", study_b.text)])

            def generate(self, question, evidence):
                return GeneratedAnswer("Study A used a randomized design with 2,400 students "
                    "(c3-aaaaaaaaaaaa). Study B used a matched-course design with 780 students "
                    "(c4-bbbbbbbbbbbb). Study A randomized sections whereas Study B matched courses "
                    "(c3-aaaaaaaaaaaa) (c4-bbbbbbbbbbbb).",
                    ["c3-aaaaaaaaaaaa", "c4-bbbbbbbbbbbb"])

            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(item["id"],
                    "partially_supported" if "Study B used" in item["text"] else "unsupported",
                    [ClaimQuote("c4-bbbbbbbbbbbb", study_b.text)] if "Study B used" in item["text"] else [],
                    []) for item in claims]

        self.client.app.state.answer_provider = PartialComparisonProvider()
        with patch("app.routers.research._retrieve", return_value=[study_a, study_b]):
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": question, "request_id": "comparison-request"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["request_id"], "comparison-request")
        self.assertEqual(result["status"], "partially_supported")
        self.assertIn("Study A's design and participants", result["answer"])
        self.assertIn("the methodological differences", result["answer"])
        self.assertNotIn("Study A used a randomized design", result["answer"])

    def test_returns_ranked_passages_with_page_and_owner_filter(self):
        response = self.client.post("/api/v1/research/search", json={
            "paper_id": "paper-1", "question": "Alpha?", "limit": 2,
        })
        self.assertEqual(response.status_code, 200)
        passages = response.json()["passages"]
        self.assertEqual([passage["text"] for passage in passages], ["Alpha result", "Beta result"])
        self.assertEqual([passage["page_number"] for passage in passages], [1, 2])
        self.assertTrue(all("secret" not in passage["text"] for passage in passages))

    def test_comparison_retrieval_does_not_force_off_topic_study(self):
        vectors = self.client.app.state.ingestion_service.vectors
        vectors.save("alice", "paper-1", [
            TextChunk(3, 4, "Study A reported integrity and privacy concerns.", "a"),
            TextChunk(4, 5, "Study B reported 120 alerts; 9 were confirmed and 111 remained unconfirmed.", "b"),
            TextChunk(5, 6, "Integrity and privacy concerns require careful review.", "other"),
        ], [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        passages = vectors.search("alice", "paper-1", [1.0, 0.0], 2,
                                  question="Compare integrity and privacy findings from both studies")
        self.assertTrue(any("Study A" in item.text for item in passages))
        self.assertFalse(any("120 alerts" in item.text for item in passages))

    def test_comparison_retrieval_includes_distinct_study_sections(self):
        vectors = self.client.app.state.ingestion_service.vectors
        vectors.save("alice", "paper-1", [
            TextChunk(3, 3, "Study A and Study B: summary of monitoring results.", "table"),
            TextChunk(4, 4, "Study A found lower misconduct in monitored sections.", "a"),
            TextChunk(5, 5, "Study B reported no statistically significant misconduct difference.", "b"),
        ], [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        passages = vectors.search("alice", "paper-1", [1.0, 0.0], 2,
                                  question="Compare Study A and Study B misconduct results")
        self.assertTrue(any("Study A found" in item.text for item in passages))
        self.assertTrue(any("Study B reported" in item.text for item in passages))

    def test_comparison_retrieval_prefers_topic_evidence_over_study_label(self):
        vectors = self.client.app.state.ingestion_service.vectors
        vectors.save("alice", "paper-1", [
            TextChunk(3, 3, "Study A found lower misconduct under surveillance.", "a"),
            TextChunk(4, 4, "Study B used three universities and one country.", "b-setting"),
            TextChunk(5, 5, "Study B found no significant difference in cheating outcomes.", "b-result"),
        ], [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        passages = vectors.search("alice", "paper-1", [1.0, 0.0], 2,
                                  question="Compare cheating outcomes in both studies")
        self.assertTrue(any("Study A found" in item.text for item in passages))
        self.assertTrue(any("Study B found" in item.text for item in passages))
        self.assertFalse(any("three universities" in item.text for item in passages))

    def test_comparison_does_not_force_unrelated_study_section(self):
        vectors = self.client.app.state.ingestion_service.vectors
        vectors.save("alice", "paper-1", [
            TextChunk(3, 3, "Study A measured student privacy concerns.", "a-privacy"),
            TextChunk(4, 4, "Study B used three universities and one country.", "b-setting"),
        ], [[1.0, 0.0], [0.0, 1.0]])
        passages = vectors.search("alice", "paper-1", [1.0, 0.0], 2,
                                  question="Compare privacy concerns in both studies")
        self.assertFalse(any("three universities" in item.text for item in passages))

    def test_calculated_answer_uses_the_same_claim_verifier_as_other_answers(self):
        source = AnswerSource(id="c1-orbit", page_number=2, chunk_index=1, content_hash="orbit",
            highlight_text="", text="Trial ORBIT reported rates of 8% in the first group and 14% in the second group.")
        class Verifier:
            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(claims[0]["id"], "supported", [
                    ClaimQuote(source.id, source.text)], [])]
        shown, _, reviews = asyncio.run(_verified_sections(Verifier(),
            "What was the percentage-point difference?", "14% − 8% = 6 percentage points [c1-orbit].", "",
            [source.id], [], {source.id: source}, {}))
        self.assertEqual(reviews[0].status, "supported")
        self.assertIn("6 percentage points", shown)
        self.assertIn(f"[{source.id}]", shown)
        bad, _, reviews = asyncio.run(_verified_sections(Verifier(),
            "What was the percentage-point difference?", "14% − 8% = 9 percentage points [c1-orbit].", "",
            [source.id], [], {source.id: source}, {}))
        self.assertEqual(reviews[0].status, "unsupported")
        self.assertNotIn("9 percentage points", bad)

    def test_explicit_pdf_gaps_do_not_block_correct_routing(self):
        legal = AnswerSource(id="c9-a", owner_id="alice", paper_id="paper-1", page_number=5,
            chunk_index=9, content_hash="a", highlight_text="", document_name="paper.pdf",
            text="These policy considerations are not a statement of current law in any country. "
                 "The studies do not test the legality of data collection in India.")
        absent = _validate_assessment(EvidenceAssessment("none", [], "", []),
            "What are the current legal requirements for biometric data collection in India?",
            {legal.id: legal})
        self.assertEqual(absent.level, "none")
        facial = AnswerSource(id="c11-b", owner_id="alice", paper_id="paper-1", page_number=6,
            chunk_index=11, content_hash="b", highlight_text="", document_name="paper.pdf",
            text="Facial recognition performance Not studied.")
        direct = _validate_assessment(EvidenceAssessment("none", [], "", []),
            "Was facial recognition accuracy studied?", {facial.id: facial})
        self.assertEqual(direct.level, "none")

    def test_display_citation_maps_to_stable_chunk_before_verification(self):
        source = AnswerSource(id="c6-study-b", owner_id="alice", paper_id="paper-1", page_number=4,
            chunk_index=6, content_hash="study-b", highlight_text="", document_name="dossier.pdf",
            text="Study B Setting: 3 universities. Participants: 780 students across 26 online courses.")
        class Verifier:
            def verify_claims(self, question, claims, evidence):
                self.claims = claims
                return [ClaimAssessment(claims[0]["id"], "supported", [
                    ClaimQuote(source.id, "3 universities"),
                    ClaimQuote(source.id, "780 students across 26 online courses")], [])]
        verifier = Verifier()
        answer, _, reviews = asyncio.run(_verified_sections(verifier,
            "How many students and universities did Study B involve?",
            "Study B involved 780 students at 3 universities across 26 courses [1].", "",
            [source.id], [], {source.id: source}, {}))
        self.assertEqual(verifier.claims[0]["cited_ids"], [source.id])
        self.assertEqual(reviews[0].status, "supported")
        self.assertIn(f"[{source.id}]", answer)

    def test_broad_questions_need_supported_assessments_not_paper_specific_overrides(self):
        texts = [
            "Study A (S-A) Study B (S-B) Misconduct result 8% vs 14%; lower with monitoring "
            "9% vs 10%; not significant Privacy concern 62% vs 18% 71% vs 24% "
            "Anxiety 34% vs 22% 41% vs 25%",
            "Study A lasted one semester at volunteer universities. Results may not generalize to every institution.",
            "Study B used a quasi-experimental design and was not randomized.",
            "Study A reports a lower misconduct rate. Study B reports a small difference. "
            "The dossier does not establish a universal effect. Privacy and anxiety findings raise concerns.",
        ]
        sources = {f"c{index}-test": AnswerSource(id=f"c{index}-test", owner_id="alice",
            paper_id="paper-1", page_number=index + 3, chunk_index=index, content_hash="test",
            highlight_text="", document_name="dossier.pdf", text=text)
            for index, text in enumerate(texts)}
        questions = [
            ("How do both studies balance academic integrity, privacy and anxiety?", ["c0-test"]),
            ("Compare Study A and Study B on privacy and misconduct.", ["c0-test"]),
            ("Should universities universally adopt remote proctoring?", ["c3-test"]),
            ("Can both studies generalize to all universities?", ["c1-test", "c2-test"]),
        ]
        for question, expected_ids in questions:
            with self.subTest(question=question):
                proposed = EvidenceAssessment("sufficient", expected_ids, supporting_quotes=[
                    EvidenceQuote(item, sources[item].text) for item in expected_ids])
                result = _validate_assessment(proposed, question, sources)
                self.assertEqual(result.level,
                    "partial" if ("both studies" in question or
                                  question.startswith("Compare Study A and Study B")) else "sufficient")
                self.assertEqual(result.chunk_ids, expected_ids)
                absent = _validate_assessment(EvidenceAssessment("none", []), question, sources)
                self.assertFalse(absent and absent.level == "sufficient")

    def test_none_assessment_does_not_authorize_search_when_pdf_has_multiple_topic_matches(self):
        source = AnswerSource(id="c1", chunk_index=1, page_number=2, content_hash="h",
            highlight_text="", text="Study A and Study B reported misconduct, privacy and anxiety outcomes.")
        result = _validate_assessment(EvidenceAssessment("none", []),
            "How do the two studies balance misconduct, privacy and anxiety?", {"c1": source})
        self.assertIsNone(result)

    def test_study_b_details_are_not_mistaken_for_missing_pdf_evidence(self):
        source = AnswerSource(id="c6-study-b", owner_id="alice", paper_id="paper-1", page_number=4,
            chunk_index=6, content_hash="study-b", highlight_text="", document_name="dossier.pdf",
            text="5. Study B: Monitoring and student trust Setting: 3 universities in one country "
                 "Participants: 780 students across 26 online courses Design: Quasi-experimental comparison")
        result = _validate_assessment(EvidenceAssessment("sufficient", [source.id],
            supporting_quotes=[EvidenceQuote(source.id, source.text)]),
            "How many students, universities and courses participated in Study B?", {source.id: source})
        self.assertEqual(result.level, "sufficient")
        self.assertEqual(result.chunk_ids, [source.id])

    def test_generalizability_uses_two_pdf_chunks_without_external_search(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        vectors.save("alice", "paper-1", [
            TextChunk(0, 3, "Study A lasted one semester and used volunteer universities. "
                      "Results may not generalize to every institution.", "a"),
            TextChunk(1, 4, "Study B used a quasi-experimental design and was not randomized.", "b"),
        ], [[0.0, 1.0], [0.0, 1.0]])
        self.client.app.state.ingestion_service = Ingestion(vectors)
        search = ExternalSearch()
        self.client.app.state.external_search = search

        class GeneralizationAnswers(AssessedAnswers):
            def generate(self, question, evidence):
                ids = [item.id for item in evidence]
                return GeneratedAnswer("Study A's volunteer setting and Study B's non-randomized design "
                                       f"limit generalizability [{ids[0]}] [{ids[1]}].", ids)

            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(claims[0]["id"], "supported", [
                    ClaimQuote(evidence[0].id, "volunteer universities. Results may not generalize"),
                    ClaimQuote(evidence[1].id, "Study B used a quasi-experimental design and was not randomized")], [])]

        self.client.app.state.answer_provider = GeneralizationAnswers(EvidenceAssessment(
            "sufficient", ["c0-a", "c1-b"], supporting_quotes=[
                EvidenceQuote("c0-a", "Study A lasted one semester and used volunteer universities"),
                EvidenceQuote("c1-b", "Study B used a quasi-experimental design and was not randomized")]))
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Can both studies generalize to all universities?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assessment"]["level"], "sufficient")
        self.assertEqual(response.json()["status"], "answered")
        self.assertEqual({item["page_number"] for item in response.json()["sources"]}, {3, 4})
        self.assertEqual(search.queries, [])

    def test_openrouter_failover_keeps_grounding_and_records_quality_provenance(self):
        budget = Mock()
        budget.reserve.side_effect = [None, None, None, None, UpstreamPoolLimit("60"), None]
        self.client.app.state.answer_provider = OpenRouterAnswerProvider("test-key", ModelRouter(), budget=budget)
        search = ExternalSearch()
        self.client.app.state.external_search = search
        outputs = [
            {"level": "sufficient", "chunk_ids": ["c0-hash-a"], "missing_information": "",
             "supporting_quotes": [{"chunk_id": "c0-hash-a", "quote": "Alpha result"}]},
            {"status": "answered", "text": "Alpha result [c0-hash-a].", "evidence_ids": ["c0-hash-a"]},
            {"claims": [{"claim_id": "claim-1", "status": "supported",
                         "supporting": [{"source_id": "c0-hash-a", "quote": "Alpha result"}],
                         "opposing": [], "explanation": "The passage states the result.",
                         "question_fit": "direct", "evidence_use": "establishes_claim",
                         "unsupported_source_ids": [], "scope_preserved": True}],
             "requirements": [{"requirement": "Identify the result", "addressed_by": ["claim-1"]}]},
        ]
        limited = Mock(status_code=429, headers={})
        limited.json.return_value = {"error": {"metadata": {"limit_source": "upstream_provider_shared_pool"}}}
        responses = [limited]
        for index, output in enumerate(outputs):
            upstream = Mock(status_code=200)
            is_generation = index == 1
            content = f'STATUS: {output["status"]}\n{output["text"]}' if is_generation else json.dumps(output)
            upstream.json.return_value = {"provider": "Google AI Studio" if is_generation else "Baidu",
                "choices": [{"finish_reason": "stop", "message": {"content": content}}]}
            responses.append(upstream)
        no_endpoint = Mock(status_code=404, headers={})
        no_endpoint.json.return_value = {"error": {"message": "No endpoints found matching your data policy."}}
        responses.insert(2, no_endpoint)
        with patch("app.providers.openrouter_answers.requests.post", side_effect=responses) as transport:
            reply = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "Alpha?"})
        self.assertEqual(reply.status_code, 200, reply.text)
        data = reply.json()
        self.assertEqual(data["verification_status"], "verified")
        self.assertEqual([source["page_number"] for source in data["sources"]], [1])
        self.assertEqual(search.queries, [])
        self.assertEqual([call.kwargs["json"]["model"] for call in transport.call_args_list],
                         [VERIFICATION_MODEL, FAST_MODEL, GENERATION_MODEL, PDF_FALLBACK_MODEL, FAST_MODEL])
        self.assertNotIn("Bob's alpha secret", repr(transport.call_args_list))
        saved = self.client.app.state.conversation_repository.items[data["conversation_id"]]["messages"][0]
        quality = saved["quality_evaluations"][0]
        self.assertTrue(quality["evaluator_version"])
        self.assertEqual([call["role"] for call in quality["model_calls"]],
                         ["evidence_assessment", "structured_output_fallback", "research_generation", "fast_generation",
                          "claim_verification", "structured_output_fallback"])
        self.assertTrue(all(call["provider"] == "openrouter" and call["latency_ms"] >= 0
                            for call in quality["model_calls"]))

    def test_both_upstream_pools_throttled_returns_retrieved_passages_for_retry(self):
        budget = Mock()
        self.client.app.state.answer_provider = OpenRouterAnswerProvider("test-key", ModelRouter(), budget=budget)
        limited = Mock(status_code=429, headers={})
        limited.json.return_value = {"error": {"metadata": {"limit_source": "upstream_provider_shared_pool"}}}
        with patch("app.providers.openrouter_answers.requests.post", return_value=limited) as post:
            response = self.client.post("/api/v1/research/ask",
                                        json={"paper_id": "paper-1", "question": "Alpha?", "limit": 1})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "60")
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "upstream_capacity")
        self.assertEqual(detail["limit_source"], "upstream_provider_shared_pool")
        self.assertEqual(detail["paper_id"], "paper-1")
        self.assertEqual([item["text"] for item in detail["passages"]], ["Alpha result"])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(self.client.app.state.conversation_repository.items, {})

    def test_local_daily_budget_blocks_assessment_without_provider_call(self):
        budget = Mock()
        budget.reserve.side_effect = AnswerRateLimitError("78827", limit_source="local_daily_budget")
        self.client.app.state.answer_provider = OpenRouterAnswerProvider("test-key", ModelRouter(), budget=budget)
        with patch("app.providers.openrouter_answers.requests.post") as post, \
                self.assertLogs("uvicorn.error", level="WARNING") as logs:
            response = self.client.post("/api/v1/research/ask",
                                        json={"paper_id": "paper-1", "question": "Alpha?", "limit": 1})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "78827")
        detail = response.json()["detail"]
        self.assertEqual(detail["limit_source"], "local_daily_budget")
        self.assertEqual(detail["paper_id"], "paper-1")
        self.assertEqual([item["page_number"] for item in detail["passages"]], [1])
        self.assertIn("stage=citewise_evidence_assessment", logs.output[0])
        self.assertIn("limit_source=local_daily_budget", logs.output[0])
        budget.reserve.assert_called_once()
        self.assertEqual(budget.reserve.call_args.args, (VERIFICATION_MODEL,))
        self.assertEqual(budget.reserve.call_args.kwargs["stage"], "citewise_evidence_assessment")
        budget.dispatch.assert_not_called()
        post.assert_not_called()

    def test_throttled_assessor_followed_by_unusable_fallback_keeps_local_passages(self):
        limited = Mock(status_code=429, headers={})
        limited.json.return_value = {"error": {"metadata": {"limit_source": "upstream_provider_shared_pool"}}}
        malformed = Mock(status_code=200)
        malformed.json.return_value = {"choices": [{"message": {"content": '{"level":"sufficient"}'}}]}
        for bad in (Timeout("private provider detail"), malformed):
            with self.subTest(failure=type(bad).__name__):
                self.client.app.state.answer_provider = OpenRouterAnswerProvider("test-key", ModelRouter())
                with patch("app.providers.openrouter_answers.requests.post", side_effect=[limited, bad]) as post, \
                        self.assertLogs("app.routers.research", level="WARNING") as logs:
                    response = self.client.post("/api/v1/research/ask",
                                                json={"paper_id": "paper-1", "question": "Alpha?", "limit": 1})
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.headers["cache-control"], "private, no-store")
                detail = response.json()["detail"]
                self.assertEqual(detail["code"], "provider_unavailable")
                self.assertEqual(detail["paper_id"], "paper-1")
                self.assertEqual([item["text"] for item in detail["passages"]], ["Alpha result"])
                self.assertNotIn("private provider detail", response.text + str(logs.output))
                self.assertIn(FAST_MODEL, str(logs.output))
                self.assertEqual(post.call_count, 2)
                self.assertEqual(self.client.app.state.conversation_repository.items, {})

    def test_unbacked_fallback_quote_abstains_with_no_second_alternate_request(self):
        self.client.app.state.answer_provider = OpenRouterAnswerProvider("test-key", ModelRouter())
        limited = Mock(status_code=429, headers={})
        limited.json.return_value = {"error": {"metadata": {"limit_source": "upstream_provider_shared_pool"}}}
        fallback = Mock(status_code=200)
        fallback.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "level": "sufficient", "chunk_ids": ["c0-hash-a"], "missing_information": "",
            "supporting_quotes": [{"chunk_id": "c0-hash-a", "quote": "An invented quotation"}],
        })}}]}
        with patch("app.providers.openrouter_answers.requests.post", side_effect=[limited, fallback]) as post:
            reply = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "Alpha?"})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(reply.json()["status"], "insufficient_evidence")
        self.assertEqual(reply.json()["sources"], [])
        self.assertEqual(post.call_count, 2)

    def test_identical_question_and_evidence_reuses_validated_classification(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10))
        self.client.app.state.answer_provider = provider
        first = EvidenceAssessment("sufficient", ["c0-hash-a"],
            supporting_quotes=[EvidenceQuote("c0-hash-a", "Alpha result")])
        with patch.object(provider, "assess", side_effect=[first, EvidenceAssessment("none", [])]) as assess, \
                patch.object(provider, "generate", return_value=GeneratedAnswer(
                    "Alpha result [c0-hash-a].", ["c0-hash-a"])), \
                patch.object(provider, "verify_claims", return_value=[ClaimAssessment(
                    "claim-1", "supported", [ClaimQuote("c0-hash-a", "Alpha result")], [])]):
            replies = [self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "Alpha?"}) for _ in range(2)]
        self.assertEqual(assess.call_count, 1)
        self.assertEqual([reply.json()["assessment"]["level"] for reply in replies],
                         ["sufficient", "sufficient"])
        self.assertTrue(all(reply.json()["search_trace"]["tavily_calls"] == 0 for reply in replies))

    def test_capabilities_preserve_same_key_cooldown_and_clear_changed_key(self):
        provider = OpenRouterAnswerProvider("first-test-key", ModelRouter(timeout=10))
        self.client.app.state.answer_provider = provider
        with patch.object(provider, "assess", side_effect=AnswerRateLimitError("90")):
            limited = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "Alpha?"})
        self.assertEqual(limited.status_code, 429)
        self.assertIn("HttpOnly", limited.headers["set-cookie"])
        self.assertNotIn("first-test-key", limited.headers["set-cookie"])
        initial = self.client.get("/api/v1/research/capabilities").json()
        self.assertGreater(initial["retry_after_seconds"], 0)
        self.assertNotIn("key_revision", initial)

        self.client.app.state.answer_provider = OpenRouterAnswerProvider("first-test-key", ModelRouter(timeout=10))
        restarted = self.client.get("/api/v1/research/capabilities").json()
        self.assertGreater(restarted["retry_after_seconds"], 0)
        self.client.app.state.answer_provider = OpenRouterAnswerProvider("second-test-key", ModelRouter(timeout=10))
        changed = self.client.get("/api/v1/research/capabilities").json()
        self.assertEqual(changed["retry_after_seconds"], 0)
        self.assertNotIn("key_revision", changed)
        self.assertNotIn("second-test-key", str(changed))

        self.client.app.state.answer_provider = provider
        with patch.object(provider, "assess", return_value=EvidenceAssessment(
                "sufficient", ["c0-hash-a"], supporting_quotes=[
                    EvidenceQuote("c0-hash-a", "Alpha result")])), \
                patch.object(provider, "generate", return_value=GeneratedAnswer(
                    "Alpha result [c0-hash-a].", ["c0-hash-a"])), \
                patch.object(provider, "verify_claims", return_value=[ClaimAssessment(
                    "claim-1", "supported", [ClaimQuote("c0-hash-a", "Alpha result")], [])]):
            answered = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "Alpha?"})
        self.assertEqual(answered.status_code, 200)
        self.assertEqual(self.client.get("/api/v1/research/capabilities").json()["retry_after_seconds"], 0)

    def test_rejects_missing_or_unready_paper_and_blank_question(self):
        for paper_id, status in (("foreign", 404), ("pending", 409)):
            response = self.client.post("/api/v1/research/search", json={
                "paper_id": paper_id, "question": "Alpha?",
            })
            self.assertEqual(response.status_code, status)
        response = self.client.post("/api/v1/research/search", json={
            "paper_id": "paper-1", "question": "   ",
        })
        self.assertEqual(response.status_code, 422)

    def test_ask_validates_citations_and_keeps_owner_filter(self):
        provider = Answers(GeneratedAnswer("Alpha is reported.", ["c0-hash-a"]))
        self.client.app.state.answer_provider = provider
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Alpha?", "limit": 2,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "answered")
        self.assertEqual(response.json()["sources"][0]["page_number"], 1)
        self.assertEqual(response.json()["sources"][0]["document_name"], "paper.pdf")
        self.assertEqual(response.json()["sources"][0]["highlight_text"], "Alpha result")
        self.assertEqual([item.id for item in provider.received[1]], ["c0-hash-a", "c1-hash-b"])
        self.assertTrue(all("secret" not in item.text for item in provider.received[1]))

        provider.result = GeneratedAnswer("Made up", ["e999"])
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Alpha?",
        })
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertIn("could not be linked", response.json()["answer"])

        provider.result = GeneratedAnswer("Alpha?", ["c0-hash-a"])
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Alpha?",
        })
        self.assertEqual(response.json()["status"], "insufficient_evidence")

    def test_ask_preserves_distinct_evidence_outcomes_and_citations(self):
        provider = Answers(GeneratedAnswer("", [], "not_mentioned"))
        self.client.app.state.answer_provider = provider
        for status, text, ids in (
            ("not_mentioned", "", []),
            ("partially_supported", "The paper supports only the first part.", ["c0-hash-a"]),
            ("insufficient_evidence", "The topic appears, but the result is inconclusive.", ["c0-hash-a"]),
            ("contradictory_evidence", "The passages report different findings.", ["c0-hash-a", "c1-hash-b"]),
        ):
            provider.result = GeneratedAnswer(text, ids, status)
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "What does the paper conclude?", "limit": 2,
            })
            self.assertEqual(response.status_code, 200)
            expected = "insufficient_evidence" if status == "contradictory_evidence" else status
            self.assertEqual(response.json()["status"], expected)
            self.assertEqual(len(response.json()["sources"]), 0 if status == "contradictory_evidence" else len(ids))
            self.assertTrue(response.json()["answer"])

    def test_assessment_controls_external_search_and_is_saved(self):
        search = ExternalSearch()
        self.client.app.state.external_search = search
        provider = AssessedAnswers(EvidenceAssessment("sufficient", ["c0-hash-a"],
            supporting_quotes=[EvidenceQuote("c0-hash-a", "Alpha result")]))
        self.client.app.state.answer_provider = provider
        payload = {"paper_id": "paper-1", "question": "Alpha?"}
        sufficient = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(sufficient.status_code, 200)
        self.assertEqual(sufficient.json()["assessment"]["level"], "sufficient")
        self.assertEqual(search.queries, [])
        self.assertEqual(sufficient.json()["search_trace"]["tavily_calls"], 0)

        provider.assessment = EvidenceAssessment("partial", ["c0-hash-a"], "Beta details",
            [EvidenceQuote("c0-hash-a", "Alpha result")])
        document_partial = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(document_partial.status_code, 200)
        self.assertEqual(document_partial.json()["status"], "partially_supported")
        self.assertEqual(document_partial.json()["search_trace"]["tavily_calls"], 1)
        self.assertEqual(len(search.queries), 1)
        partial = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Compare Alpha with external research."})
        self.assertEqual(partial.status_code, 200)
        self.assertEqual(partial.json()["assessment"]["chunk_ids"], ["c0-hash-a"])
        self.assertEqual(partial.json()["external_sources"][0]["url"], "https://example.org/study")
        self.assertIn("Beta details", search.queries[-1])
        self.assertTrue(search.queries[-1].startswith("Beta details"))
        self.assertEqual(partial.json()["search_trace"], {
            "tavily_calls": 1, "tavily_cache_hits": 0, "queries": [search.queries[-1]], "failures": 0,
            "duplicate_requests": 0, "retrieved_source_counts": [1], "rejected_irrelevant_sources": 0,
            "claim_status_counts": {"supported": 2}, "verification_failures": 0,
            "openrouter_calls": 0, "openrouter_cache_hits": 0, "openrouter_attempts": 0, "openrouter_retries": 0, "openrouter_failures": 0,
            "openrouter_rate_limits": 0, "openrouter_server_errors": 0, "openrouter_network_errors": 0,
            "openrouter_rate_limit_kind": None, "raw_external_results": 0, "invalid_external_results": 0,
            "invalid_external_urls": 0, "rejected_external_domains": 0,
            "duplicate_external_sources": 0, "accepted_external_sources": 1,
            "fallback_searches": 0, "external_search_errors": 0, "external_search_timeouts": 0})
        conversation = self.client.get(f"/api/v1/research/conversations/{partial.json()['conversation_id']}")
        self.assertEqual(conversation.json()["messages"][0]["external_sources"][0]["id"], "ext1")

        provider.assessment = EvidenceAssessment("partial", ["unknown"], "Beta details")
        count = len(search.queries)
        invalid = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(invalid.status_code, 200)
        self.assertEqual(invalid.json()["status"], "insufficient_evidence")
        self.assertEqual(len(search.queries), count)

        provider.assessment = EvidenceAssessment("partial", ["c0-hash-a"], "Beta details",
            [EvidenceQuote("c0-hash-a", "Invented result")])
        invalid_quote = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(invalid_quote.status_code, 200)
        self.assertEqual(invalid_quote.json()["status"], "insufficient_evidence")
        self.assertEqual(len(search.queries), count)

        provider.assessment = EvidenceAssessment("none", [])
        none = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "India biometric law"})
        self.assertEqual(none.status_code, 200)
        self.assertEqual(none.json()["assessment"]["level"], "none")
        self.assertEqual(none.json()["sources"], [])
        self.assertEqual(none.json()["external_sources"], [])
        self.assertEqual(none.json()["status"], "insufficient_evidence")
        self.assertEqual(provider.external_calls, 3)
        self.assertIn("external sources", none.json()["answer"])
        self.assertEqual(none.json()["search_trace"]["tavily_calls"], 1)

        specific = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1",
            "question": "What was the p-value of the study?"})
        self.assertEqual(specific.status_code, 200)
        self.assertIn("do not establish the requested statistic", specific.json()["answer"])
        self.assertEqual(len(search.queries), count + 1)

    def test_tavily_trace_distinguishes_fresh_request_from_cache_hit(self):
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment("none", []))
        self.client.app.state.external_search = TavilyExternalSearch("test-key")
        response = Mock(status_code=200)
        response.json.return_value = {"results": [
            {"title": "Library opening hours", "url": "https://example.edu/library",
             "content": "Books and lending schedules."},
            {"title": "Webcam privacy research", "url": "https://example.edu/research",
             "content": "Webcam privacy result."}]}
        payload = {"paper_id": "paper-1", "question": "external research on webcam privacy effects"}
        with patch("app.providers.external_search.requests.post", return_value=response) as post:
            fresh = self.client.post("/api/v1/research/ask", json=payload)
            cached = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(fresh.status_code, 200)
        self.assertEqual(cached.status_code, 200)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(fresh.json()["search_trace"]["tavily_calls"], 1)
        self.assertEqual(fresh.json()["search_trace"]["tavily_cache_hits"], 0)
        self.assertEqual(fresh.json()["search_trace"]["rejected_irrelevant_sources"], 1)
        self.assertEqual(cached.json()["search_trace"]["tavily_calls"], 0)
        self.assertEqual(cached.json()["search_trace"]["tavily_cache_hits"], 1)
        self.assertEqual(cached.json()["search_trace"]["rejected_irrelevant_sources"], 1)

    def test_empty_initial_search_uses_one_focused_fallback(self):
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment("none", []))
        self.client.app.state.external_search = TavilyExternalSearch("test-key")
        empty = Mock(status_code=200)
        empty.json.return_value = {"results": []}
        found = Mock(status_code=200)
        found.json.return_value = {"results": [{"title": "Webcam privacy research",
            "url": "https://example.edu/privacy", "content": "External result about webcam privacy."}]}
        with patch("app.providers.external_search.requests.post", side_effect=[empty, found]) as post:
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "What does external research say about webcam privacy?"})
        self.assertEqual(response.status_code, 200)
        trace = response.json()["search_trace"]
        self.assertEqual(post.call_count, 2)
        self.assertEqual(trace["fallback_searches"], 1)
        self.assertEqual(trace["retrieved_source_counts"], [0, 1])
        self.assertEqual(trace["accepted_external_sources"], 1)

    def test_empty_external_search_abstains_and_preserves_the_question(self):
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment("none", []))
        self.client.app.state.external_search = TavilyExternalSearch("test-key")
        empty = Mock(status_code=200)
        empty.json.return_value = {"results": []}
        question = "What happened after the intervention?"
        with patch("app.providers.external_search.requests.post", return_value=empty) as post:
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": question})
        self.assertEqual(response.status_code, 200)
        self.assertIn("no usable external evidence", response.json()["answer"])
        self.assertEqual(response.json()["external_sources"], [])
        self.assertLessEqual(post.call_count, 2)
        saved = self.client.get(f"/api/v1/research/conversations/{response.json()['conversation_id']}")
        self.assertEqual(saved.json()["messages"][0]["question"], question)

    def test_disagreeing_external_sources_remain_separate_and_cited(self):
        class DisagreeingSearch(ExternalSearch):
            def search(self, query):
                self.queries.append(query)
                return [ExternalEvidence("ext1", "Effect change report A", "https://one.example.org/result",
                                         "The effect increased by 10 units."),
                        ExternalEvidence("ext2", "Effect change report B", "https://two.example.org/result",
                                         "The effect decreased by 5 units.")]

        class DisagreeingAnswers(AssessedAnswers):
            def generate_with_external(self, question, pdf_evidence, external_evidence, missing_information):
                return ExternalAnswer("No PDF evidence.",
                    "One source reports the effect increased by 10 units [ext1]. "
                    "Another reports the effect decreased by 5 units [ext2].", [], ["ext1", "ext2"])

            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(item["id"], "supported", [ClaimQuote(
                    "ext1" if "increased" in item["text"] else "ext2",
                    "The effect increased by 10 units." if "increased" in item["text"]
                    else "The effect decreased by 5 units.")], []) for item in claims]

        self.client.app.state.answer_provider = DisagreeingAnswers(EvidenceAssessment("none", []))
        self.client.app.state.external_search = DisagreeingSearch()
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "How did the effect change?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("increased", response.json()["answer"])
        self.assertIn("decreased", response.json()["answer"])
        self.assertEqual({item["url"] for item in response.json()["external_sources"]},
                         {"https://one.example.org/result", "https://two.example.org/result"})

    def test_external_verification_retry_reuses_search_and_generation(self):
        class RateLimitedOnce(AssessedAnswers):
            verification_calls = 0

            def verify_claims(self, question, claims, evidence):
                self.verification_calls += 1
                if self.verification_calls == 1:
                    raise AnswerRateLimitError("1")
                return super().verify_claims(question, claims, evidence)

        provider = RateLimitedOnce(EvidenceAssessment("none", []))
        search = ExternalSearch()
        self.client.app.state.answer_provider = provider
        self.client.app.state.external_search = search
        first = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "What does external research report?"})
        self.assertEqual(first.status_code, 200)
        draft = first.json()
        self.assertEqual(draft["verification_status"], "unavailable")
        self.assertEqual(len(search.queries), 1)
        self.assertEqual(provider.external_calls, 1)
        retried = self.client.post(f"/api/v1/research/conversations/{draft['conversation_id']}"
                                   f"/messages/{draft['message_id']}/verify")
        self.assertEqual(retried.status_code, 200)
        self.assertEqual(retried.json()["verification_status"], "verified")
        self.assertEqual(retried.json()["answer_state"], "VERIFIED")
        self.assertEqual(len(search.queries), 1)
        self.assertEqual(provider.external_calls, 1)
        saved = self.client.get(f"/api/v1/research/conversations/{draft['conversation_id']}")
        self.assertEqual(len(saved.json()["messages"]), 1)
        self.assertEqual(saved.json()["messages"][0]["external_sources"][0]["id"], "ext1")
        internal = self.client.app.state.conversation_repository.items[draft["conversation_id"]]["messages"][0]
        self.assertEqual(next(item for item in internal["verification_checks"] if item["check_id"] == "AH3")["status"], "PASS")
        self.assertEqual([item["revision"] for item in internal["quality_evaluations"]], [1, 2])
        self.assertEqual(internal["quality_evaluations"][0]["state"], "UNAVAILABLE")
        self.assertNotIn("quality_evaluations", saved.json()["messages"][0])

    def test_web_background_is_withheld_and_promotion_remains_attributed_after_reload(self):
        question = "Do independent external sources confirm lower winter energy use?"

        class WebAnswers(AssessedAnswers):
            def generate_with_external(self, question, pdf_evidence, external_evidence, missing_information):
                self.external_calls += 1
                return ExternalAnswer("", "Winter energy use varies with temperature [ext1]. "
                                      "The heater cuts winter energy use [ext2].", [], ["ext1", "ext2"])

            def verify_claims(self, question, claims, evidence):
                by_id = {source.id: source for source in evidence}
                self.source_context = by_id["ext2"].provenance
                return [ClaimAssessment(item["id"], "supported",
                    [ClaimQuote(source.id, source.text)], [],
                    question_fit="irrelevant" if source.id == "ext1" else "direct",
                    evidence_use="attribution_only" if source.id == "ext2" else "establishes_claim")
                    for item in claims if (source := by_id[item["cited_ids"][0]])]

        provider = WebAnswers(EvidenceAssessment("none", []))
        self.client.app.state.answer_provider = provider
        self.client.app.state.external_search = TavilyExternalSearch("test-key")
        search = Mock()
        search.json.return_value = {"results": [
            {"title": "Winter energy use", "url": "https://energy.example.org/study",
             "content": "Winter energy use varies with temperature."},
            {"title": "Winter energy", "url": "https://manufacturer.example.com/announcement",
             "content": "Our heater cuts winter energy use.", "publisher": "Heater manufacturer"},
        ]}
        with patch("app.providers.external_search.requests.post", return_value=search) as post:
            result = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": question})
        self.assertEqual(result.status_code, 200, result.text)
        answer = result.json()
        self.assertEqual(answer["status"], "partially_supported")
        self.assertEqual(answer["answer_state"], "PARTIALLY_SUPPORTED")
        self.assertEqual(answer["verification_status"], "not_verified")
        self.assertNotIn("varies with temperature", answer["answer"])
        self.assertIn("Independent confirmation was not established", answer["answer"])
        self.assertEqual([item["id"] for item in answer["external_sources"]], ["ext2"])
        self.assertEqual(provider.source_context["publisher"], "Heater manufacturer")
        self.assertEqual(post.call_count, 1)
        saved = self.client.get(f"/api/v1/research/conversations/{answer['conversation_id']}").json()["messages"][0]
        for field in ("answer", "claims", "sources", "external_sources"):
            self.assertEqual(saved[field], answer[field])

    def test_external_verifier_error_returns_unverified_draft(self):
        class UnavailableVerifier(AssessedAnswers):
            def verify_claims(self, question, claims, evidence):
                raise TimeoutError("verification provider unavailable")

        self.client.app.state.answer_provider = UnavailableVerifier(EvidenceAssessment("none", []))
        self.client.app.state.external_search = ExternalSearch()
        result = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "What does external coastal research report?"})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["verification_status"], "unavailable")
        self.assertEqual(result.json()["answer_state"], "VERIFICATION_UNAVAILABLE")
        self.assertEqual(result.json()["claims"], [])

    def test_provider_failure_does_not_trigger_fallback_or_claim_absence(self):
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment("none", []))
        self.client.app.state.external_search = TavilyExternalSearch("test-key")
        with patch("app.providers.external_search.requests.post", side_effect=TimeoutError) as post:
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "What does external research say about webcam privacy?"})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(response.json()["search_trace"]["external_search_timeouts"], 1)
        self.assertEqual(response.json()["search_trace"]["fallback_searches"], 0)
        self.assertIn("External search timed out", response.json()["answer"])

    def test_external_http_failures_are_distinguished_from_missing_evidence(self):
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment("none", []))
        self.client.app.state.external_search = TavilyExternalSearch("test-key")
        for code, expected in ((401, "authentication"), (403, "authentication"),
                               (429, "rate limited"), (432, "quota"), (433, "quota"),
                               (503, "provider is unavailable")):
            with self.subTest(status=code):
                provider_response = Mock(status_code=code)
                provider_response.raise_for_status.side_effect = HTTPError(response=provider_response)
                with patch("app.providers.external_search.requests.post", return_value=provider_response) as post:
                    response = self.client.post("/api/v1/research/ask", json={
                        "paper_id": "paper-1", "question": "What evidence supports this answer?"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(post.call_count, 1)
                self.assertIn(expected, response.json()["answer"])
                self.assertEqual(response.json()["external_sources"], [])
                self.assertEqual(response.json()["search_trace"]["external_search_errors"], 1)

    def test_sample_size_does_not_bypass_wrong_none_assessment(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        vectors.save("alice", "paper-1", [TextChunk(0, 1, "The study included 412 students.", "sample")],
                     [[0.0, 1.0]])
        self.client.app.state.ingestion_service = Ingestion(vectors)
        search = ExternalSearch()
        self.client.app.state.external_search = search
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment("none", []))
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "What was the sample size of the study?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assessment"]["level"], "none")
        self.assertNotIn("412 students", response.json()["answer"])
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertEqual(search.queries, [])

    def test_sample_size_shortcut_does_not_omit_other_requested_facts(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        text = "Study A included 412 students and used a randomized design over one semester."
        vectors.save("alice", "paper-1", [TextChunk(0, 1, text, "hash-a")], [[1.0, 0.0]])
        self.client.app.state.ingestion_service = Ingestion(vectors)
        provider = AssessedAnswers(EvidenceAssessment("sufficient", ["c0-hash-a"],
            supporting_quotes=[EvidenceQuote("c0-hash-a", text)]))
        provider.result = GeneratedAnswer("Study A included 412 students and used a randomized design "
            "over one semester. [c0-hash-a]", ["c0-hash-a"])
        self.client.app.state.answer_provider = provider
        response = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1",
            "question": "What was Study A's sample size and research design?"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(provider.received)
        self.assertIn("randomized design", response.json()["answer"])

    def test_unrelated_official_biometric_device_page_is_not_legal_evidence(self):
        class DeviceSearch(ExternalSearch):
            def search(self, query):
                self.queries.append(query)
                return [ExternalEvidence("ext1", "Certified biometric devices",
                    "https://uidai.gov.in/devices", "Specifications for fingerprint scanners")]

        search = DeviceSearch()
        provider = AssessedAnswers(EvidenceAssessment("none", []))
        self.client.app.state.external_search = search
        self.client.app.state.answer_provider = provider
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "India biometric law"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["external_sources"], [])
        self.assertEqual(provider.external_calls, 0)
        self.assertEqual(len(search.queries), 1)

    def test_external_search_failure_is_recorded_once(self):
        class FailingSearch(ExternalSearch):
            def search(self, query):
                self.queries.append(query)
                raise RuntimeError("search unavailable")

        search = FailingSearch()
        self.client.app.state.external_search = search
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment("none", []))
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "What do other studies say?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["search_trace"]["tavily_calls"], 1)
        self.assertEqual(response.json()["search_trace"]["failures"], 1)
        self.assertEqual(response.json()["search_trace"]["retrieved_source_counts"], [0])
        self.assertEqual(len(search.queries), 1)

    def test_assessment_accepts_ordered_verbatim_quote_fragments(self):
        text = ("Reported misconduct was 11 percent among monitored students. "
                "The study cannot establish that surveillance caused the difference.")
        source = AnswerSource(id="c0", chunk_index=0, page_number=1, text=text,
                              content_hash="hash", highlight_text=text)
        assessment = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[EvidenceQuote(
            "c0", "Reported misconduct was 11 percent among monitored students. ... "
                  "The study cannot establish that surveillance caused the difference.")])
        self.assertIsNotNone(_validate_assessment(assessment, "surveillance misconduct", {"c0": source}))

    def test_assessment_accepts_multiple_quotes_from_each_selected_chunk(self):
        first = AnswerSource(id="c0", chunk_index=0, page_number=2,
            text="Monitoring reduced misconduct. The study lasted one semester.",
            content_hash="a", highlight_text="")
        second = AnswerSource(id="c1", chunk_index=1, page_number=3,
            text="Students reported privacy concerns.", content_hash="b", highlight_text="")
        review = _validate_assessment(EvidenceAssessment("sufficient", ["c0", "c1"],
            supporting_quotes=[EvidenceQuote("c0", "Monitoring reduced misconduct"),
                               EvidenceQuote("c0", "The study lasted one semester"),
                               EvidenceQuote("c1", "Students reported privacy concerns")]),
            "How did monitoring affect misconduct and privacy?", {"c0": first, "c1": second})
        self.assertIsNotNone(review)
        self.assertEqual(review.chunk_ids, ["c0", "c1"])
        self.assertIsNone(_validate_assessment(EvidenceAssessment("sufficient", ["c0"]),
            "How did monitoring affect misconduct?", {"c0": first}))

    def test_assessment_preserves_numbers_and_order_during_quote_normalization(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=2,
            text="The rate was 12.5%—not 18.5%. Privacy concerns were also recorded.",
            content_hash="a", highlight_text="")
        accepted = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", "The rate was 12.5 % - not 18.5 %")])
        self.assertIsNotNone(_validate_assessment(accepted, "What was the rate?", {"c0": source}))
        invented = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", "The rate was 13.5 % - not 18.5 %")])
        self.assertIsNone(_validate_assessment(invented, "What was the rate?", {"c0": source}))
        reversed_quote = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", "Privacy concerns ... The rate was 12.5%")])
        self.assertIsNone(_validate_assessment(reversed_quote, "What was the rate?", {"c0": source}))

    def test_assessment_does_not_erase_a_negative_sign(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=1,
            text="The temperature changed by −5 degrees during the experiment.",
            content_hash="a", highlight_text="")
        good = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", "The temperature changed by -5 degrees during the experiment")])
        wrong = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", "The temperature changed by 5 degrees during the experiment")])
        self.assertIsNotNone(_validate_assessment(good, "How did the temperature change?", {"c0": source}))
        self.assertIsNone(_validate_assessment(wrong, "How did the temperature change?", {"c0": source}))

    def test_assessment_rejects_malformed_quote_without_crashing(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=2,
            text="Monitoring reduced misconduct among students.",
            content_hash="a", highlight_text="")
        malformed = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", None)])
        unknown = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("missing", "Monitoring reduced misconduct among students")])
        self.assertIsNone(_validate_assessment(malformed, "Did monitoring reduce misconduct?", {"c0": source}))
        self.assertIsNone(_validate_assessment(unknown, "Did monitoring reduce misconduct?", {"c0": source}))

    def test_assessment_rejection_logs_only_structured_metadata(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=2,
            text="Private passage: monitoring reduced misconduct among students.",
            content_hash="a", highlight_text="")
        cases = [
            (EvidenceAssessment("invalid", ["c0"]), "invalid_assessment_level", 1, 0),
            (EvidenceAssessment("sufficient", ["unknown"], supporting_quotes=[
                EvidenceQuote("unknown", "Private invented quotation")]), "unknown_source_id", 1, 1),
            (EvidenceAssessment("sufficient", ["c0"]), "missing_supporting_quote", 1, 0),
            (EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
                EvidenceQuote("c0", "Private invented quotation")]),
                "quote_not_found_or_order_mismatch", 1, 1),
        ]
        for assessment, reason, cited_count, quote_count in cases:
            with self.subTest(reason=reason), self.assertLogs("uvicorn.error", level="INFO") as logs:
                self.assertIsNone(_validate_assessment(assessment, "Private question", {"c0": source},
                                                        request_id="req-123"))
            self.assertEqual(len(logs.output), 1)
            entry = logs.output[0]
            for expected in ("request_id=req-123", f"reason={reason}", "retrieved_sources=1",
                             f"cited_source_ids={cited_count}", f"supporting_quotes={quote_count}"):
                self.assertIn(expected, entry)
            self.assertNotIn("Private passage", entry)
            self.assertNotIn("Private question", entry)
            self.assertNotIn("Private invented quotation", entry)

    def test_assessment_rejection_correlates_with_ask_request(self):
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment(
            "sufficient", ["unknown"], supporting_quotes=[
                EvidenceQuote("unknown", "Private invented quotation")]))
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "Private question", "request_id": "req-456"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertEqual(response.json()["sources"], [])
        rejected = [entry for entry in logs.output if "evidence_assessment_rejected" in entry]
        self.assertEqual(len(rejected), 1)
        self.assertIn("request_id=req-456 reason=unknown_source_id level=sufficient", rejected[0])
        self.assertIn("retrieved_sources=2 cited_source_ids=1 supporting_quotes=1", rejected[0])
        self.assertNotIn("Private question", "\n".join(logs.output))
        self.assertNotIn("Private invented quotation", "\n".join(logs.output))

    def test_missing_assessment_quote_returns_safe_abstention_without_external_search(self):
        provider = AssessedAnswers(EvidenceAssessment("sufficient", ["c0-hash-a"],
            supporting_quotes=[EvidenceQuote("c0-hash-a", "An invented attendance requirement of 85 percent")]))
        external = ExternalSearch()
        self.client.app.state.answer_provider = provider
        self.client.app.state.external_search = external
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "What is the attendance requirement?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertEqual(response.json()["sources"], [])
        self.assertNotIn("85", response.json()["answer"])
        self.assertEqual(external.queries, [])
        self.assertTrue(any("reason=quote_not_found_or_order_mismatch" in line for line in logs.output))
        self.assertTrue(any("fragment_missing" in line for line in logs.output))

    def test_assessment_recovery_uses_validated_original_passage_once(self):
        class Recovered(AssessedAnswers):
            calls = 0
            def reassess(self, question, evidence):
                self.calls += 1
                return EvidenceAssessment("sufficient", ["c0-hash-a"],
                    supporting_quotes=[EvidenceQuote("c0-hash-a", "Alpha result")])
        provider = Recovered(EvidenceAssessment("sufficient", ["c0-hash-a"],
            supporting_quotes=[EvidenceQuote("c0-hash-a", "An invented alpha result")]))
        self.client.app.state.answer_provider = provider
        response = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "What is alpha?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "answered")
        self.assertEqual(provider.calls, 1)

    def test_assessment_recovery_rejects_a_second_invented_quote(self):
        class Recovered(AssessedAnswers):
            calls = 0
            def reassess(self, question, evidence):
                self.calls += 1
                return EvidenceAssessment("sufficient", ["c0-hash-a"],
                    supporting_quotes=[EvidenceQuote("c0-hash-a", "Another invented result")])
        provider = Recovered(EvidenceAssessment("sufficient", ["c0-hash-a"],
            supporting_quotes=[EvidenceQuote("c0-hash-a", "An invented alpha result")]))
        self.client.app.state.answer_provider = provider
        response = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "What alpha result?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertEqual(response.json()["sources"], [])
        self.assertEqual(provider.calls, 1)

    def test_assessment_processing_bug_is_not_hidden_as_abstention(self):
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment(
            "sufficient", ["c0-hash-a"], supporting_quotes=[EvidenceQuote("c0-hash-a", "Alpha result")]))
        with patch("app.routers.research._validate_assessment", side_effect=RuntimeError("private text")):
            response = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "Alpha?"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private text", response.text)

    def test_claim_review_logs_statuses_reasons_and_source_ids_without_content(self):
        first = AnswerSource(id="c0", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text="Private source passage. Alpha result was observed. "
                                    "Privacy concern was reported. Monitoring increased integrity.")
        second = AnswerSource(id="c1", chunk_index=1, page_number=2, content_hash="b",
            highlight_text="", text="Monitoring decreased integrity.")
        class Verifier:
            def verify_claims(self, question, claims, evidence):
                return [
                    ClaimAssessment(claims[0]["id"], "supported", [
                        ClaimQuote("c0", "Alpha result was observed")], []),
                    ClaimAssessment(claims[1]["id"], "partially_supported", [
                        ClaimQuote("c0", "Privacy concern was reported")], [],
                        "Private provider explanation"),
                    ClaimAssessment(claims[2]["id"], "unsupported", [], []),
                    ClaimAssessment(claims[3]["id"], "contradicted", [
                        ClaimQuote("c0", "Monitoring increased integrity")], [
                        ClaimQuote("c1", "Monitoring decreased integrity")]),
                ]
        answer = ("Alpha result was observed [c0]. Privacy concern was reported [c0]. "
                  "Unmeasured impact was established [c0]. Monitoring changed integrity [c0].")
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            _, _, reviews = asyncio.run(_verified_sections(Verifier(), "Private question", answer,
                "", ["c0", "c1"], [], {"c0": first, "c1": second}, {}, request_id="req-789"))
        self.assertEqual([item.status for item in reviews],
                         ["supported", "partially_supported", "unsupported", "contradicted"])
        entries = [entry for entry in logs.output if "research_claim_review" in entry]
        self.assertEqual(len(entries), 4)
        for index, (entry, status, reason) in enumerate(zip(entries,
                ["supported", "partially_supported", "unsupported", "contradicted"],
                ["accepted", "provider_partial", "provider_unsupported", "verified_contradiction"]), start=1):
            self.assertIn(f"request_id=req-789 claim_index={index} status={status}", entry)
            self.assertIn(f"reason={reason}", entry)
        self.assertIn("supporting_source_ids=['c0'] opposing_source_ids=['c1']", entries[3])
        self.assertIn("supporting_source_ids=[] opposing_source_ids=[]", entries[2])
        for secret in ("Private question", "Private source passage", "Private provider explanation",
                       "Unmeasured impact was established", "Monitoring increased integrity"):
            self.assertNotIn(secret, "\n".join(entries))

    def test_claim_review_logs_validation_rejection_code(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text="The rate was 12 percent.")
        class Verifier:
            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(claims[0]["id"], "supported", [
                    ClaimQuote("c0", "The rate was 12 percent")], [])]
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            _, _, reviews = asyncio.run(_verified_sections(Verifier(), "What was the rate?",
                "The rate was 13 percent [c0].", "", ["c0"], [], {"c0": source}, {},
                request_id="req-790"))
        self.assertEqual(reviews[0].status, "unsupported")
        self.assertIn("status=unsupported", logs.output[0])
        self.assertIn("reason=numeric_mismatch", logs.output[0])
        self.assertNotIn("13 percent", logs.output[0])

    def test_local_debug_maps_claim_index_to_generated_text(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text="Private PDF passage. The study enrolled 120 students. "
                                    "The paper mentions an outcome measure.")
        answer = ("The study enrolled 120 students [c0]. "
                  "The study included an outcome measure [c0]. "
                  "The study may describe the outcome incompletely [c0]. "
                  "The study had no unreported limitations [c0].")
        class Verifier:
            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(claim["id"],
                    "partially_supported" if index == 2 else "supported",
                    [ClaimQuote("c0", "The paper mentions an outcome measure")], [],
                    "The outcome is mentioned but its result is not established." if index == 2 else "")
                    for index, claim in enumerate(claims)]
        expected = split_claims(answer, "")[2].text
        with self.assertLogs("uvicorn.error", level="DEBUG") as logs:
            _, _, reviews = asyncio.run(_verified_sections(Verifier(), "What did the paper report?",
                answer, "", ["c0"], [], {"c0": source}, {}, request_id="req-debug",
                debug_claims=True))
        debug_entries = [entry for entry in logs.output if "research_claim_debug" in entry]
        self.assertEqual(len(debug_entries), 4)
        self.assertIn("claim_index=3", debug_entries[2])
        self.assertIn(f"claim_text={expected!r}", debug_entries[2])
        self.assertIn("status=partially_supported", debug_entries[2])
        self.assertIn("supporting_source_ids=['c0'] opposing_source_ids=[]", debug_entries[2])
        self.assertIn("provider_explanation='The outcome is mentioned but its result is not established.'",
                      debug_entries[2])
        self.assertIn("reason=provider_partial", debug_entries[2])
        self.assertNotIn("Private PDF passage", "\n".join(debug_entries))
        self.assertEqual(len(reviews), 4)
        with self.assertLogs("uvicorn.error", level="INFO") as normal_logs:
            asyncio.run(_verified_sections(Verifier(), "What did the paper report?",
                answer, "", ["c0"], [], {"c0": source}, {}, request_id="req-normal"))
        self.assertFalse(any("research_claim_debug" in entry for entry in normal_logs.output))

    def test_local_claim_debug_requires_loopback_and_debug_level(self):
        loopback = Request({"type": "http", "client": ("127.0.0.1", 1234), "app": self.client.app})
        remote = Request({"type": "http", "client": ("203.0.113.9", 1234), "app": self.client.app})
        with self.assertLogs("uvicorn.error", level="DEBUG"):
            self.assertTrue(_local_claim_debug(loopback))
            self.assertFalse(_local_claim_debug(remote))
            logging.getLogger("uvicorn.error").debug("local_debug_gate_test")

    def test_comparison_keeps_two_valid_study_passages_when_table_quote_reorders_text(self):
        sources = {
            "cA": AnswerSource(id="cA", chunk_index=1, page_number=3, content_hash="a",
                highlight_text="", text="Study A participants: 2,400 students. Design: Cluster-randomized comparison."),
            "cB": AnswerSource(id="cB", chunk_index=2, page_number=4, content_hash="b",
                highlight_text="", text="Study B participants: 780 students. Design: Matched-course comparison."),
            "cTable": AnswerSource(id="cTable", chunk_index=3, page_number=5, content_hash="t",
                highlight_text="", text="Study A Study B Design Cluster-randomized Matched-course comparison."),
        }
        question = "Compare the methodologies of Study A and Study B."
        assessment = EvidenceAssessment("sufficient", list(sources), supporting_quotes=[
            EvidenceQuote("cA", "Study A participants: 2,400 students"),
            EvidenceQuote("cB", "Study B participants: 780 students"),
            EvidenceQuote("cTable", "Design Cluster-randomized ... Study B ... Matched-course comparison")])
        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            result = _validate_assessment(assessment, question, sources)
        self.assertIsNone(result)
        self.assertTrue(any("reason=quote_not_found_or_order_mismatch" in line
                            for line in captured.output))
        missing_b = EvidenceAssessment("sufficient", ["cA", "cTable"], supporting_quotes=[
            EvidenceQuote("cA", "Study A participants: 2,400 students"),
            EvidenceQuote("cTable", "Design Cluster-randomized ... Study B ... Matched-course comparison")])
        self.assertIsNone(_validate_assessment(missing_b, question, sources))
        unrelated = sources | {"cOther": AnswerSource(id="cOther", chunk_index=4, page_number=6,
            content_hash="o", highlight_text="", text="Study B discusses an unrelated legal framework.")}
        unrelated_assessment = EvidenceAssessment("sufficient", ["cA", "cOther", "cTable"], supporting_quotes=[
            EvidenceQuote("cA", "Study A participants: 2,400 students"),
            EvidenceQuote("cOther", "Study B discusses an unrelated legal framework"),
            EvidenceQuote("cTable", "Design Cluster-randomized ... Study B ... Matched-course comparison")])
        self.assertIsNone(_validate_assessment(unrelated_assessment, question, unrelated))

    def test_labeled_methodology_fields_may_be_quoted_in_different_order(self):
        source = AnswerSource(id="cA", chunk_index=1, page_number=3, content_hash="a",
            highlight_text="", text="Study A Setting: 12 universities in three regions "
            "Participants: 2,400 undergraduate students Design: Cluster-randomized comparison "
            "Observation period: One academic semester")
        question = "Compare the methodologies used in Study A and Study B"
        reordered = EvidenceAssessment("sufficient", ["cA"], supporting_quotes=[EvidenceQuote(
            "cA", "Participants: 2,400 undergraduate students Design: Cluster-randomized comparison "
                  "Setting: 12 universities in three regions")])
        partial = _validate_assessment(reordered, question, {"cA": source})
        self.assertIsNone(partial)
        fabricated = EvidenceAssessment("sufficient", ["cA"], supporting_quotes=[EvidenceQuote(
            "cA", "Participants: 2,500 undergraduate students Design: Cluster-randomized comparison "
                  "Setting: 12 universities in three regions")])
        self.assertIsNone(_validate_assessment(fabricated, question, {"cA": source}))

    def test_multiple_grounded_quotes_from_one_chunk_are_accepted(self):
        source = AnswerSource(id="cA", chunk_index=1, page_number=3, content_hash="a",
            highlight_text="", text="Study A used a cluster-randomized comparison. "
            "Participants: 2,400 undergraduate students.")
        assessment = EvidenceAssessment("sufficient", ["cA"], supporting_quotes=[
            EvidenceQuote("cA", "Study A used a cluster-randomized comparison"),
            EvidenceQuote("cA", "Participants: 2,400 undergraduate students")])
        result = _validate_assessment(assessment, "What was Study A's methodology and sample size?", {"cA": source})
        self.assertEqual(result.level, "sufficient")
        self.assertEqual(result.chunk_ids, ["cA"])

    def test_single_study_drops_only_redundant_invalid_quote_after_full_field_coverage(self):
        source = AnswerSource(id="cA", chunk_index=4, page_number=3, content_hash="hash",
            highlight_text="", text=(
                "Study A. Design: Cluster-randomized comparison. "
                "Participants: 2,400 students. Setting: 12 universities in three regions. "
                "Observation period: One academic semester."))
        question = ("What methodology did Study A use, and what were its sample size, "
                    "university count, regions, and observation period?")
        exact = [EvidenceQuote("cA", quote) for quote in (
            "Design: Cluster-randomized comparison",
            "Participants: 2,400 students",
            "Setting: 12 universities in three regions",
            "Observation period: One academic semester")]
        redundant = EvidenceQuote("cA", "Unrelated rewritten wording that is not in the source")
        accepted = _validate_assessment(EvidenceAssessment("sufficient", ["cA"],
            supporting_quotes=exact + [redundant]), question, {"cA": source})
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted.chunk_ids, ["cA"])
        for omitted in range(len(exact)):
            incomplete = _validate_assessment(EvidenceAssessment("sufficient", ["cA"],
                supporting_quotes=exact[:omitted] + exact[omitted + 1:] + [redundant]),
                question, {"cA": source})
            self.assertIsNotNone(incomplete)
            self.assertEqual(incomplete.level, "partial")

    def test_five_sources_sixteen_quotes_recover_when_valid_quotes_cover_question(self):
        question = "What were Trial Cedar's design and participant count?"
        sources = {}
        quotes = []
        for index in range(5):
            source_id = f"c{index}"
            text = (f"Trial Cedar site {index} used a cohort design. "
                    f"Participants: {40 + index} patients. Observation period: six months.")
            sources[source_id] = AnswerSource(id=source_id, chunk_index=index,
                page_number=index + 1, content_hash=str(index), highlight_text="", text=text)
            quotes.extend([
                EvidenceQuote(source_id, f"Trial Cedar site {index} used a cohort design"),
                EvidenceQuote(source_id, f"Participants: {40 + index} patients"),
                EvidenceQuote(source_id, "Observation period: six months"),
            ])
        quotes.append(EvidenceQuote("c4", "Trial Cedar site 4 used an experimental design"))
        assessment = EvidenceAssessment("sufficient", list(sources), supporting_quotes=quotes)
        validated = _validate_assessment(assessment, question, sources)
        self.assertIsNotNone(validated)
        self.assertEqual(validated.level, "sufficient")
        self.assertEqual(validated.chunk_ids, list(sources))

        # When every participant quote is invalid, source text alone cannot
        # rescue the assessment or authorize the sufficient label.
        missing = [quote for quote in quotes if not quote.quote.startswith("Participants:")]
        missing += [EvidenceQuote(f"c{index}", f"Participants: {90 + index} patients")
                    for index in range(5)]
        partial = _validate_assessment(EvidenceAssessment(
            "sufficient", list(sources), supporting_quotes=missing), question, sources,
            request_id="quote-case")
        self.assertEqual(partial.level, "partial")
        self.assertIn("participants", partial.missing_information)

    def test_quote_matching_accepts_punctuation_but_rejects_reordered_ellipsis(self):
        source = AnswerSource(id="c0", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text="The cohort enrolled 48 patients; follow-up lasted six months.")
        punctuation = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", "The cohort enrolled 48 patients, follow-up lasted six months")])
        self.assertIsNotNone(_validate_assessment(punctuation,
            "How many patients did the cohort enroll?", {"c0": source}))
        reversed_quote = EvidenceAssessment("sufficient", ["c0"], supporting_quotes=[
            EvidenceQuote("c0", "follow-up lasted six months ... The cohort enrolled 48 patients")])
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            self.assertIsNone(_validate_assessment(reversed_quote,
                "How many patients did the cohort enroll?", {"c0": source}))
        self.assertIn("order_mismatch", "\n".join(logs.output))

    def test_invalid_quotes_leave_only_partially_covered_study(self):
        sources = {
            "cA": AnswerSource(id="cA", chunk_index=0, page_number=1, content_hash="a",
                highlight_text="", text="Trial Solstice used a randomized design. Participants: 48 nurses."),
            "cB": AnswerSource(id="cB", chunk_index=1, page_number=2, content_hash="b",
                highlight_text="", text="Trial Polaris used a cohort design. Participants: 72 nurses."),
        }
        assessment = EvidenceAssessment("sufficient", ["cA", "cB"], supporting_quotes=[
            EvidenceQuote("cA", "Trial Solstice used a randomized design"),
            EvidenceQuote("cA", "Participants: 48 nurses"),
            EvidenceQuote("cB", "Trial Polaris used a cohort design"),
            EvidenceQuote("cB", "Participants: 72 nurses ... Trial Polaris used a cohort design"),
            EvidenceQuote("cB", "cohort design ... Trial Polaris used"),
        ])
        question = "Compare both studies' designs and participants."
        result = _validate_assessment(assessment, question, sources, request_id="coverage-123")
        self.assertEqual(result.level, "partial")
        self.assertIn("Trial Polaris participants", result.missing_information)
        self.assertNotIn("Trial Solstice participants", result.missing_information)

    def test_implicit_comparison_with_four_invalid_quotes_returns_partial(self):
        source_one = AnswerSource(id="c1", chunk_index=0, page_number=1, content_hash="a",
            highlight_text="", text=("Trial Solstice used a randomized design with 48 patients "
                "at two clinics for six months. The measured outcome decreased by 4%. "
                "The authors noted selection bias."))
        source_two = AnswerSource(id="c2", chunk_index=1, page_number=2, content_hash="b",
            highlight_text="", text=("Trial Polaris used a cohort design with 72 patients "
                "at three clinics for one year. The measured outcome did not change. "
                "The authors noted a short follow-up."))
        assessment = EvidenceAssessment("sufficient", ["c1", "c2"], supporting_quotes=[
            EvidenceQuote("c1", "Trial Solstice used a randomized design with 48 patients"),
            EvidenceQuote("c1", "The measured outcome decreased by 4%"),
            EvidenceQuote("c2", "Trial Polaris used a cohort design"),
            EvidenceQuote("c1", "six months ... at two clinics"),
            EvidenceQuote("c1", "selection bias ... The authors noted"),
            EvidenceQuote("c2", "one year ... at three clinics"),
            EvidenceQuote("c2", "short follow-up ... The authors noted"),
        ])
        question = ("Compare both studies' designs, participants, settings, duration, "
                    "findings and limitations.")
        with self.assertLogs("app.routers.research", level="INFO"):
            reviewed = _validate_assessment(assessment, question, {"c1": source_one, "c2": source_two})
        self.assertIsNotNone(reviewed)
        self.assertEqual(reviewed.level, "partial")
        self.assertEqual(reviewed.chunk_ids, ["c1", "c2"])
        self.assertIn("Trial Polaris participants", reviewed.missing_information)
        self.assertIn("Trial Polaris findings", reviewed.missing_information)
        self.assertNotIn("Trial Solstice findings", reviewed.missing_information)

    def test_evidence_excerpt_selects_answering_sentence(self):
        text = ("OCLC provides several discovery services. "
                "Chinese libraries have been working with OCLC for many years. "
                "The National Library of China later formed an initiative.")
        self.assertEqual(_evidence_excerpt(
            "Which libraries have been working with OCLC for many years?", text),
            "Chinese libraries have been working with OCLC for many years.")
        self.assertIn("The National Library of China later formed an initiative.",
                      _evidence_context("Which libraries worked with OCLC?", text))
        long_text = "Background information unrelated to the methodology. " * 65
        long_text += ("Study A used a cluster-randomized design with 2,400 students "
                      "at 12 universities for one semester.")
        context = _evidence_context("What was Study A's design, sample size, universities and duration?", long_text)
        self.assertIn("cluster-randomized design", context)
        self.assertIn("12 universities", context)
        passages = [
            TextChunk(0, 2, text, "h0"),
            TextChunk(1, 13, "OCLC offers discovery services.", "h1"),
        ]
        self.assertEqual(len(_generation_passages(
            "Which libraries have been working with OCLC for many years?", passages)), 1)
        self.assertEqual(len(_generation_passages("How does OCLC work?", passages)), 2)

    def test_short_follow_up_uses_previous_question(self):
        conversation = {"messages": [{"question": "What is the FEVER task?"}]}
        self.assertIn("FEVER task", _question_with_context("What about its limitations?", conversation))
        self.assertEqual(_question_with_context("Explain retrieval augmented generation.", conversation),
                         "Explain retrieval augmented generation.")
        self.assertEqual(_question_with_context("why we use ?", {
            "messages": [{"question": "What is useMemo?"}]
        }), "Why do we use useMemo?")
        self.assertEqual(_question_with_context("What did Trial ORBIT report?", {
            "messages": [{"question": "What did the Northstar cohort report?"}]
        }), "What did Trial ORBIT report?")

    def test_vague_follow_up_retrieves_using_previous_subject(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        vectors.save("alice", "paper-1", [
            TextChunk(0, 2, "useMemo caches a calculation result so React can reuse it until dependencies change.", "memo"),
        ], [[0.0, 1.0]])
        self.client.app.state.ingestion_service = Ingestion(vectors)
        provider = Answers(GeneratedAnswer("It avoids recalculating an unchanged value.", ["c0-memo"]))
        self.client.app.state.answer_provider = provider
        first = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "What is useMemo?"})
        self.assertEqual(first.status_code, 200)
        follow_up = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "why we use ?", "conversation_id": first.json()["conversation_id"],
        })
        self.assertEqual(follow_up.status_code, 200)
        self.assertEqual(follow_up.json()["status"], "answered")
        self.assertEqual(provider.received[0], "Why do we use useMemo?")
        self.assertEqual(follow_up.json()["sources"][0]["page_number"], 2)
        third = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "What about its benefits?",
            "conversation_id": first.json()["conversation_id"],
        })
        self.assertEqual(third.status_code, 200)
        self.assertIn("useMemo", provider.received[0])

    def test_lexical_rerank_promotes_direct_match(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        vectors.save("alice", "paper-2", [
            TextChunk(0, 13, "OCLC works with local libraries.", "h0"),
            TextChunk(1, 2, "Chinese libraries have been working with OCLC for many years.", "h1"),
        ], [[1.0, 0.0], [0.0, 1.0]])
        question = "Which libraries have been working with OCLC for many years?"
        results = vectors.search("alice", "paper-2", [1.0, 0.0], 2, question=question)
        self.assertEqual(results[0].page_number, 2)

    def test_rerank_prefers_one_matching_sentence_over_scattered_table_terms(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        vectors.save("alice", "paper-2", [
            TextChunk(0, 8, "RAG scores in the table. FEVER classification results on another task.", "h0"),
            TextChunk(1, 5, "RAG handles FEVER classification by mapping labels to output tokens.", "h1"),
        ], [[1.0, 0.0], [0.0, 1.0]])
        results = vectors.search("alice", "paper-2", [1.0, 0.0], 2,
                                 question="How does RAG handle FEVER classification?")
        self.assertEqual(results[0].page_number, 5)

    def test_methodology_rerank_prefers_requested_study_and_multiple_fields(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        vectors = VectorRepository(Path(directory.name))
        vectors.save("alice", "paper-2", [
            TextChunk(0, 6, "Study A and Study B are discussed in the conclusion.", "h0"),
            TextChunk(1, 3, "Study A. Design: randomized comparison. Participants: 200 students. "
                      "Setting: 4 universities in three regions. Observation period: One semester.", "h1"),
            TextChunk(2, 4, "Study B. Design: matched courses with 80 students.", "h2"),
        ], [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        question = ("What methodology did Study A use, and what were its sample size, "
                    "university count, regions, and observation period?")
        results = vectors.search("alice", "paper-2", [1.0, 0.0], 3, question=question)
        self.assertEqual(results[0].page_number, 3)
        self.assertEqual([item.page_number for item in vectors.search(
            "bob", "paper-2", [1.0, 0.0], 3, question=question)], [])

    def test_ask_handles_configuration_and_provider_failure(self):
        self.assertFalse(self.client.get("/api/v1/research/capabilities").json()["answer_generation"])
        payload = {"paper_id": "paper-1", "question": "Alpha?"}
        response = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(response.status_code, 503)
        unavailable = UnavailableAnswers(GeneratedAnswer("Should not run", ["c0-hash-a"]))
        self.client.app.state.answer_provider = unavailable
        response = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(unavailable.received)
        self.client.app.state.answer_provider = Answers(TimeoutError())
        self.assertTrue(self.client.get("/api/v1/research/capabilities").json()["answer_generation"])
        response = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("TimeoutError", response.text)
        self.client.app.state.answer_provider = Answers(AnswerRateLimitError("12"))
        response = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "12")
        self.client.app.state.answer_provider = Answers(AnswerRateLimitError())
        response = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(response.status_code, 429)
        self.assertNotIn("retry-after", response.headers)
        response = self.client.post("/api/v1/research/ask", json={**payload, "paper_id": "foreign"})
        self.assertEqual(response.status_code, 404)

    def test_rate_limited_claim_verification_retries_without_regenerating(self):
        class LimitedVerifier(Answers):
            verification_calls = 0
            generation_calls = 0

            def generate(self, question, evidence):
                self.generation_calls += 1
                return super().generate(question, evidence)

            def verify_claims(self, question, claims, evidence):
                self.verification_calls += 1
                if self.verification_calls == 1:
                    raise AnswerRateLimitError("60")
                return super().verify_claims(question, claims, evidence)

        provider = LimitedVerifier(
            GeneratedAnswer("Alpha result. [c0-hash-a]", ["c0-hash-a"]))
        self.client.app.state.answer_provider = provider
        response = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Alpha?"})
        self.assertEqual(response.status_code, 200)
        pending = response.json()
        self.assertEqual(pending["verification_status"], "unavailable")
        self.assertEqual(pending["retry_after_seconds"], 60)
        self.assertIn("Alpha result", pending["answer"])
        retry = self.client.post(f'/api/v1/research/conversations/{pending["conversation_id"]}/messages/{pending["message_id"]}/verify')
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(retry.json()["verification_status"], "verified")
        self.assertEqual(provider.generation_calls, 1)
        self.assertEqual(provider.verification_calls, 2)
        self.assertEqual(len(self.client.app.state.conversation_repository.items[pending["conversation_id"]]["messages"]), 1)

    def test_conversation_history_is_owned_and_paper_bound(self):
        self.client.app.state.answer_provider = Answers(GeneratedAnswer("Alpha result.", ["c0-hash-a"]))
        payload = {"paper_id": "paper-1", "question": "Alpha?"}
        first = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(first.status_code, 200)
        conversation_id = first.json()["conversation_id"]
        second = self.client.post("/api/v1/research/ask", json={**payload, "conversation_id": conversation_id})
        self.assertEqual(second.status_code, 200)
        detail = self.client.get(f"/api/v1/research/conversations/{conversation_id}")
        self.assertEqual(len(detail.json()["messages"]), 2)
        self.assertEqual(len(self.client.get("/api/v1/research/conversations").json()), 1)

        self.client.app.state.conversation_repository.items["foreign"] = {
            "_id": "foreign", "owner_id": "bob", "paper_id": "paper-1", "messages": [],
            "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
        }
        self.assertEqual(self.client.get("/api/v1/research/conversations/foreign").status_code, 404)
        self.assertEqual(self.client.post("/api/v1/research/ask", json={
            **payload, "conversation_id": "foreign",
        }).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/v1/research/conversations/{conversation_id}").status_code, 204)
        self.assertEqual(self.client.get(f"/api/v1/research/conversations/{conversation_id}").status_code, 404)

    def test_conversation_ids_load_follow_ups_and_reject_missing_or_other_paper(self):
        self.client.app.state.answer_provider = Answers(GeneratedAnswer("Alpha result.", ["c0-hash-a"]))
        payload = {"paper_id": "paper-1", "question": "Alpha?"}
        first = self.client.post("/api/v1/research/ask", json=payload).json()
        second = self.client.post("/api/v1/research/ask", json={**payload, "question": "Beta?"}).json()
        first_id, second_id = first["conversation_id"], second["conversation_id"]
        self.assertNotEqual(first_id, second_id)
        follow_up = self.client.post("/api/v1/research/ask", json={
            **payload, "question": "And why?", "conversation_id": first_id,
            "reply_to_message_id": first["message_id"],
        })
        self.assertEqual(follow_up.status_code, 200)
        self.assertEqual(follow_up.json()["conversation_id"], first_id)
        self.assertEqual([item["question"] for item in self.client.get(
            f"/api/v1/research/conversations/{first_id}").json()["messages"]], ["Alpha?", "And why?"])
        self.assertEqual(len(self.client.get(
            f"/api/v1/research/conversations/{second_id}").json()["messages"]), 1)
        self.assertEqual(self.client.get("/api/v1/research/conversations/invalid-id").status_code, 404)
        self.assertEqual(self.client.post("/api/v1/research/ask", json={
            **payload, "paper_id": "another-paper", "conversation_id": first_id,
        }).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/v1/research/conversations/{first_id}").status_code, 204)
        self.assertEqual(self.client.get(f"/api/v1/research/conversations/{first_id}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/v1/research/conversations/{second_id}").status_code, 200)

    def test_conversation_metadata_is_owner_scoped_and_shared_by_all_views(self):
        self.client.app.state.answer_provider = Answers(GeneratedAnswer("Alpha result.", ["c0-hash-a"]))
        first = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "First topic?"}).json()
        second = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "Second topic?"}).json()
        first_id, second_id = first["conversation_id"], second["conversation_id"]
        path = f"/api/v1/research/conversations/{first_id}"
        updated = self.client.patch(path, json={"title": "Renamed topic", "pinned": True})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual((updated.json()["title"], updated.json()["pinned"], updated.json()["paper_filename"]),
                         ("Renamed topic", True, "paper.pdf"))
        self.assertTrue(self.client.get(path).json()["pinned"])
        self.assertFalse(self.client.patch(path, json={"saved": False}).json()["saved"])
        self.assertTrue(self.client.get(path).json()["pinned"])
        self.assertTrue(self.client.patch(path, json={"saved": True}).json()["saved"])
        follow_up = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "What about that?", "conversation_id": first_id})
        self.assertEqual(follow_up.status_code, 200)
        self.assertEqual(follow_up.json()["conversation_id"], first_id)
        listed = self.client.get("/api/v1/research/conversations").json()
        self.assertEqual({item["id"] for item in listed}, {first_id, second_id})
        self.assertEqual(next(item for item in listed if item["id"] == first_id)["title"], "Renamed topic")
        self.assertEqual(len(self.client.get(path).json()["messages"]), 2)
        self.assertEqual(self.client.patch(path, json={"pinned": False}).json()["pinned"], False)
        self.assertFalse(self.client.get(path).json()["pinned"])
        self.assertFalse(self.client.patch(path, json={"saved": False}).json()["saved"])
        self.assertFalse(self.client.get(path).json()["saved"])
        self.assertFalse(next(item for item in self.client.get("/api/v1/research/conversations").json()
                              if item["id"] == first_id)["saved"])
        self.assertTrue(self.client.patch(path, json={"saved": True}).json()["saved"])
        for invalid in ({"title": "  "}, {"pinned": None}, {"saved": None}, {}):
            self.assertEqual(self.client.patch(path, json=invalid).status_code, 422)
        self.assertEqual(self.client.patch("/api/v1/research/conversations/missing", json={"pinned": True}).status_code, 404)
        self.client.app.state.conversation_repository.items[first_id]["owner_id"] = "bob"
        self.assertEqual(self.client.patch(path, json={"title": "Stolen"}).status_code, 404)
        self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual({item["id"] for item in self.client.get("/api/v1/research/conversations").json()}, {second_id})
        self.assertTrue(self.client.patch(f"/api/v1/research/conversations/{second_id}",
                                          json={"pinned": True}).json()["pinned"])
        self.assertEqual(self.client.delete(f"/api/v1/research/conversations/{second_id}").status_code, 204)
        self.assertEqual(self.client.get("/api/v1/research/conversations").json(), [])
        self.assertEqual(asyncio.run(self.client.app.state.paper_repository.find("alice", "paper-1"))["filename"],
                         "paper.pdf")

    def test_conversation_list_includes_more_than_thirty_and_legacy_filenames(self):
        self.client.app.state.answer_provider = Answers(GeneratedAnswer("Alpha result.", ["c0-hash-a"]))
        response = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "Topic?"})
        self.assertEqual(response.status_code, 200)
        repository = self.client.app.state.conversation_repository
        original = repository.items[response.json()["conversation_id"]]
        original.pop("paper_filename")
        original.pop("title")
        original.pop("pinned")
        original.pop("saved")
        for index in range(35):
            item = {**original, "_id": f"extra-{index}", "paper_id": "paper-1",
                    "updated_at": original["updated_at"]}
            repository.items[item["_id"]] = item
        repository.items["other-owner"] = {**original, "_id": "other-owner", "owner_id": "bob"}
        listed = self.client.get("/api/v1/research/conversations")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()), 36)
        self.assertTrue(all(item["paper_filename"] == "paper.pdf" for item in listed.json()))
        self.assertTrue(all(item["title"] == "Topic?" and item["pinned"] is False
                            and item["saved"] is False for item in listed.json()))

    def test_sequential_and_concurrent_answers_keep_request_question_and_parent(self):
        class QuestionAnswers(Answers):
            def generate(self, question, evidence):
                if "first" in question:
                    time.sleep(0.04)
                return GeneratedAnswer(f"Answer for {question} [c0-hash-a].", ["c0-hash-a"])

            def verify_claims(self, question, claims, evidence):
                return [ClaimAssessment(item["id"], "supported",
                    [ClaimQuote("c0-hash-a", "Alpha result")], []) for item in claims]

        self.client.app.state.answer_provider = QuestionAnswers(None)
        first = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Explain Alpha", "request_id": "request-0"}).json()
        conversation_id = first["conversation_id"]
        requests = [
            {"paper_id": "paper-1", "conversation_id": conversation_id,
             "reply_to_message_id": first["message_id"],
             "question": "Explain first Alpha detail", "request_id": "request-1"},
            {"paper_id": "paper-1", "conversation_id": conversation_id,
             "reply_to_message_id": first["message_id"],
             "question": "Explain second Alpha detail", "request_id": "request-2"},
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda item: self.client.post("/api/v1/research/ask", json=item), requests))
        self.assertTrue(all(item.status_code == 200 for item in responses))
        for payload, response in zip(requests, responses):
            data = response.json()
            self.assertEqual(data["request_id"], payload["request_id"])
            self.assertEqual(data["question"], payload["question"])
            self.assertEqual(data["paper_id"], payload["paper_id"])
            self.assertEqual(data["conversation_id"], conversation_id)
            self.assertIn(payload["question"], data["answer"])
        saved = self.client.get(f"/api/v1/research/conversations/{conversation_id}").json()
        self.assertEqual({item["question"] for item in saved["messages"]},
                         {"Explain Alpha", requests[0]["question"], requests[1]["question"]})
        invalid_parent = self.client.post("/api/v1/research/ask", json={
            **requests[0], "reply_to_message_id": "another-conversation-message"})
        self.assertEqual(invalid_parent.status_code, 404)

    def test_follow_up_context_uses_explicit_parent_not_latest_arrival(self):
        conversation = {"messages": [
            {"id": "first", "question": "What is useMemo?"},
            {"id": "second", "question": "What is JWT?"},
        ]}
        resolved = _question_with_context("Why do we use it?", conversation, "first")
        self.assertIn("useMemo", resolved)
        self.assertNotIn("JWT", resolved)
        self.assertEqual(_question_with_context("How did Study B explain its alerts?", conversation, "first"),
                         "How did Study B explain its alerts?")

    def test_feedback_is_saved_per_answer_and_owner_scoped(self):
        self.client.app.state.answer_provider = Answers(GeneratedAnswer("Alpha result.", ["c0-hash-a"]))
        answer = self.client.post("/api/v1/research/ask", json={
            "paper_id": "paper-1", "question": "Alpha?",
        }).json()
        conversation_id, message_id = answer["conversation_id"], answer["message_id"]
        path = f"/api/v1/research/conversations/{conversation_id}/messages/{message_id}/feedback"
        response = self.client.post(path, json={"rating": "incorrect", "reason": "wrong_page"})
        self.assertEqual(response.status_code, 200)
        saved = self.client.get(f"/api/v1/research/conversations/{conversation_id}").json()
        self.assertEqual(saved["messages"][0]["feedback"], {"rating": "incorrect", "reason": "wrong_page"})
        self.assertEqual(self.client.post(path, json={"rating": "helpful", "reason": "wrong_page"}).status_code, 422)
        self.assertEqual(self.client.post(path.replace(conversation_id, "unknown"), json={"rating": "helpful"}).status_code, 404)
        self.assertEqual(self.client.post(path.replace(message_id, "unknown"), json={"rating": "helpful"}).status_code, 404)
