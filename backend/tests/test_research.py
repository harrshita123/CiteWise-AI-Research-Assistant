import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import get_current_user
from app.models import UserResponse
from app.paper_models import ChunkResponse
from app.repositories.vectors import VectorRepository
from app.providers.answers import (AnswerRateLimitError, ClaimAssessment, ClaimQuote, EvidenceAssessment, EvidenceQuote, ExternalAnswer,
                                   ExternalEvidence, GeneratedAnswer)
from app.providers.groq_answers import GroqAnswerProvider
from app.providers.external_search import TavilyExternalSearch
from app.routers.research import (_evidence_context, _evidence_excerpt, _generation_passages,
                                  _methodology_comparison_gaps, _question_with_context,
                                  _reported_calculation, _validate_assessment, _verified_sections,
                                  AnswerSource, ClaimQuoteResponse, ClaimReviewResponse, router)
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
                                     "https://www.meity.gov.in/notice", "Official personal data law excerpt")]
        return [ExternalEvidence("ext1", "Research source", "https://example.org/study", "External result")]


class Conversations:
    def __init__(self):
        self.items = {}

    async def create(self, owner_id, paper_id, message):
        now = datetime.now(timezone.utc)
        item = {"_id": f"conversation-{len(self.items) + 1}", "owner_id": owner_id,
                "paper_id": paper_id, "created_at": now, "updated_at": now, "messages": [message]}
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
        return [item for item in self.items.values() if item["owner_id"] == owner_id]

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

    def test_reported_calculations_use_dossier_inputs(self):
        source_a = AnswerSource(id="c4-a", owner_id="alice", paper_id="paper-1", page_number=3,
            chunk_index=4, content_hash="a", highlight_text="", document_name="paper.pdf",
            text="Study A reports confirmed misconduct of 8% in monitored sections and 14% in "
                 "unmonitored sections, a difference of 6 percentage points.")
        source_b = AnswerSource(id="c8-b", owner_id="alice", paper_id="paper-1", page_number=5,
            chunk_index=8, content_hash="b", highlight_text="", document_name="paper.pdf",
            text="Dimension Study A (S-A) Study B (S-B) Design Cluster-randomized Matched, "
                 "non-randomized Sample size 2,400 780")
        sources = {item.id: item for item in (source_a, source_b)}
        total = _reported_calculation("What is the combined number of participants in both studies?", sources)
        self.assertIn("2,400 + 780 = 3,180", total[0])
        self.assertIn("not one pooled experiment", total[0])
        self.assertEqual(total[1], source_b.id)
        difference = _reported_calculation("What was Study A's difference in misconduct rates?", sources)
        self.assertIn("14% − 8% = 6 percentage points", difference[0])
        self.assertEqual(difference[1], source_a.id)
        self.assertIsNone(_reported_calculation("What is the p-value of Study B?", sources))

    def test_alert_calculations_keep_confirmation_distinct_from_false_positives(self):
        source = AnswerSource(id="c7-b", owner_id="alice", paper_id="paper-1", page_number=4,
            chunk_index=7, content_hash="b", highlight_text="", document_name="paper.pdf",
            text="Total automated alerts 120 Alerts associated with confirmed misconduct 9 "
                 "Alerts not confirmed as misconduct 111 Share associated with confirmed misconduct 7.5% "
                 "The authors caution that an unconfirmed alert does not necessarily mean the system made a technical error.")
        for question in ("What proportion of alerts were confirmed?", "How many alerts were unconfirmed?",
                         "Explain Study B's confirmed and unconfirmed alerts."):
            answer = _reported_calculation(question, {source.id: source})
            self.assertIn("120 − 9 = 111", answer[0])
            self.assertIn("9 ÷ 120 × 100 = 7.5%", answer[0])
            self.assertIn("not automatically a technical false positive", answer[0])
            self.assertEqual(answer[1], source.id)

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
        self.assertEqual(direct.level, "sufficient")
        self.assertEqual(direct.chunk_ids, [facial.id])

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
                self.assertEqual(result.level, "sufficient")
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
        result = _validate_assessment(EvidenceAssessment("none", []),
            "How many students, universities and courses participated in Study B?", {source.id: source})
        self.assertEqual(result.level, "sufficient")
        self.assertEqual(result.chunk_ids, [source.id])
        direct = _reported_calculation(
            "How many students, universities and courses participated in Study B?", {source.id: source})
        self.assertIn("780 students across 3 universities and 26 online courses", direct[0])

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

    def test_identical_question_and_evidence_reuses_validated_classification(self):
        provider = GroqAnswerProvider("test-key", "test-model", 10)
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

    def test_capabilities_key_revision_changes_without_exposing_key(self):
        self.client.app.state.answer_provider = GroqAnswerProvider("first-test-key", "model", 10)
        first = self.client.get("/api/v1/research/capabilities").json()
        self.client.app.state.answer_provider = GroqAnswerProvider("second-test-key", "model", 10)
        second = self.client.get("/api/v1/research/capabilities").json()
        self.assertTrue(first["answer_generation"])
        self.assertNotEqual(first["key_revision"], second["key_revision"])
        self.assertNotIn("first-test-key", str(first))
        self.assertNotIn("second-test-key", str(second))

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
        partial = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(partial.status_code, 200)
        self.assertEqual(partial.json()["assessment"]["chunk_ids"], ["c0-hash-a"])
        self.assertEqual(partial.json()["external_sources"][0]["url"], "https://example.org/study")
        self.assertIn("Beta details", search.queries[-1])
        self.assertTrue(search.queries[-1].startswith("Beta details"))
        self.assertEqual(partial.json()["search_trace"], {
            "tavily_calls": 1, "tavily_cache_hits": 0, "queries": [search.queries[-1]], "failures": 0,
            "duplicate_requests": 0, "retrieved_source_counts": [1], "rejected_irrelevant_sources": 0,
            "claim_status_counts": {"supported": 2}, "verification_failures": 0,
            "groq_calls": 0, "groq_cache_hits": 0, "groq_attempts": 0, "groq_retries": 0, "groq_failures": 0,
            "groq_rate_limits": 0, "groq_server_errors": 0, "groq_network_errors": 0,
            "groq_rate_limit_kind": None, "raw_external_results": 0, "invalid_external_results": 0,
            "invalid_external_urls": 0, "rejected_external_domains": 0,
            "duplicate_external_sources": 0, "accepted_external_sources": 1,
            "fallback_searches": 0, "external_search_errors": 0, "external_search_timeouts": 0})
        conversation = self.client.get(f"/api/v1/research/conversations/{partial.json()['conversation_id']}")
        self.assertEqual(conversation.json()["messages"][0]["external_sources"][0]["id"], "ext1")

        provider.assessment = EvidenceAssessment("partial", ["unknown"], "Beta details")
        count = len(search.queries)
        invalid = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(invalid.status_code, 502)
        self.assertEqual(len(search.queries), count)

        provider.assessment = EvidenceAssessment("partial", ["c0-hash-a"], "Beta details",
            [EvidenceQuote("c0-hash-a", "Invented result")])
        invalid_quote = self.client.post("/api/v1/research/ask", json=payload)
        self.assertEqual(invalid_quote.status_code, 502)
        self.assertEqual(len(search.queries), count)

        provider.assessment = EvidenceAssessment("none", [])
        none = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1", "question": "India biometric law"})
        self.assertEqual(none.status_code, 200)
        self.assertEqual(none.json()["assessment"]["level"], "none")
        self.assertEqual(none.json()["sources"], [])
        self.assertTrue(none.json()["external_sources"])
        self.assertEqual(provider.external_calls, 1)
        self.assertIn("do not establish a complete", none.json()["answer"])
        self.assertEqual(none.json()["search_trace"]["tavily_calls"], 1)

        specific = self.client.post("/api/v1/research/ask", json={"paper_id": "paper-1",
            "question": "What was the p-value of the study?"})
        self.assertEqual(specific.status_code, 200)
        self.assertIn("do not report that value", specific.json()["answer"])
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
        payload = {"paper_id": "paper-1", "question": "webcam privacy effects"}
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
                "paper_id": "paper-1", "question": "What does research say about webcam privacy?"})
        self.assertEqual(response.status_code, 200)
        trace = response.json()["search_trace"]
        self.assertEqual(post.call_count, 2)
        self.assertEqual(trace["fallback_searches"], 1)
        self.assertEqual(trace["retrieved_source_counts"], [0, 1])
        self.assertEqual(trace["accepted_external_sources"], 1)

    def test_provider_failure_does_not_trigger_fallback_or_claim_absence(self):
        self.client.app.state.answer_provider = AssessedAnswers(EvidenceAssessment("none", []))
        self.client.app.state.external_search = TavilyExternalSearch("test-key")
        with patch("app.providers.external_search.requests.post", side_effect=TimeoutError) as post:
            response = self.client.post("/api/v1/research/ask", json={
                "paper_id": "paper-1", "question": "What does research say about webcam privacy?"})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(response.json()["search_trace"]["external_search_timeouts"], 1)
        self.assertEqual(response.json()["search_trace"]["fallback_searches"], 0)
        self.assertIn("does not establish that no external information exists", response.json()["answer"])

    def test_explicit_sample_size_overrides_wrong_none_without_web_search(self):
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
        self.assertEqual(response.json()["assessment"]["level"], "sufficient")
        self.assertIn("412 students", response.json()["answer"])
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
        with self.assertLogs("app.routers.research", level="INFO") as captured:
            result = _validate_assessment(assessment, question, sources)
        self.assertEqual(result.chunk_ids, ["cA", "cB"])
        self.assertTrue(any("source_id=cTable" in line for line in captured.output))
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
        self.assertEqual(partial.level, "partial")
        self.assertIn("Study B", partial.missing_information)
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
            self.assertIsNone(incomplete)

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
        response = self.client.post("/api/v1/research/ask", json={**payload, "paper_id": "foreign"})
        self.assertEqual(response.status_code, 404)

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
