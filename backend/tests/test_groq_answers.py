import json
import unittest
import requests
from unittest.mock import Mock, patch

from app.providers.answers import AnswerRateLimitError, Evidence, ExternalEvidence
from app.providers.groq_answers import GroqAnswerProvider, _retry_delay
from app.providers.external_search import TavilyExternalSearch


class GroqAnswerTests(unittest.TestCase):
    def test_structured_claim_verification_request(self):
        provider = GroqAnswerProvider("test-key", "openai/gpt-oss-120b", 120)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({"claims": [{
            "claim_id": "claim-1", "status": "supported",
            "supporting": [{"source_id": "c1", "quote": "The study included 412 students"}],
            "opposing": [], "explanation": "Directly stated",
        }]})}}]}
        with patch("app.providers.groq_answers.requests.post", return_value=response) as post:
            reviews = provider.verify_claims("How many students?", [{"id": "claim-1",
                "text": "The study included 412 students", "section": "pdf", "cited_ids": ["c1"]}],
                [Evidence("c1", "The study included 412 students.")])
        self.assertEqual(reviews[0].supporting[0].source_id, "c1")
        self.assertEqual(post.call_args.kwargs["json"]["response_format"]["json_schema"]["name"],
                         "citewise_claim_verification")
        messages = post.call_args.kwargs["json"]["messages"]
        self.assertIn("one contiguous substring", messages[0]["content"])
        self.assertIn("Do not paraphrase, reorder fields", messages[0]["content"])
        self.assertIn("multiple short exact substrings", messages[0]["content"])
        self.assertIn("use unsupported with empty supporting quotes", messages[0]["content"])
        self.assertIn("Source ID: c1\nPassage: The study included 412 students.", messages[1]["content"])

    def test_structured_assessment_and_external_answer(self):
        provider = GroqAnswerProvider("test-key", "openai/gpt-oss-120b", 120)
        assessment_response = Mock(status_code=200)
        assessment_response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "level": "partial", "chunk_ids": ["c0"], "missing_information": "privacy effects",
            "supporting_quotes": [{"chunk_id": "c0", "quote": "The study measured performance."}],
        })}}]}
        external_response = Mock(status_code=200)
        external_response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "pdf_text": "PDF [c0].", "external_text": "External [ext1].",
            "pdf_ids": ["c0"], "external_ids": ["ext1"],
        })}}]}
        with patch("app.providers.groq_answers.requests.post", side_effect=[assessment_response, external_response]) as post:
            assessment = provider.assess("What about privacy?", [Evidence("c0", "The study measured performance.")])
            answer = provider.generate_with_external("What about privacy?", [Evidence("c0", "The study measured performance.")],
                [ExternalEvidence("ext1", "Study", "https://example.org", "Privacy concern")], "privacy effects")
        self.assertEqual(assessment.level, "partial")
        self.assertEqual(assessment.chunk_ids, ["c0"])
        self.assertEqual(answer.external_ids, ["ext1"])
        self.assertIn("[ext1]", post.call_args_list[1].kwargs["json"]["messages"][1]["content"])

    def test_external_search_uses_only_real_https_results(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": [
            {"title": "Webcam privacy study", "url": "https://example.edu/study", "content": "Camera monitoring raised privacy concerns."},
            {"title": "Vendor blog", "url": "https://example.org/study", "content": "Marketing excerpt"},
            {"title": "Bad", "url": "javascript:alert(1)", "content": "Bad excerpt"},
        ]}
        with patch("app.providers.external_search.requests.post", return_value=response) as post:
            results = TavilyExternalSearch("test-key").search("webcam privacy")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.edu/study")
        self.assertEqual(post.call_args.kwargs["json"]["include_answer"], False)

    def test_indian_legal_search_keeps_only_official_sources(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": [
            {"title": "Official biometric data law", "url": "https://www.meity.gov.in/notice",
             "content": "Official text about biometric data collection in India"},
            {"title": "Unrelated official policy", "url": "https://www.niti.gov.in/ai-policy", "content": "Policy"},
            {"title": "Blog", "url": "https://example.org/interpretation", "content": "Commentary"},
        ]}
        with patch("app.providers.external_search.requests.post", return_value=response) as post:
            results = TavilyExternalSearch("test-key").search(
                "current legal requirements for biometric data collection in India")
        self.assertEqual([item.title for item in results], ["Official biometric data law"])
        self.assertIn("site:gov.in", post.call_args.kwargs["json"]["query"])

    def test_structured_request_sends_only_selected_evidence(self):
        provider = GroqAnswerProvider("test-key", "openai/gpt-oss-120b", 120)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "status": "answered", "text": "Chinese libraries", "evidence_ids": ["c4-hash"],
        })}}]}
        with patch("app.providers.groq_answers.requests.post", return_value=response) as post:
            answer = provider.generate("Which libraries?", [Evidence("c4-hash", "Chinese libraries worked with OCLC.")])

        self.assertEqual(answer.text, "Chinese libraries")
        self.assertEqual(answer.evidence_ids, ["c4-hash"])
        self.assertEqual(answer.status, "answered")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["timeout"], 120)
        self.assertEqual(kwargs["json"]["model"], "openai/gpt-oss-120b")
        self.assertTrue(kwargs["json"]["response_format"]["json_schema"]["strict"])
        self.assertIn("contradictory_evidence", kwargs["json"]["response_format"]["json_schema"]["schema"]["properties"]["status"]["enum"])
        self.assertIn("[c4-hash] Chinese libraries worked with OCLC.",
                      kwargs["json"]["messages"][1]["content"])
        self.assertIn("Write every inline citation as [source_id]", kwargs["json"]["messages"][0]["content"])
        self.assertIn("For methodological differences, state only contrasts directly established",
                      kwargs["json"]["messages"][0]["content"])
        self.assertNotIn("test-key", json.dumps(kwargs["json"]))

    def test_repeated_answer_reuses_successful_draft_without_reusing_verification(self):
        provider = GroqAnswerProvider("test-key", "model", 10)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "status": "answered", "text": "A supported fact.", "evidence_ids": ["c1"],
        })}}]}
        with patch("app.providers.groq_answers.requests.post", return_value=response) as post:
            for _ in range(2):
                provider.generate("Question?", [Evidence("c1", "A supported fact.")])
            provider._complete("citewise_claim_verification", {}, "check", "claims")
            provider._complete("citewise_claim_verification", {}, "check", "claims")
        self.assertEqual(post.call_count, 3)

    def test_answer_cache_is_scoped_to_question_and_evidence(self):
        provider = GroqAnswerProvider("test-key", "model", 10)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "status": "answered", "text": "A supported fact.", "evidence_ids": ["c1"],
        })}}]}
        with patch("app.providers.groq_answers.requests.post", return_value=response) as post:
            provider.generate("First question?", [Evidence("c1", "First passage.")])
            provider.generate("Second question?", [Evidence("c1", "First passage.")])
            provider.generate("Second question?", [Evidence("c1", "Updated passage.")])
        self.assertEqual(post.call_count, 3)

    def test_repeated_external_search_reuses_same_passages(self):
        search = TavilyExternalSearch("test-key")
        response = Mock(status_code=200)
        response.json.return_value = {"results": [{"title": "Official source",
            "url": "https://example.edu/study", "content": "Privacy research passage"}]}
        with patch("app.providers.external_search.requests.post", return_value=response) as post:
            first = search.search("privacy question")
            second = search.search("privacy question")
        self.assertEqual(first, second)
        self.assertEqual(post.call_count, 1)

    def test_external_search_cache_normalizes_query_and_tracks_fresh_call(self):
        search = TavilyExternalSearch("test-key")
        response = Mock(status_code=200)
        response.json.return_value = {"results": [{"title": "Webcam privacy study",
            "url": "https://example.edu/study", "content": "Webcam monitoring raised privacy concerns."}]}
        with patch("app.providers.external_search.requests.post", return_value=response) as post:
            fresh = search.search_with_trace("webcam   privacy")
            cached = search.search_with_trace(" WEBCAM privacy ")
        self.assertFalse(fresh.cache_hit)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(fresh.results, cached.results)
        self.assertEqual(post.call_count, 1)

    def test_external_search_rejects_irrelevant_trusted_source_and_does_not_cache_failures(self):
        search = TavilyExternalSearch("test-key")
        response = Mock(status_code=200)
        response.json.return_value = {"results": [
            {"title": "Library catalog", "url": "https://example.edu/catalog", "content": "Book lending hours."},
            {"title": "Webcam privacy", "url": "https://example.edu/privacy", "content": "Webcam privacy findings."},
        ]}
        with patch("app.providers.external_search.requests.post", return_value=response) as post:
            outcome = search.search_with_trace("webcam privacy")
            self.assertEqual([item.title for item in outcome.results], ["Webcam privacy"])
            self.assertEqual(outcome.rejected_irrelevant, 1)
            self.assertEqual(post.call_count, 1)
        malformed = Mock(status_code=200)
        malformed.json.return_value = {"unexpected": []}
        with patch("app.providers.external_search.requests.post", return_value=malformed) as post:
            with self.assertRaises(ValueError):
                search.search("new question")
            with self.assertRaises(ValueError):
                search.search("new question")
            self.assertEqual(post.call_count, 2)

    def test_tavily_http_failure_is_not_cached(self):
        search = TavilyExternalSearch("test-key")
        failed = Mock(status_code=503)
        failed.raise_for_status.side_effect = requests.HTTPError("temporary failure")
        good = Mock(status_code=200)
        good.json.return_value = {"results": [{"title": "Webcam privacy",
            "url": "https://example.edu/privacy", "content": "Webcam privacy findings."}]}
        with patch("app.providers.external_search.requests.post", side_effect=[failed, good]) as post:
            with self.assertRaises(requests.HTTPError):
                search.search("webcam privacy")
            self.assertEqual(len(search.search("webcam privacy")), 1)
            self.assertEqual(post.call_count, 2)

    def test_external_search_accepts_synonyms_and_rejects_unrelated_domains(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": [
            {"title": "Camera monitoring and student confidentiality", "url": "https://example.edu/one",
             "content": "Camera monitoring affects privacy among students."},
            {"title": "Camera monitoring and student confidentiality", "url": "https://example.edu/one?tracking=1",
             "content": "Camera monitoring affects privacy among students."},
            {"title": "Camera monitoring", "url": "https://example.edu.evil.test/two",
             "content": "Student privacy."},
            {"title": "Library catalog", "url": "https://example.edu/three", "content": "Book hours."},
            {"title": "Camera monitoring", "url": "https://user:pass@example.edu/four",
             "content": "Student privacy."},
        ]}
        with patch("app.providers.external_search.requests.post", return_value=response):
            outcome = TavilyExternalSearch("test-key").search_with_trace("webcam student privacy")
        self.assertEqual(len(outcome.results), 1)
        self.assertEqual(outcome.duplicates, 1)
        self.assertEqual(outcome.rejected_domains, 1)
        self.assertEqual(outcome.rejected_irrelevant, 1)
        self.assertEqual(outcome.invalid_urls, 1)
        self.assertEqual(outcome.raw_results, 5)

    def test_empty_search_is_cached_briefly(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": []}
        search = TavilyExternalSearch("test-key")
        with patch("app.providers.external_search.requests.post", return_value=response) as post:
            first = search.search_with_trace("webcam privacy")
            second = search.search_with_trace("webcam privacy")
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(post.call_count, 1)

    def test_relevance_across_research_topics(self):
        relevant = (
            ("webcam proctoring privacy", "Camera monitoring study", "Student confidentiality and privacy"),
            ("biometric data law India", "Indian biometric data protection act", "Rules for collection"),
            ("Python async timeout", "Python asyncio timeout documentation", "Timeout handling"),
            ("vaccine efficacy trial", "Randomized vaccine trial", "Efficacy findings"),
        )
        for query, title, excerpt in relevant:
            with self.subTest(query=query):
                self.assertTrue(TavilyExternalSearch._relevant(query, title, excerpt))
        self.assertFalse(TavilyExternalSearch._relevant(
            "webcam proctoring privacy", "University library", "Opening hours and book lending"))
        self.assertTrue(TavilyExternalSearch._trusted_host("research.gov.au"))
        self.assertFalse(TavilyExternalSearch._trusted_host("research.gov.au.evil.test"))

    def test_availability_requires_key(self):
        self.assertFalse(GroqAnswerProvider("", "openai/gpt-oss-120b", 120).available())
        self.assertTrue(GroqAnswerProvider("test-key", "openai/gpt-oss-120b", 120).available())

    def test_rate_limit_exposes_only_retry_delay(self):
        response = Mock(status_code=429, headers={"retry-after": "12"})
        with patch("app.providers.groq_answers.requests.post", return_value=response), patch("app.providers.groq_answers.time.sleep") as sleep:
            with self.assertRaises(AnswerRateLimitError) as caught:
                GroqAnswerProvider("test-key", "openai/gpt-oss-120b", 120).generate(
                    "Question?", [Evidence("c1", "Evidence.")])
        self.assertEqual(caught.exception.retry_after, "12")
        self.assertEqual(sleep.call_count, 2)

    def test_retry_transient_response_then_success(self):
        provider = GroqAnswerProvider("test-key", "model", 10)
        busy = Mock(status_code=503, headers={"retry-after": "0.5"})
        good = Mock(status_code=200)
        good.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
        with patch("app.providers.groq_answers.requests.post", side_effect=[busy, good]) as post, \
                patch("app.providers.groq_answers.time.sleep") as sleep:
            result = provider._complete("test", {}, "system", "user")
        self.assertTrue(result["ok"])
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_retry_timeout_is_bounded(self):
        provider = GroqAnswerProvider("test-key", "model", 10)
        with patch("app.providers.groq_answers.requests.post", side_effect=requests.Timeout) as post, \
                patch("app.providers.groq_answers.time.sleep") as sleep:
            with self.assertRaises(requests.Timeout):
                provider._complete("test", {}, "system", "user")
        self.assertEqual(post.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_long_retry_after_fails_without_premature_retry(self):
        provider = GroqAnswerProvider("test-key", "model", 10)
        busy = Mock(status_code=429, headers={"retry-after": "90"})
        with patch("app.providers.groq_answers.requests.post", return_value=busy) as post, \
                patch("app.providers.groq_answers.time.sleep") as sleep:
            with self.assertRaises(AnswerRateLimitError) as caught:
                provider._complete("test", {}, "system", "user")
        self.assertEqual(caught.exception.retry_after, "90")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_non_transient_error_is_not_retried(self):
        provider = GroqAnswerProvider("test-key", "model", 10)
        bad = Mock(status_code=400, headers={})
        bad.raise_for_status.side_effect = requests.HTTPError("invalid request")
        with patch("app.providers.groq_answers.requests.post", return_value=bad) as post, \
                patch("app.providers.groq_answers.time.sleep") as sleep:
            with self.assertRaises(requests.HTTPError):
                provider._complete("test", {}, "system", "user")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_retry_after_accepts_http_date(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime
        value = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=15))
        self.assertGreater(_retry_delay(value), 10)
