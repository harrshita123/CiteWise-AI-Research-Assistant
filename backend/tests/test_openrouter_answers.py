import json
import unittest
import requests
from unittest.mock import Mock, patch

from app.providers.answers import AnswerRateLimitError, Evidence, ExternalEvidence
from app.providers.openrouter_answers import (OpenRouterAnswerProvider, _retry_delay, model_runs,
                                             openrouter_cache_scope, openrouter_request_id)
from app.providers.openrouter_routing import (
    FAST_MODEL, GENERATION_MODEL, PDF_FALLBACK_MODEL, RECOMMENDATION_MODEL, RERANK_MODEL, VERIFICATION_MODEL, ModelRouter,
)
from app.providers.external_search import TavilyExternalSearch


class OpenRouterAnswerTests(unittest.TestCase):
    def test_generation_complexity_selects_model_and_cap_without_weakening_privacy(self):
        from app.providers.openrouter_routing import OutputLimits

        router = ModelRouter(output_limits=OutputLimits(simple=950, research=3100, external=3300))
        reply = Mock(status_code=200)
        reply.json.return_value = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        cases = [
            ("citewise_answer", True, False, FAST_MODEL, 950),
            ("citewise_answer", False, False, GENERATION_MODEL, 3100),
            ("citewise_external_answer", True, False, FAST_MODEL, 950),
            ("citewise_external_answer", False, False, GENERATION_MODEL, 3300),
            ("citewise_answer", True, True, GENERATION_MODEL, 3100),
        ]
        for stage, simple, zdr, model, cap in cases:
            with self.subTest(stage=stage, simple=simple, zdr=zdr), \
                    patch("app.providers.openrouter_answers.requests.post", return_value=reply) as post:
                provider = OpenRouterAnswerProvider("key", router, require_zdr=zdr)
                provider._complete(stage, {}, "system", "question", simple=simple)
                provider._complete(stage, {}, "system", "question", simple=simple)
                post.assert_called_once()
                body = post.call_args.kwargs["json"]
                self.assertEqual((body["model"], body["max_tokens"]), (model, cap))
                self.assertEqual(body["provider"]["data_collection"], "deny")
                self.assertEqual(body["provider"]["zdr"], zdr)
                self.assertTrue(all(price == 0 for price in body["provider"]["max_price"].values()))
                self.assertEqual("response_format" in body, model == FAST_MODEL)
                self.assertEqual(body["provider"]["require_parameters"], model == FAST_MODEL)

    def test_fast_pdf_path_only_falls_back_once_for_upstream_failure(self):
        busy = Mock(status_code=429, headers={})
        busy.json.return_value = {"error": {"metadata": {"limit_source": "upstream_provider_shared_pool"}}}
        working = Mock(status_code=200)
        working.json.return_value = {"choices": [{"message": {"content": "STATUS: answered\nEight [c1]."}}]}
        budget = Mock()
        provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
        with patch("app.providers.openrouter_answers.requests.post", side_effect=[busy, working]) as post:
            result = provider.generate("How many were enrolled?", [Evidence("c1", "Eight were enrolled.")], simple=True)
        self.assertEqual(result.evidence_ids, ["c1"])
        self.assertEqual([call.kwargs["json"]["model"] for call in post.call_args_list],
                         [FAST_MODEL, PDF_FALLBACK_MODEL])
        self.assertEqual(budget.reserve.call_count, 2)

        busy.json.return_value = {"error": {"metadata": {"limit_source": "openrouter_free_tier_daily"}}}
        with patch("app.providers.openrouter_answers.requests.post", return_value=busy) as post:
            with self.assertRaises(AnswerRateLimitError):
                provider.generate("How many completed?", [Evidence("c1", "Eight completed.")], simple=True)
        post.assert_called_once()

    def test_identical_concurrent_calls_share_one_dispatch_per_stage(self):
        from concurrent.futures import Future, ThreadPoolExecutor
        from threading import Event

        stages = ("citewise_question_recommendations", "citewise_evidence_assessment",
                  "citewise_answer", "citewise_claim_verification")
        for stage in stages:
            with self.subTest(stage=stage):
                entered, joined, release = Event(), Event(), Event()
                budget = Mock()
                provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
                reply = Mock(status_code=200)
                reply.json.return_value = {"choices": [{"message": {"content": "{}"}}]}

                class JoinableFuture(Future):
                    def result(self, *args, **kwargs):
                        joined.set()
                        return super().result(*args, **kwargs)

                def transport(*args, **kwargs):
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError("The second caller did not join.")
                    return reply

                with patch("app.providers.openrouter_answers.Future", JoinableFuture), \
                        patch("app.providers.openrouter_answers.requests.post", side_effect=transport) as post, \
                        ThreadPoolExecutor(max_workers=2) as pool:
                    try:
                        first = pool.submit(provider._complete, stage, {}, "system", "user")
                        self.assertTrue(entered.wait(3))
                        second = pool.submit(provider._complete, stage, {}, "system", "user")
                        self.assertTrue(joined.wait(3))
                    finally:
                        release.set()
                    self.assertEqual((first.result(), second.result()), ({}, {}))
                    post.assert_called_once()
                    budget.reserve.assert_called_once()
                    self.assertEqual(budget.reserve.call_args.kwargs["stage"], stage)
                    budget.dispatch.assert_called_once()

    def test_cached_suggestions_are_scoped_to_paper_and_owner(self):
        provider = OpenRouterAnswerProvider("key", ModelRouter())
        reply = Mock(status_code=200)
        reply.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=reply) as post:
            for scope in (("alice", "paper"), ("alice", "paper"), ("alice", "other"), ("bob", "paper")):
                token = openrouter_cache_scope.set(scope)
                try:
                    provider._complete("citewise_question_recommendations", {}, "system", "user")
                finally:
                    openrouter_cache_scope.reset(token)
        self.assertEqual(post.call_count, 3)

    def test_attempt_audit_keeps_failed_dispatches_and_links_fallback(self):
        budget = Mock()
        budget.reserve.side_effect = ["first", "second"]
        provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
        good = Mock(status_code=200)
        good.json.return_value = {"provider": "Baidu", "choices": [{"message": {"content": "{}"}}]}
        with patch("app.providers.openrouter_answers.requests.post", side_effect=[requests.Timeout, good]):
            self.assertEqual(provider._complete("citewise_evidence_assessment", {}, "system", "user"), {})
        first, second = budget.reserve.call_args_list
        self.assertEqual(first.kwargs["operation_id"], second.kwargs["operation_id"])
        self.assertEqual((first.kwargs["attempt"], second.kwargs["attempt"]), (1, 2))
        self.assertFalse(first.kwargs["fallback"])
        self.assertTrue(second.kwargs["fallback"])
        self.assertRegex(first.kwargs["key_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(first.kwargs["key_id"], second.kwargs["key_id"])
        budget.fail_before_dispatch.assert_not_called()
        self.assertEqual([item.args for item in budget.finish.call_args_list],
                         [("first", "transport_failed"), ("second", "http_success")])
        self.assertEqual(budget.finish.call_args.kwargs["upstream_provider"], "Baidu")

    def test_undispatched_ticket_never_reaches_transport(self):
        budget = Mock()
        budget.dispatch.return_value = False
        provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
        with patch("app.providers.openrouter_answers.requests.post") as post:
            with self.assertRaises(AnswerRateLimitError):
                provider._complete("citewise_question_recommendations", {}, "system", "user")
        post.assert_not_called()
        budget.finish.assert_not_called()
        budget.fail_before_dispatch.assert_called_once_with(budget.reserve.return_value)

    def test_dispatch_acknowledgement_failure_does_not_call_provider(self):
        from pymongo.errors import AutoReconnect

        budget = Mock()
        budget.reserve.return_value = "ticket"
        budget.dispatch.side_effect = AutoReconnect
        budget.fail_before_dispatch.return_value = False
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(), budget=budget)
        with patch("app.providers.openrouter_answers.requests.post") as post:
            with self.assertRaises(AutoReconnect):
                provider._complete("citewise_question_recommendations", {}, "system", "user")
        post.assert_not_called()
        budget.fail_before_dispatch.assert_called_once_with("ticket")
        budget.finish.assert_not_called()

    def test_assessment_prompt_requests_separate_ordered_comparison_quotes(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10))
        with patch.object(provider, "_complete", return_value={
            "level": "partial", "chunk_ids": ["c1"], "missing_information": "one comparison side",
            "supporting_quotes": [{"chunk_id": "c1", "quote": "The trial enrolled 48 patients"}],
        }) as complete:
            provider.assess("Compare both trials' methods.", [Evidence("c1", "The trial enrolled 48 patients.")])
        prompt = complete.call_args.args[2]
        self.assertIn("each requested dimension for each study separately", prompt)
        self.assertIn("original source order", prompt)
        self.assertEqual(complete.call_count, 1)

    def test_usage_logging_identifies_stage_without_logging_private_text(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10))
        response = Mock(status_code=200)
        response.json.return_value = {"usage": {"prompt_tokens": 34, "completion_tokens": 9, "total_tokens": 43},
                                      "choices": [{"message": {"content": '{"ok": true}'}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response), \
                self.assertLogs("app.providers.openrouter_answers", level="INFO") as logs:
            provider._complete("citewise_claim_verification", {}, "private system", "private passage")
        self.assertIn("stage=citewise_claim_verification", " ".join(logs.output))
        self.assertIn("prompt_tokens=34", " ".join(logs.output))
        self.assertNotIn("private passage", " ".join(logs.output))

    def test_calibration_uses_existing_generation_and_verification_calls(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10))
        with patch.object(provider, "_complete", side_effect=[
            {"status": "answered", "text": "A qualified finding [c1].", "evidence_ids": ["c1"]},
            {"claims": [{"claim_id": "claim-1", "status": "partially_supported",
                         "supporting": [{"source_id": "c1", "quote": "The survey observed a difference"}],
                         "opposing": [], "explanation": "The causal interpretation is not established.",
                         "question_fit": "partial", "evidence_use": "establishes_claim", "scope_preserved": False}]},
        ]) as complete:
            provider.generate("What did the survey find?", [Evidence("c1", "The survey observed a difference.")])
            provider.verify_claims("What did the survey find?", [{"id": "claim-1",
                "text": "The survey proved a causal effect", "section": "pdf", "cited_ids": ["c1"]}],
                [Evidence("c1", "The survey observed a difference.")])
        self.assertEqual(complete.call_count, 2)
        generation_prompt = complete.call_args_list[0].args[2]
        verification_prompt = complete.call_args_list[1].args[2]
        self.assertIn("numerical differences as differences", generation_prompt)
        self.assertIn("missing test is unknown", generation_prompt)
        self.assertIn("author's interpretation", verification_prompt)
        self.assertIn("does not alone establish causation", verification_prompt)

    def test_structured_claim_verification_request(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=120))
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({"claims": [{
            "claim_id": "claim-1", "status": "supported",
            "supporting": [{"source_id": "c1", "quote": "The study included 412 students"}],
            "opposing": [], "explanation": "Directly stated",
            "question_fit": "direct", "evidence_use": "establishes_claim", "scope_preserved": True,
            "unsupported_source_ids": [],
        }], "requirements": []})}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
            reviews = provider.verify_claims("How many students?", [{"id": "claim-1",
                "text": "The study included 412 students", "section": "pdf", "cited_ids": ["c1"]}],
                [Evidence("c1", "The study included 412 students.", {"origin": "pdf"})])
        self.assertEqual(reviews[0].supporting[0].source_id, "c1")
        self.assertEqual(post.call_args.kwargs["json"]["model"], VERIFICATION_MODEL)
        self.assertEqual(post.call_args.kwargs["json"]["response_format"]["json_schema"]["name"],
                         "citewise_claim_verification")
        messages = post.call_args.kwargs["json"]["messages"]
        self.assertIn("one contiguous substring", messages[0]["content"])
        self.assertIn("Do not paraphrase, reorder fields", messages[0]["content"])
        self.assertIn("multiple short exact substrings", messages[0]["content"])
        self.assertIn("use unsupported with empty supporting quotes", messages[0]["content"])
        self.assertIn('Source ID: c1\nProvenance: {"origin": "pdf"}\nPassage: The study included 412 students.',
                      messages[1]["content"])

    def test_structured_assessment_and_external_answer(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=120))
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
        with patch("app.providers.openrouter_answers.requests.post", side_effect=[assessment_response, external_response]) as post:
            assessment = provider.assess("What about privacy?", [Evidence("c0", "The study measured performance.")])
            answer = provider.generate_with_external("What about privacy?", [Evidence("c0", "The study measured performance.")],
                [ExternalEvidence("ext1", "Study", "https://example.org", "Privacy concern")], "privacy effects")
        self.assertEqual(assessment.level, "partial")
        self.assertEqual(assessment.chunk_ids, ["c0"])
        self.assertEqual(answer.external_ids, ["ext1"])
        assessment_request, external_request = [call.kwargs["json"] for call in post.call_args_list]
        self.assertEqual(assessment_request["model"], VERIFICATION_MODEL)
        self.assertTrue(assessment_request["response_format"]["json_schema"]["strict"])
        self.assertEqual(external_request["model"], GENERATION_MODEL)
        self.assertNotIn("response_format", external_request)
        self.assertFalse(external_request["provider"]["require_parameters"])
        self.assertIn('"external_ids"', external_request["messages"][0]["content"])
        self.assertIn("[ext1]", external_request["messages"][1]["content"])

    def test_claim_verifier_rejects_unstructured_relevance_and_scope(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10))
        with patch.object(provider, "_complete", return_value={"claims": [{"status": "supported"}]}) as complete:
            with self.assertRaisesRegex(ValueError, "required evidence check"):
                provider.verify_claims("Which district?", [], [Evidence("ext1", "Survey results.",
                                       {"origin": "web", "publisher": "Survey office"})])
        self.assertIn('"publisher": "Survey office"', complete.call_args.args[3])
        self.assertEqual(complete.call_count, 1)

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

    def test_external_search_discards_missing_metadata_and_keeps_stable_valid_ids(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": [
            {"title": "Recovery evidence", "content": "Recovery was measured."},
            {"title": "Recovery evidence", "url": "https://example.edu/empty"},
            {"title": "Recovery evidence", "url": "http://example.edu/insecure",
             "content": "Recovery was measured."},
            {"title": "Recovery evidence", "url": "https://example.edu/valid",
             "content": "Recovery was measured in a cohort."},
        ]}
        with patch("app.providers.external_search.requests.post", return_value=response):
            outcome = TavilyExternalSearch("test-key").search_with_trace("recovery cohort")
        self.assertEqual(outcome.raw_results, 4)
        self.assertEqual([item.id for item in outcome.results], ["ext1"])
        self.assertEqual(outcome.results[0].url, "https://example.edu/valid")
        self.assertEqual(outcome.invalid_urls, 2)
        self.assertEqual(outcome.invalid_results, 1)

    def test_legal_search_prioritizes_relevant_official_sources(self):
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
        self.assertEqual(post.call_args.kwargs["json"]["query"],
                         "current legal requirements for biometric data collection in India")

    def test_nemotron_requests_json_in_prompt_and_sends_only_selected_evidence(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=120))
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "status": "answered", "text": "Chinese libraries", "evidence_ids": ["c4-hash"],
        })}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
            answer = provider.generate("Which libraries?", [Evidence("c4-hash", "Chinese libraries worked with OCLC.")])

        self.assertEqual(answer.text, "Chinese libraries")
        self.assertEqual(answer.evidence_ids, ["c4-hash"])
        self.assertEqual(answer.status, "answered")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["timeout"], 120)
        self.assertEqual(kwargs["json"]["model"], GENERATION_MODEL)
        self.assertNotIn("response_format", kwargs["json"])
        self.assertFalse(kwargs["json"]["provider"]["require_parameters"])
        self.assertIn("Return only one JSON object", kwargs["json"]["messages"][0]["content"])
        self.assertIn('"contradictory_evidence"', kwargs["json"]["messages"][0]["content"])
        self.assertIn("[c4-hash] Chinese libraries worked with OCLC.",
                      kwargs["json"]["messages"][1]["content"])
        self.assertIn("Write every inline citation as [source_id]", kwargs["json"]["messages"][0]["content"])
        self.assertIn("For methodological differences, state only contrasts directly established",
                      kwargs["json"]["messages"][0]["content"])
        self.assertNotIn("test-key", json.dumps(kwargs["json"]))

    def test_repeated_answer_reuses_successful_draft_without_reusing_verification(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10))
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "status": "answered", "text": "A supported fact.", "evidence_ids": ["c1"],
        })}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
            for _ in range(2):
                provider.generate("Question?", [Evidence("c1", "A supported fact.")])
            provider._complete("citewise_claim_verification", {}, "check", "claims")
            provider._complete("citewise_claim_verification", {}, "check", "claims")
        self.assertEqual(post.call_count, 3)

    def test_answer_cache_is_scoped_to_question_and_evidence(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10))
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "status": "answered", "text": "A supported fact.", "evidence_ids": ["c1"],
        })}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
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

    def test_external_search_prioritizes_trusted_hosts_without_domain_allowlist(self):
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
        self.assertEqual(len(outcome.results), 2)
        self.assertEqual(outcome.results[0].url, "https://example.edu/one")
        self.assertEqual(outcome.duplicates, 1)
        self.assertEqual(outcome.rejected_domains, 0)
        self.assertEqual(outcome.rejected_irrelevant, 1)
        self.assertEqual(outcome.invalid_urls, 1)
        self.assertEqual(outcome.raw_results, 5)

    def test_external_search_keeps_distinct_record_queries_and_drops_tracking_duplicates(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": [
            {"title": "Coastal flooding records", "url": f"https://data.example.org/record?{query}",
             "content": "The coastal flooding records list observed water levels."}
            for query in ("id=alpha", "id=beta", "id=alpha&utm_source=search")
        ]}
        with patch("app.providers.external_search.requests.post", return_value=response):
            outcome = TavilyExternalSearch("test-key").search_with_trace("coastal flooding records")
        self.assertEqual(len(outcome.results), 2)
        self.assertEqual(outcome.duplicates, 1)

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

    def test_router_dispatches_structured_tasks_and_rejects_paid_models(self):
        with self.assertRaises(ValueError):
            ModelRouter(generation=GENERATION_MODEL.removesuffix(":free"))
        with self.assertRaises(ValueError):
            ModelRouter(verification=RERANK_MODEL)
        with self.assertRaises(ValueError):
            ModelRouter(verification=GENERATION_MODEL)
        with self.assertRaises(ValueError):
            ModelRouter(verification="google/gemma-4-31b-it:free")
        router = ModelRouter()
        for stage in ("citewise_answer", "citewise_external_answer"):
            self.assertEqual(router.for_stage(stage).model, "nvidia/nemotron-3-ultra-550b-a55b:free")
        self.assertEqual(router.for_stage("citewise_question_recommendations").model,
                         "inclusionai/ling-3.2-flash:free")
        for stage in ("citewise_evidence_reassessment", "citewise_claim_verification",
                      "citewise_evaluation_proposals"):
            self.assertEqual(router.for_stage(stage).model, "inclusionai/ling-3.2-flash:free")
        provider = OpenRouterAnswerProvider("key", ModelRouter(verification=RECOMMENDATION_MODEL))
        response = Mock(status_code=200)
        response.json.return_value = {"provider": "Liquid",
            "choices": [{"message": {"content": '{"questions":["What was measured?"]}'}}]}
        calls = []
        token = model_runs.set(calls)
        try:
            with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
                self.assertEqual(provider.recommend_questions("starter", "", [Evidence("c1", "Data.")], 4),
                                 ["What was measured?"])
        finally:
            model_runs.reset(token)
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["model"], RECOMMENDATION_MODEL)
        self.assertEqual(request["provider"]["max_price"], {"prompt": 0, "completion": 0, "request": 0})
        self.assertEqual(request["provider"]["data_collection"], "deny")
        self.assertTrue(request["provider"]["require_parameters"])
        self.assertFalse(request["provider"]["allow_fallbacks"])
        self.assertEqual(calls[0]["role"], "question_recommendation")
        self.assertEqual(calls[0]["upstream_provider"], "Liquid")
        self.assertEqual(provider.router.for_stage("citewise_claim_verification").model, RECOMMENDATION_MODEL)

    def test_external_model_is_independently_configurable(self):
        router = ModelRouter(external_generation=VERIFICATION_MODEL)
        self.assertEqual(router.for_stage("citewise_answer").model, GENERATION_MODEL)
        external = router.for_stage("citewise_external_answer")
        self.assertEqual((external.model, external.role), (VERIFICATION_MODEL, "external_answer_generation"))
        with self.assertRaises(ValueError):
            ModelRouter(vision="nvidia/nemotron-3.5-super-vl")
        with self.assertRaises(ValueError):
            ModelRouter(rerank="nvidia/nemotron-3.5-rerank-8b")

    def test_generation_fallback_on_no_eligible_endpoint_and_never_on_account_429(self):
        fallback = "baidu/ernie-5.5-flash-preview:free"
        budget = Mock()
        provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
        unavailable = Mock(status_code=404, headers={})
        unavailable.json.return_value = {"error": {"message": "No endpoints found matching your data policy."}}
        good = Mock(status_code=200)
        good.json.return_value = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        with patch("app.providers.openrouter_answers.requests.post", side_effect=[unavailable, good]) as post:
            self.assertTrue(provider._complete("citewise_external_answer", {}, "system", "question")["ok"])
        self.assertEqual([call.kwargs["json"]["model"] for call in post.call_args_list],
                         [GENERATION_MODEL, fallback])
        primary_request, fallback_request = [call.kwargs["json"] for call in post.call_args_list]
        self.assertNotIn("response_format", primary_request)
        self.assertTrue(fallback_request["response_format"]["json_schema"]["strict"])
        self.assertEqual(fallback_request["provider"]["data_collection"], "deny")
        self.assertEqual(fallback_request["provider"]["only"], ["baidu"])
        self.assertEqual(ModelRouter(fallback="").generation.fallback, "")
        self.assertEqual(budget.reserve.call_count, 2)
        limited = Mock(status_code=429, headers={"Retry-After": "42"})
        limited.json.return_value = {}
        # requests uses case-insensitive response headers.
        limited.headers = requests.structures.CaseInsensitiveDict(limited.headers)
        with patch("app.providers.openrouter_answers.requests.post", return_value=limited) as post:
            with self.assertRaises(AnswerRateLimitError):
                provider._complete("citewise_answer", {}, "system", "another question")
        post.assert_called_once()
        budget.cool_down.assert_called_once_with(42)
        budget.reserve.side_effect = AnswerRateLimitError("42")
        with patch("app.providers.openrouter_answers.requests.post") as post:
            with self.assertRaises(AnswerRateLimitError):
                provider.recommend_questions("starter", "", [Evidence("c1", "Data.")], 4)
            post.assert_not_called()

    def test_generation_does_not_fallback_on_unrelated_404_or_weaken_zero_retention(self):
        for require_zdr, message in ((False, "Resource not found."),
                                     (True, "No endpoints found matching your data policy.")):
            with self.subTest(require_zdr=require_zdr):
                provider = OpenRouterAnswerProvider("key", ModelRouter(), require_zdr=require_zdr)
                unavailable = Mock(status_code=404, headers={})
                unavailable.json.return_value = {"error": {"message": message}}
                unavailable.raise_for_status.side_effect = requests.HTTPError()
                with patch("app.providers.openrouter_answers.requests.post", return_value=unavailable) as post:
                    with self.assertRaises(requests.HTTPError):
                        provider.generate("Question?", [Evidence("c1", "Evidence.")])
                post.assert_called_once()
                self.assertEqual(post.call_args.kwargs["json"]["provider"]["zdr"], require_zdr)

    def test_generation_uses_plain_cited_text_on_free_fallback(self):
        for failure in (404, 429):
            with self.subTest(status=failure):
                provider = OpenRouterAnswerProvider("key", ModelRouter())
                unavailable = Mock(status_code=failure, headers={})
                unavailable.json.return_value = {"error": {"message": "No endpoints found matching your data policy."}} \
                    if failure == 404 else {"error": {"metadata": {"limit_source": "upstream_provider_shared_pool"}}}
                good = Mock(status_code=200)
                good.json.return_value = {"choices": [{"finish_reason": "stop",
                    "message": {"content": "STATUS: answered\nThe experiment used three thermometers [sample]."}}]}
                with patch("app.providers.openrouter_answers.requests.post", side_effect=[unavailable, good]) as post:
                    result = provider.generate("How many thermometers were used?",
                                               [Evidence("sample", "The experiment used three thermometers.")])
                self.assertEqual(result.status, "answered")
                self.assertEqual(result.evidence_ids, ["sample"])
                self.assertIn("[sample]", result.text)
                primary, fallback = [call.kwargs["json"] for call in post.call_args_list]
                self.assertEqual([primary["model"], fallback["model"]], [GENERATION_MODEL, PDF_FALLBACK_MODEL])
                self.assertNotIn("only", primary["provider"])
                self.assertEqual(fallback["provider"]["only"], ["google-ai-studio"])
                for body in (primary, fallback):
                    self.assertNotIn("response_format", body)
                    self.assertFalse(body["provider"]["require_parameters"])
                    self.assertEqual(body["provider"]["data_collection"], "deny")
                    self.assertEqual(body["provider"]["max_price"], {"prompt": 0, "completion": 0, "request": 0})

    def test_plain_text_fallback_does_not_accept_invalid_citations_or_retry_twice(self):
        unavailable = Mock(status_code=404, headers={})
        unavailable.json.return_value = {"error": {"message": "No endpoints found."}}
        provider = OpenRouterAnswerProvider("key", ModelRouter())
        for content in ("STATUS: answered\nThree thermometers [invented].",
                        "STATUS: answered\nThree thermometers.",
                        "STATUS: invalid\nThree thermometers [c1]."):
            with self.subTest(content=content):
                bad = Mock(status_code=200)
                bad.json.return_value = {"choices": [{"message": {"content": content}}]}
                with patch("app.providers.openrouter_answers.requests.post", side_effect=[unavailable, bad]) as post:
                    with self.assertRaises(ValueError):
                        provider.generate("Question?", [Evidence("c1", "The study used three thermometers.")])
                self.assertEqual(post.call_count, 2)

    def test_no_eligible_generator_then_upstream_429_stops_after_two_calls(self):
        unavailable = Mock(status_code=404, headers={})
        unavailable.json.return_value = {"error": {"message": "No endpoints found matching your data policy."}}
        limited = Mock(status_code=429, headers={"retry-after": "20"})
        limited.json.return_value = {"error": {"metadata": {"limit_source": "upstream_provider_shared_pool"}}}
        budget = Mock()
        provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
        with patch("app.providers.openrouter_answers.requests.post", side_effect=[unavailable, limited]) as post:
            with self.assertRaises(AnswerRateLimitError) as failure:
                provider.generate("Question?", [Evidence("c1", "The study used three thermometers.")])
        self.assertEqual(failure.exception.retry_after, "20")
        self.assertEqual(post.call_count, 2)
        budget.cool_down.assert_called_once_with(20, model=PDF_FALLBACK_MODEL)
        self.assertEqual([call.kwargs["http_status"] for call in budget.finish.call_args_list], [404, 429])
        self.assertTrue(all(call.args[1] == "provider_rejected" for call in budget.finish.call_args_list))

    def test_rejects_incomplete_structured_output_and_does_not_cache_it(self):
        budget = Mock()
        budget.reserve.side_effect = ["first", "retry"]
        provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": '{"text":"bad"}'}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "invalid structured response"):
                    provider.generate("Question?", [Evidence("c1", "Evidence.")])
        self.assertEqual(post.call_count, 2)
        self.assertEqual([call.args for call in budget.finish.call_args_list],
                         [("first", "malformed_response"), ("retry", "malformed_response")])

    def test_rerank_uses_its_dedicated_endpoint_and_validates_indices(self):
        provider = OpenRouterAnswerProvider("key", ModelRouter())
        response = Mock(status_code=200)
        response.json.return_value = {"results": [{"index": 1}, {"index": 0}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
            self.assertEqual(provider.rerank("Query", ["one", "two"], 2), [1, 0])
            self.assertEqual(post.call_args.args[0], "https://openrouter.ai/api/v1/rerank")
            body = post.call_args.kwargs["json"]
            self.assertEqual(body["model"], RERANK_MODEL)
            self.assertTrue(body["provider"]["zdr"])
            response.json.return_value = {"results": [{"index": 1}, {"index": 1}]}
            with self.assertRaisesRegex(ValueError, "invalid rerank indices"):
                provider.rerank("Query", ["one", "two"], 2)

    def test_availability_requires_key(self):
        self.assertFalse(OpenRouterAnswerProvider("", ModelRouter(timeout=120)).available())
        self.assertTrue(OpenRouterAnswerProvider("test-key", ModelRouter(timeout=120)).available())

    def test_rate_limit_exposes_only_retry_delay(self):
        response = Mock(status_code=429, headers={"retry-after": "12"})
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post, \
                patch("app.providers.openrouter_answers.time.sleep") as sleep, \
                patch("app.providers.openrouter_answers.random.uniform", return_value=0):
            with self.assertRaises(AnswerRateLimitError) as caught:
                OpenRouterAnswerProvider("test-key", ModelRouter(timeout=120)).generate(
                    "Question?", [Evidence("c1", "Evidence.")])
        self.assertEqual(caught.exception.retry_after, "12")
        post.assert_called_once()
        sleep.assert_not_called()

    def test_rate_limit_diagnostic_uses_only_safe_fields(self):
        response = Mock(status_code=429, headers={
            "retry-after": "65", "x-ratelimit-limit-requests": "30",
            "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "1m5s",
            "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "7100",
            "x-ratelimit-reset-tokens": "25s"})
        response.json.return_value = {"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded",
                                                 "message": "private question and PDF text"}}
        token = openrouter_request_id.set("request-123")
        try:
            with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post, \
                    self.assertLogs("uvicorn.error", level="WARNING") as logs:
                with self.assertRaises(AnswerRateLimitError) as caught:
                    OpenRouterAnswerProvider("private-api-key", ModelRouter(timeout=10))._complete("citewise_claim_verification", {}, "system", "user")
        finally:
            openrouter_request_id.reset(token)
        self.assertEqual(caught.exception.retry_after, "65")
        post.assert_called_once()
        event = logs.output[0]
        for field in ("request_id=request-123", "status=429", "error_type=rate_limit_error",
                      "error_code=rate_limit_exceeded", "retry_source=retry-after",
                      "x_ratelimit_remaining_requests': '0'", "x_ratelimit_reset_tokens': '25s'"):
            self.assertIn(field, event)
        for private in ("private-api-key", "private question", "PDF text", "system", "user"):
            self.assertNotIn(private, event)

    def test_rate_limit_uses_exhausted_token_reset_without_retry_after(self):
        response = Mock(status_code=429, headers={
            "x-ratelimit-remaining-requests": "12", "x-ratelimit-reset-requests": "10s",
            "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "2m15s"})
        response.json.return_value = {"error": {"type": "rate_limit_error"}}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post, \
                self.assertLogs("uvicorn.error", level="WARNING") as logs:
            with self.assertRaises(AnswerRateLimitError) as caught:
                OpenRouterAnswerProvider("key", ModelRouter(timeout=10))._complete("citewise_claim_verification", {}, "system", "user")
        self.assertEqual(caught.exception.retry_after, "135")
        self.assertIn("retry_source=quota-reset retry_seconds=135", logs.output[0])
        post.assert_called_once()

    def test_rate_limit_uses_exhausted_request_reset_without_retry_after(self):
        response = Mock(status_code=429, headers={
            "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "45s"})
        response.json.return_value = {"error": {"code": "rate_limit_exceeded"}}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
            with self.assertRaises(AnswerRateLimitError) as caught:
                OpenRouterAnswerProvider("key", ModelRouter(timeout=10))._complete("citewise_claim_verification", {}, "system", "user")
        self.assertEqual(caught.exception.retry_after, "45")
        post.assert_called_once()

    def test_rate_limit_without_usable_provider_timing_has_unknown_retry(self):
        response = Mock(status_code=429, headers={"x-ratelimit-reset-tokens": "1m",
                                                   "x-ratelimit-remaining-tokens": "unknown",
                                                   "x-ratelimit-limit-tokens": "private PDF text"})
        response.json.return_value = {"error": {"message": "private PDF text"}}
        token = openrouter_request_id.set("bad\nrequest")
        try:
            with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post, \
                    patch("app.providers.openrouter_answers.time.sleep") as sleep, \
                    self.assertLogs("uvicorn.error", level="WARNING") as logs:
                with self.assertRaises(AnswerRateLimitError) as caught:
                    OpenRouterAnswerProvider("key", ModelRouter(timeout=10))._complete("citewise_claim_verification", {}, "system", "user")
        finally:
            openrouter_request_id.reset(token)
        self.assertIsNone(caught.exception.retry_after)
        self.assertIn("request_id=none", logs.output[0])
        self.assertIn("retry_source=unknown retry_seconds=None", logs.output[0])
        self.assertNotIn("private PDF text", logs.output[0])
        post.assert_called_once()
        sleep.assert_not_called()

    def test_upstream_429_retries_once_with_free_schema_and_privacy_constraints(self):
        budget = Mock()
        provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
        busy = Mock(status_code=429, headers={"retry-after": "45"})
        busy.json.return_value = {"error": {"metadata": {"limit_source": "upstream_provider_shared_pool"}}}
        good = Mock(status_code=200)
        good.json.return_value = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        with patch("app.providers.openrouter_answers.requests.post", side_effect=[busy, good]) as post:
            self.assertTrue(provider._complete("citewise_evidence_assessment", {}, "system", "user")["ok"])
        bodies = [call.kwargs["json"] for call in post.call_args_list]
        self.assertEqual([body["model"] for body in bodies], [VERIFICATION_MODEL, FAST_MODEL])
        self.assertEqual([call.args[0] for call in budget.reserve.call_args_list], [VERIFICATION_MODEL, FAST_MODEL])
        budget.cool_down.assert_called_once_with(45, model=VERIFICATION_MODEL)
        for body in bodies:
            self.assertEqual(body["provider"]["data_collection"], "deny")
            self.assertEqual(body["provider"]["max_price"]["prompt"], 0)
            self.assertTrue(body["response_format"]["json_schema"]["strict"])
            self.assertNotIn("models", body)
        self.assertEqual(bodies[1]["provider"]["only"], ["baidu"])

    def test_account_429_with_metadata_reset_never_switches_models(self):
        budget = Mock()
        provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
        busy = Mock(status_code=429, headers={})
        busy.json.return_value = {"error": {"metadata": {"limit_source": "openrouter_free_tier_per_minute",
                                                       "headers": {"X-RateLimit-Reset": "1792448640000"}}}}
        with patch("app.providers.openrouter_answers.requests.post", return_value=busy) as post, \
                patch("app.providers.openrouter_answers.time.time", return_value=1792448600):
            with self.assertRaises(AnswerRateLimitError) as failure:
                provider._complete("citewise_evidence_assessment", {}, "system", "user")
        self.assertEqual(failure.exception.retry_after, "40")
        post.assert_called_once()
        self.assertEqual(failure.exception.limit_source, "openrouter_free_tier_per_minute")
        budget.cool_down.assert_called_once_with(40, limit_source="openrouter_free_tier_per_minute")

    def test_truncated_generation_is_neither_retried_nor_returned(self):
        provider = OpenRouterAnswerProvider("key", ModelRouter())
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"finish_reason": "length", "message": {"content":
            '{"status":"answered","text":"A claim [c1].","evidence_ids":["c1"]}'}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
            with self.assertRaisesRegex(ValueError, "output limit"):
                provider.generate("What was found?", [Evidence("c1", "A claim.")])
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 3500)

    def test_local_budget_or_zero_retention_blocks_fallback(self):
        budget = Mock()
        budget.reserve.side_effect = AnswerRateLimitError("15")
        provider = OpenRouterAnswerProvider("key", ModelRouter(), budget=budget)
        with patch("app.providers.openrouter_answers.requests.post") as post:
            with self.assertRaises(AnswerRateLimitError):
                provider._complete("citewise_evidence_assessment", {}, "system", "user")
        post.assert_not_called()

        busy = Mock(status_code=429, headers={})
        busy.json.return_value = {"error": {"metadata": {"limit_source": "upstream_provider_shared_pool"}}}
        provider = OpenRouterAnswerProvider("key", ModelRouter(), require_zdr=True)
        with patch("app.providers.openrouter_answers.requests.post", return_value=busy) as post:
            with self.assertRaises(AnswerRateLimitError):
                provider._complete("citewise_evidence_assessment", {}, "system", "user")
        post.assert_called_once()
        self.assertTrue(post.call_args.kwargs["json"]["provider"]["zdr"])

    def test_structured_timeout_and_alternate_failure_have_two_attempts_at_most(self):
        unavailable = Mock(status_code=503, headers={})
        unavailable.raise_for_status.side_effect = requests.HTTPError("unavailable")
        for primary_error in (requests.Timeout(), unavailable):
            with self.subTest(primary_error=type(primary_error).__name__):
                provider = OpenRouterAnswerProvider("key", ModelRouter())
                with patch("app.providers.openrouter_answers.requests.post",
                           side_effect=[primary_error, unavailable]) as post:
                    with self.assertRaises(requests.HTTPError):
                        provider._complete("citewise_claim_verification", {}, "system", "user")
                self.assertEqual([call.kwargs["json"]["model"] for call in post.call_args_list],
                                 [VERIFICATION_MODEL, FAST_MODEL])

    def test_bad_structured_response_does_not_trigger_failover(self):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "malformed"}}]}
        with patch("app.providers.openrouter_answers.requests.post", return_value=response) as post:
            with self.assertRaises(ValueError):
                OpenRouterAnswerProvider("key", ModelRouter()).assess("Question?", [Evidence("c1", "Evidence.")])
        post.assert_called_once()

    def test_retry_transient_response_then_success(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10, structured_fallback=""))
        busy = Mock(status_code=503, headers={"retry-after": "0.5"})
        good = Mock(status_code=200)
        good.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
        with patch("app.providers.openrouter_answers.requests.post", side_effect=[busy, good]) as post, \
                patch("app.providers.openrouter_answers.time.sleep") as sleep:
            result = provider._complete("citewise_claim_verification", {}, "system", "user")
        self.assertTrue(result["ok"])
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_retry_timeout_is_bounded(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10, structured_fallback=""))
        with patch("app.providers.openrouter_answers.requests.post", side_effect=requests.Timeout) as post, \
                patch("app.providers.openrouter_answers.time.sleep") as sleep, \
                patch("app.providers.openrouter_answers.random.uniform", return_value=0.25):
            with self.assertRaises(requests.Timeout):
                provider._complete("citewise_claim_verification", {}, "system", "user")
        self.assertEqual(post.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.25, 2.25])

    def test_long_retry_after_fails_without_premature_retry(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10))
        busy = Mock(status_code=429, headers={"retry-after": "90"})
        with patch("app.providers.openrouter_answers.requests.post", return_value=busy) as post, \
                patch("app.providers.openrouter_answers.time.sleep") as sleep:
            with self.assertRaises(AnswerRateLimitError) as caught:
                provider._complete("citewise_claim_verification", {}, "system", "user")
        self.assertEqual(caught.exception.retry_after, "90")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_non_transient_error_is_not_retried(self):
        provider = OpenRouterAnswerProvider("test-key", ModelRouter(timeout=10))
        bad = Mock(status_code=400, headers={})
        bad.raise_for_status.side_effect = requests.HTTPError("invalid request")
        with patch("app.providers.openrouter_answers.requests.post", return_value=bad) as post, \
                patch("app.providers.openrouter_answers.time.sleep") as sleep:
            with self.assertRaises(requests.HTTPError):
                provider._complete("citewise_claim_verification", {}, "system", "user")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_retry_after_accepts_http_date(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime
        value = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=15))
        self.assertGreater(_retry_delay(value), 10)
