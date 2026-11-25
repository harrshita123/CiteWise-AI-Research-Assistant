import unittest
from unittest.mock import patch, MagicMock
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.dependencies import get_current_user
from app.models import UserResponse
from app.routers.discovery import router

from app.services.paper_discovery import AcademicClient, PaperDiscoveryService, ProviderError, normalize_arxiv, normalize_doi, search_intent


class FakeClient:
    def __init__(self):
        self.item = {"DOI": "10.1234/example", "title": ["A Study"], "type": "journal-article",
                     "author": [{"given": "Ada", "family": "Smith"}], "publisher": "Publisher",
                     "published": {"date-parts": [[2020]]}}
        self.openalex = {"results": []}
        self.arxiv = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    def json(self, provider, path, params=None):
        if provider == "pubmed": return {"esearchresult": {"count": "0", "idlist": []}}
        if provider == "datacite":
            if path == "/dois": return {"data": [], "meta": {"total": 0}}
            raise ProviderError("HTTP 404")
        if provider == "openalex": return self.openalex
        if path == "/works": return {"message": {"items": [self.item]}}
        if path.endswith("notice"): return {"message": {"update-to": [{"type": "retraction"}]}}
        if path.endswith("example"): return {"message": self.item}
        raise ProviderError("not found")
    def get(self, provider, path, params=None):
        return self.arxiv


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.service = PaperDiscoveryService(self.client)

    def test_identifier_validation(self):
        self.assertEqual(normalize_doi("https://doi.org/10.1234/Example"), "10.1234/example")
        self.assertIsNone(normalize_doi("10.12/bad"))
        self.assertEqual(normalize_arxiv("arxiv:2005.11401v1"), "2005.11401v1")
        with self.assertRaises(ValueError): self.service.search("doi:broken")

    def test_registered_record_and_duplicate_openalex(self):
        self.client.openalex = {"results": [{"doi": "https://doi.org/10.1234/example", "title": "A Study", "is_retracted": False}]}
        result = self.service.search("study")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["status"], "registry_record")
        self.assertEqual(result["results"][0]["authors"], ["Ada Smith"])

    def test_mismatch_and_retraction_are_excluded(self):
        self.assertIsNone(self.service._crossref("10.1234/example", "Different title"))
        self.assertIsNone(self.service._crossref("10.1234/example", "A Study", ["Someone Else"]))
        self.client.item["updated-by"] = [{"DOI": "10.1234/notice"}]
        self.assertEqual(self.service.search("study")["results"][0]["status"], "retracted")
        self.client.item.pop("updated-by")
        self.client.openalex = {"results": [{"doi": "https://doi.org/10.1234/example", "is_retracted": True}]}
        self.assertEqual(self.service.search("study")["results"][0]["status"], "retracted")

    def test_official_preprint_and_withdrawal(self):
        self.client.arxiv = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>https://arxiv.org/abs/2005.11401v1</id><title>Test Preprint</title>
          <published>2020-05-23T00:00:00Z</published><author><name>A Smith</name></author></entry></feed>'''
        result = self.service.search("arxiv:2005.11401")
        self.assertEqual(result["results"][0]["status"], "verified_preprint")
        self.client.arxiv = self.client.arxiv.replace(b"Test Preprint", b"Withdrawn Preprint")
        self.assertEqual(self.service.search("arxiv:2005.11401")["results"][0]["status"], "retracted")

    def test_provider_failure_does_not_make_up_results(self):
        with patch.object(self.client, "json", side_effect=ProviderError("timeout")):
            result = self.service.search("test")
        self.assertEqual(result["results"], [])
        self.assertIn("crossref", result["provider_errors"])

    def test_openalex_keyless_and_backend_bearer_auth(self):
        response = MagicMock(status_code=200, headers={})
        response.iter_content.return_value = [b'{}']
        response.__enter__.return_value = response
        with patch("app.services.paper_discovery.requests.get", return_value=response) as request:
            AcademicClient().get("openalex", "/works")
            self.assertNotIn("Authorization", request.call_args.kwargs["headers"])
            AcademicClient("private-test-key").get("openalex", "/works")
            self.assertEqual(request.call_args.kwargs["headers"]["Authorization"], "Bearer private-test-key")
            self.assertNotIn("api_key", request.call_args.kwargs["params"])

    def test_rate_limit_enforces_cooldown_without_retry(self):
        response = MagicMock(status_code=429, headers={"Retry-After": "60"})
        response.__enter__.return_value = response
        client = AcademicClient()
        with patch("app.services.paper_discovery.requests.get", return_value=response) as request:
            for _ in range(2):
                with self.assertRaisesRegex(ProviderError, "rate limited"):
                    client.get("arxiv", "/api/query")
            self.assertEqual(request.call_count, 1)
        self.assertEqual(client.health()["arxiv"]["status"], "rate limited")

    def test_crossref_contact_is_sent_only_when_configured(self):
        response = MagicMock(status_code=200, headers={})
        response.iter_content.return_value = [b'{}']
        response.__enter__.return_value = response
        with patch("app.services.paper_discovery.requests.get", return_value=response) as request:
            AcademicClient(crossref_email="research@example.org").get("crossref", "/works")
        self.assertEqual(request.call_args.kwargs["params"]["mailto"], "research@example.org")

    def test_arxiv_spacing_and_duplicate_cache(self):
        response = MagicMock(status_code=200, headers={})
        response.iter_content.return_value = [b"<feed/>"]
        response.__enter__.return_value = response
        client = AcademicClient()
        with patch("app.services.paper_discovery.requests.get", return_value=response) as request, \
             patch("app.services.paper_discovery.time.monotonic", return_value=0), \
             patch("app.services.paper_discovery.time.sleep") as sleep:
            client.get("arxiv", "/api/query", {"id_list": "1234.12345"})
            client.get("arxiv", "/api/query", {"id_list": "1234.12345"})
            client.get("arxiv", "/api/query", {"id_list": "1234.12346"})
            self.assertEqual(request.call_count, 2)
            sleep.assert_called_once_with(3.0)

    def test_invalid_key_is_reported_as_authentication_error(self):
        response = MagicMock(status_code=403, headers={})
        response.__enter__.return_value = response
        client = AcademicClient("invalid-key")
        with patch("app.services.paper_discovery.requests.get", return_value=response):
            with self.assertRaisesRegex(ProviderError, "authentication error"):
                client.get("openalex", "/works")
        self.assertEqual(client.health()["openalex"]["status"], "authentication error")

    def test_arbitrary_outbound_host_is_not_permitted(self):
        with self.assertRaises(ProviderError): AcademicClient().get("http://127.0.0.1", "/")
        with self.assertRaises(ProviderError): AcademicClient().get("arxiv", "//evil.example")

    def test_exact_arxiv_title_beats_crossref_even_when_result_limit_is_one(self):
        title = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
        self.client.arxiv = f'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <id>https://arxiv.org/abs/2005.11401v1</id><title>{title}</title>
          <published>2020-05-23T00:00:00Z</published><author><name>Patrick Lewis</name></author></entry></feed>'''.encode()
        with patch.object(self.client, "get", wraps=self.client.get) as query:
            result = self.service.search(title, limit=1)
        self.assertEqual(result["results"][0]["arxiv_id"], "2005.11401v1")
        self.assertEqual(result["results"][0]["status"], "verified_preprint")
        self.assertIn('ti:"', query.call_args.args[2]["search_query"])

    def test_intent_and_provider_error_are_explicit(self):
        self.assertEqual(search_intent("author: Patrick Lewis"), "author")
        self.assertEqual(search_intent("machine learning"), "topic")
        with patch.object(self.client, "get", side_effect=ProviderError("HTTP 429")):
            result = self.service.search("arxiv:2005.11401")
        self.assertEqual(result["provider_errors"]["arxiv"], "HTTP 429")
        self.assertEqual(result["results"], [])


class DiscoveryRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: UserResponse(id="alice", email="a@example.com", role="user")
        app.state.discovery_service = PaperDiscoveryService(FakeClient())
        app.state.discovery_records = type("Records", (), {"update_one": AsyncMock()})()
        app.state.discovery_audit = type("Audit", (), {"insert_one": AsyncMock()})()
        self.client = TestClient(app)

    def test_search_checks_record_before_saving(self):
        response = self.client.get("/api/v1/discovery", params={"query": "10.1234/example"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["doi"], "10.1234/example")
        self.client.app.state.discovery_records.update_one.assert_awaited_once()

    def test_bad_identifier_never_saves_a_record(self):
        response = self.client.get("/api/v1/discovery", params={"query": "doi:broken"})
        self.assertEqual(response.status_code, 422)
        self.client.app.state.discovery_records.update_one.assert_not_awaited()
