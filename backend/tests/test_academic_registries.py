import unittest
from unittest.mock import Mock
import pymupdf

from app.services.academic_registries import pubmed, datacite
from app.services.paper_discovery import PaperDiscoveryService, normalize_doi, ProviderError
from app.services.publication_identity import verify_pdf_identity


class RegistryTests(unittest.TestCase):
    title = "Measurement of mountain snow depth"
    doi = "10.1111/snow"

    def setUp(self):
        self.api = Mock()
        self.xml = f"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
        <Article><ArticleTitle>{self.title}</ArticleTitle><AuthorList><Author><LastName>Rivera</LastName>
        <ForeName>Ana</ForeName></Author></AuthorList><Journal><Title>Snow</Title><JournalIssue>
        <PubDate><Year>2020</Year></PubDate></JournalIssue></Journal><PublicationTypeList>
        <PublicationType>Journal Article</PublicationType></PublicationTypeList></Article></MedlineCitation>
        <PubmedData><ArticleIdList><ArticleId IdType="doi">{self.doi}</ArticleId>
        <ArticleId IdType="pmc">PMC789</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
        </PubmedArticleSet>""".encode()
        self.api.get.return_value = self.xml
        self.api.json.return_value = {"esearchresult": {"idlist": ["123"], "count": "1"}}

    def test_pmid_pmc_and_doi_require_returned_identifier(self):
        for key in ["PMID:123", "PMC789", self.doi]:
            record = pubmed(self.api, key, "doi", 0, 1, normalize_doi)[0][0]
            self.assertEqual((record["pmid"], record["pmcid"], record["authors"], record["year"]),
                             ("123", "PMC789", ["Ana Rivera"], 2020))
        for key in ["PMID:789", "PMC123", "10.1111/another"]:
            self.assertEqual(pubmed(self.api, key, "doi", 0, 1, normalize_doi)[0], [])

    def test_author_search_supports_given_name_first(self):
        pubmed(self.api, "author: Ana Rivera", "author", 0, 3, normalize_doi)
        query = self.api.json.call_args.args[2]["term"]
        self.assertIn('"Ana Rivera"[Author]', query)
        self.assertIn('"Rivera Ana"[Full Author Name]', query)

    def test_retraction_excludes_pmid_and_doi(self):
        self.api.get.return_value = self.xml.replace(b"Journal Article", b"Retracted Publication")
        record = pubmed(self.api, "PMID:123", "topic", 0, 1, normalize_doi)[0][0]
        self.assertEqual((record["status"], record["pmid"], record["doi"]), ("retracted", "123", self.doi))

    def test_datacite_resource_classification(self):
        attrs = dict(doi=self.doi, state="findable", titles=[{"title": self.title}],
                     creators=[{"givenName": "Ana", "familyName": "Rivera"}], publisher="Repository", publicationYear=2020)
        self.api.json.return_value = {"data": {"id": self.doi, "attributes": attrs}}
        for kind in ("Report", "Dataset", "Software", "Text", "Preprint", "JournalArticle"):
            attrs["types"] = {"resourceTypeGeneral": kind}
            records = datacite(self.api, self.doi, "doi", 0, 1, normalize_doi)[0]
            self.assertEqual(bool(records), kind in {"Preprint", "JournalArticle"})
        attrs["state"] = "registered"
        self.assertFalse(datacite(self.api, self.doi, "doi", 0, 1, normalize_doi)[0])

    def test_pdf_controls_use_real_matcher(self):
        service = PaperDiscoveryService(self.api)
        def lookup(provider, path, params=None):
            if provider == "pubmed":
                return {"esearchresult": {"idlist": ["123"], "count": "1"}}
            if path == "/works":
                return {"message": {"items": []}} if provider == "crossref" else {"results": []}
            if path == "/dois":
                return {"data": []}
            raise ProviderError("HTTP 404")
        self.api.json.side_effect = lookup
        original_get = self.api.get.return_value
        self.api.get.side_effect = lambda provider, *args: original_get if provider == "pubmed" else b'<feed xmlns="http://www.w3.org/2005/Atom"/>'
        for first, second, expected in [
            (self.title + "\nAna Rivera", self.doi, "verified_published"),
            (self.title + "\nAna Rivera\nPMID:123", "", "verified_published"),
            (self.title + "\nAna Rivera", "10.1111/invented", "not_verified"),
            ("Tidal forces in the North Atlantic\nAna Rivera", self.doi, "not_verified"),
            ("Notice to residents\nThe main entrance will be closed tomorrow.", "", "not_verified"),
        ]:
            with self.subTest(first=first, second=second), pymupdf.open() as doc:
                doc.new_page().insert_text((40, 50), first)
                if second:
                    doc.new_page().insert_text((40, 50), second)
                self.assertEqual(verify_pdf_identity(doc.tobytes(), service)["status"], expected)
