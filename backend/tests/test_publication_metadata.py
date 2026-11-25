import unittest
from unittest.mock import Mock

from app.services.paper_discovery import PaperDiscoveryService, rank_record
from app.services.publication_metadata import normalized_title, metadata_conflict, names_agree
from app.services.publication_identity import matched_authors, verify_identity_input
from test_paper_discovery import FakeClient


class MetadataTests(unittest.TestCase):
    def test_name_and_title_normalization(self):
        self.assertTrue(names_agree("García, Elena", "E. García"))
        self.assertFalse(names_agree("Evan Smith", "Ellen Smith"))
        self.assertEqual(matched_authors(["Ana Rivera", "The Atlas Collaboration"], "Ana Rivera¹ and The Atlas Collaboration"), ["Ana Rivera","The Atlas Collaboration"])
        self.assertEqual(normalized_title("A ﬁeld: micro-\nscope"), normalized_title("A field microscope"))
        self.assertNotEqual(normalized_title("Models of x ≤ y"),normalized_title("Models of x ≥ y"))

    def test_same_title_different_authors_and_dates(self):
        base = {"title":"A Shared Title", "authors":["Jane Kim"],"year":2019}
        self.assertTrue(metadata_conflict(base, {**base,"authors":["John Kim"]}))
        self.assertTrue(metadata_conflict(base, {**base,"year":2024}))
        self.assertFalse(metadata_conflict({**base,"publication_years":[2019,2021]}, {**base,"year":2021}))

    def test_equal_titles_rank_by_evidence_without_merging(self):
        title = "Attention Is All You Need"
        registry = dict(title=title, id="doi:one", authors=[], sources=["DataCite"], status="registry_record")
        official = dict(registry, id="arxiv:two", sources=["arXiv"], status="verified_preprint", authors=["A. Author"],year=2017)
        self.assertGreater(rank_record(official,title,"title"), rank_record(registry,title,"title"))

    def test_same_identifier_disagreement_is_visible(self):
        api = FakeClient()
        api.openalex = {"results":[{"doi":"10.1234/example","title":"A Study","authorships":[{"author":{"display_name":"Other Person"}}]}]}
        self.assertEqual(PaperDiscoveryService(api).search("A Study")["results"][0]["status"],"conflicting")

    def test_equal_titles_are_ambiguous_without_a_version_link(self):
        discovery = Mock()
        base = {"title":"Ways of counting stars","authors":["Lee Kim"],"status":"verified_published","sources":["PubMed"]}
        discovery.search.return_value = {"results":[dict(base,id="one"),dict(base,id="two")]}
        first = "Ways of counting stars\nLee Kim¹\nAbstract\nWe describe the experiment."
        result = verify_identity_input(dict(first=first,text=first,metadata_title=base["title"]),discovery)
        self.assertEqual(result["status"],"conflicting")

    def test_same_surname_elsewhere_in_document_is_insufficient(self):
        self.assertEqual(matched_authors(["Ana Rivera"], "This work extends the earlier Rivera model."),[])

    def test_versions_group_only_with_provider_link_and_matching_author(self):
        api = FakeClient()
        api.arxiv = b'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:a="http://arxiv.org/schemas/atom"><entry>
          <id>https://arxiv.org/abs/2101.12345v2</id><title>A Study</title><a:doi>10.1234/example</a:doi>
          <author><name>A. Smith</name></author></entry></feed>'''
        search = PaperDiscoveryService(api)
        linked = search.search("A Study")["results"]
        self.assertEqual(len(linked), 2)
        self.assertEqual({r.get("version_group") for r in linked}, {"arxiv:2101.12345"})
        api.arxiv = api.arxiv.replace(b"A. Smith", b"Different Person")
        separate = search.search("A Study")["results"]
        self.assertIsNone(next(r for r in separate if r.get("doi")).get("version_group"))

    def test_datacite_repository_link_must_match_author_and_edition(self):
        api = FakeClient()
        doi = "10.1234/registry"
        attrs = dict(doi=doi, state="findable", titles=[{"title":"A Study"}],
                     creators=[{"name":"Ada Smith"}], publicationYear=2020, publisher="Repository",
                     types={"resourceTypeGeneral":"Preprint"}, url="https://arxiv.org/abs/2101.12345")
        original = api.json
        api.json = lambda provider,path,params=None: (
            {"data":[{"id":doi,"attributes":attrs}]} if provider=="datacite" else original(provider,path,params))
        api.arxiv = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2101.12345v2</id>
        <title>A Study</title><author><name>Ada Smith</name></author></entry></feed>'''
        def group():
            records = PaperDiscoveryService(api).search("A Study")["results"]
            return next(r for r in records if r.get("doi")==doi).get("version_group")
        self.assertEqual(group(),"arxiv:2101.12345")
        attrs["url"] += "v1"
        self.assertIsNone(group())
        attrs["url"]="https://arxiv.org/abs/2101.12345"
        attrs["creators"]=[{"name":"Different Person"}]
        self.assertIsNone(group())

    def test_proceedings_header_breaks_tie_with_earlier_preprint(self):
        title = "A Study of Mountain Snow"
        first = f"Proceedings of a Research Conference, 2022\n{title}\nAna Rivera\nAbstract\nSome results."
        author = {"title": title, "authors": ["Ana Rivera"], "sources": ["Test"]}
        published = dict(author, id="journal", doi="10.1234/conf", year=2022,
                         publication_type="proceedings-article", status="verified_published")
        preprint = dict(author, id="preprint", year=2021, status="verified_preprint")
        service = Mock()
        service.search.return_value = {"results": [preprint, published]}
        result = verify_identity_input(dict(first=first, text=first, metadata_title=title), service)
        self.assertEqual(result["doi"], "10.1234/conf")

    def test_first_page_doi_beats_identifiers_in_body(self):
        first = "A Study of Mountain Snow\nAna Rivera\nDOI: 10.1234/snow"
        service = Mock()
        service.search.return_value = {"results": []}
        verify_identity_input(dict(first=first, text=first + "\n10.1234/a 10.1234/b 10.1234/c"), service)
        service.search.assert_called_once_with("10.1234/snow", 0, 1)

    def test_apostrophe_glyphs_match_while_operators_remain_distinct(self):
        self.assertEqual(normalized_title("A Model’s Output"), normalized_title("A Model's Output"))
        self.assertNotEqual(normalized_title("Output x ≥ 2"), normalized_title("Output x ≤ 2"))

    def test_openalex_published_location_is_an_indexed_edition(self):
        work = {"id": "https://openalex.org/W123", "title": "A Study", "publication_year": 2022,
                "type": "conference-paper", "is_retracted": False, "locations": [
                    {"version": "publishedVersion", "landing_page_url": "https://example.org/proceedings/paper",
                     "source": {"type": "conference", "display_name": "Research Conference"}}]}
        record = PaperDiscoveryService._published_location(work, ["Ana Rivera"])
        self.assertEqual(record["status"], "registry_record")
        self.assertIsNone(record["doi"])
        work["locations"][0]["version"] = "submittedVersion"
        self.assertIsNone(PaperDiscoveryService._published_location(work, ["Ana Rivera"]))

    def test_explicit_subtitle_is_optional_for_same_doi_and_byline(self):
        original = {"title": "Repeated Measurements of Coastal Wind", "authors": ["Anna Rossi"],
                    "doi": "10.1234/field", "year": 2020, "venue": "Research Letters"}
        subtitle = dict(original, title=original["title"] + ": A longitudinal study")
        self.assertFalse(metadata_conflict(original, subtitle))
        self.assertTrue(metadata_conflict(original, dict(subtitle, authors=["John Rossi"])))
        self.assertTrue(metadata_conflict(original, dict(subtitle, doi="10.1234/different")))
        self.assertTrue(metadata_conflict(original, dict(subtitle, year=2025)))
