import unittest
from unittest.mock import Mock

import pymupdf

from app.services.publication_identity import extract_identity_input, extracted_paper_metadata, verify_pdf_identity


def pdf(text):
    with pymupdf.open() as doc:
        doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 550, 750), text)
        return doc.tobytes()


class PublicationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.record = {"title": "A Study of Unusual Materials", "authors": ["Ada Miller"],
                       "status": "verified_published", "doi": "10.1234/example",
                       "sources": ["Crossref"], "landing_url": "https://doi.org/10.1234/example"}
        self.service.search.return_value = {"results": [self.record], "provider_errors": {}}

    def test_layout_recovers_a_title_split_across_centered_blocks(self):
        with pymupdf.open() as document:
            page = document.new_page()
            for y, title_line in [(98, "Measuring Seasonal Changes"),
                                  (125, "Across Alpine Valleys")]:
                width = pymupdf.get_text_length(title_line, fontsize=18)
                page.insert_text(((page.rect.width - width) / 2, y), title_line, fontsize=18)
            page.insert_text((65, 155), "Ana Rivera")
            page.insert_text((35, 230), "arXiv:2101.12345v1", rotate=90, fontsize=20)
            metadata = extracted_paper_metadata(extract_identity_input(document.tobytes()))
        self.assertEqual(metadata["title"], "Measuring Seasonal Changes Across Alpine Valleys")

    def test_unlabelled_abstract_after_a_split_title_and_byline(self):
        title = "Field Measurements of Coastal Sediment"
        summary = " ".join(["We measured changes in fine sediment across the bay."] * 8)
        with pymupdf.open() as document:
            document.set_metadata({"title": title, "author": "Ada Miller"})
            page = document.new_page()
            page.insert_text((50, 75), "Field Measurements of Coastal", fontsize=18)
            page.insert_text((50, 102), "Sediment", fontsize=18)
            page.insert_text((50, 125), "Ada Miller", fontsize=12)
            page.insert_textbox(pymupdf.Rect(50, 145, 550, 270), summary, fontsize=11)
            page.insert_text((50, 225), "1. Introduction", fontsize=11)
            page.insert_textbox(pymupdf.Rect(50, 242, 285, 470), "Details from the main text. " * 8, fontsize=10)
            metadata = extracted_paper_metadata(extract_identity_input(document.tobytes()))
        self.assertEqual(metadata["abstract"], summary)

    def test_matching_identifier_title_and_author(self):
        result = verify_pdf_identity(pdf("A Study of Unusual Materials\nAda Miller\n10.1234/example"), self.service)
        self.assertEqual(result["status"], "verified_published")
        self.service.search.assert_called_once_with("10.1234/example", 0, 1)

    def test_identifier_of_a_different_paper_is_not_verification(self):
        result = verify_pdf_identity(pdf("A Different Investigation of Materials\nAda Miller\n10.1234/example"), self.service)
        self.assertEqual(result["status"], "not_verified")

    def test_similar_title_with_different_author_is_not_verified(self):
        result = verify_pdf_identity(pdf("A Study of Unusual Materials\nJohn Smith\n10.1234/example"), self.service)
        self.assertEqual(result["status"], "not_verified")

    def test_scanned_or_missing_identifiers_remain_unverified(self):
        for text in ("", "A lengthy ordinary document with no identifier or academic publication record."):
            self.assertEqual(verify_pdf_identity(pdf(text), self.service)["status"], "not_verified")
        self.service.search.assert_not_called()

    def test_provider_failure_does_not_fabricate_identity(self):
        self.service.search.return_value = {"results": [], "provider_errors": {"crossref": "timeout"}}
        result = verify_pdf_identity(pdf("A Study of Unusual Materials\nAda Miller\n10.1234/example"), self.service)
        self.assertEqual(result["status"], "verification_unavailable")
        self.assertIn("could not be checked", result["reason"])

    def test_multiple_identifiers_do_not_reuse_another_record(self):
        result = verify_pdf_identity(pdf("A Different Study of Materials\nAda Miller\n10.1234/example\n10.5678/other"), self.service)
        self.assertEqual(result["status"], "not_verified")
        self.assertEqual(self.service.search.call_count, 2)

    def test_title_metadata_search_requires_first_page_title_and_author(self):
        with pymupdf.open() as doc:
            doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 550, 750),
                "A Study of Unusual Materials\nAda Miller\nIndependent research on material properties.")
            doc.set_metadata({"title": "A Study of Unusual Materials"})
            content = doc.tobytes()
        self.assertEqual(verify_pdf_identity(content, self.service)["status"], "verified_published")
        self.service.search.assert_called_once_with("A Study of Unusual Materials", 0, 5)

    def test_identifier_on_second_page_is_checked(self):
        with pymupdf.open() as doc:
            doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 550, 750),
                "A Study of Unusual Materials\nAda Miller\nIndependent research on material properties.")
            doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 550, 750), "DOI: 10.1234/example")
            content = doc.tobytes()
        self.assertEqual(verify_pdf_identity(content, self.service)["status"], "verified_published")

    def test_identifier_in_references_is_not_searched(self):
        with pymupdf.open() as doc:
            doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 550, 750),
                "An ordinary document with a bibliography and enough words to have readable content.")
            doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 550, 750), "References\n10.1234/example")
            content = doc.tobytes()
        self.assertEqual(verify_pdf_identity(content, self.service)["status"], "not_verified")
        self.service.search.assert_not_called()

    def test_visible_byline_and_structured_summary_override_missing_or_bogus_metadata(self):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 75), "A Study of Unusual Materials", fontsize=19)
            page.insert_text((50, 105), "Ada Miller, PhD; Luis Torres, MD")
            page.insert_textbox(pymupdf.Rect(50, 140, 550, 390), "\n".join([
                "OBJECTIVE To estimate how material composition affects performance at different temperatures.",
                "RESULTS The two measurements agreed in the observed experimental conditions.",
                "CONCLUSIONS The report identifies an association between composition and the outcome.",
            ]))
            page.insert_text((50, 445), "Research Letters. 2024;10(2):21. doi:10.1234/example")
            page.insert_text((50, 470), "Published online April 3, 2024.")
            doc.set_metadata({"author": "Publisher Corporation"})
            data = doc.tobytes()
        metadata = extracted_paper_metadata(extract_identity_input(data))
        self.assertEqual(metadata["authors"], ["Ada Miller", "Luis Torres"])
        self.assertEqual(metadata["doi"], "10.1234/example")
        self.assertEqual((metadata["year"], metadata["venue"]), (2024, "Research Letters"))
        self.assertIn("CONCLUSIONS", metadata["abstract"])
        self.assertNotIn("Published online", metadata["abstract"])
        self.record["authors"] = ["A. Miller", "Luis Torres"]
        self.assertEqual(verify_pdf_identity(data, self.service)["status"], "verified_published")
        self.record["authors"] = ["Ada Miller", "Another Author"]
        self.assertEqual(verify_pdf_identity(data, self.service)["status"], "not_verified")

    def test_wrapped_doi_in_publication_footer_beats_inline_reference(self):
        identity = {"first": "Materials Research | https://doi.org/10.1234/\nexample\u2003March 6, 2024\n"
                             "Previous work is cited at doi:10.1234/other.\n"
                             "References\nDOI: 10.9999/citation"}
        metadata = extracted_paper_metadata(identity)
        self.assertEqual((metadata["doi"], metadata["venue"], metadata["year"]),
                         ("10.1234/example", "Materials Research", 2024))

    def test_repository_cover_uses_manuscript_title_byline_and_identifier(self):
        from app.services.document_classification import DocumentType, classify_document
        with pymupdf.open() as doc:
            cover = doc.new_page()
            cover.insert_textbox(pymupdf.Rect(50, 50, 540, 300),
                                 "University Repository\nThis is a repository copy.\nAuthor accepted manuscript\n"
                                 "A title supplied by the cover\nDOI: 10.9999/unrelated")
            main = doc.new_page()
            main.insert_text((50, 70), self.record["title"], fontsize=20)
            main.insert_text((50, 100), "Ada Miller")
            main.insert_textbox(pymupdf.Rect(50, 130, 540, 500),
                                "DOI: 10.1234/example\nAbstract\nA series of material samples were measured.\n"
                                "Introduction\nThe study used different materials.\nMethods\nMeasurements were repeated.")
            doc.set_metadata({"title": "A title supplied by the cover", "author": "University Repository"})
            identity = extract_identity_input(doc.tobytes(), include_tail=True)
            result = verify_pdf_identity(doc.tobytes(), self.service)
        meta = extracted_paper_metadata(identity)
        self.assertEqual(identity["front_page"], 2)
        self.assertEqual(meta["title"], self.record["title"])
        self.assertEqual(meta["authors"], ["Ada Miller"])
        self.assertEqual(meta["version"], "accepted_manuscript")
        self.assertEqual(classify_document(identity)[0], DocumentType.RESEARCH_PAPER)
        self.assertEqual(result["status"], "verified_published")
        self.assertEqual(result["verification_scope"], "publication_record")
        self.service.search.assert_called_once_with("10.1234/example", 0, 1)

    def test_title_only_match_does_not_choose_preprint_from_shared_version_group(self):
        base = {**self.record, "version_group": "arxiv:2101.12345"}
        self.service.search.return_value = {"results": [
            dict(base, id="journal"), dict(base, id="preprint", arxiv_id="2101.12345", status="verified_preprint")
        ]}
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 70), self.record["title"], fontsize=20)
            page.insert_text((50, 100), "Ada Miller")
            page.insert_text((50, 140), "Experimental observations describe material performance.")
            result = verify_pdf_identity(doc.tobytes(), self.service)
        self.assertEqual(result["status"], "conflicting")

    def test_working_paper_cover_and_byline_share_a_text_block(self):
        from app.services.document_classification import DocumentType, classify_document
        title = "Measuring Trade Across Regions"
        with pymupdf.open() as doc:
            doc.new_page().insert_textbox(pymupdf.Rect(40, 30, 550, 400),
                "INSTITUTION WORKING PAPER SERIES\nMEASURING TRADE ACROSS REGIONS\nAda Miller\nWorking Paper 9876")
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(40, 30, 550, 200),
                f"{title}\nAda Miller\nWorking Paper No. 9876\nJune 2024")
            page.insert_textbox(pymupdf.Rect(40, 220, 550, 700),
                "Abstract\nThe study estimates regional trade costs.\nIntroduction\n"
                + "Households exchange materials across regions under different economic conditions. " * 25)
            identity = extract_identity_input(doc.tobytes(), include_tail=True)
        meta = extracted_paper_metadata(identity)
        self.assertEqual(identity["front_page"], 2)
        self.assertEqual((meta["title"], meta["authors"], meta["year"]), (title, ["Ada Miller"], 2024))
        self.assertEqual(classify_document(identity)[0], DocumentType.RESEARCH_PAPER)

    def test_abstract_continues_without_including_publication_sidebar(self):
        identity = {
            "first": "Full Title of the Study\nAbstract\nBackground\nWe studied the samples.\n"
                     "Results\nInitial observations agreed.\nCitation: Example Journal.\nPublished: June 12, 2021\n"
                     "Example Journal | https://doi.org/10.1234/example",
            "next_page_text": "Further observations also agreed.\nConclusions\nThe groups were similar.\n"
                              "Author summary\nThis is a different section.",
        }
        meta = extracted_paper_metadata(identity)
        self.assertEqual(meta["abstract"], "Background We studied the samples. Results Initial observations agreed. "
                         "Further observations also agreed. Conclusions The groups were similar.")
        self.assertEqual((meta["venue"], meta["year"]), ("Example Journal", 2021))

    def test_printed_venue_and_year_distinguish_published_edition(self):
        title = "Characterizing Spatial Variation Over Time"
        published = {**self.record, "title": title, "authors": ["Ada Miller", "Astrid Vega"],
                     "venue": "Findings of the Coastal Research Meeting 2023", "year": 2023}
        preprint = {**published, "doi": None, "arxiv_id": "2301.12345", "venue": None,
                    "status": "verified_preprint", "id": "arxiv:2301.12345"}
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 50), published["venue"] + ", pages 22–29")
            page.insert_text((50, 120), title, fontsize=18)
            page.insert_text((50, 150), "Ada Miller")
            page.insert_text((127, 150), "Astrid Vega")
            page.insert_text((90, 166), "Spatial Data Lab")
            page.insert_text((65, 180), "first.last@example.org")
            page.insert_textbox(pymupdf.Rect(50, 220, 525, 450),
                                "Abstract\nWe compared yearly observations at several different coastal stations.")
            content = doc.tobytes()
        meta = extracted_paper_metadata(extract_identity_input(content))
        self.assertEqual(meta["authors"], published["authors"])
        self.assertEqual((meta["venue"], meta["year"]), (published["venue"], 2023))
        self.service.search.return_value = {"results": [preprint, published]}
        self.assertEqual(verify_pdf_identity(content, self.service)["doi"], published["doi"])

        conflicting = {**published, "status": "conflicting"}
        self.service.search.return_value = {"results": [preprint, conflicting]}
        self.assertEqual(verify_pdf_identity(content, self.service)["status"], "conflicting")

    def test_multiple_editions_without_corroborating_printed_details_are_explained(self):
        other = {**self.record, "doi": None, "id": "preprint", "status": "verified_preprint"}
        self.service.search.return_value = {"results": [self.record, other]}
        from app.services.publication_identity import verify_identity_input
        first = "A Study of Unusual Materials\nAda Miller\nAbstract\nAn experimental result."
        result = verify_identity_input({"first": first, "text": first, "metadata_title": self.record["title"]}, self.service)
        self.assertEqual(result["status"], "conflicting")
        self.assertIn("does not distinguish their editions", result["reason"])

    def test_title_fallback_after_empty_identifier_lookup_preserves_printed_doi(self):
        from app.services.publication_identity import verify_identity_input
        first = "A Study of Unusual Materials\nAda Miller\nDOI: 10.1234/example\nAbstract\nResults were reproduced."
        identity = {"first": first, "text": first, "metadata_title": self.record["title"],
                    "layout_authors": ["Ada Miller"]}
        self.service.search.side_effect = [
            {"results": [], "provider_errors": {"crossref_verification": "timeout"}},
            {"results": [{**self.record, "doi": "10.1234/other"}, self.record]}
        ]
        self.assertEqual(verify_identity_input(identity, self.service)["doi"], "10.1234/example")
        self.service.search.side_effect = [
            {"results": []}, {"results": [{**self.record, "doi": "10.1234/other"}]}
        ]
        result = verify_identity_input(identity, self.service)
        self.assertEqual(result["status"], "not_verified")
        self.assertEqual(result["reason_code"], "no_corroborated_record")

    def test_title_fallback_cannot_override_a_conflicting_printed_identifier(self):
        from app.services.publication_identity import verify_identity_input
        first = "A Study of Unusual Materials\nAda Miller\nDOI: 10.1234/example\nAbstract\nResults were reproduced."
        self.service.search.side_effect = [
            {"results": [{**self.record, "title": "A completely different publication"}]},
            {"results": [self.record]},
        ]
        result = verify_identity_input({"first": first, "text": first, "metadata_title": self.record["title"]}, self.service)
        self.assertEqual(result["status"], "not_verified")
        self.service.search.assert_called_once_with("10.1234/example", 0, 1)

    def test_standalone_structured_abstract_keeps_results_and_stops_before_footer(self):
        first = ("A Study of Unusual Materials\nAda Miller\nBACKGROUND\nSome materials react to heat.\n"
                 "METHODS\nWe tested three samples.\nRESULTS\nThe materials behaved differently.\n"
                 "CONCLUSIONS\nComposition matters.\nABSTR ACT\nCopyright belongs to the publisher.")
        metadata = extracted_paper_metadata({"first": first, "metadata_title": "A Study of Unusual Materials"})
        self.assertIn("RESULTS The materials behaved differently", metadata["abstract"])
        self.assertTrue(metadata["abstract"].endswith("Composition matters."))
