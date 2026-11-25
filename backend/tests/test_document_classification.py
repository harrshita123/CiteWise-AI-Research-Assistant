import os
import unittest
from pathlib import Path

import pymupdf

from app.services.document_classification import DocumentType, classify_document
from app.services.publication_identity import extract_identity_input, extracted_paper_metadata


def extract(text, title="", author=""):
    with pymupdf.open() as document:
        document.new_page().insert_textbox(pymupdf.Rect(40, 40, 550, 750), text)
        document.set_metadata({"title": title, "author": author})
        return extract_identity_input(document.tobytes(), include_tail=True)


class DocumentClassificationTests(unittest.TestCase):
    def test_independent_document_formats(self):
        title = "Bounds on Repeated Sampling"
        cases = [
            ("one-page theory, no DOI or abstract",
             f"{title}\nGiacomo Rossi\nTheorem 1. Every finite sample satisfies these bounds.\nProof. Apply the induction hypothesis to the reduced sample.",
             DocumentType.RESEARCH_PAPER, None),
            ("research quoting certificate wording",
             "Patterns in Administrative Language\nAbstract\nWe model certificate wording.\nIntroduction\n"
             "Certificate of Completion\nAwarded to the employee who has successfully completed training.\n"
             "Methods\nWe counted vocabulary in one hundred documents.",
             DocumentType.RESEARCH_PAPER, None),
            ("certificate",
             "Certificate of Completion\nAwarded to Jordan Park who has successfully completed the training course.",
             DocumentType.NOT_RESEARCH_PAPER, "a certificate"),
            ("certificate with scholarly headings",
             "Certificate of Completion\nAwarded to an employee who has successfully completed training.\n"
             "Abstract\nCourse outline.\nMethods\nTraining syllabus.\nResults\nCompletion granted.",
             DocumentType.NOT_RESEARCH_PAPER, "a certificate"),
            ("admission",
             "Provisional Admission Letter\nYou are admitted to the science program for the coming academic year.",
             DocumentType.NOT_RESEARCH_PAPER, "an admission letter"),
            ("résumé",
             "Curriculum Vitae\nJordan Park\nWork Experience\nEngineer on multiple public infrastructure projects.\nEducation\nBachelor of science and engineering.",
             DocumentType.NOT_RESEARCH_PAPER, "a résumé"),
            ("accented résumé heading",
             "Résumé\nJordan Park\nSkills\nSoftware testing and technical documentation.",
             DocumentType.NOT_RESEARCH_PAPER, "a résumé"),
            ("invoice",
             "Invoice 302\nBill to the company listed below.\nAmount due equals the subtotal plus tax.",
             DocumentType.NOT_RESEARCH_PAPER, "an invoice or receipt"),
            ("transcript",
             "Official Transcript\nStudent ID is listed with all credits earned this academic year.",
             DocumentType.NOT_RESEARCH_PAPER, "a transcript"),
            ("mark sheet", "Mark Sheet\nMaximum marks: 100\nMarks obtained: 76",
             DocumentType.NOT_RESEARCH_PAPER, "a transcript"),
            ("identity card", "National Identity Card\nDate of birth: January 1999",
             DocumentType.NOT_RESEARCH_PAPER, "an identity document"),
            ("employment offer", "Offer Letter\nStart date is June 1. Your annual salary is stated below.",
             DocumentType.NOT_RESEARCH_PAPER, "an employment offer"),
            ("assignment with academic headings",
             "Research Assignment 3\nDue date: March 3. Submit your work to your instructor.\n"
             "Abstract\nIntroduction\nMethods\nResults\nReferences",
             DocumentType.NOT_RESEARCH_PAPER, "an academic assignment"),
            ("slides", "Lecture Slides\nSlide 1\nAgenda\nLearning objectives for the course this week.",
             DocumentType.NOT_RESEARCH_PAPER, "a slide presentation"),
            ("personal cover letter", "Cover Letter\nDear hiring manager, I am applying to the advertised position.",
             DocumentType.NOT_RESEARCH_PAPER, "a personal document"),
            ("application",
             "Travel Application Form\nApplicant name and signature must be supplied for office use only.",
             DocumentType.NOT_RESEARCH_PAPER, "an application form"),
            ("office notice",
             "Office Notice\nThe reception desk will be closed for maintenance from Tuesday to Thursday.",
             DocumentType.NOT_RESEARCH_PAPER, "an administrative document"),
            ("empty", "", DocumentType.INCONCLUSIVE, None),
            ("unknown", "An undated note about a new idea for collecting observations during the spring.",
             DocumentType.INCONCLUSIVE, None),
        ]
        for name, text, expected, category in cases:
            with self.subTest(document=name):
                self.assertEqual(classify_document(extract(text, title=title)), (expected, category))

    def test_misleading_metadata_does_not_label_a_certificate_as_research(self):
        identity = extract(
            "Certificate of Achievement\nAwarded to Dana White\nhas completed all the lessons and projects of\n"
            "Foundations of Software Engineering\nSigned by the training director.",
            title="has completed all the lessons and projects of")
        self.assertEqual(classify_document(identity)[0], DocumentType.NOT_RESEARCH_PAPER)
        self.assertNotEqual(extracted_paper_metadata(identity).get("title"), identity["metadata_title"])

    def test_copied_metadata_and_identifier_with_a_short_label_are_inconclusive(self):
        title = "A Study of Coastal Sound"
        identity = extract(f"{title}\nAna Rivera\nhttps://doi.org/10.1234/coast\nAbstract: "
                           "Please bring your staff identification and collect a visitor badge at reception.",
                           title=title, author="Ana Rivera")
        self.assertEqual(extracted_paper_metadata(identity)["doi"], "10.1234/coast")
        self.assertEqual(classify_document(identity)[0], DocumentType.INCONCLUSIVE)

    def test_image_only_scan_has_no_classification(self):
        with pymupdf.open() as source, pymupdf.open() as scan:
            page = source.new_page()
            page.insert_text((40, 40), "Certificate of Completion")
            scan_page = scan.new_page()
            scan_page.insert_image(scan_page.rect, pixmap=page.get_pixmap())
            identity = extract_identity_input(scan.tobytes(), include_tail=True)
        self.assertEqual(classify_document(identity)[0], DocumentType.INCONCLUSIVE)

    @unittest.skipUnless(os.getenv("CITEWISE_NATURE_TEST_PDF"), "real Nature PDF not configured")
    def test_real_nature_alpha_fold_paper(self):
        content = Path(os.environ["CITEWISE_NATURE_TEST_PDF"]).read_bytes()
        identity = extract_identity_input(content, include_tail=True)
        self.assertEqual(classify_document(identity)[0], DocumentType.RESEARCH_PAPER)
        metadata = extracted_paper_metadata(identity)
        self.assertEqual(metadata["title"], "Highly accurate protein structure prediction with AlphaFold")
        self.assertIn("John Jumper", metadata["authors"])
        self.assertEqual(metadata["doi"], "10.1038/s41586-021-03819-2")
        self.assertTrue(metadata["abstract"].startswith("Proteins are essential to life"))
        self.assertNotIn("The development of computational methods", metadata["abstract"])
