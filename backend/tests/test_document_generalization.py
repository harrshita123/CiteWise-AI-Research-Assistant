"""Small synthetic PDF indexing checks; not a real-paper accuracy benchmark."""

import tempfile
import unittest
from pathlib import Path

from app.repositories.vectors import VectorRepository
from app.services.pdf_processing import PdfValidationError, create_chunks, extract_pages
from tests.test_papers import make_pdf, test_tokenizer


class DocumentGeneralizationTests(unittest.TestCase):
    def test_multiple_document_styles_keep_owner_paper_page_and_content(self):
        documents = {
            "experimental": "A randomized trial of model Orchid enrolled 120 participants.",
            "medical": "The fictional clinical study followed 44 patients after treatment.",
            "mathematics": "The theorem states that every bounded sequence has a convergent subsequence.",
            "qualitative": "Researchers interviewed twelve teachers and identified three themes.",
            "review": "The systematic review included 25 studies and reported conflicting outcomes.",
        }
        with tempfile.TemporaryDirectory() as directory:
            vectors = VectorRepository(Path(directory))
            for index, (paper_id, body) in enumerate(documents.items()):
                _, pages = extract_pages(make_pdf(body), 10, 10000)
                chunks = create_chunks(pages, test_tokenizer())
                embedding = [float(position == index) for position in range(len(documents))]
                vectors.save("alice", paper_id, chunks, [embedding for _ in chunks])
                result = vectors.search("alice", paper_id, embedding, limit=5)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].page_number, 1)
                self.assertIn(body, result[0].text)
                self.assertEqual(vectors.search("bob", paper_id, embedding, limit=5), [])

    def test_malformed_and_scanned_only_pdfs_are_rejected(self):
        for content in (b"not a pdf", make_pdf("")):
            with self.subTest(content=content[:8]), self.assertRaises(PdfValidationError):
                extract_pages(content, 10, 10000)


if __name__ == "__main__":
    unittest.main()
