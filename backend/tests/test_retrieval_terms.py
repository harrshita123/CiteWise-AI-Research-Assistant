import unittest
from unittest.mock import Mock

from app.repositories.vectors import VectorRepository


class RetrievalTermTests(unittest.TestCase):
    def test_multipart_question_preserves_close_continuation_without_duplicates(self):
        repo = object.__new__(VectorRepository)
        repo.collection = Mock()
        indexes = [10, 20, 21, 22, 23, 11]
        metadata = [{"chunk_index": index, "page_number": 3, "content_hash": str(index)} for index in indexes]
        repo.collection.query.return_value = {
            "documents": [["They used a statistical model." for _ in indexes[:-1]] + ["They computed variance."]],
            "metadatas": [metadata],
        }
        question = "What model did the researchers use, and how was uncertainty estimated?"
        selected = repo.search("alice", "paper", [0.1], question=question)
        self.assertEqual([item.chunk_index for item in selected], [10, 20, 21, 22, 11])
        self.assertEqual((selected[-1].page_number, selected[-1].content_hash), (3, "11"))
        self.assertEqual(repo.collection.query.call_args.kwargs["where"], repo.paper_filter("alice", "paper"))
        metadata[-1]["content_hash"] = "10"
        self.assertEqual([item.chunk_index for item in repo.search("alice", "paper", [0.1], question=question)], indexes[:5])
        metadata[-1].update(page_number=4, content_hash="11")
        self.assertEqual([item.chunk_index for item in repo.search("alice", "paper", [0.1], question=question)], indexes[:5])

    def test_methodology_and_participant_paraphrases_share_terms(self):
        question = VectorRepository._terms("Compare methodologies and participants")
        passage = VectorRepository._terms("A randomized design enrolled students")
        self.assertIn("student", question & passage)
        self.assertIn("design", VectorRepository._terms("methodology") & passage)


if __name__ == "__main__":
    unittest.main()
