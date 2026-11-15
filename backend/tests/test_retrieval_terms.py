import unittest

from app.repositories.vectors import VectorRepository


class RetrievalTermTests(unittest.TestCase):
    def test_methodology_and_participant_paraphrases_share_terms(self):
        question = VectorRepository._terms("Compare methodologies and participants")
        passage = VectorRepository._terms("A randomized design enrolled students")
        self.assertIn("student", question & passage)
        self.assertIn("design", VectorRepository._terms("methodology") & passage)


if __name__ == "__main__":
    unittest.main()
