import unittest

from scripts.benchmark_retrieval_quality import CASES_BY_CATEGORY, CATEGORY_REQUIREMENTS, PAPERS, score, summary


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_ground_truth_covers_general_categories(self):
        self.assertEqual(set(CASES_BY_CATEGORY), set(CATEGORY_REQUIREMENTS))
        self.assertGreaterEqual(len(PAPERS), 3)
        for category, examples in CASES_BY_CATEGORY.items():
            self.assertTrue(examples)
            for paper, question, required in examples:
                self.assertIn(paper, PAPERS)
                self.assertTrue(question.strip())
                self.assertTrue(all(0 <= index < len(PAPERS[paper]) for index in required))
                self.assertEqual(bool(required), category != "absent")

    def test_all_required_evidence_is_stricter_than_any_hit(self):
        candidates = {"dense_candidates": 3, "dense_candidate_ids": [0, 1, 2],
                      "lexical_scanned": 0, "lexical_candidates": 0}
        source_metadata = {0: {"paper_id": "exams", "page_number": 1},
                           1: {"paper_id": "exams", "page_number": 1}}
        row = score("exams", "multi_study", "compare", {0, 1}, [0],
                    candidates, [1.0, 2.0, 3.0], source_metadata)
        self.assertEqual(row["recall_at_5"], 0.5)
        self.assertEqual(row["precision_at_5"], 0.2)
        self.assertEqual(row["mrr"], 1.0)
        self.assertFalse(row["all_required_recall"])
        self.assertEqual(row["candidate_recall"], 1.0)
        self.assertEqual(row["required_studies"], ["A", "B"])
        self.assertEqual(row["returned_studies"], ["A"])
        self.assertEqual(row["returned_sources"][0]["page_number"], 1)

    def test_no_evidence_case_is_excluded_from_recall_average(self):
        candidates = {"dense_candidates": 1, "dense_candidate_ids": [0],
                      "lexical_scanned": 0, "lexical_candidates": 0}
        row = score("exams", "absent", "outside PDF", set(), [0],
                    candidates, [1.0], {0: {"paper_id": "exams", "page_number": 1}})
        self.assertIsNone(row["recall_at_5"])
        self.assertIsNone(row["all_required_recall"])
        self.assertEqual(row["precision_at_5"], 0)
        self.assertEqual(summary([row])["labeled_questions"], 0)


if __name__ == "__main__":
    unittest.main()
