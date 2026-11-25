import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.paper_models import ChunkResponse
from scripts.benchmark_retrieval_quality import (CASES_BY_CATEGORY, CATEGORY_REQUIREMENTS, PAPERS,
    baseline_for, load_research_fixture, regressions, retrieve, score, summary)


FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks/retrieval_ci/research_excerpts.json"


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_rerank_benchmark_maps_indices_to_original_chunk_ids(self):
        repo = Mock()
        repo.search.return_value = [
            ChunkResponse(chunk_index=index, page_number=1, content_hash=str(index), text=f"Passage {index}")
            for index in (4, 7, 12)
        ]
        repo.collection.candidate_ids = [4, 7, 12]
        reranker = Mock()
        reranker.rerank.return_value = [2, 0, 1]
        ids, counts = retrieve(repo, "paper", [0.1], "Question", "openrouter_rerank", reranker)
        self.assertEqual(ids, [12, 4, 7])
        self.assertEqual(counts["dense_candidate_ids"], [4, 7, 12])
        reranker.rerank.assert_called_once_with("Question", ["Passage 4", "Passage 7", "Passage 12"], 3)

    def test_actual_pdf_labels_match_the_indexed_excerpts(self):
        for fixture_name, baseline_name in (
            (FIXTURE.name, "baseline.json"),
            ("holdout_excerpts.json", "holdout_baseline.json"),
        ):
            with self.subTest(fixture=fixture_name):
                papers, cases, digest = load_research_fixture(FIXTURE.with_name(fixture_name))
                self.assertGreaterEqual(len(papers), 2)
                saved = json.loads(FIXTURE.with_name(baseline_name).read_text())
                self.assertEqual(saved["fixture_sha256"], digest)
                self.assertEqual(set(saved["papers"]), set(cases))

    def test_rejects_undocumented_absence(self):
        data = json.loads(FIXTURE.with_name("holdout_excerpts.json").read_text())
        first_paper = next(iter(data["papers"].values()))
        absent = next(query for query in first_paper["queries"] if query["category"] == "absent")
        del absent["absence_reason"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_research_fixture(path)

    def test_rejects_unbacked_relevance_judgment(self):
        data = json.loads(FIXTURE.read_text())
        first_paper = next(iter(data["papers"].values()))
        first_paper["queries"][0]["labels"][0]["quote"] = "This quotation was invented."
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "does not match a PDF excerpt"):
                load_research_fixture(path)

    def test_baseline_gate_catches_a_loss_even_when_another_paper_improves(self):
        first = {"paper": "first", "required_chunks": [0, 1], "returned_chunks": [0, 1], "scope_leak_count": 0}
        second = {"paper": "second", "required_chunks": [0, 1], "returned_chunks": [0], "scope_leak_count": 0}
        report = {"fixture_sha256": "test", "k": 5, "dependencies": {},
                  "modes": {"current_dense": {"cases": [first, second]}}}
        baseline = baseline_for(report)
        self.assertEqual(regressions(report, baseline), [])
        first["returned_chunks"], second["returned_chunks"] = [0], [0, 1]
        self.assertEqual(len(regressions(report, baseline)), 3)
        second["scope_leak_count"] = 1
        self.assertIn("out-of-scope", regressions(report, baseline)[-1])
        report["fixture_sha256"] = "different"
        self.assertIn("differ", regressions(report, baseline)[0])

    def test_baseline_gate_catches_later_support_even_when_hits_are_unchanged(self):
        case = {"paper": "paper", "required_chunks": [1],
                "returned_chunks": [1, 2], "scope_leak_count": 0}
        report = {"fixture_sha256": "test", "k": 5, "dependencies": {},
                  "modes": {"current_dense": {"cases": [case]}}}
        baseline = baseline_for(report)
        case["returned_chunks"] = [2, 1]
        failures = regressions(report, baseline)
        self.assertEqual(len(failures), 1)
        self.assertIn("sum of reciprocal ranks", failures[0])

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
        self.assertIsNone(row["precision_at_5"])
        self.assertIsNone(row["mrr"])
        self.assertEqual(summary([row])["labeled_questions"], 0)


if __name__ == "__main__":
    unittest.main()
