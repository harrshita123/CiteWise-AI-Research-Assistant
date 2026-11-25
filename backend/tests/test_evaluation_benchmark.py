import unittest

from pydantic import ValidationError

from app.services.evaluation_benchmark import (TAXONOMY, SKILLS, EvidenceRef, ResearchLabel, RetrievedChunk,
                                                BenchmarkCase, FrozenBenchmark, CaseObservation,
                                                evaluate_benchmark)


class BenchmarkModelTests(unittest.TestCase):
    def setUp(self):
        self.ref = EvidenceRef(page=2, chunk_index=3, content_hash="chunk-hash", quote="A reported finding")

    def case(self, case_id, paper_id, axis, label, skill, behavior="answer"):
        return BenchmarkCase(id=case_id, paper_id=paper_id, paper_hash=f"hash-{paper_id}",
            question=f"What does {paper_id} report?", expected_behavior=behavior,
            expected_answer="A reported finding" if behavior == "answer" else "Not reported",
            evidence=(self.ref,) if behavior == "answer" else (),
            labels=(ResearchLabel(axis=axis, label=label, confidence=0.8,
                                  evidence=(self.ref,), needs_review=False),),
            skills=(skill,), review_status="approved", reviewed_by="reviewer")

    def test_taxonomy_separates_design_and_evidence(self):
        self.assertIn("Conflicting Answers", TAXONOMY["evidence"])
        self.assertNotIn("Conflicting Answers", TAXONOMY["design"])
        self.assertIn("Contradiction", SKILLS)
        with self.assertRaises(ValidationError):
            ResearchLabel(axis="design", label="Conflicting Evidence", confidence=1)

    def test_pending_generated_cases_cannot_become_gold(self):
        pending = BenchmarkCase(id="pending", paper_id="p", paper_hash="hash", question="Why?",
                                expected_behavior="answer", expected_answer="Proposed answer")
        with self.assertRaises(ValidationError):
            FrozenBenchmark(version="v1", cases=(pending,))
        with self.assertRaises(ValidationError):
            BenchmarkCase(id="bad", paper_id="p", paper_hash="hash", question="Why?",
                          expected_behavior="answer", expected_answer="Answer", review_status="approved")

    def test_multiple_disciplines_and_reproducible_fingerprint(self):
        cases = (
            self.case("exp", "clinical", "design", "Experimental", "Bias"),
            self.case("qual", "interviews", "approach", "Qualitative", "Limitations"),
            self.case("theory", "physics", "approach", "Theoretical", "Uncertainty"),
            self.case("review", "reviews", "design", "Systematic Review", "Consensus"),
            self.case("absent", "physics", "evidence", "Conflicting Evidence", "Contradiction", "abstain"),
        )
        benchmark = FrozenBenchmark(version="1.0", cases=cases)
        self.assertEqual(benchmark.fingerprint(), FrozenBenchmark.model_validate(benchmark.model_dump()).fingerprint())
        rows = [CaseObservation(case_id=case.id, returned_pages=(2,), answer_correct=True,
                                citations_supported=1, citations_total=1, claims_cited=1,
                                claims_requiring_citation=1, refused=case.expected_behavior == "abstain",
                                latency_ms=100) for case in cases]
        report = evaluate_benchmark(benchmark, rows)
        self.assertEqual(report["overall"]["hit_at_5"], 1)
        self.assertEqual(report["overall"]["unsupported_refusal_accuracy"], 1)
        self.assertEqual(report["by_category"]["Qualitative"]["case_count"], 1)
        self.assertEqual(report["by_skill"]["Contradiction"]["case_count"], 1)
        self.assertEqual(report["benchmark_hash"], benchmark.fingerprint())

    def test_unmeasured_metrics_remain_null(self):
        benchmark = FrozenBenchmark(version="1", cases=(self.case("a", "p", "purpose", "Exploratory", "Bias"),))
        report = evaluate_benchmark(benchmark, [CaseObservation(case_id="a")])
        self.assertIsNone(report["overall"]["answer_correctness"])
        self.assertIsNone(report["overall"]["citation_precision"])
        self.assertIsNone(report["overall"]["median_answer_latency_ms"])

    def test_rate_limit_is_not_a_retrieval_miss(self):
        cases = (self.case("a", "p1", "purpose", "Exploratory", "Bias"),
                 self.case("b", "p2", "approach", "Qualitative", "Limitations"))
        report = evaluate_benchmark(FrozenBenchmark(version="1", cases=cases), [
            CaseObservation(case_id="a", returned_pages=(2,)),
            CaseObservation(case_id="b", execution_status="rate_limited")])
        self.assertEqual(report["overall"]["hit_at_5"], 1)
        self.assertEqual(report["overall"]["completed_count"], 1)
        self.assertIsNone(report["by_document"]["p2"]["hit_at_5"])

    def test_same_page_but_wrong_chunk_is_not_a_hit(self):
        benchmark = FrozenBenchmark(version="1", cases=(self.case("a", "p", "purpose", "Exploratory", "Bias"),))
        row = CaseObservation(case_id="a", returned_pages=(2,), returned_chunks=(
            RetrievedChunk(page=2, chunk_index=4, content_hash="different"),))
        self.assertEqual(evaluate_benchmark(benchmark, [row])["overall"]["hit_at_5"], 0)


if __name__ == "__main__":
    unittest.main()
