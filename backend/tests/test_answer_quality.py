import unittest

from app.providers.answers import ClaimAssessment
from app.services.answer_quality import AnswerQuality


def review(text, status, *sources):
    return {"text": text, "status": status,
            "supporting": [{"source_id": source, "quote": text} for source in sources], "opposing": []}


class AnswerQualityTests(unittest.TestCase):
    def test_citations_count_claim_source_pairs_and_preserve_repaired_claims(self):
        evaluation = AnswerQuality()
        evaluation.capture("The sediments included shale [c1] [c2]. Quartz was identified.", "", ["c1", "c2"], [], {"c1", "c2"}, {})
        evaluation.assessments = {"claim-1": ClaimAssessment("claim-1", "supported", [], [], unsupported_source_ids=("c2",))}
        results = [review(evaluation.rows[0]["text"], "supported", "c1"),
                   review(evaluation.rows[1]["text"], "supported", "c2")]
        report = evaluation.build(question="Which minerals are described?", message={"claims": results}, threshold=1, revision=1)
        metrics = report["metrics"]
        self.assertEqual(metrics["citation_coverage"]["score"], .5)
        self.assertEqual(metrics["citation_precision"]["score"], .5)
        self.assertEqual(metrics["citation_distribution"]["score"], .5)
        self.assertEqual(metrics["claim_support"]["score"], 1)
        self.assertIsNone(metrics["completeness"]["score"])
        self.assertEqual(metrics["citation_precision"]["denominator"], 2)

    def test_completeness_counts_partial_question_independently_of_correctness(self):
        evaluation = AnswerQuality()
        evaluation.capture("Trial Meridian used a randomized design [c1].", "", ["c1"], [], {"c1"}, {})
        evaluation.assessments = {"claim-1": ClaimAssessment("claim-1", "unsupported", [], [], question_fit="direct")}
        report = evaluation.build(question="What design and participants did Trial Meridian use?",
            message={"claims": [review(evaluation.rows[0]["text"], "unsupported")]}, threshold=.5, revision=1)
        self.assertEqual(report["metrics"]["completeness"]["score"], .5)
        self.assertEqual(report["metrics"]["completeness"]["status"], "PASS")
        self.assertEqual(report["metrics"]["claim_support"]["score"], 0)
        self.assertEqual([r["dimension"] for r in report["requirements"] if not r["addressed"]], ["participants"])

    def test_missing_citation_is_distinct_from_missing_verifier(self):
        evaluation = AnswerQuality()
        evaluation.capture("A harbour was flooded.", "", [], [], {}, {})
        report = evaluation.build(question="Was the harbour flooded?", message={}, threshold=1, revision=1)
        self.assertEqual(report["metrics"]["citation_coverage"]["score"], 0)
        self.assertIsNone(report["metrics"]["citation_precision"]["score"])
        evaluation.capture("A harbour was flooded [c1].", "", ["c1"], [], {"c1"}, {})
        report = evaluation.build(question="Was the harbour flooded?",
            message={"answer_state": "VERIFICATION_UNAVAILABLE"}, threshold=1, revision=2)
        self.assertEqual(report["state"], "UNAVAILABLE")
        self.assertIsNone(report["metrics"]["citation_coverage"]["score"])
        self.assertIsNone(report["metrics"]["citation_precision"]["score"])

    def test_verifier_requirements_cover_unrecognized_questions_and_ignore_unknown_claim_ids(self):
        evaluation = AnswerQuality()
        evaluation.capture("The archive lists copper coins [c1].", "", ["c1"], [], {"c1"}, {})
        evaluation.assessments = {"claim-1": ClaimAssessment("claim-1", "supported", [], [])}
        evaluation.requirements = [
            {"requirement": "Identify the listed metal", "addressed_by": ["claim-1"]},
            {"requirement": "Identify the mint", "addressed_by": ["unknown"]},
        ]
        report = evaluation.build(question="Of what and whence?",
            message={"claims": [review(evaluation.rows[0]["text"], "supported", "c1")]}, threshold=1, revision=1)
        self.assertEqual(report["metrics"]["completeness"]["score"], .5)
        self.assertEqual(report["metrics"]["completeness"]["reason"], "verifier_question_requirements")

    def test_mixed_claims_distinguish_promotional_and_contradicted_evidence(self):
        evaluation = AnswerQuality()
        evaluation.capture("The flow reached 12 liters [c1].", "The device reduced flow [ext1].", ["c1"], ["ext1"], {"c1"}, {"ext1"})
        evaluation.assessments = {
            "claim-1": ClaimAssessment("claim-1", "contradicted", [], [], unsupported_source_ids=("c1",)),
            "claim-2": ClaimAssessment("claim-2", "partially_supported", [], [], evidence_use="attribution_only"),
        }
        result = evaluation.build(question="What was the flow?", message={"claims": [
            review(evaluation.rows[0]["text"], "contradicted"),
            review(evaluation.rows[1]["text"], "partially_supported", "ext1")],
            "verification_checks": [{"check_id": "AH5", "status": "FAIL", "reason_code": "numeric_claim_not_verified"}]}, threshold=1, revision=1)
        self.assertEqual(result["evidence_type"], "mixed")
        self.assertEqual(result["metrics"]["citation_precision"]["score"], 0)
        self.assertEqual(result["metrics"]["contradiction_rate"]["score"], .5)
        self.assertEqual(result["metrics"]["source_quality"]["score"], .5)
        self.assertEqual(result["metrics"]["numerical_accuracy"]["status"], "FAIL")
