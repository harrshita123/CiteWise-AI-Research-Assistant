import unittest

from app.services.evidence_coverage import CoveragePassage, coverage_report, missing_coverage, requested_coverage


class EvidenceCoverageTests(unittest.TestCase):
    def test_context_words_do_not_create_independent_answer_parts(self):
        cases = [
            ("What findings were reported among students?", ["findings"]),
            ("What findings were reported after six months?", ["findings"]),
            ("What did participants report about recovery?", ["findings"]),
            ("What design was used with 40 participants?", ["design"]),
        ]
        for question, dimensions in cases:
            with self.subTest(question=question):
                self.assertEqual(requested_coverage(question)[1], dimensions)

    def test_explicit_participant_and_duration_requests_remain_targets(self):
        self.assertEqual(requested_coverage(
            "Compare each study's design, participants, and duration.")[1],
            ["design", "participants", "duration"])

    def test_implicit_comparison_uses_two_studies_found_in_evidence(self):
        question = ("Compare both studies' designs, participant counts, findings and duration.")
        passages = [CoveragePassage("c1", "Trial Solstice used a cohort design with 48 patients. "
                                    "Misconduct decreased by 4%. Follow-up lasted six months."),
                    CoveragePassage("c2", "Trial Polaris used a randomized design. "
                                    "Participants were discussed. The finding was measured.")]
        self.assertEqual(requested_coverage(question)[0], [])
        report = {(item.subject, item.dimension): item.status
                  for item in coverage_report(question, passages)}
        self.assertEqual(report["Trial Solstice", "design"], "information_present")
        self.assertEqual(report["Trial Solstice", "participants"], "information_present")
        self.assertEqual(report["Trial Solstice", "findings"], "information_present")
        self.assertEqual(report["Trial Solstice", "duration"], "information_present")
        self.assertEqual(report["Trial Polaris", "design"], "information_present")
        self.assertEqual(report["Trial Polaris", "participants"], "topic_mentioned")
        self.assertEqual(report["Trial Polaris", "findings"], "topic_mentioned")
        self.assertEqual(report["Trial Polaris", "duration"], "missing")
        self.assertEqual(missing_coverage(question, [item.text for item in passages]),
                         [("Trial Polaris", "participants"), ("Trial Polaris", "duration"),
                          ("Trial Polaris", "findings")])
        two_studies = coverage_report("Compare the two studies' designs.", passages)
        self.assertEqual([item.subject for item in two_studies],
                         ["Trial Solstice", "Trial Polaris"])

    def test_each_study_can_identify_more_than_two_subjects(self):
        question = "Compare each study's participant count."
        passages = [CoveragePassage("c1", "Study Aurora participants: 24 teachers."),
                    CoveragePassage("c2", "Study Boreal discussed participants."),
                    CoveragePassage("c3", "Study Cypress participants: 36 teachers.")]
        report = [(item.subject, item.status) for item in coverage_report(question, passages)]
        self.assertEqual(report, [("Study Aurora", "information_present"),
                                  ("Study Boreal", "topic_mentioned"),
                                  ("Study Cypress", "information_present")])

    def test_named_trials_and_cohorts_are_scoped_independently(self):
        question = "Compare Trial ORBIT and the Northstar cohort designs and sample sizes."
        report = coverage_report(question, [CoveragePassage("c1",
            "Trial ORBIT used a randomized design with 90 patients. "
            "Northstar cohort discussed participants but gave no count.")])
        found = {(item.subject, item.dimension): item.status for item in report}
        self.assertEqual(found["Trial ORBIT", "design"], "information_present")
        self.assertEqual(found["Trial ORBIT", "participants"], "information_present")
        self.assertEqual(found["Northstar cohort", "participants"], "topic_mentioned")

    def test_unlabeled_qualitative_and_review_papers_do_not_need_study_labels(self):
        qualitative = coverage_report("What methodology and findings were reported?", [
            CoveragePassage("c1", "Researchers conducted interviews with patients. "
                            "Participants reported increased trust.")])
        self.assertEqual([item.status for item in qualitative],
                         ["information_present", "information_present"])
        review = coverage_report("What methodology was used?", [CoveragePassage(
            "c2", "A systematic review synthesized 14 published studies.")])
        self.assertEqual(review[0].status, "information_present")

    def test_comparison_keeps_each_studys_methods_and_sample_separate(self):
        question = "Compare Study C and Study D sample sizes and designs."
        passages = [
            CoveragePassage("c1", "Study C design and participants were discussed."),
            CoveragePassage("c2", "Study D used a randomized design with 350 participants."),
        ]
        found = {(item.subject, item.dimension): item for item in coverage_report(question, passages)}
        self.assertEqual(found["Study C", "design"].status, "topic_mentioned")
        self.assertEqual(found["Study C", "participants"].status, "topic_mentioned")
        self.assertEqual(found["Study C", "participants"].candidate_ids, ("c1",))
        self.assertEqual(found["Study D", "design"].status, "information_present")
        self.assertEqual(found["Study D", "participants"].information_ids, ("c2",))
        self.assertEqual(missing_coverage(question, [item.text for item in passages]),
                         [("Study C", "design"), ("Study C", "participants")])

    def test_shared_chunk_does_not_attribute_one_studys_count_to_another(self):
        report = coverage_report("Compare Study C and Study D sample sizes.", [CoveragePassage(
            "c1", "Study C included 120 participants. Study D discussed participants but gave no count.")])
        self.assertEqual([(item.subject, item.status) for item in report],
                         [("Study C", "information_present"), ("Study D", "topic_mentioned")])

    def test_adjacent_unnamed_chunk_is_only_a_study_hint(self):
        question = "What was Study C's sample size?"
        report = coverage_report(question, [
            CoveragePassage("c1", "Study C methods continued below.", 2, 1),
            CoveragePassage("c2", "Participants: 120 students.", 2, 2),
        ])
        self.assertEqual(report[0].status, "topic_mentioned")
        self.assertEqual(report[0].inferred_subject_ids, ("c2",))
        self.assertEqual(report[0].information_ids, ())
        new_page = coverage_report(question, [
            CoveragePassage("c1", "Study C methods continued below.", 2, 1),
            CoveragePassage("c2", "Participants: 120 students.", 3, 2),
        ])
        self.assertEqual(new_page[0].status, "missing")

    def test_methodology_and_duration_require_factual_cues(self):
        question = "What were Study C's methodology and duration?"
        topic = coverage_report(question, [CoveragePassage("c1", "Study C methodology and duration were recorded.")])
        self.assertEqual([item.status for item in topic], ["topic_mentioned", "topic_mentioned"])
        actual = coverage_report(question, [CoveragePassage(
            "c2", "Study C used a prospective cohort over two semesters.")])
        self.assertEqual([item.status for item in actual], ["information_present", "information_present"])

    def test_opposing_findings_are_candidates_not_verified_contradictions(self):
        question = "Compare Study C and Study D findings about the outcome."
        report = coverage_report(question, [
            CoveragePassage("c1", "Study C reported an increase in the outcome."),
            CoveragePassage("c2", "Study D reported no significant difference in the outcome."),
        ])
        self.assertEqual([item.status for item in report], ["information_present", "information_present"])
        self.assertEqual([item.information_ids for item in report], [("c1",), ("c2",)])

    def test_mention_of_outcome_without_result_does_not_complete_synthesis(self):
        question = "Compare Study C and Study D findings about privacy."
        report = coverage_report(question, [
            CoveragePassage("c1", "Study C found increased privacy concerns."),
            CoveragePassage("c2", "Study D measured privacy outcomes."),
        ])
        self.assertEqual([item.status for item in report], ["information_present", "topic_mentioned"])

    def test_only_verifier_supplied_ids_receive_verified_status(self):
        question = "What were Study C's design and sample size?"
        passages = [CoveragePassage("claim-1", "Study C used a cohort design with 120 participants."),
                    CoveragePassage("claim-2", "Study C used a randomized design with 900 participants.")]
        unverified = coverage_report(question, passages)
        self.assertEqual([item.status for item in unverified], ["information_present", "information_present"])
        verified = coverage_report(question, passages, {"claim-1"})
        self.assertEqual([item.status for item in verified],
                         ["verified_claim_support", "verified_claim_support"])
        self.assertEqual([item.information_ids for item in verified],
                         [("claim-1", "claim-2"), ("claim-1", "claim-2")])


if __name__ == "__main__":
    unittest.main()
