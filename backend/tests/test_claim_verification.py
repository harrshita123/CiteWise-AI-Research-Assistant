import unittest
from datetime import date

from app.providers.answers import ClaimAssessment, ClaimQuote, Evidence
from app.services.claim_verification import Claim, _numbers_supported, present_claims, review_claims, split_claims


class ClaimVerificationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            Evidence("c1", "The study included 412 students and did not report a p-value."),
            Evidence("c2", "The PDF found no increase in privacy concerns with webcam monitoring."),
            Evidence("ext1", "The survey reported that 59% of students expressed privacy concerns."),
            Evidence("ext2", "A separate survey found no increase in privacy concerns."),
            Evidence("ext3", "A separate study found webcam monitoring increased privacy concerns."),
        ]

    def check(self, text, status, support, oppose=(), section="pdf", dates=None):
        claims = split_claims(text if section == "pdf" else "", text if section == "external" else "")
        proposed = [ClaimAssessment(claims[0].id, status,
            [ClaimQuote(*quote) for quote in support],
            [ClaimQuote(*quote) for quote in oppose])]
        return review_claims(claims, proposed, self.sources, dates, date(2026, 11, 13))[0]

    def test_a_direct_support(self):
        result = self.check("The study included 412 students [c1].", "supported",
            [("c1", "The study included 412 students")])
        self.assertEqual(result.status, "supported")
        self.assertIn("[c1]", present_claims([result], "pdf"))

    def test_numeric_validation_ignores_only_citation_markers(self):
        quote = ClaimQuote("c4-7afe1c5514ef", "The study included 2,400 students")
        self.assertTrue(_numbers_supported("The study included 2,400 students [c4-7afe1c5514ef]", [quote]))
        self.assertTrue(_numbers_supported("The study included 2,400 students 【c4-7afe1c5514ef】", [quote]))
        self.assertTrue(_numbers_supported("The study included 2,400 students [1]", [quote]))
        self.assertTrue(_numbers_supported("The study included 2,400 students (c4-7afe1c5514ef)", [quote]))
        self.assertFalse(_numbers_supported("The study included 2,500 students [c4-7afe1c5514ef]", [quote]))
        self.assertFalse(_numbers_supported("The 2023 study included 2,400 students [1]", [quote]))
        self.assertFalse(_numbers_supported("The study included 2,400 students (c4-7bad)", [quote]))

    def test_parenthesized_source_id_stays_associated_without_becoming_a_number(self):
        source = Evidence("c4-7afe1c5514ef", "Study A included 2,400 students at 12 universities.")
        claims = split_claims("Study A included 2,400 students at 12 universities "
                              "(c4-7afe1c5514ef).", "")
        self.assertEqual(claims[0].cited_ids, [source.id])
        self.assertNotIn("c4-7afe1c5514ef", claims[0].text)
        reviewed = review_claims(claims, [ClaimAssessment(claims[0].id, "supported", [
            ClaimQuote(source.id, "Study A included 2,400 students at 12 universities")], [])], [source])
        self.assertEqual(reviewed[0].status, "supported")
        self.assertEqual(present_claims(reviewed, "pdf").count(f"[{source.id}]"), 1)

    def test_comparison_citations_do_not_change_claims_or_source_ids(self):
        sources = [Evidence("c4-7afe1c5514ef", "Study A included 2,400 students at 12 universities."),
                   Evidence("c6-e6fce8812ec7", "Study B included 780 students at 3 universities.")]
        claims = [
            Claim("claim-1", "Study A included 2,400 students at 12 universities 【c4-7afe1c5514ef】.",
                  "pdf", ["c4-7afe1c5514ef"]),
            Claim("claim-2", "Study B included 780 students at 3 universities 【c6-e6fce8812ec7】.",
                  "pdf", ["c6-e6fce8812ec7"]),
        ]
        proposals = [ClaimAssessment(claim.id, "supported", [ClaimQuote(claim.cited_ids[0], evidence.text)], [])
                     for claim, evidence in zip(claims, sources)]
        reviewed = review_claims(claims, proposals, sources)
        self.assertEqual([item.status for item in reviewed], ["supported", "supported"])
        self.assertEqual([item.claim.text for item in reviewed], [item.text for item in claims])
        self.assertEqual([item.claim.cited_ids for item in reviewed], [item.cited_ids for item in claims])
        rendered = present_claims(reviewed, "pdf")
        self.assertIn("[c4-7afe1c5514ef]", rendered)
        self.assertIn("[c6-e6fce8812ec7]", rendered)

    def test_present_claims_links_each_supported_source_once(self):
        source = Evidence("c6-e6fce8812ec7", "Study B included 780 students in 26 courses at 3 universities.")
        claim = Claim("claim-1", "Study B included 780 students in 26 courses at 3 universities "
                      "【c6-e6fce8812ec7】.", "pdf", [source.id])
        proposal = ClaimAssessment(claim.id, "supported", [
            ClaimQuote(source.id, "Study B included 780 students"),
            ClaimQuote(source.id, "26 courses at 3 universities")], [])
        reviewed = review_claims([claim], [proposal], [source])
        self.assertEqual(reviewed[0].status, "supported")
        shown = present_claims(reviewed, "pdf")
        self.assertEqual(shown.count("[c6-e6fce8812ec7]"), 1)
        self.assertNotIn("【c6-e6fce8812ec7】", shown)

    def test_study_claims_require_exact_quotes_from_their_own_sources(self):
        sources = [Evidence("cA", "Study A used a cluster-randomized comparison. "
                            "It included 2,400 students at 12 universities."),
                   Evidence("cB", "Study B used a matched-course comparison. "
                            "It included 780 students at 3 universities.")]
        claims = [Claim("claim-1", "Study A used a cluster-randomized comparison and included "
                        "2,400 students at 12 universities.", "pdf", ["cA"]),
                  Claim("claim-2", "Study B used a matched-course comparison and included "
                        "780 students at 3 universities.", "pdf", ["cB"])]
        proposals = [
            ClaimAssessment("claim-1", "supported", [
                ClaimQuote("cA", "Study A used a cluster-randomized comparison"),
                ClaimQuote("cA", "It included 2,400 students at 12 universities")], []),
            ClaimAssessment("claim-2", "supported", [
                ClaimQuote("cB", "Study B used a matched-course comparison"),
                ClaimQuote("cB", "It included 780 students at 3 universities")], []),
        ]
        self.assertEqual([item.status for item in review_claims(claims, proposals, sources)],
                         ["supported", "supported"])
        altered = ClaimAssessment("claim-1", "supported", [
            ClaimQuote("cA", "Study A used a cluster-randomized comparison"),
            ClaimQuote("cA", "It enrolled 2,400 students at 12 universities")], [])
        invalid_id = ClaimAssessment("claim-1", "supported", [
            ClaimQuote("cB", "Study B used a matched-course comparison")], [])
        for proposal in (altered, invalid_id):
            self.assertEqual(review_claims([claims[0]], [proposal], sources)[0].status, "unsupported")

    def test_direct_methodology_contrast_requires_its_own_exact_evidence(self):
        source = Evidence("c9", "Study A randomized course sections, while Study B used matched courses "
                          "without random assignment.")
        claim = split_claims("Study A randomized course sections, while Study B used matched courses "
                             "without random assignment [c9].", "")[0]
        direct = ClaimAssessment(claim.id, "supported", [ClaimQuote("c9", source.text)], [])
        self.assertEqual(review_claims([claim], [direct], [source])[0].status, "supported")
        extra = split_claims("Study A had superior experimental control compared with Study B [c9].", "")[0]
        unsupported = ClaimAssessment(extra.id, "unsupported", [], [])
        self.assertEqual(review_claims([extra], [unsupported], [source])[0].status, "unsupported")

    def test_citation_digits_cannot_be_used_as_arithmetic_operands(self):
        quote = ClaimQuote("c4-7afe1c5514ef", "The study enrolled 2,400 students [7].")
        self.assertFalse(_numbers_supported("The combined total was 2,407 students [c4-7afe1c5514ef]", [quote]))

    def test_calculated_numbers_and_decimals_keep_strict_checks_with_citations(self):
        counts = [ClaimQuote("c4-7afe1c5514ef", "Study A enrolled 2,400 students"),
                  ClaimQuote("c6-e6fce8812ec7", "Study B enrolled 780 students")]
        self.assertTrue(_numbers_supported("The combined total was 3,180 students [c4-7afe1c5514ef] "
                                           "[c6-e6fce8812ec7]", counts))
        self.assertFalse(_numbers_supported("The combined total was 3,190 students [c4-7afe1c5514ef] "
                                            "[c6-e6fce8812ec7]", counts))
        rates = [ClaimQuote("c4-7afe1c5514ef", "18% before and 12% after monitoring")]
        self.assertTrue(_numbers_supported("The difference was 6 percentage points [c4-7afe1c5514ef]", rates))
        self.assertFalse(_numbers_supported("The difference was 6% [c4-7afe1c5514ef]", rates))
        fraction = [ClaimQuote("c6-e6fce8812ec7", "120 alerts, 9 linked to confirmed misconduct")]
        self.assertTrue(_numbers_supported("7.5% of alerts were confirmed [c6-e6fce8812ec7]", fraction))
        self.assertFalse(_numbers_supported("7.6% of alerts were confirmed [c6-e6fce8812ec7]", fraction))

    def test_b_valid_citation_does_not_support_claim(self):
        result = self.check("The study included 900 students [c1].", "supported",
            [("c1", "The study included 412 students")])
        self.assertEqual(result.status, "unsupported")
        self.assertNotIn("900 students", present_claims([result], "pdf"))

    def test_named_legal_instruments_are_not_interchangeable(self):
        source = Evidence("ext-law", "The Information Technology Rules require security safeguards.")
        claim = Claim("claim-1", "The Data Protection Act requires security safeguards.",
                      "external", ["ext-law"])
        proposed = ClaimAssessment("claim-1", "supported", [
            ClaimQuote("ext-law", "The Information Technology Rules require security safeguards")], [])
        reviewed = review_claims([claim], [proposed], [source])
        self.assertEqual(reviewed[0].status, "unsupported")
        self.assertNotIn("Data Protection Act", present_claims(reviewed, "external"))

    def test_scoped_rule_does_not_support_universal_biometric_claim(self):
        source = Evidence("ext-law", "Biometric data in the named system requires consent.")
        claim = Claim("claim-1", "All biometric data requires consent.", "external", ["ext-law"])
        proposed = ClaimAssessment("claim-1", "supported", [
            ClaimQuote("ext-law", "Biometric data in the named system requires consent")], [])
        self.assertEqual(review_claims([claim], [proposed], [source])[0].status, "unsupported")

    def test_legal_obligation_cannot_reverse_source_negation(self):
        source = Evidence("ext-law", "The rule does not require student consent.")
        claim = Claim("claim-1", "The rule requires student consent.", "external", ["ext-law"])
        proposed = ClaimAssessment("claim-1", "supported", [
            ClaimQuote("ext-law", "The rule does not require student consent")], [])
        self.assertEqual(review_claims([claim], [proposed], [source])[0].status, "unsupported")

    def test_c_partial_support(self):
        result = self.check("The survey proves webcams cause privacy concerns [ext1].",
            "partially_supported", [("ext1", "59% of students expressed privacy concerns")], section="external")
        self.assertEqual(result.status, "partially_supported")
        self.assertIn("only partly", present_claims([result], "external"))
        invented = self.check("The survey found 95% expressed privacy concerns [ext1].",
            "partially_supported", [("ext1", "59% of students expressed privacy concerns")],
            section="external")
        self.assertEqual(invented.status, "unsupported")

    def test_partial_answer_displays_only_verified_excerpt(self):
        source = Evidence("c1", "Study B used matched courses with 780 students.")
        claim = Claim("claim-1", "Study B used matched courses with 780 students and proved "
                      "the design was superior.", "pdf", ["c1"])
        assessment = ClaimAssessment("claim-1", "partially_supported", [
            ClaimQuote("c1", "Study B used matched courses with 780 students")], [])
        reviewed = review_claims([claim], [assessment], [source])
        shown = present_claims(reviewed, "pdf")
        self.assertIn("Study B used matched courses with 780 students", shown)
        self.assertIn("[c1]", shown)
        self.assertNotIn("superior", shown)
        self.assertIn("remaining detail could not be verified", shown)

    def test_d_two_external_sources_conflict(self):
        result = self.check("Webcams increase privacy concerns [ext1].", "contradicted",
            [("ext1", "59% of students expressed privacy concerns")],
            [("ext2", "no increase in privacy concerns")], section="external")
        self.assertEqual(result.status, "contradicted")
        rendered = present_claims([result], "external")
        self.assertIn("External source [ext1] reports:", rendered)
        self.assertIn("External source [ext2] reports:", rendered)
        self.assertIn("does not resolve", rendered)

    def test_e_claim_without_source(self):
        result = self.check("The study was randomized.", "supported",
            [("c1", "The study included 412 students")])
        self.assertEqual(result.status, "unsupported")

    def test_f_pdf_and_external_sources_conflict(self):
        result = self.check("The PDF found no increase in privacy concerns [c2].", "contradicted",
            [("c2", "no increase in privacy concerns with webcam monitoring")],
            [("ext3", "webcam monitoring increased privacy concerns")])
        self.assertEqual(result.status, "contradicted")
        self.assertIn("[c2]", present_claims([result], "pdf"))
        self.assertIn("[ext3]", present_claims([result], "pdf"))
        self.assertIn("PDF [c2] reports:", present_claims([result], "pdf"))
        self.assertIn("External source [ext3] reports:", present_claims([result], "pdf"))

    def test_conflict_requires_two_distinct_sources(self):
        result = self.check("The study included 412 students [c1].", "contradicted",
            [("c1", "The study included 412 students")],
            [("c1", "did not report a p-value")])
        self.assertEqual(result.status, "unsupported")
        self.assertNotIn("findings differ", present_claims([result], "pdf"))

    def test_similar_findings_are_not_presented_as_conflict(self):
        result = self.check("Webcams increase privacy concerns [ext3].", "contradicted",
            [("ext3", "webcam monitoring increased privacy concerns")],
            [("ext1", "students expressed privacy concerns")], section="external")
        self.assertEqual(result.status, "unsupported")
        self.assertNotIn("findings differ", present_claims([result], "external"))

    def test_different_populations_are_not_direct_contradictions(self):
        self.sources.extend([
            Evidence("ext4", "Webcam monitoring increased privacy concerns among first-year students."),
            Evidence("ext5", "Webcam monitoring did not increase privacy concerns among graduate students."),
        ])
        result = self.check("Webcam monitoring increases privacy concerns [ext4].", "contradicted",
            [("ext4", "Webcam monitoring increased privacy concerns among first-year students")],
            [("ext5", "Webcam monitoring did not increase privacy concerns among graduate students")],
            section="external")
        self.assertEqual(result.status, "unsupported")

    def test_non_significant_result_is_not_proof_of_no_effect(self):
        self.sources.extend([
            Evidence("ext4", "Monitoring increased integrity outcomes in a randomized trial."),
            Evidence("ext5", "Monitoring increased integrity outcomes, but the difference was not statistically significant."),
        ])
        result = self.check("Monitoring improves integrity outcomes [ext4].", "contradicted",
            [("ext4", "Monitoring increased integrity outcomes in a randomized trial")],
            [("ext5", "Monitoring increased integrity outcomes, but the difference was not statistically significant")],
            section="external")
        self.assertEqual(result.status, "unsupported")

    def test_g_outdated_verified_external_source(self):
        result = self.check("Currently, 59% of students express privacy concerns [ext1].",
            "supported", [("ext1", "59% of students expressed privacy concerns")],
            section="external", dates={"ext1": date(2020, 1, 1)})
        self.assertEqual(result.status, "partially_supported")

    def test_undated_external_source_cannot_establish_current_condition(self):
        result = self.check("Currently, 59% of students express privacy concerns [ext1].",
            "supported", [("ext1", "59% of students expressed privacy concerns")],
            section="external")
        self.assertEqual(result.status, "partially_supported")

    def test_invented_quote_fails_closed(self):
        result = self.check("The study included 412 students [c1].", "supported",
            [("c1", "The study included 412 students and was randomized")])
        self.assertEqual(result.status, "unsupported")

    def test_quote_normalization_keeps_percent_and_numeric_boundaries(self):
        self.sources.append(Evidence("c3", "The rate was 12.50%—not 18.50%."))
        valid = self.check("The rate was 12.50% [c3].", "supported",
                           [("c3", "The rate was 12.50 %")])
        self.assertEqual(valid.status, "supported")
        invented = self.check("The rate was 12.5% [c3].", "supported",
                              [("c3", "The rate was 12.5 %")])
        self.assertEqual(invented.status, "unsupported")

    def test_citation_after_punctuation_stays_with_claim(self):
        claims = split_claims("The study included 412 students. [c1] It reported no p-value. [c1]", "")
        self.assertEqual(len(claims), 2)
        self.assertEqual([item.cited_ids for item in claims], [["c1"], ["c1"]])

    def test_multiple_citations_after_sentence_stay_with_same_claim(self):
        claims = split_claims("Study A and Study B report different findings. [cA] [cB] "
                              "Both raise privacy concerns. [cC]", "")
        self.assertEqual(len(claims), 2)
        self.assertEqual([item.cited_ids for item in claims], [["cA", "cB"], ["cC"]])

    def test_citation_after_newline_and_abbreviation_stays_with_claim(self):
        claims = split_claims("Dr. Rao reported 12.5% versus 18.5%.\n[c1]\nThe result was limited. [c2]", "")
        self.assertEqual(len(claims), 2)
        self.assertEqual([item.cited_ids for item in claims], [["c1"], ["c2"]])

    def test_bullets_and_semicolons_keep_citations_with_own_claims(self):
        claims = split_claims("- Study A enrolled 2,400 students [cA]; Study B enrolled 780 students [cB].\n"
                              "- Privacy concerns rose [cC].", "")
        self.assertEqual(len(claims), 3)
        self.assertEqual([item.cited_ids for item in claims], [["cA"], ["cB"], ["cC"]])

    def test_missing_or_duplicate_assessment_only_withholds_affected_claim(self):
        claims = split_claims("The study included 412 students [c1]. It reported a p-value of 0.01 [c2].", "")
        good = ClaimAssessment(claims[0].id, "supported",
                               [ClaimQuote("c1", "The study included 412 students")], [])
        reviewed = review_claims(claims, [good], self.sources)
        self.assertEqual([item.status for item in reviewed], ["supported", "unsupported"])
        duplicate = review_claims(claims, [good, good], self.sources)
        self.assertEqual([item.status for item in duplicate], ["unsupported", "unsupported"])

    def test_paraphrase_with_source_wording_and_three_universities(self):
        self.sources.append(Evidence("c3", "Study B enrolled students at three universities."))
        result = self.check("Study B involved 3 universities [c3].", "supported",
                            [("c3", "students at three universities")])
        self.assertEqual(result.status, "supported")

    def test_calculated_total_requires_both_quoted_inputs(self):
        self.sources.extend([Evidence("c3", "Study A enrolled 1,200 participants."),
                             Evidence("c4", "Study B enrolled 1,980 participants.")])
        result = self.check("The combined total was 3,180 participants [c3] [c4].", "supported",
                            [("c3", "Study A enrolled 1,200 participants"),
                             ("c4", "Study B enrolled 1,980 participants")])
        self.assertEqual(result.status, "supported")
        wrong = self.check("The combined total was 3,190 participants [c3] [c4].", "supported",
                           [("c3", "Study A enrolled 1,200 participants"),
                            ("c4", "Study B enrolled 1,980 participants")])
        self.assertEqual(wrong.status, "unsupported")

    def test_percentage_point_difference_not_percent_change(self):
        self.sources.append(Evidence("c3", "Study A reported 18% before and 12% after monitoring."))
        result = self.check("Study A's difference was 6 percentage points [c3].", "supported",
                            [("c3", "18% before and 12% after")])
        self.assertEqual(result.status, "supported")
        wrong_unit = self.check("Study A's difference was 6% [c3].", "supported",
                                [("c3", "18% before and 12% after")])
        self.assertEqual(wrong_unit.status, "unsupported")

    def test_rounded_percentage_uses_quoted_operands(self):
        self.sources.append(Evidence("c3", "One of three respondents agreed."))
        result = self.check("About 33.3% agreed [c3].", "supported",
                            [("c3", "One of three respondents agreed")])
        self.assertEqual(result.status, "supported")
        wrong = self.check("About 34.3% agreed [c3].", "supported",
                           [("c3", "One of three respondents agreed")])
        self.assertEqual(wrong.status, "unsupported")

    def test_unconfirmed_alerts_are_derived_from_reported_counts(self):
        self.sources.append(Evidence("c3", "Study B recorded 120 alerts, of which 9 were confirmed."))
        result = self.check("Study B had 111 unconfirmed alerts [c3].", "supported",
                            [("c3", "120 alerts, of which 9 were confirmed")])
        self.assertEqual(result.status, "supported")

    def test_alert_share_is_recomputed_from_quoted_counts(self):
        self.sources.append(Evidence("c3", "Study B recorded 120 alerts, 9 linked to confirmed misconduct."))
        result = self.check("7.5% of alerts were linked to confirmed misconduct [c3].", "supported",
                            [("c3", "120 alerts, 9 linked to confirmed misconduct")])
        self.assertEqual(result.status, "supported")
        wrong = self.check("75% of alerts were linked to confirmed misconduct [c3].", "supported",
                           [("c3", "120 alerts, 9 linked to confirmed misconduct")])
        self.assertEqual(wrong.status, "unsupported")

    def test_unconfirmed_alerts_are_not_automatically_false_positives(self):
        self.sources.append(Evidence("c3", "Study B had 111 alerts not confirmed as misconduct."))
        result = self.check("All 111 alerts were false positives [c3].", "supported",
                            [("c3", "111 alerts not confirmed as misconduct")])
        self.assertEqual(result.status, "unsupported")

    def test_pdf_does_not_establish_facial_recognition_accuracy(self):
        self.sources.append(Evidence("c3", "Facial recognition accuracy was not evaluated in this study."))
        result = self.check("The study did not evaluate facial recognition accuracy [c3].", "supported",
                            [("c3", "Facial recognition accuracy was not evaluated")])
        self.assertEqual(result.status, "supported")

    def test_unsupported_claim_does_not_erase_supported_claim(self):
        supported = self.check("The study included 412 students [c1].", "supported",
                               [("c1", "The study included 412 students")])
        unsupported = self.check("The p-value was 0.01 [c1].", "supported",
                                 [("c1", "did not report a p-value")])
        shown = present_claims([supported, unsupported], "pdf")
        self.assertIn("412 students", shown)
        self.assertNotIn("p-value was 0.01", shown)
        self.assertNotIn("claims were withheld", shown)

    def test_different_study_findings_remain_separate_supported_claims(self):
        evidence = [Evidence("cA", "Study A reported 8% monitored and 14% unmonitored misconduct."),
                    Evidence("cB", "Study B reported 9% monitored and 10% comparison misconduct; "
                             "the difference was not statistically significant.")]
        claims = split_claims(
            "Study A found 8% versus 14% misconduct [cA]. "
            "Study B found 9% versus 10%, without a statistically significant difference [cB].", "")
        proposed = [
            ClaimAssessment(claims[0].id, "supported", [ClaimQuote("cA", "8% monitored and 14% unmonitored misconduct")], []),
            ClaimAssessment(claims[1].id, "supported", [ClaimQuote("cB", "9% monitored and 10% comparison misconduct; the difference was not statistically significant")], []),
        ]
        reviewed = review_claims(claims, proposed, evidence)
        self.assertEqual([item.status for item in reviewed], ["supported", "supported"])
        shown = present_claims(reviewed, "pdf")
        self.assertIn("[cA]", shown)
        self.assertIn("[cB]", shown)
        self.assertNotIn("The cited findings differ", shown)

    def test_four_methodology_facts_are_verified_independently(self):
        evidence = [Evidence("cA", "Study A setting: 12 universities. Participants: 2,400 students. "
                             "Design: Cluster-randomized comparison. Observation period: One academic semester.")]
        answer = ("Study A used a cluster-randomized comparison, included 2,400 students, "
                  "involved 12 universities, and lasted one academic semester [cA].")
        claims = split_claims(answer, "")
        self.assertEqual(len(claims), 4)
        self.assertEqual([claim.cited_ids for claim in claims], [["cA"]] * 4)
        quotes = ["Cluster-randomized comparison", "Participants: 2,400 students",
                  "Study A setting: 12 universities", "Observation period: One academic semester"]
        proposed = [ClaimAssessment(claim.id, "supported", [ClaimQuote("cA", quote)], [])
                    for claim, quote in zip(claims, quotes)]
        reviewed = review_claims(claims, proposed, evidence)
        self.assertEqual([item.status for item in reviewed], ["supported"] * 4)

    def test_one_unsupported_compound_fact_does_not_erase_other_facts(self):
        evidence = [Evidence("cA", "Study A setting: 12 universities. Participants: 2,400 students. "
                             "Design: Cluster-randomized comparison. Observation period: One academic semester.")]
        claims = split_claims("Study A used a cluster-randomized comparison, included 2,400 students, "
                              "involved 99 universities, and lasted one academic semester [cA].", "")
        quotes = ["Cluster-randomized comparison", "Participants: 2,400 students",
                  "Study A setting: 12 universities", "Observation period: One academic semester"]
        proposed = [ClaimAssessment(claim.id, "supported", [ClaimQuote("cA", quote)], [])
                    for claim, quote in zip(claims, quotes)]
        reviewed = review_claims(claims, proposed, evidence)
        self.assertEqual([item.status for item in reviewed],
                         ["supported", "supported", "unsupported", "supported"])
        shown = present_claims(reviewed, "pdf")
        self.assertIn("2,400 students", shown)
        self.assertNotIn("99 universities", shown)

    def test_compound_split_preserves_qualifier_and_does_not_split_arithmetic(self):
        claims = split_claims("The study included 2,400 undergraduate students, "
                              "was conducted at 12 universities, and lasted one academic semester [cA].", "")
        self.assertEqual(len(claims), 3)
        self.assertIn("undergraduate", claims[0].text)
        self.assertTrue(all(claim.cited_ids == ["cA"] for claim in claims))
        self.assertEqual(len(split_claims("The combined total was 3,180 students, not a pooled experiment [cA].", "")), 1)

    def test_verification_logs_reason_and_ids_without_claim_text(self):
        claims = split_claims("The study included 900 students [c1].", "")
        proposal = [ClaimAssessment(claims[0].id, "supported",
                                    [ClaimQuote("c1", "The study included 412 students")], [])]
        with self.assertLogs("app.services.claim_verification", level="INFO") as captured:
            reviewed = review_claims(claims, proposal, self.sources)
        self.assertEqual(reviewed[0].status, "unsupported")
        self.assertIn("claim_id=claim-1", captured.output[0])
        self.assertIn("source_ids=['c1']", captured.output[0])
        self.assertIn("numeric_match=False", captured.output[0])
        self.assertIn("reason=numeric_mismatch", captured.output[0])
        self.assertNotIn("900 students", captured.output[0])


if __name__ == "__main__":
    unittest.main()
