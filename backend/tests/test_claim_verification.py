import unittest
from datetime import date

from app.providers.answers import ClaimAssessment, ClaimQuote, Evidence
from app.services.claim_verification import Claim, ReviewedClaim, _numbers_supported, present_claims, review_claims, split_claims


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

    def test_partial_presentation_preserves_supported_prose_and_deduplicates_evidence(self):
        supported = self.check("The study enrolled 412 students [c1].", "supported",
            [("c1", "The study included 412 students")])
        pdf_quote = ClaimQuote("c2", self.sources[1].text)
        web_quote = ClaimQuote("ext1", self.sources[2].text)
        partial = [
            ReviewedClaim(Claim(f"claim-{index}", "The study proved the effect.", section, [quote.source_id]),
                          "partially_supported", [quote], [], "", "attribution_only")
            for index, (section, quote) in enumerate(
                [("pdf", pdf_quote), ("pdf", pdf_quote), ("external", web_quote)], 2)
        ]
        pdf = present_claims([supported, *partial], "pdf")
        external = present_claims([supported, *partial], "external")
        self.assertIn("The study enrolled 412 students. [c1]", pdf)
        self.assertEqual(pdf.count(pdf_quote.quote), 1)
        self.assertEqual(pdf.count("[c2]"), 1)
        self.assertEqual(pdf.count("Independent confirmation was not established."), 1)
        self.assertNotIn("proved the effect", pdf)
        self.assertNotIn("[ext1]", pdf)
        self.assertIn("[ext1]", external)
        self.assertNotIn("[c2]", external)

    def test_quality_checks_on_independent_evidence_fixtures(self):
        cases = [
            ("pdf direct", "The core contained quartz", "The core contained quartz.", {}, "supported"),
            ("web direct", "Wetland restoration increased bird density", "Wetland restoration increased bird density.", {}, "supported"),
            ("topic only", "Rainfall increased last spring", "Rainfall increased last spring.",
             {"question_fit": "irrelevant"}, "unsupported"),
            ("official but missing outcome", "The registry recorded the enrollment date", "The registry recorded the enrollment date.",
             {"evidence_use": "insufficient"}, "unsupported"),
            ("promotion", "The charger saves 18% of electricity", "Our charger saves 18% of electricity.",
             {"evidence_use": "attribution_only"}, "partially_supported"),
            ("preprint replication", "The audit independently replicated the classification result",
             "The preprint reports a classification result from its original audit.",
             {"evidence_use": "attribution_only"}, "partially_supported"),
            ("universal estimate", "The vacancy rate is 12% everywhere",
             "The vacancy rate was 12% in the sampled district.", {"scope_preserved": False}, "partially_supported"),
            ("time and sample lost", "The dropout rate is 8%", "The dropout rate was 8% in the 2018 cohort.",
             {"scope_preserved": False}, "partially_supported"),
            ("missing unit", "The dropout rate is 8%", "The dropout count was 8 in the cohort.", {}, "unsupported"),
            ("synthetic as observed", "The glacier lost 12 meters", "The simulated glacier lost 12 meters.", {}, "partially_supported"),
            ("synthetic in context", "The simulated glacier lost 12 meters", "The simulated glacier lost 12 meters.", {}, "supported"),
            ("question partly addressed", "The archive lists the start date", "The archive lists the start date.",
             {"question_fit": "partial"}, "partially_supported"),
        ]
        for name, text, passage, assessment, expected in cases:
            with self.subTest(name=name):
                pdf = name in {"pdf direct", "synthetic as observed", "synthetic in context"}
                source = Evidence("c1" if pdf else "ext1", passage, {"origin": "pdf" if pdf else "web"})
                claim = Claim("claim-1", text, "pdf" if pdf else "external", [source.id])
                proposed = ClaimAssessment(claim.id, "supported", [ClaimQuote(source.id, passage)], [], **assessment)
                reviewed = review_claims([claim], [proposed], [source])[0]
                self.assertEqual(reviewed.status, expected)
                if expected == "unsupported":
                    self.assertEqual(reviewed.supporting, [])
                    self.assertNotIn(f"[{source.id}]", present_claims([reviewed], claim.section))
                if assessment.get("evidence_use") == "attribution_only":
                    self.assertIn("Independent confirmation was not established", present_claims([reviewed], claim.section))

    def test_direct_refutation_can_be_retained_without_support_for_the_claim(self):
        claim = Claim("claim-1", "The program increased recycling participation", "external", ["ext1"])
        source = Evidence("ext1", "The program did not increase recycling participation.", {"origin": "web"})
        proposed = ClaimAssessment(claim.id, "contradicted", [], [ClaimQuote(source.id, source.text)])
        reviewed = review_claims([claim], [proposed], [source])[0]
        self.assertEqual(reviewed.status, "contradicted")
        self.assertIn("[ext1]", present_claims([reviewed], "external"))

    def test_web_citation_cannot_splice_away_a_qualification(self):
        source = Evidence("ext1", "The vendor claimed, without independent confirmation, that energy use fell.", {"origin": "web"})
        claim = Claim("claim-1", "The vendor demonstrated that energy use fell", "external", ["ext1"])
        proposed = ClaimAssessment(claim.id, "supported", [ClaimQuote(source.id, "The vendor claimed ... that energy use fell")], [])
        reviewed = review_claims([claim], [proposed], [source])[0]
        self.assertEqual(reviewed.reason_code, "quote_mismatch")

    def test_negative_values_are_not_converted_to_positive(self):
        source = Evidence("c-temp", "The temperature changed by −5 degrees.")
        quote = ClaimQuote(source.id, source.text)
        for value, expected in (("−5", "supported"), ("5", "unsupported")):
            claim = split_claims(f"The temperature changed by {value} degrees [c-temp].", "")[0]
            reviewed = review_claims([claim], [ClaimAssessment(claim.id, "supported", [quote], [])], [source])[0]
            self.assertEqual(reviewed.status, expected)

    def test_measured_difference_does_not_establish_significance(self):
        source = Evidence("c-rate", "Trial Meridian observed 18% versus 12% improvement. "
                          "No significance test was reported.")
        claim = split_claims("Trial Meridian found a statistically significant difference [c-rate].", "")[0]
        quote = ClaimQuote(source.id, "Trial Meridian observed 18% versus 12% improvement")
        for proposed_status in ("supported", "partially_supported"):
            reviewed = review_claims([claim], [ClaimAssessment(claim.id, proposed_status, [quote], [])], [source])[0]
            self.assertEqual(reviewed.status, "partially_supported")
            self.assertNotIn("statistically significant", present_claims([reviewed], "pdf"))
        direct = split_claims("Trial Meridian observed 18% versus 12% improvement [c-rate].", "")[0]
        accepted = review_claims([direct], [ClaimAssessment(direct.id, "supported", [quote], [])], [source])[0]
        self.assertEqual(accepted.status, "supported")
        proof = split_claims("Trial Meridian proved the intervention improved outcomes [c-rate].", "")[0]
        overstated = review_claims([proof], [ClaimAssessment(proof.id, "supported", [quote], [])], [source])[0]
        self.assertEqual(overstated.status, "partially_supported")

    def test_association_does_not_establish_causation(self):
        source = Evidence("c-cohort", "The Delta cohort found screen time associated with shorter sleep.")
        quote = ClaimQuote(source.id, source.text)
        causal = split_claims("Screen time caused shorter sleep in the Delta cohort [c-cohort].", "")[0]
        reviewed = review_claims([causal], [ClaimAssessment(causal.id, "supported", [quote], [])], [source])[0]
        self.assertEqual(reviewed.status, "partially_supported")
        self.assertEqual(reviewed.reason_code, "inference_scope_mismatch")
        qualified = split_claims("The Delta cohort found an association without establishing causation [c-cohort].", "")[0]
        calibrated = review_claims([qualified], [ClaimAssessment(qualified.id, "supported", [quote], [])], [source])[0]
        self.assertEqual(calibrated.status, "supported")

    def test_measurement_does_not_alone_establish_effectiveness(self):
        source = Evidence("c-training", "The training group completed 14 tasks; the comparison group completed 10.")
        claim = split_claims("The training was effective [c-training].", "")[0]
        quote = ClaimQuote(source.id, source.text)
        overconfident = review_claims([claim], [ClaimAssessment(claim.id, "supported", [quote], [])], [source])[0]
        self.assertEqual(overconfident.status, "partially_supported")
        self.assertNotIn("was effective", present_claims([overconfident], "pdf"))
        explicit = Evidence("c-author", "The authors interpreted the training as effective for task completion.")
        interpreted = split_claims("The authors interpreted the training as effective for task completion [c-author].", "")[0]
        supported = review_claims([interpreted], [ClaimAssessment(interpreted.id, "supported", [
            ClaimQuote(explicit.id, explicit.text)], [])], [explicit])[0]
        self.assertEqual(supported.status, "supported")
        negated = Evidence("c-negative", "The authors concluded that the training was not effective.")
        overstated = split_claims("The training was effective [c-negative].", "")[0]
        rejected = review_claims([overstated], [ClaimAssessment(overstated.id, "supported", [
            ClaimQuote(negated.id, negated.text)], [])], [negated])[0]
        self.assertNotEqual(rejected.status, "supported")

    def test_conflicting_findings_and_missing_statistic_keep_their_qualifications(self):
        higher = Evidence("c-sleep-a", "The first survey found higher sleep quality among night-shift workers.")
        lower = Evidence("c-sleep-b", "The second survey found lower sleep quality among night-shift workers.")
        claim = split_claims("The surveys report conflicting sleep quality findings [c-sleep-a].", "")[0]
        reviewed = review_claims([claim], [ClaimAssessment(claim.id, "contradicted", [
            ClaimQuote(higher.id, higher.text)], [ClaimQuote(lower.id, lower.text)])], [higher, lower])[0]
        self.assertEqual(reviewed.status, "contradicted")
        self.assertIn("[c-sleep-b]", present_claims([reviewed], "pdf"))
        missing = Evidence("c-review", "The review summarized themes but did not report a p-value.")
        gap = split_claims("The review did not report a p-value [c-review].", "")[0]
        gap_review = review_claims([gap], [ClaimAssessment(gap.id, "supported", [
            ClaimQuote(missing.id, missing.text)], [])], [missing])[0]
        self.assertEqual(gap_review.status, "supported")

    def test_independent_contrast_keeps_supported_clause_and_withholds_overclaim(self):
        source = Evidence("c-pilot", "The field study did not establish whether the program worked.")
        claims = split_claims("The field study did not establish whether the program worked, "
                              "but the program was effective [c-pilot].", "")
        self.assertEqual(len(claims), 2)
        self.assertEqual([item.cited_ids for item in claims], [[source.id], [source.id]])
        proposals = [ClaimAssessment(item.id, "supported", [ClaimQuote(source.id, source.text)], [])
                     for item in claims]
        reviewed = review_claims(claims, proposals, [source])
        self.assertEqual([item.status for item in reviewed], ["supported", "unsupported"])
        shown = present_claims(reviewed, "pdf")
        self.assertIn("did not establish", shown)
        self.assertNotIn("was effective", shown)

    def test_partial_quote_must_address_the_claimed_outcome(self):
        relevant = Evidence("c-health", "Mortality fell among patients receiving the intervention.")
        unrelated = Evidence("c-attendance", "Attendance rose after the intervention.")
        for source, expected in ((relevant, "partially_supported"), (unrelated, "unsupported")):
            claim = split_claims(f"The intervention was effective at reducing mortality [{source.id}].", "")[0]
            review = review_claims([claim], [ClaimAssessment(claim.id, "supported", [
                ClaimQuote(source.id, source.text)], [])], [source])[0]
            self.assertEqual(review.status, expected)
            if source is unrelated:
                self.assertNotIn("Attendance rose", present_claims([review], "pdf"))

    def test_statistical_evidence_can_use_p_values_or_confidence_intervals(self):
        cases = [
            ("Review Rhea found a statistically significant reduction in stroke incidence",
             "Review Rhea found lower stroke incidence: pooled risk ratio 0.72 (95% CI 0.61 to 0.85).",
             "supported"),
            ("Study Ossa found a statistically significant change in water quality",
             "Study Ossa measured a change in water quality (p = 0.03).", "supported"),
            ("Review Rhea found a statistically significant reduction in stroke incidence",
             "Review Rhea found lower stroke incidence: pooled risk ratio 0.72 (95% CI 0.61 to 1.05).",
             "partially_supported"),
            ("Study Ossa found a statistically significant change in water quality",
             "Study Ossa measured an attendance change (p = 0.03).", "unsupported"),
        ]
        for index, (statement, passage, expected) in enumerate(cases):
            with self.subTest(index=index):
                source = Evidence(f"c-stat-{index}", passage)
                claim = split_claims(f"{statement} [{source.id}].", "")[0]
                review = review_claims([claim], [ClaimAssessment(claim.id, "supported", [
                    ClaimQuote(source.id, source.text)], [])], [source])[0]
                self.assertEqual(review.status, expected)

    def test_design_and_reported_interpretation_preserve_supported_conclusions(self):
        observational = Evidence("c-cohort-x", "Cohort Vega observed exposure associated with lower lung function.")
        randomized = Evidence("c-trial-x", "Trial Sol randomized patients to treatment or control; "
                              "mortality was lower with treatment (p = 0.03).")
        for source, expected in ((observational, "partially_supported"), (randomized, "supported")):
            text = ("Exposure caused lower lung function" if source is observational else
                    "Treatment caused lower mortality")
            claim = split_claims(f"{text} [{source.id}].", "")[0]
            review = review_claims([claim], [ClaimAssessment(claim.id, "supported", [
                ClaimQuote(source.id, source.text)], [])], [source])[0]
            self.assertEqual(review.status, expected)
        qualitative = Evidence("c-interview", "Interview participants described peer support as helpful for coping.")
        synthesis = Evidence("c-synthesis", "The systematic review concluded peer support improved coping in most studies.")
        for source, text in ((qualitative, "Participants described peer support as effective for coping"),
                             (synthesis, "The review interpreted peer support as effective for coping")):
            claim = split_claims(f"{text} [{source.id}].", "")[0]
            review = review_claims([claim], [ClaimAssessment(claim.id, "supported", [
                ClaimQuote(source.id, source.text)], [])], [source])[0]
            self.assertEqual(review.status, "supported")
        negative = Evidence("c-negative-interview", "Interview participants described peer support as unhelpful for coping.")
        claim = split_claims("Participants described peer support as effective for coping [c-negative-interview].", "")[0]
        review = review_claims([claim], [ClaimAssessment(claim.id, "supported", [
            ClaimQuote(negative.id, negative.text)], [])], [negative])[0]
        self.assertNotEqual(review.status, "supported")

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
        self.assertIn("supports only part", present_claims([result], "external"))
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
        self.assertIn("supports only part", shown)

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

    def test_named_trial_clauses_keep_source_and_withhold_wrong_population(self):
        evidence = [Evidence("c-orbit", "Trial ORBIT used a randomized design and included 90 patients.")]
        claims = split_claims("Trial ORBIT used a randomized design, and included 100 patients [c-orbit].", "")
        self.assertEqual(len(claims), 2)
        self.assertEqual([item.cited_ids for item in claims], [["c-orbit"], ["c-orbit"]])
        proposals = [ClaimAssessment(item.id, "supported", [ClaimQuote("c-orbit", evidence[0].text)], [])
                     for item in claims]
        reviewed = review_claims(claims, proposals, evidence)
        self.assertEqual([item.status for item in reviewed], ["supported", "unsupported"])
        self.assertNotIn("100 patients", present_claims(reviewed, "pdf"))

    def test_repeated_contrasts_keep_independent_claims_and_local_citations(self):
        text = ("The survey reported improved attendance [c-attend], but the authors did not establish causation "
                "[c-caution], but the program proved effective [c-claim].")
        claims = split_claims(text, "")
        self.assertEqual(len(claims), 3)
        self.assertEqual([claim.cited_ids for claim in claims],
                         [["c-attend"], ["c-caution"], ["c-claim"]])
        sources = [Evidence("c-attend", "The survey reported improved attendance."),
                   Evidence("c-caution", "The authors did not establish causation."),
                   Evidence("c-claim", "The program reported improved attendance.")]
        proposed = [ClaimAssessment(claim.id, "supported", [ClaimQuote(claim.cited_ids[0], source.text)], [])
                    for claim, source in zip(claims, sources)]
        reviewed = review_claims(claims, proposed, sources)
        self.assertEqual([item.status for item in reviewed],
                         ["supported", "supported", "unsupported"])

    def test_significance_and_causation_must_match_claimed_outcome(self):
        source = Evidence("c-mixed", "A randomized trial found attendance improved (p = 0.03), "
                          "while mortality was unchanged.")
        for text in ("The trial found a statistically significant reduction in mortality",
                     "The treatment caused lower mortality"):
            with self.subTest(text=text):
                claim = split_claims(f"{text} [c-mixed].", "")[0]
                reviewed = review_claims([claim], [ClaimAssessment(claim.id, "supported", [
                    ClaimQuote(source.id, source.text)], [])], [source])[0]
                self.assertNotEqual(reviewed.status, "supported")

    def test_relevant_partial_requires_matching_proposition_and_direction(self):
        relevant = Evidence("c-lower", "Mortality fell after the intervention, but no test was reported.")
        unrelated = Evidence("c-upper", "Mortality rose after the intervention.")
        for source, expected in ((relevant, "partially_supported"), (unrelated, "unsupported")):
            claim = split_claims(f"The intervention was effective at reducing mortality [{source.id}].", "")[0]
            result = review_claims([claim], [ClaimAssessment(claim.id, "supported", [
                ClaimQuote(source.id, source.text)], [])], [source])[0]
            self.assertEqual(result.status, expected)

    def test_uncertain_qualitative_interpretation_is_not_an_established_effect(self):
        source = Evidence("c-interview-u", "Interviews described the peer program as uncertain for coping.")
        claim = split_claims("The peer program was effective for coping [c-interview-u].", "")[0]
        result = review_claims([claim], [ClaimAssessment(claim.id, "supported", [
            ClaimQuote(source.id, source.text)], [])], [source])[0]
        self.assertNotEqual(result.status, "supported")

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
