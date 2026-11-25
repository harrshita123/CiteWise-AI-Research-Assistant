"""Persist the outcomes of existing checks without exposing the audit in public APIs."""

import re
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit

from .evidence_coverage import CoveragePassage, coverage_report

Outcome = Literal["PASS", "FAIL", "UNAVAILABLE", "NOT_APPLICABLE"]


class VerificationReport:
    def __init__(self, scope: Literal["document", "answer"], previous=None):
        self.scope = scope
        self.timestamp = datetime.now(timezone.utc)
        self.checks = {item["check_id"]: item for item in previous or []}

    def record(self, check_id: str, status: Outcome, reason_code: str, references=()):
        self.checks[check_id] = {
            "check_id": check_id, "scope": self.scope,
            "priority": "HIGH" if check_id.startswith(("H", "AH")) else "MEDIUM",
            "status": status, "reason_code": reason_code,
            "evidence_refs": list(dict.fromkeys(references)), "checked_at": self.timestamp,
        }

    def passes(self, *required: str) -> bool:
        return all(self.checks.get(check, {}).get("status") == "PASS" for check in required)

    def dump(self):
        return list(self.checks.values())


def ready_for_research(paper) -> bool:
    if paper.get("status") != "ready":
        return False
    if "document_state" not in paper:
        return True  # Pre-gate library records keep their existing readiness.
    return paper["document_state"] == "READY_FOR_RESEARCH" and VerificationReport(
        "document", paper.get("verification_checks")).passes("H1", "H2", "H3", "H4")


def document_quality(report, metadata, identity, *, usable_text):
    status = identity.get("status", "pending")
    verified = status in {"verified_published", "verified_preprint", "retracted"}
    conflict = status == "conflicting"
    reason = identity.get("reason_code") or ("corroborated_identity" if verified else "identity_not_established")
    identity_conflict = conflict or reason == "identifier_not_corroborated"
    report.record("H5", "PASS" if verified else "FAIL" if identity_conflict else "UNAVAILABLE", reason)
    for check, field in (("M1", "title"), ("M2", "authors"), ("M3", "abstract"),
                         ("M4", "year"), ("M5", "venue")):
        if metadata.get(field):
            report.record(check, "PASS", "pdf_field_extracted")
        elif verified and identity.get(field):
            report.record(check, "PASS", "corroborated_record_field")
        elif field == "venue" and status == "verified_preprint":
            report.record(check, "NOT_APPLICABLE", "published_venue_not_required_for_preprint")
        else:
            report.record(check, "UNAVAILABLE", "field_not_established")
    identifiers = ("doi", "arxiv_id", "pmid", "pmcid")
    found = any(metadata.get(field) for field in identifiers)
    if reason == "identifier_not_corroborated" or (conflict and found):
        report.record("M6", "FAIL", "identifier_not_corroborated")
    elif verified and any(metadata.get(field) and metadata[field] == identity.get(field) for field in identifiers):
        report.record("M6", "PASS", "identifier_corroborated")
    else:
        report.record("M6", "UNAVAILABLE" if found else "NOT_APPLICABLE",
                      "identifier_check_incomplete" if found else "no_identifier_extracted")
    report.record("M7", "PASS" if verified or metadata.get("version") else "UNAVAILABLE",
                  "edition_identified" if verified or metadata.get("version") else "edition_not_established")
    corroborating_sources = len(set((identity.get("source") or "").split(", "))) > 1
    report.record("M8", "FAIL" if conflict else "PASS" if verified and corroborating_sources else "UNAVAILABLE",
                  "cross_source_comparison_unavailable" if verified and not corroborating_sources else reason)
    report.record("M9", "PASS" if usable_text else "UNAVAILABLE",
                  "page_mapped_chunks_indexed" if usable_text else "index_not_ready")
    missing = any(item["status"] != "PASS" for key, item in report.checks.items() if key in {"M1", "M2", "M3", "M4", "M5"})
    report.record("M10", "PASS", "metadata_gaps_recorded" if missing else "metadata_fields_present")


def answer_report(*, status, answer, claims, sources, external, question, unavailable):
    report = VerificationReport("answer")
    known = {item.id for item in [*sources, *external]}
    pdf_ids = {item.id for item in sources}
    web_ids = {item.id for item in external}
    quotes = [quote for claim in claims for quote in claim.supporting + claim.opposing]
    refs = [quote.source_id for quote in quotes if quote.source_id in known]
    supported = [claim for claim in claims if claim.status == "supported" and claim.supporting]
    grounded = bool(claims) and len(supported) == len(claims)
    citations_valid = bool(quotes) and len(refs) == len(quotes)
    provenance = bool(supported) and all(
        all(quote.source_id in (pdf_ids if claim.section == "pdf" else web_ids) for quote in claim.supporting)
        for claim in supported)
    complete = grounded and bool(answer and answer.strip()) and (
        status == "answered" or status == "not_mentioned" and any(claim.section == "external" for claim in supported))
    conflict = any(claim.status == "contradicted" for claim in claims)
    numeric = [claim for claim in claims if re.search(r"\d", re.sub(r"\[[^\]]+\]", "", claim.text))]

    def gate(check, passed, reason, *, applicable=True):
        outcome = ("UNAVAILABLE" if unavailable else "NOT_APPLICABLE" if not applicable
                   else "PASS" if passed else "FAIL")
        report.record(check, outcome, "provider_unavailable" if unavailable else reason, refs)

    gate("AH1", bool(supported), "claim_relevance_checked" if supported else "no_direct_verified_claim")
    gate("AH2", complete, "sufficient_evidence" if complete else "requested_answer_not_established")
    gate("AH3", grounded, "all_claims_grounded" if grounded else "claims_missing_or_not_fully_grounded")
    coverage = citations_valid and all(claim.supporting or claim.opposing for claim in claims)
    gate("AH4", coverage, "retained_quotes_mapped" if coverage else "missing_or_unmapped_citation")
    gate("AH5", all(claim in supported for claim in numeric),
         "numeric_claims_reviewed" if all(claim in supported for claim in numeric) else "numeric_claim_not_verified",
         applicable=bool(numeric))
    gate("AH6", provenance and not any(claim.reason_code == "attribution_only" for claim in claims),
         "source_sections_checked" if provenance else "source_provenance_not_established")
    gate("AH7", bool(claims) and not conflict, "contradictory_evidence_preserved" if conflict else "opposition_reviewed")

    report.record("AM1", report.checks["AH2"]["status"], report.checks["AH2"]["reason_code"], refs)
    readable = all(len(sentence.split()) <= 80 for sentence in re.split(r"(?<=[.!?])\s+|\n", answer or ""))
    gate("AM2", readable, "sentence_length_checked" if readable else "long_sentence_detected", applicable=bool(answer))
    gate("AM3", citations_valid and bool(supported), "retained_sources_reviewed_for_relevance", applicable=bool(sources or external))
    hosts = {urlsplit(item.url).hostname for item in external}
    gate("AM4", len(hosts) > 1, "distinct_source_hosts" if len(hosts) > 1 else "limited_source_diversity",
         applicable=bool(external) and bool(re.search(r"\b(?:compare|independent|sources|studies|replicated)\b", question, re.I)))
    passages = [CoveragePassage(str(index), claim.text) for index, claim in enumerate(supported)]
    checked_dimensions = coverage_report(question, passages, {item.id for item in passages})
    for check, dimensions in (("AM5", {"design", "participants", "duration", "intervention", "measurement"}),
                              ("AM6", {"limitations"})):
        matches = [item for item in checked_dimensions if item.dimension in dimensions]
        passed = all(item.status == "verified_claim_support" for item in matches)
        gate(check, passed, "requested_dimension_covered" if passed else "requested_dimension_incomplete",
             applicable=bool(matches))
    gate("AM7", conflict and citations_valid, "opposing_passages_retained", applicable=conflict)
    gate("AM8", bool(answer), "qualified_output_retained", applicable=not complete)
    gate("AM9", grounded and citations_valid, "material_claims_have_citations" if grounded and citations_valid else "incomplete_citation_coverage")
    gate("AM10", len((answer or "").split()) <= 350, "word_budget_checked", applicable=bool(answer))

    required = ("AH1", "AH2", "AH3", "AH4", "AH6", "AH7") + (("AH5",) if numeric else ())
    if unavailable:
        state = "VERIFICATION_UNAVAILABLE"
    elif report.passes(*required):
        state = "VERIFIED"
    elif any(claim.supporting or claim.opposing for claim in claims):
        state = "PARTIALLY_SUPPORTED"
    else:
        state = "INSUFFICIENT_EVIDENCE"
    return state, report.dump()
