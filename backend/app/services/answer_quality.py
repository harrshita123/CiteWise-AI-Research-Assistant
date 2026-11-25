"""Deterministic evaluations of generated drafts and their existing claim reviews."""

from collections import Counter
from datetime import datetime, timezone
import logging
import re
from time import perf_counter

from .claim_verification import split_claims
from .evidence_coverage import CoveragePassage, coverage_report
from ..config import get_settings

VERSION = "1.0"
SUMMARY_METRICS = ("citation_coverage", "citation_precision", "claim_support", "completeness", "relevance", "contradiction_rate")
_EVIDENCE_ABSTENTION = re.compile(r"(?i)^(?:the )?(?:available |retrieved |provided |cited )?"
                                 r"(?:evidence|passages|sources|excerpts) (?:do|does) not "
                                 r"(?:establish|support|answer|provide|show)\b")


def _priority_status(metrics):
    statuses = {metric["status"] for metric in metrics} - {"NOT_APPLICABLE"}
    for status in ("FAIL", "PARTIAL", "UNAVAILABLE", "PASS"):
        if status in statuses:
            return status
    return "NOT_APPLICABLE"


class AnswerQuality:
    def __init__(self):
        self.started = perf_counter()
        self.draft = None
        self.rows = None
        self.assessments = {}
        self.requirements = []
        self.sources = {}

    def evaluate(self, question, message, *, revision, model_calls=()):
        try:
            result = self.build(question=question, message=message, revision=revision,
                              threshold=get_settings().answer_quality_pass_threshold)
        except Exception as exc:
            logging.getLogger(__name__).warning("answer quality evaluation failed error_type=%s", type(exc).__name__)
            result = {"revision": revision, "evaluator_version": VERSION, "evaluated_at": datetime.now(timezone.utc),
                    "state": "FAILED", "reason": "quality_evaluation_failed", "evidence_type": "none",
                    "high_status": "UNAVAILABLE", "medium_status": "UNAVAILABLE", "metrics": {}}
        if model_calls:
            result["model_calls"] = [dict(call) for call in model_calls]
        return result

    def capture(self, pdf, web, pdf_ids, web_ids, known_pdf, known_web, claims=None):
        self.assessments = {}
        self.requirements = []
        self.sources = {}
        self.draft = (pdf if isinstance(pdf, str) else "", web if isinstance(web, str) else "")
        if claims is None:
            claims = split_claims(*self.draft)
        self.rows = []
        for claim in claims:
            pool, known = (pdf_ids, known_pdf) if claim.section == "pdf" else (web_ids, known_web)
            pool = pool if isinstance(pool, list) else []
            refs = list(dict.fromkeys(pool[int(ref) - 1] if ref.isascii() and ref.isdecimal() and
                     len(ref) <= 3 and 1 <= int(ref) <= len(pool) and isinstance(pool[int(ref) - 1], str)
                     else ref for ref in claim.cited_ids))
            self.rows.append({"claim_id": claim.id, "text": claim.text, "section": claim.section,
                              "attached_ids": refs, "known_ids": [ref for ref in refs if ref in known]})
            if isinstance(known, dict):
                for ref in refs:
                    if source := known.get(ref):
                        self.sources[ref] = {"id": ref, **{key: value for key in ("page_number", "document_name", "title", "url")
                            if (value := getattr(source, key, None)) is not None}}

    def build(self, *, question, message, threshold, revision):
        started = perf_counter()
        unavailable = message.get("answer_state") == "VERIFICATION_UNAVAILABLE"
        metrics = {}
        stamp = datetime.now(timezone.utc)

        def metric(key, priority, reason, *, numerator=None, denominator=None, status=None, refs=()):
            score = numerator / denominator if denominator else None
            if status is None:
                status = ("UNAVAILABLE" if score is None else "PASS" if score >= threshold
                          else "PARTIAL" if score > 0 else "FAIL")
            metrics[key] = {"id": key, "priority": priority, "status": status, "score": score,
                            "numerator": numerator, "denominator": denominator, "evidence_refs": list(refs),
                            "reason": reason, "evaluated_at": stamp, "evaluator_version": VERSION}

        reviewed = message.get("claims") or []
        rows = []
        for index, raw in enumerate(self.rows or []):
            if not raw["attached_ids"] and _EVIDENCE_ABSTENTION.match(raw["text"]):
                continue
            review = reviewed[index] if len(reviewed) == len(self.rows) and reviewed[index]["text"] == raw["text"] else None
            support = {quote["source_id"] for quote in review["supporting"]} if review and review["status"] == "supported" else set()
            refs = raw["attached_ids"]
            judged = {quote["source_id"] for quote in review["supporting"] + review["opposing"]} if review else set()
            if review and (assessment := self.assessments.get(raw["claim_id"])):
                judged.update(assessment.unsupported_source_ids)
                judged.update(quote.source_id for quote in assessment.supporting + assessment.opposing)
                support.difference_update(assessment.unsupported_source_ids)
            assessed = [ref for ref in refs if ref in judged or ref not in raw["known_ids"]]
            rows.append({**raw, "status": review["status"] if review else "verification_unavailable" if unavailable or raw["known_ids"] else "unsupported",
                         "reason": review.get("reason_code") if review else "claim_review_unavailable" if raw["known_ids"] or unavailable else "missing_or_invalid_citation",
                         "supporting": review["supporting"] if review else [], "opposing": review["opposing"] if review else [],
                         "evaluated_ids": assessed, "supporting_ids": [ref for ref in assessed if ref in support]})
        statuses = Counter(row["status"] for row in rows)
        citations = sum(len(row["evaluated_ids"]) for row in rows)
        coverage_known = bool(rows) and all(row["status"] != "verification_unavailable" and (
            row["supporting_ids"] or len(row["evaluated_ids"]) == len(row["attached_ids"])) for row in rows)
        evidence = list(dict.fromkeys(ref for row in rows for ref in row["supporting_ids"]))
        metric("citation_coverage", "HIGH", "draft_claims_with_supporting_attached_citations",
               numerator=sum(bool(row["supporting_ids"]) for row in rows) if coverage_known else None,
               denominator=len(rows) if coverage_known else None, refs=evidence)
        metric("citation_precision", "HIGH", "supporting_draft_claim_citation_pairs" if citations else "no_citation_pairs_evaluated",
               numerator=sum(len(row["supporting_ids"]) for row in rows), denominator=citations, refs=evidence)
        classified = len(rows) - statuses["verification_unavailable"]
        metric("claim_support", "HIGH", "fully_supported_claims", numerator=statuses["supported"], denominator=classified)
        contradicted = statuses["contradicted"]
        metric("contradiction_rate", "HIGH", "contradicted_claims", numerator=contradicted, denominator=classified,
               status=("FAIL" if contradicted else "PASS") if classified else "UNAVAILABLE")
        judgments = [self.assessments[row["claim_id"]] for row in rows if row["claim_id"] in self.assessments]
        fits = [value.question_fit for value in judgments]
        metric("relevance", "HIGH", "direct_question_fit_from_claim_verifier", numerator=fits.count("direct"), denominator=len(fits))

        # Completeness uses addressed requirements, including reviewed attempts not
        # judged factually correct. Coverage hints intentionally recognize only
        # identifiable dimensions and cannot serve as a correctness score.
        addressed = [CoveragePassage(row["claim_id"], row["text"]) for row in rows
                     if row["status"] != "verification_unavailable" and
                     (assessment := self.assessments.get(row["claim_id"])) and assessment.question_fit == "direct"]
        addressed_ids = {part.id for part in addressed}
        requirements = []
        seen = set()
        for item in self.requirements[:10]:
            name = " ".join(item["requirement"].split())[:240]
            if name and name.casefold() not in seen:
                seen.add(name.casefold())
                requirements.append({"subject": "", "dimension": name,
                                     "addressed": any(claim in addressed_ids for claim in item["addressed_by"])})
        basis = "verifier_question_requirements" if requirements else "identifiable_question_requirements"
        if not requirements:
            requirements = [{"subject": part.subject, "dimension": part.dimension, "addressed": part.status == "information_present"}
                            for part in coverage_report(question, addressed)]
        metric("completeness", "MEDIUM", basis if requirements else "requirements_not_identifiable",
               numerator=sum(part["addressed"] for part in requirements) if classified else None,
               denominator=len(requirements) if classified else None)

        gates = {row["check_id"]: row for row in message.get("verification_checks", [])}
        for key, priority, gate in (("numerical_accuracy", "HIGH", "AH5"), ("source_attribution", "HIGH", "AH6"),
                ("clarity", "MEDIUM", "AM2"), ("uncertainty", "MEDIUM", "AM8"),
                ("research_context", "MEDIUM", "AM5"), ("conciseness", "MEDIUM", "AM10")):
            check = gates.get(gate, {})
            metric(key, priority, check.get("reason_code", "quality_check_unavailable"), status=check.get("status", "UNAVAILABLE"),
                   refs=check.get("evidence_refs", ()))
        kinds = [item.evidence_use for item in judgments]
        metric("source_quality", "MEDIUM", "fitness_for_claim_from_existing_verifier",
               numerator=kinds.count("establishes_claim"), denominator=len(kinds))
        metric("citation_distribution", "MEDIUM", "draft_claims_with_inline_citations",
               numerator=sum(bool(row["attached_ids"]) for row in rows), denominator=len(rows))
        metric("performance", "MEDIUM", "response_preparation_and_evaluation_time_recorded", status="PASS")
        pdf = bool(message.get("sources")) or bool(self.draft and self.draft[0])
        web = bool(message.get("external_sources")) or bool(self.draft and self.draft[1])
        sources = {**self.sources, **{
            item["id"]: {key: item[key] for key in ("id", "page_number", "document_name", "title", "url")
                         if item.get(key) is not None}
            for item in (message.get("sources") or []) + (message.get("external_sources") or [])}}
        return {"revision": revision, "evaluator_version": VERSION, "evaluated_at": stamp,
                "state": "UNAVAILABLE" if unavailable or not classified else "COMPLETE",
                "evidence_type": "mixed" if pdf and web else "external" if web else "pdf" if pdf else "none",
                "high_status": _priority_status(metric for metric in metrics.values() if metric["priority"] == "HIGH"),
                "medium_status": _priority_status(metric for metric in metrics.values() if metric["priority"] == "MEDIUM"),
                "metrics": metrics, "claim_counts": dict(statuses), "claims": rows, "requirements": requirements,
                "answer": message.get("answer"), "generated_answer": "\n\n".join(filter(None, self.draft or ()))[:12000] or None,
                "sources": list(sources.values()),
                "response_ms": round((started - self.started) * 1000, 2),
                "evaluation_ms": round((perf_counter() - started) * 1000, 2)}
