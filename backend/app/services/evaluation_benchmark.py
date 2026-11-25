"""Versioned benchmark records and scoring; gold data never enters retrieval."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# These are labels for analysis, not a claim that every label is a study design.
TAXONOMY = {
    "purpose": ("Exploratory", "Descriptive", "Explanatory", "Predictive", "Optimization"),
    "design": ("Experimental", "Longitudinal", "Cross-Sectional", "Case Study", "Systematic Review", "Meta-Analysis"),
    "approach": ("Theoretical", "Empirical", "Quantitative", "Qualitative", "Mixed-Methods", "Interdisciplinary"),
    "contribution": ("Methodological", "Replication", "Validation", "Research Gap Analysis"),
    "evidence": ("Contradictory Findings", "Conflicting Evidence", "Conflicting Answers"),
    "reasoning": ("Comparative", "Causal", "Correlational", "Hypothesis Testing", "Critical Analysis", "Ethical Analysis"),
}
SKILLS = ("Agreement", "Disagreement", "Contradiction", "Trade-offs", "Uncertainty", "Bias",
          "Generalization", "Robustness", "Novelty", "Alternative Explanations", "Limitations",
          "Reproducibility", "Consensus", "Open Questions", "Future Directions")


class EvidenceRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    page: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    content_hash: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ResearchLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    axis: Literal["purpose", "design", "approach", "contribution", "evidence", "reasoning"]
    label: str
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[EvidenceRef, ...] = ()
    needs_review: bool = True

    @model_validator(mode="after")
    def label_matches_axis(self):
        if self.label not in TAXONOMY[self.axis]:
            raise ValueError("Label does not belong to its taxonomy axis")
        if not self.evidence and not self.needs_review:
            raise ValueError("A label without evidence requires review")
        return self


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    paper_hash: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1000)
    expected_behavior: Literal["answer", "abstain"]
    expected_answer: str = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = ()
    labels: tuple[ResearchLabel, ...] = ()
    skills: tuple[str, ...] = ()
    review_status: Literal["pending", "approved"] = "pending"
    reviewed_by: str | None = None

    @model_validator(mode="after")
    def validate_case(self):
        if any(skill not in SKILLS for skill in self.skills):
            raise ValueError("Unknown evaluation skill")
        if self.review_status == "approved":
            if not self.reviewed_by or not self.reviewed_by.strip():
                raise ValueError("Approved cases require a reviewer")
            if self.expected_behavior == "answer" and not self.evidence:
                raise ValueError("Approved answerable cases require evidence")
            if any(label.needs_review for label in self.labels):
                raise ValueError("Approved cases cannot contain labels pending review")
        return self


class FrozenBenchmark(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: str = Field(min_length=1)
    cases: tuple[BenchmarkCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def only_reviewed_cases(self):
        if any(case.review_status != "approved" for case in self.cases):
            raise ValueError("A benchmark can contain only approved cases")
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        return self

    def fingerprint(self) -> str:
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    page: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    content_hash: str = Field(min_length=1)


class CaseObservation(BaseModel):
    """Human judgments are optional: unavailable metrics stay null, not zero."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    execution_status: Literal["completed", "rate_limited", "provider_error", "internal_error", "not_run", "blocked"] = "completed"
    returned_pages: tuple[int, ...] = ()
    returned_chunks: tuple[RetrievedChunk, ...] | None = None
    answer_correct: bool | None = None
    citations_supported: int | None = Field(default=None, ge=0)
    citations_total: int | None = Field(default=None, ge=0)
    claims_cited: int | None = Field(default=None, ge=0)
    claims_requiring_citation: int | None = Field(default=None, ge=0)
    refused: bool | None = None
    latency_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_counts(self):
        for numerator, denominator in ((self.citations_supported, self.citations_total),
                                       (self.claims_cited, self.claims_requiring_citation)):
            if (numerator is None) != (denominator is None) or (numerator is not None and numerator > denominator):
                raise ValueError("Judgment counts must be paired and numerator cannot exceed denominator")
        return self


def evaluate_benchmark(benchmark: FrozenBenchmark, observations: list[CaseObservation]) -> dict:
    cases = {case.id: case for case in benchmark.cases}
    if len({row.case_id for row in observations}) != len(observations):
        raise ValueError("Duplicate observation")
    if any(row.case_id not in cases for row in observations):
        raise ValueError("Observation references an unknown case")

    def metrics(rows: list[CaseObservation]) -> dict:
        total_count = len(rows)
        rows = [row for row in rows if row.execution_status == "completed"]
        def hit(row: CaseObservation) -> bool:
            refs = cases[row.case_id].evidence
            if row.returned_chunks is not None:
                expected = {(ref.page, ref.chunk_index, ref.content_hash) for ref in refs}
                return any((chunk.page, chunk.chunk_index, chunk.content_hash) in expected
                           for chunk in row.returned_chunks[:5])
            # Legacy manually submitted observations identify pages only.
            return bool(set(row.returned_pages[:5]) & {ref.page for ref in refs})
        hits = [hit(row) for row in rows if cases[row.case_id].expected_behavior == "answer"]
        correctness = [row.answer_correct for row in rows if row.answer_correct is not None]
        refusals = [row.refused for row in rows if cases[row.case_id].expected_behavior == "abstain" and row.refused is not None]
        latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
        def ratio(numerator: str, denominator: str) -> float | None:
            pairs = [(getattr(row, numerator), getattr(row, denominator)) for row in rows
                     if getattr(row, denominator) is not None]
            total = sum(pair[1] for pair in pairs)
            return sum(pair[0] for pair in pairs) / total if total else None
        return {"case_count": total_count, "completed_count": len(rows),
                "hit_at_5": sum(hits) / len(hits) if hits else None,
                "answer_correctness": sum(correctness) / len(correctness) if correctness else None,
                "citation_precision": ratio("citations_supported", "citations_total"),
                "citation_coverage": ratio("claims_cited", "claims_requiring_citation"),
                "unsupported_refusal_accuracy": sum(refusals) / len(refusals) if refusals else None,
                "median_answer_latency_ms": statistics.median(latencies) if latencies else None}

    groups = defaultdict(list)
    skills = defaultdict(list)
    documents = defaultdict(list)
    for row in observations:
        case = cases[row.case_id]
        documents[case.paper_id].append(row)
        for label in {label.label for label in case.labels}:
            groups[label].append(row)
        for skill in set(case.skills):
            skills[skill].append(row)
    return {"benchmark_version": benchmark.version, "benchmark_hash": benchmark.fingerprint(),
            "overall": metrics(observations), "by_category": {key: metrics(value) for key, value in sorted(groups.items())},
            "by_skill": {key: metrics(value) for key, value in sorted(skills.items())},
            "by_document": {key: metrics(value) for key, value in sorted(documents.items())}}
