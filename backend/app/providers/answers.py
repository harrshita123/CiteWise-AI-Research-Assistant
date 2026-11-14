from dataclasses import dataclass
from typing import Literal, Protocol


EvidenceStatus = Literal[
    "answered", "not_mentioned", "partially_supported",
    "insufficient_evidence", "contradictory_evidence",
]


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    evidence_ids: list[str]
    status: EvidenceStatus = "answered"


@dataclass(frozen=True)
class EvidenceQuote:
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class EvidenceAssessment:
    level: Literal["sufficient", "partial", "none"]
    chunk_ids: list[str]
    missing_information: str = ""
    supporting_quotes: list[EvidenceQuote] | None = None


@dataclass(frozen=True)
class ExternalEvidence:
    id: str
    title: str
    url: str
    content: str
    published_date: str | None = None
    verified_publication_date: str | None = None
    publisher: str | None = None
    author: str | None = None


@dataclass(frozen=True)
class ExternalAnswer:
    pdf_text: str
    external_text: str
    pdf_ids: list[str]
    external_ids: list[str]


@dataclass(frozen=True)
class ClaimQuote:
    source_id: str
    quote: str


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    status: Literal["supported", "partially_supported", "unsupported", "contradicted"]
    supporting: list[ClaimQuote]
    opposing: list[ClaimQuote]
    explanation: str = ""


class AnswerRateLimitError(Exception):
    def __init__(self, retry_after: str = "60"):
        self.retry_after = retry_after
        super().__init__("Answer provider rate limit reached.")


class AnswerProvider(Protocol):
    def generate(self, question: str, evidence: list[Evidence]) -> GeneratedAnswer: ...
    def assess(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment: ...
    def generate_with_external(self, question: str, pdf_evidence: list[Evidence],
                               external_evidence: list[ExternalEvidence], missing_information: str) -> ExternalAnswer: ...
    def verify_claims(self, question: str, claims: list[dict[str, str]],
                      evidence: list[Evidence]) -> list[ClaimAssessment]: ...
