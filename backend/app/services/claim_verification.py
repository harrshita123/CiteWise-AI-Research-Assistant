"""Validate model claim assessments against the exact retrieved evidence."""

import re
import logging
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Literal

from ..providers.answers import ClaimAssessment, ClaimQuote, Evidence


ClaimStatus = Literal["supported", "partially_supported", "unsupported", "contradicted"]
_CITATION = re.compile(r"\[([A-Za-z0-9_-]+)\]")
_PAREN_SOURCE_CITATION = re.compile(r"\((c\d+-[0-9a-f]{8,}|ext[0-9a-f_-]{8,})\)", re.I)
_NUMERIC_CITATION = re.compile(
    r"(?:\[(?:c[A-Za-z0-9_-]+|ext[A-Za-z0-9_-]+|[1-9]\d{0,2})\]"
    r"|【(?:c[A-Za-z0-9_-]+|ext[A-Za-z0-9_-]+|[1-9]\d{0,2})】"
    r"|\((?:c\d+-[0-9a-f]{8,}|ext[0-9a-f_-]{8,})\))",
    re.I,
)
_FACT_VERB = r"(?:used|employed|included|enrolled|involved|comprised|was conducted|was carried out|lasted|spanned)"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Claim:
    id: str
    text: str
    section: Literal["pdf", "external"]
    cited_ids: list[str]


@dataclass(frozen=True)
class ReviewedClaim:
    claim: Claim
    status: ClaimStatus
    supporting: list[ClaimQuote]
    opposing: list[ClaimQuote]
    explanation: str


def split_claims(pdf_text: str, external_text: str) -> list[Claim]:
    claims = []
    for section, body in (("pdf", pdf_text), ("external", external_text)):
        # Generators often place a citation after sentence punctuation: "Fact. [id]".
        # Keep that citation with the fact when splitting into claims.
        body = re.sub(r"([.!?])\s+((?:\[[A-Za-z0-9_-]+\]\s*)+)",
                      lambda match: f" {match.group(2).strip()}{match.group(1)} ", body)
        body = re.sub(r"\b(?:Dr|Prof|Mr|Mrs|Ms|Fig|Sec|e\.g|i\.e|et al)\.",
                      lambda match: match.group().replace(".", "\ue000"), body, flags=re.I)
        for part in re.split(r"(?<=[.!?])\s+|[\n;]+", body):
            part = part.replace("\ue000", ".").strip(" -\t\r")
            if not part:
                continue
            cited = list(dict.fromkeys(_CITATION.findall(part) + _PAREN_SOURCE_CITATION.findall(part)))
            plain = _PAREN_SOURCE_CITATION.sub("", _CITATION.sub("", part)).strip()
            if not plain:
                continue
            # Split only a parallel list of independent facts about one explicit
            # subject. Keep the shared subject and sentence citation on each fact.
            subject = re.match(rf"^(Study\s+[A-Za-z0-9]+|The\s+(?:study|trial|paper))\s+{_FACT_VERB}\b",
                               plain, re.I)
            clauses = re.split(rf",\s+(?:and\s+)?(?={_FACT_VERB}\b)", plain, flags=re.I) if subject else [plain]
            for index, clause in enumerate(clauses):
                statement = clause.strip()
                if index:
                    statement = f"{subject.group(1)} {statement}"
                claims.append(Claim(f"claim-{len(claims) + 1}", statement, section, cited.copy()))
    return claims


def _verbatim(quote: ClaimQuote, evidence: dict[str, Evidence]) -> bool:
    source = evidence.get(quote.source_id)
    if source is None or len(quote.quote.strip()) < 8:
        return False
    normalize = lambda text: " ".join(re.findall(r"\d[\d,]*(?:\.\d+)?|[^\W\d_]+|%",
                                                  unicodedata.normalize("NFKC", text).casefold()))
    source_text = f" {normalize(source.text)} "
    cursor = 0
    for part in re.split(r"\s*(?:\.\.\.|…)+\s*", quote.quote):
        if not part.strip():
            continue
        fragment = f" {normalize(part)} "
        position = source_text.find(fragment, cursor)
        if position < 0:
            return False
        cursor = position + len(fragment) - 1
    return cursor > 0


def _numbers_supported(claim: str, quotes: list[ClaimQuote]) -> bool:
    """Permit only arithmetic that can be recomputed from quoted operands."""
    def rounded_match(actual: float, stated: float, decimals: int) -> bool:
        return abs(actual - stated) <= 0.5 * 10 ** (-decimals) + 1e-9

    numbers = lambda text: [float(value.replace(",", "")) for value in
                            re.findall(r"(?<![\w])\d[\d,]*(?:\.\d+)?", text)]
    numeric_claim = _NUMERIC_CITATION.sub("", claim)
    claimed = numbers(numeric_claim)
    cited_text = " ".join(_NUMERIC_CITATION.sub("", item.quote) for item in quotes)
    cited = numbers(cited_text)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    cited.extend(value for word, value in words.items()
                 if re.search(rf"\b{word}\b", cited_text, re.I))
    missing = [value for value in claimed if value not in cited]
    if not missing:
        return True
    if len(missing) != 1 or len(cited) < 2:
        return False
    value = missing[0]
    value_text = next((number for number in re.findall(r"(?<![\w])\d[\d,]*(?:\.\d+)?", numeric_claim)
                       if float(number.replace(",", "")) == value), "")
    decimals = len(value_text.partition(".")[2])
    if re.search(r"\b(?:combined|total|together|sum)\b", claim, re.I):
        return any(abs(a + b - value) < 1e-9 for a in cited for b in cited if a != b)
    if re.search(r"\b(?:difference|percentage.points?|more than|less than|unconfirmed)\b", claim, re.I):
        if "%" in claim and not re.search(r"percentage.points?", claim, re.I):
            return False
        return any(abs(abs(a - b) - value) < 1e-9 for a in cited for b in cited if a != b)
    if re.search(r"\b(?:percentage|percent|proportion|share)\b|%", claim, re.I):
        return any(b > 0 and 0 <= a <= b and rounded_match(100 * a / b, value, decimals)
                   for a in cited for b in cited if a != b)
    if re.search(r"\bratio\b", claim, re.I):
        return any(b > 0 and 0 <= a <= b and rounded_match(a / b, value, decimals)
                   for a in cited for b in cited if a != b)
    return False


def _explicit_opposition(support: list[ClaimQuote], oppose: list[ClaimQuote]) -> bool:
    """Require a visible opposite direction before displaying a contradiction."""
    negation = re.compile(r"\b(?:no|not|never|without)\b", re.I)
    increase = re.compile(r"\b(?:increas\w*|higher|ris\w*|improv\w*|more)\b", re.I)
    decrease = re.compile(r"\b(?:decreas\w*|reduc\w*|lower|fall\w*|less)\b", re.I)
    ignored = {"the", "and", "that", "with", "from", "this", "were", "was", "are", "for"}
    for first in support:
        for second in oppose:
            if (re.search(r"\b(?:not|non|no)\s+statistically\s+significant\b", first.quote, re.I)
                    or re.search(r"\b(?:not|non|no)\s+statistically\s+significant\b", second.quote, re.I)
                    or re.search(r"\bnot\s+significant\b", first.quote, re.I)
                    or re.search(r"\bnot\s+significant\b", second.quote, re.I)):
                continue
            population = lambda text: re.search(r"\bamong\s+([\w-]+(?:\s+[\w-]+){0,3}\s+students)\b",
                                                text, re.I)
            first_population, second_population = population(first.quote), population(second.quote)
            if (first_population and second_population
                    and first_population.group(1).casefold() != second_population.group(1).casefold()):
                continue
            first_words = set(re.findall(r"[a-z]{4,}", first.quote.casefold())) - ignored
            second_words = set(re.findall(r"[a-z]{4,}", second.quote.casefold())) - ignored
            opposite_direction = ((bool(increase.search(first.quote)) and bool(decrease.search(second.quote)))
                                  or (bool(decrease.search(first.quote)) and bool(increase.search(second.quote))))
            if (len(first_words & second_words) >= 2
                    and (bool(negation.search(first.quote)) != bool(negation.search(second.quote))
                         or opposite_direction)):
                return True
    return False


def _legal_scope_supported(claim: str, support: list[ClaimQuote]) -> bool:
    """Do not substitute a different named instrument or broaden a scoped rule."""
    quoted = " ".join(item.quote for item in support)
    instruments = re.findall(
        r"\b((?:[A-Z][A-Za-z0-9-]*\s+){1,8}(?:Act|Rules|Regulations|Directive)(?:,?\s+\d{4})?)\b",
        claim)
    for instrument in instruments:
        name = re.sub(r"^(?:The|A|An)\s+", "", instrument).strip().casefold()
        if name not in quoted.casefold():
            return False
    universal = re.search(r"\b(?:all|every|any)\s+(?:\w+\s+){0,3}(?:biometric|personal)\s+data\b",
                          claim, re.I)
    if universal and (not re.search(r"\b(?:all|every|any)\b", quoted, re.I)
                      or re.search(r"\bnot\s+(?:all|every|any)\b", quoted, re.I)):
        return False
    obligation = re.compile(r"\b(?:requir\w*|mandat\w*|must|shall|permitt\w*|allow\w*)\b", re.I)
    negated_obligation = re.compile(
        r"\b(?:not|never)\s+(?:\w+\s+){0,2}(?:requir\w*|mandat\w*|permitt\w*|allow\w*)\b"
        r"|\bno\s+(?:\w+\s+){0,2}(?:requirement|obligation)\b", re.I)
    if obligation.search(claim) and not obligation.search(quoted):
        return False
    if obligation.search(claim):
        if bool(negated_obligation.search(claim)) != bool(negated_obligation.search(quoted)):
            return False
    return True


def review_claims(claims: list[Claim], proposed: list[ClaimAssessment],
                  evidence: list[Evidence], verified_dates: dict[str, date] | None = None,
                  today: date | None = None) -> list[ReviewedClaim]:
    """Fail closed on omissions, invented quotes, mismatched IDs, and stale current claims."""
    source_map = {item.id: item for item in evidence}
    by_id = {item.claim_id: item for item in proposed}
    counts = Counter(item.claim_id for item in proposed)
    duplicate_ids = {claim_id for claim_id, count in counts.items() if count > 1}
    if set(by_id) - {item.id for item in claims}:
        logger.info("claim_verification_rejected reason=unknown_claim_assessment")
    result = []
    for claim in claims:
        item = by_id.get(claim.id)
        if item is None or claim.id in duplicate_ids:
            reason = "duplicate_claim_assessment" if claim.id in duplicate_ids else "missing_claim_assessment"
            logger.info("claim_verification claim_id=%s source_ids=%s proposed_status=%s "
                        "quote_match=unknown source_match=unknown numeric_match=unknown "
                        "contradiction_match=unknown reason=%s",
                        claim.id, claim.cited_ids, item.status if item else "missing", reason)
            result.append(ReviewedClaim(claim, "unsupported", [], [], reason))
            continue
        support = item.supporting
        oppose = item.opposing
        quote_match = all(_verbatim(quote, source_map) for quote in support + oppose)
        source_match = (all(quote.source_id in claim.cited_ids for quote in support)
                        and all((quote.source_id.startswith("c") if claim.section == "pdf" else
                                 quote.source_id.startswith("ext")) for quote in support))
        numeric_match = _numbers_supported(claim.text, support) if item.status in ("supported", "partially_supported") else True
        legal_scope_match = _legal_scope_supported(claim.text, support) if item.status == "supported" else True
        contradiction_match = not oppose or _explicit_opposition(support, oppose)
        valid = (item.status in ("supported", "partially_supported", "unsupported", "contradicted")
                 and quote_match and source_match and numeric_match and legal_scope_match and contradiction_match
                 and all(quote.source_id not in {item.source_id for item in support} for quote in oppose)
                 and (item.status == "unsupported" or bool(support))
                 and (item.status != "contradicted" or bool(oppose))
                 and (item.status != "unsupported" or not (support or oppose)))
        if (item.status == "supported" and re.search(r"\bfalse positives?\b", claim.text, re.I)
                and not re.search(r"\b(?:not|cannot|can't|shouldn't)\b", claim.text, re.I)
                and not re.search(r"\bfalse positives?\b", " ".join(q.quote for q in support), re.I)):
            valid = False
        reason_code = ("accepted" if valid else "quote_mismatch" if not quote_match else
                       "source_mismatch" if not source_match else "numeric_mismatch" if not numeric_match else
                       "legal_scope_mismatch" if not legal_scope_match else
                       "contradiction_mismatch" if not contradiction_match else
                       "missing_supporting_quote" if item.status != "unsupported" and not support else
                       "invalid_status_or_support")
        logger.info("claim_verification claim_id=%s source_ids=%s quote_source_ids=%s proposed_status=%s quote_match=%s "
                    "source_match=%s numeric_match=%s legal_scope_match=%s contradiction_match=%s reason=%s",
                    claim.id, claim.cited_ids, [quote.source_id for quote in support + oppose], item.status,
                    quote_match, source_match, numeric_match, legal_scope_match, contradiction_match, reason_code)
        if not valid:
            reason = ("No citation was attached to the claim." if not claim.cited_ids else
                      "The proposed supporting quote was absent from the cited passage or did not cover the claim.")
            result.append(ReviewedClaim(claim, "unsupported", [], [], reason))
            continue
        status: ClaimStatus = "contradicted" if oppose and support else item.status
        if (status == "supported" and claim.section == "external"
                and re.search(r"\b(current|currently|today|latest|now)\b", claim.text, re.I)):
            dates = verified_dates or {}
            cutoff = (today or date.today()).year - 2
            if any((stamp := dates.get(quote.source_id)) is None or stamp.year < cutoff
                   for quote in support):
                status = "partially_supported"
        result.append(ReviewedClaim(claim, status, support, oppose, item.explanation))
    return result


def present_claims(reviews: list[ReviewedClaim], section: str) -> str:
    parts = []
    withheld = []
    for item in reviews:
        if item.claim.section != section:
            continue
        refs = " ".join(f"[{source_id}]" for source_id in dict.fromkeys(
            quote.source_id for quote in item.supporting))
        display_text = re.sub(r"\s+([.,;:!?])", r"\1", _NUMERIC_CITATION.sub("", item.claim.text)).strip()
        if item.status == "supported":
            parts.append(f"{display_text} {refs}".strip())
        elif item.status == "contradicted":
            def finding(quote: ClaimQuote) -> str:
                origin = "PDF" if quote.source_id.startswith("c") else "External source"
                return f'{origin} [{quote.source_id}] reports: "{quote.quote}"'
            supporting = " ".join(finding(quote) for quote in item.supporting)
            opposing = " ".join(finding(quote) for quote in item.opposing)
            parts.append(f"The cited findings differ. {supporting} {opposing} The evidence does not resolve the disagreement.")
        elif item.status == "partially_supported":
            excerpts = " ".join(f'"{quote.quote}"' for quote in item.supporting)
            parts.append(f"The evidence only partly addresses this claim. Verified passage: {excerpts} {refs} "
                         "The remaining detail could not be verified.".strip())
        else:
            withheld.append(item.claim.text)
    if withheld and not parts:
        parts.append("The cited evidence does not establish a reliable answer to this question.")
    return " ".join(parts)
