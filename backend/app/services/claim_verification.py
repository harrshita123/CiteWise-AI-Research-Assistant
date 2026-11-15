"""Validate model claim assessments against the exact retrieved evidence."""

import re
import logging
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Literal

from ..providers.answers import ClaimAssessment, ClaimQuote, Evidence
from .research_subjects import subject_spans


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
_CLAUSE_VERB = r"(?:is|are|was|were|has|have|had|do|does|did|can|could|found|reported|observed|showed|concluded|suggested|indicated|demonstrated|proved|caused|improved|reduced|increased|decreased|remained)"
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
    reason_code: str = "accepted"


def _independent_clauses(text: str) -> list[str]:
    """Separate explicit independent assertions, preserving dependent qualifications."""
    boundary = re.compile(r",?\s+(?:but|however|whereas|and)\s+", re.I)
    pieces, start = [], 0
    for match in boundary.finditer(text):
        left, right = text[start:match.start()].strip(), text[match.end():].strip()
        if (not left or re.search(r"\bnot only\b", left, re.I)
                or not re.match(rf"(?:[A-Za-z][\w'-]*\s+){{1,6}}{_CLAUSE_VERB}\b", right, re.I)):
            continue
        pieces.append(left)
        start = match.end()
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


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
            # An explicit citation beside a clause belongs to that clause. A
            # sentence-final citation remains shared when no local one exists.
            raw_clauses = _independent_clauses(part)
            trailing = _CITATION.findall(raw_clauses[-1]) + _PAREN_SOURCE_CITATION.findall(raw_clauses[-1])
            for raw in raw_clauses:
                independent = _PAREN_SOURCE_CITATION.sub("", _CITATION.sub("", raw)).strip()
                local = list(dict.fromkeys(_CITATION.findall(raw) + _PAREN_SOURCE_CITATION.findall(raw)))
                clause_citations = local or (trailing if len(raw_clauses) > 1 else cited)
                # Keep the shared subject for parallel facts; each part retains
                # the sentence citation but receives its own verification.
                named = subject_spans(independent)
                subject = (named[0][0] if named and named[0][1] == 0
                           and re.match(rf"\s+{_FACT_VERB}\b", independent[named[0][2]:], re.I) else None)
                if subject is None:
                    generic = re.match(rf"^(The\s+(?:study|trial|paper|cohort|experiment))\s+{_FACT_VERB}\b",
                                       independent, re.I)
                    subject = generic.group(1) if generic else None
                clauses = re.split(rf",\s+(?:and\s+)?(?={_FACT_VERB}\b)", independent, flags=re.I) if subject else [independent]
                for index, clause in enumerate(clauses):
                    statement = clause.strip()
                    if index:
                        statement = f"{subject} {statement}"
                    claims.append(Claim(f"claim-{len(claims) + 1}", statement, section, clause_citations.copy()))
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
            population = lambda text: re.search(r"\bamong\s+([\w-]+(?:\s+[\w-]+){0,3}\s+"
                                                r"(?:students?|patients?|adults?|participants?|workers?|"
                                                r"respondents?|children|cohorts?))\b",
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


def _inference_scope_supported(claim: str, support: list[ClaimQuote]) -> bool:
    """Require explicit evidence for strong inferential wording, beyond a measurement alone."""
    quoted = " ".join(item.quote for item in support)
    if not quoted:
        return False
    related = _relevant_passages(claim, support)
    scoped = " ".join(related)
    # Claims explicitly saying an effect is *not* established are qualifications.
    qualification = re.search(
        r"\b(?:does not|did not|cannot|can't|not|no evidence to)\s+"
        r"(?:\w+\s+){0,4}(?:establish|prove|demonstrate|show|confirm)\b", claim, re.I)
    qualified = bool(qualification and not re.search(r"\b(?:but|however|and)\b",
                                                     claim[qualification.end():], re.I))
    negative_significance = re.search(
        r"\b(?:not|non|no|without(?: an?)?)\s+(?:statistically\s+)?significant\b", claim, re.I)
    if negative_significance and not re.search(
            r"\b(?:not|non|no)\s+(?:statistically\s+)?significant\b", quoted, re.I):
        return False
    significant = re.search(r"\bstatistically significant\b|\bsignificant (?:effect|difference|association)\b",
                            claim, re.I)
    if significant and not (qualified or negative_significance):
        if not _statistical_support(scoped):
            return False
        if not re.search(r"\bstatistically significant\b|\bsignificant (?:effect|difference|association)\b",
                         scoped, re.I) and not _relevant_partial_support(claim, support):
            return False
    if not qualified and re.search(r"\b(?:proves?|proved|proof)\b", claim, re.I) and not re.search(
            r"\b(?:proves?|proved|proof)\b", quoted, re.I):
        return False
    causal = re.search(r"\b(?:caus(?:e[ds]?|ing)|led to|resulted in|attributable to)\b", claim, re.I)
    causal_reported = re.search(r"\b(?:caus(?:e[ds]?|ing)|led to|resulted in|attributable to)\b",
                                scoped, re.I)
    causal_design = (re.search(r"\brandomiz\w*\b", quoted, re.I)
                     and _statistical_support(scoped) and _relevant_partial_support(claim, support))
    if causal and not qualified and not (causal_reported or causal_design):
        return False
    negated_causal = r"\b(?:not|never)\s+(?:\w+\s+){0,2}caus(?:e[ds]?|ing)\b"
    if causal and not qualified and bool(re.search(negated_causal, claim, re.I)) != bool(
            re.search(negated_causal, scoped, re.I)):
        return False
    effectiveness = re.search(r"\b(?:effective(?:ness)?|efficacy)\b", claim, re.I)
    interpreted = (re.search(r"\b(?:conclud\w*|interpret\w*|describ\w*|judg\w*)\b", scoped, re.I)
                   and _relevant_partial_support(claim, support)
                   and re.search(r"\b(?:helpful|beneficial|improv\w*|effective|efficacy)\b", scoped, re.I)
                   and not re.search(r"\b(?:uncertain|unclear|inconclusive|mixed|may|might|possible|ineffective|unhelpful|no benefit|failed to)\b", scoped, re.I))
    if effectiveness and not qualified and not (re.search(r"\b(?:effective(?:ness)?|efficacy)\b", scoped, re.I)
                                            or interpreted):
        return False
    if effectiveness and not qualified and re.search(r"\b(?:uncertain|unclear|inconclusive|mixed|may|might|possible)\b", scoped, re.I):
        return False
    negated_effectiveness = r"\b(?:not|never)\s+(?:\w+\s+){0,2}effective\b|\bno evidence of efficacy\b"
    if effectiveness and not qualified and bool(re.search(negated_effectiveness, claim, re.I)) != bool(
            re.search(negated_effectiveness, scoped, re.I)):
        return False
    no_effect = re.search(r"\b(?:no|zero)\s+(?:measurable\s+)?(?:effect|impact)\b", claim, re.I)
    if no_effect and not re.search(r"\b(?:no|zero)\s+(?:measurable\s+)?(?:effect|impact)\b", quoted, re.I):
        return False
    return True


def _statistical_support(quoted: str) -> bool:
    """Recognize explicit significance or a reported test that excludes its null value."""
    if re.search(r"\b(?:not|non|no)\s+(?:statistically\s+)?significant\b", quoted, re.I):
        return False
    if re.search(r"\bstatistically significant\b|\bsignificant (?:effect|difference|association)\b",
                 quoted, re.I):
        return True
    threshold = re.search(r"\b(?:alpha|significance threshold)\s*(?:=|of)\s*(0?\.\d+)\b", quoted, re.I)
    alpha = float(threshold.group(1)) if threshold else 0.05
    p_value = re.search(r"\bp\s*(<=|<|=)\s*(0?\.\d+)\b", quoted, re.I)
    if p_value and (float(p_value.group(2)) < alpha
                    or p_value.group(1) == "<" and float(p_value.group(2)) == alpha):
        return True
    interval = re.search(r"\b95\s*%?\s*(?:CI|confidence interval)\s*[:=(]?\s*"
                         r"(-?\d+(?:\.\d+)?)\s*(?:to|[-–—,])\s*(-?\d+(?:\.\d+)?)", quoted, re.I)
    if interval:
        prefix = quoted[max(0, interval.start() - 80):interval.start()]
        null = (1 if re.search(r"\b(?:risk|odds|hazard|rate) ratio\b|\b(?:RR|OR|HR)\b", prefix, re.I)
                else 0 if re.search(r"\b(?:mean |risk |absolute )?difference\b", prefix, re.I) else None)
        if null is not None:
            low, high = float(interval.group(1)), float(interval.group(2))
            return low <= high and (high < null or low > null)
    return False


def _proposition_terms(text: str) -> set[str]:
    """Use subject and outcome words, excluding generic inference vocabulary."""
    plain = _NUMERIC_CITATION.sub("", text)
    for _, start, end in reversed(subject_spans(plain)):
        plain = plain[:start] + " " + plain[end:]
    ignored = {"about", "after", "among", "and", "authors", "because", "between", "change",
               "could", "difference", "effect", "effective", "effectiveness", "efficacy",
               "established", "evidence", "found", "from", "have", "impact", "intervention",
               "into", "more", "not", "observed", "paper", "participants", "program", "proved",
               "reported", "result", "research", "significant", "statistically", "study", "than",
               "that", "their", "there", "these", "this", "those", "trial", "treatment", "were",
               "what", "when", "which", "with", "lower", "higher", "reduced", "increased",
               "improved", "caused", "causal", "interpreted", "concluded", "described", "the"}
    return {re.sub(r"e$", "", re.sub(r"(?:ment|ing|ed|s)$", "", word))
            for word in re.findall(r"[a-z]{4,}", plain.casefold()) if word not in ignored}


def _relevant_passages(claim: str, support: list[ClaimQuote]) -> list[str]:
    """Keep evidence clauses about the claimed outcome, not adjacent statistics."""
    wanted = _proposition_terms(claim)
    if not wanted:
        subjects = subject_spans(_NUMERIC_CITATION.sub("", claim))
        if not subjects:
            return []
        return [item.quote for item in support
                if subjects[0][0].casefold() in item.quote.casefold()
                and re.search(r"\b(?:difference|significant|measured|observed)\b|\d", item.quote, re.I)]
    passages = []
    for item in support:
        clauses = re.split(r"[.;]\s+|,?\s+\b(?:whereas|while|but)\b\s+", item.quote, flags=re.I)
        matched = [part for part in clauses if wanted & _proposition_terms(part)]
        passages.extend(matched)
    if not passages and wanted == {"outcom"}:
        subjects = subject_spans(_NUMERIC_CITATION.sub("", claim))
        passages = [item.quote for item in support if subjects
                    and subjects[0][0].casefold() in item.quote.casefold()
                    and re.search(r"\b(?:measured|observed|improved|reduced|increased)\b|\d", item.quote, re.I)]
    return passages


def _relevant_partial_support(claim: str, support: list[ClaimQuote]) -> bool:
    """Do not display a valid quote about a different outcome as partial support."""
    passages = _relevant_passages(claim, support)
    if not passages:
        return False
    claim_direction = re.search(r"\b(?:reduc\w*|decreas\w*|lower|fall\w*|fell|dropp\w*|less|increas\w*|higher|ris\w*|rose|improv\w*|more)\b", claim, re.I)
    if claim_direction:
        decreasing = re.compile(r"\b(?:reduc\w*|decreas\w*|lower|fall\w*|fell|dropp\w*|less)\b", re.I)
        increasing = re.compile(r"\b(?:increas\w*|higher|ris\w*|rose|improv\w*|more)\b", re.I)
        wanted_down = bool(decreasing.search(claim_direction.group()))
        if all((increasing if wanted_down else decreasing).search(part)
               and not (decreasing if wanted_down else increasing).search(part) for part in passages):
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
            result.append(ReviewedClaim(claim, "unsupported", [], [], reason, reason))
            continue
        support = item.supporting
        oppose = item.opposing
        quote_match = all(_verbatim(quote, source_map) for quote in support + oppose)
        source_match = (all(quote.source_id in claim.cited_ids for quote in support)
                        and all((quote.source_id.startswith("c") if claim.section == "pdf" else
                                 quote.source_id.startswith("ext")) for quote in support))
        numeric_match = _numbers_supported(claim.text, support) if item.status in ("supported", "partially_supported") else True
        legal_scope_match = _legal_scope_supported(claim.text, support) if item.status == "supported" else True
        inference_scope_match = _inference_scope_supported(claim.text, support) if item.status == "supported" else True
        partial_relevance_match = (_relevant_partial_support(claim.text, support)
                                   if item.status == "partially_supported" else True)
        contradiction_match = not oppose or _explicit_opposition(support, oppose)
        valid = (item.status in ("supported", "partially_supported", "unsupported", "contradicted")
                 and quote_match and source_match and numeric_match and legal_scope_match
                 and inference_scope_match and partial_relevance_match and contradiction_match
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
                       "inference_scope_mismatch" if not inference_scope_match else
                       "irrelevant_supporting_quote" if not partial_relevance_match else
                       "contradiction_mismatch" if not contradiction_match else
                       "missing_supporting_quote" if item.status != "unsupported" and not support else
                       "invalid_status_or_support")
        logger.info("claim_verification claim_id=%s source_ids=%s quote_source_ids=%s proposed_status=%s quote_match=%s "
                    "source_match=%s numeric_match=%s legal_scope_match=%s inference_scope_match=%s "
                    "partial_relevance_match=%s "
                    "contradiction_match=%s reason=%s",
                    claim.id, claim.cited_ids, [quote.source_id for quote in support + oppose], item.status,
                    quote_match, source_match, numeric_match, legal_scope_match, inference_scope_match,
                    partial_relevance_match,
                    contradiction_match, reason_code)
        if not valid:
            reason = ("No citation was attached to the claim." if not claim.cited_ids else
                      "The proposed supporting quote was absent from the cited passage or did not cover the claim.")
            if (reason_code == "inference_scope_mismatch" and contradiction_match and not oppose
                    and _relevant_partial_support(claim.text, support)):
                result.append(ReviewedClaim(claim, "partially_supported", support, [],
                                            "The cited passage supports an observation but not the stronger conclusion.",
                                            reason_code))
            else:
                result.append(ReviewedClaim(claim, "unsupported", [], [], reason, reason_code))
            continue
        status: ClaimStatus = "contradicted" if oppose and support else item.status
        if (status == "supported" and claim.section == "external"
                and re.search(r"\b(current|currently|today|latest|now)\b", claim.text, re.I)):
            dates = verified_dates or {}
            cutoff = (today or date.today()).year - 2
            if any((stamp := dates.get(quote.source_id)) is None or stamp.year < cutoff
                   for quote in support):
                status = "partially_supported"
                reason_code = "outdated_or_unverified_external_source"
        if status == "partially_supported" and reason_code == "accepted":
            reason_code = "provider_partial"
        elif status == "unsupported" and reason_code == "accepted":
            reason_code = "provider_unsupported"
        elif status == "contradicted" and reason_code == "accepted":
            reason_code = "verified_contradiction"
        result.append(ReviewedClaim(claim, status, support, oppose, item.explanation, reason_code))
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
