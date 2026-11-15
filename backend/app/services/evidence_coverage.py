"""Lightweight retrieval coverage hints, never proof that a claim is supported."""

import re
from dataclasses import dataclass

from .research_subjects import requested_subjects, scoped_segments, subject_spans


DIMENSIONS = {
    "design": (r"\b(?:methodolog\w*|designs?|randomiz\w*|experimental)\b",
               r"\b(?:methodolog\w*|design|randomiz\w*|experiment\w*|trial|cohort|survey|"
               r"qualitative|matched|interview\w*|systematic review|meta.analysis)\b"),
    "participants": (r"\b(?:participants?|students?|sample sizes?|subjects?|patients?)\b",
                     r"\b(?:participants?|students?|sample|subjects?|patients?|respondents?)\b"),
    "setting": (r"\b(?:settings?|locations?|regions?)\b|\b(?:number of|how many|count of)\s+(?:\w+\s+){0,2}(?:universit\w*|institutions?|colleges?)\b",
                r"\b(?:universit\w*|institutions?|colleges?|regions?|settings?|locations?|campuses?)\b"),
    "duration": (r"\b(?:duration|periods?|semesters?|years?|months?|weeks?|long)\b",
                 r"\b(?:duration|periods?|semesters?|years?|months?|weeks?|follow.up)\b"),
    "findings": (r"\b(?:findings?|results?|outcomes?|effects?|impact|integrity|privacy|anxiety)\b",
                 r"\b(?:findings?|results?|outcomes?|effects?|impact|misconduct|privacy|anxiety|integrity|"
                 r"increas\w*|decreas\w*|reduc\w*|higher|lower|improv\w*)\b"),
    "limitations": (r"\b(?:limitations?|generaliz\w*|bias|caveats?|uncertaint\w*)\b",
                    r"\b(?:limitations?|generaliz\w*|bias|caveats?|uncertaint\w*|non.significant|non.randomiz\w*|cannot|may not|volunteer)\b"),
}


def requested_coverage(question: str) -> tuple[list[str], list[str]]:
    """Extract explicitly named subjects and answer dimensions from a question."""
    subjects = requested_subjects(question)
    dimensions = [name for name, (prompt, _) in DIMENSIONS.items() if re.search(prompt, question, re.I)]
    if ("design" in dimensions and "setting" not in dimensions
            and re.search(r"\b(?:universit\w*|institutions?|colleges?)\b", question, re.I)
            and not re.search(r"\b(?:should|adopt|generaliz\w*|all universities)\b", question, re.I)):
        dimensions.append("setting")
    if not dimensions and re.search(r"\b(?:compare|comparison|differences?|similarities?)\b", question, re.I):
        dimensions = ["findings"]
    return subjects, dimensions


def missing_coverage(question: str, passages: list[str]) -> list[tuple[str, str]]:
    """Find requested information absent from candidate text, without approving claims."""
    report = coverage_report(question, [CoveragePassage(str(index), text)
                                        for index, text in enumerate(passages)])
    return [(item.subject, item.dimension) for item in report
            if item.status not in ("information_present", "verified_claim_support")]


@dataclass(frozen=True)
class CoveragePassage:
    id: str
    text: str
    page_number: int | None = None
    chunk_index: int | None = None


@dataclass(frozen=True)
class CoverageFinding:
    subject: str
    dimension: str
    status: str  # missing, topic_mentioned, information_present, verified_claim_support
    candidate_ids: tuple[str, ...]
    information_ids: tuple[str, ...]
    inferred_subject_ids: tuple[str, ...] = ()


_COUNT = r"(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)"
_PEOPLE = r"(?:participants?|students?|subjects?|patients?|respondents?|people|individuals?)"
_PLACES = r"(?:universit\w*|institutions?|colleges?|campuses?|regions?|countries|sites?)"
_PERIOD = r"(?:semesters?|years?|months?|weeks?|days?|exam periods?|terms?)"


def _information_present(question: str, dimension: str, text: str) -> bool:
    """Require a factual cue for the requested dimension; semantic verification follows later."""
    if dimension == "participants":
        if re.search(r"\b(?:how many|number of|sample sizes?|count of|total)\b", question, re.I):
            return bool(re.search(rf"\b{_COUNT}\s+(?:\w+\s+){{0,2}}{_PEOPLE}\b"
                                  rf"|\b{_PEOPLE}\s*(?:[:=]|of|numbered|totaled|included)\s*{_COUNT}\b", text, re.I))
        return bool(re.search(rf"\b{_PEOPLE}\b", text, re.I))
    if dimension == "setting":
        return bool(re.search(rf"\b{_COUNT}\s+(?:\w+\s+){{0,2}}{_PLACES}\b", text, re.I))
    if dimension == "duration":
        return bool(re.search(rf"\b(?:{_COUNT}|a|an|single|one)\s+{_PERIOD}\b"
                              r"|\b(?:follow.up|observation period|duration)\s*(?:[:=]|of)\s*\S+", text, re.I))
    if dimension == "design":
        return bool(re.search(r"\b(?:randomiz\w*|quasi.experiment\w*|cross.sectional|longitudinal|"
                              r"matched(?:.course| comparison| design)|case.control|cohort|survey|interview\w*|"
                              r"qualitative|quantitative|controlled trial|observational|"
                              r"systematic review|meta.analysis)\b", text, re.I))
    if dimension == "findings":
        return bool(re.search(r"\b(?:increas\w*|decreas\w*|reduc\w*|higher|lower|improv\w*|"
                              r"no (?:significant )?(?:difference|effect|change)|not significant|"
                              r"statistically significant|associated|correlat\w*)\b"
                              r"|\b\d+(?:\.\d+)?\s*%", text, re.I))
    if dimension == "limitations":
        return bool(re.search(r"\b(?:limitation\w*|limits? generaliz\w*|not generaliz\w*|"
                              r"may not generaliz\w*|selection bias|may bias|cannot|may not|"
                              r"non.randomiz\w*|small sample|short follow.up|volunteer\w*|"
                              r"non.significant)\b", text, re.I))
    return False


def coverage_report(question: str, passages: list[CoveragePassage],
                    verified_ids: set[str] | None = None) -> list[CoverageFinding]:
    """Report verified support only for IDs supplied by the claim verifier."""
    subjects, dimensions = requested_coverage(question)
    targets = subjects or [""]
    inferred: dict[str, str] = {}
    previous = None
    previous_subject = None
    for passage in sorted((item for item in passages if item.chunk_index is not None),
                          key=lambda item: item.chunk_index):
        named = {name for name, _, _ in subject_spans(passage.text)}
        if len(named) == 1:
            previous_subject = next(iter(named))
        elif named:
            previous_subject = None
        elif (previous and previous_subject and previous.page_number == passage.page_number
              and previous.chunk_index + 1 == passage.chunk_index):
            # This is a navigation hint, never evidence that the unnamed chunk
            # actually belongs to that study.
            inferred[passage.id] = previous_subject
        else:
            previous_subject = None
        previous = passage
    report = []
    for dimension in dimensions:
        for subject in targets:
            pattern = DIMENSIONS[dimension][1]
            candidates = []
            information = []
            inferred_candidates = []
            for passage in passages:
                # Split at explicit study labels so one study's result in a shared
                # chunk does not satisfy another study's requested dimension.
                segments = scoped_segments(passage.text)
                scoped = [segment for name, segment in segments if not subject or
                          name.casefold() == subject.casefold()]
                if not subject and not scoped:
                    scoped = [passage.text]
                if any(re.search(pattern, segment, re.I) for segment in scoped):
                    candidates.append(passage.id)
                elif subject and inferred.get(passage.id) == subject and re.search(pattern, passage.text, re.I):
                    candidates.append(passage.id)
                    inferred_candidates.append(passage.id)
                if any(re.search(pattern, segment, re.I)
                       and _information_present(question, dimension, segment) for segment in scoped):
                    information.append(passage.id)
            status = ("verified_claim_support" if verified_ids and set(information) & verified_ids else
                      "information_present" if information else "topic_mentioned" if candidates else "missing")
            report.append(CoverageFinding(subject, dimension, status,
                                          tuple(dict.fromkeys(candidates)), tuple(dict.fromkeys(information)),
                                          tuple(dict.fromkeys(inferred_candidates))))
    return report
