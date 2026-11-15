"""Recognize explicit research subjects without assuming two lettered studies."""

import re


SUBJECT_RE = re.compile(
    r"\b(?:(?i:study|trial|cohort|experiment)\s+(?:[A-Z][A-Za-z0-9_-]*|[a-z]|\d+)"
    r"|(?!(?:Compare|What|Which|How|Did|Does|The|This|Both|Each)\b)"
    r"[A-Z][A-Za-z0-9_-]{2,}\s+(?i:study|trial|cohort))\b"
)
_GENERIC_NAMES = {"the", "this", "that", "our", "each", "every", "another", "previous", "current"}


def _canonical(name: str) -> str:
    words = name.split()
    if words[0].casefold() in {"study", "trial", "cohort", "experiment"}:
        return f"{words[0].title()} {words[1].upper() if len(words[1]) == 1 else words[1]}"
    return name


def subject_spans(text: str) -> list[tuple[str, int, int]]:
    spans = []
    for match in SUBJECT_RE.finditer(text):
        words = match.group().split()
        if words[0].casefold() in _GENERIC_NAMES or words[-1].casefold() in _GENERIC_NAMES:
            continue
        spans.append((_canonical(match.group()), match.start(), match.end()))
    return spans


def requested_subjects(question: str) -> list[str]:
    return list(dict.fromkeys(name for name, _, _ in subject_spans(question)))


def scoped_segments(text: str) -> list[tuple[str, str]]:
    spans = subject_spans(text)
    if not spans:
        return [("", text)]
    return [(name, text[start:spans[index + 1][1] if index + 1 < len(spans) else len(text)])
            for index, (name, start, _) in enumerate(spans)]
