"""Conservative, text-based classification for research uploads."""
import re
from enum import StrEnum

from .publication_identity import extracted_paper_metadata


class DocumentType(StrEnum):
    RESEARCH_PAPER = "research_paper"
    NOT_RESEARCH_PAPER = "not_research_paper"
    INCONCLUSIVE = "inconclusive"


SECTION = r"(?im)^[ \t]*(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?(?:%s)\b(?:[ \t]*[:.\u2013\u2014-]|[ \t]*$)"

# A document heading and corroborating administrative language are required.
ADMINISTRATIVE_TYPES = (
    ("a certificate", r"(?:[\w-]+ ){0,4}certificate|this is to certify",
     r"\b(?:has (?:successfully )?completed|is awarded to|presented to|awarded to|certifies that|this is to certify|in recognition of|issuing authority|certificate number)\b"),
    ("an admission letter", r"(?:letter of (?:admission|acceptance)|(?:provisional |conditional )?admission (?:letter|offer)|offer of admission)",
     r"\b(?:admitted|offer|enrollment|enrolment|academic year|program|programme)\b"),
    ("a résumé", r"(?:resume|résumé|curriculum vitae|cv|professional experience|work experience)",
     r"(?im)^\s*(?:education|skills|employment history|work experience|professional experience|contact|career objective)\s*[:\n]"),
    ("an invoice or receipt", r"(?:tax )?invoice|(?:payment |sales )?receipt",
     r"\b(?:amount due|bill to|invoice (?:number|date)|subtotal|total paid|payment method)\b"),
    ("a transcript", r"(?:official |unofficial |academic )?transcript|academic record|mark\s*sheet|statement of marks",
     r"\b(?:grade point average|gpa|credits earned|course code|student id|marks obtained|maximum marks)\b"),
    ("an identity document", r"(?:national |student |government )?(?:identity card|identification card|id card)|passport|driver'?s? licen[cs]e",
     r"\b(?:date of birth|date of expiry|issuing authority|document number|passport number|nationality)\b"),
    ("an employment offer", r"(?:employment |job )?offer letter|(?:conditional )?offer of employment",
     r"\b(?:start date|annual salary|compensation|employment|job title|position)\b"),
    ("an academic assignment", r"(?:(?:course|graded|individual|group|academic|research|written|lab) )?assignment|homework|problem set|coursework",
     r"\b(?:due date|submit(?:ted)? (?:by|to|on|via|your)|student id|course code|grading rubric|instructor|points each)\b"),
    ("a slide presentation", r"presentation slides|slide deck|(?:lecture|seminar) slides",
     r"(?im)\b(?:slide \d+|speaker notes|agenda|learning objectives|thank you)\b"),
    ("a personal document", r"personal statement|cover letter|personal letter",
     r"\b(?:dear|sincerely|motivation for applying|i am applying|my application)\b"),
    ("an application form", r"(?:\w+ ){0,5}application form",
     r"\b(?:applicant(?: name| signature)?|date of birth|required fields|office use only)\b"),
    ("an administrative document", r"(?:office |administrative )?(?:notice|memorandum|announcement)",
     r"\b(?:office hours|will be closed|please report|effective from|all staff|subject:)\b"),
)


def classify_document(identity_input: dict) -> tuple[DocumentType, str | None]:
    first = identity_input.get("first", "")
    text = identity_input.get("text", "") or first
    if not text.strip():
        return DocumentType.INCONCLUSIVE, None

    headings = {
        "abstract": "abstract|summary|résumé|resumen|resumo|zusammenfassung",
        "intro": "introduction|introducción|introdução|einleitung|background|related work",
        "method": "methods?|methodology|materials and methods|experimental design|data and methods|méthodes|métodos|methoden",
        "findings": "results(?: and discussion)?|discussion|conclusions?|findings|résultats|resultados|ergebnisse",
        "theory": r"theorems?(?: \d+(?:\.\d+)*)?|proofs?|technical contributions",
    }
    # “Résumé” can be a CV heading or an abstract in French. Require the CV cues below.
    opening_sections = "|".join(headings.values()).replace("|résumé", "")
    opening = re.split(SECTION % opening_sections, first, maxsplit=1)[0]
    lines = [line.strip() for line in opening.splitlines() if line.strip()][:14]
    for category, heading, supporting in ADMINISTRATIVE_TYPES:
        if (any(re.match(rf"^(?:{heading})(?:\b|$)", line, re.I) for line in lines)
                and re.search(supporting, opening, re.I)):
            return DocumentType.NOT_RESEARCH_PAPER, category
    word_count = len(re.findall(r"[^\W\d_]+", text))
    if word_count < 12:
        return DocumentType.INCONCLUSIVE, None
    sections = {name for name, pattern in headings.items() if re.search(SECTION % pattern, text)}
    structure = text + "\n" + identity_input.get("tail", "")
    # Numbered citation entries can form a bibliography without a heading.
    reference_entries = re.findall(
        r"(?m)^\s*(\d{1,3})\.[ \t]*\n?[ \t]*[^\W\d_][^\n]{15,}\n?[^\n]{0,140}?\((?:19|20)\d{2}\)",
        structure)
    numbered_bibliography = len(set(reference_entries)) >= 3
    bibliography = bool(re.search(SECTION % "references|bibliography|works cited", structure))
    metadata = extracted_paper_metadata(identity_input)
    identifier = any(metadata.get(key) for key in ("doi", "arxiv_id", "pmid", "pmcid"))
    # arXiv watermarks can be extracted after the main text because they run vertically.
    repository_stamp = re.search(r"(?im)^\s*arxiv:\d{4}\.\d{4,5}(?:v\d+)?[ \t]+\[[\w.-]+\]", first)
    venue = bool(metadata.get("venue") or re.search(r"(?im)^\s*proceedings of\b", first))
    scholarly_structure = len(sections) >= 3 or bool(bibliography and len(sections) >= 2)
    if scholarly_structure:
        return DocumentType.RESEARCH_PAPER, None
    # A header, identifier or copied title cannot make an unrelated page research.
    # For unstructured journals, corroborate the title and byline with substantial
    # content and a distinct summary or bibliography.
    substantive = word_count >= 220
    if (substantive and metadata.get("title") and metadata.get("authors")
            and (numbered_bibliography or bibliography
                 or metadata.get("abstract") and (identifier or venue or sections & {"intro", "method", "findings", "theory"}))):
        return DocumentType.RESEARCH_PAPER, None
    if sections and (repository_stamp or venue or bibliography or numbered_bibliography
                     or identifier and (substantive or len(sections) >= 2)):
        return DocumentType.RESEARCH_PAPER, None
    if metadata.get("title") and ("theory" in sections or metadata.get("authors") and len(sections) >= 2):
        return DocumentType.RESEARCH_PAPER, None
    return DocumentType.INCONCLUSIVE, None
