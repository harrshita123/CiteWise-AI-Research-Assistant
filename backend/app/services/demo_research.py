"""Curated evidence from the bundled article and two published commentaries."""

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
import unicodedata

from ..paper_models import ChunkResponse
from .pdf_processing import TextChunk
from .question_recommendations import RecommendedQuestion


SAMPLE_PDF = Path(__file__).resolve().parents[1] / "fixtures" / "neves-et-al-2022.pdf"
SAMPLE_SHA256 = "4eddb46722638638935cb8a7954867f348d2aa39501454cb3d21ef25d4fced20"
SAMPLE_TITLE = "Are most published research findings false in a continuous universe?"
SAMPLE_FILENAME = "Neves et al. (2022) — Continuous universe.pdf"
SAMPLE_IDENTITY = {
    "status": "verified_published", "demo_sample": True,
    "title": SAMPLE_TITLE, "authors": ["Kleber Neves", "Pedro B. Tan", "Olavo B. Amaral"],
    "year": 2022, "venue": "PLOS ONE", "publisher": "Public Library of Science",
    "doi": "10.1371/journal.pone.0277935", "verification_scope": "publication_record",
    "source": "PLOS ONE publisher record",
    "source_url": "https://doi.org/10.1371/journal.pone.0277935",
    "reason": "The bundled PDF and its publication metadata were checked against the publisher record. CC BY 4.0.",
}


@dataclass(frozen=True)
class DemoCase:
    question: str
    page: int
    excerpt: str
    kind: str
    answer: str


CASES = (
    DemoCase("Which factors led to more false positives in the simulations?", 1,
             "the combination of selective publication, bias, low statistical power and unlikely hypotheses",
             "findings", "Selective publication, bias, low statistical power and a low prevalence of true hypotheses together led to more false positives in the simulations."),
    DemoCase("How did the authors simulate each experiment?", 2,
             "In each experiment, scientists perform t-tests between two experimental groups.",
             "methodology", "Each simulated experiment compared two groups using a t-test."),
    DemoCase("How were continuous and dichotomous effects modeled differently?", 3,
             "A dichotomous distribution containing only null and large effects makes it trivial to define true and false hypotheses; in reality, however, a real scientific field is likely to have an unknown distribution of continuous effect sizes.",
             "comparison", "The dichotomous model used only null and large effects. The continuous models allowed effect sizes to vary across a range."),
    DemoCase("What assumptions limit how broadly the results apply?", 9,
             "its assumptions of absolute publication bias, lack of replication and publication of significant findings irrespective of context",
             "limitations", "The authors flag the assumptions that publication depends entirely on statistical significance, that studies are not replicated, and that significant results are published regardless of context."),
    DemoCase("Why should the effect size threshold matter when interpreting true effects?", 3,
             "a minimum effect size of interest, above which hypotheses are considered true",
             "critical_analysis", "The threshold defines what counts as a true effect in the continuous models. The classification depends on what the model considers a meaningful effect."),
)

REPLY_RANGE = "there is a wide credibility gradient (0.1% to 85%) for different research designs and settings."
REPLY_THRESHOLD = "I did not assume that all significant p-values are around 0.05."
EXTERNAL_SOURCES = (
    {"id": "e1", "title": "Why Most Published Research Findings Are False: Problems in the Analysis",
     "url": "https://doi.org/10.1371/journal.pmed.0040168", "publisher": "PLOS Medicine",
     "author": "Steven Goodman and Sander Greenland", "verified_publication_date": "2007-04-24",
     "excerpt": "the model employed cannot be considered a proof that most published claims are untrue, but is rather a claim that no study or combination of studies can ever provide convincing evidence"},
    {"id": "e2", "title": "Why Most Published Research Findings Are False: Author's Reply to Goodman and Greenland",
     "url": "https://doi.org/10.1371/journal.pmed.0040215", "publisher": "PLOS Medicine",
     "author": "John P. A. Ioannidis", "verified_publication_date": "2007-06-26",
     "excerpt": f"{REPLY_RANGE} […] {REPLY_THRESHOLD}"},
)

EXTERNAL_QUESTIONS = (
    "Where did Goodman and Greenland disagree with Ioannidis in the 2007 exchange?",
    "What did Ioannidis say in his reply about significant p-values near 0.05?",
)

def load_sample_index(model_name: str, dimensions: int) -> tuple[int, list[TextChunk], list[list[float]]]:
    """Read the checked PDF's MiniLM index without initializing inference during guest sign-in."""
    with gzip.open(SAMPLE_PDF.with_suffix(".index.json.gz"), "rt", encoding="utf-8") as fixture:
        index = json.load(fixture)
    if index["pdf_sha256"] != SAMPLE_SHA256 or index["model"] != model_name:
        raise ValueError("Bundled sample index does not match the PDF or embedding model.")
    chunks = [TextChunk(**chunk) for chunk in index["chunks"]]
    vectors = index["vectors"]
    if not chunks or len(chunks) != len(vectors) or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Bundled sample index has invalid dimensions.")
    return index["page_count"], chunks, vectors


def question_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().strip().rstrip("?!.").split())


def is_sample(paper: dict) -> bool:
    # The upload path computes this from the bytes. A filename or printed DOI
    # cannot establish that this is the exact edition used for the citations.
    return paper.get("file_hash") == SAMPLE_SHA256


def sample_quote(case: DemoCase, chunk: ChunkResponse) -> str:
    match = re.search(r"\s+".join(map(re.escape, case.excerpt.split())), chunk.text)
    return match[0] if match else ""


def sample_source(case: DemoCase, chunks: list[ChunkResponse]) -> ChunkResponse | None:
    return next((chunk for chunk in chunks if chunk.page_number == case.page and sample_quote(case, chunk)), None)


def sample_recommendations(chunks: list[ChunkResponse], previous: list[str], limit: int) -> list[RecommendedQuestion]:
    seen = {question_key(question) for question in previous}
    return [RecommendedQuestion(f"demo-{index}", case.question, case.kind, source)
            for index, case in enumerate(CASES) if question_key(case.question) not in seen
            and (source := sample_source(case, chunks)) is not None][:limit]
