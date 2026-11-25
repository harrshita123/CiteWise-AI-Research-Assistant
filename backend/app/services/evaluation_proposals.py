"""Create untrusted review candidates from owner-scoped indexed passages."""
import re
from uuid import uuid4

from pydantic import ValidationError

from ..providers.answers import Evidence
from .evaluation_benchmark import BenchmarkCase, EvidenceRef, ResearchLabel, TAXONOMY


def generate_proposals(paper_id, paper_hash, chunks, provider):
    sources = {f"c{chunk.chunk_index}-{chunk.content_hash[:12]}": chunk for chunk in chunks}
    evidence = [Evidence(id=source_id, text=(f"Section: {chunk.section}\n" if chunk.section else "") + chunk.text)
                for source_id, chunk in sources.items()]
    generated = provider.propose_evaluation_cases(evidence)
    cases = []
    seen = set()
    for item in generated[:8]:
        if not isinstance(item, dict):
            continue
        question = item.get("question", "")
        if not isinstance(question, str) or not question.strip():
            continue
        key = re.sub(r"\W+", " ", question.casefold()).strip()
        if key in seen:
            continue
        behavior = item.get("behavior")
        source = sources.get(item.get("source_id"))
        quote = item.get("quote")
        refs = ()
        if behavior == "answer":
            if source is None or not isinstance(quote, str) or not quote.strip() or quote not in source.text:
                continue
            refs = (EvidenceRef(page=source.page_number, chunk_index=source.chunk_index,
                                content_hash=source.content_hash, quote=quote),)
        elif behavior != "abstain":
            continue
        axis, label = item.get("axis"), item.get("label")
        labels = ()
        if isinstance(axis, str) and isinstance(label, str) and axis in TAXONOMY and label in TAXONOMY[axis] and refs:
            labels = (ResearchLabel(axis=axis, label=label, confidence=0.5,
                                    evidence=refs, needs_review=True),)
        try:
            case = BenchmarkCase(id=uuid4().hex, paper_id=paper_id, paper_hash=paper_hash,
                question=question.strip(), expected_behavior=behavior,
                expected_answer=item.get("answer", ""), evidence=refs, labels=labels)
        except (ValidationError, TypeError):
            continue
        seen.add(key)
        cases.append({"status": "pending", "case": case.model_dump(mode="json")})
    return cases
