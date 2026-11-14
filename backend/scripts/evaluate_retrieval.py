"""Measure page-level Hit@k on a labelled set of owned papers.

Run from backend/: .venv313/bin/python -m scripts.evaluate_retrieval cases.jsonl
"""

import argparse
import json
from pathlib import Path

from app.providers.embeddings import LocalEmbeddings
from app.repositories.vectors import VectorRepository


def evaluate(cases: list[dict], embeddings, vectors: VectorRepository, limit: int = 5) -> dict:
    if not cases:
        raise ValueError("At least one labelled question is required.")
    results = []
    for number, case in enumerate(cases, start=1):
        owner_id = case["owner_id"]
        paper_id = case["paper_id"]
        question = case["question"].strip()
        expected_pages = set(case["expected_pages"])
        if not owner_id or not paper_id or not question or not expected_pages:
            raise ValueError(f"Case {number} needs owner_id, paper_id, question, and expected_pages.")
        embedding = embeddings.embed([question])[0]
        passages = vectors.search(owner_id, paper_id, embedding, limit, question=question)
        returned_pages = [passage.page_number for passage in passages]
        results.append({
            "case": number,
            "hit": bool(expected_pages.intersection(returned_pages)),
            "expected_pages": sorted(expected_pages),
            "returned_pages": returned_pages,
        })
    hits = sum(result["hit"] for result in results)
    return {"metric": f"page_hit_at_{limit}", "cases": len(results), "hits": hits,
            "score": hits / len(results), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate page-level retrieval on labelled PDF questions.")
    parser.add_argument("cases", type=Path, help="JSONL file with owner_id, paper_id, question, expected_pages")
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--storage-directory", type=Path, default=Path(__file__).resolve().parents[1] / "storage")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    storage = args.storage_directory
    report = evaluate(cases, LocalEmbeddings(storage / "models"),
                      VectorRepository(storage / "chroma"), args.limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
