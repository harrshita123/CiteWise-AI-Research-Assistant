"""Manually labeled synthetic retrieval benchmark; no answer or citation evaluation.

Run: cd backend && .venv/bin/python -m scripts.benchmark_retrieval_quality
The RRF branch is experimental and is not enabled in production.
"""

import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

from app.providers.embeddings import LocalEmbeddings
from app.repositories.vectors import VectorRepository
from app.services.pdf_processing import TextChunk


class TracedCollection:
    def __init__(self, collection):
        self.collection = collection
        self.candidate_ids = []
        self.scanned_count = 0

    def __getattr__(self, name):
        return getattr(self.collection, name)

    def query(self, **kwargs):
        result = self.collection.query(**kwargs)
        self.candidate_ids = [item["chunk_index"] for item in result["metadatas"][0]]
        return result

    def get(self, **kwargs):
        result = self.collection.get(**kwargs)
        self.scanned_count = len(result["ids"])
        return result


# Each tuple is (study identity, passage). Indexes are stable chunk labels.
PAPERS = {
    "exams": [
        ("A", "Study A used a cluster-randomized comparison in 12 universities, following 2,400 students for one semester."),
        ("B", "Study B used a matched-course quasi-experimental design in three universities with 780 students over two exam periods."),
        ("A", "Study A observed confirmed misconduct in 8 percent of monitored students and 14 percent of unmonitored students."),
        ("B", "Study B found no statistically significant difference in confirmed misconduct between monitored and unmonitored courses."),
        ("A", "Study A reported that 71 percent of surveyed students expressed privacy concerns about webcam monitoring."),
        ("B", "Study B recorded 120 automated alerts, of which 9 were confirmed and 111 remained unconfirmed."),
        ("A", "Study A authors warned that its three regions and one-semester follow-up limit generalizability."),
        ("B", "Study B authors warned that course matching cannot eliminate selection bias and only three universities participated."),
        ("A", "Study A did not measure facial-recognition accuracy and did not report a p-value for its misconduct comparison."),
        ("B", "Study B measured anxiety during two exam periods and did not examine long-term academic outcomes."),
    ],
    "sleep": [
        ("A", "Study A randomized 420 adults to eight weeks of sleep coaching or usual care."),
        ("B", "Study B observed 510 shift workers for six months without random assignment."),
        ("A", "Study A found a 42-minute increase in nightly sleep after coaching, compared with 18 minutes in usual care."),
        ("B", "Study B reported that longer sleep was associated with lower fatigue, but adjustment for shift schedules weakened the association."),
        ("A", "Study A excluded night-shift workers, limiting application to that population."),
        ("B", "Study B included night-shift workers but relied on self-reported sleep duration."),
        ("B", "Study B did not measure cardiovascular outcomes or blood pressure."),
    ],
    "climate": [
        ("A", "Study A compared tree-canopy coverage across 32 urban neighborhoods over five summers."),
        ("B", "Study B used satellite images from 18 cities and a cross-sectional temperature analysis."),
        ("A", "Study A found neighborhoods with more tree canopy were 2.1 degrees Celsius cooler on average."),
        ("B", "Study B found the cooling association varied by rainfall and building density, with no consistent effect in the driest cities."),
        ("A", "Study A temperature sensors sampled afternoon conditions but not nighttime heat."),
        ("B", "Study B satellite measurements captured surface temperature rather than indoor exposure."),
        ("A", "Study A did not test whether tree planting caused the observed temperature difference."),
    ],
}

# Ground-truth chunk indexes appear only here. Question wording is an input to
# evaluation, never a lookup key or rule in production retrieval.
CATEGORY_REQUIREMENTS = {
    "factual_lookup": "One passage directly states the requested fact.",
    "multi_study": "Evidence from each compared study is required.",
    "methodology": "Retrieve design, population, and period evidence when requested.",
    "numerical": "Retrieve the operands or the reported value; arithmetic is evaluated elsewhere.",
    "conflicting": "Retrieve passages describing each side of a disagreement.",
    "synthesis": "Retrieve all passages needed for a multi-part explanation.",
    "evidence_gap": "Retrieve a passage explicitly documenting an unmeasured or unreported item.",
    "absent": "No PDF passage supports the question; answerability is evaluated elsewhere.",
}
CASES_BY_CATEGORY = {
    "factual_lookup": [
        ("exams", "How many universities took part in Study B?", {1}),
        ("sleep", "How many adults were in the sleep-coaching trial?", {0}),
        ("climate", "How many neighborhoods were compared in Study A?", {0}),
    ],
    "multi_study": [
        ("exams", "Compare the designs and participants of Study A and Study B.", {0, 1}),
        ("climate", "Compare the research designs and settings of both urban heat studies.", {0, 1}),
    ],
    "methodology": [
        ("exams", "What was Study A's design, sample and observation period?", {0}),
        ("sleep", "How were participants selected and followed in both sleep studies?", {0, 1}),
    ],
    "numerical": [
        ("exams", "What was the percentage-point difference in Study A misconduct rates?", {2}),
        ("exams", "How many alerts remained unconfirmed in Study B?", {5}),
        ("sleep", "How much more did nightly sleep increase with coaching than usual care?", {2}),
    ],
    "conflicting": [
        ("exams", "Did both studies find a reduction in confirmed misconduct?", {2, 3}),
        ("sleep", "Did both sleep studies establish that sleep intervention lowers fatigue?", {0, 2, 3}),
        ("climate", "Do both studies show consistent cooling in every city?", {2, 3}),
    ],
    "synthesis": [
        ("exams", "Compare privacy concerns and limitations of the two studies.", {4, 6, 7}),
        ("sleep", "What limits applying these findings to night-shift workers?", {4, 5}),
        ("climate", "What measurement limits affect interpreting the canopy and temperature findings?", {4, 5, 6}),
    ],
    "evidence_gap": [
        ("exams", "Did Study A measure facial-recognition accuracy?", {8}),
        ("exams", "What was the p-value in Study A's misconduct comparison?", {8}),
        ("sleep", "Did Study B measure blood pressure outcomes?", {6}),
    ],
    "absent": [("exams", "What are current Indian biometric-data collection laws?", set())],
}


def documents(paper):
    return PAPERS[paper] + [(None, f"The appendix describes literature and survey procedures, section {i}.")
                            for i in range(36)]


def retrieve(repo, paper, vector, question, mode):
    owner = "benchmark-owner"
    if mode == "raw_dense":
        result = repo.collection.query(query_embeddings=[vector], n_results=5,
                                       where=repo.paper_filter(owner, paper), include=["metadatas"])
        ids = [item["chunk_index"] for item in result["metadatas"][0]]
        return ids, {"dense_candidates": len(repo.collection.candidate_ids),
                     "dense_candidate_ids": repo.collection.candidate_ids,
                     "lexical_scanned": 0, "lexical_candidates": 0}
    if mode == "current_dense":
        result = repo.search(owner, paper, vector, 5, question=question)
        return [item.chunk_index for item in result], {
            "dense_candidates": len(repo.collection.candidate_ids),
            "dense_candidate_ids": repo.collection.candidate_ids,
            "lexical_scanned": 0, "lexical_candidates": 0}
    dense = repo.search(owner, paper, vector, 25, question=question)
    terms = repo._terms(question)
    stored = repo.collection.get(where=repo.paper_filter(owner, paper), limit=256,
                                 include=["documents", "metadatas"])
    lexical = []
    if terms:
        for text, metadata in zip(stored["documents"], stored["metadatas"]):
            overlap = len(terms & repo._terms(text))
            if overlap >= min(2, len(terms)):
                lexical.append((overlap / len(terms), metadata["chunk_index"]))
    lexical.sort(key=lambda item: (-item[0], item[1]))
    scores = {}
    for ranking in ([item.chunk_index for item in dense], [index for _, index in lexical[:25]]):
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0) + 1 / (60 + rank)
    return sorted(scores, key=lambda index: (-scores[index], index))[:5], {
        "dense_candidates": len(repo.collection.candidate_ids),
        "dense_candidate_ids": repo.collection.candidate_ids,
        "lexical_scanned": len(stored["ids"]),
        "lexical_candidates": min(25, len(lexical))}


def score(paper, category, question, required, returned, candidates, latency, source_metadata):
    hits = [rank for rank, index in enumerate(returned, 1) if index in required]
    studies = PAPERS[paper]
    return {"paper": paper, "category": category, "question": question,
            "required_chunks": sorted(required), "returned_chunks": returned,
            "returned_sources": [{"paper": source_metadata.get(index, {}).get("paper_id"),
                                  "chunk_index": index,
                                  "page_number": source_metadata.get(index, {}).get("page_number"),
                                  "study": studies[index][0] if index < len(studies) else None}
                                 for index in returned],
            "required_studies": sorted({studies[i][0] for i in required if studies[i][0]}),
            "returned_studies": sorted({studies[i][0] for i in returned
                                        if i < len(studies) and studies[i][0]}),
            "recall_at_5": len(hits) / len(required) if required else None,
            "precision_at_5": len(hits) / 5,
            "mrr": 1 / hits[0] if hits else 0,
            "all_required_recall": len(hits) == len(required) if required else None,
            "candidate_recall": len(required & set(candidates["dense_candidate_ids"])) / len(required)
                if required else None,
            "scope_leak_count": sum(source_metadata.get(index, {}).get("paper_id") != paper
                                    for index in returned),
            "median_latency_ms": statistics.median(latency), **candidates}


def summary(rows):
    labeled = [row for row in rows if row["required_chunks"]]
    return {"labeled_questions": len(labeled), "absent_questions": len(rows) - len(labeled),
            "macro_recall_at_5": statistics.mean(row["recall_at_5"] for row in labeled) if labeled else None,
            "macro_precision_at_5": statistics.mean(row["precision_at_5"] for row in labeled) if labeled else None,
            "macro_mrr": statistics.mean(row["mrr"] for row in labeled) if labeled else None,
            "all_required_rate": statistics.mean(row["all_required_recall"] for row in labeled) if labeled else None,
            "macro_candidate_recall": statistics.mean(row["candidate_recall"] for row in labeled) if labeled else None,
            "median_latency_ms": statistics.median(row["median_latency_ms"] for row in rows),
            "mean_dense_candidates": statistics.mean(row["dense_candidates"] for row in rows),
            "mean_lexical_scanned": statistics.mean(row["lexical_scanned"] for row in rows)}


def run():
    embeddings = LocalEmbeddings(Path(__file__).resolve().parents[1] / "storage/models")
    modes = ("raw_dense", "current_dense", "rrf_experimental")
    output = {mode: [] for mode in modes}
    with tempfile.TemporaryDirectory() as directory:
        repo = VectorRepository(Path(directory))
        repo.collection = TracedCollection(repo.collection)
        for paper in PAPERS:
            values = documents(paper)
            vectors = embeddings.embed([text for _, text in values])
            chunks = [TextChunk(i, i // 5 + 1, text, hashlib.sha256(text.encode()).hexdigest())
                      for i, (_, text) in enumerate(values)]
            repo.save("benchmark-owner", paper, chunks, vectors)
            repo.save("other-owner", paper, [TextChunk(1000, 1, values[0][1], "private")], [vectors[0]])
            cases = [(case_paper, category, question, required)
                     for category, examples in CASES_BY_CATEGORY.items()
                     for case_paper, question, required in examples if case_paper == paper]
            queries = embeddings.embed([case[2] for case in cases])
            for (_, category, question, required), vector in zip(cases, queries):
                for mode in modes:
                    elapsed = []
                    for _ in range(3):
                        start = time.perf_counter()
                        returned, counts = retrieve(repo, paper, vector, question, mode)
                        elapsed.append((time.perf_counter() - start) * 1000)
                    source_result = repo.collection.get(
                        ids=[f"{paper}:{index}" for index in returned],
                        where=repo.paper_filter("benchmark-owner", paper), include=["metadatas"])
                    source_metadata = {item["chunk_index"]: item for item in source_result["metadatas"]}
                    output[mode].append(score(paper, category, question, required,
                                              returned, counts, elapsed, source_metadata))
    print(json.dumps({"fixture": "synthetic manually labeled passages; retrieval only", "k": 5,
                      "modes": {mode: {"summary": summary(rows), "cases": rows}
                                for mode, rows in output.items()}}, indent=2))


if __name__ == "__main__":
    run()
