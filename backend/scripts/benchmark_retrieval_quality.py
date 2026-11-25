"""Manually labeled retrieval benchmark; no answer or citation evaluation.

Run: cd backend && .venv/bin/python -m scripts.benchmark_retrieval_quality
The RRF branch is experimental and is not enabled in production.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from importlib.metadata import version
import statistics
import tempfile
import time
from pathlib import Path

from app.providers.embeddings import LocalEmbeddings
from app.repositories.vectors import VectorRepository
from app.services.pdf_processing import TextChunk

K = 5
OWNER = "benchmark-owner"
DEPENDENCIES = ("chromadb", "numpy", "onnxruntime", "tokenizers")


class TracedCollection:
    def __init__(self, collection):
        self.collection = collection
        self.candidate_ids = []
        self.scanned_count = 0

    def __getattr__(self, name):
        return getattr(self.collection, name)

    def query(self, **kwargs):
        result = self.collection.query(**kwargs)
        self.candidates = result["metadatas"][0]
        self.candidate_ids = [item["chunk_index"] for item in self.candidates]
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


def retrieve(repo, paper, vector, question, mode, reranker=None):
    owner = OWNER
    if mode == "raw_dense":
        result = repo.collection.query(query_embeddings=[vector], n_results=K,
                                       where=repo.paper_filter(owner, paper), include=["metadatas"])
        ids = [item["chunk_index"] for item in result["metadatas"][0]]
        return ids, {"dense_candidates": len(repo.collection.candidate_ids),
                     "dense_candidate_ids": repo.collection.candidate_ids,
                     "lexical_scanned": 0, "lexical_candidates": 0}
    if mode in ("current_dense", "openrouter_rerank"):
        if mode == "openrouter_rerank":
            from app.providers.openrouter_routing import RERANK_CANDIDATES
            candidates = repo.search(owner, paper, vector, RERANK_CANDIDATES, question=question)
            order = reranker.rerank(question, [item.text for item in candidates], min(K, len(candidates)))
            result = [candidates[index] for index in order]
        else:
            result = repo.search(owner, paper, vector, K, question=question)
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
    return sorted(scores, key=lambda index: (-scores[index], index))[:K], {
        "dense_candidates": len(repo.collection.candidate_ids),
        "dense_candidate_ids": repo.collection.candidate_ids,
        "lexical_scanned": len(stored["ids"]),
        "lexical_candidates": min(25, len(lexical))}


def score(paper, category, question, required, returned, candidates, latency, source_metadata, studies=None):
    if len(returned) != len(set(returned)) or len(returned) > K:
        raise ValueError("Invalid ranked retrieval results")
    hits = [rank for rank, index in enumerate(returned, 1) if index in required]
    studies = studies if studies is not None else PAPERS[paper]
    return {"paper": paper, "category": category, "question": question,
            "required_chunks": sorted(required), "returned_chunks": returned,
            "returned_sources": [{"paper": source_metadata.get(index, {}).get("paper_id"),
                                  "chunk_index": index,
                                  "page_number": source_metadata.get(index, {}).get("page_number"),
                                  "study": studies[index][0] if index < len(studies) else None}
                                 for index in returned],
            "required_studies": sorted({studies[i][0] for i in required if i < len(studies) and studies[i][0]}),
            "returned_studies": sorted({studies[i][0] for i in returned
                                        if i < len(studies) and studies[i][0]}),
            "recall_at_5": len(hits) / len(required) if required else None,
            "precision_at_5": len(hits) / K if required else None,
            "mrr": (1 / hits[0] if hits else 0) if required else None,
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


def load_research_fixture(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    papers, cases = {}, {}
    digests = set()
    for paper, document in data["papers"].items():
        digest = document["pdf_sha256"]
        if (not document["source"].startswith("https://") or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest) or digest in digests):
            raise ValueError(f"Invalid or duplicate PDF provenance for {paper}")
        digests.add(digest)
        paragraphs = document["chunks"]
        if len(paragraphs) < K or any(type(chunk["page"]) is not int or chunk["page"] < 1
                                      or not chunk["text"].strip() for chunk in paragraphs):
            raise ValueError(f"Invalid PDF excerpts for {paper}")
        papers[paper] = [TextChunk(i, chunk["page"], chunk["text"], hashlib.sha256(chunk["text"].encode()).hexdigest())
                         for i, chunk in enumerate(paragraphs)]
        queries = []
        questions = set()
        for query in document["queries"]:
            category, question = query["category"], query["question"]
            absent = category == "absent" and not query["labels"] and bool(query.get("absence_reason", "").strip())
            if (category not in CATEGORY_REQUIREMENTS or not question.strip() or question in questions
                    or (not absent and (category == "absent" or not query["labels"]))):
                raise ValueError(f"Missing labels or invalid question in {paper}: {question}")
            questions.add(question)
            required = set()
            for label in query["labels"]:
                index, quote = label["chunk"], label["quote"]
                if (type(index) is not int or not 0 <= index < len(paragraphs)
                        or not quote.strip() or quote not in paragraphs[index]["text"]):
                    raise ValueError(f"Label does not match a PDF excerpt in {paper}: {question}")
                required.add(index)
            queries.append((category, question, required))
        if len({category for category, _, labels in queries if labels}) < 2:
            raise ValueError(f"Expected multiple query types in {paper}")
        cases[paper] = queries
    if len(papers) < 2:
        raise ValueError("Expected at least two independent research PDFs")
    return papers, cases, hashlib.sha256(raw).hexdigest()


def regression_totals(report):
    """Use exact counts and fractions so a single missed hit cannot round away."""
    totals = {}
    for row in report["modes"]["current_dense"]["cases"]:
        required = set(row["required_chunks"])
        if not required:
            continue
        hits = len(required.intersection(row["returned_chunks"]))
        first = next((rank for rank, index in enumerate(row["returned_chunks"], 1) if index in required), None)
        group = totals.setdefault(row["paper"], {"queries": 0, "hits": 0, "recall_sum": Fraction(),
                                                "reciprocal_rank_sum": Fraction(), "all_required": 0})
        group["queries"] += 1
        group["hits"] += hits
        group["recall_sum"] += Fraction(hits, len(required))
        group["reciprocal_rank_sum"] += Fraction(1, first) if first else 0
        group["all_required"] += hits == len(required)
    return totals


def baseline_for(report):
    return {"fixture_sha256": report["fixture_sha256"], "k": report["k"], "dependencies": report["dependencies"],
            "papers": {paper: {key: str(value) if isinstance(value, Fraction) else value for key, value in totals.items()}
                       for paper, totals in regression_totals(report).items()}}


def regressions(report, baseline):
    current = baseline_for(report)
    if any(current[key] != baseline[key] for key in ("fixture_sha256", "k", "dependencies")):
        return ["Fixture, K, or dependency versions differ from the measured baseline."]
    if current["papers"].keys() != baseline["papers"].keys():
        return ["The scored paper set differs from the baseline."]
    errors = []
    for paper, totals in current["papers"].items():
        reference = baseline["papers"][paper]
        if totals["queries"] != reference["queries"]:
            errors.append(f"{paper}: scored query count differs from baseline.")
        for key, title in (("hits", "relevant hits (Precision@5)"), ("recall_sum", "sum of Recall@5"),
                           ("all_required", "queries with all labeled evidence"),
                           ("reciprocal_rank_sum", "sum of reciprocal ranks")):
            if key in reference and Fraction(totals[key]) < Fraction(reference[key]):
                errors.append(f"{paper}: {title} fell from {reference[key]} to {totals[key]}.")
    if any(row["scope_leak_count"] for row in report["modes"]["current_dense"]["cases"]):
        errors.append("Current retrieval included an out-of-scope candidate or result.")
    return errors


def run(fixture=None, reranker=None):
    embeddings = LocalEmbeddings(Path(__file__).resolve().parents[1] / "storage/models")
    if fixture:
        papers, cases, digest = load_research_fixture(fixture)
    else:
        papers = {paper: [TextChunk(index, index // 5 + 1, text, hashlib.sha256(text.encode()).hexdigest())
                          for index, (_, text) in enumerate(documents(paper))] for paper in PAPERS}
        cases = {paper: [(category, question, required) for category, examples in CASES_BY_CATEGORY.items()
                         for case_paper, question, required in examples if case_paper == paper] for paper in PAPERS}
        digest = None
    modes = ("current_dense",) if fixture else ("raw_dense", "current_dense", "rrf_experimental")
    if reranker is not None:
        modes += ("openrouter_rerank",)
    output = {mode: [] for mode in modes}
    with tempfile.TemporaryDirectory() as directory:
        repo = VectorRepository(Path(directory))
        repo.collection = TracedCollection(repo.collection)
        for paper, chunks in papers.items():
            vectors = embeddings.embed([chunk.text for chunk in chunks])
            repo.save(OWNER, paper, chunks, vectors)
            repo.save("other-owner", paper, [TextChunk(1000, 1, chunks[0].text, "private")], [vectors[0]])
        for paper, examples in cases.items():
            queries = embeddings.embed([question for _, question, _ in examples])
            for (category, question, required), vector in zip(examples, queries):
                for mode in modes:
                    elapsed = []
                    for _ in range(1 if mode == "openrouter_rerank" else 3):
                        start = time.perf_counter()
                        returned, counts = retrieve(repo, paper, vector, question, mode, reranker)
                        elapsed.append((time.perf_counter() - start) * 1000)
                    source_result = repo.collection.get(
                        ids=[f"{paper}:{index}" for index in returned],
                        where=repo.paper_filter(OWNER, paper), include=["metadatas"])
                    source_metadata = {item["chunk_index"]: item for item in source_result["metadatas"]}
                    row = score(paper, category, question, required, returned, counts, elapsed, source_metadata,
                                studies=[] if fixture else None)
                    row["scope_leak_count"] += sum(item.get("owner_id") != OWNER or item.get("paper_id") != paper
                                                   for item in repo.collection.candidates)
                    output[mode].append(row)
    return {"fixture": "labeled research PDF excerpts" if fixture else "synthetic manually labeled passages; retrieval only",
            "fixture_sha256": digest, "k": K, "dependencies": {name: version(name) for name in DEPENDENCIES},
            "modes": {mode: {"summary": summary(rows), "cases": rows} for mode, rows in output.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--live-rerank", action="store_true",
                        help="Compare against one paid-quota-free rerank call per public fixture query.")
    baseline_args = parser.add_mutually_exclusive_group()
    baseline_args.add_argument("--baseline", type=Path)
    baseline_args.add_argument("--write-baseline", type=Path)
    args = parser.parse_args()
    if (args.baseline or args.write_baseline) and not args.fixture:
        parser.error("Baseline operations require --fixture")
    for path in (args.fixture, args.baseline):
        if path is not None and not path.is_file():
            parser.error(f"Required retrieval file is missing: {path}")
    if args.live_rerank:
        from pymongo import MongoClient
        from app.config import get_settings
        from app.providers.openrouter_answers import OpenRouterAnswerProvider
        from app.providers.openrouter_routing import ModelRouter, SharedRequestBudget
        if not args.fixture or args.write_baseline:
            parser.error("Live reranking requires a public labeled --fixture and cannot write a baseline.")
        settings = get_settings()
        if not settings.openrouter_api_key:
            parser.error("Live reranking requires OPENROUTER_API_KEY.")
        with MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000) as client:
            budget = SharedRequestBudget(client[settings.mongodb_database]["llm_request_budget"],
                                         "openrouter", settings.openrouter_daily_limit, settings.openrouter_minute_limit)
            provider = OpenRouterAnswerProvider(settings.openrouter_api_key.get_secret_value(),
                ModelRouter(rerank=settings.openrouter_rerank_model), budget=budget)
            try:
                report = run(args.fixture, provider)
            except Exception as exc:
                parser.exit(1, f"Live reranking comparison incomplete ({type(exc).__name__}).\n")
    else:
        report = run(args.fixture)
    if args.write_baseline:
        args.write_baseline.write_text(json.dumps(baseline_for(report), indent=2) + "\n")
    report["regressions"] = regressions(report, json.loads(args.baseline.read_text())) if args.baseline else []
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["regressions"]))
