# Retrieval CI

GitHub Actions runs the existing retrieval benchmark with the production `VectorRepository.search` method, ChromaDB, and the CPU MiniLM embedder. It prints both JSON reports and uploads them in the `retrieval-results` artifact. The first run may download the checksum-verified model archive used by Chroma. Subsequent runs can reuse the model cache. The workflow needs no API keys or MongoDB service.

## Fixtures

`research_excerpts.json` contains selected page-mapped chunks extracted from two independent open-access research PDFs using CiteWise's PyMuPDF extraction and MiniLM tokenization. The fixture records the original PDF SHA-256 digests. Ten questions cover factual lookup, methodology, numerical facts, synthesis, and limitations. The relevance labels were checked against the passages in the PDFs. The loader verifies that each quoted label occurs in the specified chunk before scoring.

- Regan Doyle, Tobias Leichtle, Stephan Pauleit, and Hannes Taubenböck. *The influence of city size versus urban form on land surface temperature variation and the surface urban heat island effect*. PLOS ONE, 2026. [Article and license](https://doi.org/10.1371/journal.pone.0340060).
- Diogo Pernes, Gonçalo M. Correia, and Afonso Mendes. *Multi-Target Cross-Lingual Summarization*. Findings of EMNLP, 2024. [Article](https://aclanthology.org/2024.findings-emnlp.755/) and [ACL license](https://aclanthology.org/faq/copyright/).

`holdout_excerpts.json` was labeled and frozen before the ranking change. It covers a physics observation paper and a public-health systematic review and meta-analysis. Ten questions have quote-backed labels, including a contrast between the pooled effect and one trial's result. Two questions document information absent from their respective papers. The report records those two queries without scoring them for retrieval relevance or answer refusal.

- R. Abbott et al. *GW190521: A Binary Black Hole Merger with a Total Mass of 150 M⊙*. Physical Review Letters, 2020. [Article and license](https://doi.org/10.1103/PhysRevLett.125.101102).
- Fiona Curran et al. *Efficacy, characteristics, behavioural models and behaviour change strategies, of non-workplace interventions specifically targeting sedentary behaviour*. PLOS ONE, 2021. [Article and license](https://doi.org/10.1371/journal.pone.0256828).

The included text is adapted from the original PDFs by extraction, whitespace normalization, and chunk selection under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Baseline and failure conditions

The checked-in baseline was measured on macOS arm64 with Python 3.14 and the pinned dependencies in `requirements-retrieval-ci.txt`. The measured macro Recall@5 was **0.95**, macro Precision@5 was **0.24**, and nine of ten queries retrieved all labeled evidence.

The development fixture has 13 labeled chunk occurrences across 50 result slots, making 26% its maximum labeled Precision@5. The missing continuation originally ranked sixth for a two-part question. It contained the method that answered the second part, which used different wording from the passage. After preserving one adjacent candidate near the dense cutoff, development Recall@5 reached 100%, Precision@5 reached 26%, and 10 of 10 queries had complete labeled evidence. The holdout was unchanged at 61.7% recall, 24% precision, and 5 of 10 queries with complete evidence. Both fixtures retain their measured pre-change baselines.

The gate compares exact per-paper counts and fractions at K=5. A drop in relevant hits, aggregate recall, complete-evidence queries, or the reciprocal rank of the first relevant hit summed across queries fails the run. An improvement on one paper cannot offset a loss on the other. Out-of-scope candidates and results also fail. Candidate recall and latency are diagnostic.

Run from `backend` in an environment with the pinned dependencies:

```sh
python -m unittest discover -s tests -p 'test_retrieval*.py'
python -m scripts.benchmark_retrieval_quality \
  --fixture benchmarks/retrieval_ci/research_excerpts.json \
  --baseline benchmarks/retrieval_ci/baseline.json
python -m scripts.benchmark_retrieval_quality \
  --fixture benchmarks/retrieval_ci/holdout_excerpts.json \
  --baseline benchmarks/retrieval_ci/holdout_baseline.json
```

When intentionally changing a fixture or dependency version, use `--write-baseline` in place of `--baseline` with the corresponding baseline path. Review the per-query results and commit the measured baseline with the reviewed change.

CI scores relevance within these selected, fixed excerpts. Whole-PDF ingestion, text-extraction changes, answer accuracy, and general retrieval accuracy across research disciplines are outside this gate's coverage.
