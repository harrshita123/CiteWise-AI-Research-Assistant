# Retrieval quality benchmark

Run from `backend`:

```sh
.venv/bin/python -m scripts.benchmark_retrieval_quality
```

The script creates three temporary, synthetic PDFs represented by stored chunks: remote examinations, sleep research, and urban heat. It embeds them with CiteWise's existing local MiniLM model and queries the existing Chroma collection through the selected-paper and owner filters. Nineteen questions have manually labeled required chunk indexes and study identities. One question has no relevant PDF chunk and is excluded from recall and MRR averages. Cases are grouped by factual lookup, multi-study comparison, methodology, numerical evidence, conflicting findings, synthesis, explicit evidence gap, and absent information. The output contains every label, returned chunk and page, candidate ID, and per-question metric. No Groq or Tavily calls are made.

## Results from 14 November 2026

| Ranking | Recall@5 | Precision@5 | MRR | All required chunks found | Dense candidate recall | Median retrieval latency | Mean dense candidates | Mean lexical chunks scanned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Chroma top five | 0.851 | 0.284 | 0.816 | 0.789 | 0.851 | 0.451 ms | 5.0 | 0 |
| Current dense + term normalization and selection | 0.781 | 0.253 | 0.816 | 0.684 | 0.895 | 0.916 ms | 31.9 | 0 |
| Experimental lexical + RRF | 0.833 | 0.263 | 0.842 | 0.737 | 0.895 | 1.737 ms | 31.9 | 44.5 |

These are macro averages across labeled questions. Precision@5 divides by five even when fewer than five passages are returned. MRR is the reciprocal rank of the first relevant passage. All-required recall is true only when every manually labeled chunk appears in the final top five. Latency is the median of three warm retrieval runs per question; embedding and setup time are excluded. It is a local timing sample, not a service latency guarantee. No owner or paper scope leaks appeared in the returned chunks.

## Diagnosis

- In four of six cases where the current final top five missed required evidence, all required chunks were already among its dense candidates. These losses occur in final ranking or selection, especially methodology, multi-study comparison, and conflicting-finding cases.
- Two cases lacked required passages in the dense candidate set. Final selection alone cannot recover those cases.
- Raw Chroma top five outperformed the current selection on this fixture. This does not establish that raw ranking is better in general; the existing selection also contains safeguards for other question types.
- The benchmark-only RRF implementation scanned existing stored chunks without re-indexing. It recovered one explicit evidence-gap passage but remained below raw dense top-five recall and increased median latency. It remains disabled in production.
- The no-evidence case tests retrieval scope only. Returning a passage for that question does not by itself mean the answer pipeline would make an unsupported claim or fail to abstain.

## Next measured priority

Evaluate final passage selection against study and requested-dimension coverage **within the already retrieved candidate set**, with a guard against unrelated passages. This targets four observed misses without an extra embedding, model call, or index. Re-run the same labels and add real, permission-safe PDF labels before changing production ranking. If candidate recall remains the bottleneck, then evaluate bounded candidate expansion separately.

## Benchmark isolation audit

`backend/app` does not import this script or its fixture labels. The production research route embeds the submitted question, checks access to the selected paper for the signed-in owner, and passes that question and the same owner/paper IDs to `VectorRepository.search`. The repository applies both IDs in its Chroma `where` filter. Its study and topic heuristics inspect the current question and candidate text; they do not read benchmark questions, expected indexes, or synthetic passages. The test module imports only the benchmark's scoring helpers and fixture catalog. This is a static code audit, not a browser test.

## Limits

The passages and relevance labels are synthetic and few in number. Study identity is a benchmark label, not new production metadata. The experimental RRF uses the current 25-passage dense selection as one ranking and a bounded scan of at most 256 stored chunks as the other; other fusion designs may behave differently. The benchmark does not assess generated answer completeness, factual support, abstention, or citation correctness. It cannot establish general retrieval accuracy across real user PDFs.
