# Evaluation report — 2026-10-24

Command run:

```bash
python -m src.evaluation.cli --lexical-index-dir artifacts/indices/lexical_sample --eval-file data/evaluation/synthetic_queries.json --output-report artifacts/reports/pipeline_comparison.json
```

The lexical TF-IDF pipeline evaluated four of four supplied synthetic queries. MRR, Recall@1/3/5, and nDCG@1/3/5 were all `1.0`. In 40 in-process searches (ten repetitions of each query), p50 latency was `0.339 ms` and p95 was `0.368 ms` on the verification host.

No dense index was present and no cross-encoder was downloaded, so dense, RRF, and reranked comparisons were not run. The corpus contains only two documents and the judgments have no validation/test split; these results are a reproducibility smoke benchmark, not evidence for production quality or for enabling hybrid retrieval.
