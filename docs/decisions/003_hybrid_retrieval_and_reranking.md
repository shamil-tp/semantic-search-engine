# ADR 003: Explicit Hybrid Retrieval and Bounded Reranking

## Decision

Combine lexical TF-IDF and dense bi-encoder results with reciprocal-rank fusion (RRF). Keep it opt-in through a dedicated CLI command. An optional sentence-transformers cross-encoder reranks a bounded prefix of fused candidates.

## Rationale

RRF combines rankings without treating lexical and cosine scores as comparable. It is a small, stable baseline for determining whether lexical and dense retrieval are complementary. Cross-encoder inference is more expensive, so it receives only a configurable top-N candidate set and is never enabled implicitly.

## Evaluation policy

Compare lexical, dense, fused, and fused-plus-reranker runs on the same held-out query set. Record Recall@k, MRR, nDCG, latency, and candidate/rerank depths. Do not introduce learning-to-rank until reliable relevance labels and a measured need exist.
