# ADR 002: HNSW ANN Index for Phase 5

## Decision

Use the optional `hnswlib` HNSW implementation for scalable approximate cosine retrieval. The existing `DenseSearchEngine` remains the exact kNN implementation and evaluation oracle.

## Rationale

HNSW supports incremental vector additions and provides a practical recall/latency control through `ef_search`. It is appropriate for the project before considering vector-database infrastructure. IVF/PQ and a vector database are deferred until benchmarks demonstrate a memory, scale, metadata-filtering, or concurrency need.

## Artifact and recovery policy

An ANN index persists its HNSW binary, ordered chunk catalog, and a versioned metadata manifest. Loading rejects incomplete, incompatible, or corrupt artifacts and instructs the operator to rebuild from the known-good dense index. ANN artifacts are excluded from Git.

## Operating policy

Do not replace exact retrieval blindly. Benchmark Recall@k against the exact index, along with query latency, memory, build time, and update cost, before enabling ANN for a target corpus. Tune `ef_search` from measured recall/latency evidence.
