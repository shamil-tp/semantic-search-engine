# Semantic Search Engine — Project Execution Plan

## Project Goal
Build a semantic search engine in Python from raw documents to a tested, deployed web API. Prioritize a working, measurable baseline before adding advanced ML components. Do not implement every candidate technology; add complexity only when evaluation shows a clear benefit.

## Working Rules
- Work through phases in order. Do not start a later phase until the current phase's exit criteria pass.
- Before coding each phase: inspect the repository, summarize the plan, list files to change, and identify risks.
- Keep modules small, typed where practical, documented, and testable.
- Use configuration and environment variables for paths, model names, secrets, and deployment settings. Never commit secrets or datasets with unclear licenses.
- Add unit tests with each feature; add integration tests at API and pipeline boundaries.
- Record key decisions and trade-offs in `docs/decisions/`.
- Track metrics, datasets, model versions, and experiment settings so results are reproducible.
- Prefer simple, proven components. Avoid introducing TensorFlow and PyTorch together unless a specific requirement justifies both.
- After each phase, run formatting, linting, tests, and a small end-to-end smoke test. Report commands run, results, limitations, and the next phase.
- Do not claim a phase is complete unless its exit criteria are demonstrably met.

## Recommended Initial Stack
- Python, NumPy, pandas: data inspection, cleaning, and analysis.
- Flask: REST API.
- scikit-learn: TF-IDF, BM25-style baseline if implemented, SVM, metrics, and classical ML utilities.
- sentence-transformers / PyTorch: dense embeddings and optional cross-encoder reranking.
- FAISS or hnswlib: vector retrieval when the corpus justifies an ANN index. Start with exact cosine/dot-product search for small datasets.
- pytest, Ruff, and a dependency lock file: quality and reproducibility.
- SQLite or PostgreSQL for document metadata; keep vector index persistence separate.
- Docker for reproducible deployment.

TensorFlow is optional, not a default dependency. Add it only if a selected model or experiment specifically requires it.

## Phase 0 — Scope and Acceptance Criteria
**Tasks**
1. Define the search domain, users, document types, expected corpus size, update frequency, and language(s).
2. Define API inputs/outputs and whether documents are public, private, or user-scoped.
3. Create a small representative, legally usable dataset and a labeled query-to-relevant-document evaluation set.
4. Set initial targets for relevance (e.g. Recall@k / nDCG@k), latency, memory, and index build time.
5. Create a repository, `.gitignore`, `README.md`, `pyproject.toml`, `.env.example`, and `docs/`.

**Exit criteria**
- Scope, data rights, API contract, evaluation protocol, and target metrics are documented.
- A clean environment can install the project dependencies.

## Phase 1 — Data Ingestion and Preparation
**Tasks**
1. Implement document loaders for the initial required formats only.
2. Normalize whitespace/encoding, remove unusable records, deduplicate, and preserve document IDs and source metadata.
3. Split long documents into configurable chunks with overlap; retain source-document and chunk offsets.
4. Use pandas/NumPy for corpus profiling: missing values, duplicates, lengths, language distribution, and class imbalance if labels exist.
5. Save a versioned, reproducible cleaned corpus.

**Deliverables**
- `src/ingestion/`, data schema, cleaning report, sample dataset, and tests.

**Exit criteria**
- Re-running ingestion produces stable IDs and reproducible outputs.
- Malformed files and empty documents are handled safely.

## Phase 2 — Baseline Search
**Tasks**
1. Build a lexical baseline using tokenization + TF-IDF (and BM25 if useful).
2. Implement query preprocessing consistently with document preprocessing.
3. Return ranked document/chunk IDs, scores, and source metadata.
4. Evaluate on the labeled query set; record Recall@k, MRR, and nDCG@k.
5. Add a simple command-line evaluation script.

**Exit criteria**
- Baseline is repeatable, tested, and has recorded metrics.
- Baseline results are retained for comparison with every later approach.

## Phase 3 — Classical Text ML and Query Understanding
**Tasks**
1. Add optional intent classification only if the product needs intent-specific behavior. Start with TF-IDF + Linear SVM; compare fastText if data volume and language support justify it.
2. Add NER only if entities such as people, organizations, products, or places materially improve retrieval/filtering.
3. Add query normalization, synonyms, spelling handling, and metadata filters where evidence supports them.
4. Treat query expansion as an experiment: compare original query, rule/synonym expansion, and a small Seq2Seq/LLM-based method only if needed.
5. Prevent expansion from changing the user's intent; retain the original query and log expanded terms for evaluation.

**Exit criteria**
- Every optional component has a measurable use case, tests, and an ablation result.
- No classifier or expansion stage is enabled by default without evidence of improvement.

## Phase 4 — Dense Semantic Retrieval
**Tasks**
1. Select a compact pretrained bi-encoder such as Sentence-BERT or a suitable modern sentence-transformer.
2. Generate normalized embeddings for document chunks and queries using the same model/version.
3. Store embeddings with stable chunk IDs and metadata; record model name, revision, dimension, and preprocessing settings.
4. Implement exact vector similarity first; verify cosine/dot-product behavior and embedding dimensions.
5. Compare dense retrieval against the lexical baseline and test multilingual/domain performance if relevant.

**Exit criteria**
- Embeddings are reproducible and correctly mapped to source chunks.
- Dense retrieval is evaluated against the same query set and baseline metrics.

## Phase 5 — Vector Index and Scalable Retrieval
**Tasks**
1. Keep exact kNN for small corpora and as an evaluation oracle.
2. Benchmark an ANN option only when exact search misses latency or scale targets:
   - HNSW (e.g. hnswlib or FAISS HNSW) for a strong general-purpose starting point.
   - IVF/PQ in FAISS for larger corpora or memory-constrained deployments after benchmarking.
   - Annoy when its trade-offs and mostly static index fit the use case.
3. Measure Recall@k against exact kNN, query latency, memory, build time, and update cost.
4. Implement index persistence, rebuild/versioning, and safe recovery if an index is missing or corrupt.
5. Choose a vector database only if persistence, metadata filtering, concurrent access, or operational needs justify it; otherwise a local ANN library may be sufficient.

**Exit criteria**
- The chosen index meets agreed recall, latency, and memory targets.
- Index versioning and rebuild procedures are documented and tested.

## Phase 6 — Hybrid Retrieval and Reranking
**Tasks**
1. Combine lexical and dense candidates when evaluation shows complementary strengths. Use a simple, documented fusion method first.
2. Optionally rerank the top-N candidates with a cross-encoder; benchmark quality and latency.
3. Add XGBoost/LightGBM learning-to-rank only when there is enough reliable relevance-labeled training data and meaningful features.
4. Prevent train/test leakage; split evaluation by query or source as appropriate.
5. Tune candidate count and reranking depth on validation data, then evaluate once on a held-out test set.

**Exit criteria**
- Each added stage improves agreed metrics enough to justify its latency and operational cost.
- The final ranking pipeline is versioned and reproducibly evaluated.

## Phase 7 — Optional Knowledge Graph Integration
**Tasks**
1. Add this phase only if entity relationships improve the target search domain.
2. Extract entities/relations and store canonical entities and edges with provenance.
3. Begin with metadata/entity-based boosts or graph traversal; do not start with a Graph Neural Network by default.
4. Consider a GNN only with sufficient graph data, a defined learning objective, and an evaluation proving benefit.
5. Keep graph-derived evidence traceable to source documents.

**Exit criteria**
- Graph integration provides measurable gains and does not introduce unsupported facts or untraceable results.

## Phase 8 — Flask API and Application Integration
**Tasks**
1. Create versioned endpoints, for example:
   - `GET /health`
   - `POST /api/v1/search` with query, top_k, and supported filters
   - `GET /api/v1/documents/<id>` if document access is permitted
2. Validate request schemas, limits, types, and empty queries; return consistent JSON errors.
3. Load models/indexes once at startup where supported; avoid rebuilding them per request.
4. Add timeouts, request-size limits, structured logging, CORS only for known frontends, and rate limiting if exposed publicly.
5. Enforce document-level access controls before returning results. Never rely on vector similarity as authorization.
6. Document API usage with example requests and responses.

**Exit criteria**
- API integration tests cover valid, invalid, empty, oversized, and unauthorized requests.
- A clean startup loads the intended model and index and serves a successful search.

## Phase 9 — Evaluation, Testing, and Optimization
**Tasks**
1. Maintain unit, integration, regression, and end-to-end tests.
2. Compare lexical, dense, hybrid, and reranked pipelines on the same held-out queries.
3. Measure p50/p95 latency, throughput, memory/VRAM, index build time, and relevance metrics.
4. Profile before optimizing; batch embedding generation and cache only where beneficial.
5. Test empty corpus, duplicate documents, unseen queries, index/model mismatch, and corrupted persistence.
6. Record known limitations, failure examples, and a model/data card.

**Exit criteria**
- Tests pass and target metrics are met or deviations are explicitly documented.
- Results can be reproduced from documented commands and configuration.

## Phase 10 — Packaging and Deployment
**Tasks**
1. Add a production WSGI server (e.g. Gunicorn where supported), Dockerfile, and deployment configuration.
2. Separate development and production configuration; store secrets in deployment environment variables.
3. Build the corpus and index as a repeatable job or artifact, not on every API request.
4. Add startup/readiness checks, logs, resource limits, and a documented model/index update procedure.
5. Deploy to a chosen platform or server; verify health and search endpoints after deployment.
6. Document rollback to the previous working model/index and safe re-indexing steps.

**Exit criteria**
- Deployment can be recreated from repository instructions.
- Health checks, search smoke tests, logs, and rollback procedure are verified in the target environment.

## Phase 11 — Final Handoff
**Deliverables**
- Working source code and dependency lock file.
- README with architecture, setup, ingestion, evaluation, API, and deployment instructions.
- Architecture diagram and key decision records.
- Sample dataset or a script to obtain permitted data.
- Evaluation report with baseline-to-final comparison and latency/resource measurements.
- API examples, test report, known limitations, and future-work list.

**Definition of Done**
- A new developer can set up the project from a clean environment.
- The complete pipeline from ingestion to deployed search works end-to-end.
- Tests pass, evaluation results are reported honestly, and all optional complexity has a measured justification.

## Suggested Repository Layout
```text
semantic-search/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── retrieval/
│   │   ├── lexical.py
│   │   ├── dense.py
│   │   ├── index.py
│   │   └── reranking.py
│   ├── evaluation/
│   ├── api/
│   └── config.py
├── tests/
├── scripts/
├── data/                 # keep large/private data out of Git
├── artifacts/            # generated indexes/models; usually out of Git
├── docs/
│   └── decisions/
├── .env.example
├── .gitignore
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Initial Implementation Order
1. Scope + evaluation dataset.
2. Ingestion and chunking.
3. TF-IDF baseline.
4. Dense embeddings.
5. Exact vector search, then ANN benchmark.
6. Flask API.
7. Hybrid retrieval / cross-encoder only if metrics justify it.
8. Deployment and final evaluation.
9. Add intent classification, NER, query expansion, learning-to-rank, or graph/GNN features only for demonstrated needs.
