# Semantic Search Engine

A small, reproducible Python search service for public text and Markdown documents. It ingests normalized chunks, builds a persisted TF-IDF index, and serves validated search requests. Dense, ANN, RRF, and cross-encoder components remain optional: they are used only after matching artifacts and evaluation evidence exist.

## Architecture

```text
raw .txt/.md -> ingestion + deduplication -> chunks.jsonl
                                         |-> lexical TF-IDF index -> Flask API
                                         |-> optional dense index -> RRF (opt-in)
```

The included corpus is public sample material, so document metadata is available through the API. Do not expose `GET /api/v1/documents/<id>` for restricted corpora without adding real document-level authorization before retrieval/output.

## Clean setup

Python 3.10+ is required. The checked-in [requirements.lock](requirements.lock) records the verified environment; `pyproject.toml` remains the project source of truth.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev,production]"
```

If using the exact verified versions instead:

```bash
python -m pip install -r requirements.lock
python -m pip install -e .
```

`hnswlib` is optional in practice: it could not be installed under the verified Python 3.14 environment, so ANN tests are skipped there. Use a supported Python/runtime and `pip install -e .` to enable it.

## Build and evaluate

The repository includes a processed sample corpus. To rebuild everything from raw files:

```bash
python -m src.ingestion.cli --input-dir data/sample_corpus --output-dir data/processed --chunk-size 50 --chunk-overlap 10
python -m src.retrieval.cli build --chunks-file data/processed/chunks.jsonl --index-dir artifacts/indices/lexical_sample
python -m src.evaluation.cli --lexical-index-dir artifacts/indices/lexical_sample --eval-file data/evaluation/synthetic_queries.json --output-report artifacts/reports/pipeline_comparison.json
```

Dense retrieval is an explicit offline job; it never runs in an API request:

```bash
python -m src.retrieval.cli build-dense --chunks-file data/processed/chunks.jsonl --index-dir artifacts/indices/dense_index --model-name all-MiniLM-L6-v2
python -m src.evaluation.cli --lexical-index-dir artifacts/indices/lexical_sample --dense-index-dir artifacts/indices/dense_index --eval-file data/evaluation/synthetic_queries.json --output-report artifacts/reports/pipeline_comparison.json
```

The second command evaluates lexical, dense, and RRF on exactly the supplied queries and records p50/p95 in-process latency. It does not split the current four-query dataset: it is too small to support valid tuning/test claims. Reranking stays disabled and no knowledge graph is added; see [ADR 003](docs/decisions/003_hybrid_retrieval_and_reranking.md) and [ADR 004](docs/decisions/004_graph_retrieval_assessment.md).

## API

Copy `.env.example` values into your deployment environment. Start development mode after building the lexical artifact:

```bash
python -m src.api.app
```

Production on Linux uses Gunicorn (Windows may use Waitress):

```bash
gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 --timeout 30 "src.api.app:create_app()"
```

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/v1/search -H "Content-Type: application/json" -d '{"query":"vector embeddings","top_k":3,"filters":{"extension":".md"}}'
curl http://127.0.0.1:8000/api/v1/documents/a3090451a0a256d7
```

Successful search responses contain `query`, `top_k`, `count`, `latency_ms`, and ranked results. Invalid input returns a consistent JSON error, for example `{"error":{"code":"invalid_request","message":"query must be a non-empty string."}}`. The API accepts only nonempty string queries, `top_k` from 1–20 (configurable), and an `extension` filter. Its request body is limited to 16 KiB by default. CORS is disabled unless a request Origin exactly matches `ALLOWED_ORIGINS`.

## Tests and measured baseline

```bash
pytest --basetemp .pytest-tmp -q
ruff check src tests
```

The committed lexical baseline (`artifacts/baselines/baseline_v1.json`) evaluated all four synthetic queries at MRR, Recall@1, and nDCG@1 of 1.0. The latest measured run is recorded in [the evaluation report](docs/evaluation_report.md). This is a tiny, synthetic, non-held-out set—not a production relevance claim.

## Container deployment and operations

`Dockerfile` deliberately copies only the prebuilt lexical artifact. Build and test it locally where Docker is available:

```bash
docker build -t semantic-search .
docker run --rm -p 8000:8000 semantic-search
```

`compose.yaml` supplies read-only filesystem and CPU/memory limits. The image has a health check at `/health`.

To re-index, build into a new versioned artifact directory, run evaluation/API smoke tests against it, then atomically change `LEXICAL_INDEX_DIR` (and `DENSE_INDEX_DIR` only when they match) and restart the service. Roll back by restoring the prior environment value and restarting. Never overwrite an artifact being served.

## Limitations and future work

- The corpus/evaluation set is intentionally tiny and synthetic; obtain separate training, validation, and held-out test judgments before selecting RRF parameters or enabling a reranker.
- No authentication or document-level ACL exists because the sample corpus is public.
- Supported ingestion formats are `.txt` and `.md`.
- ANN, graph retrieval, learned ranking, and reranking require measured benefit and are not defaults.
