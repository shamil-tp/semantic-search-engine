# Semantic Search Engine

A modular, reproducible semantic search engine in Python spanning from raw document ingestion and preprocessing to vector indexing, hybrid retrieval, and REST API deployment.

## Phase 1 — Data Ingestion and Preparation

Phase 1 provides a modular, reliable pipeline that discovers raw text (`.txt`) and Markdown (`.md`) files, performs safe UTF-8 loading, normalizes whitespace and Unicode (NFC), strips duplicate and empty documents, generates configurable overlapping chunks, profiles corpus statistics using pandas/NumPy, and exports structured JSONL artifacts.

### Repository Layout

```text
├── src/
│   ├── __init__.py
│   └── ingestion/
│       ├── __init__.py
│       ├── schemas.py       # ProcessedDocument and DocumentChunk dataclass models
│       ├── loader.py        # Recursive file discovery and safe UTF-8 loading
│       ├── cleaner.py       # Unicode normalization, boundary preservation, deduplication
│       ├── chunker.py       # Overlapping word-based chunker with character offsets
│       ├── statistics.py    # pandas/NumPy corpus profiling and metrics summary
│       ├── pipeline.py      # Pipeline orchestrator
│       └── cli.py           # Command-line interface
├── tests/
│   └── test_ingestion.py    # Comprehensive unit and integration test suite
├── data/
│   ├── sample_corpus/       # Representative test documents (.txt, .md)
│   └── processed/           # Processed JSONL outputs (ignored by Git)
├── pyproject.toml           # Project metadata and test configuration
└── README.md
```

### Installation

Requires Python 3.10+:

```bash
pip install -e .
pip install pytest
```

### Running the Ingestion Pipeline

To run data ingestion on a directory of documents:

```bash
python -m src.ingestion.cli --input-dir data/sample_corpus --output-dir data/processed --chunk-size 500 --chunk-overlap 50
```

## Phase 2 — Baseline Search (TF-IDF Lexical Retrieval)

Phase 2 builds a reproducible lexical retrieval baseline using `scikit-learn`'s `TfidfVectorizer` with sparse matrix cosine similarity scoring, index persistence, and evaluation against labeled queries.

### Building the Lexical Index

To index Phase 1 chunks into a persistent TF-IDF index:

```bash
python -m src.retrieval.cli build --chunks-file data/processed/chunks.jsonl --index-dir artifacts/indices/lexical_index
```

### Searching the Index

To query the saved index without re-indexing:

```bash
python -m src.retrieval.cli search --index-dir artifacts/indices/lexical_index --query "vector embeddings in semantic search" --top-k 5
```

### Running Evaluation Benchmarks

To evaluate the index against ground-truth query relevance judgments and save a versioned baseline report:

```bash
python -m src.retrieval.cli evaluate --index-dir artifacts/indices/lexical_index --eval-file data/evaluation/synthetic_queries.json --output-report artifacts/baselines/baseline_v1.json --k-values 1,3,5,10
```

> **Note on Reports**: The evaluator refuses to silently overwrite existing reports unless `--force` is explicitly provided, ensuring baseline reproducibility and immutability.

### Search Result Schema

Each retrieved search item includes:
- `rank`: Result rank starting at 1.
- `chunk_id`: Identifier formatted as `{document_id}_c{chunk_index}`.
- `document_id`: Parent document identifier.
- `score`: Normalized TF-IDF cosine similarity score between query and chunk vector.
- `text`: Chunk text or excerpt.
- `source_path`: Original file path.
- `chunk_index`: 0-indexed position within the document.
- `metadata`: Parent document metadata (file size, timestamps, word/char counts).

### Evaluation Metrics

- **Recall@k**: Fraction of known relevant documents/chunks retrieved in the top $k$ items.
- **Mean Reciprocal Rank (MRR)**: Mean reciprocal rank ($1 / \text{rank}$) of the first retrieved relevant item across queries.
- **nDCG@k**: Normalized Discounted Cumulative Gain evaluating ranked position quality with logarithmic position discounts.

### Metric Meaning and Limitations
TF-IDF measures exact lexical token overlap and frequency inverse weights. It does not measure semantic similarity, synonyms, or conceptual closeness (which will be addressed in dense retrieval in Phase 4). Scores are lexical relevance signals rather than probabilities.

## Phase 5 — Vector Index and Scalable Retrieval

Phase 5 adds an optional persisted HNSW approximate-nearest-neighbour (ANN) index over the embeddings produced by Phase 4. Exact dense search remains available as the comparison oracle; the ANN index does not re-embed documents.

Install the new dependency from the project metadata, build the exact dense index, then build the ANN artifact from it:

```bash
python -m src.retrieval.cli build-ann --dense-index-dir artifacts/indices/dense_index --index-dir artifacts/indices/hnsw_index
python -m src.retrieval.cli search-ann --index-dir artifacts/indices/hnsw_index --query "vector embeddings in semantic search" --top-k 5
```

The HNSW artifact includes `hnsw_index.bin`, `chunks.jsonl`, and `ann_metadata.json`. If any artifact is absent, incompatible, or corrupt, rebuild it from the matching dense index rather than attempting partial recovery. Before selecting ANN in production, measure Recall@k against exact dense retrieval and record latency, memory, build time, and update cost. Tune `--ef-search` only from those measurements.

## Phase 3 — Classical Text ML and Query Understanding

Phase 3 introduces classical text machine learning components for query understanding:
- **Intent Classification**: TF-IDF + Linear SVM (`LinearSVC`) with calibrated confidence scores classifying user queries into `informational`, `technical_spec`, or `troubleshooting`.
- **Named Entity Recognition (NER) & Metadata Filtering**: Extracts structured entities (file extensions, schema attributes, document IDs, algorithms) to apply deterministic metadata filters.
- **Spelling Normalization & Controlled Expansion**: Maps typos to canonical domain terms and attaches domain synonyms while preserving the original query and logging all expansion terms.

> **Phase 3 Rule**: In accordance with `CLAUDE.md`, all query understanding stages are **disabled by default** to prevent unintended query drift until an evaluation justifies activation. See [docs/decisions/001_query_understanding_ablation.md](docs/decisions/001_query_understanding_ablation.md).

### Training the Intent Classifier

To train and persist the intent classifier:

```bash
python -m src.query.cli train-intent --output-dir artifacts/models/intent_model
```

### Analyzing Queries via CLI

To inspect detected intents, extracted entities, and proposed expansions:

```bash
python -m src.query.cli analyze --query "what is sematic search in .md files" --intent-model artifacts/models/intent_model
```

### Searching with Query Understanding Flags

Query understanding features can be selectively enabled during search:

```bash
python -m src.retrieval.cli search \
  --index-dir artifacts/indices/lexical_sample \
  --query "what is sematic search in .md files" \
  --enable-intent --intent-model artifacts/models/intent_model \
  --enable-ner --apply-filters \
  --enable-expansion
```

#### CLI Options

- `--input-dir`, `-i`: (Required) Path to the raw document directory.
- `--output-dir`, `-o`: (Required) Directory where processed JSONL files and statistics will be saved.
- `--chunk-size`: Number of words per chunk (default: `500`).
- `--chunk-overlap`: Number of overlapping words between consecutive chunks (default: `50`).
- `--extensions`: Supported file extensions (default: `.txt .md`).
- `--verbose`, `-v`: Enable debug-level logging.

### Output Schemas

Outputs are generated in the specified `--output-dir`:

1. **`documents.jsonl`**: Each line contains a cleaned, deduplicated document:
   - `document_id`: Deterministic 16-character SHA-256 hash of normalized source path.
   - `source_path`: Normalized POSIX source file path.
   - `content_hash`: SHA-256 hash of normalized text used for deduplication.
   - `text`: Cleaned text preserving paragraph boundaries.
   - `metadata`: File name, extension, size in bytes, and timestamps.

2. **`chunks.jsonl`**: Each line contains a chunk ready for indexing:
   - `chunk_id`: Deterministic ID formatted as `{document_id}_c{chunk_index}`.
   - `document_id`: Parent document ID.
   - `source_path`: Source file path.
   - `chunk_index`: 0-indexed position within the document.
   - `text`: Chunk text content.
   - `start_char`: Starting character offset in the parent document text.
   - `end_char`: Ending character offset in the parent document text.
   - `metadata`: Parent metadata extended with `word_count` and `char_count`.

3. **`statistics.json`**: Tabular profiling generated with pandas and NumPy:
   - Discovery counts, skipped unsupported files, failures with error breakdown, empty files, and duplicates.
   - Document length summary (min, max, mean, median for words and characters).
   - Chunk length summary (min, max, mean, median for words and characters).

### Running Tests

Execute the automated test suite using `pytest`:

```bash
pytest -v
```

### Architectural Decisions & Known Limitations

- **Chunking Strategy**: Word-boundary regex splitting (`\S+`) is used to guarantee that chunks split cleanly at words rather than arbitrary characters, while tracking character offsets against the parent text.
- **Deduplication**: Content deduplication uses SHA-256 hashes of NFC-normalized, whitespace-standardized text. Indiscriminate stopword or punctuation stripping is avoided to preserve downstream semantic context.
- **Deterministic IDs**: Source path hashing ensures repeatable document and chunk IDs across runs.
- **Limitations**: Currently supports `.txt` and `.md`. Document formats like PDF and DOCX or OCR workflows will require dedicated loaders in future extensions.
