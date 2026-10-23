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
