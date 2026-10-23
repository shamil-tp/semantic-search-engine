# ADR 001: Query Understanding and Text ML Ablation Analysis (Phase 3)

## Context & Requirement
In accordance with `CLAUDE.md` Phase 3 rules:
1. Intent classification, NER, query normalization, synonyms, spelling handling, and metadata filters must have demonstrable use cases, tests, and recorded ablation results.
2. Query expansion must not alter the user's intent, must retain the original query, and must log expanded terms for evaluation.
3. **Exit criterion**: *Every optional component has a measurable use case, tests, and an ablation result. No classifier or expansion stage is enabled by default without evidence of improvement.*

## Implemented Components
1. **Intent Classification** (`src/query/intent.py`):
   - Model: TF-IDF + Linear SVM (`LinearSVC`) with calibrated softmax-normalized decision margins.
   - Classes: `informational`, `technical_spec`, `troubleshooting`.
   - Decision on fastText: fastText requires compilation with C++ extensions which is not present in standard wheels for Python 3.14 on Windows; scikit-learn's `LinearSVC` fulfills the Phase 3 goal with minimal dependencies and millisecond latency.
2. **Entity Recognition & Metadata Filtering** (`src/query/ner.py`):
   - Deterministic pattern-based extractor for file extensions (`.md`, `.txt`), IR metrics (`recall@k`, `mrr`), IDs (`document_id`, `chunk_id`), algorithms (`tf-idf`, `svm`), and CLI flags.
   - Converts recognized entities into structured search filter constraints (e.g. `extension='.md'`).
3. **Query Expansion & Spelling Correction** (`src/query/expansion.py`):
   - Normalizes known technical typos (e.g., `sematic` -> `semantic`, `preprocesing` -> `preprocessing`).
   - Appends controlled domain synonyms while retaining original tokens and logging term expansions.

## Ablation Study Results

Evaluation against `data/evaluation/synthetic_queries.json`:

| Configuration | MRR | Recall@1 | Recall@5 | nDCG@1 | nDCG@5 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Phase 2, Raw Queries)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Stored in `artifacts/baselines/baseline_v1.json` |
| **+ Query Expansion & Normalization** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Stored in `artifacts/baselines/ablation_expansion.json` |

### Qualitative Findings & Drift Analysis:
- On clean benchmark queries containing exact keywords, lexical TF-IDF already retrieves the correct document at rank 1.
- Expanding synonyms on short queries introduces supplementary vocabulary (`dense retrieval`, `concept search`, `vector search`) without degrading top-1 precision, and enables matching queries with spelling errors (e.g., "sematic search").
- However, excessive expansion risks query drift on ambiguous queries by inflating secondary tokens.

## Decision
- In strict adherence to `CLAUDE.md`, **all query understanding stages (`enable_intent`, `enable_ner`, `enable_expansion`) remain disabled by default** in `QueryPipeline` and the CLI.
- Users and downstream pipelines can explicitly opt in using `--enable-intent`, `--enable-ner`, `--apply-filters`, and `--enable-expansion`.
