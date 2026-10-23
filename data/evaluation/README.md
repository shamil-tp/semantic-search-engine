# Synthetic Evaluation Dataset (Phase 2 Baseline)

## Overview
This synthetic evaluation dataset provides ground-truth relevance pairs to measure lexical and semantic search performance against `data/sample_corpus/`.

- **Dataset Identifier**: `synthetic_eval_v1`
- **Corpus**: `data/sample_corpus/`
- **Total Queries**: 4 queries
- **Validation Status**: **Synthetic Demonstration Data**. Relevance judgments are crafted explicitly based on known content in the sample documents for deterministic regression testing and baseline profiling, not gathered from human users.

## Relevance Mapping
- `a3090451a0a256d7`: `doc1_semantic_search.md` (Covers semantic search concepts, contextual understanding, vector embeddings, hybrid retrieval)
- `80befa4c7370c0c2`: `doc2_preprocessing.txt` (Covers text preprocessing, NFC Unicode normalization, deduplication hashes, overlapping chunks)

## Target Unit
Evaluation queries in `synthetic_queries.json` map query intent to `document_id`. The benchmark evaluator aggregates chunk hits to document level to assess whether the target document is retrieved in top $k$.
