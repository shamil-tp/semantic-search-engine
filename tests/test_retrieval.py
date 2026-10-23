"""Unit and integration tests for Phase 2 Lexical Search and Evaluation."""

import json
from pathlib import Path
import pytest

from src.evaluation.evaluator import BenchmarkEvaluator
from src.evaluation.metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.ingestion.schemas import DocumentChunk
from src.retrieval.lexical import LexicalSearchEngine, preprocess_text


# ---------------------------------------------------------------------------
# 1. Metrics Tests
# ---------------------------------------------------------------------------

def test_recall_at_k():
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c2", "c4", "c5"}

    # Top 1: ['c1'] -> hits=0/3 = 0.0
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(0.0)
    # Top 2: ['c1', 'c2'] -> hits=1/3 = 1/3
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3)
    # Top 4: ['c1', 'c2', 'c3', 'c4'] -> hits=2/3
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)
    # Empty relevant
    assert recall_at_k(retrieved, set(), k=5) == 0.0
    # k <= 0
    assert recall_at_k(retrieved, relevant, k=0) == 0.0


def test_reciprocal_rank_and_mrr():
    # First relevant at rank 1 -> RR = 1.0
    assert reciprocal_rank(["c1", "c2"], {"c1"}) == 1.0
    # First relevant at rank 2 -> RR = 0.5
    assert reciprocal_rank(["c1", "c2"], {"c2"}) == 0.5
    # First relevant at rank 3 -> RR = 1/3
    assert reciprocal_rank(["c1", "c2", "c3"], {"c3"}) == pytest.approx(1 / 3)
    # No relevant retrieved -> RR = 0.0
    assert reciprocal_rank(["c1", "c2"], {"c3"}) == 0.0

    # MRR across 2 queries: [1.0, 0.5] -> 0.75
    all_ret = [["c1", "c2"], ["c1", "c2"]]
    all_rel = [{"c1"}, {"c2"}]
    assert mean_reciprocal_rank(all_ret, all_rel) == pytest.approx(0.75)


def test_ndcg_at_k_binary_and_graded():
    import math

    # Binary: relevant are 'c1', 'c2'
    retrieved = ["c1", "c2", "c3"]
    relevant_set = {"c1", "c2"}
    # Perfect ranking: IDCG = DCG = (1/log2(2) + 1/log2(3)) -> nDCG = 1.0
    assert ndcg_at_k(retrieved, relevant_set, k=2) == pytest.approx(1.0)

    # Suboptimal ranking: retrieved has irrelevant item first
    # Retrieved at rank 1: c3 (irrel -> 0), rank 2: c1 (rel -> 1)
    retrieved_sub = ["c3", "c1"]
    # DCG@2 = 0 / log2(2) + 1 / log2(3) = 1 / log2(3)
    # IDCG@2: ideal ranking has 2 relevant items: 1/log2(2) + 1/log2(3)
    expected_dcg = 1.0 / math.log2(3)
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    expected_ndcg = expected_dcg / expected_idcg
    assert ndcg_at_k(retrieved_sub, relevant_set, k=2) == pytest.approx(expected_ndcg)

    # Empty judgments
    assert ndcg_at_k(retrieved, set(), k=3) == 0.0


# ---------------------------------------------------------------------------
# 2. Retrieval Engine Tests
# ---------------------------------------------------------------------------

def test_lexical_search_ranking():
    chunks = [
        DocumentChunk(
            chunk_id="chunk_python",
            document_id="doc_py",
            source_path="/path/py.txt",
            chunk_index=0,
            text="Python is a high-level programming language known for readable syntax.",
            start_char=0,
            end_char=70,
            metadata={"topic": "python"},
        ),
        DocumentChunk(
            chunk_id="chunk_rust",
            document_id="doc_rust",
            source_path="/path/rust.txt",
            chunk_index=0,
            text="Rust is a systems programming language focusing on memory safety and speed.",
            start_char=0,
            end_char=75,
            metadata={"topic": "rust"},
        ),
    ]

    engine = LexicalSearchEngine()
    engine.build_index(chunks)

    # Search for python
    results_py = engine.search("readable python language", top_k=2)
    assert len(results_py) >= 1
    assert results_py[0].chunk_id == "chunk_python"
    assert results_py[0].document_id == "doc_py"
    assert results_py[0].score > 0.0
    assert results_py[0].metadata["topic"] == "python"

    # Search for rust
    results_rust = engine.search("memory safety in rust", top_k=2)
    assert len(results_rust) >= 1
    assert results_rust[0].chunk_id == "chunk_rust"


def test_lexical_search_empty_queries_and_vocab():
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            source_path="/path/1.txt",
            chunk_index=0,
            text="Machine learning models learn patterns from data.",
            start_char=0,
            end_char=49,
        )
    ]
    engine = LexicalSearchEngine()
    engine.build_index(chunks)

    # Empty string or whitespace
    assert engine.search("") == []
    assert engine.search("   ") == []

    # Query with no shared vocabulary
    assert engine.search("zebra dinosaur quantum") == []


def test_empty_corpus():
    engine = LexicalSearchEngine()
    engine.build_index([])
    assert engine.search("test query") == []


def test_deterministic_tie_breaking():
    # Two chunks with identical text (equal score)
    chunks = [
        DocumentChunk(
            chunk_id="c_beta",
            document_id="doc_beta",
            source_path="/path/beta.txt",
            chunk_index=0,
            text="Identical content for testing tie-break.",
            start_char=0,
            end_char=40,
        ),
        DocumentChunk(
            chunk_id="c_alpha",
            document_id="doc_alpha",
            source_path="/path/alpha.txt",
            chunk_index=0,
            text="Identical content for testing tie-break.",
            start_char=0,
            end_char=40,
        ),
    ]
    engine = LexicalSearchEngine()
    engine.build_index(chunks)

    results = engine.search("testing tie-break", top_k=2)
    assert len(results) == 2
    assert results[0].score == results[1].score
    # doc_alpha must precede doc_beta alphabetically
    assert results[0].document_id == "doc_alpha"
    assert results[1].document_id == "doc_beta"


# ---------------------------------------------------------------------------
# 3. Persistence and Loading Tests
# ---------------------------------------------------------------------------

def test_index_save_and_reload(tmp_path: Path):
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            source_path="/path/d1.txt",
            chunk_index=0,
            text="Information retrieval evaluation metrics.",
            start_char=0,
            end_char=41,
        )
    ]
    index_dir = tmp_path / "lexical_index"
    engine = LexicalSearchEngine()
    engine.build_index(chunks)
    engine.save(index_dir)

    # Reload
    loaded_engine = LexicalSearchEngine.load(index_dir)
    res_orig = engine.search("information retrieval", top_k=1)
    res_loaded = loaded_engine.search("information retrieval", top_k=1)

    assert len(res_orig) == 1
    assert len(res_loaded) == 1
    assert res_orig[0].chunk_id == res_loaded[0].chunk_id
    assert res_orig[0].score == pytest.approx(res_loaded[0].score)


def test_index_corrupt_or_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LexicalSearchEngine.load(tmp_path / "non_existent")

    # Incomplete index directory
    partial = tmp_path / "partial_index"
    partial.mkdir()
    (partial / "index_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Missing required index artifact"):
        LexicalSearchEngine.load(partial)


# ---------------------------------------------------------------------------
# 4. Evaluator Integration Tests
# ---------------------------------------------------------------------------

def test_benchmark_evaluator():
    chunks = [
        DocumentChunk(
            chunk_id="c_search",
            document_id="doc_search",
            source_path="/path/search.txt",
            chunk_index=0,
            text="Vector embeddings for semantic search.",
            start_char=0,
            end_char=38,
        ),
        DocumentChunk(
            chunk_id="c_clean",
            document_id="doc_clean",
            source_path="/path/clean.txt",
            chunk_index=0,
            text="Data preprocessing and text cleaning.",
            start_char=0,
            end_char=37,
        ),
    ]
    engine = LexicalSearchEngine()
    engine.build_index(chunks)

    evaluator = BenchmarkEvaluator(engine=engine, k_values=[1, 5], target_unit="document_id")

    queries = [
        {
            "query_id": "q1",
            "query": "vector embeddings search",
            "relevant_ids": ["doc_search"],
        },
        {
            "query_id": "q2",
            "query": "preprocessing cleaning",
            "relevant_ids": ["doc_clean"],
        },
        # Query with missing judgments should be skipped cleanly
        {
            "query_id": "q3_skipped",
            "query": "some query",
            "relevant_ids": [],
        },
    ]

    report = evaluator.evaluate_queries(queries)
    assert report["evaluation_metadata"]["valid_queries_evaluated"] == 2
    assert len(report["evaluation_metadata"]["skipped_queries"]) == 1
    assert report["aggregate_metrics"]["mrr"] == 1.0
    assert report["aggregate_metrics"]["recall@1"] == 1.0
    assert report["aggregate_metrics"]["ndcg@1"] == 1.0
