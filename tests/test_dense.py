"""Pytest tests for Phase 4 Dense Semantic Retrieval."""

from pathlib import Path
import numpy as np
import pytest

from src.ingestion.schemas import DocumentChunk
from src.retrieval.dense import DenseSearchEngine, preprocess_dense_text
from src.retrieval.schemas import SearchResult


def test_preprocess_dense_text():
    raw = "  Line with   extra \r\n spaces  and NFC: e\u0301  "
    cleaned = preprocess_dense_text(raw)
    assert "\r" not in cleaned
    assert "é" in cleaned
    assert "Line with extra\nspaces and NFC: é" in cleaned


def test_dense_search_mocked_embeddings(monkeypatch):
    """Test exact vector similarity and ranking logic with deterministic mocked embeddings."""
    dim = 4
    # Chunk 0 embedding: [1, 0, 0, 0]
    # Chunk 1 embedding: [0, 1, 0, 0]
    # Chunk 2 embedding: [0, 0, 1, 0]
    chunk0 = DocumentChunk(
        chunk_id="c0",
        document_id="doc_a",
        source_path="/path/a.md",
        chunk_index=0,
        text="Alpha text on artificial intelligence.",
        start_char=0,
        end_char=38,
        metadata={"extension": ".md"},
    )
    chunk1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc_b",
        source_path="/path/b.txt",
        chunk_index=0,
        text="Beta text on database storage.",
        start_char=0,
        end_char=31,
        metadata={"extension": ".txt"},
    )

    engine = DenseSearchEngine()
    engine.chunks = [chunk0, chunk1]
    engine.embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    engine.embedding_dim = dim
    engine._is_indexed = True

    # Mock query encoding to return vector close to chunk0: [0.9, 0.1, 0, 0] normalized
    def mock_encode_query(query: str):
        if "alpha" in query.lower():
            v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            v = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return v / np.linalg.norm(v)

    monkeypatch.setattr(engine, "encode_query", mock_encode_query)

    # Search for alpha
    results_alpha = engine.search("alpha query", top_k=2)
    assert len(results_alpha) == 2
    assert results_alpha[0].chunk_id == "c0"
    assert results_alpha[0].score == pytest.approx(1.0)
    assert results_alpha[1].score == pytest.approx(0.0)

    # Metadata filter test
    results_filtered = engine.search("alpha query", metadata_filters={"extension": ".txt"})
    assert len(results_filtered) == 1
    assert results_filtered[0].chunk_id == "c1"


def test_dense_search_empty_queries_and_corpus():
    engine = DenseSearchEngine()
    engine.build_index([])
    assert engine.search("query") == []
    assert engine.search("") == []
    assert engine.search("   ") == []


def test_dense_index_save_and_reload(tmp_path: Path):
    chunk = DocumentChunk(
        chunk_id="c_test",
        document_id="doc_test",
        source_path="/path/test.txt",
        chunk_index=0,
        text="Testing dense index persistence.",
        start_char=0,
        end_char=32,
        metadata={"tag": "unit_test"},
    )
    engine = DenseSearchEngine(model_name="mock-model")
    engine.chunks = [chunk]
    engine.embeddings = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    engine.embedding_dim = 4
    engine._is_indexed = True

    save_dir = tmp_path / "dense_index"
    engine.save(save_dir)

    # Reload
    loaded = DenseSearchEngine.load(save_dir)
    assert loaded.embedding_dim == 4
    assert len(loaded.chunks) == 1
    assert loaded.chunks[0].chunk_id == "c_test"
    assert np.allclose(loaded.embeddings, engine.embeddings)
