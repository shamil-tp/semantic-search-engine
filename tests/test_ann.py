"""Unit and persistence tests for scikit-learn VectorSearchEngine."""

from pathlib import Path
import numpy as np
import pytest

from src.ingestion.schemas import DocumentChunk
from src.retrieval.ann import VectorSearchEngine


def _sample_chunks():
    return [
        DocumentChunk(
            chunk_id="c0",
            document_id="doc_a",
            source_path="/path/a.md",
            chunk_index=0,
            text="First document about machine learning algorithms.",
            start_char=0,
            end_char=46,
            metadata={"extension": ".md"},
        ),
        DocumentChunk(
            chunk_id="c1",
            document_id="doc_b",
            source_path="/path/b.txt",
            chunk_index=0,
            text="Second document about database indexing.",
            start_char=0,
            end_char=40,
            metadata={"extension": ".txt"},
        ),
    ]


def test_vector_search_engine_build_and_search(monkeypatch):
    chunks = _sample_chunks()
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    engine = VectorSearchEngine(model_name="test-model").build(chunks, embeddings)

    # Mock query encoding to return [1.0, 0.0, 0.0]
    monkeypatch.setattr(
        engine._query_encoder,
        "encode_query",
        lambda q: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    results = engine.search("machine learning", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c0"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].chunk_id == "c1"
    assert results[1].score == pytest.approx(0.0, abs=1e-5)


def test_vector_search_metadata_filter(monkeypatch):
    chunks = _sample_chunks()
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    engine = VectorSearchEngine().build(chunks, embeddings)
    monkeypatch.setattr(
        engine._query_encoder,
        "encode_query",
        lambda q: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    results = engine.search("query", top_k=2, metadata_filters={"extension": ".txt"})
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_vector_search_save_and_reload(tmp_path: Path):
    chunks = _sample_chunks()
    embeddings = np.array(
        [
            [0.6, 0.8, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    engine = VectorSearchEngine(model_name="test-model").build(chunks, embeddings)
    engine.save(tmp_path)

    loaded = VectorSearchEngine.load(tmp_path)
    assert len(loaded.chunks) == 2
    assert loaded.embedding_dim == 3
    assert np.allclose(loaded.embeddings, embeddings)
    assert loaded.corpus_fingerprint == engine.corpus_fingerprint


def test_vector_search_invalid_inputs():
    chunks = _sample_chunks()
    with pytest.raises(ValueError, match="Cannot build an index for an empty corpus"):
        VectorSearchEngine().build([], np.zeros((0, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="must contain only finite values"):
        VectorSearchEngine().build(chunks, np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32))
