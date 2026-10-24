"""Phase 5 HNSW persistence and stable-ID tests (requires optional hnswlib)."""

import numpy as np
import pytest

hnswlib = pytest.importorskip("hnswlib", reason="optional Phase 5 ANN backend is not installed")

from src.ingestion.schemas import DocumentChunk
from src.retrieval.ann import HnswSearchEngine


def _chunks():
    return [
        DocumentChunk("c0", "d0", "/a.txt", 0, "alpha", 0, 5),
        DocumentChunk("c1", "d1", "/b.txt", 0, "beta", 0, 4),
    ]


def test_ann_round_trip_preserves_stable_chunk_ids(tmp_path, monkeypatch):
    engine = HnswSearchEngine(model_name="test-model").build(
        _chunks(), np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    monkeypatch.setattr(engine._query_encoder, "encode_query", lambda _: np.array([1.0, 0.0], dtype=np.float32))
    engine.save(tmp_path)
    loaded = HnswSearchEngine.load(tmp_path)
    monkeypatch.setattr(loaded._query_encoder, "encode_query", lambda _: np.array([1.0, 0.0], dtype=np.float32))
    assert [item.chunk_id for item in loaded.search("alpha", top_k=2)] == ["c0", "c1"]


def test_ann_rejects_invalid_vectors():
    with pytest.raises(ValueError, match="finite"):
        HnswSearchEngine().build(_chunks(), np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32))
