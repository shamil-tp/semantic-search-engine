"""Persisted HNSW approximate-nearest-neighbour retrieval for dense embeddings.

The exact :class:`DenseSearchEngine` remains the evaluation oracle.  This
module intentionally accepts its already-normalized embeddings rather than
re-encoding documents, so an ANN artifact is tied to a specific dense index.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.ingestion.schemas import DocumentChunk
from src.retrieval.dense import DEFAULT_DENSE_MODEL, DenseSearchEngine
from src.retrieval.schemas import SearchResult

ANN_INDEX_FILE = "hnsw_index.bin"
ANN_CATALOG_FILE = "chunks.jsonl"
ANN_METADATA_FILE = "ann_metadata.json"
ANN_FORMAT_VERSION = 1
ANN_METRIC = "cosine"


def _require_hnswlib():
    """Import the optional backend only when ANN retrieval is used."""
    try:
        import hnswlib
    except ImportError as error:
        raise RuntimeError(
            "HNSW retrieval requires the optional 'hnswlib' dependency. "
            "Install the project dependencies, then rebuild the ANN index."
        ) from error
    return hnswlib


def _fingerprint(chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> str:
    """Return a stable fingerprint binding an ANN artifact to its dense corpus."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(embeddings, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _catalog_fingerprint(chunks: Sequence[DocumentChunk]) -> str:
    """Fingerprint the ordered catalog used to translate ANN labels to chunks."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(json.dumps(chunk.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class HnswSearchEngine:
    """Approximate cosine search over normalized vectors using HNSW.

    ``ef_search`` is deliberately configurable: larger values generally
    improve recall at the cost of latency.  Measure it against exact search
    before choosing production defaults for a corpus.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        device: Optional[str] = None,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
    ) -> None:
        if m < 2 or ef_construction < 2 or ef_search < 1:
            raise ValueError("m and ef_construction must be at least 2; ef_search must be positive.")
        self.model_name = model_name
        self.device = device or "cpu"
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.chunks: List[DocumentChunk] = []
        self.embedding_dim: Optional[int] = None
        self.corpus_fingerprint: Optional[str] = None
        self._index = None
        self._query_encoder = DenseSearchEngine(model_name=model_name, device=self.device)

    def build(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> "HnswSearchEngine":
        """Build an HNSW index from normalized embeddings produced by dense retrieval."""
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional matrix.")
        if vectors.shape[0] != len(chunks):
            raise ValueError("Embedding row count must match chunk count.")
        if not len(chunks):
            raise ValueError("Cannot build an ANN index for an empty corpus.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values.")

        hnswlib = _require_hnswlib()
        self.chunks = list(chunks)
        self.embedding_dim = int(vectors.shape[1])
        index = hnswlib.Index(space=ANN_METRIC, dim=self.embedding_dim)
        index.init_index(
            max_elements=len(self.chunks),
            ef_construction=self.ef_construction,
            M=self.m,
        )
        index.add_items(vectors, np.arange(len(self.chunks), dtype=np.int64))
        index.set_ef(self.ef_search)
        self._index = index
        self.corpus_fingerprint = _fingerprint(self.chunks, vectors)
        return self

    @classmethod
    def from_dense(cls, dense: DenseSearchEngine, **kwargs) -> "HnswSearchEngine":
        """Build ANN from a fitted exact dense index without re-embedding data."""
        if dense.embeddings is None or not dense._is_indexed:
            raise ValueError("A fitted dense index is required to build ANN retrieval.")
        return cls(model_name=dense.model_name, device=dense.device, **kwargs).build(dense.chunks, dense.embeddings)

    def search(self, query: str, top_k: int = 10, max_excerpt_len: Optional[int] = None) -> List[SearchResult]:
        """Return approximate cosine-ranked results for a non-empty query."""
        if not query or not query.strip() or self._index is None or not self.chunks:
            return []
        if top_k < 1:
            return []

        self._query_encoder.embedding_dim = self.embedding_dim
        query_vector = self._query_encoder.encode_query(query)
        if not len(query_vector) or np.all(query_vector == 0):
            return []
        requested = min(top_k, len(self.chunks))
        labels, distances = self._index.knn_query(query_vector, k=requested)

        results: List[SearchResult] = []
        for rank, (label, distance) in enumerate(zip(labels[0], distances[0]), start=1):
            chunk = self.chunks[int(label)]
            text = chunk.text
            if max_excerpt_len and len(text) > max_excerpt_len:
                text = text[:max_excerpt_len] + "..."
            results.append(
                SearchResult(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(round(1.0 - float(distance), 6)),
                    text=text,
                    source_path=chunk.source_path,
                    chunk_index=chunk.chunk_index,
                    metadata=dict(chunk.metadata),
                )
            )
        return results

    def save(self, index_dir: str | Path) -> None:
        """Persist the index, public chunk catalog, and compatibility metadata."""
        if self._index is None or self.embedding_dim is None or self.corpus_fingerprint is None:
            raise ValueError("Cannot save an unfitted ANN index.")
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(path / ANN_INDEX_FILE))
        with (path / ANN_CATALOG_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        metadata = {
            "format_version": ANN_FORMAT_VERSION,
            "index_type": "hnsw_cosine",
            "similarity_metric": ANN_METRIC,
            "library": {"name": "hnswlib", "version": importlib.metadata.version("hnswlib")},
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "chunk_count": len(self.chunks),
            "corpus_fingerprint": self.corpus_fingerprint,
            "catalog_fingerprint": _catalog_fingerprint(self.chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {"m": self.m, "ef_construction": self.ef_construction, "ef_search": self.ef_search},
        }
        with (path / ANN_METADATA_FILE).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @classmethod
    def load(cls, index_dir: str | Path, device: Optional[str] = None) -> "HnswSearchEngine":
        """Load a validated ANN artifact; rebuild if any validation fails."""
        path = Path(index_dir)
        required = [path / ANN_INDEX_FILE, path / ANN_CATALOG_FILE, path / ANN_METADATA_FILE]
        if not path.is_dir():
            raise FileNotFoundError(f"ANN index directory does not exist: {path}")
        for artifact in required:
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing required ANN index artifact: {artifact.name}; rebuild the index.")
        try:
            metadata = json.loads((path / ANN_METADATA_FILE).read_text(encoding="utf-8"))
            if (
                metadata.get("format_version") != ANN_FORMAT_VERSION
                or metadata.get("index_type") != "hnsw_cosine"
                or metadata.get("similarity_metric") != ANN_METRIC
                or metadata.get("library", {}).get("name") != "hnswlib"
            ):
                raise ValueError("Unsupported ANN index metadata; rebuild the index.")
            chunks = [DocumentChunk(**json.loads(line)) for line in (path / ANN_CATALOG_FILE).read_text(encoding="utf-8").splitlines() if line]
            if len(chunks) != metadata.get("chunk_count") or not chunks:
                raise ValueError("ANN chunk catalog is invalid; rebuild the index.")
            if metadata.get("catalog_fingerprint") != _catalog_fingerprint(chunks):
                raise ValueError("ANN chunk catalog does not match its manifest; rebuild the index.")
            config = metadata.get("config", {})
            engine = cls(
                model_name=metadata["model_name"], device=device,
                m=int(config["m"]), ef_construction=int(config["ef_construction"]), ef_search=int(config["ef_search"]),
            )
            hnswlib = _require_hnswlib()
            index = hnswlib.Index(space=ANN_METRIC, dim=int(metadata["embedding_dim"]))
            index.load_index(str(path / ANN_INDEX_FILE), max_elements=len(chunks))
            if index.get_current_count() != len(chunks):
                raise ValueError("ANN item count does not match chunk catalog; rebuild the index.")
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Invalid or corrupt ANN index at {path}; rebuild the index.") from error

        engine.chunks = chunks
        engine.embedding_dim = int(metadata["embedding_dim"])
        engine.corpus_fingerprint = str(metadata["corpus_fingerprint"])
        engine._index = index
        engine._index.set_ef(engine.ef_search)
        return engine
