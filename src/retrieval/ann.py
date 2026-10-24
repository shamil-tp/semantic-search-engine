"""Persisted Vector retrieval using scikit-learn NearestNeighbors.

Replaces external vector databases / hnswlib with a reproducible, lightweight,
in-memory scikit-learn NearestNeighbors model using cosine distance.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.ingestion.schemas import DocumentChunk
from src.retrieval.dense import DEFAULT_DENSE_MODEL, DenseSearchEngine
from src.retrieval.schemas import SearchResult

VECTOR_EMBEDDINGS_FILE = "embeddings.npy"
VECTOR_CATALOG_FILE = "chunks.jsonl"
VECTOR_METADATA_FILE = "vector_metadata.json"
VECTOR_FORMAT_VERSION = 1
VECTOR_METRIC = "cosine"


def _fingerprint(chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> str:
    """Return a stable fingerprint binding a vector index artifact to its dense corpus."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(embeddings, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _catalog_fingerprint(chunks: Sequence[DocumentChunk]) -> str:
    """Fingerprint the ordered catalog used to translate indices to chunks."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(json.dumps(chunk.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class VectorSearchEngine:
    """Vector search over normalized dense embeddings using scikit-learn NearestNeighbors."""

    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        device: Optional[str] = None,
        algorithm: str = "brute",
    ) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        self.algorithm = algorithm
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dim: Optional[int] = None
        self.corpus_fingerprint: Optional[str] = None
        self._nn: Optional[NearestNeighbors] = None
        self._query_encoder = DenseSearchEngine(model_name=model_name, device=self.device)

    def build(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> "VectorSearchEngine":
        """Build a NearestNeighbors index from normalized dense embeddings."""
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional matrix.")
        if vectors.shape[0] != len(chunks):
            raise ValueError("Embedding row count must match chunk count.")
        if not len(chunks):
            raise ValueError("Cannot build an index for an empty corpus.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values.")

        self.chunks = list(chunks)
        self.embeddings = vectors
        self.embedding_dim = int(vectors.shape[1])

        # Fit scikit-learn NearestNeighbors with cosine metric
        nn = NearestNeighbors(
            n_neighbors=min(10, len(self.chunks)),
            algorithm=self.algorithm,
            metric=VECTOR_METRIC,
        )
        nn.fit(vectors)
        self._nn = nn
        self.corpus_fingerprint = _fingerprint(self.chunks, vectors)
        return self

    @classmethod
    def from_dense(cls, dense: DenseSearchEngine, **kwargs) -> "VectorSearchEngine":
        """Build vector search engine from a fitted exact dense index without re-embedding."""
        if dense.embeddings is None or not dense._is_indexed:
            raise ValueError("A fitted dense index is required to build vector retrieval.")
        return cls(model_name=dense.model_name, device=dense.device, **kwargs).build(
            dense.chunks, dense.embeddings
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        max_excerpt_len: Optional[int] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Return cosine-ranked results for a non-empty query with optional metadata filtering."""
        if not query or not query.strip() or self._nn is None or not self.chunks or self.embeddings is None:
            return []
        if top_k < 1:
            return []

        self._query_encoder.embedding_dim = self.embedding_dim
        query_vector = self._query_encoder.encode_query(query)
        if not len(query_vector) or np.all(query_vector == 0):
            return []

        query_2d = query_vector.reshape(1, -1)

        # If metadata filters are active, query more or filter candidates
        if metadata_filters:
            # Match filtered indices
            matching_indices = set()
            for idx, chunk in enumerate(self.chunks):
                match = True
                for filter_k, filter_v in metadata_filters.items():
                    if hasattr(chunk, filter_k):
                        if getattr(chunk, filter_k) != filter_v:
                            match = False
                            break
                    elif filter_k in chunk.metadata:
                        if chunk.metadata[filter_k] != filter_v:
                            match = False
                            break
                    else:
                        match = False
                        break
                if match:
                    matching_indices.add(idx)

            if not matching_indices:
                return []

            # Search across all candidates to filter
            k_search = len(self.chunks)
            distances, indices = self._nn.kneighbors(query_2d, n_neighbors=k_search)
            candidate_pairs = [
                (idx, dist)
                for idx, dist in zip(indices[0], distances[0])
                if idx in matching_indices
            ][:top_k]
        else:
            k_search = min(top_k, len(self.chunks))
            distances, indices = self._nn.kneighbors(query_2d, n_neighbors=k_search)
            candidate_pairs = list(zip(indices[0], distances[0]))

        # Sort with deterministic tie-breaking:
        # Cosine similarity = 1.0 - cosine distance
        # Sort key: (-score, doc_id, chunk_index)
        scored_candidates = []
        for idx, dist in candidate_pairs:
            sim = 1.0 - float(dist)
            scored_candidates.append((idx, sim))

        scored_candidates.sort(
            key=lambda item: (
                -float(item[1]),
                self.chunks[item[0]].document_id,
                self.chunks[item[0]].chunk_index,
            )
        )

        results: List[SearchResult] = []
        for rank, (idx, score) in enumerate(scored_candidates[:top_k], start=1):
            chunk = self.chunks[idx]
            text = chunk.text
            if max_excerpt_len and len(text) > max_excerpt_len:
                text = text[:max_excerpt_len] + "..."
            results.append(
                SearchResult(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(round(score, 6)),
                    text=text,
                    source_path=chunk.source_path,
                    chunk_index=chunk.chunk_index,
                    metadata=dict(chunk.metadata),
                )
            )
        return results

    def save(self, index_dir: str | Path) -> None:
        """Persist embeddings, chunk catalog, and metadata safely without pickle."""
        if self.embeddings is None or self.embedding_dim is None or self.corpus_fingerprint is None:
            raise ValueError("Cannot save an unfitted vector index.")
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings safely as numpy binary
        np.save(path / VECTOR_EMBEDDINGS_FILE, self.embeddings)

        # Save chunk catalog as JSONL
        with (path / VECTOR_CATALOG_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        # Save metadata manifest
        metadata = {
            "format_version": VECTOR_FORMAT_VERSION,
            "index_type": "sklearn_cosine",
            "similarity_metric": VECTOR_METRIC,
            "library": {"name": "scikit-learn"},
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "chunk_count": len(self.chunks),
            "corpus_fingerprint": self.corpus_fingerprint,
            "catalog_fingerprint": _catalog_fingerprint(self.chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {"algorithm": self.algorithm},
        }
        with (path / VECTOR_METADATA_FILE).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @classmethod
    def load(cls, index_dir: str | Path, device: Optional[str] = None) -> "VectorSearchEngine":
        """Load persisted embeddings and chunk catalog, recreating the NearestNeighbors index."""
        path = Path(index_dir)
        required = [
            path / VECTOR_EMBEDDINGS_FILE,
            path / VECTOR_CATALOG_FILE,
            path / VECTOR_METADATA_FILE,
        ]
        if not path.is_dir():
            raise FileNotFoundError(f"Vector index directory does not exist: {path}")
        for artifact in required:
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing required vector index artifact: {artifact.name}")

        try:
            metadata = json.loads((path / VECTOR_METADATA_FILE).read_text(encoding="utf-8"))
            if (
                metadata.get("format_version") != VECTOR_FORMAT_VERSION
                or metadata.get("index_type") != "sklearn_cosine"
                or metadata.get("similarity_metric") != VECTOR_METRIC
            ):
                raise ValueError("Unsupported vector index metadata format.")

            chunks = [
                DocumentChunk(**json.loads(line))
                for line in (path / VECTOR_CATALOG_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(chunks) != metadata.get("chunk_count") or not chunks:
                raise ValueError("Vector chunk catalog is invalid or empty.")
            if metadata.get("catalog_fingerprint") != _catalog_fingerprint(chunks):
                raise ValueError("Vector chunk catalog does not match its manifest.")

            embeddings = np.load(path / VECTOR_EMBEDDINGS_FILE)
            if embeddings.shape[0] != len(chunks):
                raise ValueError("Embedding count does not match chunk count.")

            config = metadata.get("config", {})
            engine = cls(
                model_name=metadata["model_name"],
                device=device,
                algorithm=config.get("algorithm", "brute"),
            )
            engine.build(chunks, embeddings)
            return engine
        except Exception as error:
            raise ValueError(f"Invalid or corrupt vector index at {path}: {error}") from error


# Alias for backward compatibility if referenced elsewhere
HnswSearchEngine = VectorSearchEngine
