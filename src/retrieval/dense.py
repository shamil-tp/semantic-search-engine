"""Dense vector retrieval using compact pretrained Sentence-Transformers bi-encoder."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.ingestion.cleaner import normalize_text
from src.ingestion.schemas import DocumentChunk
from src.retrieval.schemas import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_DENSE_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_CATALOG_FILE = "chunks.jsonl"
DENSE_METADATA_FILE = "dense_metadata.json"


def preprocess_dense_text(text: str) -> str:
    """Shared preprocessing for documents and queries before dense embedding."""
    if not text:
        return ""
    return normalize_text(text)


class DenseSearchEngine:
    """
    Dense Semantic Search Engine using bi-encoder sentence representations.
    Computes exact dot-product similarity over L2-normalized embeddings (cosine similarity).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device or "cpu"
        self.batch_size = batch_size
        self._model = None

        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, D), float32, normalized
        self.embedding_dim: Optional[int] = None
        self._is_indexed = False

    def _get_model(self):
        """Lazy load sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SentenceTransformer model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def build_index(self, chunks: Sequence[DocumentChunk]) -> "DenseSearchEngine":
        """
        Encode all document chunks into normalized embeddings and store in memory.
        """
        self.chunks = list(chunks)
        if not self.chunks:
            logger.warning("Empty chunk list provided to build_index.")
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            self.embedding_dim = 0
            self._is_indexed = True
            return self

        model = self._get_model()
        texts = [preprocess_dense_text(c.text) for c in self.chunks]

        logger.info(f"Encoding {len(texts)} chunks with {self.model_name} (batch_size={self.batch_size})...")
        encoded = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalize so dot product equals cosine similarity
            convert_to_numpy=True,
        )

        self.embeddings = encoded.astype(np.float32)
        self.embedding_dim = self.embeddings.shape[1]
        self._is_indexed = True
        logger.info(f"Built dense index: {self.embeddings.shape} (dim={self.embedding_dim})")
        return self

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into an L2-normalized embedding vector."""
        clean_q = preprocess_dense_text(query)
        if not clean_q:
            return np.zeros(self.embedding_dim or 0, dtype=np.float32)

        model = self._get_model()
        q_emb = model.encode(
            [clean_q],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return q_emb.astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = -1.0,
        max_excerpt_len: Optional[int] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Retrieve and rank top_k chunks using exact cosine similarity over normalized embeddings.
        
        Deterministic tie-breaking:
        1. Higher score first (-score)
        2. Lower document_id alphabetically
        3. Lower chunk_index numerically
        """
        if not query or not query.strip():
            return []

        if not self._is_indexed or self.embeddings is None or len(self.chunks) == 0:
            return []

        q_vec = self.encode_query(query)
        if np.all(q_vec == 0):
            return []

        # Cosine similarity: (N, D) dot (D,) -> (N,)
        scores = np.dot(self.embeddings, q_vec)

        # Candidate indices matching score threshold
        candidate_indices = np.where(scores >= min_score)[0]
        if len(candidate_indices) == 0:
            return []

        # Apply metadata filters if provided
        if metadata_filters:
            filtered_indices = []
            for idx in candidate_indices:
                chunk = self.chunks[idx]
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
                    filtered_indices.append(idx)
            candidate_indices = np.array(filtered_indices, dtype=int)
            if len(candidate_indices) == 0:
                return []

        # Deterministic sorting
        sorted_indices = sorted(
            candidate_indices,
            key=lambda idx: (
                -float(scores[idx]),
                self.chunks[idx].document_id,
                self.chunks[idx].chunk_index,
            ),
        )

        top_indices = sorted_indices[:top_k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            text = chunk.text
            if max_excerpt_len and len(text) > max_excerpt_len:
                text = text[:max_excerpt_len] + "..."

            results.append(
                SearchResult(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(round(float(scores[idx]), 6)),
                    text=text,
                    source_path=chunk.source_path,
                    chunk_index=chunk.chunk_index,
                    metadata=dict(chunk.metadata),
                )
            )

        return results

    def save(self, index_dir: str | Path) -> None:
        """Persist dense embeddings, chunk records, and model metadata."""
        if not self._is_indexed or self.embeddings is None:
            raise ValueError("Cannot save an unfitted dense index.")

        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Save chunks catalog
        chunks_path = path / CHUNKS_CATALOG_FILE
        with open(chunks_path, "w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")

        # 2. Save embeddings matrix as binary numpy
        emb_path = path / EMBEDDINGS_FILE
        np.save(emb_path, self.embeddings)

        # 3. Save metadata
        meta = {
            "index_type": "dense_bi_encoder",
            "model_name": self.model_name,
            "embedding_dim": int(self.embedding_dim or 0),
            "chunk_count": len(self.chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "device": self.device,
            },
        }
        with open(path / DENSE_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved dense index ({len(self.chunks)} chunks, dim={self.embedding_dim}) to {path.resolve()}")

    @classmethod
    def load(cls, index_dir: str | Path, device: Optional[str] = None) -> "DenseSearchEngine":
        """Load persisted dense index artifacts."""
        path = Path(index_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Dense index directory does not exist: {path}")

        meta_path = path / DENSE_METADATA_FILE
        chunks_path = path / CHUNKS_CATALOG_FILE
        emb_path = path / EMBEDDINGS_FILE

        for req in [meta_path, chunks_path, emb_path]:
            if not req.exists():
                raise FileNotFoundError(f"Missing required dense index artifact: {req.name}")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        chunks: List[DocumentChunk] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    chunks.append(DocumentChunk(**json.loads(line_str)))

        embeddings = np.load(emb_path)
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding row count {embeddings.shape[0]} does not match chunk count {len(chunks)}"
            )

        model_name = metadata.get("model_name", DEFAULT_DENSE_MODEL)
        engine = cls(model_name=model_name, device=device or "cpu")
        engine.chunks = chunks
        engine.embeddings = embeddings.astype(np.float32)
        engine.embedding_dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
        engine._is_indexed = True

        return engine
