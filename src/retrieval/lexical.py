"""Lexical retrieval baseline using scikit-learn TfidfVectorizer."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ingestion.cleaner import normalize_text
from src.ingestion.schemas import DocumentChunk
from src.retrieval.index import load_lexical_index, save_lexical_index
from src.retrieval.schemas import SearchResult

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """
    Apply shared text normalization for documents and queries.
    Preserves numbers, technical punctuation, and tokens without aggressive stemming.
    """
    if not text:
        return ""
    # Use Phase 1 standard normalization (NFC, standardized whitespace and lines)
    return normalize_text(text).lower()


class LexicalSearchEngine:
    """
    TF-IDF Lexical Search Engine.
    Uses scikit-learn's TfidfVectorizer with sparse matrix cosine similarity scoring.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: Optional[int] = None,
        sublinear_tf: bool = True,
    ):
        self.config = {
            "ngram_range": list(ngram_range),
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
            "sublinear_tf": sublinear_tf,
        }
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            token_pattern=r"(?u)\b\w[\w\.-]*\w\b|\b\w+\b",  # Keeps identifiers like v1.0, 3.14, word-name
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            norm="l2",
        )
        self.tfidf_matrix: Optional[sp.csr_matrix] = None
        self.chunks: List[DocumentChunk] = []
        self._is_fitted = False

    def build_index(self, chunks: Sequence[DocumentChunk]) -> "LexicalSearchEngine":
        """
        Fit vectorizer on chunks and build sparse TF-IDF matrix.
        """
        self.chunks = list(chunks)
        if not self.chunks:
            logger.warning("Empty chunk list provided to build_index.")
            self.tfidf_matrix = sp.csr_matrix((0, 0))
            self._is_fitted = True
            return self

        corpus_texts = [c.text for c in self.chunks]
        try:
            matrix = self.vectorizer.fit_transform(corpus_texts)
            self.tfidf_matrix = matrix.tocsr()
            self._is_fitted = True
            logger.info(
                f"Built index over {len(self.chunks)} chunks. "
                f"Vocabulary size: {len(self.vectorizer.vocabulary_)}, "
                f"Matrix shape: {self.tfidf_matrix.shape}"
            )
        except ValueError as exc:
            # Handles empty vocabulary e.g. all empty docs or stop-words
            logger.warning(f"Failed to fit TfidfVectorizer: {exc}")
            self.tfidf_matrix = sp.csr_matrix((len(self.chunks), 0))
            self._is_fitted = True

        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        max_excerpt_len: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Retrieve and rank top_k chunks for the given query.
        
        Deterministic tie-breaking:
        1. Higher score first (-score)
        2. Lower document_id alphabetically
        3. Lower chunk_index numerically
        """
        if not query or not query.strip():
            return []

        if not self._is_fitted or self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            return []

        if self.tfidf_matrix.shape[1] == 0:
            # Empty vocabulary
            return []

        # Vectorize query (norm='l2' is applied by TfidfVectorizer)
        query_vec = self.vectorizer.transform([query])
        if query_vec.nnz == 0:
            # Query has no overlapping vocabulary
            return []

        # Cosine similarity between query and all document chunk vectors: (1, V) dot (N, V).T -> (1, N)
        scores = self.tfidf_matrix.dot(query_vec.T).toarray().ravel()

        # Filter positive or above min_score matches
        candidate_indices = np.where(scores > min_score)[0]
        if len(candidate_indices) == 0:
            return []

        # Deterministic sorting
        # Sort key: (-score, doc_id, chunk_index)
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
        """Persist index to directory."""
        if not self._is_fitted or self.tfidf_matrix is None:
            raise ValueError("Cannot save an unfitted index.")
        save_lexical_index(
            index_dir=index_dir,
            vectorizer=self.vectorizer,
            tfidf_matrix=self.tfidf_matrix,
            chunks=self.chunks,
            config=self.config,
        )

    @classmethod
    def load(cls, index_dir: str | Path) -> "LexicalSearchEngine":
        """Load index from directory."""
        vec, matrix, chunks, meta = load_lexical_index(index_dir)
        cfg = meta.get("config", {})
        engine = cls(
            ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
            min_df=cfg.get("min_df", 1),
            max_df=cfg.get("max_df", 1.0),
            max_features=cfg.get("max_features"),
            sublinear_tf=cfg.get("sublinear_tf", True),
        )
        engine.vectorizer = vec
        engine.tfidf_matrix = matrix
        engine.chunks = chunks
        engine._is_fitted = True
        return engine
