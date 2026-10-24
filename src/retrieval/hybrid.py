"""Explicit opt-in hybrid fusion and cross-encoder reranking.

The implementation stays deliberately small: lexical and dense engines produce
bounded candidate lists, reciprocal-rank fusion combines their rank signals,
and an optional cross-encoder reranks only the leading fused candidates.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from src.retrieval.schemas import SearchResult

DEFAULT_RRF_K = 60
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SearchEngine(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = -1.0,
        max_excerpt_len: Optional[int] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]: ...


def reciprocal_rank_fusion(
    result_lists: Iterable[Sequence[SearchResult]],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 10,
) -> List[SearchResult]:
    """Combine ranked lists without assuming their scores share a scale."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive.")
    if top_k < 1:
        return []

    fused: Dict[str, tuple[SearchResult, float]] = {}
    for results in result_lists:
        for result in results:
            contribution = 1.0 / (rrf_k + result.rank)
            existing = fused.get(result.chunk_id)
            if existing is None:
                fused[result.chunk_id] = (result, contribution)
            else:
                fused[result.chunk_id] = (existing[0], existing[1] + contribution)

    ordered = sorted(
        fused.values(),
        key=lambda item: (-item[1], item[0].document_id, item[0].chunk_index),
    )[:top_k]
    return [
        SearchResult(
            rank=rank,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            score=round(score, 6),
            text=result.text,
            source_path=result.source_path,
            chunk_index=result.chunk_index,
            metadata=dict(result.metadata),
        )
        for rank, (result, score) in enumerate(ordered, start=1)
    ]


class CrossEncoderReranker:
    """Lazy cross-encoder reranker, loaded only when explicitly requested."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, candidates: Sequence[SearchResult], top_k: int) -> List[SearchResult]:
        """Score the bounded candidate set and return deterministic reranked results."""
        if not query or not query.strip() or top_k < 1:
            return []
        if not candidates:
            return []
        scores = self._get_model().predict([(query, candidate.text) for candidate in candidates])
        ordered = sorted(
            zip(candidates, scores),
            key=lambda item: (-float(item[1]), item[0].document_id, item[0].chunk_index),
        )[:top_k]
        return [
            SearchResult(
                rank=rank,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                score=round(float(score), 6),
                text=result.text,
                source_path=result.source_path,
                chunk_index=result.chunk_index,
                metadata=dict(result.metadata),
            )
            for rank, (result, score) in enumerate(ordered, start=1)
        ]


class HybridSearchEngine:
    """Fuse lexical and dense candidates; reranking is opt-in and bounded."""

    def __init__(
        self,
        lexical_engine: SearchEngine,
        dense_engine: SearchEngine,
        rrf_k: int = DEFAULT_RRF_K,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_top_n: int = 20,
    ) -> None:
        if rerank_top_n < 1:
            raise ValueError("rerank_top_n must be positive.")
        self.lexical_engine = lexical_engine
        self.dense_engine = dense_engine
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
        max_excerpt_len: Optional[int] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Return RRF results, optionally reranking only the leading candidates."""
        if not query or not query.strip() or top_k < 1:
            return []
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k.")
        lexical = self.lexical_engine.search(
            query, top_k=candidate_k, max_excerpt_len=max_excerpt_len, metadata_filters=metadata_filters
        )
        dense = self.dense_engine.search(
            query, top_k=candidate_k, max_excerpt_len=max_excerpt_len, metadata_filters=metadata_filters
        )
        fused = reciprocal_rank_fusion((lexical, dense), rrf_k=self.rrf_k, top_k=candidate_k)
        if self.reranker is None:
            return fused[:top_k]
        return self.reranker.rerank(query, fused[: self.rerank_top_n], top_k=top_k)
