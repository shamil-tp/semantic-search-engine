from src.ingestion.schemas import DocumentChunk
from src.retrieval.hybrid import HybridSearchEngine, reciprocal_rank_fusion
from src.retrieval.lexical import LexicalSearchEngine
from src.retrieval.schemas import SearchResult


def _result(chunk_id: str, rank: int) -> SearchResult:
    return SearchResult(rank, chunk_id, chunk_id, 1.0, chunk_id, "/x", 0, {})


def test_rrf_rewards_agreement_and_is_deterministic():
    results = reciprocal_rank_fusion(
        [[_result("a", 1), _result("b", 2)], [_result("b", 1), _result("a", 2)]],
        top_k=2,
    )
    assert [result.chunk_id for result in results] == ["a", "b"]
    assert results[0].score == results[1].score


def test_hybrid_rejects_candidate_count_below_requested_top_k():
    engine = LexicalSearchEngine().build_index(
        [DocumentChunk("c", "d", "/x", 0, "semantic search", 0, 15)]
    )
    hybrid = HybridSearchEngine(engine, engine)
    try:
        hybrid.search("semantic", top_k=2, candidate_k=1)
    except ValueError as exc:
        assert "candidate_k" in str(exc)
    else:
        raise AssertionError("expected candidate count validation")
