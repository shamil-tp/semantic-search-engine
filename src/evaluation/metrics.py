"""Evaluation metrics: Recall@k, MRR, and nDCG@k."""

import math
from typing import Any, Dict, List, Optional, Sequence, Set


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Recall@k.
    The fraction of known relevant items retrieved in top k results.
    
    If relevant_ids is empty, returns 0.0.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / float(len(relevant_ids))


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """
    Compute Reciprocal Rank (RR).
    1 / (rank of first relevant item). 1-indexed.
    If no relevant item is retrieved, returns 0.0.
    """
    if not relevant_ids:
        return 0.0

    for rank_idx, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank_idx
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: Sequence[Sequence[str]],
    all_relevant: Sequence[Set[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) across queries.
    Returns 0.0 if empty list of queries.
    """
    if not all_retrieved or not all_relevant:
        return 0.0
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("all_retrieved and all_relevant must have the same length")

    rrs = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    return sum(rrs) / float(len(rrs))


def dcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_scores: Dict[str, float],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain at k (DCG@k).
    Uses standard formula: sum_{i=1}^k (rel_i / log2(i + 1)).
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, item_id in enumerate(top_k, start=1):
        rel = relevance_scores.get(item_id, 0.0)
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_scores: Dict[str, float] | Set[str],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k (nDCG@k).
    Supports either binary relevance (Set[str]) or graded relevance (Dict[str, float]).
    Returns 0.0 if ideal DCG is 0.0.
    """
    if k <= 0:
        return 0.0

    # Convert binary set to dict with 1.0 relevance
    if isinstance(relevance_scores, (set, list, tuple)):
        scores_map = {item: 1.0 for item in relevance_scores}
    else:
        scores_map = dict(relevance_scores)

    # Positive relevance items
    positive_scores = [v for v in scores_map.values() if v > 0.0]
    if not positive_scores:
        return 0.0

    actual_dcg = dcg_at_k(retrieved_ids, scores_map, k)

    # Ideal ranking: sorted positive relevance descending
    ideal_sorted_scores = sorted(positive_scores, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_sorted_scores, start=1))

    if idcg == 0.0:
        return 0.0

    return actual_dcg / idcg
