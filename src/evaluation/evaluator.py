"""Evaluation framework for running queries and calculating benchmark metrics."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set
import sklearn

from src.evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k, reciprocal_rank
from src.retrieval.lexical import LexicalSearchEngine

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Evaluates a search engine instance against a labeled query set.
    Supports chunk-level and document-level matching modes.
    """

    def __init__(
        self,
        engine: LexicalSearchEngine,
        k_values: Sequence[int] = (1, 3, 5, 10),
        target_unit: str = "document_id",  # 'document_id' or 'chunk_id'
    ):
        self.engine = engine
        self.k_values = list(k_values)
        if target_unit not in ("document_id", "chunk_id"):
            raise ValueError(f"target_unit must be 'document_id' or 'chunk_id', got {target_unit}")
        self.target_unit = target_unit

    def evaluate_queries(
        self,
        queries_data: List[Dict[str, Any]],
        dataset_name: str = "synthetic_eval_v1",
        corpus_name: str = "sample_corpus",
    ) -> Dict[str, Any]:
        """
        Run evaluation across labeled queries.
        
        Expected queries_data schema:
        [
            {
                "query_id": "q1",
                "query": "query text",
                "relevant_ids": ["doc_id1", "doc_id2"],
                "relevance_scores": {"doc_id1": 1.0, "doc_id2": 0.5} (optional)
            }
        ]
        """
        valid_queries = []
        skipped_queries = []

        max_k = max(self.k_values) if self.k_values else 10

        per_query_results = []
        all_retrieved: List[List[str]] = []
        all_relevant: List[Set[str]] = []

        # Metric accumulators
        recall_sums = {k: 0.0 for k in self.k_values}
        ndcg_sums = {k: 0.0 for k in self.k_values}

        for q_entry in queries_data:
            query_id = q_entry.get("query_id")
            query_text = q_entry.get("query", "").strip()
            relevant_ids = set(q_entry.get("relevant_ids", []))
            relevance_scores = q_entry.get("relevance_scores", {r: 1.0 for r in relevant_ids})

            if not query_id or not query_text:
                skipped_queries.append({"query_id": query_id, "reason": "empty_query_or_id"})
                continue

            if not relevant_ids:
                skipped_queries.append({"query_id": query_id, "reason": "no_relevance_judgments"})
                continue

            valid_queries.append(q_entry)

            # Retrieve top_k results
            search_results = self.engine.search(query_text, top_k=max_k)

            # Map retrieved results to target unit (document_id or chunk_id)
            if self.target_unit == "document_id":
                retrieved_units: List[str] = []
                seen_docs: Set[str] = set()
                for res in search_results:
                    if res.document_id not in seen_docs:
                        seen_docs.add(res.document_id)
                        retrieved_units.append(res.document_id)
            else:
                retrieved_units = [res.chunk_id for res in search_results]

            all_retrieved.append(retrieved_units)
            all_relevant.append(relevant_ids)

            # Calculate individual query metrics
            rr = reciprocal_rank(retrieved_units, relevant_ids)
            q_recalls = {k: recall_at_k(retrieved_units, relevant_ids, k) for k in self.k_values}
            q_ndcgs = {k: ndcg_at_k(retrieved_units, relevance_scores, k) for k in self.k_values}

            for k in self.k_values:
                recall_sums[k] += q_recalls[k]
                ndcg_sums[k] += q_ndcgs[k]

            per_query_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "relevant_count": len(relevant_ids),
                    "retrieved_count": len(retrieved_units),
                    "reciprocal_rank": round(rr, 4),
                    "recall": {f"k_{k}": round(v, 4) for k, v in q_recalls.items()},
                    "ndcg": {f"k_{k}": round(v, 4) for k, v in q_ndcgs.items()},
                    "top_retrieved": [
                        {"rank": r.rank, "id": getattr(r, self.target_unit), "score": r.score}
                        for r in search_results[:3]
                    ],
                }
            )

        n_valid = len(valid_queries)
        if n_valid > 0:
            avg_mrr = mean_reciprocal_rank(all_retrieved, all_relevant)
            avg_recall = {f"recall@{k}": round(recall_sums[k] / n_valid, 4) for k in self.k_values}
            avg_ndcg = {f"ndcg@{k}": round(ndcg_sums[k] / n_valid, 4) for k in self.k_values}
        else:
            avg_mrr = 0.0
            avg_recall = {f"recall@{k}": 0.0 for k in self.k_values}
            avg_ndcg = {f"ndcg@{k}": 0.0 for k in self.k_values}

        report = {
            "evaluation_metadata": {
                "dataset_name": dataset_name,
                "corpus_name": corpus_name,
                "target_unit": self.target_unit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_queries": len(queries_data),
                "valid_queries_evaluated": n_valid,
                "skipped_queries": skipped_queries,
                "scikit_learn_version": sklearn.__version__,
                "python_version": sys.version,
            },
            "aggregate_metrics": {
                "mrr": round(avg_mrr, 4),
                **avg_recall,
                **avg_ndcg,
            },
            "per_query_results": per_query_results,
        }

        return report
