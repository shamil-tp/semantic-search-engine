"""Reproducible evaluation and latency comparison for persisted pipelines."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import BenchmarkEvaluator
from src.retrieval.dense import DenseSearchEngine
from src.retrieval.hybrid import HybridSearchEngine
from src.retrieval.lexical import LexicalSearchEngine


def _measure_latency(
    engine: Any, queries: list[dict[str, Any]], repeats: int, top_k: int
) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(repeats):
        for item in queries:
            query = item.get("query", "")
            if not isinstance(query, str) or not query.strip():
                continue
            started = time.perf_counter()
            engine.search(query, top_k=top_k)
            samples.append((time.perf_counter() - started) * 1_000)
    if not samples:
        return {"sample_count": 0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))
    return {
        "sample_count": len(samples),
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(ordered[p95_index], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare persisted lexical, dense, and RRF pipelines."
    )
    parser.add_argument("--lexical-index-dir", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument(
        "--dense-index-dir",
        help="Optional exact dense index; enables dense and hybrid rows.",
    )
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.top_k < 1:
        parser.error("--repeats and --top-k must be positive.")

    queries = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    engines: dict[str, Any] = {
        "lexical": LexicalSearchEngine.load(args.lexical_index_dir)
    }
    if args.dense_index_dir:
        dense = DenseSearchEngine.load(args.dense_index_dir)
        engines["dense"] = dense
        engines["hybrid_rrf"] = HybridSearchEngine(engines["lexical"], dense)

    report: dict[str, Any] = {
        "evaluation_input": {
            "eval_file": str(Path(args.eval_file)),
            "top_k": args.top_k,
            "latency_repeats": args.repeats,
        },
        "pipelines": {},
        "notes": [
            "Metrics use all supplied relevance judgments. This command does not create a validation/test split.",
            "Reranking is excluded unless separately evaluated with a suitable held-out set.",
        ],
    }
    for name, engine in engines.items():
        metrics = BenchmarkEvaluator(
            engine, k_values=[1, 3, 5], target_unit="document_id"
        ).evaluate_queries(
            queries, dataset_name=Path(args.eval_file).stem, corpus_name=name
        )
        report["pipelines"][name] = {
            "metrics": metrics["aggregate_metrics"],
            "latency": _measure_latency(engine, queries, args.repeats, args.top_k),
        }

    target = Path(args.output_report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
