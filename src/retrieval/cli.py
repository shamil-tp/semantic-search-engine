"""CLI for Phase 2 Lexical Indexing, Search, and Evaluation."""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.evaluation.evaluator import BenchmarkEvaluator
from src.ingestion.schemas import DocumentChunk
from src.query.expansion import QueryExpander
from src.query.intent import IntentClassifier
from src.query.ner import EntityRecognizer
from src.query.pipeline import QueryPipeline
from src.retrieval.dense import DenseSearchEngine
from src.retrieval.ann import HnswSearchEngine
from src.retrieval.hybrid import CrossEncoderReranker, HybridSearchEngine
from src.retrieval.lexical import LexicalSearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lexical_cli")


def load_chunks_from_jsonl(chunks_path: str | Path) -> list[DocumentChunk]:
    """Load DocumentChunk objects from chunks.jsonl."""
    path = Path(chunks_path)
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file does not exist: {path}")

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            chunks.append(DocumentChunk(**data))
    return chunks


def build_command(args):
    """Build and save lexical index from Phase 1 chunks."""
    logger.info(f"Loading chunks from {args.chunks_file}")
    chunks = load_chunks_from_jsonl(args.chunks_file)
    logger.info(f"Loaded {len(chunks)} chunks.")

    ngram_range = tuple(map(int, args.ngram_range.split(",")))
    engine = LexicalSearchEngine(
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )
    engine.build_index(chunks)
    engine.save(args.index_dir)
    print(f"\nSuccessfully built and saved lexical index to: {args.index_dir}")


def search_command(args):
    """Query an existing lexical index with optional query understanding."""
    logger.info(f"Loading index from {args.index_dir}")
    engine = LexicalSearchEngine.load(args.index_dir)

    # Configure query pipeline if flags specified
    intent_clf = None
    if args.enable_intent and args.intent_model:
        intent_clf = IntentClassifier.load(args.intent_model)

    pipeline = QueryPipeline(
        intent_classifier=intent_clf,
        enable_intent=args.enable_intent,
        enable_ner=args.enable_ner,
        enable_expansion=args.enable_expansion,
    )

    processed_q = pipeline.process_query(args.query)

    if args.enable_intent and processed_q.intent:
        print(f"\n[Intent Detected] {processed_q.intent.intent} (confidence: {processed_q.intent.confidence:.2f})")
    if args.enable_ner and processed_q.entities:
        ent_strs = [f"{e.text} ({e.label})" for e in processed_q.entities]
        print(f"[Entities Recognized] {', '.join(ent_strs)}")
        if processed_q.metadata_filters:
            print(f"[Metadata Filters] {processed_q.metadata_filters}")
    if args.enable_expansion and processed_q.expansion and processed_q.expansion.is_expanded:
        print(f"[Query Expansion] '{args.query}' -> '{processed_q.effective_query}'")

    results = engine.search(
        query=processed_q.effective_query,
        top_k=args.top_k,
        max_excerpt_len=args.excerpt_len,
        metadata_filters=processed_q.metadata_filters if (args.enable_ner and args.apply_filters) else None,
    )

    print(f"\nSearch results for query: '{args.query}' (top {args.top_k}):")
    if not results:
        print("No matching results found.")
        return

    for res in results:
        print(f"\n[Rank {res.rank}] Score: {res.score:.4f} | Chunk ID: {res.chunk_id}")
        print(f"Document ID: {res.document_id} | Path: {res.source_path}")
        print(f"Text excerpt:\n{res.text}")


def evaluate_command(args):
    """Evaluate lexical index against labeled evaluation dataset."""
    logger.info(f"Loading index from {args.index_dir}")
    engine = LexicalSearchEngine.load(args.index_dir)

    eval_path = Path(args.eval_file)
    if not eval_path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    # Configure query pipeline if flags specified
    query_pipe = None
    if args.enable_expansion or args.enable_ner:
        query_pipe = QueryPipeline(
            enable_ner=args.enable_ner,
            enable_expansion=args.enable_expansion,
        )

    k_values = [int(k) for k in args.k_values.split(",")]
    evaluator = BenchmarkEvaluator(
        engine=engine,
        k_values=k_values,
        target_unit=args.target_unit,
    )

    report = evaluator.evaluate_queries(
        queries_data=queries_data,
        dataset_name=eval_path.stem,
        corpus_name=Path(args.index_dir).name,
        query_pipeline=query_pipe,
        enable_filters=args.apply_filters,
    )

    print("\n=== Evaluation Results ===")
    print(json.dumps(report["aggregate_metrics"], indent=2))

    if args.output_report:
        out_path = Path(args.output_report)
        if out_path.exists() and not args.force:
            raise FileExistsError(
                f"Report already exists at {out_path}. Use --force to overwrite or specify a new versioned path."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull evaluation report saved to: {out_path.resolve()}")


def build_dense_command(args):
    """Build and save dense semantic vector index using SentenceTransformers."""
    logger.info(f"Loading chunks from {args.chunks_file}")
    chunks = load_chunks_from_jsonl(args.chunks_file)
    logger.info(f"Loaded {len(chunks)} chunks.")

    engine = DenseSearchEngine(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    engine.build_index(chunks)
    engine.save(args.index_dir)
    print(f"\nSuccessfully built and saved dense index to: {args.index_dir}")


def search_dense_command(args):
    """Search dense semantic vector index."""
    logger.info(f"Loading dense index from {args.index_dir}")
    engine = DenseSearchEngine.load(args.index_dir, device=args.device)

    # Configure query pipeline if flags specified
    intent_clf = None
    if args.enable_intent and args.intent_model:
        intent_clf = IntentClassifier.load(args.intent_model)

    pipeline = QueryPipeline(
        intent_classifier=intent_clf,
        enable_intent=args.enable_intent,
        enable_ner=args.enable_ner,
        enable_expansion=args.enable_expansion,
    )

    processed_q = pipeline.process_query(args.query)

    if args.enable_intent and processed_q.intent:
        print(f"\n[Intent Detected] {processed_q.intent.intent} (confidence: {processed_q.intent.confidence:.2f})")
    if args.enable_ner and processed_q.entities:
        ent_strs = [f"{e.text} ({e.label})" for e in processed_q.entities]
        print(f"[Entities Recognized] {', '.join(ent_strs)}")
        if processed_q.metadata_filters:
            print(f"[Metadata Filters] {processed_q.metadata_filters}")
    if args.enable_expansion and processed_q.expansion and processed_q.expansion.is_expanded:
        print(f"[Query Expansion] '{args.query}' -> '{processed_q.effective_query}'")

    results = engine.search(
        query=processed_q.effective_query,
        top_k=args.top_k,
        max_excerpt_len=args.excerpt_len,
        metadata_filters=processed_q.metadata_filters if (args.enable_ner and args.apply_filters) else None,
    )

    print(f"\n[Dense Search Results] Query: '{args.query}' (top {args.top_k}):")
    if not results:
        print("No matching results found.")
        return

    for res in results:
        print(f"\n[Rank {res.rank}] Score: {res.score:.4f} | Chunk ID: {res.chunk_id}")
        print(f"Document ID: {res.document_id} | Path: {res.source_path}")
        print(f"Text excerpt:\n{res.text}")


def evaluate_dense_command(args):
    """Evaluate dense index against labeled evaluation dataset."""
    logger.info(f"Loading dense index from {args.index_dir}")
    engine = DenseSearchEngine.load(args.index_dir, device=args.device)

    eval_path = Path(args.eval_file)
    if not eval_path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    query_pipe = None
    if args.enable_expansion or args.enable_ner:
        query_pipe = QueryPipeline(
            enable_ner=args.enable_ner,
            enable_expansion=args.enable_expansion,
        )

    k_values = [int(k) for k in args.k_values.split(",")]
    evaluator = BenchmarkEvaluator(
        engine=engine,
        k_values=k_values,
        target_unit=args.target_unit,
    )

    report = evaluator.evaluate_queries(
        queries_data=queries_data,
        dataset_name=eval_path.stem,
        corpus_name=Path(args.index_dir).name,
        query_pipeline=query_pipe,
        enable_filters=args.apply_filters,
    )

    print("\n=== Dense Evaluation Results ===")
    print(json.dumps(report["aggregate_metrics"], indent=2))

    if args.output_report:
        out_path = Path(args.output_report)
        if out_path.exists() and not args.force:
            raise FileExistsError(
                f"Report already exists at {out_path}. Use --force to overwrite or specify a new versioned path."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull dense evaluation report saved to: {out_path.resolve()}")


def build_ann_command(args):
    """Build HNSW ANN artifacts from an existing exact dense index."""
    dense = DenseSearchEngine.load(args.dense_index_dir, device=args.device)
    engine = HnswSearchEngine.from_dense(
        dense,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    engine.save(args.index_dir)
    print(f"\nSuccessfully built and saved HNSW ANN index to: {args.index_dir}")


def search_ann_command(args):
    """Search a persisted HNSW ANN index."""
    engine = HnswSearchEngine.load(args.index_dir, device=args.device)
    results = engine.search(args.query, top_k=args.top_k, max_excerpt_len=args.excerpt_len)
    print(f"\n[ANN Search Results] Query: '{args.query}' (top {args.top_k}):")
    if not results:
        print("No matching results found.")
        return
    for result in results:
        print(f"\n[Rank {result.rank}] Score: {result.score:.4f} | Chunk ID: {result.chunk_id}")
        print(f"Document ID: {result.document_id} | Path: {result.source_path}")
        print(f"Text excerpt:\n{result.text}")


def search_hybrid_command(args):
    """Fuse lexical and dense candidates, optionally reranking the bounded top-N."""
    lexical = LexicalSearchEngine.load(args.lexical_index_dir)
    dense = DenseSearchEngine.load(args.dense_index_dir, device=args.device)
    reranker = None
    if args.rerank_model:
        reranker = CrossEncoderReranker(model_name=args.rerank_model, device=args.device)
    engine = HybridSearchEngine(
        lexical_engine=lexical,
        dense_engine=dense,
        rrf_k=args.rrf_k,
        reranker=reranker,
        rerank_top_n=args.rerank_top_n,
    )
    results = engine.search(
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_excerpt_len=args.excerpt_len,
    )
    mode = "RRF + cross-encoder" if reranker else "RRF"
    print(f"\n[Hybrid Search Results: {mode}] Query: '{args.query}' (top {args.top_k}):")
    if not results:
        print("No matching results found.")
        return
    for result in results:
        print(f"\n[Rank {result.rank}] Score: {result.score:.4f} | Chunk ID: {result.chunk_id}")
        print(f"Document ID: {result.document_id} | Path: {result.source_path}")
        print(f"Text excerpt:\n{result.text}")
    parser = argparse.ArgumentParser(description="Phase 2 Lexical Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build index subcommand
    build_parser = subparsers.add_parser("build", help="Build and save lexical TF-IDF index")
    build_parser.add_argument("--chunks-file", "-c", required=True, help="Path to chunks.jsonl")
    build_parser.add_argument("--index-dir", "-o", required=True, help="Directory to save index")
    build_parser.add_argument("--ngram-range", default="1,2", help="N-gram range, e.g. '1,2'")
    build_parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency")
    build_parser.add_argument("--max-df", type=float, default=1.0, help="Maximum document frequency")
    build_parser.add_argument("--max-features", type=int, default=None, help="Maximum vocabulary features")
    build_parser.set_defaults(func=build_command)

    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Search the lexical index")
    search_parser.add_argument("--index-dir", "-i", required=True, help="Path to saved index")
    search_parser.add_argument("--query", "-q", required=True, help="Search query string")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--excerpt-len", type=int, default=200, help="Max length of text excerpt")
    search_parser.add_argument("--enable-intent", action="store_true", help="Enable intent classification")
    search_parser.add_argument("--intent-model", help="Path to trained intent classifier directory")
    search_parser.add_argument("--enable-ner", action="store_true", help="Enable entity recognition")
    search_parser.add_argument("--apply-filters", action="store_true", help="Apply entity-derived metadata filters")
    search_parser.add_argument("--enable-expansion", action="store_true", help="Enable synonym & spelling expansion")
    search_parser.set_defaults(func=search_command)

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate index against labeled queries")
    eval_parser.add_argument("--index-dir", "-i", required=True, help="Path to saved index")
    eval_parser.add_argument("--eval-file", "-e", required=True, help="Path to queries JSON file")
    eval_parser.add_argument("--output-report", "-o", help="Path to save evaluation report JSON")
    eval_parser.add_argument("--k-values", default="1,3,5,10", help="Comma-separated k values")
    eval_parser.add_argument("--target-unit", default="document_id", choices=["document_id", "chunk_id"])
    eval_parser.add_argument("--enable-expansion", action="store_true", help="Enable query expansion in evaluation")
    eval_parser.add_argument("--enable-ner", action="store_true", help="Enable NER filter extraction in evaluation")
    eval_parser.add_argument("--apply-filters", action="store_true", help="Apply extracted metadata filters")
    eval_parser.add_argument("--force", action="store_true", help="Overwrite existing report file")
    eval_parser.set_defaults(func=evaluate_command)

    # Build dense subcommand
    build_dense_parser = subparsers.add_parser("build-dense", help="Build and save dense vector index")
    build_dense_parser.add_argument("--chunks-file", "-c", required=True, help="Path to chunks.jsonl")
    build_dense_parser.add_argument("--index-dir", "-o", required=True, help="Directory to save dense index")
    build_dense_parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    build_dense_parser.add_argument("--device", default="cpu", help="Device to run inference on (cpu, cuda)")
    build_dense_parser.add_argument("--batch-size", type=int, default=32, help="Embedding encoding batch size")
    build_dense_parser.set_defaults(func=build_dense_command)

    # Search dense subcommand
    search_dense_parser = subparsers.add_parser("search-dense", help="Search the dense vector index")
    search_dense_parser.add_argument("--index-dir", "-i", required=True, help="Path to saved dense index")
    search_dense_parser.add_argument("--query", "-q", required=True, help="Search query string")
    search_dense_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return")
    search_dense_parser.add_argument("--excerpt-len", type=int, default=200, help="Max length of text excerpt")
    search_dense_parser.add_argument("--device", default="cpu", help="Device to run inference on (cpu, cuda)")
    search_dense_parser.add_argument("--enable-intent", action="store_true", help="Enable intent classification")
    search_dense_parser.add_argument("--intent-model", help="Path to trained intent classifier directory")
    search_dense_parser.add_argument("--enable-ner", action="store_true", help="Enable entity recognition")
    search_dense_parser.add_argument("--apply-filters", action="store_true", help="Apply entity-derived metadata filters")
    search_dense_parser.add_argument("--enable-expansion", action="store_true", help="Enable synonym & spelling expansion")
    search_dense_parser.set_defaults(func=search_dense_command)

    # Evaluate dense subcommand
    eval_dense_parser = subparsers.add_parser("evaluate-dense", help="Evaluate dense index against labeled queries")
    eval_dense_parser.add_argument("--index-dir", "-i", required=True, help="Path to saved dense index")
    eval_dense_parser.add_argument("--eval-file", "-e", required=True, help="Path to queries JSON file")
    eval_dense_parser.add_argument("--output-report", "-o", help="Path to save evaluation report JSON")
    eval_dense_parser.add_argument("--k-values", default="1,3,5,10", help="Comma-separated k values")
    eval_dense_parser.add_argument("--target-unit", default="document_id", choices=["document_id", "chunk_id"])
    eval_dense_parser.add_argument("--device", default="cpu", help="Device to run inference on (cpu, cuda)")
    eval_dense_parser.add_argument("--enable-expansion", action="store_true", help="Enable query expansion in evaluation")
    eval_dense_parser.add_argument("--enable-ner", action="store_true", help="Enable NER filter extraction in evaluation")
    eval_dense_parser.add_argument("--apply-filters", action="store_true", help="Apply extracted metadata filters")
    eval_dense_parser.add_argument("--force", action="store_true", help="Overwrite existing report file")
    eval_dense_parser.set_defaults(func=evaluate_dense_command)

    build_ann_parser = subparsers.add_parser("build-ann", help="Build a persisted HNSW index from a dense index")
    build_ann_parser.add_argument("--dense-index-dir", required=True, help="Path to the exact dense index")
    build_ann_parser.add_argument("--index-dir", "-o", required=True, help="Directory for ANN artifacts")
    build_ann_parser.add_argument("--m", type=int, default=16, help="HNSW graph degree")
    build_ann_parser.add_argument("--ef-construction", type=int, default=200, help="HNSW construction search breadth")
    build_ann_parser.add_argument("--ef-search", type=int, default=100, help="HNSW query search breadth")
    build_ann_parser.add_argument("--device", default="cpu", help="Device to encode queries on")
    build_ann_parser.set_defaults(func=build_ann_command)

    search_ann_parser = subparsers.add_parser("search-ann", help="Search a persisted HNSW ANN index")
    search_ann_parser.add_argument("--index-dir", "-i", required=True, help="Path to ANN artifacts")
    search_ann_parser.add_argument("--query", "-q", required=True, help="Search query string")
    search_ann_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return")
    search_ann_parser.add_argument("--excerpt-len", type=int, default=200, help="Maximum text excerpt length")
    search_ann_parser.add_argument("--device", default="cpu", help="Device to encode queries on")
    search_ann_parser.set_defaults(func=search_ann_command)

    search_hybrid_parser = subparsers.add_parser("search-hybrid", help="Fuse lexical and dense retrieval with optional reranking")
    search_hybrid_parser.add_argument("--lexical-index-dir", required=True, help="Path to lexical TF-IDF index")
    search_hybrid_parser.add_argument("--dense-index-dir", required=True, help="Path to exact dense index")
    search_hybrid_parser.add_argument("--query", "-q", required=True, help="Search query string")
    search_hybrid_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of final results")
    search_hybrid_parser.add_argument("--candidate-k", type=int, default=50, help="Candidates retrieved per engine")
    search_hybrid_parser.add_argument("--rrf-k", type=int, default=60, help="Reciprocal-rank fusion constant")
    search_hybrid_parser.add_argument("--rerank-model", help="Cross-encoder model; omit to use RRF only")
    search_hybrid_parser.add_argument("--rerank-top-n", type=int, default=20, help="Maximum fused candidates passed to reranker")
    search_hybrid_parser.add_argument("--excerpt-len", type=int, default=200, help="Maximum text excerpt length")
    search_hybrid_parser.add_argument("--device", default="cpu", help="Device for dense and reranker models")
    search_hybrid_parser.set_defaults(func=search_hybrid_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
