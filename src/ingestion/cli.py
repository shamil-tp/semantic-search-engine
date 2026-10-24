"""CLI entrypoint for Phase 1 Data Ingestion."""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.ingestion.pipeline import IngestionPipeline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Phase 1 Data Ingestion and Preparation Pipeline"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="Input directory containing raw documents (.txt, .md)",
    )
    parser.add_argument(
        "--crawler-file",
        "-f",
        type=str,
        help="Input crawler JSON / JSONL export file (e.g. data/crawler.sites.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to ingest from crawler file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Output directory to save processed documents, chunks, and statistics",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target number of words per chunk (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Number of overlapping words between consecutive chunks (default: 50)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".txt", ".md"],
        help="Supported file extensions (default: .txt .md)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    if not args.input_dir and not args.crawler_file:
        parser.error("Either --input-dir (-i) or --crawler-file (-f) must be specified.")

    logger = logging.getLogger("ingestion_cli")
    source_desc = f"directory: {args.input_dir}" if args.input_dir else f"crawler file: {args.crawler_file}"
    logger.info(f"Starting ingestion from {source_desc}")
    logger.info(f"Output target: {args.output_dir}")
    logger.info(f"Chunk size: {args.chunk_size}, Chunk overlap: {args.chunk_overlap}")

    pipeline = IngestionPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        supported_extensions=set(args.extensions),
    )

    try:
        if args.crawler_file:
            result = pipeline.run_crawler_file(
                crawler_file=args.crawler_file,
                output_dir=args.output_dir,
                limit=args.limit,
            )
        else:
            result = pipeline.run(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
            )
        stats = result["statistics"]
        print("\n=== Ingestion Completed Successfully ===")
        print(json.dumps(stats, indent=2))
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
