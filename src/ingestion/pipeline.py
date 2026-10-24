"""Pipeline orchestrator for Phase 1 Data Ingestion and Preparation."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import chunk_document
from src.ingestion.loader import discover_files, load_crawler_json, load_file
from src.ingestion.schemas import DocumentChunk, ProcessedDocument
from src.ingestion.statistics import calculate_corpus_statistics

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates:
    1. Discovery of files (.txt, .md).
    2. Safe loading and UTF-8 handling.
    3. Normalization, empty skipping, deduplication.
    4. Overlapping chunk generation.
    5. Dataset statistics profiling with pandas/NumPy.
    6. JSONL and JSON statistics persistence.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        supported_extensions: Optional[Set[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = supported_extensions
        self.cleaner = TextCleaner()

    def run(
        self,
        input_dir: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute ingestion pipeline across input_dir and optionally persist results to output_dir.
        """
        input_path = Path(input_dir)
        discovered_paths = discover_files(input_path, self.supported_extensions)

        processed_docs: List[ProcessedDocument] = []
        all_chunks: List[DocumentChunk] = []

        failed_count = 0
        empty_count = 0
        duplicate_count = 0
        failure_reasons: Dict[str, int] = {}

        for file_path in discovered_paths:
            raw_doc, err = load_file(file_path)
            if err is not None:
                failed_count += 1
                failure_reasons[err] = failure_reasons.get(err, 0) + 1
                continue

            assert raw_doc is not None
            processed_doc, clean_status = self.cleaner.clean_document(raw_doc)

            if clean_status == "empty":
                empty_count += 1
                continue
            elif clean_status == "duplicate":
                duplicate_count += 1
                continue

            assert processed_doc is not None
            processed_docs.append(processed_doc)

            doc_chunks = chunk_document(
                processed_doc,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            all_chunks.extend(doc_chunks)

        stats = calculate_corpus_statistics(
            discovered_count=len(discovered_paths),
            skipped_unsupported_count=0,
            failed_count=failed_count,
            empty_count=empty_count,
            duplicate_count=duplicate_count,
            processed_documents=processed_docs,
            chunks=all_chunks,
            failure_reasons=failure_reasons,
        )

        if output_dir:
            self._save_results(output_dir, processed_docs, all_chunks, stats)

        return {
            "processed_documents": processed_docs,
            "chunks": all_chunks,
            "statistics": stats,
        }

    def run_crawler_file(
        self,
        crawler_file: str | Path,
        output_dir: Optional[str | Path] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute ingestion pipeline on a crawler JSON / JSONL export file.
        """
        path = Path(crawler_file)
        logger.info(f"Ingesting crawler data from {path.resolve()} (limit={limit})")

        processed_docs: List[ProcessedDocument] = []
        all_chunks: List[DocumentChunk] = []

        empty_count = 0
        duplicate_count = 0
        total_discovered = 0

        for raw_doc in load_crawler_json(path, limit=limit):
            total_discovered += 1
            processed_doc, clean_status = self.cleaner.clean_document(raw_doc)

            if clean_status == "empty":
                empty_count += 1
                continue
            elif clean_status == "duplicate":
                duplicate_count += 1
                continue

            assert processed_doc is not None
            processed_docs.append(processed_doc)

            doc_chunks = chunk_document(
                processed_doc,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            all_chunks.extend(doc_chunks)

        stats = calculate_corpus_statistics(
            discovered_count=total_discovered,
            skipped_unsupported_count=0,
            failed_count=0,
            empty_count=empty_count,
            duplicate_count=duplicate_count,
            processed_documents=processed_docs,
            chunks=all_chunks,
        )

        if output_dir:
            self._save_results(output_dir, processed_docs, all_chunks, stats)

        return {
            "processed_documents": processed_docs,
            "chunks": all_chunks,
            "statistics": stats,
        }

    def _save_results(
        self,
        output_dir: str | Path,
        documents: List[ProcessedDocument],
        chunks: List[DocumentChunk],
        stats: Dict[str, Any],
    ) -> None:
        """Save documents, chunks, and statistics into structured JSON/JSONL format."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        docs_file = out_path / "documents.jsonl"
        with open(docs_file, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

        chunks_file = out_path / "chunks.jsonl"
        with open(chunks_file, "w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")

        stats_file = out_path / "statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Ingestion results written to {out_path.resolve()}")
