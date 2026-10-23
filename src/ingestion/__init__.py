"""Phase 1 Data Ingestion and Preparation Package."""

from src.ingestion.cleaner import TextCleaner, compute_content_hash, normalize_text
from src.ingestion.chunker import chunk_document
from src.ingestion.loader import discover_files, generate_document_id, load_file
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.schemas import DocumentChunk, ProcessedDocument
from src.ingestion.statistics import calculate_corpus_statistics

__all__ = [
    "ProcessedDocument",
    "DocumentChunk",
    "discover_files",
    "load_file",
    "generate_document_id",
    "normalize_text",
    "compute_content_hash",
    "TextCleaner",
    "chunk_document",
    "calculate_corpus_statistics",
    "IngestionPipeline",
]
