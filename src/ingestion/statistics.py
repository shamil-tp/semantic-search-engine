"""Corpus profiling and dataset statistics using pandas and NumPy."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.ingestion.schemas import DocumentChunk, ProcessedDocument


def calculate_corpus_statistics(
    discovered_count: int,
    skipped_unsupported_count: int,
    failed_count: int,
    empty_count: int,
    duplicate_count: int,
    processed_documents: List[ProcessedDocument],
    chunks: List[DocumentChunk],
    failure_reasons: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """
    Generate dataset statistics and profiling report using pandas and NumPy.
    Safely handles empty corpus without throwing ZeroDivisionError or NaN issues.
    """
    total_processed = len(processed_documents)
    total_chunks = len(chunks)

    # Document length statistics
    if total_processed > 0:
        doc_df = pd.DataFrame(
            [
                {
                    "char_length": len(d.text),
                    "word_length": len(d.text.split()),
                    "has_metadata": bool(d.metadata),
                }
                for d in processed_documents
            ]
        )
        doc_char_lengths = doc_df["char_length"].to_numpy()
        doc_word_lengths = doc_df["word_length"].to_numpy()

        doc_stats = {
            "min_chars": int(np.min(doc_char_lengths)),
            "max_chars": int(np.max(doc_char_lengths)),
            "mean_chars": float(round(float(np.mean(doc_char_lengths)), 2)),
            "median_chars": float(round(float(np.median(doc_char_lengths)), 2)),
            "min_words": int(np.min(doc_word_lengths)),
            "max_words": int(np.max(doc_word_lengths)),
            "mean_words": float(round(float(np.mean(doc_word_lengths)), 2)),
            "median_words": float(round(float(np.median(doc_word_lengths)), 2)),
            "missing_metadata_count": int((~doc_df["has_metadata"]).sum()),
        }
    else:
        doc_stats = {
            "min_chars": 0,
            "max_chars": 0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "min_words": 0,
            "max_words": 0,
            "mean_words": 0.0,
            "median_words": 0.0,
            "missing_metadata_count": 0,
        }

    # Chunk length statistics
    if total_chunks > 0:
        chunk_df = pd.DataFrame(
            [
                {
                    "char_length": len(c.text),
                    "word_length": len(c.text.split()),
                    "has_metadata": bool(c.metadata),
                }
                for c in chunks
            ]
        )
        chunk_char_lengths = chunk_df["char_length"].to_numpy()
        chunk_word_lengths = chunk_df["word_length"].to_numpy()

        chunk_stats = {
            "min_chars": int(np.min(chunk_char_lengths)),
            "max_chars": int(np.max(chunk_char_lengths)),
            "mean_chars": float(round(float(np.mean(chunk_char_lengths)), 2)),
            "median_chars": float(round(float(np.median(chunk_char_lengths)), 2)),
            "min_words": int(np.min(chunk_word_lengths)),
            "max_words": int(np.max(chunk_word_lengths)),
            "mean_words": float(round(float(np.mean(chunk_word_lengths)), 2)),
            "median_words": float(round(float(np.median(chunk_word_lengths)), 2)),
            "missing_metadata_count": int((~chunk_df["has_metadata"]).sum()),
        }
    else:
        chunk_stats = {
            "min_chars": 0,
            "max_chars": 0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "min_words": 0,
            "max_words": 0,
            "mean_words": 0.0,
            "median_words": 0.0,
            "missing_metadata_count": 0,
        }

    return {
        "file_summary": {
            "files_discovered": discovered_count,
            "files_skipped_unsupported": skipped_unsupported_count,
            "files_failed": failed_count,
            "files_empty": empty_count,
            "files_duplicate": duplicate_count,
            "unique_processed_documents": total_processed,
            "generated_chunks": total_chunks,
            "failure_reasons": failure_reasons or {},
        },
        "document_statistics": doc_stats,
        "chunk_statistics": chunk_stats,
    }
