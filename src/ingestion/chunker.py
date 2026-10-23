"""Document chunking module with configurable size and overlap."""

import hashlib
import logging
import re
from typing import Iterator, List

from src.ingestion.schemas import DocumentChunk, ProcessedDocument

logger = logging.getLogger(__name__)


def chunk_document(
    doc: ProcessedDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[DocumentChunk]:
    """
    Split a ProcessedDocument into chunks of approximate word count with overlap.
    Preserves:
    - Stable chunk_id (`{doc_id}_c{chunk_index}`)
    - Parent document_id
    - Source path
    - Chunk index
    - start_char and end_char offsets in doc.text
    - Parent metadata with chunk-specific enhancements
    
    Args:
        doc: Cleaned processed document.
        chunk_size: Target words per chunk (> 0).
        chunk_overlap: Target overlapping words (0 <= chunk_overlap < chunk_size).
        
    Returns:
        List of non-empty DocumentChunk objects.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    text = doc.text
    if not text.strip():
        return []

    # Tokenize by whitespace spans so we know exact character boundaries for each word
    word_matches = list(re.finditer(r"\S+", text))
    if not word_matches:
        return []

    total_words = len(word_matches)

    # If document is shorter than or equal to chunk_size, produce a single chunk
    if total_words <= chunk_size:
        chunk_id = f"{doc.document_id}_c0"
        return [
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=doc.document_id,
                source_path=doc.source_path,
                chunk_index=0,
                text=text,
                start_char=0,
                end_char=len(text),
                metadata={
                    **doc.metadata,
                    "word_count": total_words,
                    "char_count": len(text),
                },
            )
        ]

    step = chunk_size - chunk_overlap
    chunks: List[DocumentChunk] = []
    chunk_index = 0
    start_idx = 0

    while start_idx < total_words:
        end_idx = min(start_idx + chunk_size, total_words)

        start_char = word_matches[start_idx].start()
        end_char = word_matches[end_idx - 1].end()

        chunk_text = text[start_char:end_char]
        chunk_words = end_idx - start_idx
        chunk_id = f"{doc.document_id}_c{chunk_index}"

        chunk_obj = DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc.document_id,
            source_path=doc.source_path,
            chunk_index=chunk_index,
            text=chunk_text,
            start_char=start_char,
            end_char=end_char,
            metadata={
                **doc.metadata,
                "word_count": chunk_words,
                "char_count": len(chunk_text),
            },
        )
        chunks.append(chunk_obj)
        chunk_index += 1

        if end_idx >= total_words:
            break

        start_idx += step

    return chunks
