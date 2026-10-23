"""Text cleaning, normalization, and deduplication module."""

import hashlib
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from src.ingestion.loader import RawDocument
from src.ingestion.schemas import ProcessedDocument

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Clean and normalize text safely:
    - Normalizes Unicode (NFC form).
    - Standardizes line breaks (\\r\\n and \\r -> \\n).
    - Preserves meaningful paragraph boundaries while cleaning unnecessary whitespace:
      - Strips trailing/leading whitespace per line.
      - Collapses consecutive whitespace characters (excluding newlines) to a single space.
      - Collapses 3 or more consecutive newlines into double newlines (paragraph boundary).
    - Avoids destructive token/punctuation removal to preserve semantic meaning.
    """
    if not text:
        return ""

    # 1. Unicode normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Clean line by line: collapse horizontal whitespace, strip each line
    lines: List[str] = []
    for line in text.split("\n"):
        # Replace non-newline whitespace sequences (tabs, non-breaking spaces, spaces) with a single space
        cleaned_line = re.sub(r"[^\S\n]+", " ", line).strip()
        lines.append(cleaned_line)

    normalized = "\n".join(lines)

    # 4. Collapse multiple blank lines (>2 newlines to 2 newlines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # 5. Trim leading and trailing whitespace of entire text
    return normalized.strip()


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TextCleaner:
    """
    Cleans raw documents, filters empty records, and performs duplicate detection.
    """

    def __init__(self, seen_hashes: Optional[Set[str]] = None):
        # Set of seen content hashes across processing run
        self.seen_hashes: Set[str] = set() if seen_hashes is None else set(seen_hashes)

    def clean_document(
        self, raw_doc: RawDocument
    ) -> Tuple[Optional[ProcessedDocument], Optional[str]]:
        """
        Process a single RawDocument.
        
        Returns:
            (ProcessedDocument, None) if successfully cleaned and unique.
            (None, "empty") if text is empty after normalization.
            (None, "duplicate") if document content hash was already encountered.
        """
        cleaned_text = normalize_text(raw_doc.raw_text)

        if not cleaned_text:
            logger.info(f"Empty document skipped: {raw_doc.source_path}")
            return None, "empty"

        content_hash = compute_content_hash(cleaned_text)

        if content_hash in self.seen_hashes:
            logger.info(
                f"Duplicate document skipped (hash: {content_hash[:8]}): {raw_doc.source_path}"
            )
            return None, "duplicate"

        self.seen_hashes.add(content_hash)

        processed = ProcessedDocument(
            document_id=raw_doc.document_id,
            source_path=raw_doc.source_path,
            content_hash=content_hash,
            text=cleaned_text,
            metadata=dict(raw_doc.metadata),
        )
        return processed, None
