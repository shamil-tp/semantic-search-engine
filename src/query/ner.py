"""Lightweight Named Entity Recognition and metadata filtering extractor."""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class ExtractedEntity:
    """Represents a recognized entity or structured token in a query."""

    text: str
    label: str  # e.g., 'FILE_EXTENSION', 'METRIC', 'SECTION_REF', 'ALGORITHM', 'SYSTEM_FLAG'
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Configurable entity patterns for semantic search and documentation domain
DEFAULT_ENTITY_PATTERNS = [
    # File extensions (either standalone e.g. .md or part of filename e.g. doc.md)
    (r"(\.(?:txt|md|jsonl|json|py|csv|npz|pkl))\b", "FILE_EXTENSION"),
    # Common IR metrics
    (r"\b(?:recall@\d+|ndcg@\d+|mrr|dcg@\d+|map|f1[-_]?score)\b", "METRIC"),
    # Document IDs or chunk IDs (e.g., a3090451a0a256d7, doc1, chunk_0)
    (r"\b[0-9a-f]{16}(?:_c\d+)?\b", "DOCUMENT_OR_CHUNK_ID"),
    # Algorithms and machine learning models
    (r"\b(?:tf-idf|tfidf|linear\s*svm|bm25|faiss|hnsw|sentence-bert|sbert|svm|ann)\b", "ALGORITHM"),
    # Technical schemas and fields
    (r"\b(?:start_char|end_char|document_id|chunk_id|content_hash|source_path)\b", "SCHEMA_FIELD"),
    # CLI parameters
    (r"--(?:input-dir|output-dir|chunk-size|chunk-overlap|top-k|ngram-range)\b", "CLI_FLAG"),
]


class EntityRecognizer:
    """
    Rule and pattern-based entity recognizer for query understanding and filter extraction.
    Provides fast, deterministic extraction without heavy external dependencies.
    """

    def __init__(self, patterns: Optional[List[Tuple[str, str]]] = None):
        raw_patterns = patterns or DEFAULT_ENTITY_PATTERNS
        self.compiled_patterns = [
            (re.compile(pat, re.IGNORECASE), label)
            for pat, label in raw_patterns
        ]

    def extract_entities(self, query: str) -> List[ExtractedEntity]:
        """Extract all recognized entities with character offsets."""
        if not query:
            return []

        entities: List[ExtractedEntity] = []
        for regex, label in self.compiled_patterns:
            for match in regex.finditer(query):
                # If there is a capture group (like file extension), use group 1 offsets and text
                if match.groups():
                    text = match.group(1)
                    start = match.start(1)
                    end = match.end(1)
                else:
                    text = match.group(0)
                    start = match.start()
                    end = match.end()

                entities.append(
                    ExtractedEntity(
                        text=text,
                        label=label,
                        start=start,
                        end=end,
                    )
                )

        # Sort entities by start offset
        entities.sort(key=lambda e: (e.start, e.end))
        return entities

    def extract_metadata_filters(self, query: str) -> Dict[str, Any]:
        """
        Derive metadata filters from recognized entities.
        For example:
        - Mention of '.md' -> {'extension': '.md'}
        - Mention of 'document_id' -> metadata constraint
        """
        filters: Dict[str, Any] = {}
        entities = self.extract_entities(query)

        for ent in entities:
            if ent.label == "FILE_EXTENSION":
                filters["extension"] = ent.text.lower()
            elif ent.label == "DOCUMENT_OR_CHUNK_ID":
                if "_c" in ent.text:
                    filters["chunk_id"] = ent.text
                else:
                    filters["document_id"] = ent.text

        return filters
