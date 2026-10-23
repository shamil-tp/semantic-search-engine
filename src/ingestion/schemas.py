"""Data schemas for processed documents and chunks in Phase 1."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ProcessedDocument:
    """Represents a cleaned and deduplicated document ready for chunking."""

    document_id: str
    source_path: str
    content_hash: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a text chunk derived from a parent document."""

    chunk_id: str
    document_id: str
    source_path: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)
