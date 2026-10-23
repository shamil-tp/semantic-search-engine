"""Schema definitions for lexical search results and indexed items."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class SearchResult:
    """Represents an individual ranked search result."""

    rank: int
    chunk_id: str
    document_id: str
    score: float
    text: str
    source_path: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to dictionary."""
        return asdict(self)
