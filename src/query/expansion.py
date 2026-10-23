"""Query expansion, spelling normalization, and synonym handling module."""

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Domain synonym dictionary mapping canonical or variant phrases to expanded terms
DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "semantic search": ["dense retrieval", "concept search", "vector search"],
    "vector search": ["dense retrieval", "embedding search", "semantic search"],
    "vector embeddings": ["dense vectors", "sentence embeddings"],
    "dense embeddings": ["vector embeddings", "sentence representations"],
    "hybrid retrieval": ["lexical and dense fusion", "reciprocal rank fusion"],
    "text preprocessing": ["text normalization", "data cleaning", "tokenization"],
    "chunking": ["text splitting", "passage splitting", "overlapping chunks"],
    "deduplication": ["duplicate detection", "content hash deduplication"],
    "unicode normalization": ["nfc normalization", "character encoding normalization"],
    "mrr": ["mean reciprocal rank"],
    "ndcg": ["normalized discounted cumulative gain"],
}

# Domain spelling correction mapping common typos to correct terminology
DEFAULT_SPELLING_CORRECTIONS: Dict[str, str] = {
    "sematic": "semantic",
    "semamtic": "semantic",
    "embedings": "embeddings",
    "embeding": "embedding",
    "preprocesing": "preprocessing",
    "pre-processing": "preprocessing",
    "chunck": "chunk",
    "chuncking": "chunking",
    "dedup": "deduplication",
    "deduplicate": "deduplication",
    "retreival": "retrieval",
    "tfidf": "tf-idf",
    "ndcg": "ndcg",
}


@dataclass(frozen=True)
class QueryExpansionResult:
    """Represents the expansion output, preserving the original query and traceability."""

    original_query: str
    normalized_query: str
    spelling_corrections: Dict[str, str]
    expanded_terms: List[str]
    expanded_query: str
    is_expanded: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryExpander:
    """
    Expands and normalizes queries while preserving the original intent.
    - Applies spelling corrections.
    - Appends controlled domain synonyms without removing the original terms.
    - Logs all expanded terms for evaluation and auditability.
    """

    def __init__(
        self,
        synonyms: Optional[Dict[str, List[str]]] = None,
        spelling_dict: Optional[Dict[str, str]] = None,
        max_expansion_terms: int = 4,
    ):
        self.synonyms = synonyms if synonyms is not None else DEFAULT_SYNONYMS
        self.spelling_dict = spelling_dict if spelling_dict is not None else DEFAULT_SPELLING_CORRECTIONS
        self.max_expansion_terms = max_expansion_terms

    def correct_spelling(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Normalize known domain typos and spelling variations.
        Returns (corrected_text, corrections_applied_dict).
        """
        if not text:
            return "", {}

        corrections: Dict[str, str] = {}
        tokens = re.findall(r"\b\w+(?:-\w+)?\b|\S", text)
        corrected_tokens = []

        for token in tokens:
            lower_token = token.lower()
            if lower_token in self.spelling_dict:
                replacement = self.spelling_dict[lower_token]
                # Preserve case if original was upper
                if token.isupper() and len(token) > 1:
                    replacement = replacement.upper()
                corrections[token] = replacement
                corrected_tokens.append(replacement)
            else:
                corrected_tokens.append(token)

        corrected_text = " ".join(corrected_tokens)
        return corrected_text, corrections

    def expand_query(self, query: str) -> QueryExpansionResult:
        """
        Perform spelling normalization and controlled synonym expansion.
        Always preserves the original query and original terms.
        """
        if not query or not query.strip():
            return QueryExpansionResult(
                original_query=query,
                normalized_query="",
                spelling_corrections={},
                expanded_terms=[],
                expanded_query="",
                is_expanded=False,
            )

        # 1. Spelling correction
        normalized_query, corrections = self.correct_spelling(query)
        lower_normalized = normalized_query.lower()

        # 2. Synonym matching
        added_terms: List[str] = []
        seen_words: Set[str] = set(re.findall(r"\b\w+\b", lower_normalized))

        for key_phrase, synonyms_list in self.synonyms.items():
            pattern = rf"\b{re.escape(key_phrase)}\b"
            if re.search(pattern, lower_normalized):
                for syn in synonyms_list:
                    # Only add if terms are not already in query
                    syn_words = set(re.findall(r"\b\w+\b", syn.lower()))
                    if not syn_words.issubset(seen_words):
                        added_terms.append(syn)
                        seen_words.update(syn_words)
                        if len(added_terms) >= self.max_expansion_terms:
                            break
            if len(added_terms) >= self.max_expansion_terms:
                break

        is_expanded = len(added_terms) > 0 or len(corrections) > 0
        if added_terms:
            expanded_query_str = f"{normalized_query} " + " ".join(added_terms)
        else:
            expanded_query_str = normalized_query

        if is_expanded:
            logger.info(
                f"Query expanded: '{query}' -> '{expanded_query_str}' "
                f"(corrections: {corrections}, added_terms: {added_terms})"
            )

        return QueryExpansionResult(
            original_query=query,
            normalized_query=normalized_query,
            spelling_corrections=corrections,
            expanded_terms=added_terms,
            expanded_query=expanded_query_str,
            is_expanded=is_expanded,
        )
