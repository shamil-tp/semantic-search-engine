"""Query pipeline orchestrator integrating intent, NER, normalization, and expansion."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.query.expansion import QueryExpander, QueryExpansionResult
from src.query.intent import IntentClassifier, IntentPrediction
from src.query.ner import EntityRecognizer, ExtractedEntity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedQuery:
    """Consolidated representation of an analyzed search query."""

    original_query: str
    effective_query: str
    intent: Optional[IntentPrediction] = None
    entities: List[ExtractedEntity] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    expansion: Optional[QueryExpansionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "effective_query": self.effective_query,
            "intent": self.intent.to_dict() if self.intent else None,
            "entities": [e.to_dict() for e in self.entities],
            "metadata_filters": self.metadata_filters,
            "expansion": self.expansion.to_dict() if self.expansion else None,
        }


class QueryPipeline:
    """
    Orchestrates query understanding stages.
    
    CRITICAL WORKING RULE (Phase 3 exit criterion):
    No classifier or expansion stage is enabled by default without evidence of improvement.
    enable_intent, enable_ner, and enable_expansion default to FALSE.
    """

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        entity_recognizer: Optional[EntityRecognizer] = None,
        query_expander: Optional[QueryExpander] = None,
        enable_intent: bool = False,
        enable_ner: bool = False,
        enable_expansion: bool = False,
    ):
        self.intent_classifier = intent_classifier
        self.entity_recognizer = entity_recognizer or EntityRecognizer()
        self.query_expander = query_expander or QueryExpander()

        self.enable_intent = enable_intent
        self.enable_ner = enable_ner
        self.enable_expansion = enable_expansion

    def process_query(
        self,
        query: str,
        override_enable_intent: Optional[bool] = None,
        override_enable_ner: Optional[bool] = None,
        override_enable_expansion: Optional[bool] = None,
    ) -> ProcessedQuery:
        """
        Process query through enabled query understanding modules.
        Preserves original query and logs all terms.
        """
        do_intent = self.enable_intent if override_enable_intent is None else override_enable_intent
        do_ner = self.enable_ner if override_enable_ner is None else override_enable_ner
        do_expansion = self.enable_expansion if override_enable_expansion is None else override_enable_expansion

        raw_query = query or ""
        effective_query = raw_query

        # 1. Intent Classification
        intent_pred: Optional[IntentPrediction] = None
        if do_intent and self.intent_classifier is not None:
            intent_pred = self.intent_classifier.predict(raw_query)

        # 2. NER & Metadata Filter Extraction
        entities: List[ExtractedEntity] = []
        filters: Dict[str, Any] = {}
        if do_ner:
            entities = self.entity_recognizer.extract_entities(raw_query)
            filters = self.entity_recognizer.extract_metadata_filters(raw_query)

        # 3. Query Expansion & Spelling Correction
        expansion_result: Optional[QueryExpansionResult] = None
        if do_expansion:
            expansion_result = self.query_expander.expand_query(raw_query)
            if expansion_result.expanded_query:
                effective_query = expansion_result.expanded_query

        return ProcessedQuery(
            original_query=raw_query,
            effective_query=effective_query,
            intent=intent_pred,
            entities=entities,
            metadata_filters=filters,
            expansion=expansion_result,
        )
