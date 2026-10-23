"""Comprehensive pytest suite for Phase 3 Query Understanding and Text ML."""

from pathlib import Path
import pytest

from src.ingestion.schemas import DocumentChunk
from src.query.expansion import (
    DEFAULT_SPELLING_CORRECTIONS,
    DEFAULT_SYNONYMS,
    QueryExpander,
    QueryExpansionResult,
)
from src.query.intent import (
    DEFAULT_TRAINING_DATA,
    IntentClassifier,
    IntentPrediction,
)
from src.query.ner import (
    DEFAULT_ENTITY_PATTERNS,
    EntityRecognizer,
    ExtractedEntity,
)
from src.query.pipeline import ProcessedQuery, QueryPipeline
from src.retrieval.lexical import LexicalSearchEngine


# ---------------------------------------------------------------------------
# 1. Intent Classifier Tests
# ---------------------------------------------------------------------------

def test_intent_classifier_training_and_prediction():
    classifier = IntentClassifier(c_param=1.0)
    classifier.train(DEFAULT_TRAINING_DATA)

    # Informational query
    pred1 = classifier.predict("what is semantic search")
    assert pred1.intent == "informational"
    assert pred1.confidence > 0.3
    assert "informational" in pred1.decision_scores

    # Technical spec query
    pred2 = classifier.predict("document_id and start_char schema specification")
    assert pred2.intent == "technical_spec"
    assert pred2.confidence > 0.3

    # Troubleshooting query
    pred3 = classifier.predict("handle UnicodeDecodeError on utf-8 file")
    assert pred3.intent == "troubleshooting"
    assert pred3.confidence > 0.3


def test_intent_classifier_empty_query_and_unfitted():
    # Unfitted
    unfitted = IntentClassifier()
    assert unfitted.predict("query").intent == "unknown"

    # Fitted but empty query
    fitted = IntentClassifier().train(DEFAULT_TRAINING_DATA)
    assert fitted.predict("").intent == "unknown"
    assert fitted.predict("   ").intent == "unknown"

    # OOV vocabulary
    oov_pred = fitted.predict("zyxwvu abracadabra")
    assert oov_pred.intent == "unknown"


def test_intent_classifier_save_and_load(tmp_path: Path):
    model_dir = tmp_path / "intent_model"
    clf = IntentClassifier().train(DEFAULT_TRAINING_DATA)
    clf.save(model_dir)

    loaded = IntentClassifier.load(model_dir)
    res_orig = clf.predict("how do vector embeddings work")
    res_loaded = loaded.predict("how do vector embeddings work")

    assert res_orig.intent == res_loaded.intent
    assert res_orig.confidence == pytest.approx(res_loaded.confidence, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. Named Entity Recognition & Metadata Filtering Tests
# ---------------------------------------------------------------------------

def test_entity_recognition_patterns():
    ner = EntityRecognizer()
    query = "Inspect document_id in doc1.md using recall@5 and tf-idf model"
    entities = ner.extract_entities(query)

    extracted_labels = {e.label: e.text for e in entities}
    assert "SCHEMA_FIELD" in extracted_labels
    assert extracted_labels["SCHEMA_FIELD"] == "document_id"

    assert "FILE_EXTENSION" in extracted_labels
    assert extracted_labels["FILE_EXTENSION"] == ".md"

    assert "METRIC" in extracted_labels
    assert extracted_labels["METRIC"] == "recall@5"

    assert "ALGORITHM" in extracted_labels
    assert extracted_labels["ALGORITHM"] == "tf-idf"


def test_metadata_filter_extraction():
    ner = EntityRecognizer()
    query = "Find notes in .md files regarding chunk_id a3090451a0a256d7_c1"
    filters = ner.extract_metadata_filters(query)

    assert filters.get("extension") == ".md"
    assert filters.get("chunk_id") == "a3090451a0a256d7_c1"


def test_lexical_search_with_metadata_filters():
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="doc_md",
            source_path="/dir/doc.md",
            chunk_index=0,
            text="General overview text in markdown format.",
            start_char=0,
            end_char=42,
            metadata={"extension": ".md"},
        ),
        DocumentChunk(
            chunk_id="c2",
            document_id="doc_txt",
            source_path="/dir/doc.txt",
            chunk_index=0,
            text="General overview text in plain text format.",
            start_char=0,
            end_char=43,
            metadata={"extension": ".txt"},
        ),
    ]
    engine = LexicalSearchEngine()
    engine.build_index(chunks)

    # Search with filter for extension .md
    results_md = engine.search("general overview", metadata_filters={"extension": ".md"})
    assert len(results_md) == 1
    assert results_md[0].document_id == "doc_md"

    # Search with filter for extension .txt
    results_txt = engine.search("general overview", metadata_filters={"extension": ".txt"})
    assert len(results_txt) == 1
    assert results_txt[0].document_id == "doc_txt"


# ---------------------------------------------------------------------------
# 3. Query Expansion & Spelling Correction Tests
# ---------------------------------------------------------------------------

def test_spelling_correction():
    expander = QueryExpander()
    text = "sematic search and chuncking preprocesing"
    corrected, fixes = expander.correct_spelling(text)

    assert corrected == "semantic search and chunking preprocessing"
    assert fixes["sematic"] == "semantic"
    assert fixes["chuncking"] == "chunking"
    assert fixes["preprocesing"] == "preprocessing"


def test_controlled_synonym_expansion():
    expander = QueryExpander(max_expansion_terms=2)
    # Query contains 'semantic search'
    res = expander.expand_query("semantic search overview")

    assert res.original_query == "semantic search overview"
    assert res.is_expanded is True
    # Must preserve the original terms in the expanded string
    assert "semantic search overview" in res.expanded_query
    # Must add controlled synonyms
    assert any(syn in res.expanded_terms for syn in ["dense retrieval", "concept search", "vector search"])


def test_expansion_does_not_mutate_unrelated_terms():
    expander = QueryExpander()
    res = expander.expand_query("unrelated astronomy galaxies query")
    assert res.is_expanded is False
    assert res.expanded_terms == []
    assert res.expanded_query == "unrelated astronomy galaxies query"


# ---------------------------------------------------------------------------
# 4. Query Pipeline & Default-Disabled Rule Verification
# ---------------------------------------------------------------------------

def test_query_pipeline_defaults_are_disabled():
    # Phase 3 Working Rule:
    # "No classifier or expansion stage is enabled by default without evidence of improvement."
    pipeline = QueryPipeline()
    assert pipeline.enable_intent is False
    assert pipeline.enable_ner is False
    assert pipeline.enable_expansion is False

    processed = pipeline.process_query("sematic search in .md files")
    assert processed.original_query == "sematic search in .md files"
    assert processed.effective_query == "sematic search in .md files"
    assert processed.intent is None
    assert processed.entities == []
    assert processed.metadata_filters == {}
    assert processed.expansion is None


def test_query_pipeline_explicitly_enabled():
    clf = IntentClassifier().train(DEFAULT_TRAINING_DATA)
    pipeline = QueryPipeline(
        intent_classifier=clf,
        enable_intent=True,
        enable_ner=True,
        enable_expansion=True,
    )

    processed = pipeline.process_query("sematic search in .md files")
    assert processed.intent is not None
    assert processed.intent.intent == "informational"
    assert len(processed.entities) > 0
    assert processed.metadata_filters.get("extension") == ".md"
    assert processed.expansion is not None
    assert "semantic" in processed.effective_query
