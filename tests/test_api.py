from __future__ import annotations

from pathlib import Path
import pytest
from src.ingestion.schemas import DocumentChunk
from src.retrieval.lexical import LexicalSearchEngine
from src.api.app import create_app


@pytest.fixture
def test_engine():
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            source_path="/corpus/doc1.md",
            chunk_index=0,
            text="Semantic search with vector embeddings and information retrieval.",
            start_char=0,
            end_char=68,
            metadata={"file_name": "doc1.md", "extension": ".md"},
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="doc_2",
            source_path="/corpus/doc2.txt",
            chunk_index=0,
            text="Lexical search using TF-IDF matching.",
            start_char=0,
            end_char=37,
            metadata={"file_name": "doc2.txt", "extension": ".txt"},
        ),
    ]
    return LexicalSearchEngine().build_index(chunks)


@pytest.fixture
def client(test_engine):
    app = create_app(
        {"SEARCH_MODE": "lexical", "MAX_TOP_K": 10},
        lexical_engine=test_engine,
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_homepage_and_health(client):
    res_home = client.get("/")
    assert res_home.status_code == 200
    assert b"Semantic Search Engine" in res_home.data
    assert b"search-form" in res_home.data

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.get_json() == {"status": "ok", "search_mode": "lexical"}


def test_api_search_endpoint(client):
    res = client.post("/api/search", json={"query": "semantic search", "top_k": 3})
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["query"] == "semantic search"
    assert payload["count"] >= 1
    assert payload["results"][0]["document_id"] == "doc_1"


def test_search_rejects_invalid_requests(client):
    # Missing query
    res = client.post("/api/search", json={})
    assert res.status_code == 400

    # Negative top_k
    res = client.post("/api/search", json={"query": "test", "top_k": -1})
    assert res.status_code == 400

    # Oversized top_k
    res = client.post("/api/search", json={"query": "test", "top_k": 100})
    assert res.status_code == 400


def test_document_and_static_assets(client):
    res_doc = client.get("/api/v1/documents/doc_1")
    assert res_doc.status_code == 200
    assert res_doc.get_json()["document_id"] == "doc_1"

    res_missing = client.get("/api/v1/documents/non_existent")
    assert res_missing.status_code == 404

    # Static CSS
    res_css = client.get("/static/css/style.css")
    assert res_css.status_code == 200

    # Static JS
    res_js = client.get("/static/js/app.js")
    assert res_js.status_code == 200
