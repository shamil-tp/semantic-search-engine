"""API integration tests using an in-memory lexical index."""

import pytest

flask = pytest.importorskip("flask")

from src.api.app import create_app
from src.ingestion.schemas import DocumentChunk
from src.retrieval.lexical import LexicalSearchEngine


@pytest.fixture()
def client():
    engine = LexicalSearchEngine().build_index(
        [
            DocumentChunk(
                "c1",
                "doc1",
                "/public/guide.md",
                0,
                "Semantic search uses vector embeddings.",
                0,
                37,
                {"extension": ".md"},
            ),
            DocumentChunk(
                "c2",
                "doc2",
                "/public/notes.txt",
                0,
                "Unicode normalization prepares text.",
                0,
                33,
                {"extension": ".txt"},
            ),
        ]
    )
    app = create_app({"TESTING": True, "MAX_TOP_K": 3}, lexical_engine=engine)
    return app.test_client()


def test_health_search_and_document(client):
    assert client.get("/health").get_json()["status"] == "ok"
    response = client.post(
        "/api/v1/search", json={"query": "vector embeddings", "top_k": 1}
    )
    assert response.status_code == 200
    assert response.get_json()["results"][0]["document_id"] == "doc1"
    assert client.get("/api/v1/documents/doc1").status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": " "},
        {"query": 2},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "filters": {"topic": "x"}},
    ],
)
def test_search_rejects_invalid_requests(client, payload):
    response = client.post("/api/v1/search", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_search_filter_and_not_found(client):
    response = client.post(
        "/api/v1/search", json={"query": "text", "filters": {"extension": ".txt"}}
    )
    assert response.status_code == 200
    assert response.get_json()["results"][0]["document_id"] == "doc2"
    assert client.get("/api/v1/documents/missing").status_code == 404
