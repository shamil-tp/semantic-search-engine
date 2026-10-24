"""Test suite for document upload and in-document semantic/lexical search."""

import io
from pathlib import Path
import pytest
from src.api.app import create_app
from src.retrieval.lexical import LexicalSearchEngine


@pytest.fixture
def client():
    # Build minimal dummy lexical engine to pass app startup
    from src.ingestion.schemas import DocumentChunk
    engine = LexicalSearchEngine().build_index([
        DocumentChunk("c0", "d0", "path", 0, "Test base text", 0, 14)
    ])
    app = create_app({"SEARCH_MODE": "lexical"}, lexical_engine=engine)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_upload_and_search_txt_document(client):
    content = """
    Artificial Intelligence and Robotics.
    Robotic process automation utilizes software bots to automate repetitive business tasks.
    In contrast, autonomous robots rely on computer vision and neural networks to navigate physical spaces.
    Deep learning models allow robots to perform complex manipulation of delicate objects.
    """
    file_bytes = content.encode("utf-8")
    data = {
        "file": (io.BytesIO(file_bytes), "robotics_guide.txt")
    }

    # 1. Test Upload
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    res_data = res.get_json()
    assert res_data["status"] == "ok"
    assert "session_id" in res_data
    assert res_data["filename"] == "robotics_guide.txt"
    assert res_data["total_chunks"] >= 1

    session_id = res_data["session_id"]

    # 2. Test Semantic Search within document
    search_res = client.post("/api/search-doc", json={
        "session_id": session_id,
        "query": "neural networks and computer vision navigation",
        "top_k": 2,
        "mode": "semantic"
    })
    assert search_res.status_code == 200
    search_data = search_res.get_json()
    assert search_data["count"] >= 1
    assert "autonomous robots rely on computer vision" in search_data["results"][0]["text"]
    assert search_data["results"][0]["score"] > 0.0

    # 3. Test Lexical Search within document
    lex_res = client.post("/api/search-doc", json={
        "session_id": session_id,
        "query": "repetitive business tasks",
        "top_k": 2,
        "mode": "lexical"
    })
    assert lex_res.status_code == 200
    lex_data = lex_res.get_json()
    assert lex_data["count"] >= 1
    assert "repetitive business tasks" in lex_data["results"][0]["text"]


def test_upload_invalid_file(client):
    # Missing file
    res = client.post("/api/upload", data={})
    assert res.status_code == 400

    # Unsupported extension
    res = client.post("/api/upload", data={
        "file": (io.BytesIO(b"binary"), "image.png")
    }, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "unsupported_extension" in res.get_json()["error"]["code"]
