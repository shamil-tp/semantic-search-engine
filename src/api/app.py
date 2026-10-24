"""Flask application factory for persisted lexical, dense, and hybrid search indexes."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path when running app.py directly via script path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from src.retrieval.dense import DenseSearchEngine
from src.retrieval.hybrid import HybridSearchEngine
from src.retrieval.lexical import LexicalSearchEngine

DEFAULT_MAX_TOP_K = 20
DEFAULT_MAX_REQUEST_BYTES = 16 * 1024


class JsonFormatter(logging.Formatter):
    """Small structured logger that does not require another dependency."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "time": self.formatTime(record, self.datefmt),
            }
        )


def _configure_logging(app: Flask) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    app.logger.handlers[:] = [handler]
    app.logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def _error(message: str, status: int, code: str) -> tuple[Response, int]:
    return jsonify({"error": {"code": code, "message": message}}), status


def _result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "score": result.score,
        "text": result.text,
        "source_path": result.source_path,
        "chunk_index": result.chunk_index,
        "metadata": result.metadata,
    }


def _parse_search_body(
    body: Mapping[str, Any], max_top_k: int
) -> tuple[str, int, dict[str, str] | None]:
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if len(query) > 2_000:
        raise ValueError("query must be at most 2000 characters.")

    top_k = body.get("top_k", min(5, max_top_k))
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= max_top_k
    ):
        raise ValueError(f"top_k must be an integer between 1 and {max_top_k}.")

    filters = body.get("filters")
    if filters is None:
        return query.strip(), top_k, None
    if not isinstance(filters, dict) or set(filters) - {"extension"}:
        raise ValueError("filters may contain only the extension field.")
    extension = filters.get("extension")
    if (
        not isinstance(extension, str)
        or not extension.startswith(".")
        or len(extension) > 16
    ):
        raise ValueError("filters.extension must be a file extension such as '.md'.")
    return query.strip(), top_k, {"extension": extension.lower()}


def create_app(
    config: Mapping[str, Any] | None = None,
    *,
    lexical_engine: LexicalSearchEngine | None = None,
    dense_engine: DenseSearchEngine | None = None,
) -> Flask:
    """Create an app and load configured persisted artifacts once per process."""
    template_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )
    app.config.from_mapping(
        APP_TITLE=os.getenv("APP_TITLE", "Semantic Search Engine"),
        LEXICAL_INDEX_DIR=os.getenv(
            "LEXICAL_INDEX_DIR", "artifacts/indices/lexical_sample"
        ),
        DENSE_INDEX_DIR=os.getenv("DENSE_INDEX_DIR", ""),
        SEARCH_MODE=os.getenv("SEARCH_MODE", "lexical"),
        MAX_TOP_K=int(os.getenv("MAX_TOP_K", str(DEFAULT_MAX_TOP_K))),
        MAX_CONTENT_LENGTH=int(
            os.getenv("MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES))
        ),
        ALLOWED_ORIGINS=os.getenv("ALLOWED_ORIGINS", ""),
    )
    if config:
        app.config.update(config)
    _configure_logging(app)

    # Load engines once at startup
    if lexical_engine is None and app.config["LEXICAL_INDEX_DIR"]:
        lex_path = Path(app.config["LEXICAL_INDEX_DIR"])
        if lex_path.exists():
            lexical_engine = LexicalSearchEngine.load(lex_path)

    if dense_engine is None and app.config["DENSE_INDEX_DIR"]:
        dense_path = Path(app.config["DENSE_INDEX_DIR"])
        if dense_path.exists():
            dense_engine = DenseSearchEngine.load(dense_path)

    mode = app.config["SEARCH_MODE"]
    if mode not in {"lexical", "dense", "hybrid"}:
        raise ValueError("SEARCH_MODE must be 'lexical', 'dense', or 'hybrid'.")

    if mode == "hybrid":
        if lexical_engine is None or dense_engine is None:
            raise ValueError("Both LEXICAL_INDEX_DIR and DENSE_INDEX_DIR are required when SEARCH_MODE=hybrid.")
        engine: Any = HybridSearchEngine(lexical_engine, dense_engine)
        catalog_chunks = lexical_engine.chunks
    elif mode == "dense":
        if dense_engine is None:
            raise ValueError("DENSE_INDEX_DIR is required when SEARCH_MODE=dense.")
        engine = dense_engine
        catalog_chunks = dense_engine.chunks
    else:
        if lexical_engine is None:
            raise ValueError("LEXICAL_INDEX_DIR is required when SEARCH_MODE=lexical.")
        engine = lexical_engine
        catalog_chunks = lexical_engine.chunks

    app.extensions["search_engine"] = engine
    app.extensions["documents"] = {
        chunk.document_id: chunk for chunk in catalog_chunks
    }
    app.extensions["search_mode"] = mode

    @app.after_request
    def add_cors(response: Response) -> Response:
        origin = request.headers.get("Origin")
        origins = {
            item.strip()
            for item in app.config["ALLOWED_ORIGINS"].split(",")
            if item.strip()
        }
        if origin and origin in origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_: RequestEntityTooLarge):
        return _error(
            "request body exceeds the configured limit.", 413, "request_too_large"
        )

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return _error(
            exc.description, exc.code or 500, exc.name.lower().replace(" ", "_")
        )

    @app.errorhandler(Exception)
    def unhandled_error(exc: Exception):
        app.logger.exception("unhandled_request_error: %s", exc)
        return _error("internal server error.", 500, "internal_error")

    # Front-end UI route
    @app.get("/")
    def index():
        return render_template(
            "index.html",
            app_title=app.config["APP_TITLE"],
            search_mode=app.extensions["search_mode"],
        )

    # Health check route
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "search_mode": app.extensions["search_mode"]})

    # Search handler function shared across /api/search and /api/v1/search
    def handle_search():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _error("request body must be a JSON object.", 400, "invalid_json")
        try:
            query, top_k, filters = _parse_search_body(body, app.config["MAX_TOP_K"])
        except ValueError as exc:
            return _error(str(exc), 400, "invalid_request")
        started = time.perf_counter()
        results = app.extensions["search_engine"].search(
            query, top_k=top_k, max_excerpt_len=500, metadata_filters=filters
        )
        latency_ms = round((time.perf_counter() - started) * 1_000, 3)
        app.logger.info(
            "search mode=%s results=%d latency_ms=%s", mode, len(results), latency_ms
        )
        return jsonify(
            {
                "query": query,
                "top_k": top_k,
                "count": len(results),
                "latency_ms": latency_ms,
                "results": [_result_to_dict(result) for result in results],
            }
        )

    @app.post("/api/search")
    def api_search():
        return handle_search()

    @app.post("/api/v1/search")
    def api_v1_search():
        return handle_search()

    @app.get("/api/v1/documents/<document_id>")
    def document(document_id: str):
        chunk = app.extensions["documents"].get(document_id)
        if chunk is None:
            return _error("document not found.", 404, "not_found")
        return jsonify(
            {
                "document_id": chunk.document_id,
                "source_path": chunk.source_path,
                "metadata": chunk.metadata,
            }
        )

    return app


def main() -> None:
    """CLI Entry point for Flask web application server."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}

    print(f"Starting Semantic Search Web Application on http://{host}:{port} (debug={debug})")
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
