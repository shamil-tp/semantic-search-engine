"""In-memory scoped session indexing for user-uploaded documents."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.ingestion.chunker import chunk_document
from src.ingestion.cleaner import TextCleaner
from src.ingestion.loader import RawDocument, extract_text_from_file_bytes
from src.ingestion.schemas import DocumentChunk, ProcessedDocument
from src.retrieval.dense import DEFAULT_DENSE_MODEL, DenseSearchEngine
from src.retrieval.lexical import LexicalSearchEngine
from src.retrieval.schemas import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class UploadedDocSession:
    """Represents an uploaded document indexed in-memory for scoped search."""

    session_id: str
    filename: str
    doc_id: str
    total_chunks: int
    char_length: int
    chunks: List[DocumentChunk]
    lexical_engine: LexicalSearchEngine
    dense_engine: DenseSearchEngine


class DocumentSessionManager:
    """
    Manages in-memory document search sessions.
    Allows users to upload a document (.pdf, .txt, .md, .json) and immediately
    perform high-precision semantic or lexical search strictly inside that document.
    """

    def __init__(self, dense_model_name: str = DEFAULT_DENSE_MODEL, device: str = "cpu"):
        self.dense_model_name = dense_model_name
        self.device = device
        # Cache single shared dense model instance to avoid reloading weights
        self._shared_dense = DenseSearchEngine(model_name=dense_model_name, device=device)
        self.sessions: Dict[str, UploadedDocSession] = {}

    def index_uploaded_file(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = 150,
        chunk_overlap: int = 25,
    ) -> UploadedDocSession:
        """
        Extract text, chunk, and index an uploaded document both lexically and semantically.
        """
        raw_text = extract_text_from_file_bytes(file_bytes, filename)
        if not raw_text.strip():
            raise ValueError(f"Uploaded file '{filename}' contains no extractable text.")

        session_id = uuid.uuid4().hex[:12]
        doc_id = f"upload_{session_id}"

        raw_doc = RawDocument(
            document_id=doc_id,
            source_path=filename,
            raw_text=raw_text,
            metadata={"file_name": filename, "is_uploaded": True},
        )

        cleaner = TextCleaner()
        processed_doc, clean_err = cleaner.clean_document(raw_doc)
        if not processed_doc:
            raise ValueError(f"Document became empty after cleaning ({clean_err}).")

        chunks = chunk_document(
            processed_doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise ValueError("No chunks could be produced from document.")

        # 1. Build Lexical Engine
        lex_engine = LexicalSearchEngine().build_index(chunks)

        # 2. Build Dense Semantic Engine using shared model weights
        dense_engine = DenseSearchEngine(model_name=self.dense_model_name, device=self.device)
        dense_engine._model = self._shared_dense._get_model()  # Reuse loaded model
        dense_engine.build_index(chunks)

        session = UploadedDocSession(
            session_id=session_id,
            filename=filename,
            doc_id=doc_id,
            total_chunks=len(chunks),
            char_length=len(processed_doc.text),
            chunks=chunks,
            lexical_engine=lex_engine,
            dense_engine=dense_engine,
        )

        self.sessions[session_id] = session
        logger.info(
            f"Created upload session {session_id} for '{filename}': "
            f"{len(chunks)} chunks, {len(processed_doc.text)} chars."
        )
        return session

    def search_session(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        search_mode: str = "semantic",  # 'semantic' or 'lexical'
    ) -> List[SearchResult]:
        """Search strictly inside an uploaded document."""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found or expired.")

        if search_mode == "lexical":
            return session.lexical_engine.search(query, top_k=top_k)
        else:
            # Semantic search
            return session.dense_engine.search(query, top_k=top_k)
