"""TF-IDF Matrix and Vectorizer Index Persistence."""

import hashlib
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ingestion.schemas import DocumentChunk

logger = logging.getLogger(__name__)

INDEX_METADATA_FILE = "index_metadata.json"
CHUNKS_CATALOG_FILE = "chunks.jsonl"
TFIDF_MATRIX_FILE = "tfidf_matrix.npz"
VECTORIZER_FILE = "vectorizer.pkl"


def compute_chunks_fingerprint(chunks: List[DocumentChunk]) -> str:
    """Compute deterministic SHA-256 fingerprint for indexed chunks list."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(str(len(chunk.text)).encode("utf-8"))
    return hasher.hexdigest()


def save_lexical_index(
    index_dir: str | Path,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: sp.spmatrix,
    chunks: List[DocumentChunk],
    config: Dict[str, Any],
) -> None:
    """
    Persist fitted vectorizer, sparse matrix, chunk records, and metadata.
    
    Security note: Vectorizer is stored via pickle. Only load index artifacts
    from trusted sources.
    """
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)

    fingerprint = compute_chunks_fingerprint(chunks)

    # 1. Save chunks catalog
    chunks_path = path / CHUNKS_CATALOG_FILE
    with open(chunks_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")

    # 2. Save sparse TF-IDF matrix (compressed scipy npz)
    matrix_path = path / TFIDF_MATRIX_FILE
    sp.save_npz(matrix_path, tfidf_matrix)

    # 3. Save vectorizer
    vec_path = path / VECTORIZER_FILE
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # 4. Save metadata
    meta = {
        "format_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "corpus_fingerprint": fingerprint,
        "chunk_count": len(chunks),
        "vocabulary_size": len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_ else 0,
        "matrix_shape": list(tfidf_matrix.shape),
        "config": config,
    }
    meta_path = path / INDEX_METADATA_FILE
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved lexical index to {path.resolve()}")


def load_lexical_index(
    index_dir: str | Path,
) -> Tuple[TfidfVectorizer, sp.csr_matrix, List[DocumentChunk], Dict[str, Any]]:
    """
    Load persisted index artifacts and validate integrity.
    
    Returns:
        (vectorizer, tfidf_matrix, chunks, metadata)
    """
    path = Path(index_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Index directory does not exist: {path}")

    meta_path = path / INDEX_METADATA_FILE
    chunks_path = path / CHUNKS_CATALOG_FILE
    matrix_path = path / TFIDF_MATRIX_FILE
    vec_path = path / VECTORIZER_FILE

    for req_file in [meta_path, chunks_path, matrix_path, vec_path]:
        if not req_file.exists():
            raise FileNotFoundError(f"Missing required index artifact: {req_file.name} in {path}")

    # Load metadata
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        raise ValueError(f"Corrupt index metadata: {exc}") from exc

    # Load chunks
    chunks: List[DocumentChunk] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            chunks.append(DocumentChunk(**data))

    # Verify chunk count
    if len(chunks) != metadata.get("chunk_count"):
        raise ValueError(
            f"Chunk count mismatch: expected {metadata.get('chunk_count')}, got {len(chunks)}"
        )

    # Load sparse matrix
    try:
        tfidf_matrix = sp.load_npz(matrix_path).tocsr()
    except Exception as exc:
        raise ValueError(f"Failed to load sparse matrix: {exc}") from exc

    if tfidf_matrix.shape[0] != len(chunks):
        raise ValueError(
            f"Matrix row count {tfidf_matrix.shape[0]} does not match chunk count {len(chunks)}"
        )

    # Load vectorizer
    try:
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to load vectorizer: {exc}") from exc

    return vectorizer, tfidf_matrix, chunks, metadata
