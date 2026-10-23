"""Document loading and recursive directory discovery."""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class RawDocument:
    """Raw uncleaned document loaded from file system."""

    document_id: str
    source_path: str
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_document_id(source_path: str) -> str:
    """
    Generate deterministic document ID from normalized source path.
    Uses SHA-256 hash of normalized path relative/absolute string representation.
    """
    normalized_path = Path(source_path).resolve().as_posix()
    return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:16]


def discover_files(
    directory: str | Path,
    supported_extensions: Optional[Set[str]] = None,
) -> List[Path]:
    """
    Recursively discover files with supported extensions in the given directory.
    Returns sorted list of Paths for deterministic processing order.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return []

    if not dir_path.is_dir():
        logger.warning(f"Path is not a directory: {dir_path}")
        return []

    if supported_extensions is None:
        supported_extensions = DEFAULT_SUPPORTED_EXTENSIONS
    else:
        supported_extensions = {ext.lower() for ext in supported_extensions}

    discovered: List[Path] = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in supported_extensions:
                discovered.append(file_path)

    discovered.sort()
    return discovered


def load_file(file_path: str | Path) -> Tuple[Optional[RawDocument], Optional[str]]:
    """
    Safely load a single text/markdown file.
    Handles UTF-8 decoding, missing files, permission errors, and malformed files.
    
    Returns:
        (RawDocument, None) on success
        (None, error_reason) on failure
    """
    path = Path(file_path)
    if not path.is_file():
        return None, f"File does not exist or is not a regular file: {path}"

    try:
        # Check file size metadata
        stat = path.stat()
        file_size = stat.st_size
    except Exception as exc:
        return None, f"Failed to access file stat for {path}: {exc}"

    # Read UTF-8 safely
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError as exc:
        # Fallback attempted or log unreadable malformed encoding error
        logger.warning(f"UTF-8 decode failed for {path}: {exc}")
        return None, f"UnicodeDecodeError: {exc}"
    except Exception as exc:
        logger.warning(f"Failed to read file {path}: {exc}")
        return None, f"ReadError: {exc}"

    doc_id = generate_document_id(str(path))
    metadata: Dict[str, Any] = {
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "file_size_bytes": file_size,
        "created_time": stat.st_ctime,
        "modified_time": stat.st_mtime,
    }

    return RawDocument(
        document_id=doc_id,
        source_path=path.resolve().as_posix(),
        raw_text=content,
        metadata=metadata,
    ), None
