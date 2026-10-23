"""Comprehensive pytest suite for Phase 1 Data Ingestion and Preparation."""

import json
from pathlib import Path
import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.cleaner import TextCleaner, compute_content_hash, normalize_text
from src.ingestion.loader import (
    DEFAULT_SUPPORTED_EXTENSIONS,
    discover_files,
    generate_document_id,
    load_file,
    RawDocument,
)
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.schemas import DocumentChunk, ProcessedDocument
from src.ingestion.statistics import calculate_corpus_statistics


# ---------------------------------------------------------------------------
# 1. Loader & Discovery Tests
# ---------------------------------------------------------------------------

def test_discover_files_recursive_and_supported_extensions(tmp_path: Path):
    # Setup nested directories with supported and unsupported files
    sub1 = tmp_path / "sub1"
    sub2 = tmp_path / "sub2" / "nested"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    (tmp_path / "doc1.txt").write_text("doc1", encoding="utf-8")
    (sub1 / "doc2.md").write_text("doc2", encoding="utf-8")
    (sub2 / "doc3.TXT").write_text("doc3", encoding="utf-8")
    (sub2 / "ignore.pdf").write_text("pdf", encoding="utf-8")
    (sub2 / "ignore.py").write_text("code", encoding="utf-8")

    discovered = discover_files(tmp_path)
    discovered_names = [p.name.lower() for p in discovered]

    assert len(discovered) == 3
    assert "doc1.txt" in discovered_names
    assert "doc2.md" in discovered_names
    assert "doc3.txt" in discovered_names
    assert "ignore.pdf" not in discovered_names


def test_discover_files_non_existent_dir(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist"
    assert discover_files(non_existent) == []


def test_load_file_utf8_success(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    content = "Hello, world! 🚀 Testing Unicode: café, résumé, ñ."
    test_file.write_text(content, encoding="utf-8")

    doc, err = load_file(test_file)
    assert err is None
    assert doc is not None
    assert doc.raw_text == content
    assert doc.metadata["file_name"] == "sample.txt"
    assert doc.metadata["extension"] == ".txt"
    assert doc.metadata["file_size_bytes"] > 0


def test_load_file_malformed_encoding(tmp_path: Path):
    test_file = tmp_path / "malformed.txt"
    # Write invalid UTF-8 bytes (e.g. 0x80)
    with open(test_file, "wb") as f:
        f.write(b"Valid ASCII \x80 Invalid UTF-8 \xfe")

    doc, err = load_file(test_file)
    assert doc is None
    assert err is not None
    assert "UnicodeDecodeError" in err


def test_load_non_existent_file(tmp_path: Path):
    test_file = tmp_path / "missing.txt"
    doc, err = load_file(test_file)
    assert doc is None
    assert "does not exist" in err


def test_deterministic_document_id(tmp_path: Path):
    file_a = tmp_path / "doc.txt"
    file_a.write_text("content", encoding="utf-8")

    id1 = generate_document_id(str(file_a))
    id2 = generate_document_id(str(file_a))
    assert id1 == id2
    assert len(id1) == 16


# ---------------------------------------------------------------------------
# 2. Text Normalization and Cleaning Tests
# ---------------------------------------------------------------------------

def test_normalize_text_unicode_and_line_endings():
    # Decomposed e + accent -> NFC
    decomposed = "re\u0301sume\u0301"
    text_with_crlf = f"Header\r\n\r\nParagraph 1: {decomposed}   with   spaces.\r\n\r\n\r\n\r\nParagraph 2."
    
    normalized = normalize_text(text_with_crlf)
    assert "\u00e9" in normalized  # NFC form of é
    assert "\r" not in normalized
    # Max consecutive newlines is 2 (paragraph break)
    assert "\n\n\n" not in normalized
    assert "Paragraph 1: résumé with spaces." in normalized


def test_cleaner_empty_document():
    cleaner = TextCleaner()
    raw = RawDocument(
        document_id="d1",
        source_path="/path/empty.txt",
        raw_text="   \n\n  \t  \n  ",
    )
    processed, status = cleaner.clean_document(raw)
    assert processed is None
    assert status == "empty"


def test_cleaner_duplicate_detection():
    cleaner = TextCleaner()
    raw1 = RawDocument(
        document_id="d1",
        source_path="/path/doc1.txt",
        raw_text="This is unique content.",
    )
    raw2 = RawDocument(
        document_id="d2",
        source_path="/path/doc2.txt",
        raw_text="This is unique content.",
    )
    # Different spacing but normalized to same text
    raw3 = RawDocument(
        document_id="d3",
        source_path="/path/doc3.txt",
        raw_text="  This   is unique   content.  \n",
    )

    doc1, status1 = cleaner.clean_document(raw1)
    assert doc1 is not None
    assert status1 is None

    doc2, status2 = cleaner.clean_document(raw2)
    assert doc2 is None
    assert status2 == "duplicate"

    doc3, status3 = cleaner.clean_document(raw3)
    assert doc3 is None
    assert status3 == "duplicate"


def test_preserves_punctuation_and_numbers():
    cleaner = TextCleaner()
    raw = RawDocument(
        document_id="d1",
        source_path="/path/doc1.txt",
        raw_text="Section 3.14: In 2026, the AI index was $100. Is this 100% accurate?",
    )
    doc, status = cleaner.clean_document(raw)
    assert doc is not None
    assert "Section 3.14: In 2026, the AI index was $100. Is this 100% accurate?" in doc.text


# ---------------------------------------------------------------------------
# 3. Document Chunking Tests
# ---------------------------------------------------------------------------

def test_chunking_short_document():
    doc = ProcessedDocument(
        document_id="doc_short",
        source_path="/path/short.txt",
        content_hash="abc",
        text="A brief document containing just a few words.",
    )
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "doc_short_c0"
    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == len(doc.text)
    assert chunk.text == doc.text


def test_chunking_long_document_and_overlap():
    words = [f"word{i}" for i in range(120)]
    full_text = " ".join(words)

    doc = ProcessedDocument(
        document_id="doc_long",
        source_path="/path/long.txt",
        content_hash="hash123",
        text=full_text,
    )
    # chunk_size=50, chunk_overlap=10 -> step = 40 words
    # Chunk 0: 0..50
    # Chunk 1: 40..90
    # Chunk 2: 80..120
    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 3

    # Check IDs and offsets
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"doc_long_c{idx}"
        assert c.chunk_index == idx
        assert c.document_id == "doc_long"
        assert c.source_path == "/path/long.txt"
        # Validate character slice against original text
        assert doc.text[c.start_char : c.end_char] == c.text

    # Verify overlap content
    chunk0_words = chunks[0].text.split()
    chunk1_words = chunks[1].text.split()
    assert len(chunk0_words) == 50
    assert len(chunk1_words) == 50
    # Overlap of 10 words: chunk0[-10:] == chunk1[:10]
    assert chunk0_words[-10:] == chunk1_words[:10]


def test_chunking_invalid_parameters():
    doc = ProcessedDocument("d1", "p", "h", "Some text")
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_document(doc, chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        chunk_document(doc, chunk_size=10, chunk_overlap=-1)

    with pytest.raises(ValueError, match="strictly less than"):
        chunk_document(doc, chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError, match="strictly less than"):
        chunk_document(doc, chunk_size=10, chunk_overlap=15)


# ---------------------------------------------------------------------------
# 4. Statistics Tests
# ---------------------------------------------------------------------------

def test_empty_corpus_statistics():
    stats = calculate_corpus_statistics(
        discovered_count=0,
        skipped_unsupported_count=0,
        failed_count=0,
        empty_count=0,
        duplicate_count=0,
        processed_documents=[],
        chunks=[],
    )
    assert stats["file_summary"]["files_discovered"] == 0
    assert stats["file_summary"]["unique_processed_documents"] == 0
    assert stats["file_summary"]["generated_chunks"] == 0
    assert stats["document_statistics"]["mean_words"] == 0.0
    assert stats["chunk_statistics"]["median_chars"] == 0.0


# ---------------------------------------------------------------------------
# 5. Full Pipeline End-to-End & Reproducibility Tests
# ---------------------------------------------------------------------------

def test_pipeline_reproducibility_and_deterministic_output(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir_1 = tmp_path / "out1"
    output_dir_2 = tmp_path / "out2"
    input_dir.mkdir()

    # Create synthetic documents
    doc1 = input_dir / "alpha.txt"
    doc1.write_text("First document content with some words to chunk.\n\nSecond paragraph.", encoding="utf-8")

    doc2 = input_dir / "beta.md"
    doc2.write_text("# Markdown Title\nHere is information about semantic search.", encoding="utf-8")

    # Duplicate of doc1
    doc3 = input_dir / "alpha_copy.txt"
    doc3.write_text("First document content with some words to chunk.\n\nSecond paragraph.", encoding="utf-8")

    # Empty file
    empty_doc = input_dir / "empty.txt"
    empty_doc.write_text("   \n   ", encoding="utf-8")

    # Run 1
    p1 = IngestionPipeline(chunk_size=10, chunk_overlap=2)
    res1 = p1.run(input_dir=input_dir, output_dir=output_dir_1)

    # Run 2
    p2 = IngestionPipeline(chunk_size=10, chunk_overlap=2)
    res2 = p2.run(input_dir=input_dir, output_dir=output_dir_2)

    # Validate statistics
    stats1 = res1["statistics"]
    assert stats1["file_summary"]["files_discovered"] == 4
    assert stats1["file_summary"]["files_empty"] == 1
    assert stats1["file_summary"]["files_duplicate"] == 1
    assert stats1["file_summary"]["unique_processed_documents"] == 2
    assert stats1["file_summary"]["generated_chunks"] > 0

    # Verify deterministic output files
    docs1_lines = (output_dir_1 / "documents.jsonl").read_text(encoding="utf-8").strip().splitlines()
    docs2_lines = (output_dir_2 / "documents.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert docs1_lines == docs2_lines

    chunks1_lines = (output_dir_1 / "chunks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    chunks2_lines = (output_dir_2 / "chunks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert chunks1_lines == chunks2_lines

    stats1_json = (output_dir_1 / "statistics.json").read_text(encoding="utf-8")
    stats2_json = (output_dir_2 / "statistics.json").read_text(encoding="utf-8")
    assert stats1_json == stats2_json
