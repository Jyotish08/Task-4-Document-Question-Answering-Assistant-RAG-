"""Automated unit tests for boundary-aware document chunking."""

import pytest

from src.chunker import DocumentChunker, split_into_sentences
from src.document_loader import Document, DocumentPage


def test_split_into_sentences_with_abbreviations():
    """Verify sentence splitter does not break on common abbreviations like e.g. and i.e."""
    text = "Python includes many built-in functions, e.g. print() and len(). Next sentence begins here."
    sentences = split_into_sentences(text)
    assert len(sentences) == 2
    assert "e.g. print()" in sentences[0]
    assert sentences[1] == "Next sentence begins here."


def test_chunker_invalid_overlap_raises():
    """Verify chunker rejects overlap >= chunk_size."""
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=200, chunk_overlap=250)


def test_chunker_short_document_single_chunk():
    """Verify a document shorter than chunk_size produces exactly one chunk."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    page = DocumentPage(page_number=1, text="Short document content.")
    chunks = chunker.chunk_page("doc1", "Short Doc", "short.txt", page)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc1_p1_c001"
    assert chunks[0].text == "Short document content."
    assert chunks[0].page_number == 1


def test_chunker_metadata_preservation():
    """Verify chunker preserves all required metadata attributes."""
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    text = (
        "# Indentation Rules\n\n"
        "Python relies on whitespace indentation rather than curly brackets to delimit code blocks. "
        "A standard Python block uses 4 spaces per indentation level. "
        "Mixing tabs and spaces in Python 3 raises an IndentationError immediately during parsing."
    )
    page = DocumentPage(page_number=2, text=text)
    chunks = chunker.chunk_page("doc_test", "Syntax Guide", "syntax.md", page)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.doc_id == "doc_test"
        assert c.source_title == "Syntax Guide"
        assert c.source_filename == "syntax.md"
        assert c.page_number == 2
        assert c.char_count > 0
        assert c.token_count > 0
        assert c.start_char >= 0
        assert c.end_char > c.start_char


def test_chunker_handles_unicode():
    """Verify chunker handles international characters and multi-byte Unicode cleanly."""
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
    unicode_text = "Python 支持多语言。こんにちは世界。Привет мир! Python is used worldwide."
    page = DocumentPage(page_number=1, text=unicode_text)
    chunks = chunker.chunk_page("doc_uni", "Unicode Test", "unicode.txt", page)

    assert len(chunks) >= 1
    combined = " ".join(c.text for c in chunks)
    assert "こんにちは" in combined
    assert "Привет" in combined


def test_chunker_empty_input():
    """Verify chunker returns empty list for empty page text."""
    chunker = DocumentChunker()
    page = DocumentPage(page_number=1, text="   \n\n   ")
    chunks = chunker.chunk_page("doc_empty", "Empty", "empty.txt", page)
    assert chunks == []
