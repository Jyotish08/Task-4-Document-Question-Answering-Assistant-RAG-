"""Automated unit tests for citation construction, formatting, and validation."""

import pytest

from src.chunker import TextChunk
from src.citation import Citation, CitationValidator, create_citation_from_retrieval
from src.retriever import RetrievalResult


@pytest.fixture
def corpus_fixture():
    return [
        TextChunk(
            chunk_id="chunk_a",
            doc_id="doc_syntax",
            source_title="Syntax Guide",
            source_filename="syntax.md",
            page_number=1,
            section="Indentation",
            text="A standard Python block uses 4 spaces per indentation level.",
            start_char=0,
            end_char=60,
        )
    ]


def test_create_citation_from_retrieval():
    """Verify Citation constructor extracts correct properties from RetrievalResult."""
    result = RetrievalResult(
        chunk_id="chunk_a",
        doc_id="doc_syntax",
        source_name="syntax.md",
        page_number=1,
        section="Indentation",
        text="A standard Python block uses 4 spaces per indentation level.",
        score=0.98,
        method="hybrid",
        rank=1,
        citation_metadata={"source_title": "Syntax Guide"},
    )
    cite = create_citation_from_retrieval(result)
    assert cite.chunk_id == "chunk_a"
    assert cite.doc_id == "doc_syntax"
    assert cite.source_filename == "syntax.md"
    assert cite.page_number == 1
    assert cite.passage_text == "A standard Python block uses 4 spaces per indentation level."
    assert "syntax.md, Page 1" in cite.to_formatted_citation()


def test_citation_validator_valid(corpus_fixture):
    """Verify validator passes genuine citations."""
    cite = Citation(
        citation_id="cite_01",
        chunk_id="chunk_a",
        doc_id="doc_syntax",
        source_title="Syntax Guide",
        source_filename="syntax.md",
        page_number=1,
        section="Indentation",
        snippet="A standard Python block uses 4 spaces",
        score=0.95,
    )
    is_valid, errors = CitationValidator.validate_citations([cite], corpus_fixture)
    assert is_valid is True
    assert errors == []


def test_citation_validator_detects_nonexistent_chunk(corpus_fixture):
    """Verify validator catches fabricated chunk IDs."""
    fake_cite = Citation(
        citation_id="cite_fake",
        chunk_id="chunk_hallucinated",
        doc_id="doc_syntax",
        source_title="Syntax Guide",
        source_filename="syntax.md",
        page_number=1,
        section="Indentation",
        snippet="Some fabricated snippet",
        score=0.95,
    )
    is_valid, errors = CitationValidator.validate_citations([fake_cite], corpus_fixture)
    assert is_valid is False
    assert any("nonexistent chunk_id" in err for err in errors)


def test_citation_validator_detects_metadata_mismatch(corpus_fixture):
    """Verify validator catches incorrect doc_id or page_number."""
    mismatched_cite = Citation(
        citation_id="cite_02",
        chunk_id="chunk_a",
        doc_id="doc_WRONG",
        source_title="Syntax Guide",
        source_filename="syntax.md",
        page_number=99,  # Genuine chunk has page_number=1
        section="Indentation",
        snippet="A standard Python block uses 4 spaces",
        score=0.95,
    )
    is_valid, errors = CitationValidator.validate_citations([mismatched_cite], corpus_fixture)
    assert is_valid is False
    assert any("doc_id mismatch" in err for err in errors)
    assert any("page number mismatch" in err for err in errors)


def test_citation_validator_detects_hallucinated_snippet(corpus_fixture):
    """Verify validator catches fabricated snippet text."""
    hallucinated_cite = Citation(
        citation_id="cite_03",
        chunk_id="chunk_a",
        doc_id="doc_syntax",
        source_title="Syntax Guide",
        source_filename="syntax.md",
        page_number=1,
        section="Indentation",
        snippet="Quantum computers evaluate Python loops in polynomial time.",
        score=0.95,
    )
    is_valid, errors = CitationValidator.validate_citations([hallucinated_cite], corpus_fixture)
    assert is_valid is False
    assert any("cannot be found in target passage" in err for err in errors)
