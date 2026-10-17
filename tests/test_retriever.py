"""Automated unit tests for BM25, TF-IDF, and Hybrid retrieval engines."""

import pytest

from src.chunker import TextChunk
from src.retriever import BM25Retriever, HybridRetriever, TFIDFRetriever, build_retriever


@pytest.fixture
def sample_chunks():
    """Fixture providing a diverse set of synthetic text chunks."""
    return [
        TextChunk(
            chunk_id="chunk_01",
            doc_id="doc_syntax",
            source_title="Syntax Guide",
            source_filename="syntax.md",
            page_number=1,
            section="Indentation",
            text="A standard Python block uses 4 spaces per indentation level. Tabs and spaces should not be mixed.",
            start_char=0,
            end_char=99,
        ),
        TextChunk(
            chunk_id="chunk_02",
            doc_id="doc_structures",
            source_title="Data Structures",
            source_filename="structures.md",
            page_number=1,
            section="Lists",
            text="Python's list.sort() method sorts the items in place using the Timsort algorithm.",
            start_char=0,
            end_char=81,
        ),
        TextChunk(
            chunk_id="chunk_03",
            doc_id="doc_structures",
            source_title="Data Structures",
            source_filename="structures.md",
            page_number=1,
            section="Sets",
            text="An empty set must be created with set() because curly braces {} create an empty dictionary.",
            start_char=82,
            end_char=173,
        ),
        TextChunk(
            chunk_id="chunk_04",
            doc_id="doc_memory",
            source_title="Memory Guide",
            source_filename="memory.pdf",
            page_number=2,
            section="Garbage Collection",
            text="CPython uses reference counting as its primary memory management mechanism alongside cyclic GC.",
            start_char=0,
            end_char=96,
        ),
    ]


def test_bm25_retrieval(sample_chunks):
    """Verify BM25 retrieves correct passage and returns positive scores."""
    retriever = BM25Retriever()
    retriever.index(sample_chunks)

    results = retriever.retrieve("Timsort algorithm sorting", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "chunk_02"
    assert "Timsort" in results[0].text
    assert results[0].score > 0.0
    assert results[0].method == "bm25"


def test_tfidf_retrieval(sample_chunks):
    """Verify TF-IDF retrieves correct passage with cosine similarity."""
    retriever = TFIDFRetriever()
    retriever.index(sample_chunks)

    results = retriever.retrieve("reference counting memory management", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "chunk_04"
    assert "reference counting" in results[0].text
    assert 0.0 < results[0].score <= 1.0
    assert results[0].method == "tfidf"


def test_hybrid_retrieval_and_normalization(sample_chunks):
    """Verify Hybrid retriever combines engines and produces normalized scores [0.0, 1.0]."""
    retriever = HybridRetriever()
    retriever.index(sample_chunks)

    results = retriever.retrieve("empty set syntax", top_k=3)
    assert len(results) > 0
    assert results[0].chunk_id == "chunk_03"
    assert 0.0 <= results[0].score <= 1.0
    assert results[0].method == "hybrid"


def test_retriever_document_filter(sample_chunks):
    """Verify metadata filtering restricts results to allowed document IDs."""
    retriever = HybridRetriever()
    retriever.index(sample_chunks)

    # Filter to only doc_syntax
    results = retriever.retrieve("Python list sort empty set", top_k=3, doc_filter={"doc_syntax"})
    for r in results:
        assert r.doc_id == "doc_syntax"


def test_retriever_factory():
    """Verify factory returns appropriate retriever instances."""
    assert isinstance(build_retriever("bm25"), BM25Retriever)
    assert isinstance(build_retriever("tfidf"), TFIDFRetriever)
    assert isinstance(build_retriever("hybrid"), HybridRetriever)
    with pytest.raises(ValueError):
        build_retriever("invalid_retriever")
