"""Automated integration tests for the complete Document QA Pipeline."""

from pathlib import Path
import pytest

from src.abstention import EvidenceStatus
from src.config import config
from src.pipeline import DocumentQAPipeline


@pytest.fixture
def test_pipeline(tmp_path: Path):
    """Fixture providing a temporary pipeline with sample documents."""
    pipeline = DocumentQAPipeline(
        retrieval_method="hybrid",
        storage_dir=tmp_path / "storage",
    )
    # Ingest the real sample docs directory
    pipeline.ingest_directory(config.paths.sample_docs_dir)
    return pipeline


def test_pipeline_ingest_and_query_supported(test_pipeline):
    """Test full pipeline query on a supported factual question."""
    answer = test_pipeline.query("How many spaces per indentation level are standard in a Python block?")

    assert answer.status == EvidenceStatus.SUPPORTED
    assert answer.is_abstention is False
    assert "4 spaces" in answer.answer_text
    assert len(answer.citations) >= 1
    assert answer.validation_passed is True


def test_pipeline_query_unsupported_abstention(test_pipeline):
    """Test full pipeline query on an unsupported out-of-domain question."""
    answer = test_pipeline.query("What is the capital of Australia?")

    assert answer.status == EvidenceStatus.UNSUPPORTED
    assert answer.is_abstention is True
    assert answer.answer_text == "Information not found in provided document."
    assert len(answer.citations) == 0


def test_pipeline_query_partial_evidence(test_pipeline):
    """Test pipeline handling of partially supported query."""
    answer = test_pipeline.query("Does Python support complex numbers, and what is the exact formula for quantum spin matrices in Python?")

    assert answer.status == EvidenceStatus.PARTIAL
    assert answer.is_abstention is False
    assert "complex" in answer.answer_text.lower()
    assert answer.uncertainty_note is not None
    assert len(answer.citations) >= 1


def test_pipeline_remove_document_and_rebuild(test_pipeline):
    """Test removing a document updates chunk registry and index."""
    initial_chunks = len(test_pipeline.chunks)
    assert initial_chunks > 0

    first_doc_id = list(test_pipeline.documents.keys())[0]
    removed = test_pipeline.remove_document(first_doc_id)
    assert removed is True
    assert first_doc_id not in test_pipeline.documents
    assert len(test_pipeline.chunks) < initial_chunks


def test_pipeline_state_persistence(tmp_path: Path):
    """Test saving and restoring pipeline state to/from disk."""
    storage_dir = tmp_path / "state_store"
    storage_dir.mkdir()

    p1 = DocumentQAPipeline(storage_dir=storage_dir)
    p1.ingest_directory(config.paths.sample_docs_dir)
    doc_count_1 = len(p1.documents)
    chunk_count_1 = len(p1.chunks)

    # Initialize a new pipeline instance pointing to the same storage
    p2 = DocumentQAPipeline(storage_dir=storage_dir)
    loaded = p2.load_state()
    assert loaded is True
    assert len(p2.documents) == doc_count_1
    assert len(p2.chunks) == chunk_count_1

    # Verify query works on restored pipeline
    ans = p2.query("What sorting algorithm is used by Python's list.sort() method?")
    assert ans.status in (EvidenceStatus.SUPPORTED, EvidenceStatus.PARTIAL)
    assert "Timsort" in ans.answer_text


def test_pipeline_citation_contains_full_supporting_passage(test_pipeline):
    """Verify that every ordinary answer contains an explicit supporting source passage."""
    answer = test_pipeline.query("How many spaces per indentation level are standard in a Python block?")
    assert answer.status == EvidenceStatus.SUPPORTED
    assert len(answer.citations) >= 1
    top_cite = answer.citations[0]
    assert top_cite.passage_text != ""
    assert "4 spaces" in top_cite.passage_text
    assert top_cite.snippet in top_cite.passage_text


def test_pipeline_sequential_multi_document_queries(test_pipeline):
    """Verify sequential queries across distinct documents do not return stale citations."""
    # Query 1: Targets document 01 (syntax)
    ans1 = test_pipeline.query("How many spaces per indentation level are standard in a Python block?")
    assert ans1.status == EvidenceStatus.SUPPORTED
    for c in ans1.citations:
        assert c.source_filename == "01_python_syntax_and_types.md"

    # Query 2: Targets document 02 (structures)
    ans2 = test_pipeline.query("What sorting algorithm is used by Python's list.sort() method?")
    assert ans2.status == EvidenceStatus.SUPPORTED
    for c in ans2.citations:
        assert c.source_filename == "02_data_structures.md"
        assert c.source_filename != "01_python_syntax_and_types.md"

    # Query 3: Targets document 05 (modules)
    ans3 = test_pipeline.query("What global variable holds a module's name as a string?")
    assert ans3.status == EvidenceStatus.SUPPORTED
    for c in ans3.citations:
        assert c.source_filename == "05_modules_and_packages.md"


def test_pipeline_paraphrased_query_supported(test_pipeline):
    """Verify that a paraphrased query successfully retrieves grounded evidence."""
    ans = test_pipeline.query("Which sorting algorithm is used by Python to sort lists in place?")
    assert ans.status == EvidenceStatus.SUPPORTED
    assert "Timsort" in ans.answer_text
    assert len(ans.citations) >= 1


def test_pipeline_index_fingerprint_and_freshness(tmp_path: Path):
    """Verify corpus fingerprint calculation and index freshness detection."""
    storage_dir = tmp_path / "fingerprint_store"
    p = DocumentQAPipeline(storage_dir=storage_dir)
    p.ingest_directory(config.paths.sample_docs_dir)

    fp = p.compute_corpus_fingerprint()
    assert fp != ""
    assert len(fp) == 16

    # Verify freshness check
    assert p.is_index_fresh(config.paths.sample_docs_dir) is True

    # If an extra file is introduced, freshness check reports False
    fake_doc = config.paths.sample_docs_dir / "temp_fake_doc.txt"
    try:
        fake_doc.write_text("Temporary test file", encoding="utf-8")
        assert p.is_index_fresh(config.paths.sample_docs_dir) is False
    finally:
        if fake_doc.exists():
            fake_doc.unlink()
