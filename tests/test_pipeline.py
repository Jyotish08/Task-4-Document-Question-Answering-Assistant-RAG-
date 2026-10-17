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
