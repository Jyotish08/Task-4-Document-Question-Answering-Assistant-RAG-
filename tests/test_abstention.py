"""Automated unit tests for the deterministic abstention policy."""

import pytest

from src.abstention import AbstentionPolicy, EvidenceStatus
from src.config import config
from src.retriever import RetrievalResult


@pytest.fixture
def policy():
    return AbstentionPolicy()


def test_mandatory_abstention_phrase_exact():
    """Verify exact mandatory abstention string required by internship spec."""
    assert config.abstention.abstention_message == "Information not found in provided document."


def test_empty_query_abstains(policy):
    """Verify empty or whitespace-only queries trigger abstention."""
    eval_res = policy.evaluate("   ", [])
    assert eval_res.should_abstain is True
    assert eval_res.status == EvidenceStatus.EMPTY_QUERY


def test_empty_retrieval_abstains(policy):
    """Verify that when no chunks match, policy abstains."""
    eval_res = policy.evaluate("What is the capital of Australia?", [])
    assert eval_res.should_abstain is True
    assert eval_res.status == EvidenceStatus.UNSUPPORTED


def test_below_threshold_score_abstains(policy):
    """Verify query with top score below calibrated threshold is rejected."""
    weak_result = RetrievalResult(
        chunk_id="c1",
        doc_id="d1",
        source_name="doc.txt",
        page_number=1,
        section="General",
        text="Unrelated text about something else.",
        score=0.15,  # Below hybrid threshold of 0.40
        method="hybrid",
        rank=1,
    )
    eval_res = policy.evaluate("What is Python's Global Interpreter Lock?", [weak_result], method="hybrid")
    assert eval_res.should_abstain is True
    assert eval_res.status == EvidenceStatus.UNSUPPORTED


def test_out_of_domain_query_abstains(policy):
    """Verify queries containing non-existent domain concepts abstain."""
    sample_res = RetrievalResult(
        chunk_id="c1",
        doc_id="d1",
        source_name="python.txt",
        page_number=1,
        section="General",
        text="Python syntax relies on whitespace indentation.",
        score=0.50,
        method="hybrid",
        rank=1,
    )
    corpus_text = "Python syntax relies on whitespace indentation."
    eval_res = policy.evaluate(
        query="What is the recipe for baking sourdough bread?",
        retrieval_results=[sample_res],
        method="hybrid",
        corpus_text=corpus_text,
    )
    assert eval_res.should_abstain is True
    assert eval_res.status == EvidenceStatus.UNSUPPORTED
    assert eval_res.abstention_message == "Information not found in provided document."


def test_supported_query_passes(policy):
    """Verify clear grounded question passes without abstention."""
    supported_res = RetrievalResult(
        chunk_id="c1",
        doc_id="d1",
        source_name="python.txt",
        page_number=1,
        section="Syntax",
        text="A standard Python block uses 4 spaces per indentation level.",
        score=0.98,
        method="hybrid",
        rank=1,
    )
    corpus_text = "A standard Python block uses 4 spaces per indentation level."
    eval_res = policy.evaluate(
        query="How many spaces per indentation level are standard in a Python block?",
        retrieval_results=[supported_res],
        method="hybrid",
        corpus_text=corpus_text,
    )
    assert eval_res.should_abstain is False
    assert eval_res.status == EvidenceStatus.SUPPORTED


def test_same_term_distractor_query_abstains(policy):
    """Verify query containing common corpus entities but asking for an absent fact strictly abstains."""
    sample_res = RetrievalResult(
        chunk_id="c1",
        doc_id="d1",
        source_name="syntax.md",
        page_number=1,
        section="Syntax",
        text="A standard Python block uses 4 spaces per indentation level. Python is dynamically typed.",
        score=0.45,
        method="hybrid",
        rank=1,
    )
    corpus_text = "A standard Python block uses 4 spaces per indentation level. Python is dynamically typed."
    eval_res = policy.evaluate(
        query="What quantum cryptography algorithm is used by Python code blocks?",
        retrieval_results=[sample_res],
        method="hybrid",
        corpus_text=corpus_text,
    )
    assert eval_res.should_abstain is True
    assert eval_res.status == EvidenceStatus.UNSUPPORTED
    assert eval_res.abstention_message == "Information not found in provided document."


def test_high_ranked_irrelevant_passage_does_not_become_answer(policy):
    """Verify that a high-ranked passage without sufficient evidence does not produce a false positive."""
    distractor_res = RetrievalResult(
        chunk_id="c_distractor",
        doc_id="doc_distractor",
        source_name="structures.md",
        page_number=1,
        section="Lists",
        text="Python lists are mutable sequences. You can append items using list.append().",
        score=0.42,
        method="hybrid",
        rank=1,
    )
    corpus_text = "Python lists are mutable sequences. You can append items using list.append()."
    eval_res = policy.evaluate(
        query="What is the default TCP socket port for Python list sorting?",
        retrieval_results=[distractor_res],
        method="hybrid",
        corpus_text=corpus_text,
    )
    assert eval_res.should_abstain is True
    assert eval_res.status == EvidenceStatus.UNSUPPORTED
    assert eval_res.abstention_message == "Information not found in provided document."
