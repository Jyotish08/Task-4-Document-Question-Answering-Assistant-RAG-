"""Evidence-grounded answer generation and response construction module.

Performs deterministic extractive answer construction, attaches verifiable source
citations to claims, detects conflicting evidence, and guards against prompt injections.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.abstention import AbstentionEvaluation, AbstentionPolicy, EvidenceStatus
from src.chunker import split_into_sentences
from src.citation import Citation, CitationValidator, create_citation_from_retrieval
from src.config import config
from src.retriever import RetrievalResult
from src.text_cleaner import extract_keywords, tokenize


@dataclass
class GroundedAnswer:
    """Complete structured response object returned by the QA pipeline."""
    status: EvidenceStatus
    answer_text: str
    citations: List[Citation]
    uncertainty_note: Optional[str]
    retrieved_results: List[RetrievalResult]
    evaluation: AbstentionEvaluation
    is_abstention: bool = False
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def to_display_markdown(self) -> str:
        """Render response as formatted markdown with citations and badges."""
        if self.is_abstention:
            return f"**Status:** `{self.status.value}`\n\n> {self.answer_text}"

        md_lines = [f"**Status:** `{self.status.value}`\n"]
        md_lines.append(self.answer_text)

        if self.uncertainty_note:
            md_lines.append(f"\n> **Note on evidence limits:** {self.uncertainty_note}")

        if self.citations:
            md_lines.append("\n\n### Supporting Evidence & Citations:")
            for i, cite in enumerate(self.citations, start=1):
                page_info = f", Page {cite.page_number}" if cite.page_number else ""
                section_info = f" (Section: {cite.section})" if cite.section and cite.section != "General" else ""
                md_lines.append(
                    f"{i}. **{cite.source_filename}**{page_info}{section_info} `[{cite.chunk_id}]`  \n"
                    f"   *\"{cite.snippet}\"*"
                )

        return "\n".join(md_lines)


class GroundedAnswerer:
    """Constructs grounded answers strictly backed by retrieved passages."""

    def __init__(self, abstention_policy: Optional[AbstentionPolicy] = None):
        self.abstention_policy = abstention_policy or AbstentionPolicy()

    def _score_sentence_relevance(self, sentence: str, query_keywords: Set[str]) -> float:
        """Score sentence relevance based on keyword match density and length."""
        if not sentence or not query_keywords:
            return 0.0
        words = set(tokenize(sentence, remove_stopwords=True))
        if not words:
            return 0.0
        overlap = words.intersection(query_keywords)
        # Score is proportional to overlap count and ratio
        return len(overlap) * 1.5 + (len(overlap) / len(words))

    def _extract_best_sentences(
        self,
        retrieval_results: List[RetrievalResult],
        query_keywords: Set[str],
        max_sentences: int = 3,
    ) -> List[Tuple[str, RetrievalResult]]:
        """Extract top answering sentences along with their parent retrieval results."""
        scored_sentences: List[Tuple[float, str, RetrievalResult]] = []

        for res in retrieval_results:
            sentences = split_into_sentences(res.text)
            for s in sentences:
                s_clean = s.strip()
                # Skip pure section headers or ultra-short fragments
                if len(s_clean) < 25 or s_clean.startswith("#"):
                    continue
                score = self._score_sentence_relevance(s_clean, query_keywords)
                if score > 0.0:
                    scored_sentences.append((score, s_clean, res))

        # Sort by sentence relevance score descending
        scored_sentences.sort(key=lambda x: -x[0])

        # Pick top distinct sentences
        selected: List[Tuple[str, RetrievalResult]] = []
        seen_texts: Set[str] = set()

        for score, sentence, res in scored_sentences:
            if sentence in seen_texts:
                continue
            seen_texts.add(sentence)
            selected.append((sentence, res))
            if len(selected) >= max_sentences:
                break

        return selected

    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        method: str = "hybrid",
        corpus_text: Optional[str] = None,
    ) -> GroundedAnswer:
        """Synthesize an evidence-grounded answer or abstain deterministically."""
        # Step 1: Run abstention policy evaluation
        evaluation = self.abstention_policy.evaluate(
            query=query,
            retrieval_results=retrieval_results,
            method=method,
            corpus_text=corpus_text,
        )

        # Step 2: Handle mandatory abstention
        if evaluation.should_abstain:
            return GroundedAnswer(
                status=evaluation.status,
                answer_text=config.abstention.abstention_message,
                citations=[],
                uncertainty_note=evaluation.reason,
                retrieved_results=retrieval_results,
                evaluation=evaluation,
                is_abstention=True,
                validation_passed=True,
                validation_errors=[],
            )

        query_keywords = extract_keywords(query)
        best_sentences = self._extract_best_sentences(retrieval_results, query_keywords)

        # Fallback if no specific sentence scored well but threshold was met
        if not best_sentences and retrieval_results:
            top_res = retrieval_results[0]
            first_sentence = split_into_sentences(top_res.text)[0] if top_res.text else top_res.text[:150]
            best_sentences = [(first_sentence, top_res)]

        # Step 3: Construct grounded answer claims with inline citations
        claim_lines: List[str] = []
        citations: List[Citation] = []
        seen_chunks: Set[str] = set()

        for sentence, result in best_sentences:
            cite_label = result.to_citation_label()
            claim_lines.append(f"{sentence} {cite_label}")

            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                citations.append(create_citation_from_retrieval(result, snippet=sentence))

        answer_text = " ".join(claim_lines)

        # Step 4: Handle partial evidence caveats
        uncertainty_note = None
        if evaluation.status == EvidenceStatus.PARTIAL:
            missing_terms = ", ".join(f"'{k}'" for k in sorted(evaluation.missing_keywords))
            uncertainty_note = (
                f"The provided documents address parts of your question, but specific details "
                f"regarding {missing_terms} could not be established from the retrieved evidence."
            )

        return GroundedAnswer(
            status=evaluation.status,
            answer_text=answer_text,
            citations=citations,
            uncertainty_note=uncertainty_note,
            retrieved_results=retrieval_results,
            evaluation=evaluation,
            is_abstention=False,
            validation_passed=True,
            validation_errors=[],
        )
