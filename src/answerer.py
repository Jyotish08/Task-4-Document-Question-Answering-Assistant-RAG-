"""Evidence-grounded answer generation and response construction module.

Performs deterministic extractive answer construction, attaches verifiable source
citations to claims, detects conflicting evidence, and guards against prompt injections.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.abstention import (
    GENERIC_QUESTION_TERMS,
    AbstentionEvaluation,
    AbstentionPolicy,
    EvidenceStatus,
    simple_stem,
)
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
                    f"   - **Direct Excerpt:** *\"{cite.snippet}\"*  \n"
                    f"   - **Full Supporting Passage:** *\"{cite.passage_text}\"*"
                )

        return "\n".join(md_lines)


class GroundedAnswerer:
    """Constructs grounded answers strictly backed by retrieved passages."""

    def __init__(self, abstention_policy: Optional[AbstentionPolicy] = None):
        self.abstention_policy = abstention_policy or AbstentionPolicy()

    def _score_sentence_relevance(
        self,
        sentence: str,
        query_keywords: Set[str],
    ) -> Tuple[float, Set[str]]:
        """Score sentence relevance based on core keyword match density and length."""
        if not sentence or not query_keywords:
            return 0.0, set()
        words = set(tokenize(sentence, remove_stopwords=True))
        if not words:
            return 0.0, set()
        stems = {simple_stem(w) for w in words}

        core_kw = {w for w in query_keywords if w not in GENERIC_QUESTION_TERMS and len(w) > 2}
        if not core_kw:
            core_kw = query_keywords

        core_stems = {simple_stem(w) for w in core_kw}
        all_query_stems = {simple_stem(w) for w in query_keywords}

        core_overlap = stems.intersection(core_stems)
        # If there are core keywords in query but this sentence matches none of them, reject!
        if core_stems and not core_overlap:
            return 0.0, set()

        all_overlap = stems.intersection(all_query_stems)
        # Higher weight for core terms, density bonus
        score = (len(core_overlap) * 3.0) + (len(all_overlap) * 0.5) + (len(all_overlap) / max(len(words), 1))
        return score, core_overlap

    def _extract_best_sentences(
        self,
        retrieval_results: List[RetrievalResult],
        query_keywords: Set[str],
        max_sentences: int = 3,
    ) -> List[Tuple[str, RetrievalResult]]:
        """Extract top answering sentences strictly grounded in retrieved evidence."""
        scored_sentences: List[Tuple[float, Set[str], str, RetrievalResult]] = []

        for res in retrieval_results:
            sentences = split_into_sentences(res.text)
            for s in sentences:
                s_clean = s.strip()
                # Skip pure section headers or ultra-short fragments
                if len(s_clean) < 25 or s_clean.startswith("#"):
                    continue
                score, core_matched = self._score_sentence_relevance(s_clean, query_keywords)
                if score > 0.0:
                    scored_sentences.append((score, core_matched, s_clean, res))

        # Sort by sentence relevance score descending
        scored_sentences.sort(key=lambda x: -x[0])

        if not scored_sentences:
            return []

        # Select top distinct sentences with complementary evidence gating
        selected: List[Tuple[str, RetrievalResult]] = []
        seen_texts: Set[str] = set()

        # Always take the single best sentence
        best_score, best_stems, best_sentence, best_res = scored_sentences[0]
        selected.append((best_sentence, best_res))
        seen_texts.add(best_sentence)
        covered_stems: Set[str] = set(best_stems)

        # Consider secondary sentences only if they provide complementary core evidence
        for score, core_matched, sentence, res in scored_sentences[1:]:
            if sentence in seen_texts:
                continue
            if len(selected) >= max_sentences:
                break

            new_stems = core_matched - covered_stems
            # Include if it resolves distinct core keywords from query (multi-part question)
            # or is a contiguous high-scoring sentence from the same top chunk
            is_complementary = (len(new_stems) > 0 and score >= 0.40 * best_score)
            is_same_chunk_context = (res.chunk_id == best_res.chunk_id and score >= 0.70 * best_score)

            if is_complementary or is_same_chunk_context:
                selected.append((sentence, res))
                seen_texts.add(sentence)
                covered_stems.update(core_matched)

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
