"""Citation extraction, formatting, and validation module.

Ensures strict source grounding, verifies that citations reference actual chunks
in the corpus, and prevents fabrication of sources, page numbers, or snippets.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.chunker import TextChunk
from src.retriever import RetrievalResult


@dataclass
class Citation:
    """Verifiable source citation tied to an exact document passage."""
    citation_id: str
    chunk_id: str
    doc_id: str
    source_title: str
    source_filename: str
    page_number: Optional[int]
    section: str
    snippet: str
    score: float
    passage_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_formatted_citation(self) -> str:
        """Formatted human-readable citation string."""
        page_str = f", Page {self.page_number}" if self.page_number else ""
        section_str = f" (Section: {self.section})" if self.section and self.section != "General" else ""
        return f"[{self.source_filename}{page_str}{section_str} | ID: {self.chunk_id}]"


def create_citation_from_retrieval(result: RetrievalResult, snippet: Optional[str] = None) -> Citation:
    """Construct a Citation object directly from a verified RetrievalResult."""
    chosen_snippet = snippet if snippet else (result.text[:180] + ("..." if len(result.text) > 180 else ""))
    return Citation(
        citation_id=f"cite_{result.chunk_id}",
        chunk_id=result.chunk_id,
        doc_id=result.doc_id,
        source_title=result.citation_metadata.get("source_title", result.source_name),
        source_filename=result.source_name,
        page_number=result.page_number,
        section=result.section,
        snippet=chosen_snippet.strip(),
        score=result.score,
        passage_text=result.text,
        metadata=result.citation_metadata,
    )


class CitationValidator:
    """Validates citations against the true corpus chunks to prevent hallucinations."""

    @staticmethod
    def validate_citations(
        citations: List[Citation],
        corpus_chunks: List[TextChunk],
    ) -> Tuple[bool, List[str]]:
        """Verify that all citations exist in the corpus and their snippets are genuine.

        Returns:
            (is_valid, list_of_validation_errors)
        """
        chunk_map: Dict[str, TextChunk] = {c.chunk_id: c for c in corpus_chunks}
        errors: List[str] = []

        for cite in citations:
            # 1. Existence check
            if cite.chunk_id not in chunk_map:
                errors.append(f"Citation references nonexistent chunk_id: '{cite.chunk_id}'")
                continue

            target_chunk = chunk_map[cite.chunk_id]

            # 2. Document and filename integrity
            if cite.doc_id != target_chunk.doc_id:
                errors.append(
                    f"Citation doc_id mismatch for chunk '{cite.chunk_id}': "
                    f"expected '{target_chunk.doc_id}', got '{cite.doc_id}'"
                )

            if cite.source_filename != target_chunk.source_filename:
                errors.append(
                    f"Citation filename mismatch for chunk '{cite.chunk_id}': "
                    f"expected '{target_chunk.source_filename}', got '{cite.source_filename}'"
                )

            # 3. Page number integrity
            if cite.page_number != target_chunk.page_number:
                errors.append(
                    f"Citation page number mismatch for chunk '{cite.chunk_id}': "
                    f"expected '{target_chunk.page_number}', got '{cite.page_number}'"
                )

            # 4. Snippet grounding check (snippet text or substantial substring must exist in chunk)
            clean_snippet = cite.snippet.replace("...", "").strip()
            # Test if a meaningful substring (at least 20 chars) exists in target chunk text
            if len(clean_snippet) >= 20 and clean_snippet[:30] not in target_chunk.text:
                # Secondary check: does any significant sentence from snippet exist?
                words = clean_snippet.split()
                if len(words) > 5:
                    phrase = " ".join(words[:5])
                    if phrase not in target_chunk.text:
                        errors.append(
                            f"Citation snippet for chunk '{cite.chunk_id}' cannot be found in target passage."
                        )

        is_valid = len(errors) == 0
        return is_valid, errors
