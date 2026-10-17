"""Semantic and boundary-aware document chunking module.

Implements sentence- and paragraph-aware text chunking with configurable
sizes, overlaps, section header tracking, and metadata preservation.
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional

from src.config import config
from src.document_loader import Document, DocumentPage
from src.text_cleaner import tokenize


@dataclass
class TextChunk:
    """Represents a coherent passage segment derived from a document."""
    chunk_id: str
    doc_id: str
    source_title: str
    source_filename: str
    page_number: Optional[int]
    section: str
    text: str
    start_char: int
    end_char: int
    char_count: int = field(init=False)
    token_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)
        self.token_count = len(tokenize(self.text))


# Sentence boundary splitter regex respecting common abbreviations
ABBREVIATION_REGEX = re.compile(
    r"\b(e\.g\.|i\.e\.|etc\.|vs\.|dr\.|mr\.|mrs\.|ms\.|prof\.|fig\.|no\.|p\.|pp\.)\s*$",
    re.IGNORECASE,
)
SECTION_HEADER_REGEX = re.compile(
    r"^(?:#{1,6}\s+|(?:\d+\.)+\s+)([^\n]+)",
    re.MULTILINE,
)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting common English abbreviations."""
    if not text:
        return []

    # Candidate splits on sentence-ending punctuation followed by whitespace
    raw_splits = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences: List[str] = []
    buffer = ""

    for segment in raw_splits:
        if not segment:
            continue
        if buffer:
            buffer += " " + segment
        else:
            buffer = segment

        # Check if the buffer ends with a known abbreviation
        if ABBREVIATION_REGEX.search(buffer):
            continue  # Do not split yet; continue accumulating

        sentences.append(buffer)
        buffer = ""

    if buffer:
        sentences.append(buffer)

    return sentences


def detect_section_header(text: str) -> Optional[str]:
    """Detect if a text segment begins with or contains a prominent section heading."""
    match = SECTION_HEADER_REGEX.search(text)
    if match:
        return match.group(1).strip()
    return None


class DocumentChunker:
    """Boundary-aware document chunker preserving structural context and metadata."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_length: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.min_chunk_length = min_chunk_length or config.chunking.min_chunk_length

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def chunk_page(
        self,
        doc_id: str,
        source_title: str,
        source_filename: str,
        page: DocumentPage,
        current_section: str = "General",
    ) -> List[TextChunk]:
        """Chunk an individual document page respecting paragraph and sentence boundaries."""
        page_text = page.text.strip()
        if not page_text:
            return []

        # If total page text is smaller than chunk_size, return it as a single chunk
        if len(page_text) <= self.chunk_size:
            header = detect_section_header(page_text) or current_section
            return [
                TextChunk(
                    chunk_id=f"{doc_id}_p{page.page_number}_c001",
                    doc_id=doc_id,
                    source_title=source_title,
                    source_filename=source_filename,
                    page_number=page.page_number,
                    section=header,
                    text=page_text,
                    start_char=0,
                    end_char=len(page_text),
                )
            ]

        # Split page into structural blocks/paragraphs
        paragraphs = re.split(r"\n\s*\n+", page_text)
        chunks: List[TextChunk] = []
        chunk_idx = 1

        active_section = current_section
        current_sentences: List[str] = []
        current_len = 0
        window_start_offset = 0

        # Break paragraphs into individual sentences with section header awareness
        linear_items: List[dict] = []
        char_cursor = 0

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue

            # Update active section header if encountered
            found_header = detect_section_header(p_clean)
            if found_header:
                active_section = found_header

            sentences = split_into_sentences(p_clean)
            for s in sentences:
                s_len = len(s)
                linear_items.append({
                    "text": s,
                    "section": active_section,
                    "start": char_cursor,
                    "end": char_cursor + s_len,
                })
                char_cursor += s_len + 1  # approximate inter-sentence whitespace

        if not linear_items:
            return []

        i = 0
        total_items = len(linear_items)

        while i < total_items:
            chunk_items: List[dict] = []
            accumulated_chars = 0
            chunk_section = linear_items[i]["section"]
            start_offset = linear_items[i]["start"]

            j = i
            while j < total_items:
                item = linear_items[j]
                item_len = len(item["text"])

                # Check if adding this sentence exceeds chunk_size
                if accumulated_chars + item_len > self.chunk_size and chunk_items:
                    break

                chunk_items.append(item)
                accumulated_chars += item_len + 1
                j += 1

            # Build chunk text
            chunk_text = " ".join(item["text"] for item in chunk_items).strip()
            end_offset = chunk_items[-1]["end"] if chunk_items else start_offset + len(chunk_text)

            if len(chunk_text) >= self.min_chunk_length:
                chunk_id = f"{doc_id}_p{page.page_number}_c{chunk_idx:03d}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        source_title=source_title,
                        source_filename=source_filename,
                        page_number=page.page_number,
                        section=chunk_section,
                        text=chunk_text,
                        start_char=start_offset,
                        end_char=end_offset,
                    )
                )
                chunk_idx += 1

            # Advance window taking overlap into account
            if j >= total_items:
                break

            # Calculate how many items to step back for overlap
            overlap_chars = 0
            step_back = 0
            for k in range(len(chunk_items) - 1, -1, -1):
                overlap_chars += len(chunk_items[k]["text"])
                if overlap_chars >= self.chunk_overlap:
                    step_back = len(chunk_items) - k
                    break

            # Ensure forward progress
            advance = max(1, len(chunk_items) - step_back)
            i += advance

        return chunks

    def chunk_document(self, doc: Document) -> List[TextChunk]:
        """Chunk an entire document across all pages preserving section context."""
        all_chunks: List[TextChunk] = []
        current_section = "General"

        for page in doc.pages:
            page_chunks = self.chunk_page(
                doc_id=doc.doc_id,
                source_title=doc.title,
                source_filename=doc.filename,
                page=page,
                current_section=current_section,
            )
            if page_chunks:
                current_section = page_chunks[-1].section
                all_chunks.extend(page_chunks)

        return all_chunks

    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """Chunk a collection of documents."""
        all_chunks: List[TextChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
