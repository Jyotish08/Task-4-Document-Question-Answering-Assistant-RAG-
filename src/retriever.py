"""Information retrieval engines: BM25, TF-IDF, and Hybrid lexical retrieval.

Implements structured passage retrieval with deterministic tie-breaking,
document metadata filtering, and transparent score attribution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import TextChunk
from src.config import config
from src.text_cleaner import tokenize


@dataclass
class RetrievalResult:
    """Structured representation of a single retrieved document passage."""
    chunk_id: str
    doc_id: str
    source_name: str
    page_number: Optional[int]
    section: str
    text: str
    score: float  # Raw or normalized retrieval score (NOT a calibrated probability)
    method: str   # "bm25", "tfidf", "hybrid"
    rank: int     # 1-indexed rank within top-k results
    citation_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation_label(self) -> str:
        """Create a human-readable citation badge for answers."""
        page_str = f", Page {self.page_number}" if self.page_number else ""
        section_str = f" (Section: {self.section})" if self.section and self.section != "General" else ""
        return f"[{self.source_name}{page_str}{section_str} | Chunk {self.chunk_id}]"


class BaseRetriever(ABC):
    """Abstract base class for document chunk retrieval engines."""

    def __init__(self):
        self.chunks: List[TextChunk] = []

    @abstractmethod
    def index(self, chunks: List[TextChunk]) -> None:
        """Construct the retrieval index over text chunks."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        doc_filter: Optional[Set[str]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve top-k relevant chunks for a given query."""
        pass


class BM25Retriever(BaseRetriever):
    """Lexical passage retrieval using the Okapi BM25 ranking function."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__()
        self.k1 = k1
        self.b = b
        self.bm25_index: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []

    def index(self, chunks: List[TextChunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self.bm25_index = None
            self.corpus_tokens = []
            return

        self.corpus_tokens = [tokenize(c.text, remove_stopwords=False) for c in self.chunks]
        self.bm25_index = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        doc_filter: Optional[Set[str]] = None,
    ) -> List[RetrievalResult]:
        if not self.bm25_index or not self.chunks:
            return []

        query_tokens = tokenize(query, remove_stopwords=False)
        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(query_tokens)

        # Build candidate tuples: (score, chunk_id, index) for deterministic tie-breaking
        scored_candidates = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
            if doc_filter and chunk.doc_id not in doc_filter and chunk.source_filename not in doc_filter:
                continue
            if score > 0.0:
                # Use negative chunk_id for ascending tie-break when scores match
                scored_candidates.append((float(score), chunk.chunk_id, idx, chunk))

        # Sort by score descending; if scores are equal, sort alphabetically by chunk_id
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        results: List[RetrievalResult] = []
        for rank, (score, _, _, chunk) in enumerate(scored_candidates[:top_k], start=1):
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_name=chunk.source_filename,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    score=round(score, 4),
                    method="bm25",
                    rank=rank,
                    citation_metadata={
                        "source_title": chunk.source_title,
                        "source_filename": chunk.source_filename,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    },
                )
            )

        return results


class TFIDFRetriever(BaseRetriever):
    """Lexical passage retrieval using TF-IDF vectorization with cosine similarity."""

    def __init__(self, ngram_range=(1, 2), sublinear_tf: bool = True):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda text: tokenize(text, remove_stopwords=False),
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            token_pattern=None,
        )
        self.tfidf_matrix = None

    def index(self, chunks: List[TextChunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self.tfidf_matrix = None
            return

        corpus = [c.text for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        doc_filter: Optional[Set[str]] = None,
    ) -> List[RetrievalResult]:
        if self.tfidf_matrix is None or not self.chunks:
            return []

        query_tokens = tokenize(query, remove_stopwords=False)
        if not query_tokens:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        scored_candidates = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, similarities)):
            if doc_filter and chunk.doc_id not in doc_filter and chunk.source_filename not in doc_filter:
                continue
            if score > 0.0:
                scored_candidates.append((float(score), chunk.chunk_id, idx, chunk))

        # Sort by cosine similarity descending, then chunk_id ascending
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        results: List[RetrievalResult] = []
        for rank, (score, _, _, chunk) in enumerate(scored_candidates[:top_k], start=1):
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_name=chunk.source_filename,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    score=round(score, 4),
                    method="tfidf",
                    rank=rank,
                    citation_metadata={
                        "source_title": chunk.source_title,
                        "source_filename": chunk.source_filename,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    },
                )
            )

        return results


class HybridRetriever(BaseRetriever):
    """Ensemble hybrid retriever combining BM25 and TF-IDF via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, bm25_retriever: Optional[BM25Retriever] = None, tfidf_retriever: Optional[TFIDFRetriever] = None, rrf_k: int = 60):
        super().__init__()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.tfidf = tfidf_retriever or TFIDFRetriever()
        self.rrf_k = rrf_k

    def index(self, chunks: List[TextChunk]) -> None:
        self.chunks = list(chunks)
        self.bm25.index(chunks)
        self.tfidf.index(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        doc_filter: Optional[Set[str]] = None,
    ) -> List[RetrievalResult]:
        if not self.chunks:
            return []

        # Retrieve a broader pool of candidates from each engine
        pool_size = max(top_k * 3, 10)
        bm25_res = self.bm25.retrieve(query, top_k=pool_size, doc_filter=doc_filter)
        tfidf_res = self.tfidf.retrieve(query, top_k=pool_size, doc_filter=doc_filter)

        if not bm25_res and not tfidf_res:
            return []

        # Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank))
        chunk_map: Dict[str, TextChunk] = {c.chunk_id: c for c in self.chunks}
        rrf_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}
        tfidf_scores: Dict[str, float] = {}

        for rank, res in enumerate(bm25_res, start=1):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            bm25_scores[res.chunk_id] = res.score

        for rank, res in enumerate(tfidf_res, start=1):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            tfidf_scores[res.chunk_id] = res.score

        # Deterministic sorting by RRF score descending, chunk_id ascending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))
        max_possible_rrf = (1.0 / (self.rrf_k + 1)) * 2.0

        results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            chunk = chunk_map[cid]
            raw_rrf = rrf_scores[cid]
            norm_score = min(1.0, raw_rrf / max_possible_rrf)
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_name=chunk.source_filename,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    score=round(norm_score, 4),
                    method="hybrid",
                    rank=rank,
                    citation_metadata={
                        "source_title": chunk.source_title,
                        "source_filename": chunk.source_filename,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "raw_rrf_score": round(raw_rrf, 5),
                        "bm25_score": bm25_scores.get(cid, 0.0),
                        "tfidf_score": tfidf_scores.get(cid, 0.0),
                    },
                )
            )

        return results


def build_retriever(method: str = "hybrid") -> BaseRetriever:
    """Factory function for initializing retriever implementations."""
    method_clean = method.lower().strip()
    if method_clean == "bm25":
        return BM25Retriever()
    elif method_clean == "tfidf":
        return TFIDFRetriever()
    elif method_clean == "hybrid":
        return HybridRetriever()
    else:
        raise ValueError(f"Unknown retrieval method '{method}'. Supported methods: 'bm25', 'tfidf', 'hybrid'")
