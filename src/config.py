"""Configuration module for the Document Question-Answering Assistant.

Centralizes settings for document ingestion, text cleaning, chunking,
retrieval algorithms, evidence thresholds, and abstention behavior.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Set


@dataclass
class PathConfig:
    """Filesystem path configurations."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    sample_docs_dir: Path = field(init=False)
    uploads_dir: Path = field(init=False)
    storage_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.base_dir / "data"
        self.sample_docs_dir = self.data_dir / "sample_docs"
        self.uploads_dir = self.data_dir / "uploads"
        self.storage_dir = self.base_dir / "storage"
        self.reports_dir = self.base_dir / "reports"

        # Ensure runtime directories exist
        for directory in [self.data_dir, self.sample_docs_dir, self.uploads_dir, self.storage_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class IngestionConfig:
    """Document ingestion and security constraints."""
    allowed_extensions: Set[str] = field(default_factory=lambda: {".txt", ".md", ".pdf"})
    max_file_size_bytes: int = 15 * 1024 * 1024  # 15 MB
    max_documents: int = 100
    encoding: str = "utf-8"


@dataclass
class ChunkingConfig:
    """Document chunking and segmentation settings."""
    chunk_size: int = int(os.getenv("DOC_QA_CHUNK_SIZE", "500"))  # characters
    chunk_overlap: int = int(os.getenv("DOC_QA_CHUNK_OVERLAP", "100"))  # characters
    split_on_sentences: bool = True
    min_chunk_length: int = 40  # minimum characters to keep as a valid chunk


@dataclass
class RetrievalConfig:
    """Information retrieval engine parameters."""
    method: str = os.getenv("DOC_QA_RETRIEVER_METHOD", "hybrid")  # "bm25", "tfidf", "hybrid"
    top_k: int = int(os.getenv("DOC_QA_TOP_K", "4"))
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_alpha: float = 0.5  # Weight for BM25 score in hybrid fusion (1-alpha for TF-IDF)


@dataclass
class AbstentionConfig:
    """Deterministic abstention and evidence sufficiency rules."""
    # Mandatory exact phrase required by internship specifications
    abstention_message: str = "Information not found in provided document."

    # Minimum relevance score thresholds calibrated on validation queries
    bm25_min_score: float = float(os.getenv("DOC_QA_BM25_THRESHOLD", "3.00"))
    tfidf_min_score: float = float(os.getenv("DOC_QA_TFIDF_THRESHOLD", "0.08"))
    hybrid_min_score: float = float(os.getenv("DOC_QA_HYBRID_THRESHOLD", "0.40"))

    # Minimum ratio of content words in question found in retrieved passages
    min_keyword_overlap_ratio: float = float(os.getenv("DOC_QA_KEYWORD_OVERLAP_THRESHOLD", "0.35"))
    max_oov_ratio_threshold: float = float(os.getenv("DOC_QA_MAX_OOV_THRESHOLD", "0.60"))
    partial_keyword_overlap_ratio: float = float(os.getenv("DOC_QA_PARTIAL_OVERLAP_THRESHOLD", "0.70"))


@dataclass
class AppConfig:
    """Global composite configuration for the Document QA Assistant."""
    paths: PathConfig = field(default_factory=PathConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)


# Default singleton instance
config = AppConfig()
