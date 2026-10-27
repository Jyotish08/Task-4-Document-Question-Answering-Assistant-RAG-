"""End-to-end Document QA Pipeline coordinator.

Connects document loading, text cleaning, chunking, indexing, retrieval,
abstention evaluation, grounded answer synthesis, and citation validation.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.abstention import AbstentionPolicy
from src.answerer import GroundedAnswer, GroundedAnswerer
from src.chunker import DocumentChunker, TextChunk
from src.citation import CitationValidator
from src.config import config
from src.document_loader import Document, load_directory, load_document
from src.retriever import BaseRetriever, RetrievalResult, build_retriever


class DocumentQAPipeline:
    """Orchestrates ingestion, chunking, indexing, retrieval, and answering."""

    def __init__(
        self,
        retrieval_method: str = "hybrid",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        top_k: Optional[int] = None,
        storage_dir: Optional[Path] = None,
    ):
        self.retrieval_method = retrieval_method
        self.top_k = top_k or config.retrieval.top_k
        self.storage_dir = storage_dir or config.paths.storage_dir

        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever: BaseRetriever = build_retriever(self.retrieval_method)
        self.abstention_policy = AbstentionPolicy()
        self.answerer = GroundedAnswerer(abstention_policy=self.abstention_policy)

        # Internal in-memory registry
        self.documents: Dict[str, Document] = {}
        self.chunks: List[TextChunk] = []

    def load_and_index_documents(self, documents: List[Document]) -> int:
        """Ingest, chunk, and index a list of pre-parsed Document objects.

        Returns:
            Total number of chunks indexed.
        """
        for doc in documents:
            self.documents[doc.doc_id] = doc

        # Re-chunk all active documents
        self.chunks = self.chunker.chunk_documents(list(self.documents.values()))

        # Re-index the retriever
        self.retriever.index(self.chunks)
        self.save_state()
        return len(self.chunks)

    def ingest_file(self, file_path: Path, custom_name: Optional[str] = None) -> Document:
        """Ingest a single file, chunk it, and update the index."""
        doc = load_document(file_path, custom_filename=custom_name)
        self.documents[doc.doc_id] = doc
        self.chunks = self.chunker.chunk_documents(list(self.documents.values()))
        self.retriever.index(self.chunks)
        self.save_state()
        return doc

    def ingest_directory(self, dir_path: Path) -> int:
        """Ingest all supported documents in a directory and build the index."""
        docs = load_directory(dir_path)
        return self.load_and_index_documents(docs)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the pipeline and rebuild the index."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.chunks = self.chunker.chunk_documents(list(self.documents.values()))
            self.retriever.index(self.chunks)
            self.save_state()
            return True
        return False

    def clear(self) -> None:
        """Clear all indexed documents and reset retriever state."""
        self.documents.clear()
        self.chunks.clear()
        self.retriever.index([])
        self.save_state()

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        doc_filter: Optional[Set[str]] = None,
    ) -> GroundedAnswer:
        """Execute the full question-answering workflow on a user question."""
        k = top_k or self.top_k

        # 1. Retrieve candidate chunks
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=k,
            doc_filter=doc_filter,
        )

        # 2. Generate grounded answer or abstain
        corpus_text = " ".join(c.text for c in self.chunks) if self.chunks else None
        answer = self.answerer.generate_answer(
            query=question,
            retrieval_results=retrieval_results,
            method=self.retrieval_method,
            corpus_text=corpus_text,
        )

        # 3. Validate citations against corpus
        if answer.citations:
            is_valid, validation_errors = CitationValidator.validate_citations(
                citations=answer.citations,
                corpus_chunks=self.chunks,
            )
            answer.validation_passed = is_valid
            answer.validation_errors = validation_errors

        return answer

    def get_document_summary(self) -> List[Dict[str, Any]]:
        """Return structured summary of currently indexed documents."""
        summary = []
        for doc in self.documents.values():
            doc_chunks = [c for c in self.chunks if c.doc_id == doc.doc_id]
            summary.append({
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "title": doc.title,
                "file_type": doc.file_type.upper(),
                "pages": len(doc.pages),
                "total_chars": doc.total_chars,
                "chunk_count": len(doc_chunks),
                "source_url": doc.source_url or "Local File",
                "license": doc.license_info or "Not Specified",
            })
        return summary

    def compute_corpus_fingerprint(self) -> str:
        """Compute a deterministic SHA-256 fingerprint of currently indexed documents."""
        hasher = hashlib.sha256()
        for doc_id in sorted(self.documents.keys()):
            doc = self.documents[doc_id]
            doc_str = f"{doc.doc_id}:{doc.filename}:{doc.full_text}"
            hasher.update(doc_str.encode("utf-8", errors="ignore"))
        return hasher.hexdigest()[:16]

    @staticmethod
    def compute_directory_fingerprint(dir_path: Path) -> str:
        """Compute a deterministic hash representing file contents in a directory."""
        if not dir_path.exists() or not dir_path.is_dir():
            return ""
        hasher = hashlib.sha256()
        allowed = config.ingestion.allowed_extensions
        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in allowed:
                stat = file_path.stat()
                file_sig = f"{file_path.name}:{stat.st_size}:{int(stat.st_mtime)}"
                hasher.update(file_sig.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def save_state(self) -> Path:
        """Serialize current corpus state, chunk metadata, and fingerprint to storage."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        save_file = self.storage_dir / "index_metadata.json"
        serializable_data = {
            "retrieval_method": self.retrieval_method,
            "corpus_fingerprint": self.compute_corpus_fingerprint(),
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "filename": d.filename,
                    "title": d.title,
                    "file_type": d.file_type,
                    "pages": [{"page_number": p.page_number, "text": p.text} for p in d.pages],
                    "source_url": d.source_url,
                    "license_info": d.license_info,
                    "metadata": d.metadata,
                }
                for d in self.documents.values()
            ],
            "chunks": [asdict(c) for c in self.chunks],
        }

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        return save_file

    def load_state(self) -> bool:
        """Restore pipeline documents and chunks from storage if available."""
        save_file = self.storage_dir / "index_metadata.json"
        if not save_file.exists():
            return False

        try:
            with open(save_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            from src.document_loader import DocumentPage

            restored_docs: Dict[str, Document] = {}
            for d in data.get("documents", []):
                pages = [DocumentPage(page_number=p["page_number"], text=p["text"]) for p in d["pages"]]
                doc = Document(
                    doc_id=d["doc_id"],
                    filename=d["filename"],
                    title=d["title"],
                    file_type=d["file_type"],
                    pages=pages,
                    source_url=d.get("source_url"),
                    license_info=d.get("license_info"),
                    metadata=d.get("metadata", {}),
                )
                restored_docs[doc.doc_id] = doc

            self.documents = restored_docs
            self.chunks = self.chunker.chunk_documents(list(self.documents.values()))
            self.retriever.index(self.chunks)

            # Invalidate if restored corpus does not match saved fingerprint
            saved_fp = data.get("corpus_fingerprint")
            current_fp = self.compute_corpus_fingerprint()
            if saved_fp and saved_fp != current_fp:
                print(f"[Warning] Index fingerprint mismatch: saved {saved_fp} != current {current_fp}")

            return True
        except Exception as e:
            print(f"[Warning] Failed to restore saved state: {e}")
            return False

    def is_index_fresh(self, dir_path: Path) -> bool:
        """Check whether the saved index contains all files and exact filenames from dir_path."""
        if not self.documents:
            return False
        if not dir_path.exists() or not dir_path.is_dir():
            return True
        allowed = config.ingestion.allowed_extensions
        dir_files = {p.name for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in allowed}
        indexed_files = {d.filename for d in self.documents.values()}
        return dir_files == indexed_files

    def sync_directory(self, dir_path: Path) -> int:
        """Synchronize the index with the contents of a directory, rebuilding if stale."""
        if self.load_state():
            if self.is_index_fresh(dir_path):
                return len(self.chunks)
        # If no saved state or index is stale, re-ingest directory
        return self.ingest_directory(dir_path)
