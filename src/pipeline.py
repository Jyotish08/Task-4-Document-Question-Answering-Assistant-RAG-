"""End-to-end Document QA Pipeline coordinator.

Connects document loading, text cleaning, chunking, indexing, retrieval,
abstention evaluation, grounded answer synthesis, and citation validation.
"""

from dataclasses import asdict
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

    def save_state(self) -> Path:
        """Serialize current corpus state and chunk metadata to storage."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        save_file = self.storage_dir / "index_metadata.json"
        serializable_data = {
            "retrieval_method": self.retrieval_method,
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
            return True
        except Exception as e:
            print(f"[Warning] Failed to restore saved state: {e}")
            return False
