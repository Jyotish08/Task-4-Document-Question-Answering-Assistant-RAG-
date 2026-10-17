"""Document ingestion and validation module.

Handles secure file validation, path traversal prevention, file size checks,
and text extraction from TXT, MD, and text-based PDF documents.
"""

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set

import pymupdf

from src.config import config
from src.text_cleaner import clean_document_text, parse_frontmatter


class DocumentLoaderError(Exception):
    """Base exception for document loading errors."""
    pass


class UnsupportedFileTypeError(DocumentLoaderError):
    """Raised when an unsupported file extension is provided."""
    pass


class FileSecurityError(DocumentLoaderError):
    """Raised when a file fails security checks (path traversal, size, etc.)."""
    pass


class EmptyDocumentError(DocumentLoaderError):
    """Raised when an ingested document contains no readable text."""
    pass


class OcrRequiredError(EmptyDocumentError):
    """Raised when a PDF has pages but no extractable text (scanned / image-only)."""
    pass


class DocumentExtractionError(DocumentLoaderError):
    """Raised when a file is corrupted or cannot be parsed."""
    pass


@dataclass
class DocumentPage:
    """Represents a single page or distinct section of a document."""
    page_number: int  # 1-indexed
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class Document:
    """Represents an ingested and validated document."""
    doc_id: str
    filename: str
    title: str
    file_type: str  # "txt", "md", "pdf"
    pages: List[DocumentPage]
    total_chars: int = field(init=False)
    source_url: Optional[str] = None
    license_info: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.total_chars = sum(p.char_count for p in self.pages)

    @property
    def full_text(self) -> str:
        """Concatenated text of all pages."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


def sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename to prevent path traversal and shell injection.

    Strips directory separators, null bytes, and non-whitelisted characters.
    """
    if not filename:
        return "unnamed_document"
    # Strip trailing slashes to isolate file basename
    stripped = filename.rstrip("/\\")
    if not stripped:
        return "unnamed_document"
    # Take only the basename component
    base = os.path.basename(stripped)
    # Remove null bytes and path traversal patterns
    base = base.replace("\0", "").replace("..", "")
    # Whitelist alphanumeric, dots, underscores, dashes, and spaces
    clean_base = re.sub(r"[^a-zA-Z0-9._\- ]", "_", base).strip()
    return clean_base or "document"


def generate_doc_id(seed_text: str, prefix: str = "doc") -> str:
    """Generate a stable, collision-resistant document ID using SHA-256."""
    h = hashlib.sha256(seed_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{h}"


def validate_file(
    file_path: Path,
    allowed_extensions: Optional[Set[str]] = None,
    max_size_bytes: Optional[int] = None,
) -> Path:
    """Perform security and integrity validation on a filesystem path."""
    allowed = allowed_extensions or config.ingestion.allowed_extensions
    max_size = max_size_bytes or config.ingestion.max_file_size_bytes

    resolved = file_path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not resolved.is_file():
        raise FileSecurityError(f"Target path is not a regular file: {file_path}")

    # Extension check
    suffix = resolved.suffix.lower()
    if suffix not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{suffix}'. Allowed formats are: {allowed_list}"
        )

    # Size check
    file_size = resolved.stat().st_size
    if file_size == 0:
        raise EmptyDocumentError(f"File is completely empty (0 bytes): {resolved.name}")

    if file_size > max_size:
        raise FileSecurityError(
            f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit "
            f"of {max_size / (1024*1024):.2f} MB"
        )

    return resolved


def load_text_or_markdown(file_path: Path, sanitized_name: str) -> Document:
    """Extract and parse text or markdown file."""
    # Attempt reading with UTF-8 first, fallback to latin-1 / cp1252 if decoding fails
    content = ""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentExtractionError(f"Unable to decode text file with supported encodings: {file_path.name}")

    cleaned = clean_document_text(content)
    if not cleaned.strip():
        raise EmptyDocumentError(f"Document contains no readable text content: {file_path.name}")

    # Parse YAML frontmatter if Markdown
    meta, body = parse_frontmatter(cleaned)
    body_cleaned = clean_document_text(body)

    file_type = "md" if file_path.suffix.lower() == ".md" else "txt"
    title = meta.get("title") or file_path.stem.replace("_", " ").replace("-", " ").title()
    source_url = meta.get("source_url")
    license_info = meta.get("license") or meta.get("license_info")
    doc_id = generate_doc_id(f"{sanitized_name}:{body_cleaned[:200]}")

    page = DocumentPage(page_number=1, text=body_cleaned)
    return Document(
        doc_id=doc_id,
        filename=sanitized_name,
        title=title,
        file_type=file_type,
        pages=[page],
        source_url=source_url,
        license_info=license_info,
        metadata=meta,
    )


def load_pdf(file_path: Path, sanitized_name: str) -> Document:
    """Extract page-level text from text-based PDF using PyMuPDF."""
    try:
        doc = pymupdf.open(str(file_path))
    except Exception as e:
        raise DocumentExtractionError(f"Failed to open PDF (corrupted or unreadable format): {e}")

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise EmptyDocumentError(f"PDF contains 0 pages: {file_path.name}")

        pages: List[DocumentPage] = []
        total_text_chars = 0

        for page_idx in range(total_pages):
            page_obj = doc[page_idx]
            raw_text = page_obj.get_text()
            cleaned_page_text = clean_document_text(raw_text)
            if cleaned_page_text.strip():
                pages.append(DocumentPage(page_number=page_idx + 1, text=cleaned_page_text))
                total_text_chars += len(cleaned_page_text)

        # Check if PDF contains no extractable text (e.g. scanned or image-only)
        if total_text_chars == 0:
            raise OcrRequiredError(
                f"PDF '{file_path.name}' contains {total_pages} page(s) but no extractable text. "
                "The file appears to be image-only or scanned. An optical character recognition "
                "(OCR) pre-processing step is required to extract text from this document."
            )

        # Read metadata from PDF document properties
        pdf_meta = doc.metadata or {}
        title = pdf_meta.get("title") or file_path.stem.replace("_", " ").replace("-", " ").title()
        doc_id = generate_doc_id(f"{sanitized_name}:{pages[0].text[:200]}")

        return Document(
            doc_id=doc_id,
            filename=sanitized_name,
            title=title,
            file_type="pdf",
            pages=pages,
            metadata={"total_pdf_pages": total_pages, **pdf_meta},
        )
    finally:
        doc.close()


def load_document(file_path: Path, custom_filename: Optional[str] = None) -> Document:
    """Validate and load a single document from filesystem path."""
    validated_path = validate_file(file_path)
    clean_name = sanitize_filename(custom_filename or validated_path.name)
    suffix = validated_path.suffix.lower()

    if suffix in (".txt", ".md"):
        return load_text_or_markdown(validated_path, clean_name)
    elif suffix == ".pdf":
        return load_pdf(validated_path, clean_name)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file format: {suffix}")


def load_directory(dir_path: Path) -> List[Document]:
    """Ingest and validate all supported documents in a directory."""
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    documents: List[Document] = []
    allowed = config.ingestion.allowed_extensions

    for file_path in sorted(dir_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in allowed:
            try:
                doc = load_document(file_path)
                documents.append(doc)
            except DocumentLoaderError as err:
                # Log error cleanly without crashing batch load
                print(f"[Warning] Skipped invalid document {file_path.name}: {err}")

    return documents
