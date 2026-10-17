"""Automated unit and integration tests for document ingestion and text extraction."""

from pathlib import Path
import pytest
import pymupdf

from src.document_loader import (
    DocumentLoaderError,
    DocumentPage,
    EmptyDocumentError,
    FileSecurityError,
    OcrRequiredError,
    UnsupportedFileTypeError,
    generate_doc_id,
    load_document,
    load_pdf,
    load_text_or_markdown,
    sanitize_filename,
    validate_file,
)


def test_sanitize_filename():
    """Verify filename sanitization prevents path traversal and unsafe characters."""
    assert sanitize_filename("simple.txt") == "simple.txt"
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("report;rm -rf /") == "report_rm -rf"
    assert sanitize_filename("") == "unnamed_document"


def test_generate_doc_id_stability():
    """Verify doc_id generation is deterministic and unique for distinct content."""
    id1 = generate_doc_id("file1.txt:contentA")
    id2 = generate_doc_id("file1.txt:contentA")
    id3 = generate_doc_id("file1.txt:contentB")

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("doc_")


def test_load_text_document(tmp_path: Path):
    """Test standard plain text document ingestion."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Python is an interpreted high-level general-purpose programming language.", encoding="utf-8")

    doc = load_document(txt_file)
    assert doc.file_type == "txt"
    assert doc.filename == "sample.txt"
    assert len(doc.pages) == 1
    assert "programming language" in doc.full_text


def test_load_markdown_document_with_frontmatter(tmp_path: Path):
    """Test markdown document ingestion and frontmatter metadata extraction."""
    md_file = tmp_path / "guide.md"
    md_content = """---
title: "Python Control Flow"
source_url: "https://docs.python.org/3/tutorial/"
license: "PSFL"
---
# Control Flow
Python includes if statements and while loops for branching.
"""
    md_file.write_text(md_content, encoding="utf-8")

    doc = load_document(md_file)
    assert doc.file_type == "md"
    assert doc.title == "Python Control Flow"
    assert doc.source_url == "https://docs.python.org/3/tutorial/"
    assert doc.license_info == "PSFL"
    assert "# Control Flow" in doc.full_text


def test_load_pdf_multi_page(tmp_path: Path):
    """Test PDF text extraction across multiple pages preserving page numbers."""
    pdf_path = tmp_path / "multipage.pdf"
    pdf = pymupdf.open()

    p1 = pdf.new_page()
    p1.insert_text((50, 100), "This is page 1 content discussing syntax.", fontsize=12)

    p2 = pdf.new_page()
    p2.insert_text((50, 100), "This is page 2 content discussing memory management.", fontsize=12)

    pdf.save(str(pdf_path))
    pdf.close()

    doc = load_document(pdf_path)
    assert doc.file_type == "pdf"
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "page 1 content" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "page 2 content" in doc.pages[1].text


def test_empty_file_rejected(tmp_path: Path):
    """Verify that completely empty files raise EmptyDocumentError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(empty_file)


def test_unsupported_file_extension(tmp_path: Path):
    """Verify that unsupported extensions (.exe, .csv, .docx) are rejected."""
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b,c\n1,2,3", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        load_document(bad_file)
    assert ".csv" in str(exc_info.value)


def test_empty_or_scanned_pdf_detects_ocr_required(tmp_path: Path):
    """Verify that empty or image-only PDFs raise OcrRequiredError with helpful explanation."""
    pdf_path = tmp_path / "blank.pdf"
    pdf = pymupdf.open()
    pdf.new_page()  # Blank page with no text
    pdf.save(str(pdf_path))
    pdf.close()

    with pytest.raises(OcrRequiredError) as exc_info:
        load_document(pdf_path)
    assert "optical character recognition" in str(exc_info.value).lower()
