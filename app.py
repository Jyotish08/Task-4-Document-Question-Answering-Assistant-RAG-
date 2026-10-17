"""Streamlit Interactive Web Application for Document Question-Answering Assistant (RAG).

Provides an evidence-grounded, local-first search and QA interface with:
- Multi-document upload (.txt, .md, .pdf)
- Corpus indexing and document management
- Grounded answer presentation with strict inline citations
- Expandable source passages and page-level metadata
- Deterministic abstention display for out-of-domain questions
- Session query history
"""

import os
from pathlib import Path
import tempfile
import time
from typing import List

import streamlit as st

from src.abstention import EvidenceStatus
from src.config import config
from src.document_loader import (
    DocumentLoaderError,
    EmptyDocumentError,
    FileSecurityError,
    OcrRequiredError,
    UnsupportedFileTypeError,
    load_document,
)
from src.pipeline import DocumentQAPipeline


# Set Streamlit page configuration
st.set_page_config(
    page_title="Document QA Assistant (RAG)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_pipeline_instance() -> DocumentQAPipeline:
    """Initialize and persist singleton pipeline across user re-runs."""
    pipeline = DocumentQAPipeline(retrieval_method="hybrid")
    # Automatically load persisted state or default sample corpus if available
    saved_meta = config.paths.storage_dir / "index_metadata.json"
    if saved_meta.exists():
        pipeline.load_state()
    elif config.paths.sample_docs_dir.exists():
        pipeline.ingest_directory(config.paths.sample_docs_dir)
    return pipeline


# Initialize session state
pipeline = get_pipeline_instance()

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "current_question" not in st.session_state:
    st.session_state.current_question = ""


# --- SIDEBAR: Configuration & Document Management ---
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    st.markdown("**Local-first, evidence-grounded search & QA.**")
    st.caption("No paid APIs or external network dependencies.")

    st.divider()

    # Retrieval Engine Settings
    st.subheader("Retrieval Engine")
    retrieval_method = st.selectbox(
        "Retrieval Algorithm",
        options=["hybrid", "bm25", "tfidf"],
        index=0,
        help="BM25: Okapi BM25 lexical ranker. TF-IDF: Sublinear TF-IDF with cosine similarity. Hybrid: Reciprocal Rank Fusion of both.",
    )
    if retrieval_method != pipeline.retrieval_method:
        pipeline.retrieval_method = retrieval_method

    top_k = st.slider("Top Passages (Top-K)", min_value=1, max_value=8, value=4)

    st.divider()

    # Document Upload Section
    st.subheader("📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Documents (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        help="Accepts plain text, markdown, and text-based PDFs up to 15MB each.",
    )

    if uploaded_files:
        if st.button("Process & Index Uploaded Files", use_container_width=True, type="primary"):
            success_count = 0
            for uploaded_file in uploaded_files:
                # Write to temp file for secure loading
                suffix = Path(uploaded_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)

                try:
                    doc = pipeline.ingest_file(tmp_path, custom_name=uploaded_file.name)
                    st.success(f"Indexed: **{doc.filename}** ({len(doc.pages)} page(s))")
                    success_count += 1
                except OcrRequiredError as ocr_err:
                    st.error(f"⚠️ OCR Required: {ocr_err}")
                except EmptyDocumentError as empty_err:
                    st.error(f"⚠️ Empty Document: {empty_err}")
                except UnsupportedFileTypeError as type_err:
                    st.error(f"⚠️ Unsupported Type: {type_err}")
                except FileSecurityError as sec_err:
                    st.error(f"⚠️ Security Error: {sec_err}")
                except Exception as ex:
                    st.error(f"⚠️ Extraction Error ({uploaded_file.name}): {ex}")
                finally:
                    if tmp_path.exists():
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

            if success_count > 0:
                st.rerun()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Load Sample Docs", use_container_width=True):
            pipeline.clear()
            pipeline.ingest_directory(config.paths.sample_docs_dir)
            st.rerun()

    with col_btn2:
        if st.button("Clear Index", use_container_width=True):
            pipeline.clear()
            st.rerun()

    st.divider()

    # Indexed Documents Summary
    st.subheader(f"📚 Indexed Corpus ({len(pipeline.documents)} docs, {len(pipeline.chunks)} chunks)")
    doc_summaries = pipeline.get_document_summary()
    if doc_summaries:
        for d in doc_summaries:
            with st.expander(f"{d['filename']} ({d['file_type']})"):
                st.write(f"**Title:** {d['title']}")
                st.write(f"**Pages:** {d['pages']} | **Chunks:** {d['chunk_count']}")
                st.write(f"**Total Characters:** {d['total_chars']:,}")
                st.write(f"**Source:** {d['source_url']}")
                st.write(f"**License:** {d['license']}")
                if st.button(f"Remove {d['filename']}", key=f"del_{d['doc_id']}"):
                    pipeline.remove_document(d["doc_id"])
                    st.rerun()
    else:
        st.info("No documents are currently indexed. Upload files or load sample corpus.")


# --- MAIN CONTENT AREA ---
st.title("📚 Document Question-Answering Assistant")
st.markdown(
    "Ask factual questions about the indexed documents. The assistant retrieves relevant passages, "
    "attaches verifiable citations, and **strictly refuses to invent answers** when documents do not contain the required evidence."
)

# Example Queries
st.markdown("##### 💡 Try an Example Question:")
col_ex1, col_ex2, col_ex3 = st.columns(3)
with col_ex1:
    if st.button("✅ Supported Query\n(Indentation Spaces)", use_container_width=True):
        st.session_state.current_question = "How many spaces per indentation level are standard in a Python block?"
with col_ex2:
    if st.button("⚠️ Partial Query\n(Complex Numbers & Quantum)", use_container_width=True):
        st.session_state.current_question = "Does Python support complex numbers, and what is the exact formula for quantum spin matrices in Python?"
with col_ex3:
    if st.button("🚫 Unsupported Query\n(Capital of Australia)", use_container_width=True):
        st.session_state.current_question = "What is the capital of Australia?"

# Question Input Box
with st.form("qa_form"):
    question = st.text_input(
        "Enter your question:",
        value=st.session_state.current_question,
        placeholder="e.g., What sorting algorithm is used by Python's list.sort() method?",
    )
    submit_col1, submit_col2 = st.columns([1, 5])
    with submit_col1:
        submitted = st.form_submit_button("Ask Assistant", type="primary", use_container_width=True)

# Run Query
if submitted and question:
    if not pipeline.chunks:
        st.warning("⚠️ No documents are indexed! Please upload documents or click 'Load Sample Docs' in the sidebar.")
    else:
        start_time = time.perf_counter()
        answer = pipeline.query(question, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Save to session history
        st.session_state.query_history.insert(
            0,
            {
                "question": question,
                "status": answer.status.value,
                "answer": answer.answer_text,
                "latency_ms": elapsed_ms,
                "citations_count": len(answer.citations),
            },
        )

        st.subheader("Answer")

        # Status Badge Mapping
        if answer.status == EvidenceStatus.SUPPORTED:
            st.success(f"**STATUS: SUPPORTED BY EVIDENCE** (Retrieved via `{pipeline.retrieval_method.upper()}` in {elapsed_ms:.1f} ms)")
            st.markdown(f"### {answer.answer_text}")

        elif answer.status == EvidenceStatus.PARTIAL:
            st.warning(f"**STATUS: PARTIAL / UNCERTAIN EVIDENCE** (Retrieved via `{pipeline.retrieval_method.upper()}` in {elapsed_ms:.1f} ms)")
            st.markdown(f"### {answer.answer_text}")
            if answer.uncertainty_note:
                st.info(f"ℹ️ **Evidence Limitation:** {answer.uncertainty_note}")

        elif answer.status == EvidenceStatus.UNSUPPORTED:
            st.error(f"**STATUS: INFORMATION NOT FOUND** (Abstention evaluated in {elapsed_ms:.1f} ms)")
            st.markdown(f"### > *\"{config.abstention.abstention_message}\"*")
            if answer.uncertainty_note:
                st.caption(f"**Abstention Reason:** {answer.uncertainty_note}")

        elif answer.status == EvidenceStatus.EMPTY_QUERY:
            st.warning("Please enter a non-empty question.")

        # Supporting Passages & Citations
        if answer.citations:
            st.markdown("#### 📖 Supporting Evidence Passages & Citations")
            for idx, cite in enumerate(answer.citations, start=1):
                page_str = f" • Page {cite.page_number}" if cite.page_number else ""
                section_str = f" • {cite.section}" if cite.section and cite.section != "General" else ""
                header_title = f"Citation #{idx}: {cite.source_filename}{page_str}{section_str} (Score: {cite.score:.4f})"

                with st.expander(header_title, expanded=(idx == 1)):
                    st.markdown(f"**Direct Evidence Extract:**")
                    st.info(f"\"{cite.snippet}\"")
                    st.caption(f"**Chunk ID:** `{cite.chunk_id}` | **Document ID:** `{cite.doc_id}` | **Title:** {cite.source_title}")

        # Citation Verification Badge
        if answer.citations:
            if answer.validation_passed:
                st.caption("✅ **Citation Integrity Verified**: All citations point to genuine, non-hallucinated chunks in the local corpus.")
            else:
                st.error(f"❌ Citation Integrity Warning: {answer.validation_errors}")

        # Diagnostic Expander for Researchers / QA
        with st.expander("🔍 QA & Retrieval Diagnostics (Research Mode)"):
            diag_col1, diag_col2, diag_col3 = st.columns(3)
            diag_col1.metric("Top Score", f"{answer.evaluation.top_score:.4f}")
            diag_col2.metric("Calibrated Threshold", f"{answer.evaluation.threshold:.4f}")
            diag_col3.metric("Keyword Overlap", f"{answer.evaluation.keyword_overlap_ratio:.1%}")

            st.write(f"**Matched Keywords:** `{', '.join(sorted(answer.evaluation.matched_keywords)) or 'None'}`")
            st.write(f"**Missing Keywords:** `{', '.join(sorted(answer.evaluation.missing_keywords)) or 'None'}`")
            st.write(f"**Out-of-Vocabulary (OOV) Terms:** `{', '.join(sorted(answer.evaluation.oov_keywords)) or 'None'}`")

            st.markdown("**All Retrieved Candidate Chunks:**")
            for r in answer.retrieved_results:
                st.markdown(f"- **Rank {r.rank}:** `{r.source_name}` (Page {r.page_number}) - Score: `{r.score:.4f}` | `[{r.chunk_id}]`")
                st.text(r.text[:220] + "...")


# --- SESSION QUERY HISTORY ---
if st.session_state.query_history:
    st.divider()
    st.subheader("🕒 Session Query History")
    for item in st.session_state.query_history[:10]:
        status_icon = "🟢" if item["status"] == "SUPPORTED" else ("🟡" if item["status"] == "PARTIAL_EVIDENCE" else "🔴")
        with st.expander(f"{status_icon} \"{item['question']}\" ({item['status']}) - {item['latency_ms']:.1f}ms"):
            st.write(f"**Status:** `{item['status']}`")
            st.write(f"**Answer:** {item['answer']}")
            st.caption(f"Citations attached: {item['citations_count']}")
