# Document QA Assistant Evaluation Report

**Execution Date:** 2026-10-27 20:03:58  
**Retrieval Method:** `HYBRID`  
**Total Evaluated Queries:** 37  
- Answerable Queries: 16  
- Unsupported Queries: 16  
- Partially Supported Queries: 5  

## 1. Quantitative Performance Metrics

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **Retrieval Recall@1** | 100.0% | ≥ 80% | ✅ Pass |
| **Retrieval Recall@3** | 100.0% | ≥ 90% | ✅ Pass |
| **Retrieval Recall@5** | 100.0% | ≥ 95% | ✅ Pass |
| **Mean Reciprocal Rank (MRR)** | 1.0000 | ≥ 0.8500 | ✅ Pass |
| **Answer Support Rate** | 87.5% | ≥ 90% | ⚠️ Review |
| **Citation Correctness** | 100.0% | 100% | ✅ Pass |
| **Abstention Precision** | 100.0% | ≥ 95% | ✅ Pass |
| **Abstention Recall** | 100.0% | ≥ 95% | ✅ Pass |
| **False Answer Rate (Unsupported)** | 0.0% | 0.0% | ✅ Pass |
| **Mean Query Latency** | 4.90 ms | < 50 ms | ✅ Pass |
| **P95 Query Latency** | 6.43 ms | < 100 ms | ✅ Pass |

## 2. Sample Query Transcripts

### A. Supported Query Example
- **Question:** "How many spaces per indentation level are standard in a Python block?"
- **Status:** `SUPPORTED`
- **Generated Grounded Answer:** A standard Python block uses 4 spaces per indentation level. [01_python_syntax_and_types.md, Page 1 (Section: Python Syntax, Variables, and Primitive Types) | Chunk doc_00bde289297e_p1_c001]
- **Citation Sources:** `['01_python_syntax_and_types.md']`
- **Latency:** 5.71 ms

### B. Unsupported Query Example (Mandatory Abstention)
- **Question:** "How do you define a struct in C++20 with designated initializers?"
- **Status:** `UNSUPPORTED`
- **Output:** "Information not found in provided document."
- **Expected Abstain:** True | **Actual Abstain:** True
- **Latency:** 3.05 ms

### C. Partially Supported Query Example
- **Question:** "Does Python support complex numbers, and what is the exact formula for quantum spin matrices in Python?"
- **Status:** `PARTIAL_EVIDENCE`
- **Output:** **Complex numbers (`complex`)**: Written in the form `real + imag*j`, where `j` represents the imaginary unit. [01_python_syntax_and_types.md, Page 1 (Section: 3. Numeric Types) | Chunk doc_00bde289297e_p1_c007]
- **Citation Sources:** `['01_python_syntax_and_types.md']`
- **Latency:** 5.31 ms
