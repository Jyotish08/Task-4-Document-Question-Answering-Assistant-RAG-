# Document QA Assistant Evaluation Report

**Execution Date:** 2026-10-17 16:33:27  
**Retrieval Method:** `HYBRID`  
**Total Evaluated Queries:** 35  
- Answerable Queries: 15  
- Unsupported Queries: 15  
- Partially Supported Queries: 5  

## 1. Quantitative Performance Metrics

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **Retrieval Recall@1** | 100.0% | ≥ 80% | ✅ Pass |
| **Retrieval Recall@3** | 100.0% | ≥ 90% | ✅ Pass |
| **Retrieval Recall@5** | 100.0% | ≥ 95% | ✅ Pass |
| **Mean Reciprocal Rank (MRR)** | 1.0000 | ≥ 0.8500 | ✅ Pass |
| **Answer Support Rate** | 86.7% | ≥ 90% | ⚠️ Review |
| **Citation Correctness** | 100.0% | 100% | ✅ Pass |
| **Abstention Precision** | 100.0% | ≥ 95% | ✅ Pass |
| **Abstention Recall** | 100.0% | ≥ 95% | ✅ Pass |
| **False Answer Rate (Unsupported)** | 0.0% | 0.0% | ✅ Pass |
| **Mean Query Latency** | 4.75 ms | < 50 ms | ✅ Pass |
| **P95 Query Latency** | 7.75 ms | < 100 ms | ✅ Pass |

## 2. Sample Query Transcripts

### A. Supported Query Example
- **Question:** "How many spaces per indentation level are standard in a Python block?"
- **Status:** `SUPPORTED`
- **Generated Grounded Answer:** A standard Python block uses 4 spaces per indentation level. [01_python_syntax_and_types.md, Page 1 (§ Python Syntax, Variables, and Primitive Types) | Chunk doc_00bde289297e_p1_c001] Mixing tabs and spaces in Python 3 raises an `IndentationError`. [01_python_syntax_and_types.md, Page 1 (§ Python Syntax, Variables, and Primitive Types) | Chunk doc_00bde289297e_p1_c001] Syntax and Indentation
Python relies on whitespace indentation rather than curly brackets `{}` or keywords like `begin`/`end` to delimit code blocks. [01_python_syntax_and_types.md, Page 1 (§ Python Syntax, Variables, and Primitive Types) | Chunk doc_00bde289297e_p1_c001]
- **Citation Sources:** `['01_python_syntax_and_types.md']`
- **Latency:** 4.96 ms

### B. Unsupported Query Example (Mandatory Abstention)
- **Question:** "How do you define a struct in C++20 with designated initializers?"
- **Status:** `UNSUPPORTED`
- **Output:** "Information not found in provided document."
- **Expected Abstain:** True | **Actual Abstain:** True
- **Latency:** 3.36 ms

### C. Partially Supported Query Example
- **Question:** "Does Python support complex numbers, and what is the exact formula for quantum spin matrices in Python?"
- **Status:** `PARTIAL_EVIDENCE`
- **Output:** **Complex numbers (`complex`)**: Written in the form `real + imag*j`, where `j` represents the imaginary unit. [01_python_syntax_and_types.md, Page 1 (§ 3. Numeric Types) | Chunk doc_00bde289297e_p1_c007] Values can be any arbitrary Python object. [02_data_structures.md, Page 1 (§ 4. Dictionaries (`dict`)) | Chunk doc_13801f4030be_p1_c012] In Python 3.7+, dictionaries are guaranteed to maintain insertion order. [02_data_structures.md, Page 1 (§ 4. Dictionaries (`dict`)) | Chunk doc_13801f4030be_p1_c012]
- **Citation Sources:** `['01_python_syntax_and_types.md', '02_data_structures.md']`
- **Latency:** 11.05 ms
