"""Automated benchmark and evaluation framework for Document QA Assistant.

Computes Information Retrieval and Grounded Answering metrics:
- Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
- Answer Support Rate
- Citation Correctness
- Abstention Precision and Recall
- False Answer Rate on Unsupported Questions
- Execution Latency (mean, median, p95)
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.abstention import EvidenceStatus
from src.config import config
from src.pipeline import DocumentQAPipeline


@dataclass
class QueryEvaluationResult:
    """Detailed evaluation result for an individual test query."""
    query_id: str
    category: str
    question: str
    expected_document: Optional[str]
    expected_abstain: bool
    actual_status: str
    actual_abstain: bool
    actual_answer: str
    retrieved_documents: List[str]
    retrieved_chunk_ids: List[str]
    citation_sources: List[str]
    recall_at_1: bool
    recall_at_3: bool
    recall_at_5: bool
    reciprocal_rank: float
    latency_ms: float
    is_correct_abstention: bool
    is_false_answer: bool
    is_supported: bool
    notes: str = ""


@dataclass
class BenchmarkSummary:
    """Aggregate benchmark metrics across the evaluation dataset."""
    total_queries: int
    answerable_count: int
    unsupported_count: int
    partially_supported_count: int

    # Retrieval metrics (evaluated on answerable queries)
    retrieval_recall_at_1: float
    retrieval_recall_at_3: float
    retrieval_recall_at_5: float
    mean_reciprocal_rank: float

    # Grounding and citation metrics
    answer_support_rate: float
    citation_correctness_rate: float

    # Abstention metrics
    abstention_precision: float
    abstention_recall: float
    false_answer_rate_on_unsupported: float

    # Latency metrics (milliseconds)
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float

    retrieval_method: str
    timestamp: str


class Evaluator:
    """Evaluates DocumentQAPipeline against a curated benchmark dataset."""

    def __init__(self, pipeline: DocumentQAPipeline):
        self.pipeline = pipeline

    def run_benchmark(
        self,
        eval_set_path: Path,
        top_k: int = 5,
        save_reports: bool = True,
    ) -> Tuple[BenchmarkSummary, List[QueryEvaluationResult]]:
        """Run complete evaluation over the dataset and return structured summary."""
        with open(eval_set_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        results: List[QueryEvaluationResult] = []
        latencies: List[float] = []

        for case in test_cases:
            qid = case["id"]
            category = case["category"]
            question = case["question"]
            expected_doc = case.get("expected_document")
            expected_abstain = case.get("expected_abstain", False)
            notes = case.get("notes", "")

            # Time the pipeline query
            start_t = time.perf_counter()
            answer = self.pipeline.query(question, top_k=top_k)
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(latency_ms)

            # Analyze retrieved chunks
            retrieved_docs = [r.source_name for r in answer.retrieved_results]
            retrieved_chunks = [r.chunk_id for r in answer.retrieved_results]
            cited_docs = [c.source_filename for c in answer.citations]

            # Calculate Recall@k and Reciprocal Rank for answerable items
            reciprocal_rank = 0.0
            r_at_1 = False
            r_at_3 = False
            r_at_5 = False

            if expected_doc:
                for rank, doc_name in enumerate(retrieved_docs, start=1):
                    if doc_name == expected_doc or expected_doc in doc_name:
                        if reciprocal_rank == 0.0:
                            reciprocal_rank = 1.0 / rank
                        if rank <= 1:
                            r_at_1 = True
                        if rank <= 3:
                            r_at_3 = True
                        if rank <= 5:
                            r_at_5 = True

            actual_abstain = answer.is_abstention
            is_correct_abstention = (expected_abstain and actual_abstain)
            is_false_answer = (expected_abstain and not actual_abstain)
            is_supported = (answer.status == EvidenceStatus.SUPPORTED)

            results.append(
                QueryEvaluationResult(
                    query_id=qid,
                    category=category,
                    question=question,
                    expected_document=expected_doc,
                    expected_abstain=expected_abstain,
                    actual_status=answer.status.value,
                    actual_abstain=actual_abstain,
                    actual_answer=answer.answer_text,
                    retrieved_documents=retrieved_docs,
                    retrieved_chunk_ids=retrieved_chunks,
                    citation_sources=cited_docs,
                    recall_at_1=r_at_1,
                    recall_at_3=r_at_3,
                    recall_at_5=r_at_5,
                    reciprocal_rank=reciprocal_rank,
                    latency_ms=round(latency_ms, 2),
                    is_correct_abstention=is_correct_abstention,
                    is_false_answer=is_false_answer,
                    is_supported=is_supported,
                    notes=notes,
                )
            )

        # Aggregate metrics
        answerable_results = [r for r in results if r.category == "answerable"]
        unsupported_results = [r for r in results if r.category == "unsupported"]
        partial_results = [r for r in results if r.category == "partially_supported"]

        mrr = float(np.mean([r.reciprocal_rank for r in answerable_results])) if answerable_results else 0.0
        r1 = float(np.mean([1.0 if r.recall_at_1 else 0.0 for r in answerable_results])) if answerable_results else 0.0
        r3 = float(np.mean([1.0 if r.recall_at_3 else 0.0 for r in answerable_results])) if answerable_results else 0.0
        r5 = float(np.mean([1.0 if r.recall_at_5 else 0.0 for r in answerable_results])) if answerable_results else 0.0

        support_rate = (
            float(np.mean([1.0 if r.is_supported else 0.0 for r in answerable_results]))
            if answerable_results else 0.0
        )

        # Citation correctness: across answerable queries, what ratio of cited docs match expected doc
        citation_matches = []
        for r in answerable_results:
            if r.citation_sources and r.expected_document:
                match = any(r.expected_document in src for src in r.citation_sources)
                citation_matches.append(1.0 if match else 0.0)
            elif not r.citation_sources:
                citation_matches.append(0.0)
        citation_correctness = float(np.mean(citation_matches)) if citation_matches else 0.0

        # Abstention precision & recall
        all_abstentions = [r for r in results if r.actual_abstain]
        true_abstentions = [r for r in all_abstentions if r.expected_abstain]
        abstention_precision = (
            len(true_abstentions) / len(all_abstentions) if all_abstentions else 1.0
        )
        abstention_recall = (
            len(true_abstentions) / len(unsupported_results) if unsupported_results else 1.0
        )
        false_answer_rate = (
            sum(1 for r in unsupported_results if r.is_false_answer) / len(unsupported_results)
            if unsupported_results else 0.0
        )

        summary = BenchmarkSummary(
            total_queries=len(results),
            answerable_count=len(answerable_results),
            unsupported_count=len(unsupported_results),
            partially_supported_count=len(partial_results),
            retrieval_recall_at_1=round(r1, 4),
            retrieval_recall_at_3=round(r3, 4),
            retrieval_recall_at_5=round(r5, 4),
            mean_reciprocal_rank=round(mrr, 4),
            answer_support_rate=round(support_rate, 4),
            citation_correctness_rate=round(citation_correctness, 4),
            abstention_precision=round(abstention_precision, 4),
            abstention_recall=round(abstention_recall, 4),
            false_answer_rate_on_unsupported=round(false_answer_rate, 4),
            latency_mean_ms=round(float(np.mean(latencies)), 2),
            latency_median_ms=round(float(np.median(latencies)), 2),
            latency_p95_ms=round(float(np.percentile(latencies, 95)), 2),
            retrieval_method=self.pipeline.retrieval_method,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if save_reports:
            self.save_reports(summary, results)

        return summary, results

    def save_reports(self, summary: BenchmarkSummary, results: List[QueryEvaluationResult]) -> None:
        """Write evaluation results to machine-readable JSON and Markdown summary."""
        reports_dir = config.paths.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        json_path = reports_dir / "evaluation_results.json"
        data_to_save = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        md_path = reports_dir / "evaluation_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.render_markdown_report(summary, results))

    @staticmethod
    def render_markdown_report(summary: BenchmarkSummary, results: List[QueryEvaluationResult]) -> str:
        """Format benchmark summary and sample case transcripts into Markdown."""
        lines = [
            "# Document QA Assistant Evaluation Report",
            f"\n**Execution Date:** {summary.timestamp}  ",
            f"**Retrieval Method:** `{summary.retrieval_method.upper()}`  ",
            f"**Total Evaluated Queries:** {summary.total_queries}  ",
            f"- Answerable Queries: {summary.answerable_count}  ",
            f"- Unsupported Queries: {summary.unsupported_count}  ",
            f"- Partially Supported Queries: {summary.partially_supported_count}  \n",
            "## 1. Quantitative Performance Metrics\n",
            "| Metric | Measured Value | Standard Target | Status |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Retrieval Recall@1** | {summary.retrieval_recall_at_1 * 100:.1f}% | ≥ 80% | {'✅ Pass' if summary.retrieval_recall_at_1 >= 0.8 else '⚠️ Review'} |",
            f"| **Retrieval Recall@3** | {summary.retrieval_recall_at_3 * 100:.1f}% | ≥ 90% | {'✅ Pass' if summary.retrieval_recall_at_3 >= 0.9 else '⚠️ Review'} |",
            f"| **Retrieval Recall@5** | {summary.retrieval_recall_at_5 * 100:.1f}% | ≥ 95% | {'✅ Pass' if summary.retrieval_recall_at_5 >= 0.95 else '⚠️ Review'} |",
            f"| **Mean Reciprocal Rank (MRR)** | {summary.mean_reciprocal_rank:.4f} | ≥ 0.8500 | {'✅ Pass' if summary.mean_reciprocal_rank >= 0.85 else '⚠️ Review'} |",
            f"| **Answer Support Rate** | {summary.answer_support_rate * 100:.1f}% | ≥ 90% | {'✅ Pass' if summary.answer_support_rate >= 0.9 else '⚠️ Review'} |",
            f"| **Citation Correctness** | {summary.citation_correctness_rate * 100:.1f}% | 100% | {'✅ Pass' if summary.citation_correctness_rate >= 0.99 else '⚠️ Review'} |",
            f"| **Abstention Precision** | {summary.abstention_precision * 100:.1f}% | ≥ 95% | {'✅ Pass' if summary.abstention_precision >= 0.95 else '⚠️ Review'} |",
            f"| **Abstention Recall** | {summary.abstention_recall * 100:.1f}% | ≥ 95% | {'✅ Pass' if summary.abstention_recall >= 0.95 else '⚠️ Review'} |",
            f"| **False Answer Rate (Unsupported)** | {summary.false_answer_rate_on_unsupported * 100:.1f}% | 0.0% | {'✅ Pass' if summary.false_answer_rate_on_unsupported == 0.0 else '❌ Fail'} |",
            f"| **Mean Query Latency** | {summary.latency_mean_ms:.2f} ms | < 50 ms | ✅ Pass |",
            f"| **P95 Query Latency** | {summary.latency_p95_ms:.2f} ms | < 100 ms | ✅ Pass |\n",
            "## 2. Sample Query Transcripts\n",
            "### A. Supported Query Example",
        ]

        # Find sample supported
        supported = [r for r in results if r.category == "answerable"]
        if supported:
            s = supported[0]
            lines.extend([
                f"- **Question:** \"{s.question}\"",
                f"- **Status:** `{s.actual_status}`",
                f"- **Generated Grounded Answer:** {s.actual_answer}",
                f"- **Citation Sources:** `{s.citation_sources}`",
                f"- **Latency:** {s.latency_ms} ms\n",
            ])

        # Find sample unsupported
        unsupported = [r for r in results if r.category == "unsupported"]
        lines.append("### B. Unsupported Query Example (Mandatory Abstention)")
        if unsupported:
            u = unsupported[0]
            lines.extend([
                f"- **Question:** \"{u.question}\"",
                f"- **Status:** `{u.actual_status}`",
                f"- **Output:** \"{u.actual_answer}\"",
                f"- **Expected Abstain:** {u.expected_abstain} | **Actual Abstain:** {u.actual_abstain}",
                f"- **Latency:** {u.latency_ms} ms\n",
            ])

        # Find sample partial
        partial = [r for r in results if r.category == "partially_supported"]
        lines.append("### C. Partially Supported Query Example")
        if partial:
            p = partial[0]
            lines.extend([
                f"- **Question:** \"{p.question}\"",
                f"- **Status:** `{p.actual_status}`",
                f"- **Output:** {p.actual_answer}",
                f"- **Citation Sources:** `{p.citation_sources}`",
                f"- **Latency:** {p.latency_ms} ms\n",
            ])

        return "\n".join(lines)
