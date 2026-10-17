"""Command-Line Interface (CLI) for the Document Question-Answering Assistant.

Provides interactive commands to ingest documents, run grounded retrieval queries,
execute benchmark evaluations, inspect corpus status, and manage the local index.
"""

import argparse
from pathlib import Path
import sys

from src.config import config
from src.evaluation import Evaluator
from src.pipeline import DocumentQAPipeline


def get_pipeline(method: str = "hybrid") -> DocumentQAPipeline:
    """Initialize pipeline and attempt to restore saved index state."""
    pipeline = DocumentQAPipeline(retrieval_method=method)
    if (config.paths.storage_dir / "index_metadata.json").exists():
        pipeline.load_state()
    return pipeline


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a file or directory into the pipeline."""
    target_path = Path(args.path)
    if not target_path.exists():
        print(f"[Error] Target path does not exist: {target_path}")
        sys.exit(1)

    pipeline = get_pipeline(args.method)

    if target_path.is_file():
        doc = pipeline.ingest_file(target_path)
        print(f"Successfully ingested file: {doc.filename}")
        print(f"  Title: {doc.title}")
        print(f"  Pages: {len(doc.pages)} | Characters: {doc.total_chars}")
        print(f"  Total chunks indexed in corpus: {len(pipeline.chunks)}")
    elif target_path.is_dir():
        count = pipeline.ingest_directory(target_path)
        print(f"Successfully ingested directory: {target_path}")
        print(f"  Documents indexed: {len(pipeline.documents)}")
        print(f"  Total chunks indexed: {count}")
    else:
        print(f"[Error] Invalid path type: {target_path}")
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    """Run a grounded QA query against the indexed documents."""
    pipeline = get_pipeline(args.method)
    if not pipeline.chunks:
        print("[Warning] No documents are currently indexed.")
        print(f"Run `python cli.py ingest {config.paths.sample_docs_dir}` to load sample corpus.")
        sys.exit(1)

    answer = pipeline.query(args.question, top_k=args.top_k)
    print("\n" + "=" * 60)
    print(f"Question: {args.question}")
    print(f"Status: {answer.status.value}")
    print("=" * 60)
    print(f"\n{answer.answer_text}\n")

    if answer.uncertainty_note:
        print(f"Note on evidence limits: {answer.uncertainty_note}\n")

    if answer.citations:
        print("-" * 60)
        print("Supporting Evidence & Source Citations:")
        for idx, cite in enumerate(answer.citations, start=1):
            page_str = f", Page {cite.page_number}" if cite.page_number else ""
            section_str = f" (§ {cite.section})" if cite.section and cite.section != "General" else ""
            print(f"  [{idx}] {cite.source_filename}{page_str}{section_str} [ID: {cite.chunk_id}]")
            print(f"      Score: {cite.score:.4f}")
            print(f"      Excerpt: \"{cite.snippet}\"")
        print("-" * 60)

    if not answer.validation_passed:
        print(f"[Warning] Citation validation errors: {answer.validation_errors}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run the comprehensive benchmark evaluation."""
    eval_set_path = Path(args.eval_set) if args.eval_set else (config.paths.data_dir / "evaluation_set.json")
    if not eval_set_path.exists():
        print(f"[Error] Evaluation dataset not found at: {eval_set_path}")
        sys.exit(1)

    pipeline = get_pipeline(args.method)
    if not pipeline.chunks:
        print("Corpus is empty. Ingesting sample docs for benchmark...")
        pipeline.ingest_directory(config.paths.sample_docs_dir)

    print(f"Running evaluation benchmark on {eval_set_path.name} using {args.method.upper()}...")
    evaluator = Evaluator(pipeline)
    summary, _ = evaluator.run_benchmark(eval_set_path, top_k=args.top_k, save_reports=True)

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Evaluated Queries: {summary.total_queries}")
    print(f"  Retrieval Recall@1: {summary.retrieval_recall_at_1 * 100:.1f}%")
    print(f"  Retrieval Recall@3: {summary.retrieval_recall_at_3 * 100:.1f}%")
    print(f"  Retrieval Recall@5: {summary.retrieval_recall_at_5 * 100:.1f}%")
    print(f"  Mean Reciprocal Rank (MRR): {summary.mean_reciprocal_rank:.4f}")
    print(f"  Answer Support Rate: {summary.answer_support_rate * 100:.1f}%")
    print(f"  Citation Correctness: {summary.citation_correctness_rate * 100:.1f}%")
    print(f"  Abstention Precision: {summary.abstention_precision * 100:.1f}%")
    print(f"  Abstention Recall: {summary.abstention_recall * 100:.1f}%")
    print(f"  False Answer Rate (Unsupported): {summary.false_answer_rate_on_unsupported * 100:.1f}%")
    print(f"  Mean Latency: {summary.latency_mean_ms:.2f} ms")
    print(f"  P95 Latency: {summary.latency_p95_ms:.2f} ms")
    print("=" * 60)
    print(f"Reports saved to:")
    print(f"  JSON: {config.paths.reports_dir / 'evaluation_results.json'}")
    print(f"  Markdown: {config.paths.reports_dir / 'evaluation_summary.md'}")


def cmd_status(args: argparse.Namespace) -> None:
    """Display index and corpus status."""
    pipeline = get_pipeline()
    docs = pipeline.get_document_summary()
    print("\n" + "=" * 60)
    print("DOCUMENT QA ASSISTANT STATUS")
    print("=" * 60)
    print(f"Indexed Documents: {len(docs)}")
    print(f"Total Chunks: {len(pipeline.chunks)}")
    print(f"Retrieval Method: {pipeline.retrieval_method.upper()}")
    print(f"Storage Directory: {pipeline.storage_dir}")
    print("-" * 60)
    if docs:
        for d in docs:
            print(f"- {d['filename']} [{d['file_type']}]")
            print(f"    Title: {d['title']}")
            print(f"    Pages: {d['pages']} | Chunks: {d['chunk_count']} | Chars: {d['total_chars']}")
            print(f"    Source: {d['source_url']}")
    else:
        print("No documents currently indexed.")
    print("=" * 60)


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear all indexed documents and reset storage."""
    pipeline = get_pipeline()
    pipeline.clear()
    print("Corpus index cleared successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Local-First Document Question-Answering Assistant (RAG)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest command
    p_ingest = subparsers.add_parser("ingest", help="Ingest a document file or directory")
    p_ingest.add_argument("path", help="Path to file (.txt, .md, .pdf) or directory")
    p_ingest.add_argument("--method", default="hybrid", choices=["bm25", "tfidf", "hybrid"], help="Retriever method")

    # Query command
    p_query = subparsers.add_parser("query", help="Ask a question against the corpus")
    p_query.add_argument("question", help="Question text")
    p_query.add_argument("--method", default="hybrid", choices=["bm25", "tfidf", "hybrid"], help="Retriever method")
    p_query.add_argument("--top-k", type=int, default=4, help="Number of passages to retrieve")

    # Evaluate command
    p_eval = subparsers.add_parser("evaluate", help="Run benchmark evaluation")
    p_eval.add_argument("--eval-set", help="Path to evaluation_set.json")
    p_eval.add_argument("--method", default="hybrid", choices=["bm25", "tfidf", "hybrid"], help="Retriever method")
    p_eval.add_argument("--top-k", type=int, default=5, help="Number of passages to retrieve")

    # Status command
    subparsers.add_parser("status", help="Show currently indexed documents and index status")

    # Clear command
    subparsers.add_parser("clear", help="Clear all indexed documents")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "clear":
        cmd_clear(args)


if __name__ == "__main__":
    main()
