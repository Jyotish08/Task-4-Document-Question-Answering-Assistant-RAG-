"""Automated tests for evaluation framework and benchmark metric computations."""

from pathlib import Path
import json
import pytest

from src.config import config
from src.evaluation import Evaluator
from src.pipeline import DocumentQAPipeline


@pytest.fixture(scope="module")
def shared_pipeline():
    """Create and index sample corpus once for evaluation tests."""
    pipeline = DocumentQAPipeline(retrieval_method="hybrid")
    pipeline.ingest_directory(config.paths.sample_docs_dir)
    return pipeline


def test_evaluation_dataset_structure():
    """Verify evaluation dataset meets the mandatory minimum count requirements."""
    eval_path = config.paths.data_dir / "evaluation_set.json"
    assert eval_path.exists(), "data/evaluation_set.json must exist"

    with open(eval_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    answerable = [c for c in cases if c["category"] == "answerable"]
    unsupported = [c for c in cases if c["category"] == "unsupported"]
    partial = [c for c in cases if c["category"] == "partially_supported"]

    assert len(answerable) >= 15, "Must have at least 15 answerable queries"
    assert len(unsupported) >= 15, "Must have at least 15 unsupported queries"
    assert len(partial) >= 5, "Must have at least 5 partially supported queries"
    assert len(cases) >= 35, "Total evaluation set must have at least 35 test cases"


def test_evaluator_runs_benchmark_and_meets_targets(shared_pipeline, tmp_path: Path):
    """Verify benchmark metrics meet all quality standards."""
    eval_path = config.paths.data_dir / "evaluation_set.json"
    evaluator = Evaluator(shared_pipeline)

    summary, results = evaluator.run_benchmark(eval_path, top_k=5, save_reports=False)

    assert summary.total_queries >= 35
    assert summary.retrieval_recall_at_1 >= 0.80
    assert summary.retrieval_recall_at_3 >= 0.90
    assert summary.mean_reciprocal_rank >= 0.85
    assert summary.citation_correctness_rate >= 0.95
    assert summary.abstention_precision >= 0.95
    assert summary.abstention_recall >= 0.95
    assert summary.false_answer_rate_on_unsupported == 0.0
    assert summary.latency_mean_ms < 50.0  # Fast local execution


def test_evaluator_generates_markdown_and_json(shared_pipeline, tmp_path: Path):
    """Verify evaluator creates valid machine-readable JSON and Markdown reports."""
    eval_path = config.paths.data_dir / "evaluation_set.json"
    evaluator = Evaluator(shared_pipeline)

    # Temporary override of reports_dir
    config.paths.reports_dir = tmp_path / "test_reports"
    summary, results = evaluator.run_benchmark(eval_path, top_k=3, save_reports=True)

    json_report = config.paths.reports_dir / "evaluation_results.json"
    md_report = config.paths.reports_dir / "evaluation_summary.md"

    assert json_report.exists()
    assert md_report.exists()

    with open(json_report, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "summary" in data
    assert "results" in data
    assert len(data["results"]) == summary.total_queries

    md_content = md_report.read_text(encoding="utf-8")
    assert "# Document QA Assistant Evaluation Report" in md_content
    assert "Quantitative Performance Metrics" in md_content
