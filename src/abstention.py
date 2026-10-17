"""Deterministic abstention policy and evidence sufficiency assessment.

Implements rigorous multi-tier checks against empty questions, out-of-domain queries,
unsupported claims, and insufficient evidence thresholds.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set

from src.config import config
from src.retriever import RetrievalResult
from src.text_cleaner import extract_keywords, tokenize


# Words commonly appearing in technical queries that do not specify the domain topic
GENERIC_QUESTION_TERMS: Set[str] = {
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    "does", "do", "did", "is", "are", "was", "were", "can", "could", "would",
    "should", "using", "use", "used", "make", "write", "find", "handle", "define",
    "difference", "between", "python", "explain", "example", "work", "occur",
    "happens", "happen", "many", "much", "give", "show", "tell", "state",
    "describe", "list", "create", "creating", "code", "run", "call", "called"
}


def simple_stem(word: str) -> str:
    """Lightweight rule-based English suffix normalizer for plural and tense inflections."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    return w


class EvidenceStatus(str, Enum):
    """Categorical status of evidence support for an answer."""
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    EMPTY_QUERY = "EMPTY_QUERY"


@dataclass
class AbstentionEvaluation:
    """Detailed diagnostic result of the abstention assessment."""
    status: EvidenceStatus
    should_abstain: bool
    reason: str
    top_score: float
    threshold: float
    keyword_overlap_ratio: float
    matched_keywords: Set[str]
    missing_keywords: Set[str]
    oov_keywords: Set[str]
    abstention_message: str = config.abstention.abstention_message


class AbstentionPolicy:
    """Evaluates whether retrieved passages provide sufficient grounded evidence to answer a query."""

    def __init__(
        self,
        bm25_threshold: Optional[float] = None,
        tfidf_threshold: Optional[float] = None,
        hybrid_threshold: Optional[float] = None,
        min_keyword_overlap: Optional[float] = None,
        max_oov_ratio: Optional[float] = None,
        partial_overlap_ratio: Optional[float] = None,
    ):
        self.bm25_threshold = bm25_threshold or config.abstention.bm25_min_score
        self.tfidf_threshold = tfidf_threshold or config.abstention.tfidf_min_score
        self.hybrid_threshold = hybrid_threshold or config.abstention.hybrid_min_score
        self.min_keyword_overlap = min_keyword_overlap or config.abstention.min_keyword_overlap_ratio
        self.max_oov_ratio = max_oov_ratio or config.abstention.max_oov_ratio_threshold
        self.partial_overlap_ratio = partial_overlap_ratio or config.abstention.partial_keyword_overlap_ratio

    def get_threshold_for_method(self, method: str) -> float:
        """Return the calibrated relevance threshold for a specific retrieval method."""
        method_lower = method.lower()
        if method_lower == "bm25":
            return self.bm25_threshold
        elif method_lower == "tfidf":
            return self.tfidf_threshold
        elif method_lower == "hybrid":
            return self.hybrid_threshold
        return self.hybrid_threshold

    def evaluate(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        method: str = "hybrid",
        corpus_text: Optional[str] = None,
    ) -> AbstentionEvaluation:
        """Run deterministic multi-tier evidence sufficiency checks."""
        # 1. Empty or whitespace question check
        clean_query = query.strip()
        if not clean_query:
            return AbstentionEvaluation(
                status=EvidenceStatus.EMPTY_QUERY,
                should_abstain=True,
                reason="Query is empty or contains only whitespace.",
                top_score=0.0,
                threshold=0.0,
                keyword_overlap_ratio=0.0,
                matched_keywords=set(),
                missing_keywords=set(),
                oov_keywords=set(),
            )

        threshold = self.get_threshold_for_method(method)

        # 2. Check for empty retrieval results
        if not retrieval_results:
            return AbstentionEvaluation(
                status=EvidenceStatus.UNSUPPORTED,
                should_abstain=True,
                reason="No matching passages found in the document corpus.",
                top_score=0.0,
                threshold=threshold,
                keyword_overlap_ratio=0.0,
                matched_keywords=set(),
                missing_keywords=extract_keywords(clean_query),
                oov_keywords=extract_keywords(clean_query),
            )

        top_result = retrieval_results[0]
        top_score = top_result.score

        # Combine text of top retrieved results for keyword overlap
        top_passages_text = " ".join(r.text for r in retrieval_results[:3])
        passage_stems = {simple_stem(w) for w in extract_keywords(top_passages_text)}

        # Extract core domain terms from query
        query_keywords = extract_keywords(clean_query)
        core_kw = {w for w in query_keywords if w not in GENERIC_QUESTION_TERMS and len(w) > 2}
        if not core_kw:
            core_kw = query_keywords if query_keywords else set(tokenize(clean_query, remove_stopwords=False))

        core_stems = {simple_stem(w) for w in core_kw}
        core_matched = core_stems.intersection(passage_stems)
        core_missing = core_stems - core_matched
        core_ratio = len(core_matched) / max(len(core_stems), 1)

        # Out-Of-Vocabulary check relative to corpus
        if corpus_text:
            corpus_stems = {simple_stem(w) for w in extract_keywords(corpus_text)}
        else:
            corpus_stems = passage_stems

        oov_stems = core_stems - corpus_stems
        oov_ratio = len(oov_stems) / max(len(core_stems), 1)

        # 3. Score threshold check
        if top_score < threshold:
            return AbstentionEvaluation(
                status=EvidenceStatus.UNSUPPORTED,
                should_abstain=True,
                reason=(
                    f"Top retrieval score ({top_score:.4f}) is below the calibrated "
                    f"confidence threshold ({threshold:.4f}) for {method.upper()}."
                ),
                top_score=top_score,
                threshold=threshold,
                keyword_overlap_ratio=core_ratio,
                matched_keywords=core_matched,
                missing_keywords=core_missing,
                oov_keywords=oov_stems,
            )

        # 4. Out-of-domain / Unsupported concept check
        if oov_ratio >= 0.65 or len(core_matched) == 0:
            return AbstentionEvaluation(
                status=EvidenceStatus.UNSUPPORTED,
                should_abstain=True,
                reason=(
                    f"Query domain concepts ({', '.join(sorted(oov_stems)) or 'all concepts'}) "
                    f"cannot be found in the document corpus."
                ),
                top_score=top_score,
                threshold=threshold,
                keyword_overlap_ratio=core_ratio,
                matched_keywords=core_matched,
                missing_keywords=core_missing,
                oov_keywords=oov_stems,
            )

        # 5. Weak lexical alignment with high out-of-vocabulary terms
        if oov_ratio >= 0.50 and len(core_matched) < 2:
            return AbstentionEvaluation(
                status=EvidenceStatus.UNSUPPORTED,
                should_abstain=True,
                reason=(
                    f"Significant proportion of query concepts ({oov_ratio:.1%}) are missing "
                    "from the document corpus."
                ),
                top_score=top_score,
                threshold=threshold,
                keyword_overlap_ratio=core_ratio,
                matched_keywords=core_matched,
                missing_keywords=core_missing,
                oov_keywords=oov_stems,
            )

        # 6. Partial evidence determination
        if oov_ratio > 0.0 or core_ratio < self.partial_overlap_ratio or len(core_missing) >= 2:
            return AbstentionEvaluation(
                status=EvidenceStatus.PARTIAL,
                should_abstain=False,
                reason=(
                    "Retrieved passages provide partial context regarding the topic, "
                    "but specific question concepts could not be fully resolved."
                ),
                top_score=top_score,
                threshold=threshold,
                keyword_overlap_ratio=core_ratio,
                matched_keywords=core_matched,
                missing_keywords=core_missing,
                oov_keywords=oov_stems,
            )

        # 7. Fully supported evidence
        return AbstentionEvaluation(
            status=EvidenceStatus.SUPPORTED,
            should_abstain=False,
            reason="Retrieved passages provide clear and grounded evidence for the query.",
            top_score=top_score,
            threshold=threshold,
            keyword_overlap_ratio=core_ratio,
            matched_keywords=core_matched,
            missing_keywords=core_missing,
            oov_keywords=oov_stems,
        )
