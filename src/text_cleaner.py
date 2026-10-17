"""Text cleaning and normalization module.

Provides deterministic cleaning, Unicode normalization, security sanitization,
Markdown frontmatter parsing, and tokenization utilities.
"""

import html
import re
import unicodedata
from typing import Dict, List, Set, Tuple


# Standard lightweight English stopwords for keyword overlap and BM25 tokenization
# Does not require downloading bulky external NLTK data.
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def sanitize_text(text: str) -> str:
    """Sanitize raw text against null bytes and illegal control characters."""
    if not text:
        return ""
    # Strip null bytes and non-printable control characters except tab and newline
    cleaned = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or unicodedata.category(ch)[0] != "C")
    return cleaned


def normalize_whitespace(text: str) -> str:
    """Normalize line endings, trim trailing whitespace, and collapse excessive blank lines."""
    if not text:
        return ""
    # Normalize Windows and legacy Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3 or more consecutive newlines into double newlines (paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_document_text(text: str) -> str:
    """Complete cleaning pipeline: NFKC normalization, control char strip, whitespace cleanup."""
    if not text:
        return ""
    # Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)
    # Strip null characters and malicious control characters
    sanitized = sanitize_text(normalized)
    # Normalize whitespace
    return normalize_whitespace(sanitized)


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Extract YAML-style frontmatter headers if present at the top of a Markdown document.

    Returns:
        A tuple of (metadata_dict, remaining_body_text).
    """
    metadata: Dict[str, str] = {}
    if not text or not text.startswith("---"):
        return metadata, text

    parts = text.split("---", 2)
    if len(parts) >= 3:
        frontmatter_block = parts[1]
        body_text = parts[2].lstrip()
        for line in frontmatter_block.splitlines():
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                clean_key = key.strip().lower()
                clean_val = val.strip().strip('"').strip("'")
                metadata[clean_key] = clean_val
        return metadata, body_text

    return metadata, text


def tokenize(text: str, remove_stopwords: bool = False, min_length: int = 2) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25 and lexical overlap."""
    if not text:
        return []
    # Match alphanumeric words
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) >= min_length]
    elif min_length > 1:
        tokens = [t for t in tokens if len(t) >= min_length]
    return tokens


def extract_keywords(text: str) -> Set[str]:
    """Extract unique meaningful content keywords from text."""
    return set(tokenize(text, remove_stopwords=True, min_length=2))


def safe_html_escape(text: str) -> str:
    """Escape text for safe HTML rendering in UI to prevent injection."""
    return html.escape(text, quote=True)
