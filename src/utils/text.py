"""Lightweight, dependency-free text utilities.

Deliberately avoids numpy/sklearn so the core scoring logic works
offline with zero heavyweight dependencies. Good enough for
approximate lexical similarity; evaluators can swap in an LLM judge
or embedding client for higher fidelity when API keys are configured.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "as", "and", "or",
    "but", "if", "so", "that", "this", "these", "those", "it", "its",
    "from", "into", "than", "then", "there", "their", "they", "them",
    "he", "she", "his", "her", "you", "your", "we", "our", "i", "do",
    "does", "did", "has", "have", "had", "not", "no", "which", "who",
    "whom", "what", "when", "where", "how", "why",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return _WORD_RE.findall((text or "").lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with common stopwords removed, for lexical overlap scoring."""
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter, good enough for claim decomposition."""
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def term_frequencies(tokens: list[str]) -> Counter:
    return Counter(tokens)


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity over raw term-frequency vectors (bag-of-words).

    Returns a value in [0, 1]. Empty strings yield 0.0.
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    vec_a = term_frequencies(tokens_a)
    vec_b = term_frequencies(tokens_b)
    shared = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    set_a = set(content_tokens(text_a))
    set_b = set(content_tokens(text_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def token_overlap_ratio(claim: str, source: str) -> float:
    """Fraction of claim's content tokens that also appear in source.

    Used as a cheap grounding/support proxy: how much of a claim's
    vocabulary is backed by the source text.
    """
    claim_tokens = set(content_tokens(claim))
    if not claim_tokens:
        return 0.0
    source_tokens = set(content_tokens(source))
    if not source_tokens:
        return 0.0
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def best_match_score(text: str, candidates: list[str], scorer=cosine_similarity) -> tuple[float, int]:
    """Return (best_score, best_index) of `text` against a list of candidates."""
    if not candidates:
        return 0.0, -1
    scores = [scorer(text, c) for c in candidates]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return scores[best_idx], best_idx
