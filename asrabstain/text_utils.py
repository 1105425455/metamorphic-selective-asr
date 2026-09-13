"""Text normalisation and tokenisation, English only.

Lowercase, strip punctuation, split on whitespace. Word error rate uses
kaldialign when available, with a pure-Python fallback.
"""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s']")
_WS_RE = re.compile(r"\s+")


def normalize_en(text: str) -> str:
    """English normalisation: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def tokenize_en(text: str) -> list[str]:
    """Tokenise normalised text on whitespace."""
    norm = normalize_en(text)
    return norm.split() if norm else []


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance, via kaldialign when available.

    The fallback keeps evaluation usable without kaldialign; the algorithm is
    equivalent (word-level edit distance).
    """
    try:
        from kaldialign import edit_distance as _ed
        result = _ed(ref, hyp)
        return result.get("total", 0) if isinstance(result, dict) else result
    except ImportError:
        m, n = len(ref), len(hyp)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            cur = [i] + [0] * n
            ri = ref[i - 1]
            for j in range(1, n + 1):
                cost = 0 if ri == hyp[j - 1] else 1
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            prev = cur
        return prev[n]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: edit distance over the reference length."""
    ref_tokens = tokenize_en(reference)
    hyp_tokens = tokenize_en(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def norm_edit_distance(reference: str, hypothesis: str) -> float:
    """Normalised edit distance: Levenshtein over the longer token count."""
    ref_tokens = tokenize_en(reference)
    hyp_tokens = tokenize_en(hypothesis)
    return _edit_distance(ref_tokens, hyp_tokens) / max(
        len(ref_tokens), len(hyp_tokens), 1)
