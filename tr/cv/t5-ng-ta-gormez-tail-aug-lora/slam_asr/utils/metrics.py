"""Evaluation metrics and text normalization for ASR benchmarks (Turkish & Multilingual)."""

import re
import string
from typing import Callable, List, Optional, Tuple, Union


def normalize_text_turkish(text: str) -> str:
    """Accurate Turkish text normalization for ASR evaluation:
    - Handles Turkish dotted and undotted I: 'İ' -> 'i', 'I' -> 'ı'
    - Preserves Turkish characters: ç, ğ, ı, ö, ş, ü, â, î, û
    - Strips punctuation and symbols
    - Collapses multiple whitespace
    """
    if not text:
        return ""
    # Explicit Turkish uppercase mapping before .lower()
    # (Python str.lower() incorrectly converts 'I' to 'i' on standard locales)
    text = text.replace("İ", "i").replace("I", "ı")
    # Circumflex vowels
    text = text.replace("Â", "a").replace("â", "a")
    text = text.replace("Î", "i").replace("î", "i")
    text = text.replace("Û", "u").replace("û", "u")
    text = text.lower()

    # Remove punctuation except word characters and whitespace
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Remove underscores (which \w includes)
    text = text.replace("_", " ")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_librispeech(text: str) -> str:
    """Standard text normalization for LibriSpeech evaluation:
    - Lowercase
    - Remove punctuation
    - Collapse multiple spaces
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str, language: str = "tr") -> str:
    """Select appropriate normalizer based on language."""
    if language == "tr":
        return normalize_text_turkish(text)
    return normalize_text_librispeech(text)


def compute_wer(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
    language: str = "tr",
) -> float:
    """Compute Word Error Rate (WER) between hypothesis and reference transcripts.

    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    total_dist = 0
    total_words = 0

    for pred, ref in zip(predictions, references):
        if normalize:
            pred = normalize_text(pred, language=language)
            ref = normalize_text(ref, language=language)

        pred_words = pred.split()
        ref_words = ref.split()

        total_words += len(ref_words)
        total_dist += levenshtein_distance(pred_words, ref_words)

    if total_words == 0:
        return 0.0

    return (total_dist / total_words) * 100.0


def compute_cer(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
    language: str = "tr",
) -> float:
    """Compute Character Error Rate (CER)."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    total_dist = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):
        if normalize:
            pred = normalize_text(pred, language=language)
            ref = normalize_text(ref, language=language)

        pred_chars = list(pred.replace(" ", ""))
        ref_chars = list(ref.replace(" ", ""))

        total_chars += len(ref_chars)
        total_dist += levenshtein_distance(pred_chars, ref_chars)

    if total_chars == 0:
        return 0.0

    return (total_dist / total_chars) * 100.0


def levenshtein_distance(seq1: List[Union[str, int]], seq2: List[Union[str, int]]) -> int:
    """Dynamic programming implementation of Levenshtein distance."""
    n = len(seq1)
    m = len(seq2)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[n][m]
