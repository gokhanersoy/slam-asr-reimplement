"""Evaluation metrics and text normalization for ASR benchmarks."""

import re
import string
from typing import List, Tuple, Union


def normalize_text_librispeech(text: str) -> str:
    """Standard text normalization for LibriSpeech evaluation:

    - Upper / lower casing (lowercase)
    - Remove punctuation (keep apostrophes if within words or remove)
    - Collapse multiple spaces
    - Strip whitespace
    """
    text = text.lower()
    # Remove punctuation except apostrophes in words
    text = re.sub(r"[^\w\s']", " ", text)
    # Remove stray numbers or symbols
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
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
            pred = normalize_text_librispeech(pred)
            ref = normalize_text_librispeech(ref)

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
) -> float:
    """Compute Character Error Rate (CER)."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    total_dist = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):
        if normalize:
            pred = normalize_text_librispeech(pred)
            ref = normalize_text_librispeech(ref)

        pred_chars = list(pred)
        ref_chars = list(ref)

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
