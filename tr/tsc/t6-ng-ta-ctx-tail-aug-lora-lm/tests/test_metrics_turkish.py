"""Unit tests for Turkish text normalization and ASR metrics."""

import unittest
from slam_asr.utils.metrics import (
    compute_cer,
    compute_wer,
    levenshtein_distance,
    normalize_text_turkish,
)


class TestTurkishMetrics(unittest.TestCase):
    def test_turkish_case_folding(self):
        # Critical test: 'I' -> 'ı' and 'İ' -> 'i'
        raw = "IŞIK ILIK İSTANBUL İÇİN"
        expected = "ışık ılık istanbul için"
        self.assertEqual(normalize_text_turkish(raw), expected)

    def test_turkish_special_characters(self):
        raw = "ÇALIŞMA GÖZLEMİ ŞÜPHELİ ÇAĞLAYAN ÖRDEK ÜZÜM"
        expected = "çalışma gözlemi şüpheli çağlayan ördek üzüm"
        self.assertEqual(normalize_text_turkish(raw), expected)

    def test_punctuation_removal(self):
        raw = "Merhaba, dünya! Bugün hava çok güzel; değil mi?"
        expected = "merhaba dünya bugün hava çok güzel değil mi"
        self.assertEqual(normalize_text_turkish(raw), expected)

    def test_wer_exact_match(self):
        refs = ["türkiye cumhuriyeti yüzüncü yılını kutluyor"]
        preds = ["türkiye cumhuriyeti yüzüncü yılını kutluyor"]
        wer = compute_wer(preds, refs, language="tr")
        self.assertEqual(wer, 0.0)

    def test_wer_with_normalization(self):
        refs = ["Türkiye Cumhuriyeti, 100. yılını kutluyor!"]
        preds = ["türkiye cumhuriyeti 100 yılını kutluyor"]
        wer = compute_wer(preds, refs, language="tr")
        self.assertEqual(wer, 0.0)

    def test_wer_single_word_substitution(self):
        refs = ["bu bir yapay zeka modelidir"]
        preds = ["bu bir derin zeka modelidir"]
        # 1 substitution out of 5 words = 20%
        wer = compute_wer(preds, refs, language="tr")
        self.assertAlmostEqual(wer, 20.0, places=2)

    def test_cer_calculation(self):
        refs = ["kelime"]
        preds = ["kalime"]
        # 1 char difference out of 6 chars = 16.67%
        cer = compute_cer(preds, refs, language="tr")
        self.assertAlmostEqual(cer, 100.0 / 6.0, places=2)


if __name__ == "__main__":
    unittest.main()
