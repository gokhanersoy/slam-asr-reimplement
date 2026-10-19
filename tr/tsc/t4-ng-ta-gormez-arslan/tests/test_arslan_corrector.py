"""Unit tests for Recep Sinan Arslan Output Correction module (Doktora Tezi, 2020)."""

import unittest
from slam_asr.utils.arslan_corrector import (
    ArslanOutputCorrector,
    fast_levenshtein,
    normalized_levenshtein,
)


class TestArslanCorrector(unittest.TestCase):
    def setUp(self):
        self.corrector = ArslanOutputCorrector(
            max_nl_distance=0.33,
            min_word_len=4,
            apply_colloquial_rules=True,
            apply_spelling_variants=True,
            apply_conjunction_rules=True,
        )

    def test_fast_levenshtein(self):
        self.assertEqual(fast_levenshtein("ankara", "ankara"), 0)
        self.assertEqual(fast_levenshtein("gel", "gelme"), 2)
        self.assertEqual(fast_levenshtein("kitap", "katip"), 2)

    def test_normalized_levenshtein(self):
        self.assertEqual(normalized_levenshtein("türkçe", "türkçe"), 0.0)
        # 1 substitution in 6 characters: 1/6 = 0.1667
        self.assertAlmostEqual(normalized_levenshtein("türkçe", "türkce"), 1 / 6, places=4)
        # Empty string handling
        self.assertEqual(normalized_levenshtein("", ""), 0.0)

    def test_in_vocabulary_preservation(self):
        """Valid words in the vocabulary must never be modified."""
        words = ["insanlar", "türkiye", "işsizlik", "çalışıyor", "bakanlık", "ekonomi"]
        for w in words:
            self.assertEqual(self.corrector.find_best_candidate(w), w)

    def test_short_word_protection(self):
        """Arslan Golden Rule: words with len <= 3 must not be modified."""
        short_words = ["ve", "de", "da", "bu", "şu", "o", "mi", "mu", "en", "ne", "ki", "son"]
        for w in short_words:
            self.assertEqual(self.corrector.find_best_candidate(w), w)

    def test_oov_correction_under_threshold(self):
        """OOV words within threshold <= 0.33 should be corrected to closest valid word."""
        # 'insanlat' (8 chars, 1 substitution 't'->'r' = 0.125 NL <= 0.33) -> 'insanlar'
        self.assertEqual(self.corrector.find_best_candidate("insanlat"), "insanlar")
        # 'çalışıyot' (9 chars, 1 substitution 't'->'r' = 0.111 NL <= 0.33) -> 'çalışıyor'
        self.assertEqual(self.corrector.find_best_candidate("çalışıyot"), "çalışıyor")

    def test_oov_rejected_over_threshold(self):
        """Arslan Golden Rule: distant OOV words (NL > 0.33) must NOT be replaced with unrelated words."""
        # 'xyzqw' is not close to any Turkish word
        self.assertEqual(self.corrector.find_best_candidate("xyzqw"), "xyzqw")

    def test_colloquial_verbalization(self):
        """Colloquial spoken forms must be normalized to canonical orthography."""
        text = "biliyosun yarın geliyo ve biz de yapcaz"
        corrected = self.corrector.correct_sentence(text)
        self.assertIn("biliyorsun", corrected)
        self.assertIn("geliyor", corrected)
        self.assertIn("yapacağız", corrected)

    def test_loanword_standardization(self):
        """Common loanword spelling variants must be mapped to standard Turkish."""
        text = "coronanın etkisi henüz bitmedi"
        corrected = self.corrector.correct_sentence(text)
        self.assertIn("koronanın", corrected)

    def test_conjunction_heuristics(self):
        """Disambiguate conjunction vs suffix spacing."""
        text = "arama zaman da bitti"
        corrected = self.corrector.correct_sentence(text)
        self.assertIn("zamanda", corrected)

    def test_full_pipeline_sample(self):
        """End-to-end sentence correction matching realistic ASR error patterns."""
        sample_asr = "mart biliyosun coronanın daha tam olarak etkilemediği"
        corrected = self.corrector.correct_sentence(sample_asr)
        self.assertEqual(corrected, "mart biliyorsun koronanın daha tam olarak etkilemediği")


if __name__ == "__main__":
    unittest.main()
