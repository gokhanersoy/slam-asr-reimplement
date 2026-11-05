"""Unit tests for Zemberek Turkish Morphological Language Model (slam_asr.utils.zemberek_lm)."""

import unittest
from slam_asr.utils.zemberek_lm import (
    ZemberekLanguageModel,
    check_vowel_harmony,
    fast_levenshtein,
)


class TestZemberekLM(unittest.TestCase):
    """Test suite for Zemberek Turkish Language Model & Output Rescorer."""

    def setUp(self):
        self.lm = ZemberekLanguageModel(ngram_order=3, alpha_smoothing=0.05)

    def test_initialization(self):
        """Verify model loads core vocabulary and sets valid token counters."""
        self.assertGreater(len(self.lm.vocab), 100)
        self.assertGreater(self.lm.total_tokens, 100)
        self.assertTrue(self.lm.is_valid_word("ev"))
        self.assertTrue(self.lm.is_valid_word("insan"))
        self.assertTrue(self.lm.is_valid_word("kitap"))

    def test_vowel_harmony(self):
        """Verify Turkish major vowel harmony checks (Kalınlık-İncelik)."""
        # Back vowels (a, ı, o, u) + back suffix (lar, da, dan)
        self.assertTrue(check_vowel_harmony("kitap", "lar"))
        self.assertTrue(check_vowel_harmony("araba", "da"))
        self.assertTrue(check_vowel_harmony("okul", "dan"))

        # Front vowels (e, i, ö, ü) + front suffix (ler, de, den)
        self.assertTrue(check_vowel_harmony("ev", "ler"))
        self.assertTrue(check_vowel_harmony("köy", "de"))
        self.assertTrue(check_vowel_harmony("göz", "den"))

        # Incompatible harmony
        self.assertFalse(check_vowel_harmony("ev", "lar"))
        self.assertFalse(check_vowel_harmony("kitap", "ler"))

    def test_morphological_validity(self):
        """Verify root + suffix combination validity checks."""
        self.assertTrue(self.lm.is_valid_word("evler"))
        self.assertTrue(self.lm.is_valid_word("kitaplar"))
        self.assertTrue(self.lm.is_valid_word("odada"))
        self.assertTrue(self.lm.is_valid_word("geldi"))
        self.assertTrue(self.lm.is_valid_word("görüyor"))

    def test_fast_levenshtein(self):
        """Verify edit distance computation."""
        self.assertEqual(fast_levenshtein("ev", "ev"), 0)
        self.assertEqual(fast_levenshtein("ev", "evler"), 3)
        self.assertEqual(fast_levenshtein("kitap", "katap"), 1)

    def test_ngram_scoring_and_perplexity(self):
        """Verify n-gram sentence log likelihood and perplexity."""
        # Train on a small Turkish corpus
        corpus = [
            "türkiye büyük millet meclisi bugün toplandı",
            "yeni bir konuşma tanıma modeli geliştirdik",
            "bu araştırma kapsamında türkçe veri kümesi incelendi",
        ]
        for s in corpus:
            self.lm.train_on_sentence(s)

        score_known = self.lm.score_sentence("türkiye büyük millet meclisi bugün toplandı")
        score_rare = self.lm.score_sentence("uçak tren meclisi garip biçimde koştu")

        # Known frequent sequence should have higher log-likelihood (less negative)
        self.assertGreater(score_known, score_rare)

        # Perplexity should be finite and positive
        ppl = self.lm.get_perplexity("yeni bir konuşma tanıma modeli geliştirdik")
        self.assertGreater(ppl, 1.0)
        self.assertLess(ppl, 10000.0)

    def test_candidate_generation(self):
        """Verify that plausible candidates are generated for misspelled words."""
        candidates = self.lm.generate_candidates("evler")
        self.assertIn("evler", candidates)
        self.assertIn("ev", candidates)

    def test_sentence_rescoring(self):
        """Verify sentence rescoring preserves natural valid Turkish words."""
        sentence = "bu araştırma kapsamında türkçe konuşma tanıma yapıldı"
        for s in [sentence]:
            self.lm.train_on_sentence(s)

        corrected = self.lm.correct_sentence(sentence)
        self.assertEqual(corrected, sentence)


if __name__ == "__main__":
    unittest.main()
