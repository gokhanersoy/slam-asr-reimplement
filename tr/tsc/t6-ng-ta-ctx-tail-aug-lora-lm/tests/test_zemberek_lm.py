"""Unit tests for Zemberek Language Model and Morphological Rescorer."""

import os
import shutil
import tempfile
import unittest
import json

from slam_asr.utils.zemberek_lm import (
    ZemberekLanguageModel,
    TurkishMorphologyAnalyzer,
    check_vowel_harmony,
    fast_levenshtein,
)


class TestZemberekLM(unittest.TestCase):
    """Test suite for Zemberek Language Model and Turkish morphology processing."""

    def setUp(self):
        self.sample_sentences = [
            "türkiye büyük millet meclisi ankarada toplandı",
            "konuşma tanıma sistemleri yapay zeka ile gelişiyor",
            "bu bölümde dinleyicileri aldık",
            "öğrenciler kütüphanede ders çalışıyor",
            "yeni nesil derin öğrenme modelleri başarı gösteriyor",
        ]
        self.temp_dir = tempfile.mkdtemp()
        self.manifest_path = os.path.join(self.temp_dir, "test_manifest.jsonl")
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for s in self.sample_sentences:
                f.write(json.dumps({"transcript": s}, ensure_ascii=False) + "\n")

        self.lm = ZemberekLanguageModel(manifest_path=self.manifest_path)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_vowel_harmony(self):
        """Verify vowel harmony checks."""
        self.assertTrue(check_vowel_harmony("adımlarımızdan"))
        self.assertTrue(check_vowel_harmony("evlerimizden"))
        self.assertTrue(check_vowel_harmony("çocuklar"))
        self.assertTrue(check_vowel_harmony("kitaplar"))  # last two vowels harmonize

    def test_morphology_analyzer(self):
        """Verify morphology analyzer strips suffixes and identifies roots."""
        analyzer = TurkishMorphologyAnalyzer(known_roots={"konuşma", "dinleyici", "kitap", "ders"})
        analyses = analyzer.analyze("dinleyicileri")
        self.assertTrue(len(analyses) > 0)
        self.assertTrue(any("dinleyici" in a["root"] for a in analyses))
        self.assertTrue(analyzer.is_valid_morphology("dinleyicileri"))

    def test_lm_scoring(self):
        """Verify LM gives higher score to natural in-domain sentences than random words."""
        natural_score = self.lm.score_sentence("konuşma tanıma sistemleri yapay zeka ile gelişiyor")
        unnatural_score = self.lm.score_sentence("meclisi zeka ders kütüphanede yapay ankarada")
        self.assertGreater(natural_score, unnatural_score)

    def test_perplexity(self):
        """Verify perplexity returns a positive finite number."""
        ppl = self.lm.perplexity("konuşma tanıma sistemleri gelişiyor")
        self.assertGreater(ppl, 1.0)
        self.assertFalse(float("inf") == ppl)

    def test_spelling_correction(self):
        """Verify misspelled word candidate correction."""
        # 'öğrenciler' is in vocab, 'öğrencilerr' has a double-r typo
        corrected = self.lm.correct_word("öğrencilerr")
        self.assertEqual(corrected, "öğrenciler")

    def test_hypothesis_rescoring(self):
        """Verify N-best hypotheses rescoring favors fluent sentence."""
        hyps = [
            {"text": "meclisi zeka ders kütüphanede", "score": -1.0},
            {"text": "konuşma tanıma sistemleri yapay zeka ile gelişiyor", "score": -1.5},
        ]
        rescored = self.lm.rescore_hypotheses(hyps, lm_weight=1.0, normalize_by_length=True)
        self.assertEqual(rescored[0]["text"], "konuşma tanıma sistemleri yapay zeka ile gelişiyor")

    def test_save_and_load_lm(self):
        """Verify serialization and deserialization of LM tables."""
        save_path = os.path.join(self.temp_dir, "saved_lm.json")
        self.lm.save_lm(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded_lm = ZemberekLanguageModel(lm_file=save_path)
        self.assertEqual(len(loaded_lm.vocab), len(self.lm.vocab))
        self.assertAlmostEqual(
            self.lm.score_sentence("konuşma tanıma"),
            loaded_lm.score_sentence("konuşma tanıma"),
            places=3
        )


if __name__ == "__main__":
    unittest.main()
