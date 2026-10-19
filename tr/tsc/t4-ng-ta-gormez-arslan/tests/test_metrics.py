"""Unit tests for text normalization, Levenshtein distance, WER, and CER."""

import unittest
from slam_asr.utils.metrics import (
    normalize_text_librispeech,
    levenshtein_distance,
    compute_wer,
    compute_cer,
)


class TestMetrics(unittest.TestCase):
    def test_text_normalization(self):
        text = "Hello, WORLD! This is a test... with 123 punctuation."
        expected = "hello world this is a test with 123 punctuation"
        self.assertEqual(normalize_text_librispeech(text), expected)

    def test_levenshtein_distance(self):
        # exact match
        self.assertEqual(levenshtein_distance(["hello", "world"], ["hello", "world"]), 0)
        # substitution
        self.assertEqual(levenshtein_distance(["hello", "world"], ["hello", "there"]), 1)
        # insertion
        self.assertEqual(levenshtein_distance(["hello"], ["hello", "world"]), 1)
        # deletion
        self.assertEqual(levenshtein_distance(["hello", "world"], ["hello"]), 1)
        # empty
        self.assertEqual(levenshtein_distance([], ["hello"]), 1)
        self.assertEqual(levenshtein_distance([], []), 0)

    def test_compute_wer(self):
        refs = ["THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"]
        hyps = ["THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"]
        self.assertAlmostEqual(compute_wer(hyps, refs), 0.0)

        # 1 substitution out of 9 words: 1/9 * 100% = 11.11%
        hyps_sub = ["THE QUICK BROWN CAT JUMPS OVER THE LAZY DOG"]
        self.assertAlmostEqual(compute_wer(hyps_sub, refs), (1 / 9) * 100.0, places=2)

        # 1 deletion out of 9 words
        hyps_del = ["THE QUICK BROWN FOX JUMPS OVER THE LAZY"]
        self.assertAlmostEqual(compute_wer(hyps_del, refs), (1 / 9) * 100.0, places=2)

        # 1 insertion
        hyps_ins = ["THE QUICK BROWN FOX JUMPS OVER ALL THE LAZY DOG"]
        self.assertAlmostEqual(compute_wer(hyps_ins, refs), (1 / 9) * 100.0, places=2)

    def test_compute_cer(self):
        refs = ["CAT"]
        hyps = ["BAT"]
        self.assertAlmostEqual(compute_cer(hyps, refs), (1 / 3) * 100.0, places=2)


if __name__ == "__main__":
    unittest.main()
