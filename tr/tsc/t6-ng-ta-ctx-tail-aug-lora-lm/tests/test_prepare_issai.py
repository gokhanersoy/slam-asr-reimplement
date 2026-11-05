"""Unit tests for ISSAI Turkish Speech Corpus data preparation logic."""

import json
import os
import shutil
import tempfile
import unittest

from scripts.prepare_issai_tsc import locate_split_directories, scan_paired_directory
from scripts.create_toy_turkish_dataset import generate_synthetic_wav


class TestPrepareISSAI(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.extract_dir = os.path.join(self.test_dir, "extracted")

        # Create ISSAI_TSC_218 directory structure: Train, Dev, Test
        self.train_dir = os.path.join(self.extract_dir, "Train")
        self.dev_dir = os.path.join(self.extract_dir, "Dev")
        self.test_dir_split = os.path.join(self.extract_dir, "Test")

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.dev_dir, exist_ok=True)
        os.makedirs(self.test_dir_split, exist_ok=True)

        # Create paired 1.wav, 1.txt in Train
        wav1 = os.path.join(self.train_dir, "1.wav")
        txt1 = os.path.join(self.train_dir, "1.txt")
        generate_synthetic_wav(wav1, duration_sec=1.5)
        with open(txt1, "w", encoding="utf-8") as f:
            f.write("TÜRKİYE CUMHURİYETİ İÇİN ÇALIŞIYORUZ\n")

        # Create paired 2.wav, 2.txt in Train
        wav2 = os.path.join(self.train_dir, "2.wav")
        txt2 = os.path.join(self.train_dir, "2.txt")
        generate_synthetic_wav(wav2, duration_sec=1.2)
        with open(txt2, "w", encoding="utf-8") as f:
            f.write("YAPAY ZEKA VE DERİN ÖĞRENME\n")

        # Create paired 1.wav, 1.txt in Dev
        dev_wav = os.path.join(self.dev_dir, "1.wav")
        dev_txt = os.path.join(self.dev_dir, "1.txt")
        generate_synthetic_wav(dev_wav, duration_sec=1.0)
        with open(dev_txt, "w", encoding="utf-8") as f:
            f.write("DOĞRULAMA KÜMESİ TESTİ\n")

        # Create paired 1.wav, 1.txt in Test
        test_wav = os.path.join(self.test_dir_split, "1.wav")
        test_txt = os.path.join(self.test_dir_split, "1.txt")
        generate_synthetic_wav(test_wav, duration_sec=1.0)
        with open(test_txt, "w", encoding="utf-8") as f:
            f.write("TEST KÜMESİ TESTİ\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_locate_split_directories(self):
        splits = locate_split_directories(self.extract_dir)
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)
        self.assertEqual(os.path.basename(splits["train"]), "Train")
        self.assertEqual(os.path.basename(splits["val"]), "Dev")
        self.assertEqual(os.path.basename(splits["test"]), "Test")

    def test_scan_paired_directory(self):
        train_samples = scan_paired_directory(self.train_dir, split_name="train")
        self.assertEqual(len(train_samples), 2)

        # Check sample 1
        s1 = train_samples[0]
        self.assertEqual(s1["key"], "issai_train_1")
        # Check Turkish normalization applied: 'İ' -> 'i', 'I' -> 'ı'
        self.assertEqual(s1["text"], "türkiye cumhuriyeti için çalışıyoruz")
        self.assertEqual(s1["target"], "türkiye cumhuriyeti için çalışıyoruz")
        self.assertTrue(s1["audio_filepath"].endswith("1.wav"))
        self.assertTrue(s1["source"].endswith("1.wav"))
        self.assertAlmostEqual(s1["duration"], 1.5, places=1)

        # Check Dev sample
        dev_samples = scan_paired_directory(self.dev_dir, split_name="val")
        self.assertEqual(len(dev_samples), 1)
        self.assertEqual(dev_samples[0]["text"], "doğrulama kümesi testi")


if __name__ == "__main__":
    unittest.main()
