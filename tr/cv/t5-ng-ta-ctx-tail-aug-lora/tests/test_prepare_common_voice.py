"""Unit tests for Mozilla Common Voice Turkish data preparation logic."""

import json
import os
import shutil
import tempfile
import unittest

from scripts.prepare_common_voice import locate_common_voice_files, parse_common_voice_tsv
from scripts.create_toy_turkish_dataset import generate_synthetic_wav


class TestPrepareCommonVoice(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.extract_dir = os.path.join(self.test_dir, "extracted", "cv-corpus-tr")
        self.clips_dir = os.path.join(self.extract_dir, "clips")

        os.makedirs(self.clips_dir, exist_ok=True)

        # Create synthetic audio clips
        wav1 = os.path.join(self.clips_dir, "common_voice_tr_001.wav")
        wav2 = os.path.join(self.clips_dir, "common_voice_tr_002.wav")
        generate_synthetic_wav(wav1, duration_sec=2.0)
        generate_synthetic_wav(wav2, duration_sec=1.5)

        # Create train.tsv
        train_tsv = os.path.join(self.extract_dir, "train.tsv")
        with open(train_tsv, "w", encoding="utf-8") as f:
            f.write("client_id\tpath\tsentence\tup_votes\tdown_votes\n")
            f.write("user1\tcommon_voice_tr_001.wav\tTÜRKİYE CUMHURİYETİ İÇİN ÇALIŞIYORUZ\t2\t0\n")
            f.write("user2\tcommon_voice_tr_002.wav\tYAPAY ZEKA VE DERİN ÖĞRENME\t3\t0\n")

        # Create test.tsv
        test_tsv = os.path.join(self.extract_dir, "test.tsv")
        with open(test_tsv, "w", encoding="utf-8") as f:
            f.write("client_id\tpath\tsentence\n")
            f.write("user3\tcommon_voice_tr_001.wav\tDOĞRULAMA KÜMESİ TESTİ\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_locate_common_voice_files(self):
        tsv_files, clips_dir = locate_common_voice_files(self.test_dir)
        self.assertIsNotNone(clips_dir)
        self.assertTrue(clips_dir.endswith("clips"))
        self.assertIn("train", tsv_files)
        self.assertIn("test", tsv_files)

    def test_parse_common_voice_tsv(self):
        tsv_files, clips_dir = locate_common_voice_files(self.test_dir)
        samples = parse_common_voice_tsv(tsv_files["train"], clips_dir, split_name="train")
        self.assertEqual(len(samples), 2)

        # Check sample 1
        s1 = samples[0]
        self.assertEqual(s1["key"], "cv_train_common_voice_tr_001")
        # Check Turkish normalization applied
        self.assertEqual(s1["text"], "türkiye cumhuriyeti için çalışıyoruz")
        self.assertEqual(s1["target"], "türkiye cumhuriyeti için çalışıyoruz")
        self.assertTrue(s1["audio_filepath"].endswith("common_voice_tr_001.wav"))
        self.assertAlmostEqual(s1["duration"], 2.0, places=1)


if __name__ == "__main__":
    unittest.main()
