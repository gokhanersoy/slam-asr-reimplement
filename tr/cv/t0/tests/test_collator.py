"""Unit tests for ASRDataCollator."""

import unittest

try:
    import torch
    from slam_asr.data.collator import ASRDataCollator
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for collator tests")
class TestCollator(unittest.TestCase):
    def test_variable_length_waveform_collation(self):
        collator = ASRDataCollator(is_whisper=False)
        batch = [
            {
                "key": "sample_1",
                "speech": torch.randn(16000),  # 1s
                "target": "hello world",
                "prompt": "Transcribe speech to text.",
            },
            {
                "key": "sample_2",
                "speech": torch.randn(32000),  # 2s
                "target": "this is a second test",
                "prompt": "Transcribe speech to text.",
            },
        ]

        collated = collator(batch)
        self.assertEqual(collated["speech_inputs"].shape, (2, 32000))
        self.assertEqual(collated["speech_attention_mask"].shape, (2, 32000))

        # Check mask values
        self.assertEqual(collated["speech_attention_mask"][0, :16000].sum().item(), 16000)
        self.assertEqual(collated["speech_attention_mask"][0, 16000:].sum().item(), 0)
        self.assertEqual(collated["speech_attention_mask"][1].sum().item(), 32000)

        self.assertEqual(collated["keys"], ["sample_1", "sample_2"])
        self.assertEqual(collated["transcripts"], ["hello world", "this is a second test"])

    def test_whisper_mel_collation(self):
        collator = ASRDataCollator(is_whisper=True)
        batch = [
            {
                "key": "sample_1",
                "speech": torch.randn(80, 3000),
                "target": "whisper sample",
                "prompt": "Transcribe speech to text.",
            },
            {
                "key": "sample_2",
                "speech": torch.randn(80, 3000),
                "target": "another sample",
                "prompt": "Transcribe speech to text.",
            },
        ]
        collated = collator(batch)
        self.assertEqual(collated["speech_inputs"].shape, (2, 80, 3000))
        self.assertIsNone(collated["speech_attention_mask"])


if __name__ == "__main__":
    unittest.main()
