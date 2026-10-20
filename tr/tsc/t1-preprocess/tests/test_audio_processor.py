"""Unit tests for AudioProcessor with Dynamic Range Compression and Tail Buffering."""

import unittest

try:
    import torch
    from slam_asr.data.audio_processor import AudioProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for AudioProcessor tests")
class TestAudioProcessor(unittest.TestCase):
    def test_audio_processor_initialization(self):
        proc = AudioProcessor(
            sampling_rate=16000,
            apply_drc=True,
            drc_threshold_db=-24.0,
            drc_ratio=3.0,
            trailing_silence_sec=0.3,
            leading_silence_sec=0.1,
        )
        self.assertEqual(proc.sampling_rate, 16000)
        self.assertTrue(proc.apply_drc)
        self.assertEqual(proc.trailing_silence_sec, 0.3)
        self.assertEqual(proc.leading_silence_sec, 0.1)

    def test_drc_compression_boosts_quiet_tail(self):
        proc = AudioProcessor(
            sampling_rate=16000,
            apply_drc=True,
            trailing_silence_sec=0.0,
            leading_silence_sec=0.0,
            is_whisper=False,
        )
        # Create a signal where second half is much quieter (trailing energy drop)
        t = torch.linspace(0, 1, 16000)
        loud_part = torch.sin(2 * 3.14159 * 200 * t[:8000]) * 0.8
        quiet_tail = torch.sin(2 * 3.14159 * 200 * t[8000:]) * 0.05
        waveform = torch.cat([loud_part, quiet_tail])

        orig_ratio = loud_part.abs().mean() / quiet_tail.abs().mean()
        processed = proc.process(waveform)

        proc_loud = processed[:8000].abs().mean()
        proc_quiet = processed[8000:].abs().mean()
        proc_ratio = proc_loud / proc_quiet

        # Compressed ratio should be smaller (quiet tail boosted relative to loud part)
        self.assertLess(proc_ratio, orig_ratio)

    def test_trailing_and_leading_silence_padding(self):
        proc = AudioProcessor(
            sampling_rate=16000,
            apply_drc=False,
            trailing_silence_sec=0.2,  # 3200 samples
            leading_silence_sec=0.1,   # 1600 samples
            is_whisper=False,
        )
        waveform = torch.ones(8000)  # 0.5s of audio
        processed = proc.process(waveform)

        # Expected length = 1600 + 8000 + 3200 = 12800 samples
        self.assertEqual(processed.shape[0], 12800)
        # Verify leading zeros
        self.assertTrue((processed[:1600] == 0.0).all())
        # Verify trailing zeros
        self.assertTrue((processed[-3200:] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
