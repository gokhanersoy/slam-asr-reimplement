"""Unit tests for targeted speech tail augmentations in SLAM-ASR (t1-tail-aug)."""

import unittest

try:
    import torch
    from slam_asr.data.tail_augmentations import (
        TailSpeedPerturbation,
        TailEnergyDecay,
        TailSpecMasking,
        TailAugmentationPipeline,
    )
    from slam_asr.data.audio_processor import AudioProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for tail augmentation tests")
class TestTailAugmentations(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        # 2.0 seconds of audio: 32,000 samples
        t = torch.linspace(0, 2.0, 32000)
        self.waveform = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz sine tone

    def test_tail_speedup_reduces_length(self):
        """Accelerating the tail by 1.25x should shorten the total duration."""
        speedup = TailSpeedPerturbation(
            tail_ratio_range=(0.30, 0.30),
            speed_factor_range=(1.25, 1.25),
            crossfade_ms=40,
            sample_rate=self.sr,
            p=1.0,
        )
        augmented = speedup(self.waveform, tail_ratio=0.30, speed_factor=1.25)
        self.assertLess(augmented.size(-1), self.waveform.size(-1))
        self.assertFalse(torch.isnan(augmented).any())
        self.assertFalse(torch.isinf(augmented).any())

    def test_tail_speedup_preserves_head(self):
        """The beginning of the utterance must remain identical prior to crossfade."""
        speedup = TailSpeedPerturbation(
            tail_ratio_range=(0.30, 0.30),
            speed_factor_range=(1.25, 1.25),
            crossfade_ms=40,
            sample_rate=self.sr,
            p=1.0,
        )
        augmented = speedup(self.waveform, tail_ratio=0.30, speed_factor=1.25)
        # Check first 50% of the audio: should be bitwise identical
        half = int(self.waveform.size(-1) * 0.50)
        self.assertTrue(torch.allclose(self.waveform[:half], augmented[:half], atol=1e-5))

    def test_tail_energy_decay_lowers_tail_rms(self):
        """Tail energy decay should reduce RMS amplitude strictly in the tail."""
        decay = TailEnergyDecay(
            tail_ratio_range=(0.30, 0.30),
            decay_db_range=(-6.0, -6.0),
            p=1.0,
        )
        augmented = decay(self.waveform, tail_ratio=0.30, decay_db=-6.0)

        split_point = int(self.waveform.size(-1) * 0.70)
        orig_head = self.waveform[:split_point]
        aug_head = augmented[:split_point]
        self.assertTrue(torch.allclose(orig_head, aug_head, atol=1e-6))

        orig_tail_rms = torch.sqrt(torch.mean(self.waveform[split_point:] ** 2))
        aug_tail_rms = torch.sqrt(torch.mean(augmented[split_point:] ** 2))
        self.assertLess(aug_tail_rms.item(), orig_tail_rms.item())

    def test_tail_spec_masking(self):
        """Tail spec masking should mask cells in the tail frames while preserving early frames."""
        masking = TailSpecMasking(
            tail_ratio_range=(0.30, 0.30),
            freq_mask_param=8,
            time_mask_param=10,
            mask_count=2,
            mask_value=0.0,
            p=1.0,
        )
        spec = torch.ones(80, 200)
        augmented = masking(spec, speech_time_frames=200)

        # Early frames (first 50%) should not be masked
        self.assertTrue((augmented[:, :100] == 1.0).all())
        # Tail frames should contain at least some 0.0 masks
        self.assertTrue((augmented[:, 140:] == 0.0).any())

    def test_eval_mode_bypasses_augmentation(self):
        """AudioProcessor in eval mode should not alter waveforms."""
        processor = AudioProcessor(
            sampling_rate=16000,
            normalize=False,
            is_whisper=False,
            apply_tail_speedup=True,
            apply_tail_energy_decay=True,
            is_training=False,
        )
        processed = processor.process(self.waveform)
        self.assertTrue(torch.allclose(processed, self.waveform))


if __name__ == "__main__":
    unittest.main()
