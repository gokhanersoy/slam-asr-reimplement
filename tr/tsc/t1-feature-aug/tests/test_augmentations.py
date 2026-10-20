"""Unit tests for SLAM-ASR audio and spectrogram augmentations (t1-feature-aug)."""

import unittest

try:
    import torch
    from slam_asr.data.augmentations import (
        SpeedPerturbation,
        VolumePerturbation,
        SpecAugment,
        AudioAugmentationPipeline,
    )
    from slam_asr.data.audio_processor import AudioProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for augmentation tests")
class TestAugmentations(unittest.TestCase):
    def test_speed_perturbation_resampling(self):
        waveform = torch.randn(16000)  # 1 second at 16kHz
        speed = SpeedPerturbation(factors=[0.9, 1.0, 1.1])

        # Test faster factor (1.1x -> shorter)
        faster = speed(waveform, factor=1.1)
        expected_len = int(16000 / 1.1)
        self.assertEqual(faster.shape[0], expected_len)

        # Test slower factor (0.9x -> longer)
        slower = speed(waveform, factor=0.9)
        expected_slower = int(16000 / 0.9)
        self.assertEqual(slower.shape[0], expected_slower)

        # Test identity factor (1.0x -> identical)
        same = speed(waveform, factor=1.0)
        self.assertEqual(same.shape[0], 16000)
        self.assertTrue(torch.equal(same, waveform))

    def test_volume_perturbation_clamping(self):
        waveform = torch.tensor([0.5, -0.5, 0.2, -0.2])
        vol = VolumePerturbation(min_db=-4.0, max_db=4.0)

        # Test known dB shift
        boosted = vol(waveform, db_shift=6.0)  # ~2x
        self.assertAlmostEqual(boosted[0].item(), 0.5 * (10.0 ** (6.0 / 20.0)), places=3)

        # Ensure bounds [-1.0, 1.0]
        hot_waveform = torch.tensor([0.9, -0.9])
        clamped = vol(hot_waveform, db_shift=12.0)
        self.assertLessEqual(clamped.max().item(), 1.0)
        self.assertGreaterEqual(clamped.min().item(), -1.0)

    def test_spec_augment_masking(self):
        # 80 mel bins, 100 time frames filled with ones
        mel_spec = torch.ones(80, 100)
        spec_aug = SpecAugment(
            freq_mask_param=15,
            time_mask_param=20,
            freq_mask_count=2,
            time_mask_count=2,
            mask_value=0.0,
        )

        augmented = spec_aug(mel_spec)
        self.assertEqual(augmented.shape, (80, 100))
        # Masked elements should exist (zeros present)
        zeros_count = (augmented == 0.0).sum().item()
        self.assertGreater(zeros_count, 0)
        # Non-masked elements should still be 1.0
        ones_count = (augmented == 1.0).sum().item()
        self.assertGreater(ones_count, 0)
        self.assertEqual(zeros_count + ones_count, 80 * 100)

    def test_audio_processor_train_vs_eval_mode(self):
        proc = AudioProcessor(
            sampling_rate=16000,
            is_whisper=False,
            apply_speed_perturbation=True,
            speed_factors=[1.1],
            apply_volume_perturbation=False,
            apply_spec_augment=False,
            is_training=False,  # Start in eval mode
        )

        waveform = torch.randn(16000)
        # Eval mode: no perturbation
        eval_out = proc.process(waveform)
        self.assertEqual(eval_out.shape[0], 16000)

        # Switch to train mode: speed perturbation active
        proc.train(True)
        train_out = proc.process(waveform)
        self.assertEqual(train_out.shape[0], int(16000 / 1.1))

        # Switch back to eval mode
        proc.eval()
        eval_out2 = proc.process(waveform)
        self.assertEqual(eval_out2.shape[0], 16000)


if __name__ == "__main__":
    unittest.main()
