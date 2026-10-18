"""Unit tests for SLAM-ASR projector, downsampler, and enhanced modules (t1-enhanced)."""

import unittest

try:
    import torch
    from slam_asr.models.projector import (
        FrameDownsampler,
        Conv1dDownsampler,
        LinearProjector,
        EnhancedMLPProjector,
        MLPProjector,
        SpeechProjector,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for projector tests")
class TestProjector(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.encoder_dim = 1024
        self.llm_dim = 4096
        self.k = 5

    def test_downsampler_discard_remainder(self):
        downsampler = FrameDownsampler(k=self.k, padding_mode="discard")
        # 53 frames: 53 % 5 = 3 remainder -> should keep 50 frames -> 10 downsampled frames
        x = torch.randn(self.batch_size, 53, self.encoder_dim)
        mask = torch.ones(self.batch_size, 53, dtype=torch.long)

        out, out_mask = downsampler(x, mask)
        expected_time = 10
        expected_dim = self.encoder_dim * self.k

        self.assertEqual(out.shape, (self.batch_size, expected_time, expected_dim))
        self.assertEqual(out_mask.shape, (self.batch_size, expected_time))

    def test_downsampler_pad_remainder(self):
        downsampler = FrameDownsampler(k=self.k, padding_mode="pad")
        # 53 frames: pad by 2 to reach 55 -> 11 downsampled frames
        x = torch.randn(self.batch_size, 53, self.encoder_dim)
        mask = torch.ones(self.batch_size, 53, dtype=torch.long)

        out, out_mask = downsampler(x, mask)
        expected_time = 11
        expected_dim = self.encoder_dim * self.k

        self.assertEqual(out.shape, (self.batch_size, expected_time, expected_dim))
        self.assertEqual(out_mask.shape, (self.batch_size, expected_time))

    def test_downsampler_variable_k(self):
        for k in (4, 6):
            downsampler = FrameDownsampler(k=k, padding_mode="discard")
            x = torch.randn(self.batch_size, 60, self.encoder_dim)
            out, _ = downsampler(x)
            expected_time = 60 // k
            expected_dim = self.encoder_dim * k
            self.assertEqual(out.shape, (self.batch_size, expected_time, expected_dim))

    def test_conv1d_downsampler(self):
        downsampler = Conv1dDownsampler(in_dim=self.encoder_dim, out_dim=self.encoder_dim, k=self.k)
        x = torch.randn(self.batch_size, 50, self.encoder_dim)
        mask = torch.ones(self.batch_size, 50, dtype=torch.long)
        out, out_mask = downsampler(x, mask)
        self.assertEqual(out.shape, (self.batch_size, 10, self.encoder_dim))
        self.assertEqual(out_mask.shape, (self.batch_size, 10))

    def test_linear_projector(self):
        in_dim = self.encoder_dim * self.k
        projector = LinearProjector(in_dim=in_dim, out_dim=self.llm_dim)
        x = torch.randn(self.batch_size, 10, in_dim)
        out = projector(x)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_enhanced_mlp_projector_gelu(self):
        in_dim = self.encoder_dim * self.k
        projector = EnhancedMLPProjector(in_dim=in_dim, out_dim=self.llm_dim, activation="gelu")
        x = torch.randn(self.batch_size, 10, in_dim)
        out = projector(x)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_enhanced_mlp_projector_swiglu(self):
        in_dim = self.encoder_dim * self.k
        projector = EnhancedMLPProjector(in_dim=in_dim, out_dim=self.llm_dim, activation="swiglu")
        x = torch.randn(self.batch_size, 10, in_dim)
        out = projector(x)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_speech_projector_mlp_end_to_end(self):
        speech_projector = SpeechProjector(
            encoder_dim=self.encoder_dim,
            llm_dim=self.llm_dim,
            downsample_rate=5,
            projector_type="mlp",
            mlp_activation="gelu",
        )
        features = torch.randn(self.batch_size, 50, self.encoder_dim)
        out, _ = speech_projector(features)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_speech_projector_conv_mlp_end_to_end(self):
        speech_projector = SpeechProjector(
            encoder_dim=self.encoder_dim,
            llm_dim=self.llm_dim,
            downsample_rate=5,
            projector_type="conv_mlp",
        )
        features = torch.randn(self.batch_size, 50, self.encoder_dim)
        out, _ = speech_projector(features)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_speech_projector_linear_end_to_end(self):
        speech_projector = SpeechProjector(
            encoder_dim=self.encoder_dim,
            llm_dim=self.llm_dim,
            downsample_rate=5,
            projector_type="linear",
        )
        features = torch.randn(self.batch_size, 50, self.encoder_dim)
        out, _ = speech_projector(features)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))


if __name__ == "__main__":
    unittest.main()
