"""Unit tests for SLAM-ASR projector, downsamplers, and Phoneme Attention."""

import unittest

try:
    import torch
    from slam_asr.models.projector import (
        FrameDownsampler,
        LinearProjector,
        MLPProjector,
        PhonemeAttentionProjector,
        PhonemeSelfAttentionBlock,
        PhoneticPrototypeAttention,
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

    def test_linear_projector(self):
        in_dim = self.encoder_dim * self.k
        projector = LinearProjector(in_dim=in_dim, out_dim=self.llm_dim)
        x = torch.randn(self.batch_size, 10, in_dim)
        out = projector(x)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_mlp_projector(self):
        in_dim = self.encoder_dim * self.k
        hidden_dim = 2048
        projector = MLPProjector(in_dim=in_dim, out_dim=self.llm_dim, hidden_dim=hidden_dim)
        x = torch.randn(self.batch_size, 10, in_dim)
        out = projector(x)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_phonetic_prototype_attention(self):
        num_proto = 32
        ppa = PhoneticPrototypeAttention(dim=self.encoder_dim, num_prototypes=num_proto, num_heads=4)
        x = torch.randn(self.batch_size, 50, self.encoder_dim)
        out, weights = ppa(x)
        self.assertEqual(out.shape, (self.batch_size, 50, self.encoder_dim))
        self.assertEqual(weights.shape, (self.batch_size, 50, num_proto))
        # Attention weights should sum to ~1.0
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(self.batch_size, 50), atol=1e-5))

    def test_phoneme_self_attention_block(self):
        psa = PhonemeSelfAttentionBlock(dim=self.llm_dim, num_heads=4, dropout=0.0)
        x = torch.randn(self.batch_size, 10, self.llm_dim)
        mask = torch.ones(self.batch_size, 10, dtype=torch.long)
        out = psa(x, mask)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_phoneme_attention_projector_end_to_end(self):
        pap = PhonemeAttentionProjector(
            encoder_dim=self.encoder_dim,
            llm_dim=self.llm_dim,
            downsample_rate=self.k,
            num_phoneme_prototypes=32,
            num_attention_heads=4,
        )
        x = torch.randn(self.batch_size, 50, self.encoder_dim)
        mask = torch.ones(self.batch_size, 50, dtype=torch.long)
        out, out_mask = pap(x, mask)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))
        self.assertEqual(out_mask.shape, (self.batch_size, 10))

    def test_speech_projector_phoneme_attention_end_to_end(self):
        speech_projector = SpeechProjector(
            encoder_dim=self.encoder_dim,
            llm_dim=self.llm_dim,
            downsample_rate=self.k,
            projector_type="phoneme_attention",
            num_phoneme_prototypes=32,
            num_attention_heads=4,
        )
        features = torch.randn(self.batch_size, 50, self.encoder_dim)
        mask = torch.ones(self.batch_size, 50, dtype=torch.long)
        out, out_mask = speech_projector(features, mask)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))
        self.assertEqual(out_mask.shape, (self.batch_size, 10))

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
