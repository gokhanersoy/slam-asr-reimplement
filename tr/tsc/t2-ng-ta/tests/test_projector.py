"""Unit tests for SLAM-ASR projector, downsampler, and NGram-Temporal Hybrid Attention modules (t2-ng-ta)."""

import unittest

try:
    import torch
    from slam_asr.models.projector import (
        FrameDownsampler,
        LinearProjector,
        MLPProjector,
        MultiOrderNGramConv,
        TemporalAttentionPooling,
        TemporalAttentionDownsampler,
        NGramContextAttention,
        TemporalSelfAttentionBlock,
        HybridTemporalNGramAttentionBlock,
        NGramTemporalHybridProjector,
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

    def test_multi_order_ngram_conv(self):
        conv = MultiOrderNGramConv(
            dim=self.encoder_dim,
            ngram_orders=(1, 3, 5, 9),
            dropout=0.0,
        )
        conv.eval()
        x = torch.randn(self.batch_size, 50, self.encoder_dim)
        out, weights = conv(x)
        self.assertEqual(out.shape, (self.batch_size, 50, self.encoder_dim))
        self.assertEqual(weights.shape, (self.batch_size, 50, 4))
        # Gating weights across branches should sum to ~1.0 per frame
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(self.batch_size, 50), atol=1e-5))

    def test_temporal_attention_pooling(self):
        pool = TemporalAttentionPooling(dim=self.encoder_dim)
        pool.eval()
        x_win = torch.randn(self.batch_size, 10, self.k, self.encoder_dim)
        attended, weights = pool(x_win)
        self.assertEqual(attended.shape, (self.batch_size, 10, self.encoder_dim))
        self.assertEqual(weights.shape, (self.batch_size, 10, self.k))
        # Intra-window attention weights should sum to 1.0
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(self.batch_size, 10), atol=1e-5))

    def test_temporal_attention_downsampler_hybrid(self):
        downsampler = TemporalAttentionDownsampler(
            k=self.k,
            encoder_dim=self.encoder_dim,
            pooling_mode="hybrid",
        )
        downsampler.eval()
        x = torch.randn(self.batch_size, 50, self.encoder_dim)
        mask = torch.ones(self.batch_size, 50, dtype=torch.long)
        out, out_mask = downsampler(x, mask)
        # hybrid: (k + 1) * encoder_dim
        expected_dim = (self.k + 1) * self.encoder_dim
        self.assertEqual(out.shape, (self.batch_size, 10, expected_dim))
        self.assertEqual(out_mask.shape, (self.batch_size, 10))

    def test_hybrid_temporal_ngram_attention_split_heads(self):
        attn = HybridTemporalNGramAttentionBlock(
            dim=self.llm_dim,
            window_size=5,
            num_heads=4,
            mode="split_heads",
            dropout=0.0,
        )
        attn.eval()
        x = torch.randn(self.batch_size, 10, self.llm_dim)
        mask = torch.ones(self.batch_size, 10, dtype=torch.long)
        out = attn(x, mask)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_hybrid_temporal_ngram_attention_sequential(self):
        attn = HybridTemporalNGramAttentionBlock(
            dim=self.llm_dim,
            window_size=5,
            num_heads=4,
            mode="sequential",
            dropout=0.0,
        )
        attn.eval()
        x = torch.randn(self.batch_size, 10, self.llm_dim)
        mask = torch.ones(self.batch_size, 10, dtype=torch.long)
        out = attn(x, mask)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))

    def test_ngram_temporal_hybrid_projector_end_to_end(self):
        proj = NGramTemporalHybridProjector(
            encoder_dim=self.encoder_dim,
            llm_dim=self.llm_dim,
            downsample_rate=self.k,
            ngram_orders=(1, 3, 5, 9),
            ngram_window_size=5,
            temporal_pooling_mode="hybrid",
            hybrid_attention_mode="split_heads",
            num_attention_heads=4,
        )
        proj.eval()
        x = torch.randn(self.batch_size, 50, self.encoder_dim)
        mask = torch.ones(self.batch_size, 50, dtype=torch.long)
        out, out_mask = proj(x, mask)
        self.assertEqual(out.shape, (self.batch_size, 10, self.llm_dim))
        self.assertEqual(out_mask.shape, (self.batch_size, 10))

    def test_speech_projector_hybrid_end_to_end(self):
        speech_projector = SpeechProjector(
            encoder_dim=self.encoder_dim,
            llm_dim=self.llm_dim,
            downsample_rate=self.k,
            projector_type="ngram_temporal_hybrid",
            ngram_orders=(1, 3, 5, 9),
            ngram_window_size=5,
            temporal_pooling_mode="hybrid",
            hybrid_attention_mode="split_heads",
            num_attention_heads=4,
        )
        speech_projector.eval()
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
